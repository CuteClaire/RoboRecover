from __future__ import annotations

"""
RLDS writer helper for eval-time rollouts.

Writes episodes in a TFDS/RLDS compatible format so that
`openpi/examples/libero/convert_libero_data_to_lerobot.py` can load them from
`tfds.load(<suite>_no_noops, data_dir=...)`.

Features schema is intended to match `DATASETS/modified_libero_rlds`.
"""

import pathlib
import shutil
from typing import Any, DefaultDict, Dict, Iterable, List, Tuple


REFERENCE_MODIFIED_LIBERO_RLDS_ROOT = pathlib.Path("/path/to/modified_libero_rlds")


def rlds_features():
    # Lazy imports for TFDS/TF.
    import numpy as np
    import tensorflow_datasets as tfds

    return tfds.features.FeaturesDict(
        {
            "steps": tfds.features.Dataset(
                {
                    "observation": tfds.features.FeaturesDict(
                        {
                            "image": tfds.features.Image(
                                shape=(256, 256, 3),
                                dtype=np.uint8,
                                encoding_format="jpeg",
                            ),
                            "wrist_image": tfds.features.Image(
                                shape=(256, 256, 3),
                                dtype=np.uint8,
                                encoding_format="jpeg",
                            ),
                            "state": tfds.features.Tensor(shape=(8,), dtype=np.float32),
                            "joint_state": tfds.features.Tensor(shape=(7,), dtype=np.float32),
                        }
                    ),
                    "action": tfds.features.Tensor(shape=(7,), dtype=np.float32),
                    "discount": tfds.features.Scalar(dtype=np.float32),
                    "reward": tfds.features.Scalar(dtype=np.float32),
                    "is_first": tfds.features.Scalar(dtype=np.bool_),
                    "is_last": tfds.features.Scalar(dtype=np.bool_),
                    "is_terminal": tfds.features.Scalar(dtype=np.bool_),
                    "language_instruction": tfds.features.Text(),
                }
            ),
            "episode_metadata": tfds.features.FeaturesDict(
                {
                    "file_path": tfds.features.Text(),
                }
            ),
        }
    )


def _copy_reference_features_json(*, dataset_name: str, out_dataset_version: pathlib.Path) -> None:
    ref = REFERENCE_MODIFIED_LIBERO_RLDS_ROOT / dataset_name / "1.0.0" / "features.json"
    if not ref.exists():
        raise FileNotFoundError(f"Reference features.json not found: {ref}")
    out_dataset_version.mkdir(parents=True, exist_ok=True)
    shutil.copy(ref, out_dataset_version / "features.json")


def write_rlds_episodes(
    *,
    episodes_by_dataset: Dict[str, List[Dict[str, Any]]],
    rlds_data_dir: pathlib.Path,
    num_shards: int = 1,
    overwrite: bool = True,
    split_name: str = "train",
) -> None:
    """
    episodes_by_dataset keys should be RLDS dataset names like:
      libero_10_no_noops / libero_goal_no_noops / ...

    Each episode should be a dict: {"steps": [...], "episode_metadata": {"file_path": str}}
    where every step dict matches the schema from `rlds_features()`.
    """

    import tensorflow as tf
    import tensorflow_datasets as tfds

    RLDS_FEATURES = rlds_features()
    serializer = tfds.core.example_serializer.ExampleSerializer(RLDS_FEATURES.get_serialized_info())

    for dataset_name, episodes in episodes_by_dataset.items():
        output_version = rlds_data_dir / dataset_name / "1.0.0"
        if overwrite and output_version.exists():
            shutil.rmtree(output_version)
        output_version.mkdir(parents=True, exist_ok=True)

        _copy_reference_features_json(dataset_name=dataset_name, out_dataset_version=output_version)

        n = len(episodes)
        # Shard distribution (contiguous episodes).
        if num_shards < 1:
            raise ValueError(f"num_shards must be >= 1, got {num_shards}")
        if n == 0:
            shard_ranges: List[Tuple[int, int]] = [(0, 0)]
        else:
            shard_size = (n + num_shards - 1) // num_shards
            shard_ranges = []
            for shard_id in range(num_shards):
                start = shard_id * shard_size
                end = min(start + shard_size, n)
                if start < end:
                    shard_ranges.append((start, end))
            if not shard_ranges:
                shard_ranges = [(0, 0)]

        shard_lengths: List[str] = []
        shard_tfrecord_paths: List[pathlib.Path] = []

        for actual_shard_idx, (start, end) in enumerate(shard_ranges):
            tfrecord_path = (
                output_version
                / f"{dataset_name}-{split_name}.tfrecord-{actual_shard_idx:05d}-of-{len(shard_ranges):05d}"
            )
            shard_tfrecord_paths.append(tfrecord_path)

            with tf.io.TFRecordWriter(str(tfrecord_path)) as w:
                for i in range(start, end):
                    ex = RLDS_FEATURES.encode_example(episodes[i])
                    w.write(serializer.serialize_example(ex))

            # Count examples for dataset_info.json.
            cnt = sum(1 for _ in tf.data.TFRecordDataset(str(tfrecord_path)))
            shard_lengths.append(str(cnt))

        import json

        # Prefer copying the reference dataset_info.json so that TFDS can
        # infer any extra required metadata.
        ref_info_path = REFERENCE_MODIFIED_LIBERO_RLDS_ROOT / dataset_name / "1.0.0" / "dataset_info.json"
        if ref_info_path.exists():
            with open(ref_info_path, "r") as f:
                info = json.load(f)

            # Update shard lengths for the split we wrote.
            updated = False
            for split in info.get("splits", []):
                if split.get("name") == split_name:
                    split["shardLengths"] = shard_lengths
                    updated = True
                    if "numBytes" in split:
                        split["numBytes"] = str(sum(p.stat().st_size for p in shard_tfrecord_paths))
            if not updated:
                # Fallback to minimal structure if split name mismatches.
                info = {
                    "name": dataset_name,
                    "version": "1.0.0",
                    "fileFormat": "tfrecord",
                    "splits": [
                        {
                            "name": split_name,
                            "filepathTemplate": "{DATASET}-{SPLIT}.{FILEFORMAT}-{SHARD_X_OF_Y}",
                            "shardLengths": shard_lengths,
                            "numBytes": str(sum(p.stat().st_size for p in shard_tfrecord_paths)),
                        }
                    ],
                }
        else:
            info = {
                "name": dataset_name,
                "version": "1.0.0",
                "fileFormat": "tfrecord",
                "splits": [
                    {
                        "name": split_name,
                        "filepathTemplate": "{DATASET}-{SPLIT}.{FILEFORMAT}-{SHARD_X_OF_Y}",
                        "shardLengths": shard_lengths,
                        "numBytes": str(sum(p.stat().st_size for p in shard_tfrecord_paths)),
                    }
                ],
            }

        with open(output_version / "dataset_info.json", "w") as f:
            json.dump(info, f, indent=2)

