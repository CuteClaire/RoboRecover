#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Filtered repro eval for Cosmos Policy on LIBERO."""

import collections
import dataclasses
import fcntl
import json
import logging
import math
import os
import pathlib
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

import draccus
import numpy as np
import torch
import torch.multiprocessing as mp

from libero.libero import benchmark

from cosmos_policy.experiments.robot.cosmos_utils import (
    WorkerPoolManager,
    get_action,
    get_model,
    get_planning_model,
    init_t5_text_embeddings_cache,
    load_dataset_stats,
    query_model_parallel,
)
from cosmos_policy.experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
)
from cosmos_policy.experiments.robot.libero.run_libero_eval import PolicyEvalConfig, validate_config
from cosmos_policy.experiments.robot.robot_utils import get_image_resize_size
from cosmos_policy.utils.utils import set_seed_everywhere

SUITE_MAX_INFER_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


@dataclass
class SampleRef:
    source: str
    folder: str
    file_path: pathlib.Path
    repeat_idx: int = 0


@dataclass
class EvalResult:
    source: str
    file: str
    folder: str
    repeat_idx: int | None
    task_suite_name: str | None
    task_id: int | None
    episode_idx: int | None
    seed: int | None
    status: str
    reason: str
    replay_steps: int
    wait_steps: int
    infer_steps: int
    max_infer_steps: int
    replay_done: bool
    wait_done: bool
    infer_done: bool
    init_xyz_err: float | None
    init_rpy_err: float | None
    init_gripper_err: float | None
    infer_vs_annotated_xyz_err: float | None
    infer_vs_annotated_rpy_err: float | None
    infer_vs_annotated_gripper_err: float | None
    video_agent: str | None
    video_wrist: str | None
    failure_detail: str | None = None
    crash_kind: str | None = None
    retry_count: int = 0
    duration_sec: float | None = None


@dataclass
class ManifestPolicyEvalConfig(PolicyEvalConfig):
    sample_file_list: str = ""
    filtered_roots: str = ""
    trials_per_sample: int = 1
    limit_samples: int = -1
    num_workers: int = 1
    worker_id: int = 0
    resume_eval: bool = False
    repeat_seed_step: int = 1
    manifest_run_name: str = ""
    manifest_model_tag: str = ""
    manifest_video_root: str = ""
    manifest_save_video: bool = True
    manifest_save_wrist_video: bool = False
    manifest_video_fps: int = 30
    manifest_max_infer_steps: int = -1
    eval_mode: str = "seed_plus_replay_then_infer"
    num_steps_wait_after_replay: int = 0
    skip_state_mismatch_check: bool = False
    xyz_tol: float = 0.05
    rpy_tol: float = 0.20
    gripper_tol: float = 0.05
    max_infra_retries: int = 2


def _norm_path(path_like: str) -> str:
    try:
        return str(pathlib.Path(path_like).expanduser().resolve())
    except Exception:
        return str(pathlib.Path(path_like))


def _quat_to_rpy(quat_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = quat_xyzw
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)
    else:
        pitch = math.asin(sinp)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return np.array([roll, pitch, yaw], dtype=np.float64)


def _quat_geodesic_error_xyzw(q1: np.ndarray, q2: np.ndarray) -> float:
    a = np.asarray(q1, dtype=np.float64).reshape(4)
    b = np.asarray(q2, dtype=np.float64).reshape(4)
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    dot = float(np.clip(abs(np.dot(a, b)), 0.0, 1.0))
    return float(2.0 * math.acos(dot))


def _rotation_error_obs_vs_expected_quat(obs_quat_xyzw: np.ndarray, expected_quat: np.ndarray) -> float:
    q_obs = np.asarray(obs_quat_xyzw, dtype=np.float64).reshape(4)
    q_raw = np.asarray(expected_quat, dtype=np.float64).reshape(4)
    as_xyzw = q_raw
    wxyz_as_xyzw = np.array([q_raw[1], q_raw[2], q_raw[3], q_raw[0]], dtype=np.float64)
    return min(
        _quat_geodesic_error_xyzw(q_obs, as_xyzw),
        _quat_geodesic_error_xyzw(q_obs, wxyz_as_xyzw),
    )


def _obs_state(obs: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, float]:
    xyz = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
    quat = np.asarray(obs["robot0_eef_quat"], dtype=np.float64)
    rpy = _quat_to_rpy(quat)
    gripper_raw = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float64).reshape(-1)
    gripper = float(gripper_raw[0]) if gripper_raw.size > 0 else 0.0
    return xyz, rpy, gripper


def _state_error(obs: dict[str, Any], expected: dict[str, Any]) -> tuple[float, float, float]:
    xyz, rpy, gripper = _obs_state(obs)
    ex_xyz = np.asarray(expected["xyz"], dtype=np.float64)
    ex_rpy = np.asarray(expected["rpy"], dtype=np.float64)
    ex_gripper = float(expected["gripper"])
    xyz_err = float(np.max(np.abs(xyz - ex_xyz)))
    gripper_err = abs(gripper - ex_gripper)
    raw_quat = expected.get("quat")
    if raw_quat is not None:
        obs_q = np.asarray(obs["robot0_eef_quat"], dtype=np.float64)
        rpy_err = _rotation_error_obs_vs_expected_quat(obs_q, raw_quat)
    else:
        rpy_err = float(np.max(np.abs(rpy - ex_rpy)))
    return xyz_err, rpy_err, gripper_err


def _state_error_from_triplet(
    xyz: np.ndarray,
    rpy: np.ndarray,
    gripper: float,
    expected_xyz: np.ndarray,
    expected_rpy: np.ndarray,
    expected_gripper: float,
) -> tuple[float, float, float]:
    xyz_err = float(np.max(np.abs(xyz - expected_xyz)))
    rpy_err = float(np.max(np.abs(rpy - expected_rpy)))
    gripper_err = abs(gripper - expected_gripper)
    return xyz_err, rpy_err, gripper_err


def _required_json_fields(data: dict[str, Any]) -> list[str]:
    return ["task_suite_name", "task_id", "episode_idx", "seed", "actions", "initial_state"]


def _load_sample(path: pathlib.Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    missing = [k for k in _required_json_fields(data) if k not in data]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")
    actions = data["actions"]
    if not isinstance(actions, list) or len(actions) == 0:
        raise ValueError("actions must be a non-empty list")
    step = data.get("annotated_step")
    if step is not None:
        step = int(step)
        if step < 0:
            raise ValueError(f"Invalid annotated_step: {step}")
        if len(actions) > step + 1:
            data["actions"] = actions[: step + 1]
    return data


def _parse_filtered_roots(raw_value: str) -> list[pathlib.Path]:
    roots: list[pathlib.Path] = []
    for chunk in raw_value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        roots.append(pathlib.Path(chunk).expanduser().resolve())
    return roots


def _derive_sample_source(path: pathlib.Path, filtered_roots: list[pathlib.Path]) -> tuple[str, str]:
    for root in filtered_roots:
        try:
            rel = path.resolve().relative_to(root)
        except ValueError:
            continue
        if rel.parts:
            return rel.parts[0], path.parent.name
        return root.name, path.parent.name

    parent = path.parent.name
    parts = list(path.parts)
    if "OOD_data" in parts:
        idx = parts.index("OOD_data")
        if idx + 1 < len(parts) - 1:
            return parts[idx + 1], parent
    if "json" in parts:
        idx = parts.index("json")
        if idx > 0:
            return parts[idx - 1], parent
    return parent, parent


def _collect_samples_from_file_list(
    list_path: pathlib.Path,
    *,
    filtered_roots: list[pathlib.Path],
    trials_per_sample: int,
    limit_samples: int,
    num_workers: int,
    worker_id: int,
    completed: set[tuple[str, int]] | None = None,
) -> list[SampleRef]:
    if trials_per_sample <= 0:
        raise ValueError(f"trials_per_sample must be positive, got {trials_per_sample}")

    samples: list[SampleRef] = []
    seen: set[tuple[pathlib.Path, int]] = set()
    with open(list_path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue

            explicit_repeat: int | None = None
            path_text = line
            if "\t" in line:
                path_text, repeat_text = line.rsplit("\t", 1)
                path_text = path_text.strip()
                repeat_text = repeat_text.strip()
                if repeat_text:
                    explicit_repeat = int(repeat_text)

            path_obj = pathlib.Path(path_text).expanduser().resolve()
            if not path_obj.is_file():
                raise FileNotFoundError(f"sample file not found: {path_obj}")
            if path_obj.suffix.lower() != ".json":
                raise ValueError(f"not a json file: {path_obj}")

            repeat_indices = [explicit_repeat] if explicit_repeat is not None else list(range(trials_per_sample))
            for repeat_idx in repeat_indices:
                assert repeat_idx is not None
                key = (path_obj, int(repeat_idx))
                if key in seen:
                    continue
                seen.add(key)
                source, folder = _derive_sample_source(path_obj, filtered_roots)
                samples.append(SampleRef(source=source, folder=folder, file_path=path_obj, repeat_idx=int(repeat_idx)))

    if num_workers <= 0:
        raise ValueError(f"num_workers must be positive, got {num_workers}")
    if num_workers > 1:
        wid = worker_id % num_workers
        samples = [sample for idx, sample in enumerate(samples) if idx % num_workers == wid]

    if completed:
        samples = [
            sample
            for sample in samples
            if (_norm_path(str(sample.file_path)), int(sample.repeat_idx)) not in completed
        ]

    if limit_samples > 0:
        samples = samples[:limit_samples]

    return samples


def _load_completed_results(jsonl_path: pathlib.Path) -> set[tuple[str, int]]:
    completed: set[tuple[str, int]] = set()
    if not jsonl_path.exists():
        return completed
    for raw in jsonl_path.read_text(encoding="utf-8").splitlines():
        row = raw.strip()
        if not row:
            continue
        try:
            obj = json.loads(row)
        except Exception:
            continue
        file_v = obj.get("file")
        repeat_v = obj.get("repeat_idx", 0)
        if isinstance(file_v, str) and file_v:
            completed.add((_norm_path(file_v), int(repeat_v)))
    return completed


def _load_results(jsonl_path: pathlib.Path) -> list[EvalResult]:
    results: list[EvalResult] = []
    if not jsonl_path.exists():
        return results
    for raw in jsonl_path.read_text(encoding="utf-8").splitlines():
        row = raw.strip()
        if not row:
            continue
        try:
            obj = json.loads(row)
            field_names = {field.name for field in dataclasses.fields(EvalResult)}
            filtered = {key: value for key, value in obj.items() if key in field_names}
            results.append(EvalResult(**filtered))
        except Exception:
            logging.warning("Skipping malformed result row in %s", jsonl_path)
    return results


def _counter(items: list[str]) -> dict[str, int]:
    return dict(collections.Counter(items))


def _to_summary(results: list[EvalResult]) -> dict[str, Any]:
    status_counts = _counter([result.status for result in results])
    total = len(results)
    success = status_counts.get("success", 0)
    effective_failure = status_counts.get("timeout_failure", 0) + status_counts.get("env_crash", 0)
    unique_files = len({result.file for result in results})

    def _bucket(key_fn):
        bucket: dict[str, collections.Counter] = {}
        for result in results:
            key = key_fn(result)
            bucket.setdefault(key, collections.Counter())[result.status] += 1
        return {key: dict(value) for key, value in bucket.items()}

    crash_kind_counts = _counter([result.crash_kind or "" for result in results if result.status == "env_crash"])
    mean_infer_steps = float(np.mean([result.infer_steps for result in results]) if results else 0.0)

    return {
        "total_unique_samples": unique_files,
        "total_samples": total,
        "success_count": success,
        "success_rate": (success / total if total else 0.0),
        "status_counts": status_counts,
        "effective_failure": effective_failure,
        "effective_failure_rate": (effective_failure / total if total else 0.0),
        "state_mismatch_rate": (status_counts.get("state_mismatch", 0) / total if total else 0.0),
        "crash_kind_counts": {key: value for key, value in crash_kind_counts.items() if key},
        "mean_infer_steps": mean_infer_steps,
        "by_source": _bucket(lambda result: result.source),
        "by_folder": _bucket(lambda result: result.folder),
        "by_suite": _bucket(lambda result: result.task_suite_name or "unknown"),
    }


def _is_env_terminated(env: Any) -> bool:
    cur = env
    visited: set[int] = set()
    for _ in range(8):
        if cur is None:
            return False
        obj_id = id(cur)
        if obj_id in visited:
            break
        visited.add(obj_id)
        done = getattr(cur, "done", None)
        if isinstance(done, (bool, np.bool_)):
            return bool(done)
        cur = getattr(cur, "env", None)
    return False


def _safe_step(env: Any, action: list[float] | np.ndarray) -> tuple[dict[str, Any], bool] | None:
    if _is_env_terminated(env):
        return None
    try:
        obs, _, done, _ = env.step(action)
        return obs, bool(done)
    except ValueError as exc:
        if "executing action in terminated episode" in str(exc).lower():
            return None
        raise


def _classify_crash(exc: BaseException, crash_phase: str) -> str:
    if isinstance(exc, (ConnectionError, TimeoutError, BrokenPipeError)):
        return "infrastructure"
    if isinstance(exc, (ValueError, KeyError, FileNotFoundError)):
        return "data"
    msg = str(exc).lower()
    if "cuda" in msg or "out of memory" in msg or "oom" in msg:
        return "infrastructure"
    if crash_phase in ("replay", "wait", "infer", "create_env", "init_state"):
        if any(token in msg for token in ("mujoco", "robosuite", "mj_", "mjcf", "simulation")):
            return "simulation"
    return "unknown"


def _prepare_observation(obs: dict[str, Any], flip_images: bool) -> dict[str, np.ndarray]:
    primary_image = get_libero_image(obs, flip_images)
    wrist_image = get_libero_wrist_image(obs, flip_images)
    proprio = np.concatenate((obs["robot0_gripper_qpos"], obs["robot0_eef_pos"], obs["robot0_eef_quat"]))
    return {
        "primary_image": primary_image,
        "wrist_image": wrist_image,
        "proprio": proprio,
    }


def _query_best_actions(
    cfg: Any,
    *,
    observation: dict[str, Any],
    task_description: str,
    model: Any,
    planning_model: Any,
    dataset_stats: dict[str, Any],
    worker_pool: WorkerPoolManager | None,
) -> tuple[list[np.ndarray], dict[str, Any] | None, float]:
    num_queries = int(cfg.num_queries_best_of_n)

    if cfg.use_parallel_inference and num_queries > 1 and worker_pool and worker_pool.initialized:
        query_results = query_model_parallel(cfg, observation, task_description, worker_pool, cfg.parallel_timeout)
    else:
        query_results = []
        for query_idx in range(num_queries):
            return_dict: dict[str, Any] = {}
            action_return_dict = get_action(
                cfg,
                model,
                dataset_stats,
                observation,
                task_description,
                seed=cfg.seed + query_idx,
                randomize_seed=cfg.randomize_seed,
                num_denoising_steps_action=cfg.num_denoising_steps_action,
                generate_future_state_and_value_in_parallel=not (
                    cfg.ar_future_prediction or cfg.ar_value_prediction or cfg.ar_qvalue_prediction
                ),
            )
            return_dict["actions"] = action_return_dict["actions"]
            return_dict["sampling_seed"] = int(action_return_dict.get("sampling_seed", cfg.seed + query_idx))
            if cfg.ar_future_prediction:
                raise NotImplementedError("Manifest eval does not yet support autoregressive future-state prediction.")
            return_dict["future_image_predictions"] = action_return_dict["future_image_predictions"]
            if cfg.ar_value_prediction or cfg.ar_qvalue_prediction:
                raise NotImplementedError("Manifest eval does not yet support autoregressive value/Q-value prediction.")
            return_dict["value_prediction"] = action_return_dict["value_prediction"]
            query_results.append(return_dict)

    best_query_idx, best_return_dict = max(enumerate(query_results), key=lambda item: item[1]["value_prediction"])
    best_actions = [np.asarray(action, dtype=np.float32) for action in best_return_dict["actions"]]
    best_future_predictions = best_return_dict["future_image_predictions"]
    best_value_prediction = float(best_return_dict["value_prediction"])
    best_sampling_seed = int(best_return_dict.get("sampling_seed", cfg.seed + best_query_idx))
    logging.info(
        "Selected query %s with sampling seed %s and value %.4f",
        best_query_idx,
        best_sampling_seed,
        best_value_prediction,
    )
    return best_actions, best_future_predictions, best_value_prediction


def _video_paths(
    *,
    video_root: pathlib.Path,
    model_tag: str,
    source: str,
    suite: str,
    sample_path: pathlib.Path,
    status: str,
    repeat_idx: int,
) -> tuple[pathlib.Path, pathlib.Path]:
    output_dir = video_root / model_tag / source / suite
    output_dir.mkdir(parents=True, exist_ok=True)
    base = sample_path.stem
    repeat_tag = f"rep{int(repeat_idx):02d}"
    return (
        output_dir / f"{base}_{repeat_tag}_{status}_agent.mp4",
        output_dir / f"{base}_{repeat_tag}_{status}_wrist.mp4",
    )


def _save_video(frames: list[np.ndarray], path: pathlib.Path, fps: int) -> str:
    import imageio

    imageio.mimwrite(str(path), [np.asarray(frame) for frame in frames], fps=fps)
    return str(path)


def _max_infer_steps_for_suite(suite: str, cfg_value: int) -> int:
    if cfg_value > 0:
        return cfg_value
    if suite not in SUITE_MAX_INFER_STEPS:
        raise ValueError(f"Unknown task suite for default max steps: {suite}")
    return SUITE_MAX_INFER_STEPS[suite]


def _eval_one_sample_attempt(
    sample: SampleRef,
    cfg: Any,
    *,
    model: Any,
    planning_model: Any,
    dataset_stats: dict[str, Any],
    worker_pool: WorkerPoolManager | None,
    resize_size: int,
    benchmark_cache: dict[str, Any],
    save_video: bool,
    save_wrist_video: bool,
    video_fps: int,
    video_root: pathlib.Path,
    model_tag: str,
    phase_ref: dict[str, str],
) -> EvalResult:
    def _set_phase(phase: str) -> None:
        phase_ref["phase"] = phase

    started_at = time.time()
    eval_mode = _normalize_eval_mode(cfg.eval_mode)
    crash_phase = "load_sample"
    _set_phase(crash_phase)
    env = None
    suite = None
    task_id = None
    episode_idx = None
    seed = None
    max_infer_steps = 0
    replay_steps = 0
    wait_steps = 0
    infer_steps = 0
    replay_done = False
    wait_done = False
    infer_done = False
    status = "invalid_sample"
    reason = "unknown"
    failure_detail: str | None = None
    init_xyz_err = None
    init_rpy_err = None
    init_gripper_err = None
    infer_vs_annotated_xyz_err = None
    infer_vs_annotated_rpy_err = None
    infer_vs_annotated_gripper_err = None
    video_agent = None
    video_wrist = None
    agent_frames: list[np.ndarray] = []
    wrist_frames: list[np.ndarray] = []
    annotated_xyz = None
    annotated_rpy = None
    annotated_gripper = None

    original_cfg_seed = int(cfg.seed)
    effective_seed = int(original_cfg_seed) + int(sample.repeat_idx) * int(cfg.repeat_seed_step)
    cfg.seed = effective_seed
    set_seed_everywhere(effective_seed)

    raw = _load_sample(sample.file_path)
    suite = str(raw["task_suite_name"])
    task_id = int(raw["task_id"])
    episode_idx = int(raw["episode_idx"])
    seed = int(raw["seed"])
    expected_init = raw["initial_state"]
    actions = raw["actions"]
    max_infer_steps = _max_infer_steps_for_suite(suite, int(cfg.manifest_max_infer_steps))

    crash_phase = "benchmark"
    _set_phase(crash_phase)
    if suite not in benchmark_cache:
        benchmark_cache[suite] = benchmark.get_benchmark_dict()[suite]()
    task_suite = benchmark_cache[suite]
    task = task_suite.get_task(task_id)
    task_description = task.language
    initial_states = task_suite.get_task_init_states(task_id)
    if episode_idx < 0 or episode_idx >= len(initial_states):
        raise ValueError(f"episode_idx out of range: {episode_idx}")

    crash_phase = "create_env"
    _set_phase(crash_phase)
    env, _ = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res, seed=seed)
    try:
        crash_phase = "init_state"
        _set_phase(crash_phase)
        env.reset()
        obs = env.set_init_state(initial_states[episode_idx])
        init_xyz_err, init_rpy_err, init_gripper_err = _state_error(obs, expected_init)

        if not cfg.skip_state_mismatch_check:
            if (
                init_xyz_err > float(cfg.xyz_tol)
                or init_rpy_err > float(cfg.rpy_tol)
                or init_gripper_err > float(cfg.gripper_tol)
            ):
                status = "state_mismatch"
                reason = (
                    f"init pose exceeds tol: xyz_err={init_xyz_err:.6f}(tol {cfg.xyz_tol}) "
                    f"rpy_err={init_rpy_err:.6f}(tol {cfg.rpy_tol}) "
                    f"grip_err={init_gripper_err:.6f}(tol {cfg.gripper_tol})"
                )
                failure_detail = "init_state_mismatch"
                return EvalResult(
                    source=sample.source,
                    file=str(sample.file_path),
                    folder=sample.folder,
                    repeat_idx=int(sample.repeat_idx),
                    task_suite_name=suite,
                    task_id=task_id,
                    episode_idx=episode_idx,
                    seed=seed,
                    status=status,
                    reason=reason,
                    replay_steps=0,
                    wait_steps=0,
                    infer_steps=0,
                    max_infer_steps=max_infer_steps,
                    replay_done=False,
                    wait_done=False,
                    infer_done=False,
                    init_xyz_err=init_xyz_err,
                    init_rpy_err=init_rpy_err,
                    init_gripper_err=init_gripper_err,
                    infer_vs_annotated_xyz_err=None,
                    infer_vs_annotated_rpy_err=None,
                    infer_vs_annotated_gripper_err=None,
                    video_agent=None,
                    video_wrist=None,
                    failure_detail=failure_detail,
                    crash_kind=None,
                    retry_count=0,
                    duration_sec=time.time() - started_at,
                )

        crash_phase = "replay"
        _set_phase(crash_phase)
        if eval_mode == "seed_plus_replay_then_infer":
            for action in actions:
                stepped = _safe_step(env, np.asarray(action, dtype=np.float64))
                if stepped is None:
                    replay_done = True
                    status = "replay_already_done"
                    reason = "env_terminated_before_or_during_replay_step"
                    failure_detail = "replay_env_terminated"
                    break
                obs, done = stepped
                replay_steps += 1
                if done:
                    replay_done = True
                    status = "replay_already_done"
                    reason = "env_done_during_replay"
                    failure_detail = "replay_env_done"
                    break

        annotated_xyz, annotated_rpy, annotated_gripper = _obs_state(obs)

        crash_phase = "wait"
        _set_phase(crash_phase)
        if not replay_done and int(cfg.num_steps_wait_after_replay) > 0:
            dummy_action = get_libero_dummy_action(cfg.model_family)
            for _ in range(int(cfg.num_steps_wait_after_replay)):
                stepped = _safe_step(env, dummy_action)
                if stepped is None:
                    wait_done = True
                    status = "replay_already_done"
                    reason = "env_terminated_before_or_during_wait_step"
                    failure_detail = "wait_env_terminated"
                    break
                obs, done = stepped
                wait_steps += 1
                if done:
                    wait_done = True
                    status = "replay_already_done"
                    reason = "env_done_during_wait_after_replay"
                    failure_detail = "wait_env_done"
                    break

        crash_phase = "infer"
        _set_phase(crash_phase)
        pending_actions: list[np.ndarray] = []
        if save_video:
            observation = _prepare_observation(obs, cfg.flip_images)
            agent_frames.append(np.array(observation["primary_image"], copy=True))
            if save_wrist_video:
                wrist_frames.append(np.array(observation["wrist_image"], copy=True))

        while not replay_done and not wait_done and infer_steps < max_infer_steps:
            if len(pending_actions) == 0:
                observation = _prepare_observation(obs, cfg.flip_images)
                best_actions, _, _ = _query_best_actions(
                    cfg,
                    observation=observation,
                    task_description=task_description,
                    model=model,
                    planning_model=planning_model,
                    dataset_stats=dataset_stats,
                    worker_pool=worker_pool,
                )
                rollout_horizon = min(int(cfg.num_open_loop_steps), len(best_actions))
                if rollout_horizon <= 0:
                    raise ValueError(f"Invalid num_open_loop_steps={cfg.num_open_loop_steps}")
                pending_actions = best_actions[:rollout_horizon]

            action_to_env = pending_actions.pop(0)
            stepped = _safe_step(env, action_to_env.tolist())
            if stepped is None:
                status = "timeout_failure"
                reason = "env_terminated_before_or_during_infer_step_without_success"
                failure_detail = "infer_env_terminated"
                break
            obs, done = stepped
            infer_steps += 1
            if save_video:
                observation = _prepare_observation(obs, cfg.flip_images)
                agent_frames.append(np.array(observation["primary_image"], copy=True))
                if save_wrist_video:
                    wrist_frames.append(np.array(observation["wrist_image"], copy=True))
            if done:
                infer_done = True
                status = "success"
                reason = "env_done_during_infer"
                failure_detail = "infer_ok"
                break

        if not infer_done and failure_detail is None and not replay_done and not wait_done:
            status = "timeout_failure"
            reason = "max_infer_steps_reached_without_success"
            failure_detail = "infer_timeout"

        crash_phase = "compare_final_state"
        _set_phase(crash_phase)
        final_xyz, final_rpy, final_gripper = _obs_state(obs)
        if annotated_xyz is not None and annotated_rpy is not None and annotated_gripper is not None:
            (
                infer_vs_annotated_xyz_err,
                infer_vs_annotated_rpy_err,
                infer_vs_annotated_gripper_err,
            ) = _state_error_from_triplet(
                final_xyz,
                final_rpy,
                final_gripper,
                annotated_xyz,
                annotated_rpy,
                annotated_gripper,
            )

        crash_phase = "save_video"
        _set_phase(crash_phase)
        if save_video and len(agent_frames) > 0:
            agent_path, wrist_path = _video_paths(
                video_root=video_root,
                model_tag=model_tag,
                source=sample.source,
                suite=suite,
                sample_path=sample.file_path,
                status=status,
                repeat_idx=int(sample.repeat_idx),
            )
            video_agent = _save_video(agent_frames, agent_path, video_fps)
            if save_wrist_video and len(wrist_frames) > 0:
                video_wrist = _save_video(wrist_frames, wrist_path, video_fps)

        if failure_detail is None:
            if status == "replay_already_done":
                failure_detail = "replay_or_wait_task_done"
            elif status == "success":
                failure_detail = "infer_ok"
        if failure_detail is not None:
            failure_detail = f"{failure_detail}|mode={eval_mode}"

        return EvalResult(
            source=sample.source,
            file=str(sample.file_path),
            folder=sample.folder,
            repeat_idx=int(sample.repeat_idx),
            task_suite_name=suite,
            task_id=task_id,
            episode_idx=episode_idx,
            seed=seed,
            status=status,
            reason=reason,
            replay_steps=replay_steps,
            wait_steps=wait_steps,
            infer_steps=infer_steps,
            max_infer_steps=max_infer_steps,
            replay_done=replay_done,
            wait_done=wait_done,
            infer_done=infer_done,
            init_xyz_err=init_xyz_err,
            init_rpy_err=init_rpy_err,
            init_gripper_err=init_gripper_err,
            infer_vs_annotated_xyz_err=infer_vs_annotated_xyz_err,
            infer_vs_annotated_rpy_err=infer_vs_annotated_rpy_err,
            infer_vs_annotated_gripper_err=infer_vs_annotated_gripper_err,
            video_agent=video_agent,
            video_wrist=video_wrist,
            failure_detail=failure_detail,
            crash_kind=None,
            retry_count=0,
            duration_sec=time.time() - started_at,
        )
    finally:
        cfg.seed = original_cfg_seed
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


def _eval_one_sample(
    sample: SampleRef,
    cfg: Any,
    *,
    model: Any,
    planning_model: Any,
    dataset_stats: dict[str, Any],
    worker_pool: WorkerPoolManager | None,
    resize_size: int,
    benchmark_cache: dict[str, Any],
    save_video: bool,
    save_wrist_video: bool,
    video_fps: int,
    video_root: pathlib.Path,
    model_tag: str,
) -> EvalResult:
    last_exc: BaseException | None = None
    last_tb = ""
    max_retries = max(0, int(cfg.max_infra_retries))

    for attempt in range(max_retries + 1):
        phase_ref: dict[str, str] = {"phase": "load_sample"}
        try:
            return _eval_one_sample_attempt(
                sample,
                cfg,
                model=model,
                planning_model=planning_model,
                dataset_stats=dataset_stats,
                worker_pool=worker_pool,
                resize_size=resize_size,
                benchmark_cache=benchmark_cache,
                save_video=save_video,
                save_wrist_video=save_wrist_video,
                video_fps=video_fps,
                video_root=video_root,
                model_tag=model_tag,
                phase_ref=phase_ref,
            )
        except Exception as exc:
            last_exc = exc
            last_tb = traceback.format_exc()
            last_phase = phase_ref.get("phase", "unknown")
            kind = _classify_crash(exc, last_phase)
            if kind != "infrastructure" or attempt >= max_retries:
                return EvalResult(
                    source=sample.source,
                    file=str(sample.file_path),
                    folder=sample.folder,
                    repeat_idx=int(sample.repeat_idx),
                    task_suite_name=None,
                    task_id=None,
                    episode_idx=None,
                    seed=None,
                    status="env_crash",
                    reason=f"phase={last_phase} exception={exc!r}\n{last_tb}",
                    replay_steps=0,
                    wait_steps=0,
                    infer_steps=0,
                    max_infer_steps=0,
                    replay_done=False,
                    wait_done=False,
                    infer_done=False,
                    init_xyz_err=None,
                    init_rpy_err=None,
                    init_gripper_err=None,
                    infer_vs_annotated_xyz_err=None,
                    infer_vs_annotated_rpy_err=None,
                    infer_vs_annotated_gripper_err=None,
                    video_agent=None,
                    video_wrist=None,
                    failure_detail="exception",
                    crash_kind=kind,
                    retry_count=attempt,
                    duration_sec=None,
                )
            logging.warning(
                "Infrastructure failure attempt %d/%d for %s during %s: %s",
                attempt + 1,
                max_retries + 1,
                sample.file_path,
                last_phase,
                exc,
            )

    last_phase = phase_ref.get("phase", "unknown")
    return EvalResult(
        source=sample.source,
        file=str(sample.file_path),
        folder=sample.folder,
        repeat_idx=int(sample.repeat_idx),
        task_suite_name=None,
        task_id=None,
        episode_idx=None,
        seed=None,
        status="env_crash",
        reason=f"phase={last_phase} exception={last_exc!r}\n{last_tb}",
        replay_steps=0,
        wait_steps=0,
        infer_steps=0,
        max_infer_steps=0,
        replay_done=False,
        wait_done=False,
        infer_done=False,
        init_xyz_err=None,
        init_rpy_err=None,
        init_gripper_err=None,
        infer_vs_annotated_xyz_err=None,
        infer_vs_annotated_rpy_err=None,
        infer_vs_annotated_gripper_err=None,
        video_agent=None,
        video_wrist=None,
        failure_detail="exception",
        crash_kind="unknown",
        retry_count=max_retries,
        duration_sec=None,
    )


def _model_tag_from_cfg(cfg: Any) -> str:
    if getattr(cfg, "manifest_model_tag", "").strip():
        return cfg.manifest_model_tag.strip()
    ckpt_path = str(getattr(cfg, "ckpt_path", "")).strip()
    if ckpt_path:
        return pathlib.Path(ckpt_path).expanduser().name
    return "cosmos_policy"


def _run_name_from_cfg(cfg: Any) -> str:
    if getattr(cfg, "manifest_run_name", "").strip():
        return cfg.manifest_run_name.strip()
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _normalize_eval_mode(eval_mode: str) -> str:
    normalized = str(eval_mode).strip()
    aliases = {
        "seed_plus_replay": "seed_plus_replay_then_infer",
        "seed_plus_replay_then_infer": "seed_plus_replay_then_infer",
        "seed_only": "seed_only_then_infer",
        "seed_only_then_infer": "seed_only_then_infer",
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported eval_mode={eval_mode}")
    return aliases[normalized]


def _validate_manifest_config(cfg: ManifestPolicyEvalConfig) -> None:
    validate_config(cfg)
    if not str(cfg.sample_file_list).strip():
        raise ValueError("sample_file_list must be provided for filtered repro eval.")
    if int(cfg.num_workers) <= 0:
        raise ValueError(f"num_workers must be positive, got {cfg.num_workers}")
    if int(cfg.trials_per_sample) <= 0:
        raise ValueError(f"trials_per_sample must be positive, got {cfg.trials_per_sample}")
    cfg.eval_mode = _normalize_eval_mode(cfg.eval_mode)


def _resolve_local_checkpoint_path(checkpoint_path: str) -> str:
    raw = str(checkpoint_path).strip()
    if not raw:
        return raw
    path_obj = pathlib.Path(raw).expanduser()
    if not path_obj.is_dir():
        return raw

    pt_files = sorted(path_obj.glob("*.pt"))
    if len(pt_files) == 1:
        return str(pt_files[0].resolve())
    if len(pt_files) > 1:
        raise ValueError(
            f"Checkpoint directory contains multiple .pt files; please pass an explicit checkpoint file: {path_obj}"
        )
    raise ValueError(f"Checkpoint directory does not contain a .pt file: {path_obj}")


def _warm_libero_assets_once(samples: list[SampleRef], cfg: Any, benchmark_cache: dict[str, Any]) -> None:
    if not samples:
        return

    lock_path = pathlib.Path("/tmp/cosmos_policy_libero_assets.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    warm_sample = samples[0]
    raw = _load_sample(warm_sample.file_path)
    suite = str(raw["task_suite_name"])
    task_id = int(raw["task_id"])
    seed = int(raw["seed"])

    with open(lock_path, "w", encoding="utf-8") as lock_file:
        logging.info("Waiting for LIBERO asset warmup lock: %s", lock_path)
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        logging.info("Acquired LIBERO asset warmup lock using sample: %s", warm_sample.file_path)
        env = None
        try:
            if suite not in benchmark_cache:
                benchmark_cache[suite] = benchmark.get_benchmark_dict()[suite]()
            task_suite = benchmark_cache[suite]
            task = task_suite.get_task(task_id)
            env, _ = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res, seed=seed)
            logging.info("LIBERO asset warmup completed with suite=%s task_id=%s", suite, task_id)
        finally:
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def eval_libero_from_manifest(cfg: Any) -> float:
    cfg.eval_mode = _normalize_eval_mode(cfg.eval_mode)
    cfg.ckpt_path = _resolve_local_checkpoint_path(cfg.ckpt_path)
    if getattr(cfg, "planning_model_ckpt_path", ""):
        cfg.planning_model_ckpt_path = _resolve_local_checkpoint_path(cfg.planning_model_ckpt_path)
    assert not (cfg.deterministic and cfg.randomize_seed), (
        "Cannot enable both deterministic mode and randomize seed mode!"
    )
    if cfg.deterministic:
        os.environ["DETERMINISTIC"] = "True"
    else:
        os.environ.pop("DETERMINISTIC", None)
    if cfg.use_parallel_inference:
        mp.set_start_method("spawn", force=True)

    if int(cfg.num_workers) <= 0:
        raise ValueError(f"num_workers must be positive, got {cfg.num_workers}")
    if int(cfg.trials_per_sample) <= 0:
        raise ValueError(f"trials_per_sample must be positive, got {cfg.trials_per_sample}")

    run_name = _run_name_from_cfg(cfg)
    out_dir = pathlib.Path(cfg.local_log_dir).expanduser().resolve() / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "eval.log"
    jsonl_path = out_dir / "results.jsonl"
    summary_path = out_dir / "summary.json"

    video_root_base = (
        pathlib.Path(cfg.manifest_video_root).expanduser().resolve()
        if str(cfg.manifest_video_root).strip()
        else (pathlib.Path(cfg.local_log_dir).expanduser().resolve() / "videos")
    )
    video_root = video_root_base / run_name

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path)],
        force=True,
    )

    sample_list_path = pathlib.Path(cfg.sample_file_list).expanduser().resolve()
    filtered_roots = _parse_filtered_roots(getattr(cfg, "filtered_roots", ""))
    completed = _load_completed_results(jsonl_path) if cfg.resume_eval else set()
    samples = _collect_samples_from_file_list(
        sample_list_path,
        filtered_roots=filtered_roots,
        trials_per_sample=int(cfg.trials_per_sample),
        limit_samples=int(cfg.limit_samples),
        num_workers=int(cfg.num_workers),
        worker_id=int(cfg.worker_id),
        completed=completed,
    )

    logging.info("Output dir: %s", out_dir)
    logging.info("Run name: %s", run_name)
    logging.info("Sample file list: %s", sample_list_path)
    logging.info("Sample tasks after sharding/resume: %d", len(samples))
    logging.info("Unique sample files after sharding/resume: %d", len({str(s.file_path) for s in samples}))
    logging.info("Worker id / num workers: %s / %s", cfg.worker_id, cfg.num_workers)
    logging.info("Trials per sample: %s", cfg.trials_per_sample)
    logging.info("Repeat seed step: %s", cfg.repeat_seed_step)
    logging.info("Resume eval: %s", cfg.resume_eval)
    logging.info("Eval mode: %s", cfg.eval_mode)
    logging.info("Wait after replay: %s", cfg.num_steps_wait_after_replay)
    logging.info("Model tag: %s", _model_tag_from_cfg(cfg))
    logging.info("Resolved checkpoint path: %s", cfg.ckpt_path)

    if not samples:
        summary = _to_summary(_load_results(jsonl_path) if cfg.resume_eval else [])
        with open(summary_path, "w", encoding="utf-8") as file_obj:
            json.dump(summary, file_obj, indent=2, ensure_ascii=False)
        logging.info("No pending samples after sharding/resume. Summary saved to %s", summary_path)
        return 0.0

    benchmark_cache: dict[str, Any] = {}
    _warm_libero_assets_once(samples, cfg, benchmark_cache)

    set_seed_everywhere(int(cfg.seed))
    init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path)
    dataset_stats = load_dataset_stats(cfg.dataset_stats_path)

    worker_pool = None
    if cfg.use_parallel_inference:
        available_gpus = [int(gpu.strip()) for gpu in str(cfg.available_gpus).split(",") if gpu.strip()]
        available_gpus = available_gpus[: int(cfg.num_queries_best_of_n)]
        worker_pool = WorkerPoolManager(cfg, dataset_stats, available_gpus)
        model = None
        planning_model = None
    else:
        model, cosmos_config = get_model(cfg)
        assert cfg.chunk_size == cosmos_config.dataloader_train.dataset.chunk_size, (
            "Mismatch found between train and test chunk sizes! "
            f"Train: {cosmos_config.dataloader_train.dataset.chunk_size}, Test: {cfg.chunk_size}"
        )
        if cfg.planning_model_ckpt_path != "":
            planning_model, _ = get_planning_model(cfg)
        else:
            planning_model = None

    resize_size = get_image_resize_size(cfg.model_family)

    if cfg.use_parallel_inference and worker_pool:
        logging.info("Parallel inference enabled on GPUs: %s", available_gpus)
        logging.info("Parallel timeout: %ss", cfg.parallel_timeout)
        logging.info("Multiprocessing start method: %s", mp.get_start_method())
        worker_pool.start_workers()
    else:
        logging.info("Using serial inference (parallel inference disabled)")

    all_results: list[EvalResult] = _load_results(jsonl_path) if cfg.resume_eval else []
    model_tag = _model_tag_from_cfg(cfg)

    try:
        with open(jsonl_path, "a", encoding="utf-8") as jsonl_file:
            for sample in samples:
                result = _eval_one_sample(
                    sample,
                    cfg,
                    model=model,
                    planning_model=planning_model,
                    dataset_stats=dataset_stats,
                    worker_pool=worker_pool,
                    resize_size=resize_size,
                    benchmark_cache=benchmark_cache,
                    save_video=bool(cfg.manifest_save_video),
                    save_wrist_video=bool(cfg.manifest_save_wrist_video),
                    video_fps=int(cfg.manifest_video_fps),
                    video_root=video_root,
                    model_tag=model_tag,
                )
                all_results.append(result)
                jsonl_file.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
                jsonl_file.flush()
                logging.info(
                    "[%s] src=%s rep=%s file=%s | suite=%s task=%s ep=%s | replay=%d wait=%d infer=%d | "
                    "failure_detail=%s crash_kind=%s retry=%s | init(xyz/rpy/g)=%s/%s/%s | video=%s | reason[:120]=%s",
                    result.status,
                    result.source,
                    result.repeat_idx,
                    pathlib.Path(result.file).name,
                    result.task_suite_name,
                    result.task_id,
                    result.episode_idx,
                    result.replay_steps,
                    result.wait_steps,
                    result.infer_steps,
                    result.failure_detail,
                    result.crash_kind,
                    result.retry_count,
                    "None" if result.init_xyz_err is None else f"{result.init_xyz_err:.6f}",
                    "None" if result.init_rpy_err is None else f"{result.init_rpy_err:.6f}",
                    "None" if result.init_gripper_err is None else f"{result.init_gripper_err:.6f}",
                    result.video_agent,
                    (result.reason or "")[:120],
                )

        summary = _to_summary(all_results)
        with open(summary_path, "w", encoding="utf-8") as file_obj:
            json.dump(summary, file_obj, indent=2, ensure_ascii=False)

        logging.info("Summary saved: %s", summary_path)
        logging.info("Status counts: %s", summary["status_counts"])
        logging.info("Success rate: %.4f", summary["success_rate"])
        logging.info(
            "Effective failure: %d (rate=%.4f)",
            summary["effective_failure"],
            summary["effective_failure_rate"],
        )
        return float(summary["success_rate"])
    finally:
        if worker_pool:
            try:
                worker_pool.shutdown()
            except Exception:
                logging.exception("Failed to shut down parallel worker pool cleanly")


@draccus.wrap()
def main(cfg: ManifestPolicyEvalConfig) -> float:
    _validate_manifest_config(cfg)
    return eval_libero_from_manifest(cfg)


if __name__ == "__main__":
    main()
