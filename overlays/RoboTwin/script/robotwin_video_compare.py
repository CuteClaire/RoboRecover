"""Exact frame-by-frame video comparison for RoboTwin replay verification."""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class VideoCompareResult:
    camera: str
    ref_path: str
    replay_path: str
    match: bool
    compared_frames: int
    width: int | None = None
    height: int | None = None
    ref_total_frames: int | None = None
    replay_total_frames: int | None = None
    first_mismatch_frame: int | None = None
    reason: str = ""
    pixel_tol: int = 0
    exact_match_frames: int | None = None
    tolerant_match_frames: int | None = None
    first_tolerant_mismatch_frame: int | None = None
    max_pixel_diff: int | None = None
    mean_pixel_diff: float | None = None
    worst_frame: int | None = None
    worst_frame_max_diff: int | None = None
    worst_frame_mean_diff: float | None = None
    worst_frame_pct_pixels_diff: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _run_command(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)


def probe_video(path: Path) -> dict[str, Any]:
    out = _run_command(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_packets",
            "-show_entries",
            "stream=width,height,nb_read_packets,r_frame_rate",
            "-of",
            "json",
            str(path),
        ]
    )
    payload = json.loads(out)
    streams = payload.get("streams") or []
    if not streams:
        raise ValueError(f"no video stream in {path}")
    stream = streams[0]
    frame_count = stream.get("nb_read_packets")
    frame_count = int(frame_count) if frame_count is not None else None
    width = int(stream["width"])
    height = int(stream["height"])
    fps = None
    rate = stream.get("r_frame_rate", "0/1")
    if "/" in rate:
        num, den = rate.split("/", 1)
        if float(den) != 0:
            fps = float(num) / float(den)
    return {
        "width": width,
        "height": height,
        "frame_count": frame_count,
        "fps": fps,
    }


def decode_rgb_frames(path: Path, *, num_frames: int, width: int, height: int) -> np.ndarray:
    frame_bytes = width * height * 3
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-frames:v",
        str(num_frames),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=True)
    raw = proc.stdout
    expected = frame_bytes * num_frames
    if len(raw) < expected:
        actual_frames = len(raw) // frame_bytes
        raise ValueError(
            f"{path} decoded {actual_frames} frames, expected {num_frames} "
            f"(bytes {len(raw)} < {expected})"
        )
    raw = raw[:expected]
    return np.frombuffer(raw, dtype=np.uint8).reshape(num_frames, height, width, 3)


def _frame_diff_stats(ref_frame: np.ndarray, replay_frame: np.ndarray) -> dict[str, float | int]:
    diff = np.abs(ref_frame.astype(np.int16) - replay_frame.astype(np.int16))
    total_pixels = diff.size
    changed = int(np.count_nonzero(diff))
    return {
        "max_diff": int(diff.max()),
        "mean_diff": float(diff.mean()),
        "pct_pixels_diff": 100.0 * changed / total_pixels if total_pixels else 0.0,
    }


def compare_videos_exact(
    ref_path: Path,
    replay_path: Path,
    *,
    camera: str,
    num_frames: int,
    pixel_tol: int = 0,
) -> VideoCompareResult:
    ref_path = ref_path.resolve()
    replay_path = replay_path.resolve()
    if not ref_path.is_file():
        return VideoCompareResult(
            camera=camera,
            ref_path=str(ref_path),
            replay_path=str(replay_path),
            match=False,
            compared_frames=0,
            reason=f"reference video not found: {ref_path}",
        )
    if not replay_path.is_file():
        return VideoCompareResult(
            camera=camera,
            ref_path=str(ref_path),
            replay_path=str(replay_path),
            match=False,
            compared_frames=0,
            reason=f"replay video not found: {replay_path}",
        )
    if num_frames <= 0:
        return VideoCompareResult(
            camera=camera,
            ref_path=str(ref_path),
            replay_path=str(replay_path),
            match=False,
            compared_frames=0,
            reason="num_frames must be positive",
        )

    try:
        ref_meta = probe_video(ref_path)
        replay_meta = probe_video(replay_path)
    except (subprocess.CalledProcessError, ValueError, json.JSONDecodeError) as exc:
        return VideoCompareResult(
            camera=camera,
            ref_path=str(ref_path),
            replay_path=str(replay_path),
            match=False,
            compared_frames=0,
            reason=f"ffprobe failed: {exc}",
        )

    ref_w, ref_h = ref_meta["width"], ref_meta["height"]
    replay_w, replay_h = replay_meta["width"], replay_meta["height"]
    if (ref_w, ref_h) != (replay_w, replay_h):
        return VideoCompareResult(
            camera=camera,
            ref_path=str(ref_path),
            replay_path=str(replay_path),
            match=False,
            compared_frames=0,
            width=ref_w,
            height=ref_h,
            ref_total_frames=ref_meta.get("frame_count"),
            replay_total_frames=replay_meta.get("frame_count"),
            reason=f"resolution mismatch: ref={ref_w}x{ref_h}, replay={replay_w}x{replay_h}",
        )

    ref_total = ref_meta.get("frame_count")
    replay_total = replay_meta.get("frame_count")
    if ref_total is not None and ref_total < num_frames:
        return VideoCompareResult(
            camera=camera,
            ref_path=str(ref_path),
            replay_path=str(replay_path),
            match=False,
            compared_frames=0,
            width=ref_w,
            height=ref_h,
            ref_total_frames=ref_total,
            replay_total_frames=replay_total,
            reason=f"reference has only {ref_total} frames, need {num_frames}",
        )
    if replay_total is not None and replay_total < num_frames:
        return VideoCompareResult(
            camera=camera,
            ref_path=str(ref_path),
            replay_path=str(replay_path),
            match=False,
            compared_frames=0,
            width=ref_w,
            height=ref_h,
            ref_total_frames=ref_total,
            replay_total_frames=replay_total,
            reason=f"replay has only {replay_total} frames, need {num_frames}",
        )

    try:
        ref_frames = decode_rgb_frames(ref_path, num_frames=num_frames, width=ref_w, height=ref_h)
        replay_frames = decode_rgb_frames(replay_path, num_frames=num_frames, width=ref_w, height=ref_h)
    except (subprocess.CalledProcessError, ValueError) as exc:
        return VideoCompareResult(
            camera=camera,
            ref_path=str(ref_path),
            replay_path=str(replay_path),
            match=False,
            compared_frames=0,
            width=ref_w,
            height=ref_h,
            ref_total_frames=ref_total,
            replay_total_frames=replay_total,
            reason=f"decode failed: {exc}",
        )

    first_mismatch = None
    first_tolerant_mismatch = None
    exact_match_frames = 0
    tolerant_match_frames = 0
    per_frame_max: list[int] = []
    per_frame_mean: list[float] = []
    worst_frame = 0
    worst_max = -1

    for frame_idx in range(num_frames):
        stats = _frame_diff_stats(ref_frames[frame_idx], replay_frames[frame_idx])
        per_frame_max.append(stats["max_diff"])
        per_frame_mean.append(stats["mean_diff"])
        if stats["max_diff"] == 0:
            exact_match_frames += 1
        elif first_mismatch is None:
            first_mismatch = frame_idx
        if stats["max_diff"] <= pixel_tol:
            tolerant_match_frames += 1
        elif first_tolerant_mismatch is None:
            first_tolerant_mismatch = frame_idx
        if stats["max_diff"] > worst_max:
            worst_max = stats["max_diff"]
            worst_frame = frame_idx

    tolerant_match = first_tolerant_mismatch is None
    if tolerant_match:
        reason = "exact_match" if first_mismatch is None else f"tolerant_match_within_{pixel_tol}"
    else:
        reason = f"pixel_mismatch_at_frame_{first_tolerant_mismatch}_tol_{pixel_tol}"

    worst_stats = _frame_diff_stats(ref_frames[worst_frame], replay_frames[worst_frame])

    return VideoCompareResult(
        camera=camera,
        ref_path=str(ref_path),
        replay_path=str(replay_path),
        match=tolerant_match,
        compared_frames=num_frames,
        width=ref_w,
        height=ref_h,
        ref_total_frames=ref_total,
        replay_total_frames=replay_total,
        first_mismatch_frame=first_mismatch,
        reason=reason,
        pixel_tol=pixel_tol,
        exact_match_frames=exact_match_frames,
        tolerant_match_frames=tolerant_match_frames,
        first_tolerant_mismatch_frame=first_tolerant_mismatch,
        max_pixel_diff=max(per_frame_max) if per_frame_max else 0,
        mean_pixel_diff=float(np.mean(per_frame_mean)) if per_frame_mean else 0.0,
        worst_frame=worst_frame,
        worst_frame_max_diff=worst_stats["max_diff"],
        worst_frame_mean_diff=worst_stats["mean_diff"],
        worst_frame_pct_pixels_diff=worst_stats["pct_pixels_diff"],
    )


def compare_camera_sets_exact(
    ref_paths: dict[str, Path],
    replay_paths: dict[str, Path],
    *,
    cameras: list[str],
    num_frames: int,
    pixel_tol: int = 0,
) -> dict[str, Any]:
    results: list[VideoCompareResult] = []
    for camera in cameras:
        ref = ref_paths.get(camera)
        replay = replay_paths.get(camera)
        if ref is None:
            results.append(
                VideoCompareResult(
                    camera=camera,
                    ref_path="",
                    replay_path=str(replay) if replay else "",
                    match=False,
                    compared_frames=0,
                    reason=f"missing reference path for {camera}",
                )
            )
            continue
        if replay is None:
            results.append(
                VideoCompareResult(
                    camera=camera,
                    ref_path=str(ref),
                    replay_path="",
                    match=False,
                    compared_frames=0,
                    reason=f"missing replay path for {camera}",
                )
            )
            continue
        results.append(
            compare_videos_exact(ref, replay, camera=camera, num_frames=num_frames, pixel_tol=pixel_tol)
        )
    return {
        "all_cameras_match": all(r.match for r in results),
        "pixel_tol": pixel_tol,
        "num_frames": num_frames,
        "cameras": [r.to_dict() for r in results],
    }
