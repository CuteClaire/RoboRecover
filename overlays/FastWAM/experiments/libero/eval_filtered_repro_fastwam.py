from __future__ import annotations

import argparse
import collections
import json
import logging
import math
import os
import pathlib
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Optional

import hydra
import numpy as np
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.libero.action_ensembler import ActionEnsembler

LIBERO_ENV_RESOLUTION = 256


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
class EvalContext:
    cfg: DictConfig
    model: Any
    processor: Any
    action_horizon: int
    input_w: int
    input_h: int
    model_device: str
    predict_action_chunk: Any
    set_global_seed: Any


def _import_eval_runtime_helpers() -> dict[str, Any]:
    import torch
    from hydra.utils import instantiate

    from experiments.libero.eval_libero_single import (
        _load_model_checkpoint,
        _mixed_precision_to_model_dtype,
        _predict_action_chunk,
        _resolve_dataset_stats_path,
        _resolve_eval_device,
        _resolve_worker_gpu_index,
    )
    from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
    from fastwam.utils.pytorch_utils import set_global_seed

    return {
        "torch": torch,
        "instantiate": instantiate,
        "load_model_checkpoint": _load_model_checkpoint,
        "mixed_precision_to_model_dtype": _mixed_precision_to_model_dtype,
        "predict_action_chunk": _predict_action_chunk,
        "resolve_dataset_stats_path": _resolve_dataset_stats_path,
        "resolve_eval_device": _resolve_eval_device,
        "resolve_worker_gpu_index": _resolve_worker_gpu_index,
        "load_dataset_stats_from_json": load_dataset_stats_from_json,
        "set_global_seed": set_global_seed,
    }


def _import_libero_runtime_helpers() -> dict[str, Any]:
    from libero.libero import benchmark

    from experiments.libero.libero_utils import get_libero_dummy_action, get_libero_env, get_libero_image

    return {
        "benchmark": benchmark,
        "get_libero_dummy_action": get_libero_dummy_action,
        "get_libero_env": get_libero_env,
        "get_libero_image": get_libero_image,
    }


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
        raise ValueError(f"missing required fields for strict replay: {missing}")
    actions = data["actions"]
    if not isinstance(actions, list) or len(actions) == 0:
        raise ValueError("actions empty")
    step = data.get("annotated_step")
    if step is not None:
        step = int(step)
        if step < 0:
            raise ValueError(f"invalid annotated_step: {step}")
        if len(actions) > step + 1:
            data["actions"] = actions[: step + 1]
    return data


def _derive_sample_source(path: pathlib.Path) -> tuple[str, str]:
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


def _collect_samples_from_file_list(list_path: pathlib.Path) -> list[SampleRef]:
    samples: list[SampleRef] = []
    seen: set[tuple[pathlib.Path, int]] = set()
    with open(list_path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            repeat_idx = 0
            path_text = line
            if "\t" in line:
                path_text, repeat_text = line.rsplit("\t", 1)
                path_text = path_text.strip()
                repeat_text = repeat_text.strip()
                if repeat_text:
                    repeat_idx = int(repeat_text)
            p = pathlib.Path(path_text).expanduser().resolve()
            key = (p, repeat_idx)
            if key in seen:
                continue
            seen.add(key)
            if not p.is_file():
                raise FileNotFoundError(f"sample file not found: {p}")
            if p.suffix.lower() != ".json":
                raise ValueError(f"not a json file: {p}")
            source, folder = _derive_sample_source(p)
            samples.append(SampleRef(source=source, folder=folder, file_path=p, repeat_idx=repeat_idx))
    if not samples:
        raise RuntimeError(f"no paths in sample file list: {list_path}")
    return samples


def _video_paths(
    video_root: pathlib.Path,
    model_tag: str,
    source: str,
    suite: str,
    sample_path: pathlib.Path,
    status: str,
    repeat_idx: int,
) -> tuple[pathlib.Path, pathlib.Path]:
    d = video_root / model_tag / source / suite
    d.mkdir(parents=True, exist_ok=True)
    base = sample_path.stem
    repeat_tag = f"rep{int(repeat_idx):02d}"
    return (
        d / f"{base}_{repeat_tag}_{status}_agent.mp4",
        d / f"{base}_{repeat_tag}_{status}_wrist.mp4",
    )


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


def _safe_step(
    env: Any,
    action: list[float] | np.ndarray,
) -> tuple[dict[str, Any], bool] | None:
    if _is_env_terminated(env):
        return None
    try:
        obs, _, done, _ = env.step(action)
        return obs, bool(done)
    except ValueError as e:
        if "executing action in terminated episode" in str(e).lower():
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
        if any(x in msg for x in ("mujoco", "robosuite", "mj_", "mjcf", "simulation")):
            return "simulation"
    return "unknown"


def _counter(items: list[str]) -> dict[str, int]:
    return dict(collections.Counter(items))


def _to_summary(results: list[EvalResult]) -> dict[str, Any]:
    status_counts = _counter([r.status for r in results])
    total = len(results)
    success = status_counts.get("success", 0)
    effective_failure = status_counts.get("timeout_failure", 0) + status_counts.get("env_crash", 0)
    unique_files = len({r.file for r in results})

    def _bucket(key_fn):
        b: dict[str, collections.Counter] = {}
        for r in results:
            k = key_fn(r)
            b.setdefault(k, collections.Counter())[r.status] += 1
        return {k: dict(v) for k, v in b.items()}

    crash_kind_counts = _counter([r.crash_kind or "" for r in results if r.status == "env_crash"])
    mean_infer_steps = float(
        np.mean([r.infer_steps for r in results if r.infer_steps is not None]) if results else 0.0
    )

    return {
        "total_unique_samples": unique_files,
        "total_samples": total,
        "success_count": success,
        "success_rate": (success / total if total else 0.0),
        "status_counts": status_counts,
        "effective_failure": effective_failure,
        "effective_failure_rate": (effective_failure / total if total else 0.0),
        "state_mismatch_rate": (status_counts.get("state_mismatch", 0) / total if total else 0.0),
        "crash_kind_counts": {k: v for k, v in crash_kind_counts.items() if k},
        "mean_infer_steps": mean_infer_steps,
        "by_source": _bucket(lambda r: r.source),
        "by_folder": _bucket(lambda r: r.folder),
        "by_suite": _bucket(lambda r: r.task_suite_name or "unknown"),
    }


def _compose_cfg(task_choice: str, ckpt: str, gpu_id: int, hydra_overrides: list[str]) -> DictConfig:
    config_dir = PROJECT_ROOT / "configs"
    overrides = [
        f"task={task_choice}",
        f"ckpt={ckpt}",
        f"gpu_id={gpu_id}",
    ]
    overrides.extend(hydra_overrides)
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        cfg = compose(config_name="sim_libero.yaml", overrides=overrides)
    return cfg


def _build_eval_context(cfg: DictConfig) -> EvalContext:
    runtime = _import_eval_runtime_helpers()
    torch = runtime["torch"]
    instantiate = runtime["instantiate"]
    worker_gpu_index = runtime["resolve_worker_gpu_index"](cfg)
    if worker_gpu_index is not None:
        torch.cuda.set_device(int(worker_gpu_index))

    if cfg.get("seed") is not None:
        runtime["set_global_seed"](int(cfg.seed), get_worker_init_fn=False)

    model_device = runtime["resolve_eval_device"](cfg)
    model_dtype = runtime["mixed_precision_to_model_dtype"](cfg.get("mixed_precision", "bf16"))
    model = instantiate(cfg.model, model_dtype=model_dtype, device=model_device)
    runtime["load_model_checkpoint"](model, str(cfg.ckpt))
    model = model.to(model_device).eval()

    dataset_stats_path = runtime["resolve_dataset_stats_path"](cfg)
    dataset_stats = runtime["load_dataset_stats_from_json"](str(dataset_stats_path))
    processor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(dataset_stats)

    action_horizon_cfg = cfg.EVALUATION.get("action_horizon", None)
    if action_horizon_cfg is None:
        action_horizon = int(cfg.data.train.num_frames) - 1
    else:
        action_horizon = int(action_horizon_cfg)
    if action_horizon <= 0:
        raise ValueError(f"EVALUATION.action_horizon must be positive, got {action_horizon}")

    video_size = cfg.data.train.get("video_size", [224, 224])
    if len(video_size) != 2:
        raise ValueError(f"data.train.video_size must be [H, W], got {video_size}")
    input_h = int(video_size[0])
    input_w = int(video_size[1])

    return EvalContext(
        cfg=cfg,
        model=model,
        processor=processor,
        action_horizon=action_horizon,
        input_w=input_w,
        input_h=input_h,
        model_device=model_device,
        predict_action_chunk=runtime["predict_action_chunk"],
        set_global_seed=runtime["set_global_seed"],
    )


def _max_infer_steps_for_suite(suite: str, cfg_value: int) -> int:
    if cfg_value > 0:
        return cfg_value
    if suite not in SUITE_MAX_INFER_STEPS:
        raise ValueError(f"unknown task suite for default max steps: {suite}")
    return SUITE_MAX_INFER_STEPS[suite]


def _eval_one_sample_attempt(
    sample: SampleRef,
    ctx: EvalContext,
    *,
    exec_chunk_size: int,
    repeat_seed_step: int,
    wait_after_replay: int,
    max_infer_steps_cfg: int,
    xyz_tol: float,
    rpy_tol: float,
    gripper_tol: float,
    save_video: bool,
    save_wrist_video: bool,
    video_fps: int,
    video_root: pathlib.Path,
    model_tag: str,
    skip_state_mismatch: bool,
    eval_mode: str,
) -> EvalResult:
    started_at = time.time()
    libero_runtime = _import_libero_runtime_helpers()
    original_cfg_seed = None if ctx.cfg.get("seed") is None else int(ctx.cfg.seed)
    if original_cfg_seed is not None:
        effective_seed = int(original_cfg_seed) + int(sample.repeat_idx) * int(repeat_seed_step)
        ctx.cfg.seed = effective_seed
        ctx.set_global_seed(effective_seed, get_worker_init_fn=False)
    crash_phase = "load_sample"
    env = None
    suite: str | None = None
    task_id: int | None = None
    episode_idx: int | None = None
    seed: int | None = None
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

    raw = _load_sample(sample.file_path)
    suite = str(raw["task_suite_name"])
    task_id = int(raw["task_id"])
    episode_idx = int(raw["episode_idx"])
    seed = int(raw["seed"])
    expected_init = raw["initial_state"]
    actions = raw["actions"]
    max_infer_steps = _max_infer_steps_for_suite(suite, max_infer_steps_cfg)

    crash_phase = "benchmark"
    bm = libero_runtime["benchmark"].get_benchmark_dict()
    task_suite = bm[suite]()
    task = task_suite.get_task(task_id)
    task_description = task.language
    initial_states = task_suite.get_task_init_states(task_id)
    if episode_idx < 0 or episode_idx >= len(initial_states):
        raise ValueError(f"episode_idx out of range: {episode_idx}")

    crash_phase = "create_env"
    env, _ = libero_runtime["get_libero_env"](task, LIBERO_ENV_RESOLUTION, seed)
    try:
        crash_phase = "init_state"
        env.reset()
        obs = env.set_init_state(initial_states[episode_idx])
        init_xyz_err, init_rpy_err, init_gripper_err = _state_error(obs, expected_init)

        if not skip_state_mismatch:
            if init_xyz_err > xyz_tol or init_rpy_err > rpy_tol or init_gripper_err > gripper_tol:
                status = "state_mismatch"
                reason = (
                    f"init pose exceeds tol: xyz_err={init_xyz_err:.6f}(tol {xyz_tol}) "
                    f"rpy_err={init_rpy_err:.6f}(tol {rpy_tol}) "
                    f"grip_err={init_gripper_err:.6f}(tol {gripper_tol})"
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

        if eval_mode == "seed_plus_replay":
            crash_phase = "replay"
            for action in actions:
                stepped = _safe_step(env, action)
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
        if not replay_done and wait_after_replay > 0:
            dummy_action = libero_runtime["get_libero_dummy_action"]()
            for _ in range(wait_after_replay):
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
        if not replay_done and not wait_done:
            use_action_ensembler = bool(ctx.cfg.EVALUATION.get("use_action_ensembler", False))
            ensembler = ActionEnsembler() if use_action_ensembler else None
            if ensembler is not None:
                ensembler.reset()

            pending_actions: list[list[float]] = []
            if save_video:
                imgs = libero_runtime["get_libero_image"](obs)
                agent_frames.append(np.array(imgs["image"], copy=True))
                if save_wrist_video:
                    wrist_frames.append(np.array(imgs["wrist_image"], copy=True))

            while infer_steps < max_infer_steps:
                if len(pending_actions) == 0:
                    action_chunk, _, _ = ctx.predict_action_chunk(
                        obs=obs,
                        task_description=task_description,
                        model=ctx.model,
                        processor=ctx.processor,
                        cfg=ctx.cfg,
                        action_horizon=ctx.action_horizon,
                        input_w=ctx.input_w,
                        input_h=ctx.input_h,
                        model_device=ctx.model_device,
                    )
                    rollout_horizon = min(int(exec_chunk_size), int(action_chunk.shape[0]))
                    if rollout_horizon <= 0:
                        raise ValueError(f"invalid exec_chunk_size={exec_chunk_size}")
                    if ensembler is not None:
                        ensembler.add_actions(action_chunk, infer_steps)
                        pending_actions = [
                            ensembler.get_action(ts).tolist()
                            for ts in range(infer_steps, infer_steps + rollout_horizon)
                        ]
                    else:
                        pending_actions = action_chunk[:rollout_horizon].tolist()

                action_to_env = [float(x) for x in pending_actions.pop(0)]
                stepped = _safe_step(env, action_to_env)
                if stepped is None:
                    status = "timeout_failure"
                    reason = "env_terminated_before_or_during_infer_step_without_success"
                    failure_detail = "infer_env_terminated"
                    break
                obs, done = stepped
                infer_steps += 1
                if save_video:
                    imgs = libero_runtime["get_libero_image"](obs)
                    agent_frames.append(np.array(imgs["image"], copy=True))
                    if save_wrist_video:
                        wrist_frames.append(np.array(imgs["wrist_image"], copy=True))
                if done:
                    infer_done = True
                    status = "success"
                    reason = "env_done_during_infer"
                    failure_detail = "infer_ok"
                    break

            if not infer_done and failure_detail is None:
                status = "timeout_failure"
                reason = "max_infer_steps_reached_without_success"
                failure_detail = "infer_timeout"

        crash_phase = "compare_final_state"
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
        if save_video and len(agent_frames) > 0:
            import imageio

            agent_path, wrist_path = _video_paths(
                video_root=video_root,
                model_tag=model_tag,
                source=sample.source,
                suite=suite,
                sample_path=sample.file_path,
                status=status,
                repeat_idx=int(sample.repeat_idx),
            )
            imageio.mimwrite(str(agent_path), [np.asarray(x) for x in agent_frames], fps=video_fps)
            video_agent = str(agent_path)
            if save_wrist_video and len(wrist_frames) > 0:
                imageio.mimwrite(str(wrist_path), [np.asarray(x) for x in wrist_frames], fps=video_fps)
                video_wrist = str(wrist_path)

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
        if original_cfg_seed is not None:
            ctx.cfg.seed = original_cfg_seed
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


def eval_one_sample(
    sample: SampleRef,
    ctx: EvalContext,
    *,
    exec_chunk_size: int,
    repeat_seed_step: int,
    wait_after_replay: int,
    max_infer_steps_cfg: int,
    xyz_tol: float,
    rpy_tol: float,
    gripper_tol: float,
    save_video: bool,
    save_wrist_video: bool,
    video_fps: int,
    video_root: pathlib.Path,
    model_tag: str,
    max_infra_retries: int,
    skip_state_mismatch: bool,
    eval_mode: str,
) -> EvalResult:
    last_exc: BaseException | None = None
    last_tb = ""
    last_phase = "load_sample"

    for attempt in range(max(0, max_infra_retries) + 1):
        try:
            return _eval_one_sample_attempt(
                sample=sample,
                ctx=ctx,
                exec_chunk_size=exec_chunk_size,
                repeat_seed_step=repeat_seed_step,
                wait_after_replay=wait_after_replay,
                max_infer_steps_cfg=max_infer_steps_cfg,
                xyz_tol=xyz_tol,
                rpy_tol=rpy_tol,
                gripper_tol=gripper_tol,
                save_video=save_video,
                save_wrist_video=save_wrist_video,
                video_fps=video_fps,
                video_root=video_root,
                model_tag=model_tag,
                skip_state_mismatch=skip_state_mismatch,
                eval_mode=eval_mode,
            )
        except Exception as e:
            last_exc = e
            last_tb = traceback.format_exc()
            kind = _classify_crash(e, last_phase)
            if kind != "infrastructure" or attempt >= max_infra_retries:
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
                    reason=f"exception={e!r}\n{last_tb}",
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
                "Infrastructure failure attempt %d/%d for %s: %s",
                attempt + 1,
                max_infra_retries + 1,
                sample.file_path,
                e,
            )

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
        reason=f"exception={last_exc!r}\n{last_tb}",
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
        retry_count=max_infra_retries,
        duration_sec=None,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="FastWAM filtered repro: replay annotated steps -> optional wait -> FastWAM infer."
    )
    parser.add_argument("--task-choice", type=str, required=True, help="Hydra task choice, e.g. libero_uncond_2cam224_1e-4")
    parser.add_argument("--ckpt", type=str, default="", help="FastWAM checkpoint path. Optional only with --dry-run.")
    parser.add_argument("--sample-file-list", type=str, required=True, help="Text file containing one JSON sample path per line.")
    parser.add_argument("--output-dir", type=str, default="eval_logs/fastwam_filtered_repro")
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--model-tag", type=str, default="", help="Used in video output paths. Defaults to checkpoint basename.")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--exec-chunk-size", type=int, default=None, help="Open-loop steps before replanning.")
    parser.add_argument("--replan-steps", type=int, default=None, help="Alias for --exec-chunk-size.")
    parser.add_argument(
        "--repeat-seed-step",
        type=int,
        default=1,
        help="Model sampling seed offset per repeat_idx. 0 means all repeats use the same cfg.seed.",
    )
    parser.add_argument("--num-steps-wait-after-replay", type=int, default=0)
    parser.add_argument("--max-infer-steps", type=int, default=-1, help="-1 uses filtered-repro suite defaults.")
    parser.add_argument("--xyz-tol", type=float, default=0.05)
    parser.add_argument("--rpy-tol", type=float, default=0.20)
    parser.add_argument("--gripper-tol", type=float, default=0.05)
    parser.add_argument("--skip-state-mismatch-check", action="store_true")
    parser.add_argument("--max-infra-retries", type=int, default=0)
    parser.add_argument("--no-save-video", action="store_true", help="Disable MP4 export.")
    parser.add_argument("--save-wrist-video", action="store_true")
    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument("--video-root", type=str, default="eval_logs/fastwam_filtered_repro/videos")
    parser.add_argument("--eval-mode", type=str, choices=["seed_plus_replay", "seed_only"], default="seed_plus_replay")
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--limit-samples", type=int, default=-1)
    parser.add_argument("--dry-run", action="store_true", help="Only validate config/sample sharding and exit.")
    return parser


def main():
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    parser = _build_parser()
    args, hydra_overrides = parser.parse_known_args()

    if not args.dry_run and not str(args.ckpt).strip():
        parser.error("--ckpt is required unless --dry-run is set")

    exec_chunk = args.exec_chunk_size
    if args.replan_steps is not None:
        exec_chunk = int(args.replan_steps)

    sample_list_path = pathlib.Path(args.sample_file_list).expanduser().resolve()
    samples = _collect_samples_from_file_list(sample_list_path)
    if args.num_workers > 1:
        wid = args.worker_id % args.num_workers
        samples = [s for i, s in enumerate(samples) if i % args.num_workers == wid]
    if args.limit_samples > 0:
        samples = samples[: args.limit_samples]
    if not samples:
        raise RuntimeError("No matched samples after sharding/filtering.")

    run_name = args.run_name.strip() or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = pathlib.Path(args.output_dir).expanduser().resolve() / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "eval.log"
    jsonl_path = out_dir / "results.jsonl"
    summary_path = out_dir / "summary.json"
    video_root = pathlib.Path(args.video_root).expanduser().resolve() / run_name

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path)],
        force=True,
    )

    model_tag = args.model_tag.strip()
    if not model_tag:
        if str(args.ckpt).strip():
            model_tag = pathlib.Path(args.ckpt).expanduser().name
        else:
            model_tag = "fastwam_dry_run"

    ckpt_for_cfg = str(args.ckpt).strip() or "__DRY_RUN__"
    cfg = _compose_cfg(
        task_choice=args.task_choice,
        ckpt=ckpt_for_cfg,
        gpu_id=int(args.gpu_id),
        hydra_overrides=hydra_overrides,
    )

    if exec_chunk is None:
        exec_chunk = int(cfg.EVALUATION.get("replan_steps", 10))
    cfg.EVALUATION.replan_steps = int(exec_chunk)

    logging.info("Output dir: %s", out_dir)
    logging.info("Run name: %s", run_name)
    logging.info("Sample tasks after sharding: %d", len(samples))
    logging.info("Unique sample files after sharding: %d", len({str(s.file_path) for s in samples}))
    logging.info("Task choice: %s", args.task_choice)
    logging.info("Checkpoint: %s", args.ckpt if args.ckpt else "<dry-run>")
    logging.info("GPU id: %s", args.gpu_id)
    logging.info("Exec chunk size: %s", exec_chunk)
    logging.info("Repeat seed step: %s", args.repeat_seed_step)
    logging.info("Wait after replay: %s", args.num_steps_wait_after_replay)
    logging.info("Eval mode: %s", args.eval_mode)
    logging.info("Hydra overrides: %s", hydra_overrides)

    if args.dry_run:
        dry_run_summary = {
            "run_name": run_name,
            "task_choice": args.task_choice,
            "sample_file_list": str(sample_list_path),
            "num_sample_tasks": len(samples),
            "num_unique_samples": len({str(s.file_path) for s in samples}),
            "worker_id": args.worker_id,
            "num_workers": args.num_workers,
            "first_sample": str(samples[0].file_path) if samples else None,
            "first_repeat_idx": int(samples[0].repeat_idx) if samples else None,
            "output_dir": str(out_dir),
            "exec_chunk_size": int(exec_chunk),
            "repeat_seed_step": int(args.repeat_seed_step),
            "eval_mode": args.eval_mode,
            "hydra_overrides": hydra_overrides,
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(dry_run_summary, f, indent=2, ensure_ascii=False)
        logging.info("Dry run complete. Summary saved to %s", summary_path)
        return

    ctx = _build_eval_context(cfg)

    all_results: list[EvalResult] = []
    with open(jsonl_path, "w", encoding="utf-8") as fw:
        for sample in samples:
            result = eval_one_sample(
                sample=sample,
                ctx=ctx,
                exec_chunk_size=int(exec_chunk),
                repeat_seed_step=int(args.repeat_seed_step),
                wait_after_replay=int(args.num_steps_wait_after_replay),
                max_infer_steps_cfg=int(args.max_infer_steps),
                xyz_tol=float(args.xyz_tol),
                rpy_tol=float(args.rpy_tol),
                gripper_tol=float(args.gripper_tol),
                save_video=not args.no_save_video,
                save_wrist_video=bool(args.save_wrist_video),
                video_fps=int(args.video_fps),
                video_root=video_root,
                model_tag=model_tag,
                max_infra_retries=int(args.max_infra_retries),
                skip_state_mismatch=bool(args.skip_state_mismatch_check),
                eval_mode=args.eval_mode,
            )
            all_results.append(result)
            fw.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
            fw.flush()
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
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logging.info("Summary saved: %s", summary_path)
    logging.info("Status counts: %s", summary["status_counts"])
    logging.info("Success rate: %.4f", summary["success_rate"])
    logging.info("Effective failure: %d (rate=%.4f)", summary["effective_failure"], summary["effective_failure_rate"])


if __name__ == "__main__":
    main()
