# Copyright (c) 2026 BeingBeyond Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""
Filtered OOD JSON strict repro for Being-H: replay annotated actions -> optional wait (default 0) -> Policy_Libero.

Outputs match openpi `eval_filtered_repro.py`: results.jsonl, summary.json, eval.log; per-row fields include
failure_detail, crash_kind, retry_count, init_*_err, infer_vs_annotated_*, video_agent paths.

Usage (start run_server_vla first, from Being-H repo root):
  export PYTHONPATH=.:$PWD:/path/to/LIBERO
  python -m BeingH.benchmark.libero.eval_filtered_repro_beingh \\
    --host 127.0.0.1 --port 18880 --model-tag Being-H05-2B_libero

Default filtered roots (existing dirs only): openpi, Being-H, unifolm-vla .../filtered_data/OOD_data
Override with repeated --filtered-root. Env FILTERED_ROOTS=path1:path2 also works when no CLI roots.
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import os
import math
import pathlib
import sys
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any


def _boot(msg: str) -> None:
    """Unbuffered stderr marker for import-time / early main segfaults (logging may never run)."""
    sys.stderr.write(f"[eval_boot] {msg}\n")
    sys.stderr.flush()


_boot("stdlib imports ok")

import imageio
import numpy as np

_boot("imageio, numpy ok")

from libero.libero import benchmark
from libero.libero import get_libero_path

_boot("libero benchmark ok")

from libero.libero.envs import OffScreenRenderEnv

_boot("OffScreenRenderEnv import ok (next: Being-H Policy)")

from BeingH.benchmark.utils.policy import Obses_to_Policy_Obs_Dict
from BeingH.benchmark.utils.policy import Policy_Libero as Policy

_boot("BeingH Policy import ok")

# Use robosuite quat2axisangle — NOT experiments.LIBERO.libero_utils, which does `import tensorflow`
# at module import time. Loading TF before MuJoCo/EGL env creation is a common source of segfaults (exit 139).
import robosuite.utils.transform_utils as _robosuite_T

_boot("robosuite transform_utils ok — module load complete (RLDS writer lazy-imported in main if --save-rlds)")

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256

_DEFAULT_REPO = pathlib.Path(__file__).resolve().parents[3]
DEFAULT_FILTERED_ROOTS = [
    str(_DEFAULT_REPO.parent / "openpi" / "filtered_data" / "OOD_data"),
    str(_DEFAULT_REPO / "filtered_data" / "OOD_data"),
    str(_DEFAULT_REPO.parent / "unifolm-vla" / "filtered_data" / "OOD_data"),
]

SUITE_MAX_STEPS = {
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


@dataclass
class EvalResult:
    source: str
    file: str
    folder: str
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


def _quat_to_rpy(quat_xyzw: np.ndarray) -> np.ndarray:
    # robosuite / LIBERO obs["robot0_eef_quat"] is (x, y, z, w), not (w, x, y, z).
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
    """Angular distance in radians; quaternions (x, y, z, w) robosuite convention."""
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
    gripper_raw = obs["robot0_gripper_qpos"]
    if isinstance(gripper_raw, np.ndarray):
        gripper = float(gripper_raw[0])
    else:
        gripper = float(gripper_raw)
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


def _get_libero_env(task, resolution: int, seed: int):
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=task_bddl_file,
        camera_heights=resolution,
        camera_widths=resolution,
    )
    env.seed(seed)
    return env


def _frame_pair(obs: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    agent = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    return agent, wrist


def _suite_to_rlds_dataset_name(suite: str) -> str:
    return f"{suite}_no_noops"


def _capture_rlds_step(
    *,
    obs: dict[str, Any],
    action: list[float] | np.ndarray,
    task_description: str,
    is_first: bool,
    is_last: bool,
    is_terminal: bool,
    reward: float,
) -> dict[str, Any]:
    agent_img, wrist_img = _frame_pair(obs)
    agent_img = np.asarray(agent_img, dtype=np.uint8)
    wrist_img = np.asarray(wrist_img, dtype=np.uint8)

    eef_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float32).reshape(-1)[:3]
    eef_quat = np.asarray(obs["robot0_eef_quat"], dtype=np.float32).reshape(4)
    eef_axis_angle = np.asarray(_robosuite_T.quat2axisangle(eef_quat), dtype=np.float32).reshape(-1)[:3]

    gripper_qpos = np.asarray(obs.get("robot0_gripper_qpos", np.zeros(2, dtype=np.float32)), dtype=np.float32).reshape(-1)
    if gripper_qpos.size == 1:
        gripper_qpos = np.concatenate([gripper_qpos, gripper_qpos], axis=0)
    if gripper_qpos.size < 2:
        gripper_qpos = np.pad(gripper_qpos, (0, 2 - gripper_qpos.size), mode="constant")
    gripper_qpos = gripper_qpos[:2]

    state8 = np.concatenate([eef_pos, eef_axis_angle, gripper_qpos], axis=0).astype(np.float32)

    joint_pos = np.asarray(obs.get("robot0_joint_pos", np.zeros(7, dtype=np.float32)), dtype=np.float32).reshape(-1)
    if joint_pos.size < 7:
        joint_pos = np.pad(joint_pos, (0, 7 - joint_pos.size), mode="constant")
    joint_state7 = joint_pos[:7]

    action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
    if action_arr.size < 7:
        action_arr = np.pad(action_arr, (0, 7 - action_arr.size), mode="constant")
    action7 = action_arr[:7]

    return {
        "observation": {
            "image": agent_img,
            "wrist_image": wrist_img,
            "state": state8,
            "joint_state": joint_state7,
        },
        "action": action7,
        "discount": 1.0,
        "reward": float(reward),
        "is_first": bool(is_first),
        "is_last": bool(is_last),
        "is_terminal": bool(is_terminal),
        "language_instruction": task_description,
    }


def _log_rlds_first_step_shapes(step: dict[str, Any], *, where: str) -> None:
    """Log dtype/shape of the first captured RLDS step (for debugging layout issues)."""
    obs = step["observation"]
    logging.info(
        "[RLDS debug] first step (%s): image=%s %s wrist=%s %s state=%s joint=%s action=%s",
        where,
        obs["image"].shape,
        obs["image"].dtype,
        obs["wrist_image"].shape,
        obs["wrist_image"].dtype,
        obs["state"].shape,
        obs["joint_state"].shape,
        np.asarray(step["action"]).shape,
    )


def _required_json_fields(data: dict[str, Any]) -> list[str]:
    return ["task_suite_name", "task_id", "episode_idx", "seed", "actions", "initial_state"]


def _load_sample(path: pathlib.Path) -> dict[str, Any]:
    with open(path, "r") as f:
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


def _max_infer_steps_for_suite(suite: str, cfg_value: int) -> int:
    if cfg_value > 0:
        return cfg_value
    if suite not in SUITE_MAX_STEPS:
        raise ValueError(f"unknown task suite for default max steps: {suite}")
    return SUITE_MAX_STEPS[suite]


def _collect_samples(
    filtered_roots: list[pathlib.Path], prefixes: list[str], limit_per_folder: int
) -> list[SampleRef]:
    samples: list[SampleRef] = []
    for root in filtered_roots:
        if not root.is_dir():
            logging.warning("Skipping missing filtered root: %s", root)
            continue
        source = root.parts[-3] if len(root.parts) >= 3 else root.name
        folders = [d for d in sorted(root.iterdir()) if d.is_dir()]
        if prefixes:
            folders = [d for d in folders if any(d.name.startswith(p) for p in prefixes)]
        for folder in folders:
            files = sorted(folder.glob("*.json"))
            if limit_per_folder > 0:
                files = files[:limit_per_folder]
            for f in files:
                samples.append(SampleRef(source=source, folder=folder.name, file_path=f))
    return samples


def _collect_samples_from_file_list(
    list_path: pathlib.Path, filtered_roots: list[pathlib.Path]
) -> list[SampleRef]:
    paths: list[pathlib.Path] = []
    with open(list_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            paths.append(pathlib.Path(line).expanduser().resolve())

    if not paths:
        raise RuntimeError(f"no paths in sample file list: {list_path}")

    roots_resolved = [r.resolve() for r in filtered_roots]
    samples: list[SampleRef] = []
    seen: set[pathlib.Path] = set()
    for p in paths:
        if p in seen:
            continue
        seen.add(p)
        if not p.is_file():
            raise FileNotFoundError(f"sample file not found: {p}")
        if p.suffix.lower() != ".json":
            raise ValueError(f"not a json file: {p}")
        under: pathlib.Path | None = None
        for root in roots_resolved:
            try:
                p.relative_to(root)
                under = root
                break
            except ValueError:
                continue
        if under is None:
            raise ValueError(f"sample path not under any --filtered-root: {p}\nroots={filtered_roots}")
        source = under.parts[-3] if len(under.parts) >= 3 else under.name
        samples.append(SampleRef(source=source, folder=p.parent.name, file_path=p))
    return samples


def _video_paths(
    video_root: pathlib.Path,
    model_tag: str,
    source: str,
    suite: str,
    sample_path: pathlib.Path,
    status: str,
):
    d = video_root / model_tag / source / suite
    d.mkdir(parents=True, exist_ok=True)
    base = sample_path.stem
    return (
        d / f"{base}_{status}_agent.mp4",
        d / f"{base}_{status}_wrist.mp4",
    )


def _classify_crash(exc: BaseException, crash_phase: str) -> str:
    """Return crash_kind: infrastructure | data | simulation | unknown."""
    if isinstance(exc, (ConnectionError, TimeoutError, BrokenPipeError)):
        return "infrastructure"
    if isinstance(exc, (ValueError, KeyError)):
        return "data"
    if isinstance(exc, FileNotFoundError):
        return "data"
    msg = str(exc).lower()
    if "connection" in msg or "timeout" in msg or "refused" in msg or "errno 111" in msg:
        return "infrastructure"
    if "cuda" in msg or "out of memory" in msg or "oom" in msg:
        return "infrastructure"
    if isinstance(exc, OSError) and getattr(exc, "errno", None) in (101, 104, 110, 111):
        return "infrastructure"
    if crash_phase in ("replay", "wait", "infer", "create_env", "init_state", "annotate_state"):
        if any(x in msg for x in ("mujoco", "robosuite", "mj_", "mjcf", "simulation")):
            return "simulation"
    return "unknown"


def _is_env_terminated(env: Any) -> bool:
    """Best-effort probe for terminal flag through nested wrappers."""
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
    *,
    phase: str,
) -> tuple[dict[str, Any], bool] | None:
    """Step once unless environment is already terminated.

    Returns:
      - (obs, done) when stepping succeeds
      - None when the env is already terminated (or becomes terminated before this step)
    """
    if _is_env_terminated(env):
        return None
    try:
        obs, _, done, _ = env.step(action)
        return obs, bool(done)
    except ValueError as e:
        # robosuite raises this when step() is called after episode termination.
        if "executing action in terminated episode" in str(e).lower():
            return None
        raise


def _eval_one_sample_attempt(
    sample: SampleRef,
    policy: Policy,
    libero_obses_to_policy_obs_dict,
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
    phase_ref: dict[str, str],
    save_rlds: bool,
    rlds_seed_plus_replay_mode: str,
    rlds_save_which: str,
    rlds_episodes_by_dataset: dict[str, list[dict[str, Any]]],
    rlds_debug: bool = False,
    save_hdf5: bool = False,
    hdf5_save_which: str = "success_only",
    hdf5_data_dir: pathlib.Path | None = None,
    hdf5_counter: list[int] | None = None,
) -> EvalResult:
    def _set_phase(p: str) -> None:
        phase_ref["phase"] = p

    crash_phase = "load_sample"
    _set_phase(crash_phase)
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
    video_agent = None
    video_wrist = None
    init_xyz_err = None
    init_rpy_err = None
    init_gripper_err = None
    infer_vs_annotated_xyz_err = None
    infer_vs_annotated_rpy_err = None
    infer_vs_annotated_gripper_err = None
    annotated_xyz = None
    annotated_rpy = None
    annotated_gripper = None
    agent_frames: list[np.ndarray] = []
    wrist_frames: list[np.ndarray] = []
    rlds_steps: list[dict[str, Any]] = []
    rlds_step_idx = 0

    save_replay_steps = (save_rlds or save_hdf5) and rlds_seed_plus_replay_mode == "replay_and_infer"
    save_wait_steps = save_replay_steps and wait_after_replay > 0
    save_infer_steps = save_rlds or save_hdf5

    raw = _load_sample(sample.file_path)
    suite = str(raw["task_suite_name"])
    if (save_rlds or save_hdf5) and rlds_debug:
        logging.info(
            "[RLDS debug] sample start file=%s suite=%s mode=%s replay_save=%s wait_save=%s infer_save=%s",
            sample.file_path,
            suite,
            eval_mode,
            save_replay_steps,
            save_wait_steps,
            save_infer_steps,
        )
    task_id = int(raw["task_id"])
    episode_idx = int(raw["episode_idx"])
    seed = int(raw["seed"])
    actions = raw["actions"]
    expected_init = raw["initial_state"]
    max_infer_steps = _max_infer_steps_for_suite(suite, max_infer_steps_cfg)

    crash_phase = "benchmark"
    _set_phase(crash_phase)
    bm = benchmark.get_benchmark_dict()
    task_suite = bm[suite]()
    task = task_suite.get_task(task_id)
    task_description = task.language
    initial_states = task_suite.get_task_init_states(task_id)
    if episode_idx < 0 or episode_idx >= len(initial_states):
        raise ValueError(f"episode_idx out of range: {episode_idx}")

    crash_phase = "create_env"
    _set_phase(crash_phase)
    env = _get_libero_env(task, LIBERO_ENV_RESOLUTION, seed)
    try:
        crash_phase = "init_state"
        _set_phase(crash_phase)
        env.reset()
        obs = env.set_init_state(initial_states[episode_idx])
        init_xyz_err, init_rpy_err, init_gripper_err = _state_error(obs, expected_init)

        if not skip_state_mismatch:
            if init_xyz_err > xyz_tol or init_rpy_err > rpy_tol or init_gripper_err > gripper_tol:
                status = "state_mismatch"
                reason = (
                    f"init pose exceeds tol: xyz_err={init_xyz_err:.6f}(tol {xyz_tol}) "
                    f"rpy_err={init_rpy_err:.6f}(tol {rpy_tol}) grip_err={init_gripper_err:.6f}(tol {gripper_tol})"
                )
                failure_detail = "init_state_mismatch"
                return EvalResult(
                    source=sample.source,
                    file=str(sample.file_path),
                    folder=sample.folder,
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
                )

        if eval_mode == "seed_plus_replay":
            crash_phase = "replay"
            _set_phase(crash_phase)
            for action in actions:
                stepped = _safe_step(env, action, phase=crash_phase)
                if stepped is None:
                    replay_done = True
                    status = "replay_already_done"
                    reason = "env_terminated_before_or_during_replay_step"
                    failure_detail = "replay_env_terminated"
                    break
                if save_replay_steps:
                    rlds_steps.append(
                        _capture_rlds_step(
                            obs=obs,
                            action=action,
                            task_description=task_description,
                            is_first=rlds_step_idx == 0,
                            is_last=False,
                            is_terminal=False,
                            reward=0.0,
                        )
                    )
                    rlds_step_idx += 1
                    if rlds_debug and len(rlds_steps) == 1:
                        _log_rlds_first_step_shapes(rlds_steps[-1], where="replay")
                obs, done = stepped
                replay_steps += 1
                if done:
                    replay_done = True
                    status = "replay_already_done"
                    reason = "env_done_during_replay"
                    failure_detail = "replay_env_done"
                    break

        crash_phase = "annotate_state"
        _set_phase(crash_phase)
        annotated_xyz, annotated_rpy, annotated_gripper = _obs_state(obs)

        crash_phase = "wait"
        _set_phase(crash_phase)
        if not replay_done and wait_after_replay > 0:
            for _ in range(wait_after_replay):
                stepped = _safe_step(env, LIBERO_DUMMY_ACTION, phase=crash_phase)
                if stepped is None:
                    wait_done = True
                    status = "replay_already_done"
                    reason = "env_terminated_before_or_during_wait_step"
                    failure_detail = "wait_env_terminated"
                    break
                if save_wait_steps:
                    rlds_steps.append(
                        _capture_rlds_step(
                            obs=obs,
                            action=LIBERO_DUMMY_ACTION,
                            task_description=task_description,
                            is_first=rlds_step_idx == 0,
                            is_last=False,
                            is_terminal=False,
                            reward=0.0,
                        )
                    )
                    rlds_step_idx += 1
                    if rlds_debug and len(rlds_steps) == 1:
                        _log_rlds_first_step_shapes(rlds_steps[-1], where="wait")
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
        if not replay_done and not wait_done:
            policy.reset()
            if save_video:
                a, w = _frame_pair(obs)
                agent_frames.append(a)
                if save_wrist_video:
                    wrist_frames.append(w)
            for _ in range(max_infer_steps):
                obs_dict = libero_obses_to_policy_obs_dict(obs, task_description)
                action = policy.get_action(obs_dict)
                action = np.asarray(action, dtype=np.float64).copy()
                action[-1] = -2 * action[-1] + 1
                action_list = action.tolist() if hasattr(action, "tolist") else list(action)
                if save_infer_steps:
                    rlds_steps.append(
                        _capture_rlds_step(
                            obs=obs,
                            action=action_list,
                            task_description=task_description,
                            is_first=rlds_step_idx == 0,
                            is_last=False,
                            is_terminal=False,
                            reward=0.0,
                        )
                    )
                    rlds_step_idx += 1
                    if rlds_debug and len(rlds_steps) == 1:
                        _log_rlds_first_step_shapes(rlds_steps[-1], where="infer")
                stepped = _safe_step(env, action_list, phase=crash_phase)
                if stepped is None:
                    status = "timeout_failure"
                    reason = "env_terminated_before_or_during_infer_step_without_success"
                    failure_detail = "infer_env_terminated"
                    break
                obs, done = stepped
                infer_steps += 1
                if save_video:
                    a, w = _frame_pair(obs)
                    agent_frames.append(a)
                    if save_wrist_video:
                        wrist_frames.append(w)
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
        _set_phase(crash_phase)
        final_xyz, final_rpy, final_gripper = _obs_state(obs)
        infer_vs_annotated_xyz_err, infer_vs_annotated_rpy_err, infer_vs_annotated_gripper_err = (
            _state_error_from_triplet(
                final_xyz,
                final_rpy,
                final_gripper,
                annotated_xyz,
                annotated_rpy,
                annotated_gripper,
            )
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

        # RLDS buffer + optional per-episode HDF5 (same trajectory layout as RLDS capture).
        if save_rlds or save_hdf5:
            if not rlds_steps:
                if rlds_debug:
                    logging.info(
                        "[RLDS debug] no steps captured (skip buffer): file=%s status=%s infer_steps=%s",
                        sample.file_path,
                        status,
                        infer_steps,
                    )
            else:
                should_save_rlds = False
                should_save_hdf5 = False
                if save_rlds:
                    if rlds_save_which == "success_only":
                        should_save_rlds = status == "success"
                    elif rlds_save_which == "success_and_error":
                        should_save_rlds = (status == "success") or (status != "success" and infer_steps > 0)
                    else:
                        raise ValueError(f"unknown rlds_save_which: {rlds_save_which}")

                if rlds_debug:
                    logging.info(
                        "[RLDS debug] episode decision: file=%s n_steps=%d status=%s should_save_rlds=%s save_which=%s",
                        pathlib.Path(sample.file_path).name,
                        len(rlds_steps),
                        status,
                        should_save_rlds,
                        rlds_save_which,
                    )

                if save_hdf5:
                    if hdf5_save_which == "success_only":
                        should_save_hdf5 = status == "success"
                    elif hdf5_save_which == "success_and_error":
                        should_save_hdf5 = (status == "success") or (status != "success" and infer_steps > 0)
                    else:
                        raise ValueError(f"unknown hdf5_save_which: {hdf5_save_which}")

                need_finalize = (save_rlds and should_save_rlds) or (save_hdf5 and should_save_hdf5)
                if need_finalize:
                    reward_final = 1.0 if status == "success" else 0.0
                    rlds_steps[-1]["is_last"] = True
                    rlds_steps[-1]["is_terminal"] = True
                    rlds_steps[-1]["reward"] = float(reward_final)

                    dataset_name = _suite_to_rlds_dataset_name(suite)

                    if save_hdf5 and should_save_hdf5 and hdf5_data_dir is not None and hdf5_counter is not None:
                        try:
                            from experiments.LIBERO.hdf5_writer_for_eval import (
                                next_hdf5_episode_path,
                                write_episode_hdf5,
                            )

                            idx = hdf5_counter[0]
                            out_p = next_hdf5_episode_path(
                                hdf5_data_dir=hdf5_data_dir,
                                dataset_name=dataset_name,
                                episode_index=idx,
                                sample_path=pathlib.Path(sample.file_path),
                                status_tag="succ" if status == "success" else "error",
                            )
                            write_episode_hdf5(
                                out_path=out_p,
                                steps=rlds_steps,
                                episode_metadata={"file_path": str(sample.file_path)},
                                dataset_name=dataset_name,
                            )
                            hdf5_counter[0] = idx + 1
                            logging.info("Wrote HDF5 episode (%s) -> %s", status, out_p)
                        except Exception:
                            logging.exception("HDF5 write failed for %s", sample.file_path)

                    if save_rlds and should_save_rlds:
                        rlds_episodes_by_dataset.setdefault(dataset_name, []).append(
                            {
                                "steps": rlds_steps,
                                "episode_metadata": {"file_path": str(sample.file_path)},
                            }
                        )
                        if rlds_debug:
                            logging.info(
                                "[RLDS debug] appended to in-memory buffer: dataset=%s total_episodes_in_dataset=%d",
                                dataset_name,
                                len(rlds_episodes_by_dataset[dataset_name]),
                            )

        return EvalResult(
            source=sample.source,
            file=str(sample.file_path),
            folder=sample.folder,
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
        )
    finally:
        try:
            env.close()
        except Exception:
            pass


def eval_one_sample(
    sample: SampleRef,
    host: str,
    port: int,
    exec_chunk_size: int,
    action_type: str,
    data_config_name: str,
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
    save_rlds: bool,
    rlds_seed_plus_replay_mode: str,
    rlds_save_which: str,
    rlds_episodes_by_dataset: dict[str, list[dict[str, Any]]],
    rlds_debug: bool = False,
    save_hdf5: bool = False,
    hdf5_save_which: str = "success_only",
    hdf5_data_dir: pathlib.Path | None = None,
    hdf5_counter: list[int] | None = None,
) -> EvalResult:
    libero_obses_to_policy_obs_dict = Obses_to_Policy_Obs_Dict[data_config_name]

    last_exc: BaseException | None = None
    last_tb = ""
    phase_ref: dict[str, str] = {"phase": "load_sample"}

    for attempt in range(max(0, max_infra_retries) + 1):
        policy = Policy(
            host=host,
            port=port,
            exec_chunk_size=exec_chunk_size,
            action_type=action_type,
        )
        phase_ref = {"phase": "load_sample"}
        try:
            return _eval_one_sample_attempt(
                sample=sample,
                policy=policy,
                libero_obses_to_policy_obs_dict=libero_obses_to_policy_obs_dict,
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
                phase_ref=phase_ref,
                save_rlds=save_rlds,
                rlds_seed_plus_replay_mode=rlds_seed_plus_replay_mode,
                rlds_save_which=rlds_save_which,
                rlds_episodes_by_dataset=rlds_episodes_by_dataset,
                rlds_debug=rlds_debug,
                save_hdf5=save_hdf5,
                hdf5_save_which=hdf5_save_which,
                hdf5_data_dir=hdf5_data_dir,
                hdf5_counter=hdf5_counter,
            )
        except Exception as e:
            last_exc = e
            last_tb = traceback.format_exc()
            crash_phase = phase_ref.get("phase", "unknown")
            kind = _classify_crash(e, crash_phase)
            if kind != "infrastructure" or attempt >= max_infra_retries:
                tb = last_tb
                return EvalResult(
                    source=sample.source,
                    file=str(sample.file_path),
                    folder=sample.folder,
                    task_suite_name=None,
                    task_id=None,
                    episode_idx=None,
                    seed=None,
                    status="env_crash",
                    reason=f"phase={crash_phase} exception={e!r}\n{tb}",
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
                )
            logging.warning(
                "Infrastructure failure attempt %d/%d for %s: %s — retrying",
                attempt + 1,
                max_infra_retries + 1,
                sample.file_path,
                e,
            )

    tb = last_tb or ""
    crash_phase = phase_ref.get("phase", "unknown")
    return EvalResult(
        source=sample.source,
        file=str(sample.file_path),
        folder=sample.folder,
        task_suite_name=None,
        task_id=None,
        episode_idx=None,
        seed=None,
        status="env_crash",
        reason=f"phase={crash_phase} exception={last_exc!r}\n{tb}",
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
        crash_kind="infrastructure",
        retry_count=max_infra_retries,
    )


def _counter(items: list[str]) -> dict[str, int]:
    c = collections.Counter(items)
    return dict(c)


def _to_summary(results: list[EvalResult]) -> dict[str, Any]:
    status_counts = _counter([r.status for r in results])
    total = len(results)
    effective_failure = status_counts.get("timeout_failure", 0) + status_counts.get("env_crash", 0)

    def _bucket(key_fn):
        b: dict[str, collections.Counter] = {}
        for r in results:
            k = key_fn(r)
            b.setdefault(k, collections.Counter())[r.status] += 1
        return {k: dict(v) for k, v in b.items()}

    crash_kind_counts = _counter([r.crash_kind or "" for r in results if r.status == "env_crash"])

    return {
        "total_samples": total,
        "status_counts": status_counts,
        "effective_failure": effective_failure,
        "effective_failure_rate": (effective_failure / total if total else 0.0),
        "state_mismatch_rate": (status_counts.get("state_mismatch", 0) / total if total else 0.0),
        "crash_kind_counts": {k: v for k, v in crash_kind_counts.items() if k},
        "by_source": _bucket(lambda r: r.source),
        "by_folder": _bucket(lambda r: r.folder),
        "by_suite": _bucket(lambda r: r.task_suite_name or "unknown"),
    }


def _resolve_default_roots(cli_roots: list[str]) -> list[pathlib.Path]:
    if cli_roots:
        paths = [pathlib.Path(p).expanduser().resolve() for p in cli_roots]
    elif os.environ.get("FILTERED_ROOTS"):
        paths = [
            pathlib.Path(p.strip()).expanduser().resolve()
            for p in os.environ["FILTERED_ROOTS"].split(":")
            if p.strip()
        ]
    else:
        paths = [pathlib.Path(p).expanduser().resolve() for p in DEFAULT_FILTERED_ROOTS]
    roots: list[pathlib.Path] = []
    for p in paths:
        if p.is_dir():
            roots.append(p)
        else:
            logging.warning("Skipping missing filtered root: %s", p)
    if not roots:
        raise RuntimeError("No valid filtered root directories. Pass --filtered-root or create OOD_data.")
    return roots


def main():
    _boot("main() entered")

    parser = argparse.ArgumentParser(
        description="Being-H filtered repro: replay -> optional wait -> Policy_Libero infer (native server)."
    )
    parser.add_argument("--filtered-root", action="append", default=[], help="Repeatable roots for scanning JSON samples.")
    parser.add_argument("--folder-prefix", action="append", default=[], help="Repeatable folder name prefix filter.")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--exec-chunk-size", type=int, default=8, help="Policy chunk size (matches num_open_loop_steps in standard eval).")
    parser.add_argument(
        "--replan-steps",
        type=int,
        default=None,
        help="Alias for --exec-chunk-size (openpi naming).",
    )
    parser.add_argument("--action-type", type=str, default="world_delta")
    parser.add_argument("--data-config-name", type=str, default="libero")
    parser.add_argument("--num-steps-wait-after-replay", type=int, default=0)
    parser.add_argument("--max-infer-steps", type=int, default=-1, help="-1: suite default max steps.")
    parser.add_argument("--limit-per-folder", type=int, default=-1)
    parser.add_argument("--xyz-tol", type=float, default=0.05)
    parser.add_argument("--rpy-tol", type=float, default=0.20)
    parser.add_argument("--gripper-tol", type=float, default=0.05)
    parser.add_argument(
        "--skip-state-mismatch-check",
        action="store_true",
        help="Do not emit state_mismatch when init error exceeds tolerances.",
    )
    parser.add_argument("--max-infra-retries", type=int, default=2)
    parser.add_argument("--no-save-video", action="store_true", help="Disable MP4 export (default: save agent video).")
    parser.add_argument("--save-wrist-video", action="store_true")
    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument("--video-root", type=str, default="eval_logs/filtered_repro_beingh/videos")
    parser.add_argument("--output-dir", type=str, default="eval_logs/filtered_repro_beingh")
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--model-tag", type=str, default="Being-H05-2B_libero")
    # RLDS trajectory saving (optional)
    parser.add_argument("--save-rlds", action="store_true", help="Save replay/infer step data as RLDS TFRecords.")
    parser.add_argument("--rlds-data-dir", type=str, default="", help="RLDS output root dir. See writer helper.")
    parser.add_argument(
        "--rlds-seed-plus-replay-mode",
        type=str,
        choices=["replay_and_infer", "infer_only_after_replay"],
        default="infer_only_after_replay",
        help="Whether to save replay steps too, or only save from infer-start onward.",
    )
    parser.add_argument(
        "--rlds-save-which",
        type=str,
        choices=["success_only", "success_and_error"],
        default="success_only",
        help="Whether to save only successful episodes, or success + infer failures.",
    )
    parser.add_argument("--rlds-num-shards", type=int, default=1)
    parser.add_argument("--rlds-overwrite", action="store_true", help="Overwrite RLDS output directories.")
    parser.add_argument(
        "--rlds-debug",
        action="store_true",
        help="Verbose [RLDS debug] logs: first-step shapes, buffer decisions, write errors (Python only).",
    )
    parser.add_argument(
        "--save-hdf5",
        action="store_true",
        help="Stream selected episodes to separate HDF5 files (RLDS-compatible layout, no TensorFlow).",
    )
    parser.add_argument(
        "--hdf5-save-which",
        type=str,
        choices=["success_only", "success_and_error"],
        default="success_only",
        help="Whether HDF5 saves only successful episodes, or success + infer failures.",
    )
    parser.add_argument(
        "--hdf5-data-dir",
        type=str,
        default="",
        help="Root directory for per-episode .h5 files (subdirs per dataset_name). Required when --save-hdf5.",
    )
    parser.add_argument(
        "--eval-mode",
        type=str,
        choices=["seed_plus_replay", "seed_only"],
        default="seed_plus_replay",
        help="seed_plus_replay: replay annotated steps then infer; seed_only: infer directly from init state.",
    )
    parser.add_argument("--sample-file-list", type=str, default="")
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    args = parser.parse_args()
    if args.save_hdf5 and not str(args.hdf5_data_dir).strip():
        parser.error("--hdf5-data-dir is required when --save-hdf5")

    _boot("main(): argparse parsed")

    exec_chunk = args.exec_chunk_size
    if args.replan_steps is not None:
        exec_chunk = int(args.replan_steps)

    roots = _resolve_default_roots(list(args.filtered_root or []))

    run_name = args.run_name.strip() or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = pathlib.Path(args.output_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "eval.log"
    jsonl_path = out_dir / "results.jsonl"
    summary_path = out_dir / "summary.json"
    video_root = pathlib.Path(args.video_root) / run_name
    save_video = not args.no_save_video
    if save_video:
        video_root.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path)],
    )
    _boot(f"logging configured -> {log_path}")
    logging.info("Output dir: %s", out_dir)
    logging.info("Roots: %s", [str(r) for r in roots])
    logging.info("Prefixes: %s", args.folder_prefix)
    logging.info(
        "host=%s port=%d wait_after_replay=%d max_infer=%d save_video=%s wrist=%s eval_mode=%s save_rlds=%s save_hdf5=%s rlds_debug=%s",
        args.host,
        args.port,
        args.num_steps_wait_after_replay,
        args.max_infer_steps,
        save_video,
        args.save_wrist_video,
        args.eval_mode,
        args.save_rlds,
        args.save_hdf5,
        args.rlds_debug,
    )

    sample_list_path = args.sample_file_list.strip()
    if sample_list_path:
        samples = _collect_samples_from_file_list(pathlib.Path(sample_list_path), roots)
    else:
        samples = _collect_samples(
            filtered_roots=roots,
            prefixes=args.folder_prefix,
            limit_per_folder=args.limit_per_folder,
        )
    if not samples:
        raise RuntimeError("No matched samples.")

    if args.num_workers > 1:
        wid = args.worker_id % args.num_workers
        samples = [s for i, s in enumerate(samples) if i % args.num_workers == wid]
        logging.info("Worker shard id=%s num_workers=%s -> %d samples", args.worker_id, args.num_workers, len(samples))

    logging.info("Total samples: %d", len(samples))

    all_results: list[EvalResult] = []
    save_rlds = bool(args.save_rlds)
    save_hdf5 = bool(args.save_hdf5)
    rlds_data_dir = pathlib.Path(args.rlds_data_dir) if args.rlds_data_dir else (out_dir / "rlds")
    hdf5_data_dir = pathlib.Path(args.hdf5_data_dir).expanduser().resolve() if save_hdf5 else None
    hdf5_counter: list[int] = [0] if save_hdf5 else []
    rlds_episodes_by_dataset: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)

    with open(jsonl_path, "w") as fw:
        for s in samples:
            result = eval_one_sample(
                sample=s,
                host=args.host,
                port=args.port,
                exec_chunk_size=exec_chunk,
                action_type=args.action_type,
                data_config_name=args.data_config_name,
                wait_after_replay=args.num_steps_wait_after_replay,
                max_infer_steps_cfg=args.max_infer_steps,
                xyz_tol=args.xyz_tol,
                rpy_tol=args.rpy_tol,
                gripper_tol=args.gripper_tol,
                save_video=save_video,
                save_wrist_video=args.save_wrist_video,
                video_fps=args.video_fps,
                video_root=video_root,
                model_tag=args.model_tag,
                max_infra_retries=args.max_infra_retries,
                skip_state_mismatch=args.skip_state_mismatch_check,
                eval_mode=args.eval_mode,
                save_rlds=save_rlds,
                rlds_seed_plus_replay_mode=args.rlds_seed_plus_replay_mode,
                rlds_save_which=args.rlds_save_which,
                rlds_episodes_by_dataset=rlds_episodes_by_dataset,
                rlds_debug=bool(args.rlds_debug),
                save_hdf5=save_hdf5,
                hdf5_save_which=args.hdf5_save_which,
                hdf5_data_dir=hdf5_data_dir,
                hdf5_counter=hdf5_counter if save_hdf5 else None,
            )

            all_results.append(result)
            fw.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
            fw.flush()

            logging.info(
                "[%s] src=%s file=%s | suite=%s task=%s ep=%s | replay=%d wait=%d infer=%d | "
                "failure_detail=%s crash_kind=%s retry=%s | init(xyz/rpy/g)=%s/%s/%s | video=%s | reason[:120]=%s",
                result.status,
                result.source,
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
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logging.info("Summary saved: %s", summary_path)
    logging.info("Status counts: %s", summary["status_counts"])
    logging.info("crash_kind_counts: %s", summary.get("crash_kind_counts", {}))
    logging.info("effective_failure=%d (rate=%.3f)", summary["effective_failure"], summary["effective_failure_rate"])

    if save_rlds:
        logging.info(
            "Writing RLDS episodes: datasets=%d to %s (num_shards=%d overwrite=%s)",
            len(rlds_episodes_by_dataset),
            rlds_data_dir,
            args.rlds_num_shards,
            args.rlds_overwrite,
        )
        if args.rlds_debug:
            for ds_name, eps in rlds_episodes_by_dataset.items():
                n_steps = [len(e.get("steps") or []) for e in eps]
                logging.info(
                    "[RLDS debug] pre-write summary: dataset=%s num_episodes=%d step_counts(min/max)=%s/%s",
                    ds_name,
                    len(eps),
                    min(n_steps) if n_steps else 0,
                    max(n_steps) if n_steps else 0,
                )
        try:
            from experiments.LIBERO.rlds_writer_for_eval import write_rlds_episodes

            write_rlds_episodes(
                episodes_by_dataset=rlds_episodes_by_dataset,
                rlds_data_dir=rlds_data_dir,
                num_shards=args.rlds_num_shards,
                overwrite=bool(args.rlds_overwrite),
            )
            logging.info("RLDS TFRecord write finished: %s", rlds_data_dir)
        except Exception:
            logging.exception("[RLDS debug] write_rlds_episodes failed (Python exception)")
            raise


if __name__ == "__main__":
    _boot("__main__: calling main()")
    main()
