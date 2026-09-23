#!/usr/bin/env python3
"""
Strict repro eval for filtered OOD JSON: replay annotated actions -> optional dummy wait -> policy infer (websocket).

Usage (from openpi repo root, with policy server already listening):
  export PYTHONPATH=$PWD:$PWD/third_party/libero
  python examples/libero/eval_filtered_repro.py \\
    --host 127.0.0.1 --port 8000 \\
    --model-tag pi05 \\
    --output-dir eval_logs/filtered_repro --run-name myrun

Default data roots (each must exist as a directory; missing roots are skipped):
  openpi/filtered_data/OOD_data, Being-H/filtered_data/OOD_data, unifolm-vla/filtered_data/OOD_data
  Override with one or more --filtered-root PATH (repeatable).

Outputs (under output-dir/run-name/):
  results.jsonl   — one JSON object per sample (same schema for all models in this repo)
  summary.json    — aggregates
  eval.log        — text log
Videos (default ON, disable with --no-save-video):
  under video-root/run-name/<model_tag>/<source>/<suite>/*_agent.mp4 and *_wrist.mp4 if --save-wrist-video

Init rotation check: obs quat is robosuite (x,y,z,w). If initial_state.quat is present, rotation error is
quaternion geodesic (rad), min over (wxyz) vs (xyzw) interpretation; else max(|rpy_delta|) vs stored rpy.

Defaults aligned with batch eval: --num-steps-wait-after-replay 0 (replay then infer immediately).

Optional RoboMonkey verifier (multi-sample chunk selection):
  Start verifier HTTP server (POST /process), then add e.g.
  --use-verifier --verifier-host 127.0.0.1 --verifier-port 3100
  --verifier-n-samples 8 --verifier-score-mode first_action
  Default JSON: image_b64 (base64 PNG), instruction, action (2D list); response rewards.
  Override field names with --verifier-image-field if needed (e.g. image_path for shared FS).
"""
from __future__ import annotations

import argparse
import base64
import collections
import io
import json
import logging
import math
import os
import pathlib
import shutil
import traceback
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

import imageio
import numpy as np
from PIL import Image
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy

def write_rlds_episodes(*args, **kwargs):
    from experiments.LIBERO.rlds_writer_for_eval import write_rlds_episodes as writer
    return writer(*args, **kwargs)

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256

_OPENPI_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _default_filtered_roots() -> list[str]:
    return [
        # str(_OPENPI_ROOT / "filtered_data" / "OOD_data"),
        # str(_OPENPI_ROOT.parent / "Being-H" / "filtered_data" / "OOD_data"),
        str(_OPENPI_ROOT.parent / "unifolm-vla" / "filtered_data" / "OOD_data"),
    ]


def _resolve_filtered_roots(cli_roots: list[str]) -> list[pathlib.Path]:
    """Use explicit --filtered-root if any; else defaults + env FILTERED_ROOTS (colon-separated)."""
    if cli_roots:
        paths = [pathlib.Path(p).expanduser().resolve() for p in cli_roots]
    elif os.environ.get("FILTERED_ROOTS"):
        paths = [
            pathlib.Path(p.strip()).expanduser().resolve()
            for p in os.environ["FILTERED_ROOTS"].split(":")
            if p.strip()
        ]
    else:
        paths = [pathlib.Path(p).expanduser().resolve() for p in _default_filtered_roots()]
    out: list[pathlib.Path] = []
    for p in paths:
        if p.is_dir():
            out.append(p)
        else:
            logging.warning("Skipping missing filtered root: %s", p)
    if not out:
        raise RuntimeError("No valid filtered root directories. Create OOD_data or pass --filtered-root.")
    return out


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
    verifier_best_idx: int | None = None
    verifier_scores: list[float] | None = None


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = math.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


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
    """Angular distance in radians between orientations; quaternions are robosuite (x, y, z, w)."""
    a = np.asarray(q1, dtype=np.float64).reshape(4)
    b = np.asarray(q2, dtype=np.float64).reshape(4)
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    dot = float(np.clip(abs(np.dot(a, b)), 0.0, 1.0))
    return float(2.0 * math.acos(dot))


def _rotation_error_obs_vs_expected_quat(obs_quat_xyzw: np.ndarray, expected_quat: np.ndarray) -> float:
    """JSON may store quat as (x,y,z,w) or (w,x,y,z); use min geodesic vs obs."""
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


def _format_obs(obs: dict[str, Any], task_description: str, resize_size: int) -> dict[str, Any]:
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, resize_size, resize_size))
    wrist_img = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist_img, resize_size, resize_size))
    return {
        "observation/image": img,
        "observation/wrist_image": wrist_img,
        "observation/state": np.concatenate(
            (
                obs["robot0_eef_pos"],
                _quat2axisangle(obs["robot0_eef_quat"].copy()),
                obs["robot0_gripper_qpos"],
            )
        ),
        "prompt": str(task_description),
    }


def _frame_pair(obs: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    agent = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    return agent, wrist

def _is_terminated_episode_error(exc: BaseException) -> bool:
    return isinstance(exc, ValueError) and "executing action in terminated episode" in str(exc).lower()

def _env_already_done(env: Any) -> bool:
    if bool(getattr(env, "done", False)):
        return True
    inner = getattr(env, "env", None)
    if inner is not None and bool(getattr(inner, "done", False)):
        return True
    return False

def _safe_env_step(env: Any, action: list[float] | np.ndarray) -> tuple[tuple[Any, Any, bool, Any] | None, bool]:
    # Some robosuite versions raise ValueError on terminal episodes instead of returning done=True.
    if _env_already_done(env):
        return None, True
    try:
        return env.step(action), False
    except Exception as exc:
        if _is_terminated_episode_error(exc):
            return None, True
        raise


def _verifier_agentview_uint8(obs: dict[str, Any]) -> np.ndarray:
    """Same orientation as _frame_pair agent view (uint8 HxWx3)."""
    return np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])


def _encode_image_png_base64(image_hwc_uint8: np.ndarray) -> str:
    pil_img = Image.fromarray(np.asarray(image_hwc_uint8, dtype=np.uint8))
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _http_post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except HTTPError as e:
        try:
            body = e.read().decode(errors="replace")
        except Exception:
            body = ""
        if not body.strip():
            body = getattr(e, "reason", None) or str(e)
        raise RuntimeError(f"verifier HTTP {e.code}: {body}") from e
    except URLError as e:
        raise RuntimeError(f"verifier connection error: {e}") from e


def _verifier_score_actions(
    process_url: str,
    timeout: float,
    image_uint8: np.ndarray,
    instruction: str,
    actions: np.ndarray,
    *,
    image_field: str,
    instruction_field: str,
    actions_field: str,
) -> np.ndarray:
    """POST /process: default image_b64 (PNG) + action + rewards (works when client/server have no shared FS)."""
    actions = np.clip(np.asarray(actions, dtype=np.float32), -1.0, 1.0)
    payload = {
        instruction_field: instruction,
        image_field: _encode_image_png_base64(image_uint8),
        actions_field: actions.tolist(),
    }
    data = _http_post_json(process_url, payload, timeout)
    if "rewards" in data:
        scores = np.array(data["rewards"], dtype=np.float32)
    elif "scores" in data:
        scores = np.array(data["scores"], dtype=np.float32)
    else:
        raise RuntimeError(f"verifier response missing 'rewards' or 'scores': keys={list(data.keys())}")
    if scores.shape != (len(actions),):
        raise RuntimeError(f"verifier scores shape {scores.shape} expected ({len(actions)},)")
    return scores


def _extract_verifier_rep_actions(chunks: list[np.ndarray], mode: str, k: int) -> np.ndarray:
    reps: list[np.ndarray] = []
    for chunk in chunks:
        c = np.asarray(chunk, dtype=np.float32)
        if mode == "first_action":
            reps.append(c[0])
        elif mode == "mean_k":
            kk = min(k, len(c))
            if kk <= 0:
                raise RuntimeError("empty chunk for mean_k")
            reps.append(c[:kk].mean(axis=0))
        else:
            raise ValueError(f"unknown verifier score mode: {mode!r} (use first_action or mean_k)")
    return np.stack(reps, axis=0)


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
    # Reuse openpi's obs->(image,wrist,state) formatting; then add joint_state + actual action.
    formatted = _format_obs(obs, task_description, resize_size=LIBERO_ENV_RESOLUTION)
    image = np.asarray(formatted["observation/image"], dtype=np.uint8)
    wrist_image = np.asarray(formatted["observation/wrist_image"], dtype=np.uint8)
    state8 = np.asarray(formatted["observation/state"], dtype=np.float32).reshape(-1)[:8]

    joint_pos = np.asarray(obs.get("robot0_joint_pos", np.zeros(7, dtype=np.float32)), dtype=np.float32).reshape(-1)
    if joint_pos.size >= 7:
        joint_state7 = joint_pos[:7]
    else:
        joint_state7 = np.pad(joint_pos, (0, 7 - joint_pos.size), mode="constant")

    action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
    if action_arr.size >= 7:
        action7 = action_arr[:7]
    else:
        action7 = np.pad(action_arr, (0, 7 - action_arr.size), mode="constant")

    return {
        "observation": {
            "image": image,
            "wrist_image": wrist_image,
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


def _trim_replay_actions(actions: list[Any], replay_trim_steps: int) -> list[Any]:
    """Replay only the first len(actions) - replay_trim_steps annotated steps.

    If len(actions) <= replay_trim_steps, skip replay entirely (return empty list).
    """
    trim = int(replay_trim_steps)
    if trim <= 0:
        return actions
    if len(actions) <= trim:
        return []
    return actions[: len(actions) - trim]


def _max_infer_steps_for_suite(suite: str, cfg_value: int) -> int:
    if cfg_value > 0:
        return cfg_value
    if suite not in SUITE_MAX_STEPS:
        raise ValueError(f"unknown task suite for default max steps: {suite}")
    return SUITE_MAX_STEPS[suite]


def _collect_samples(filtered_roots: list[pathlib.Path], prefixes: list[str], limit_per_folder: int) -> list[SampleRef]:
    samples: list[SampleRef] = []
    for root in filtered_roots:
        if not root.is_dir():
            logging.warning("filtered root not found, skip: %s", root)
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


def _collect_samples_from_file_list(list_path: pathlib.Path, filtered_roots: list[pathlib.Path]) -> list[SampleRef]:
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


def _load_existing_results(jsonl_path: pathlib.Path) -> tuple[list[EvalResult], set[str]]:
    existing_results: list[EvalResult] = []
    completed_files: set[str] = set()
    if not jsonl_path.exists() or jsonl_path.stat().st_size == 0:
        return existing_results, completed_files

    valid_fields = set(EvalResult.__dataclass_fields__.keys())
    with open(jsonl_path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if not isinstance(data, dict):
                    raise ValueError("existing result row is not a JSON object")
                result = EvalResult(**{k: data[k] for k in valid_fields if k in data})
            except Exception as exc:
                logging.warning("Skipping malformed existing result line %d in %s: %s", line_no, jsonl_path, exc)
                continue
            existing_results.append(result)
            if result.file:
                completed_files.add(str(pathlib.Path(result.file).expanduser().resolve()))
    return existing_results, completed_files


def _make_unhandled_failure_result(sample: SampleRef, exc: BaseException) -> EvalResult:
    return EvalResult(
        source=sample.source,
        file=str(sample.file_path),
        folder=sample.folder,
        task_suite_name=None,
        task_id=None,
        episode_idx=None,
        seed=None,
        status="env_crash",
        reason=f"unhandled exception={exc!r}\n{traceback.format_exc()}",
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
        retry_count=0,
        verifier_best_idx=None,
        verifier_scores=None,
    )


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
    if isinstance(exc, (ConnectionError, TimeoutError, BrokenPipeError)):
        return "infrastructure"
    if isinstance(exc, (ValueError, KeyError, FileNotFoundError)):
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


def _eval_one_sample_attempt(
    sample: SampleRef,
    client: _websocket_client_policy.WebsocketClientPolicy,
    resize_size: int,
    replan_steps: int,
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
    save_hdf5: bool = False,
    hdf5_save_which: str = "success_only",
    hdf5_data_dir: pathlib.Path | None = None,
    hdf5_counter: list[int] | None = None,
    use_verifier: bool = False,
    verifier_process_url: str = "",
    verifier_timeout: float = 120.0,
    verifier_n_samples: int = 8,
    verifier_score_mode: str = "first_action",
    verifier_score_k: int = 5,
    verifier_image_field: str = "image_b64",
    verifier_instruction_field: str = "instruction",
    verifier_actions_field: str = "action",
    replay_trim_steps: int = 0,
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
    verifier_best_idx: int | None = None
    verifier_scores: list[float] | None = None

    save_replay_steps = (save_rlds or save_hdf5) and rlds_seed_plus_replay_mode == "replay_and_infer"
    save_wait_steps = save_replay_steps and wait_after_replay > 0
    save_infer_steps = save_rlds or save_hdf5

    raw = _load_sample(sample.file_path)
    suite = str(raw["task_suite_name"])
    task_id = int(raw["task_id"])
    episode_idx = int(raw["episode_idx"])
    seed = int(raw["seed"])
    actions = _trim_replay_actions(raw["actions"], replay_trim_steps)
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

                step_out, forced_done = _safe_env_step(env, action)
                replay_steps += 1
                done = bool(forced_done)
                if not done and step_out is not None:
                    obs, _, done, _ = step_out
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
                step_out, forced_done = _safe_env_step(env, LIBERO_DUMMY_ACTION)
                wait_steps += 1
                done = bool(forced_done)
                if not done and step_out is not None:
                    obs, _, done, _ = step_out
                if done:
                    wait_done = True
                    status = "replay_already_done"
                    reason = "env_done_during_wait_after_replay"
                    failure_detail = "wait_env_done"
                    break

        crash_phase = "infer"
        _set_phase(crash_phase)
        infer_ran = False
        if not replay_done and not wait_done:
            infer_ran = True
            action_plan = collections.deque()
            if save_video:
                a, w = _frame_pair(obs)
                agent_frames.append(a)
                if save_wrist_video:
                    wrist_frames.append(w)
            for _ in range(max_infer_steps):
                if not action_plan:
                    element = _format_obs(obs, task_description, resize_size)
                    if use_verifier:
                        if not verifier_process_url:
                            raise RuntimeError("use_verifier=True but verifier_process_url is empty")
                        img_v = _verifier_agentview_uint8(obs)
                        chunks: list[np.ndarray] = []
                        for _vs in range(verifier_n_samples):
                            raw = client.infer(element)["actions"]
                            chunk = np.asarray(raw, dtype=np.float32)
                            chunks.append(chunk)
                        if verifier_score_mode == "all_k_mean":
                            score_list: list[float] = []
                            for ch in chunks:
                                kk = min(verifier_score_k, len(ch))
                                if kk <= 0:
                                    raise RuntimeError("empty chunk for all_k_mean")
                                k_actions = ch[:kk]
                                s = _verifier_score_actions(
                                    verifier_process_url,
                                    verifier_timeout,
                                    img_v,
                                    task_description,
                                    k_actions,
                                    image_field=verifier_image_field,
                                    instruction_field=verifier_instruction_field,
                                    actions_field=verifier_actions_field,
                                )
                                score_list.append(float(np.mean(s)))
                            scores = np.asarray(score_list, dtype=np.float32)
                        elif verifier_score_mode in ("first_action", "mean_k"):
                            rep = _extract_verifier_rep_actions(
                                chunks, verifier_score_mode, verifier_score_k
                            )
                            scores = _verifier_score_actions(
                                verifier_process_url,
                                verifier_timeout,
                                img_v,
                                task_description,
                                rep,
                                image_field=verifier_image_field,
                                instruction_field=verifier_instruction_field,
                                actions_field=verifier_actions_field,
                            )
                        else:
                            raise ValueError(f"unknown verifier_score_mode: {verifier_score_mode!r}")
                        best_idx = int(np.argmax(scores))
                        verifier_best_idx = best_idx
                        verifier_scores = scores.tolist()
                        action_chunk = chunks[best_idx]
                    else:
                        action_chunk = client.infer(element)["actions"]
                    if len(action_chunk) < replan_steps:
                        raise RuntimeError(
                            f"policy output len={len(action_chunk)} < replan_steps={replan_steps}"
                        )
                    action_plan.extend(action_chunk[:replan_steps])

                action = action_plan.popleft()
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
                step_out, forced_done = _safe_env_step(env, action_list)
                infer_steps += 1
                done = bool(forced_done)
                if not done and step_out is not None:
                    obs, _, done, _ = step_out
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
            if infer_ran and not infer_done:
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

        # RLDS buffer + optional per-episode HDF5.
        if save_rlds or save_hdf5:
            if not rlds_steps:
                pass
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
            verifier_best_idx=verifier_best_idx,
            verifier_scores=verifier_scores,
        )
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


def eval_one_sample(
    sample: SampleRef,
    client: _websocket_client_policy.WebsocketClientPolicy,
    resize_size: int,
    replan_steps: int,
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
    save_hdf5: bool = False,
    hdf5_save_which: str = "success_only",
    hdf5_data_dir: pathlib.Path | None = None,
    hdf5_counter: list[int] | None = None,
    use_verifier: bool = False,
    verifier_process_url: str = "",
    verifier_timeout: float = 120.0,
    verifier_n_samples: int = 8,
    verifier_score_mode: str = "first_action",
    verifier_score_k: int = 5,
    verifier_image_field: str = "image_b64",
    verifier_instruction_field: str = "instruction",
    verifier_actions_field: str = "action",
    replay_trim_steps: int = 0,
) -> EvalResult:
    phase_ref: dict[str, str] = {"phase": "load_sample"}
    last_exc: BaseException | None = None
    last_tb = ""

    for attempt in range(max(0, max_infra_retries) + 1):
        phase_ref = {"phase": "load_sample"}
        try:
            return _eval_one_sample_attempt(
                sample=sample,
                client=client,
                resize_size=resize_size,
                replan_steps=replan_steps,
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
                save_hdf5=save_hdf5,
                hdf5_save_which=hdf5_save_which,
                hdf5_data_dir=hdf5_data_dir,
                hdf5_counter=hdf5_counter,
                use_verifier=use_verifier,
                verifier_process_url=verifier_process_url,
                verifier_timeout=verifier_timeout,
                verifier_n_samples=verifier_n_samples,
                verifier_score_mode=verifier_score_mode,
                verifier_score_k=verifier_score_k,
                verifier_image_field=verifier_image_field,
                verifier_instruction_field=verifier_instruction_field,
                verifier_actions_field=verifier_actions_field,
                replay_trim_steps=replay_trim_steps,
            )
        except Exception as e:
            last_exc = e
            last_tb = traceback.format_exc()
            crash_phase = phase_ref.get("phase", "unknown")
            kind = _classify_crash(e, crash_phase)
            if kind != "infrastructure" or attempt >= max_infra_retries:
                return EvalResult(
                    source=sample.source,
                    file=str(sample.file_path),
                    folder=sample.folder,
                    task_suite_name=None,
                    task_id=None,
                    episode_idx=None,
                    seed=None,
                    status="env_crash",
                    reason=f"phase={crash_phase} exception={e!r}\n{last_tb}",
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
        reason=f"phase={crash_phase} exception={last_exc!r}\n{last_tb}",
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

    crash_kind_counts: dict[str, int] = {}
    for r in results:
        if r.status == "env_crash" and r.crash_kind:
            crash_kind_counts[r.crash_kind] = crash_kind_counts.get(r.crash_kind, 0) + 1

    return {
        "total_samples": total,
        "status_counts": status_counts,
        "effective_failure": effective_failure,
        "effective_failure_rate": (effective_failure / total if total else 0.0),
        "state_mismatch_rate": (status_counts.get("state_mismatch", 0) / total if total else 0.0),
        "crash_kind_counts": crash_kind_counts,
        "by_source": _bucket(lambda r: r.source),
        "by_folder": _bucket(lambda r: r.folder),
        "by_suite": _bucket(lambda r: r.task_suite_name or "unknown"),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Strict repro eval: replay -> optional wait -> infer on filtered OOD json (openpi reference implementation)."
    )
    parser.add_argument("--filtered-root", action="append", default=[], help="Repeatable roots; default scans openpi/Being-H/unifolm-vla OOD_data.")
    parser.add_argument("--folder-prefix", action="append", default=[], help="Repeatable folder name prefix filter.")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--num-steps-wait-after-replay", type=int, default=0, help="Dummy steps after replay before infer (default 0).")
    parser.add_argument("--max-infer-steps", type=int, default=-1, help="-1 means use suite default.")
    parser.add_argument("--limit-per-folder", type=int, default=-1)
    parser.add_argument("--xyz-tol", type=float, default=0.05)
    parser.add_argument("--rpy-tol", type=float, default=0.20)
    parser.add_argument("--gripper-tol", type=float, default=0.05)
    parser.add_argument("--skip-state-mismatch-check", action="store_true", help="Do not emit state_mismatch when init error exceeds tol.")
    parser.add_argument("--max-infra-retries", type=int, default=2)
    parser.add_argument("--no-save-video", action="store_true", help="Disable saving agent/wrist MP4 (default: save agent video).")
    parser.add_argument("--save-wrist-video", action="store_true", help="Also save wrist camera MP4.")
    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument("--video-root", type=str, default="eval_logs/filtered_repro/videos")
    parser.add_argument("--output-dir", type=str, default="eval_logs/filtered_repro")
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--model-tag", type=str, default="model")
    parser.add_argument(
        "--use-verifier",
        action="store_true",
        help="Sample N websocket policy chunks; RoboMonkey HTTP /process scores them; execute replan_steps from the best chunk.",
    )
    parser.add_argument("--verifier-host", type=str, default="127.0.0.1")
    parser.add_argument("--verifier-port", type=int, default=3100)
    parser.add_argument(
        "--verifier-timeout",
        type=float,
        default=120.0,
        help="HTTP timeout seconds for each /process call (batch reward can be slow).",
    )
    parser.add_argument("--verifier-n-samples", type=int, default=8)
    parser.add_argument(
        "--verifier-score-mode",
        type=str,
        choices=["first_action", "mean_k", "all_k_mean"],
        default="first_action",
    )
    parser.add_argument("--verifier-score-k", type=int, default=5)
    parser.add_argument(
        "--verifier-image-field",
        type=str,
        default="image_b64",
        help="JSON key for base64-encoded PNG (default image_b64). Use image_path only if server shares filesystem.",
    )
    parser.add_argument("--verifier-instruction-field", type=str, default="instruction")
    parser.add_argument(
        "--verifier-actions-field",
        type=str,
        default="action",
        help="JSON key for 2D action list (RoboMonkey infer_server: action).",
    )
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
        help="Root directory for per-episode .h5 files. Required when --save-hdf5.",
    )
    parser.add_argument(
        "--eval-mode",
        type=str,
        choices=["seed_plus_replay", "seed_only"],
        default="seed_plus_replay",
        help="seed_plus_replay: replay annotated steps then infer; seed_only: infer directly from init state.",
    )
    parser.add_argument(
        "--replay-trim-steps",
        type=int,
        default=0,
        help=(
            "In seed_plus_replay mode, replay only the first (len(actions) - N) annotated steps "
            "(default 0 = full replay). If len(actions) <= N, skip replay entirely."
        ),
    )
    parser.add_argument(
        "--sample-file-list",
        type=str,
        default="",
        help="Optional path: one JSON path per line (must be under a filtered root).",
    )
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    args = parser.parse_args()
    if args.save_hdf5 and not str(args.hdf5_data_dir).strip():
        parser.error("--hdf5-data-dir is required when --save-hdf5")

    roots = _resolve_filtered_roots(list(args.filtered_root or []))

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
    logging.info("Output dir: %s", out_dir)
    logging.info("Roots: %s", [str(r) for r in roots])
    logging.info("Prefixes: %s", args.folder_prefix)
    logging.info(
        "host=%s port=%d replan=%d wait_after_replay=%d max_infer=%d save_video=%s wrist=%s eval_mode=%s",
        args.host,
        args.port,
        args.replan_steps,
        args.num_steps_wait_after_replay,
        args.max_infer_steps,
        save_video,
        args.save_wrist_video,
        args.eval_mode,
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

    existing_results, completed_files = _load_existing_results(jsonl_path)
    resume_existing = jsonl_path.exists() and jsonl_path.stat().st_size > 0
    if resume_existing:
        backup_path = jsonl_path.with_name(f"{jsonl_path.name}.bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        shutil.copy2(jsonl_path, backup_path)
        logging.info("Resuming from existing results: %s (backup=%s)", jsonl_path, backup_path)
        if existing_results:
            samples = [s for s in samples if str(s.file_path.expanduser().resolve()) not in completed_files]
            logging.info("Remaining samples after resume skip: %d", len(samples))
        else:
            logging.warning("Existing results file has no parseable rows; appending new rows without skip filtering")

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    vhost = args.verifier_host
    if vhost in ("0.0.0.0", "::", ""):
        vhost = "127.0.0.1"
    verifier_process_url = f"http://{vhost}:{int(args.verifier_port)}/process"
    if args.use_verifier:
        logging.info(
            "Verifier enabled: url=%s n_samples=%d mode=%s k=%d fields(image/instr/act)=%s/%s/%s",
            verifier_process_url,
            args.verifier_n_samples,
            args.verifier_score_mode,
            args.verifier_score_k,
            args.verifier_image_field,
            args.verifier_instruction_field,
            args.verifier_actions_field,
        )
    all_results: list[EvalResult] = []
    save_rlds = bool(args.save_rlds)
    save_hdf5 = bool(args.save_hdf5)
    rlds_data_dir = pathlib.Path(args.rlds_data_dir) if args.rlds_data_dir else (out_dir / "rlds")
    hdf5_data_dir = pathlib.Path(args.hdf5_data_dir).expanduser().resolve() if save_hdf5 else None
    hdf5_counter: list[int] = [0] if save_hdf5 else []
    rlds_episodes_by_dataset: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)

    if existing_results:
        all_results.extend(existing_results)

    write_mode = "a" if resume_existing else "w"
    with open(jsonl_path, write_mode) as fw:
        for s in samples:
            try:
                result = eval_one_sample(
                    sample=s,
                    client=client,
                    resize_size=args.resize_size,
                    replan_steps=args.replan_steps,
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
                    replay_trim_steps=args.replay_trim_steps,
                    save_rlds=save_rlds,
                    rlds_seed_plus_replay_mode=args.rlds_seed_plus_replay_mode,
                    rlds_save_which=args.rlds_save_which,
                    rlds_episodes_by_dataset=rlds_episodes_by_dataset,
                    save_hdf5=save_hdf5,
                    hdf5_save_which=args.hdf5_save_which,
                    hdf5_data_dir=hdf5_data_dir,
                    hdf5_counter=hdf5_counter if save_hdf5 else None,
                    use_verifier=bool(args.use_verifier),
                    verifier_process_url=verifier_process_url if args.use_verifier else "",
                    verifier_timeout=float(args.verifier_timeout),
                    verifier_n_samples=int(args.verifier_n_samples),
                    verifier_score_mode=str(args.verifier_score_mode),
                    verifier_score_k=int(args.verifier_score_k),
                    verifier_image_field=str(args.verifier_image_field),
                    verifier_instruction_field=str(args.verifier_instruction_field),
                    verifier_actions_field=str(args.verifier_actions_field),
                )
            except Exception as exc:
                logging.exception("Unhandled failure for %s", s.file_path)
                result = _make_unhandled_failure_result(s, exc)

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
        write_rlds_episodes(
            episodes_by_dataset=rlds_episodes_by_dataset,
            rlds_data_dir=rlds_data_dir,
            num_shards=args.rlds_num_shards,
            overwrite=bool(args.rlds_overwrite),
        )


if __name__ == "__main__":
    main()
