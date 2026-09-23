#!/usr/bin/env python3
"""Replay RoboTwin OOD traces, then hand off to an XVLA HTTP policy."""

from __future__ import annotations

import argparse
import base64
import faulthandler
import json
import logging
import os
import sys
import traceback
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import requests
from scipy.spatial.transform import Rotation as R

try:
    import json_numpy
except ModuleNotFoundError:
    json_numpy = None

os.environ.setdefault("ROBOTWIN_SKIP_CUROBO_PLANNER", "0")
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.6")
# The evaluation hosts may have a read-only home cache. Keep Curobo's JIT
# artifacts in a writable per-run location when the caller did not provide one.
if "TORCH_EXTENSIONS_DIR" not in os.environ:
    output_dir = Path("/tmp/ood_xvla_torch_extensions")
    argv = sys.argv[1:]
    for idx, arg in enumerate(argv[:-1]):
        if arg == "--output-dir":
            output_dir = Path(argv[idx + 1]).expanduser() / "_torch_extensions"
            break
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_EXTENSIONS_DIR"] = str(output_dir)
cuda_home = Path(os.environ.get("ROBOTWIN_CUDA_HOME", "/usr/local/cuda"))
if cuda_home.is_dir():
    os.environ.setdefault("CUDA_HOME", str(cuda_home))
    cuda_bin = str(cuda_home / "bin")
    path_parts = os.environ.get("PATH", "").split(os.pathsep)
    if cuda_bin not in path_parts:
        os.environ["PATH"] = os.pathsep.join([cuda_bin, *path_parts])
    cuda_lib = str(cuda_home / "targets/x86_64-linux/lib")
    ld_parts = os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep)
    if cuda_lib not in ld_parts:
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join([cuda_lib, *ld_parts])
faulthandler.enable(all_threads=True)

XVLA_ROOT = Path(__file__).resolve().parents[2]
ROBOTWIN_ROOT = Path(os.environ.get("ROBOTWIN_ROOT", "/path/to/workspace/RoboTwin")).resolve()
ROBOTWIN_SCRIPT_DIR = ROBOTWIN_ROOT / "script"
for path in (ROBOTWIN_SCRIPT_DIR, ROBOTWIN_ROOT, ROBOTWIN_ROOT / "policy", ROBOTWIN_ROOT / "description" / "utils"):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from robotwin_trace_utils import (  # noqa: E402
    DEFAULT_VIDEO_VIEWS,
    InitialStateErrors,
    build_task_args,
    class_decorator,
    discover_trace_json_paths,
    initial_state_error,
    load_trace_json,
    sample_stem_from_path,
    setup_robotwin_paths,
    start_eval_video_recording,
    verify_trace_json,
)
from episode_dataset_recorder import EpisodeDatasetRecorder  # noqa: E402

setup_robotwin_paths()

DEFAULT_JSON_ROOT = Path("/path/to/workspace/disturb-bench/data/RecoverBench_RoboTwin_OOD")


@dataclass
class EvalResult:
    file: str
    model_name: str
    repeat_idx: int
    sample_id: str
    task_name: str
    task_config: str
    seed: int
    episode_idx: int
    instruction: str
    error_type: str | None
    s_stage: str | None
    d_stage: str | None
    status: str
    reason: str
    replay_steps: int
    wait_steps: int
    infer_steps: int
    actions_len: int
    annotated_step: int | None
    init_joint_err: float | None = None
    init_left_endpose_pos_err: float | None = None
    init_left_endpose_rot_err: float | None = None
    init_right_endpose_pos_err: float | None = None
    init_right_endpose_rot_err: float | None = None
    init_left_gripper_err: float | None = None
    init_right_gripper_err: float | None = None
    replay_done: bool = False
    wait_done: bool = False
    infer_done: bool = False
    eval_success_at_end: bool | None = None
    video_paths: dict[str, str] | None = None
    failure_detail: str | None = None
    traceback: str | None = None
    fresh_step_limit_after_replay: bool = False
    env_step_limit_before_replay: int | None = None
    env_step_count_after_replay: int | None = None
    env_step_limit_for_infer: int | None = None


def quat_to_rotate6d(q: np.ndarray) -> np.ndarray:
    return R.from_quat(q).as_matrix()[..., :, :2].reshape(q.shape[:-1] + (6,))


def rotate6d_to_quat(v6: np.ndarray) -> np.ndarray:
    v6 = np.asarray(v6)
    if v6.shape[-1] != 6:
        raise ValueError(f"Last dimension must be 6, got {v6.shape[-1]}")
    a1 = v6[..., 0:5:2]
    a2 = v6[..., 1:6:2]
    b1 = a1 / np.maximum(np.linalg.norm(a1, axis=-1, keepdims=True), 1e-12)
    proj = np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = a2 - proj
    b2 = b2 / np.maximum(np.linalg.norm(b2, axis=-1, keepdims=True), 1e-12)
    b3 = np.cross(b1, b2)
    return R.from_matrix(np.stack((b1, b2, b3), axis=-1)).as_quat()


def obs_to_xvla_proprio(obs: dict[str, Any]) -> np.ndarray:
    left_ee = np.asarray(obs["endpose"]["left_endpose"], dtype=np.float32).reshape(1, 7)
    right_ee = np.asarray(obs["endpose"]["right_endpose"], dtype=np.float32).reshape(1, 7)
    left_grip = np.asarray(obs["endpose"]["left_gripper"], dtype=np.float32).reshape(1)
    right_grip = np.asarray(obs["endpose"]["right_gripper"], dtype=np.float32).reshape(1)
    left_grip = 1.0 - left_grip * 2.0
    right_grip = 1.0 - right_grip * 2.0
    return np.concatenate(
        [
            left_ee[:, :3],
            quat_to_rotate6d(left_ee[:, 3:]),
            left_grip[:, None],
            right_ee[:, :3],
            quat_to_rotate6d(right_ee[:, 3:]),
            right_grip[:, None],
        ],
        axis=-1,
    ).squeeze(0)


def xvla_action_to_robotwin_ee(action: np.ndarray) -> np.ndarray:
    action = np.asarray(action, dtype=np.float64).reshape(-1)
    if action.shape[0] != 20:
        raise ValueError(f"XVLA ee6d action must be 20D, got {action.shape}")
    left_xyz = action[:3].reshape(1, 3)
    left_quat = rotate6d_to_quat(action[3:9].reshape(1, 6))
    left_grip = 1.0 - 2.0 * (action[9:10].reshape(1, 1) > 0.7)
    right_xyz = action[10:13].reshape(1, 3)
    right_quat = rotate6d_to_quat(action[13:19].reshape(1, 6))
    right_grip = 1.0 - 2.0 * (action[19:20].reshape(1, 1) > 0.7)
    return np.concatenate([left_xyz, left_quat, left_grip, right_xyz, right_quat, right_grip], axis=1).reshape(16)


class XVLAHttpPolicy:
    def __init__(
        self,
        host: str,
        port: int,
        *,
        timeout: float,
        domain_id: int,
        steps: int,
        instruction: str,
    ):
        self.url = f"http://{host}:{int(port)}/act"
        self.timeout = float(timeout)
        self.domain_id = int(domain_id)
        self.steps = int(steps)
        self.instruction = instruction
        self.session = requests.Session()

    def close(self) -> None:
        self.session.close()

    @staticmethod
    def dumps_array(array: np.ndarray) -> str:
        array = np.asarray(array)
        if json_numpy is not None:
            return json_numpy.dumps(array)
        return json.dumps(
            {
                "__numpy__": base64.b64encode(np.ascontiguousarray(array).tobytes()).decode("ascii"),
                "dtype": array.dtype.str,
                "shape": list(array.shape),
            }
        )

    def predict(self, obs: dict[str, Any]) -> np.ndarray:
        camera_obs = obs["observation"]
        payload = {
            "domain_id": self.domain_id,
            "proprio": self.dumps_array(obs_to_xvla_proprio(obs)),
            "language_instruction": self.instruction,
            "image0": self.dumps_array(camera_obs["head_camera"]["rgb"]),
            "image1": self.dumps_array(camera_obs["left_camera"]["rgb"]),
            "image2": self.dumps_array(camera_obs["right_camera"]["rgb"]),
            "steps": self.steps,
        }
        response = self.session.post(self.url, json=payload, timeout=self.timeout)
        if not response.ok:
            raise RuntimeError(f"XVLA HTTP {response.status_code}: {response.text[:1000]}")
        data = response.json()
        if "action" not in data:
            raise RuntimeError(f"XVLA response missing action: {data}")
        actions = np.asarray(data["action"], dtype=np.float64)
        if actions.ndim != 2 or actions.shape[1] != 20 or actions.shape[0] == 0:
            raise RuntimeError(f"invalid XVLA action shape: {actions.shape}")
        return actions


def _apply_init_errors(result: EvalResult, errors: InitialStateErrors) -> EvalResult:
    err = errors.to_dict()
    result.init_joint_err = err["joint_err"]
    result.init_left_endpose_pos_err = err["left_endpose_pos_err"]
    result.init_left_endpose_rot_err = err["left_endpose_rot_err"]
    result.init_right_endpose_pos_err = err["right_endpose_pos_err"]
    result.init_right_endpose_rot_err = err["right_endpose_rot_err"]
    result.init_left_gripper_err = err["left_gripper_err"]
    result.init_right_gripper_err = err["right_gripper_err"]
    return result


def _result_from_data(
    path: Path,
    data: dict[str, Any],
    *,
    model_name: str,
    repeat_idx: int,
    status: str,
    reason: str,
    replay_steps: int = 0,
    wait_steps: int = 0,
    infer_steps: int = 0,
    eval_success_at_end: bool | None = None,
    video_paths: dict[str, str] | None = None,
    failure_detail: str | None = None,
    tb: str | None = None,
    fresh_step_limit_after_replay: bool = False,
    env_step_limit_before_replay: int | None = None,
    env_step_count_after_replay: int | None = None,
    env_step_limit_for_infer: int | None = None,
) -> EvalResult:
    return EvalResult(
        file=str(path),
        model_name=model_name,
        repeat_idx=repeat_idx,
        sample_id=sample_stem_from_path(path),
        task_name=data["task_name"],
        task_config=data["task_config"],
        seed=int(data["seed"]),
        episode_idx=int(data.get("episode_idx", 0)),
        instruction=data["instruction"],
        error_type=data.get("error_type"),
        s_stage=data.get("s_stage"),
        d_stage=data.get("d_stage"),
        status=status,
        reason=reason,
        replay_steps=replay_steps,
        wait_steps=wait_steps,
        infer_steps=infer_steps,
        actions_len=len(data["actions"]),
        annotated_step=data.get("annotated_step"),
        replay_done=replay_steps > 0 or status in {"replay_already_done", "infer_ok", "infer_failed", "max_infer_steps_reached"},
        wait_done=wait_steps > 0,
        infer_done=infer_steps > 0,
        eval_success_at_end=eval_success_at_end,
        video_paths=video_paths,
        failure_detail=failure_detail,
        traceback=tb,
        fresh_step_limit_after_replay=fresh_step_limit_after_replay,
        env_step_limit_before_replay=env_step_limit_before_replay,
        env_step_count_after_replay=env_step_count_after_replay,
        env_step_limit_for_infer=env_step_limit_for_infer,
    )


def _closed_gripper_action(obs: dict[str, Any]) -> np.ndarray:
    action = np.zeros_like(np.asarray(obs["joint_action"]["vector"], dtype=np.float64))
    if action.shape[0] >= 2:
        action[-2:] = -1.0
    return action


def _set_video_ffmpeg_closed(task_env) -> None:
    try:
        task_env._del_eval_video_ffmpeg()
    except Exception:
        pass


def _run_xvla_chunk(
    task_env,
    policy: XVLAHttpPolicy,
    infer_trace: dict[str, Any],
    *,
    max_actions: int | None = None,
    recorder: EpisodeDatasetRecorder | None = None,
    infer_call_idx: int = 0,
) -> tuple[int, bool]:
    obs = task_env.get_obs()
    xvla_actions = policy.predict(obs)
    if max_actions is not None:
        xvla_actions = xvla_actions[:max_actions]
    executed = 0
    for chunk_offset, xvla_action in enumerate(xvla_actions):
        ee_action = xvla_action_to_robotwin_ee(xvla_action)
        infer_trace["actions"].append(np.asarray(xvla_action, dtype=np.float64).tolist())
        if recorder is not None:
            recorder.set_action_context(
                phase="infer",
                policy_action_raw=xvla_action,
                infer_call_idx=infer_call_idx,
                chunk_offset=chunk_offset,
            )
        task_env.take_action(ee_action, action_type="ee")
        executed += 1
        if task_env.eval_success:
            return executed, True
        if task_env.take_action_cnt >= task_env.step_lim:
            return executed, False
    return executed, False


def append_result_jsonl(path: Path, result: EvalResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def write_result_json(path: Path, result: EvalResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(result), f, ensure_ascii=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def write_results_jsonl(path: Path, results: list[EvalResult]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")


def build_summary(results: list[EvalResult], *, args: argparse.Namespace) -> dict[str, Any]:
    status_counts = Counter(result.status for result in results)
    task_groups: dict[str, list[EvalResult]] = defaultdict(list)
    error_groups: dict[str, list[EvalResult]] = defaultdict(list)
    failed_files: list[str] = []
    for result in results:
        task_groups[result.task_name].append(result)
        error_groups[result.error_type or "unknown"].append(result)
        if result.status not in {"infer_ok", "replay_ok"}:
            failed_files.append(result.file)

    def summarize(groups: dict[str, list[EvalResult]]) -> dict[str, dict[str, Any]]:
        out = {}
        for key, items in sorted(groups.items()):
            success_count = sum(1 for item in items if bool(item.eval_success_at_end))
            out[key] = {
                "total": len(items),
                "success_count": success_count,
                "success_rate": success_count / len(items) if items else 0.0,
                "status_counts": dict(Counter(item.status for item in items)),
            }
        return out

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "json_root": str(args.json_root) if args.json_root else "",
        "sample_file_list": str(args.sample_file_list) if args.sample_file_list else "",
        "policy_name": args.policy_name,
        "model_name": args.model_name,
        "eval_mode": args.eval_mode,
        "num_steps_wait_after_replay": args.num_steps_wait_after_replay,
        "fresh_step_limit_after_replay": args.fresh_step_limit_after_replay,
        "max_infer_steps": args.max_infer_steps,
        "xvla_steps": args.xvla_steps,
        "xvla_domain_id": args.xvla_domain_id,
        "total": len(results),
        "status_counts": dict(status_counts),
        "task_summary": summarize(task_groups),
        "error_type_summary": summarize(error_groups),
        "failed_files": failed_files,
    }


def write_summary_json(path: Path, results: list[EvalResult], *, args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(build_summary(results, args=args), f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())


def evaluate_one_json(path: Path, args: argparse.Namespace, policy: XVLAHttpPolicy | None) -> EvalResult:
    verify_trace_json(path)
    data = load_trace_json(path)
    task_args = build_task_args(data["task_name"], data["task_config"])
    task_env = None
    recorder: EpisodeDatasetRecorder | None = None
    video_paths = None
    replay_steps = 0
    wait_steps = 0
    infer_steps = 0
    env_step_limit_before_replay = None
    env_step_count_after_replay = None
    env_step_limit_for_infer = None

    def result_from_data(**kwargs) -> EvalResult:
        return _result_from_data(
            path,
            data,
            fresh_step_limit_after_replay=args.fresh_step_limit_after_replay,
            env_step_limit_before_replay=env_step_limit_before_replay,
            env_step_count_after_replay=env_step_count_after_replay,
            env_step_limit_for_infer=env_step_limit_for_infer,
            **kwargs,
        )

    try:
        task_env = class_decorator(data["task_name"])
        task_env.setup_demo(now_ep_num=int(data.get("episode_idx", 0)), seed=int(data["seed"]), is_test=True, **task_args)
        task_env.set_instruction(instruction=data["instruction"])
        if args.save_dataset_episode:
            recorder = EpisodeDatasetRecorder(
                args.output_dir,
                {
                    "source_json": str(path.resolve()),
                    "sample_id": sample_stem_from_path(path),
                    "model_name": args.model_name,
                    "repeat_idx": args.repeat_idx,
                    "task_name": data["task_name"],
                    "task_config": data["task_config"],
                    "seed": int(data["seed"]),
                    "episode_idx": int(data.get("episode_idx", 0)),
                    "instruction": data["instruction"],
                    "error_type": data.get("error_type"),
                },
                image_format=args.dataset_image_format,
            )
            recorder.attach(task_env)
        env_step_limit_before_replay = int(task_env.step_lim)

        if args.save_video and args.output_dir is not None:
            video_paths = start_eval_video_recording(
                task_env,
                args.output_dir,
                sample_stem_from_path(path),
                task_args,
                video_fps=args.video_fps,
                video_views=args.video_views,
            )

        obs = task_env.get_obs()
        init_errors = initial_state_error(obs, data["initial_state"])
        if init_errors.exceeds_tolerance(
            joint_tol=args.joint_tol,
            endpose_pos_tol=args.endpose_pos_tol,
            endpose_rot_tol=args.endpose_rot_tol,
            gripper_tol=args.gripper_tol,
        ):
            if not args.skip_state_mismatch_check:
                result = result_from_data(
                    model_name=args.model_name,
                    repeat_idx=args.repeat_idx,
                    status="state_mismatch",
                    reason="initial_state_exceeds_tolerance",
                    video_paths=video_paths,
                    failure_detail="init_state_mismatch",
                )
                return _apply_init_errors(result, init_errors)
            logging.warning("[%s] initial_state_exceeds_tolerance; continuing due to --skip-state-mismatch-check", path.name)

        if args.eval_mode == "seed_plus_replay":
            replay_action_type = data.get("action_type", "qpos")
            if recorder is not None:
                recorder.set_phase("replay")
            if args.fresh_step_limit_after_replay:
                task_env.step_lim = max(
                    int(task_env.step_lim),
                    int(task_env.take_action_cnt) + len(data["actions"]) + int(args.num_steps_wait_after_replay),
                )
            for action in data["actions"]:
                task_env.take_action(np.asarray(action, dtype=np.float64), action_type=replay_action_type)
                replay_steps += 1
                if task_env.eval_success or (
                    not args.fresh_step_limit_after_replay and task_env.take_action_cnt >= task_env.step_lim
                ):
                    env_step_count_after_replay = int(task_env.take_action_cnt)
                    env_step_limit_for_infer = int(task_env.step_lim)
                    result = result_from_data(
                        model_name=args.model_name,
                        repeat_idx=args.repeat_idx,
                        status="replay_already_done",
                        reason="env_done_during_replay",
                        replay_steps=replay_steps,
                        video_paths=video_paths,
                        eval_success_at_end=bool(task_env.eval_success),
                        failure_detail="replay_env_done",
                    )
                    return _apply_init_errors(result, init_errors)

            obs = task_env.get_obs()
            if recorder is not None:
                recorder.set_phase("replay_wait")
            for _ in range(args.num_steps_wait_after_replay):
                task_env.take_action(_closed_gripper_action(obs))
                wait_steps += 1
                if task_env.eval_success or (
                    not args.fresh_step_limit_after_replay and task_env.take_action_cnt >= task_env.step_lim
                ):
                    env_step_count_after_replay = int(task_env.take_action_cnt)
                    env_step_limit_for_infer = int(task_env.step_lim)
                    result = result_from_data(
                        model_name=args.model_name,
                        repeat_idx=args.repeat_idx,
                        status="replay_already_done",
                        reason="env_done_during_wait_after_replay",
                        replay_steps=replay_steps,
                        wait_steps=wait_steps,
                        video_paths=video_paths,
                        eval_success_at_end=bool(task_env.eval_success),
                        failure_detail="wait_env_done",
                    )
                    return _apply_init_errors(result, init_errors)
                obs = task_env.get_obs()

        env_step_count_after_replay = int(task_env.take_action_cnt)
        env_step_limit_for_infer = int(task_env.step_lim)
        if args.fresh_step_limit_after_replay and args.eval_mode == "seed_plus_replay":
            task_env.take_action_cnt = 0
            env_step_limit_for_infer = env_step_limit_before_replay
            task_env.step_lim = env_step_limit_for_infer

        if args.max_infer_steps == 0:
            result = result_from_data(
                model_name=args.model_name,
                repeat_idx=args.repeat_idx,
                status="replay_ok",
                reason="replay_completed_without_infer",
                replay_steps=replay_steps,
                wait_steps=wait_steps,
                video_paths=video_paths,
                eval_success_at_end=bool(task_env.eval_success),
            )
            return _apply_init_errors(result, init_errors)

        if policy is None:
            raise RuntimeError("XVLA policy is required for infer, but policy was not initialized")

        if recorder is not None:
            recorder.set_phase("infer")
        policy.instruction = data["instruction"]
        infer_trace = {"actions": []}
        max_infer_steps = args.max_infer_steps if args.max_infer_steps > 0 else None
        infer_call_idx = 0
        while task_env.take_action_cnt < task_env.step_lim:
            remaining = None if max_infer_steps is None else max(max_infer_steps - infer_steps, 0)
            if remaining == 0:
                result = result_from_data(
                    model_name=args.model_name,
                    repeat_idx=args.repeat_idx,
                    status="max_infer_steps_reached",
                    reason="infer_budget_exhausted",
                    replay_steps=replay_steps,
                    wait_steps=wait_steps,
                    infer_steps=infer_steps,
                    video_paths=video_paths,
                    eval_success_at_end=bool(task_env.eval_success),
                )
                return _apply_init_errors(result, init_errors)
            executed, success = _run_xvla_chunk(
                task_env,
                policy,
                infer_trace,
                max_actions=remaining,
                recorder=recorder,
                infer_call_idx=infer_call_idx,
            )
            infer_call_idx += 1
            infer_steps += executed
            if success or task_env.eval_success:
                result = result_from_data(
                    model_name=args.model_name,
                    repeat_idx=args.repeat_idx,
                    status="infer_ok",
                    reason="eval_success",
                    replay_steps=replay_steps,
                    wait_steps=wait_steps,
                    infer_steps=infer_steps,
                    video_paths=video_paths,
                    eval_success_at_end=True,
                )
                return _apply_init_errors(result, init_errors)
            if max_infer_steps is not None and infer_steps >= max_infer_steps:
                result = result_from_data(
                    model_name=args.model_name,
                    repeat_idx=args.repeat_idx,
                    status="max_infer_steps_reached",
                    reason="infer_budget_exhausted",
                    replay_steps=replay_steps,
                    wait_steps=wait_steps,
                    infer_steps=infer_steps,
                    video_paths=video_paths,
                    eval_success_at_end=bool(task_env.eval_success),
                )
                return _apply_init_errors(result, init_errors)
            if executed == 0:
                break

        result = result_from_data(
            model_name=args.model_name,
            repeat_idx=args.repeat_idx,
            status="infer_failed",
            reason="env_terminated_without_success",
            replay_steps=replay_steps,
            wait_steps=wait_steps,
            infer_steps=infer_steps,
            video_paths=video_paths,
            eval_success_at_end=bool(task_env.eval_success),
            failure_detail="infer_env_done",
        )
        return _apply_init_errors(result, init_errors)
    except Exception as exc:
        if task_env is not None:
            step_lim = getattr(task_env, "step_lim", None)
            if step_lim is not None:
                try:
                    env_step_limit_for_infer = int(step_lim)
                except Exception:
                    pass
        result = result_from_data(
            model_name=args.model_name,
            repeat_idx=args.repeat_idx,
            status="eval_crash",
            reason=str(exc),
            replay_steps=replay_steps,
            wait_steps=wait_steps,
            infer_steps=infer_steps,
            video_paths=video_paths,
            eval_success_at_end=bool(getattr(task_env, "eval_success", False)) if task_env is not None else None,
            failure_detail=type(exc).__name__,
            tb=traceback.format_exc(),
        )
        return result
    finally:
        if recorder is not None:
            recorded_result = locals().get("result")
            try:
                recorder.finalize(
                    task_env=task_env,
                    status=getattr(recorded_result, "status", "eval_crash"),
                    success=getattr(recorded_result, "eval_success_at_end", None),
                    error=getattr(recorded_result, "reason", None),
                )
            except Exception:
                logging.exception("dataset_recorder_finalize_failed sample=%s", path)
        if task_env is not None:
            if args.save_video:
                _set_video_ffmpeg_closed(task_env)
            if not args.skip_close_env:
                try:
                    task_env.close_env()
                except Exception:
                    logging.exception("failed to close task_env after evaluation")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-path", type=str, default="")
    parser.add_argument("--json-root", type=str, default=str(DEFAULT_JSON_ROOT))
    parser.add_argument("--sample-file-list", type=str, default="")
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--policy-name", type=str, default="xvla")
    parser.add_argument("--model-name", type=str, default="X-VLA-RoboTwin2")
    parser.add_argument("--repeat-idx", type=int, default=1)
    parser.add_argument("--xvla-host", type=str, default="127.0.0.1")
    parser.add_argument("--xvla-port", type=int, default=0)
    parser.add_argument("--xvla-timeout", type=float, default=120.0)
    parser.add_argument("--xvla-domain-id", type=int, default=6)
    parser.add_argument("--xvla-steps", type=int, default=10)
    parser.add_argument("--eval-mode", choices=["seed_plus_replay", "seed_only"], default="seed_plus_replay")
    parser.add_argument("--num-steps-wait-after-replay", type=int, default=0)
    parser.add_argument(
        "--fresh-step-limit-after-replay",
        action="store_true",
        default=True,
        help="Reset take_action_cnt after replay/wait so infer starts from 0 with the original RoboTwin step_lim.",
    )
    parser.add_argument(
        "--count-replay-steps-in-step-limit",
        action="store_false",
        dest="fresh_step_limit_after_replay",
        help="Keep legacy behavior where replay/wait actions consume RoboTwin step_lim.",
    )
    parser.add_argument("--max-infer-steps", type=int, default=-1)
    parser.add_argument("--joint-tol", type=float, default=0.05)
    parser.add_argument("--endpose-pos-tol", type=float, default=0.02)
    parser.add_argument("--endpose-rot-tol", type=float, default=0.15)
    parser.add_argument("--gripper-tol", type=float, default=0.05)
    parser.add_argument("--skip-state-mismatch-check", action="store_true")
    parser.add_argument("--output-dir", type=str, default="eval_logs/xvla_ood_replay")
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--output-dir-is-run-dir", action="store_true")
    parser.add_argument("--result-json-path", type=str, default="")
    parser.add_argument("--append-results-jsonl", type=str, default="")
    parser.add_argument("--summary-json-path", type=str, default="")
    parser.add_argument("--log-path", type=str, default="")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--no-save-video", action="store_false", dest="save_video")
    parser.set_defaults(save_video=False)
    parser.add_argument("--save-dataset-episode", action="store_true")
    parser.add_argument("--dataset-image-format", choices=["png"], default="png")
    parser.add_argument("--video-fps", type=int, default=10)
    parser.add_argument("--video-views", type=str, default=",".join(DEFAULT_VIDEO_VIEWS))
    parser.add_argument("--skip-close-env", action="store_true")
    parser.add_argument("--hard-exit-after-result", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.policy_name != "xvla":
        print(f"Only xvla is supported by this script, got {args.policy_name}", file=sys.stderr)
        return 1

    json_path = Path(args.json_path).expanduser().resolve() if args.json_path else None
    json_root = Path(args.json_root).expanduser().resolve() if args.json_root else None
    sample_file_list = Path(args.sample_file_list).expanduser().resolve() if args.sample_file_list else None
    if json_path is not None:
        json_root = None
        sample_file_list = None
    elif sample_file_list is not None:
        json_root = None

    paths = discover_trace_json_paths(json_path=json_path, json_root=json_root, sample_file_list=sample_file_list, limit=args.limit)
    if not paths:
        print("No JSON files found.", file=sys.stderr)
        return 1

    output_root = Path(args.output_dir).expanduser().resolve()
    if args.output_dir_is_run_dir:
        args.output_dir = output_root
    else:
        args.output_dir = output_root / (args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.video_views = [item.strip() for item in args.video_views.split(",") if item.strip()]

    result_json_path = Path(args.result_json_path).expanduser().resolve() if args.result_json_path else None
    append_results_jsonl_path = Path(args.append_results_jsonl).expanduser().resolve() if args.append_results_jsonl else None
    summary_json_path = Path(args.summary_json_path).expanduser().resolve() if args.summary_json_path else None
    log_path = Path(args.log_path).expanduser().resolve() if args.log_path else args.output_dir / "eval.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_path, encoding="utf-8")],
        force=True,
    )
    logging.info(
        "samples=%d policy=%s model=%s eval_mode=%s wait=%d fresh_step_limit_after_replay=%s max_infer=%d xvla_steps=%d save_video=%s",
        len(paths),
        args.policy_name,
        args.model_name,
        args.eval_mode,
        args.num_steps_wait_after_replay,
        args.fresh_step_limit_after_replay,
        args.max_infer_steps,
        args.xvla_steps,
        args.save_video,
    )

    policy = None
    if not (args.eval_mode == "seed_plus_replay" and args.max_infer_steps == 0):
        if args.xvla_port <= 0:
            raise RuntimeError("--xvla-port must be > 0 when infer is enabled")
        policy = XVLAHttpPolicy(
            args.xvla_host,
            args.xvla_port,
            timeout=args.xvla_timeout,
            domain_id=args.xvla_domain_id,
            steps=args.xvla_steps,
            instruction="",
        )

    results: list[EvalResult] = []
    try:
        for idx, path in enumerate(paths, start=1):
            logging.info("[%d/%d] %s", idx, len(paths), path)
            result = evaluate_one_json(path, args, policy)
            results.append(result)
            if append_results_jsonl_path is not None:
                append_result_jsonl(append_results_jsonl_path, result)
            if result_json_path is not None and len(paths) == 1:
                write_result_json(result_json_path, result)
            if summary_json_path is not None:
                write_summary_json(summary_json_path, results, args=args)
            logging.info(
                "  -> status=%s replay=%d wait=%d infer=%d success=%s",
                result.status,
                result.replay_steps,
                result.wait_steps,
                result.infer_steps,
                result.eval_success_at_end,
            )
    finally:
        if policy is not None:
            policy.close()

    write_results_jsonl(args.output_dir / "results.jsonl", results)
    write_summary_json(args.output_dir / "summary.json", results, args=args)

    ok_statuses = {"infer_ok", "replay_ok"}
    exit_code = 0 if all(result.status in ok_statuses for result in results) else 2
    if args.hard_exit_after_result:
        logging.shutdown()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exit_code)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
