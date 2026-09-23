#!/usr/bin/env python3
"""Replay RoboTwin OOD traces, then evaluate LingBot-VLA via its official websocket server."""

from __future__ import annotations

import argparse
import csv
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

LINGBOT_ROOT = Path(__file__).resolve().parents[2]
ROBOTWIN_ROOT = Path(os.environ.get("ROBOTWIN_ROOT", "/path/to/workspace/RoboTwin"))
ROBOTWIN_SCRIPT = ROBOTWIN_ROOT / "script"
for path in (str(LINGBOT_ROOT), str(ROBOTWIN_SCRIPT), str(ROBOTWIN_ROOT), str(ROBOTWIN_ROOT / "policy")):
    if path not in sys.path:
        sys.path.insert(0, path)


def _configure_curobo_for_replay() -> None:
    """Enable cuRobo only for traces whose replay actions are end-effector actions."""
    paths: list[Path] = []
    argv = sys.argv[1:]
    if "--eval-mode" in argv:
        mode_idx = argv.index("--eval-mode")
        if mode_idx + 1 < len(argv) and argv[mode_idx + 1] == "seed_only":
            os.environ["ROBOTWIN_SKIP_CUROBO_PLANNER"] = "1"
            return
    idx = 0
    while idx < len(argv):
        arg = argv[idx]
        if arg == "--json-path" and idx + 1 < len(argv):
            paths.append(Path(argv[idx + 1]).expanduser())
            idx += 2
            continue
        if arg == "--sample-file-list" and idx + 1 < len(argv):
            sample_list = Path(argv[idx + 1]).expanduser()
            try:
                for line in sample_list.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line:
                        paths.append(Path(line).expanduser())
            except OSError:
                pass
            idx += 2
            continue
        if arg == "--sample-csv" and idx + 1 < len(argv):
            sample_csv = Path(argv[idx + 1]).expanduser()
            try:
                with sample_csv.open("r", encoding="utf-8-sig", newline="") as f:
                    for row in csv.DictReader(f):
                        raw_path = Path(row["json_path"]).expanduser()
                        if not raw_path.is_file():
                            try:
                                relative = raw_path.relative_to("/path/to/workspace")
                            except ValueError:
                                relative = None
                            if relative is not None:
                                mapped = Path("/path/to/workspace") / relative
                                if mapped.is_file():
                                    raw_path = mapped
                        paths.append(raw_path)
            except (OSError, KeyError):
                pass
            idx += 2
            continue
        idx += 1

    needs_curobo = False
    for path in paths:
        try:
            with path.open("r", encoding="utf-8") as f:
                needs_curobo = json.load(f).get("action_type", "qpos") == "ee"
        except Exception:
            needs_curobo = False
        if needs_curobo:
            break
    os.environ["ROBOTWIN_SKIP_CUROBO_PLANNER"] = "0" if needs_curobo else "1"
    if needs_curobo:
        # End-effector traces need CuRobo.  Match the known-good XVLA toolchain
        # instead of accidentally choosing the conda-bundled nvcc/GCC pair,
        # which cannot build CuRobo's CUDA extensions on the 3090 hosts.
        cuda_home = Path(os.environ.get("ROBOTWIN_CUDA_HOME", "/usr/local/cuda"))
        if cuda_home.is_dir():
            os.environ.setdefault("ROBOTWIN_CUDA_HOME", str(cuda_home))
            os.environ.setdefault("CUDA_HOME", str(cuda_home))
            os.environ.setdefault("CC", "/usr/bin/gcc-9")
            os.environ.setdefault("CXX", "/usr/bin/g++-9")
            os.environ.setdefault("CUDAHOSTCXX", "/usr/bin/g++-9")
            os.environ.setdefault("NVCC_PREPEND_FLAGS", "--allow-unsupported-compiler")
            cuda_bin = str(cuda_home / "bin")
            if cuda_bin not in os.environ.get("PATH", "").split(":"):
                os.environ["PATH"] = cuda_bin + ":" + os.environ.get("PATH", "")
            cuda_lib = str(cuda_home / "targets/x86_64-linux/lib")
            if cuda_lib not in os.environ.get("LD_LIBRARY_PATH", "").split(":"):
                os.environ["LD_LIBRARY_PATH"] = cuda_lib + ":" + os.environ.get("LD_LIBRARY_PATH", "")
        # Leave TORCH_CUDA_ARCH_LIST unset unless the launcher/user supplied it.
        # PyTorch then targets the visible GPU (8.6 on 3090, 8.9 on 4090),
        # instead of reusing a binary compiled for the wrong host architecture.
        torch_extensions_dir = Path("/tmp/ood_robotwin_torch_extensions")
        torch_extensions_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(torch_extensions_dir))


_configure_curobo_for_replay()
faulthandler.enable(all_threads=True)

from deploy.websocket_client_policy import WebsocketClientPolicy
from episode_dataset_recorder import EpisodeDatasetRecorder
from robotwin_trace_utils import (
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


class LingBotClient:
    def __init__(self, host: str, port: int) -> None:
        self.client = WebsocketClientPolicy(host=host, port=port)

    def reset_model(self) -> None:
        self.client.reset("robotwin")

    def infer_from_robotwin(self, task_env, observation: dict[str, Any], *, max_actions: int | None) -> np.ndarray:
        obs_images = observation.get("observation", {})
        payload = {
            "task": task_env.get_instruction(),
            "observation.images.cam_high": _rgb_uint8(obs_images["head_camera"]["rgb"], "head_camera"),
            "observation.images.cam_left_wrist": _rgb_uint8(obs_images["left_camera"]["rgb"], "left_camera"),
            "observation.images.cam_right_wrist": _rgb_uint8(obs_images["right_camera"]["rgb"], "right_camera"),
            "observation.state": np.asarray(observation["joint_action"]["vector"], dtype=np.float32),
        }
        response = self.client.infer(payload)
        actions = _lingbot_response_to_joint_actions(response)
        if max_actions is not None:
            actions = actions[:max_actions]
        return actions


def _rgb_uint8(image: Any, name: str) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"{name} rgb image must be HWC with 3 channels, got {arr.shape}")
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _as_action_chunk(value: Any, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2:
        raise ValueError(f"{name} must be [T,D], got {arr.shape}")
    return arr


def _lingbot_response_to_joint_actions(response: dict[str, Any]) -> np.ndarray:
    if "action" in response and response["action"] is not None:
        actions = _as_action_chunk(response["action"], "action")
        if actions.shape[1] != 14:
            raise ValueError(f"expected action [T,14], got {actions.shape}")
        return actions

    if "action.arm.position" not in response or "action.effector.position" not in response:
        raise ValueError(f"LingBot response missing action keys: {sorted(response.keys())}")

    arm = _as_action_chunk(response["action.arm.position"], "action.arm.position")
    effector = _as_action_chunk(response["action.effector.position"], "action.effector.position")
    if arm.shape[0] != effector.shape[0] or arm.shape[1] != 12 or effector.shape[1] != 2:
        raise ValueError(f"expected arm [T,12] and effector [T,2], got {arm.shape} and {effector.shape}")
    return np.concatenate([arm[:, :6], effector[:, :1], arm[:, 6:12], effector[:, 1:2]], axis=1)


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
) -> EvalResult:
    return EvalResult(
        file=str(path),
        model_name=model_name,
        repeat_idx=repeat_idx,
        sample_id=sample_stem_from_path(path),
        task_name=data.get("task_name", ""),
        task_config=data.get("task_config", ""),
        seed=int(data.get("seed", 0)),
        episode_idx=int(data.get("episode_idx", 0)),
        instruction=data.get("instruction", ""),
        error_type=data.get("error_type"),
        s_stage=data.get("s_stage"),
        d_stage=data.get("d_stage"),
        status=status,
        reason=reason,
        replay_steps=replay_steps,
        wait_steps=wait_steps,
        infer_steps=infer_steps,
        actions_len=len(data.get("actions") or []),
        annotated_step=data.get("annotated_step"),
        replay_done=replay_steps > 0 or status in {"replay_already_done", "infer_ok", "infer_failed", "max_infer_steps_reached"},
        wait_done=wait_steps > 0,
        infer_done=infer_steps > 0,
        eval_success_at_end=eval_success_at_end,
        video_paths=video_paths,
        failure_detail=failure_detail,
        traceback=tb,
    )


def _closed_gripper_action(obs: dict[str, Any]) -> np.ndarray:
    action = np.zeros_like(np.asarray(obs["joint_action"]["vector"], dtype=np.float64))
    if action.shape[0] >= 14:
        action[6] = -1.0
        action[13] = -1.0
    elif action.shape[0] >= 2:
        action[-2:] = -1.0
    return action


def _close_video(task_env) -> None:
    try:
        task_env._del_eval_video_ffmpeg()
    except Exception:
        pass


def _run_lingbot_chunk(
    task_env,
    model: LingBotClient,
    observation: dict[str, Any],
    *,
    max_actions: int | None,
    recorder: EpisodeDatasetRecorder | None = None,
    infer_call_idx: int = 0,
) -> tuple[int, bool]:
    actions = model.infer_from_robotwin(task_env, observation, max_actions=max_actions)
    if actions.ndim != 2 or actions.shape[1] != 14:
        raise ValueError(f"expected LingBot action chunk [T,14], got {actions.shape}")

    executed = 0
    for chunk_offset, action in enumerate(actions):
        if recorder is not None:
            recorder.set_action_context(
                phase="infer",
                policy_action_raw=action,
                infer_call_idx=infer_call_idx,
                chunk_offset=chunk_offset,
            )
        task_env.take_action(action)
        executed += 1
        if task_env.eval_success:
            return executed, True
        if task_env.take_action_cnt >= task_env.step_lim:
            return executed, False
    return executed, False


def evaluate_one_json(path: Path, args: argparse.Namespace, model: LingBotClient | None) -> EvalResult:
    verify_trace_json(path)
    data = load_trace_json(path)
    task_args = build_task_args(data["task_name"], data["task_config"])
    task_env = None
    recorder: EpisodeDatasetRecorder | None = None
    replay_steps = 0
    wait_steps = 0
    infer_steps = 0
    video_paths = None
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
        original_step_lim = int(task_env.step_lim)
        infer_step_budget = min(args.max_infer_steps, original_step_lim) if args.max_infer_steps > 0 else original_step_lim
        if args.eval_mode == "seed_plus_replay":
            task_env.step_lim = max(task_env.step_lim, task_env.take_action_cnt + len(data["actions"]) + args.num_steps_wait_after_replay)

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
        ) and not args.skip_state_mismatch_check:
            result = _result_from_data(
                path,
                data,
                model_name=args.model_name,
                repeat_idx=args.repeat_idx,
                status="state_mismatch",
                reason="initial_state_exceeds_tolerance",
                video_paths=video_paths,
                failure_detail="init_state_mismatch",
            )
            return _apply_init_errors(result, init_errors)

        if args.eval_mode == "seed_plus_replay":
            replay_action_type = data.get("action_type", "qpos")
            if recorder is not None:
                recorder.set_phase("replay")
            for action in data["actions"]:
                task_env.take_action(np.asarray(action, dtype=np.float64), action_type=replay_action_type)
                replay_steps += 1
                if task_env.eval_success:
                    result = _result_from_data(
                        path,
                        data,
                        model_name=args.model_name,
                        repeat_idx=args.repeat_idx,
                        status="replay_already_done",
                        reason="env_done_during_replay",
                        replay_steps=replay_steps,
                        video_paths=video_paths,
                        eval_success_at_end=True,
                        failure_detail="replay_env_done",
                    )
                    return _apply_init_errors(result, init_errors)

            obs = task_env.get_obs()
            if recorder is not None:
                recorder.set_phase("replay_wait")
            for _ in range(args.num_steps_wait_after_replay):
                task_env.take_action(_closed_gripper_action(obs))
                wait_steps += 1
                if task_env.eval_success:
                    result = _result_from_data(
                        path,
                        data,
                        model_name=args.model_name,
                        repeat_idx=args.repeat_idx,
                        status="replay_already_done",
                        reason="env_done_during_wait_after_replay",
                        replay_steps=replay_steps,
                        wait_steps=wait_steps,
                        video_paths=video_paths,
                        eval_success_at_end=True,
                        failure_detail="wait_env_done",
                    )
                    return _apply_init_errors(result, init_errors)
                obs = task_env.get_obs()

        if args.max_infer_steps == 0:
            result = _result_from_data(
                path,
                data,
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

        if model is None:
            raise RuntimeError("LingBot model server is required for infer")
        if recorder is not None:
            recorder.set_phase("infer")
        model.reset_model()
        task_env.take_action_cnt = 0
        task_env.step_lim = original_step_lim
        infer_call_idx = 0

        while task_env.take_action_cnt < task_env.step_lim:
            remaining = infer_step_budget - infer_steps
            if remaining <= 0:
                break
            obs = task_env.get_obs()
            executed, success = _run_lingbot_chunk(
                task_env,
                model,
                obs,
                max_actions=remaining,
                recorder=recorder,
                infer_call_idx=infer_call_idx,
            )
            infer_call_idx += 1
            infer_steps += executed
            if success or task_env.eval_success:
                result = _result_from_data(
                    path,
                    data,
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
            if infer_steps >= infer_step_budget:
                result = _result_from_data(
                    path,
                    data,
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

        result = _result_from_data(
            path,
            data,
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
        result = _result_from_data(
            path,
            data,
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
                _close_video(task_env)
            if not args.skip_close_env:
                task_env.close_env()


def append_result_jsonl(path: Path, result: EvalResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")


def write_result_json(path: Path, result: EvalResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(result), f, ensure_ascii=False)
        f.write("\n")


def build_summary(results: list[EvalResult], *, args: argparse.Namespace) -> dict[str, Any]:
    task_groups: dict[str, list[EvalResult]] = defaultdict(list)
    error_groups: dict[str, list[EvalResult]] = defaultdict(list)
    for result in results:
        task_groups[result.task_name].append(result)
        error_groups[result.error_type or "unknown"].append(result)
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "json_root": str(args.json_root) if args.json_root else "",
        "sample_file_list": str(args.sample_file_list) if args.sample_file_list else "",
        "policy_name": args.policy_name,
        "model_name": args.model_name,
        "eval_mode": args.eval_mode,
        "num_steps_wait_after_replay": args.num_steps_wait_after_replay,
        "max_infer_steps": args.max_infer_steps,
        "total": len(results),
        "status_counts": dict(Counter(result.status for result in results)),
        "task_summary": {
            task: {
                "total": len(items),
                "success_count": sum(1 for item in items if bool(item.eval_success_at_end)),
                "success_rate": sum(1 for item in items if bool(item.eval_success_at_end)) / len(items),
                "status_counts": dict(Counter(item.status for item in items)),
            }
            for task, items in sorted(task_groups.items())
        },
        "error_type_summary": {
            err: {
                "total": len(items),
                "success_count": sum(1 for item in items if bool(item.eval_success_at_end)),
                "success_rate": sum(1 for item in items if bool(item.eval_success_at_end)) / len(items),
                "status_counts": dict(Counter(item.status for item in items)),
            }
            for err, items in sorted(error_groups.items())
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RoboTwin OOD replay then LingBot-VLA infer")
    parser.add_argument("--json-path", default="")
    parser.add_argument("--json-root", default=str(DEFAULT_JSON_ROOT))
    parser.add_argument("--sample-file-list", default="")
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--policy-name", default="lingbot-vla")
    parser.add_argument("--model-name", default="lingbot-vla-4b-posttrain-robotwin")
    parser.add_argument("--repeat-idx", type=int, default=1)
    parser.add_argument("--model-server-host", default="127.0.0.1")
    parser.add_argument("--model-server-port", type=int, default=0)
    parser.add_argument("--eval-mode", choices=["seed_plus_replay", "seed_only"], default="seed_plus_replay")
    parser.add_argument("--num-steps-wait-after-replay", type=int, default=0)
    parser.add_argument("--max-infer-steps", type=int, default=-1)
    parser.add_argument("--joint-tol", type=float, default=0.05)
    parser.add_argument("--endpose-pos-tol", type=float, default=0.02)
    parser.add_argument("--endpose-rot-tol", type=float, default=0.15)
    parser.add_argument("--gripper-tol", type=float, default=0.05)
    parser.add_argument("--skip-state-mismatch-check", action="store_true")
    parser.add_argument("--output-dir", default="eval_logs/lingbot_ood_replay")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--output-dir-is-run-dir", action="store_true")
    parser.add_argument("--result-json-path", default="")
    parser.add_argument("--append-results-jsonl", default="")
    parser.add_argument("--summary-json-path", default="")
    parser.add_argument("--log-path", default="")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--no-save-video", action="store_false", dest="save_video")
    parser.set_defaults(save_video=False)
    parser.add_argument("--save-dataset-episode", action="store_true")
    parser.add_argument("--dataset-image-format", choices=["png"], default="png")
    parser.add_argument("--video-fps", type=int, default=10)
    parser.add_argument("--video-views", default=",".join(DEFAULT_VIDEO_VIEWS))
    parser.add_argument("--skip-close-env", action="store_true")
    parser.add_argument("--hard-exit-after-result", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    json_path = Path(args.json_path).expanduser().resolve() if args.json_path else None
    json_root = Path(args.json_root).expanduser().resolve() if args.json_root else None
    sample_file_list = Path(args.sample_file_list).expanduser().resolve() if args.sample_file_list else None
    if json_path:
        json_root = None
        sample_file_list = None
    elif sample_file_list:
        json_root = None
    paths = discover_trace_json_paths(json_path=json_path, json_root=json_root, sample_file_list=sample_file_list, limit=args.limit)
    if not paths:
        print("No JSON files found.", file=sys.stderr)
        return 1

    output_root = Path(args.output_dir).expanduser().resolve()
    args.output_dir = output_root if args.output_dir_is_run_dir else output_root / (args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.video_views = [item.strip() for item in args.video_views.split(",") if item.strip()]
    log_path = Path(args.log_path).expanduser().resolve() if args.log_path else args.output_dir / "eval.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_path, encoding="utf-8")],
        force=True,
    )

    model = None
    if not (args.eval_mode == "seed_plus_replay" and args.max_infer_steps == 0):
        if args.model_server_port <= 0:
            raise ValueError("--model-server-port must be > 0 when max_infer_steps != 0")
        model = LingBotClient(args.model_server_host, args.model_server_port)

    results: list[EvalResult] = []
    for idx, path in enumerate(paths, start=1):
        logging.info("[%d/%d] %s", idx, len(paths), path)
        result = evaluate_one_json(path, args, model)
        results.append(result)
        if args.result_json_path:
            write_result_json(Path(args.result_json_path).expanduser().resolve(), result)
        if args.append_results_jsonl:
            append_result_jsonl(Path(args.append_results_jsonl).expanduser().resolve(), result)
        logging.info(
            "status=%s success=%s replay=%d infer=%d reason=%s",
            result.status,
            result.eval_success_at_end,
            result.replay_steps,
            result.infer_steps,
            result.reason,
        )

    summary_path = Path(args.summary_json_path).expanduser().resolve() if args.summary_json_path else args.output_dir / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(build_summary(results, args=args), f, indent=2, ensure_ascii=False)
        f.write("\n")

    if args.hard_exit_after_result:
        os._exit(0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
