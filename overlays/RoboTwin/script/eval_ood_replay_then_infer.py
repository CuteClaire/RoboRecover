#!/usr/bin/env python3
"""Strict RoboTwin OOD eval: replay annotated actions, then hand off to pi05."""

from __future__ import annotations

import argparse
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
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.6")
        torch_extensions_dir = Path("/tmp/ood_robotwin_torch_extensions")
        torch_extensions_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(torch_extensions_dir))
        cuda_home = Path(os.environ.get("ROBOTWIN_CUDA_HOME", "/usr/local/cuda"))
        if cuda_home.is_dir():
            os.environ.setdefault("ROBOTWIN_CUDA_HOME", str(cuda_home))
            os.environ.setdefault("CUDA_HOME", str(cuda_home))
            os.environ.setdefault("CC", "/usr/bin/gcc-9")
            os.environ.setdefault("CXX", "/usr/bin/g++-9")
            os.environ.setdefault("CUDAHOSTCXX", "/usr/bin/g++-9")
            os.environ.setdefault("NVCC_PREPEND_FLAGS", "--allow-unsupported-compiler")
            cuda_bin = str(cuda_home / "bin")
            path_parts = os.environ.get("PATH", "").split(os.pathsep)
            if cuda_bin not in path_parts:
                os.environ["PATH"] = os.pathsep.join([cuda_bin, *path_parts])
            cuda_lib = str(cuda_home / "targets/x86_64-linux/lib")
            ld_parts = os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep)
            if cuda_lib not in ld_parts:
                os.environ["LD_LIBRARY_PATH"] = os.pathsep.join([cuda_lib, *ld_parts])


_configure_curobo_for_replay()
faulthandler.enable(all_threads=True)

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
from model_rpc import ModelRpcClient
from episode_dataset_recorder import EpisodeDatasetRecorder

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
    # Filled for external Pi05 RPC runs; retained as zero for local policies.
    rpc_infer_calls: int = 0
    rpc_infer_elapsed_s: float = 0.0
    rpc_infer_mean_s: float = 0.0


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


def write_summary_json(path: Path, results: list[EvalResult], *, args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = build_summary(results, args=args)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())


def _load_pi05_model(args: argparse.Namespace):
    if args.policy_name != "pi05":
        raise ValueError(f"unsupported policy_name={args.policy_name!r}; only pi05 is supported")
    if args.model_server_port > 0:
        return ModelRpcClient(args.model_server_host, args.model_server_port, timeout=args.model_server_timeout)
    from policy.pi05.deploy_policy import get_model

    return get_model(
        {
            "train_config_name": args.train_config_name,
            "model_name": args.model_name,
            "checkpoint_id": args.checkpoint_id,
            "pi0_step": args.pi0_step,
        }
    )


def _reset_pi05_model(model) -> None:
    if isinstance(model, ModelRpcClient):
        model.call("reset_model")
        return
    from policy.pi05.deploy_policy import reset_model

    reset_model(model)


def _run_pi05_chunk(
    task_env,
    model,
    observation: dict[str, Any],
    infer_trace: dict[str, Any],
    *,
    pi0_step: int,
    max_actions: int | None = None,
    recorder: EpisodeDatasetRecorder | None = None,
    infer_call_idx: int = 0,
) -> tuple[int, bool]:
    if isinstance(model, ModelRpcClient):
        payload = {
            "instruction": task_env.get_instruction(),
            "images": [
                observation["observation"]["head_camera"]["rgb"],
                observation["observation"]["right_camera"]["rgb"],
                observation["observation"]["left_camera"]["rgb"],
            ],
            "state": observation["joint_action"]["vector"],
            "pi0_step": pi0_step,
        }
        actions = np.asarray(model.call("infer_from_robotwin", payload), dtype=np.float64)
        if max_actions is not None:
            actions = actions[:max_actions]
        executed = 0
        for chunk_offset, action in enumerate(actions):
            infer_trace["actions"].append(np.asarray(action, dtype=np.float64).tolist())
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
    from policy.pi05.deploy_policy import encode_obs

    if model.observation_window is None:
        model.set_language(task_env.get_instruction())

    input_rgb_arr, input_state = encode_obs(observation)
    model.update_observation_window(input_rgb_arr, input_state)
    actions = model.get_action()[: model.pi0_step]
    if max_actions is not None:
        actions = actions[:max_actions]

    executed = 0
    for chunk_offset, action in enumerate(actions):
        infer_trace["actions"].append(np.asarray(action, dtype=np.float64).tolist())
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
        observation = task_env.get_obs()
        input_rgb_arr, input_state = encode_obs(observation)
        model.update_observation_window(input_rgb_arr, input_state)
    return executed, False


def evaluate_one_json(path: Path, args: argparse.Namespace, model) -> EvalResult:
    verify_trace_json(path)
    data = load_trace_json(path)
    task_args = build_task_args(data["task_name"], data["task_config"])
    task_env = None
    recorder: EpisodeDatasetRecorder | None = None
    sample_id = sample_stem_from_path(path)
    try:
        task_env = class_decorator(data["task_name"])
        task_env.setup_demo(now_ep_num=int(data.get("episode_idx", 0)), seed=int(data["seed"]), is_test=True, **task_args)
        task_env.set_instruction(instruction=data["instruction"])
        if args.save_dataset_episode:
            recorder = EpisodeDatasetRecorder(
                args.output_dir,
                {
                    "source_json": str(path.resolve()),
                    "sample_id": sample_id,
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
        replay_step_budget = len(data["actions"]) + int(args.num_steps_wait_after_replay)
        if args.eval_mode == "seed_plus_replay":
            task_env.step_lim = max(task_env.step_lim, task_env.take_action_cnt + replay_step_budget)

        video_paths = None
        if args.save_video and args.output_dir is not None:
            video_paths = start_eval_video_recording(
                task_env,
                args.output_dir,
                sample_id,
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
            logging.warning("[%s] initial_state_exceeds_tolerance; continuing due to --skip-state-mismatch-check", path.name)

        replay_steps = 0
        wait_steps = 0
        infer_steps = 0
        replay_action_type = data.get("action_type", "qpos")

        if args.eval_mode == "seed_plus_replay":
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
                        eval_success_at_end=bool(task_env.eval_success),
                        failure_detail="wait_env_done",
                    )
                    return _apply_init_errors(result, init_errors)
                obs = task_env.get_obs()
        else:
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
            raise RuntimeError("pi05 model is required for infer, but model was not initialized")
        if recorder is not None:
            recorder.set_phase("infer")
        _reset_pi05_model(model)
        infer_trace = {"actions": []}
        max_infer_steps = infer_step_budget
        task_env.take_action_cnt = 0
        task_env.step_lim = original_step_lim
        infer_call_idx = 0
        while task_env.take_action_cnt < task_env.step_lim:
            obs = task_env.get_obs()
            remaining_infer_steps = max_infer_steps - infer_steps
            if remaining_infer_steps <= 0:
                break
            executed, success = _run_pi05_chunk(
                task_env,
                model,
                obs,
                infer_trace,
                pi0_step=args.pi0_step,
                max_actions=remaining_infer_steps,
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
            if infer_steps >= max_infer_steps:
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
            video_paths=locals().get("video_paths"),
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
                task_env.close_env()


def write_results_jsonl(path: Path, results: list[EvalResult]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")


def build_summary(results: list[EvalResult], *, args: argparse.Namespace) -> dict[str, Any]:
    status_counts = Counter(result.status for result in results)
    by_task: dict[str, dict[str, Any]] = {}
    by_error: dict[str, dict[str, Any]] = {}
    failed_files: list[str] = []

    task_groups: dict[str, list[EvalResult]] = defaultdict(list)
    error_groups: dict[str, list[EvalResult]] = defaultdict(list)
    for result in results:
        task_groups[result.task_name].append(result)
        error_groups[result.error_type or "unknown"].append(result)
        if result.status not in {"infer_ok", "replay_ok"}:
            failed_files.append(result.file)

    for task_name, items in sorted(task_groups.items()):
        by_task[task_name] = {
            "total": len(items),
            "success_count": sum(1 for item in items if bool(item.eval_success_at_end)),
            "success_rate": sum(1 for item in items if bool(item.eval_success_at_end)) / len(items),
            "status_counts": dict(Counter(item.status for item in items)),
        }

    for error_type, items in sorted(error_groups.items()):
        by_error[error_type] = {
            "total": len(items),
            "success_count": sum(1 for item in items if bool(item.eval_success_at_end)),
            "success_rate": sum(1 for item in items if bool(item.eval_success_at_end)) / len(items),
            "status_counts": dict(Counter(item.status for item in items)),
        }

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "json_root": str(args.json_root) if args.json_root else "",
        "sample_file_list": str(args.sample_file_list) if args.sample_file_list else "",
        "policy_name": args.policy_name,
        "train_config_name": args.train_config_name,
        "model_name": args.model_name,
        "checkpoint_id": args.checkpoint_id,
        "pi0_step": args.pi0_step,
        "eval_mode": args.eval_mode,
        "num_steps_wait_after_replay": args.num_steps_wait_after_replay,
        "max_infer_steps": args.max_infer_steps,
        "total": len(results),
        "status_counts": dict(status_counts),
        "task_summary": by_task,
        "error_type_summary": by_error,
        "failed_files": failed_files,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay RoboTwin OOD traces, then evaluate pi05 from replayed states.")
    parser.add_argument("--json-path", type=str, default="", help="Single OOD JSON to evaluate")
    parser.add_argument("--json-root", type=str, default=str(DEFAULT_JSON_ROOT), help="Root containing OOD JSON files")
    parser.add_argument("--sample-file-list", type=str, default="", help="Optional text file with one JSON path per line")
    parser.add_argument("--limit", type=int, default=-1, help="Max number of samples to process")
    parser.add_argument("--policy-name", type=str, default="pi05")
    parser.add_argument("--train-config-name", type=str, default="pi05_base_finetune_on_robotwin_clean_randomized_joint_training")
    parser.add_argument("--model-name", type=str, default="pi05_robotwin2")
    parser.add_argument("--checkpoint-id", type=str, default="final")
    parser.add_argument("--pi0-step", type=int, default=32)
    parser.add_argument("--repeat-idx", type=int, default=1)
    parser.add_argument("--model-server-host", type=str, default="127.0.0.1")
    parser.add_argument("--model-server-port", type=int, default=0)
    parser.add_argument("--model-server-timeout", type=float, default=120.0)
    parser.add_argument("--eval-mode", choices=["seed_plus_replay", "seed_only"], default="seed_plus_replay")
    parser.add_argument("--num-steps-wait-after-replay", type=int, default=0)
    parser.add_argument("--max-infer-steps", type=int, default=-1)
    parser.add_argument("--joint-tol", type=float, default=0.05)
    parser.add_argument("--endpose-pos-tol", type=float, default=0.02)
    parser.add_argument("--endpose-rot-tol", type=float, default=0.15)
    parser.add_argument("--gripper-tol", type=float, default=0.05)
    parser.add_argument("--skip-state-mismatch-check", action="store_true")
    parser.add_argument("--output-dir", type=str, default="eval_logs/ood_replay_then_infer")
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--output-dir-is-run-dir", action="store_true")
    parser.add_argument("--result-json-path", type=str, default="", help="Optional single-result JSON output path")
    parser.add_argument("--append-results-jsonl", type=str, default="", help="Append each sample result to this JSONL path")
    parser.add_argument("--summary-json-path", type=str, default="", help="Optional summary JSON output path")
    parser.add_argument("--log-path", type=str, default="", help="Optional log file path")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--no-save-video", action="store_false", dest="save_video")
    parser.set_defaults(save_video=False)
    parser.add_argument("--save-dataset-episode", action="store_true")
    parser.add_argument("--dataset-image-format", choices=["png"], default="png")
    parser.add_argument("--video-fps", type=int, default=10)
    parser.add_argument("--video-views", type=str, default=",".join(DEFAULT_VIDEO_VIEWS))
    parser.add_argument("--skip-close-env", action="store_true", help="Skip task_env.close_env(); useful for single-sample isolated runs")
    parser.add_argument("--hard-exit-after-result", action="store_true", help="Use os._exit after writing outputs")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    json_path = Path(args.json_path).expanduser().resolve() if args.json_path else None
    json_root = Path(args.json_root).expanduser().resolve() if args.json_root else None
    sample_file_list = Path(args.sample_file_list).expanduser().resolve() if args.sample_file_list else None
    if json_path is not None:
        json_root = None
        sample_file_list = None
    elif sample_file_list is not None:
        json_root = None
    paths = discover_trace_json_paths(
        json_path=json_path,
        json_root=json_root,
        sample_file_list=sample_file_list,
        limit=args.limit,
    )
    if not paths:
        print("No JSON files found.", file=sys.stderr)
        return 1

    output_root = Path(args.output_dir).expanduser().resolve()
    if args.output_dir_is_run_dir:
        args.output_dir = output_root
    else:
        run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = output_root / run_name
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.video_views = [item.strip() for item in args.video_views.split(",") if item.strip()]
    result_json_path = Path(args.result_json_path).expanduser().resolve() if args.result_json_path else None
    append_results_jsonl_path = Path(args.append_results_jsonl).expanduser().resolve() if args.append_results_jsonl else None
    summary_json_path = Path(args.summary_json_path).expanduser().resolve() if args.summary_json_path else None
    log_path = Path(args.log_path).expanduser().resolve() if args.log_path else args.output_dir / "eval.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
        force=True,
    )
    logging.info(
        "samples=%d policy=%s model=%s eval_mode=%s wait=%d max_infer=%d save_video=%s",
        len(paths),
        args.policy_name,
        args.model_name,
        args.eval_mode,
        args.num_steps_wait_after_replay,
        args.max_infer_steps,
        args.save_video,
    )

    model = None
    if not (args.eval_mode == "seed_plus_replay" and args.max_infer_steps == 0):
        model = _load_pi05_model(args)
    results: list[EvalResult] = []
    for idx, path in enumerate(paths, start=1):
        logging.info("[%d/%d] %s", idx, len(paths), path)
        rpc_before = model.rpc_stats() if isinstance(model, ModelRpcClient) else None
        result = evaluate_one_json(path, args, model)
        if isinstance(model, ModelRpcClient) and rpc_before is not None:
            rpc_after = model.rpc_stats()
            calls = int(rpc_after["infer_calls"] - rpc_before["infer_calls"])
            elapsed = float(rpc_after["infer_elapsed_s"] - rpc_before["infer_elapsed_s"])
            result.rpc_infer_calls = calls
            result.rpc_infer_elapsed_s = elapsed
            result.rpc_infer_mean_s = elapsed / calls if calls else 0.0
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

    default_results_jsonl = args.output_dir / "results.jsonl"
    default_summary_json = args.output_dir / "summary.json"
    write_results_jsonl(default_results_jsonl, results)
    write_summary_json(default_summary_json, results, args=args)
    logging.info("Wrote %s", default_results_jsonl)
    logging.info("Wrote %s", default_summary_json)

    ok_statuses = {"infer_ok", "replay_ok"}
    exit_code = 0 if all(result.status in ok_statuses for result in results) else 2
    if isinstance(model, ModelRpcClient):
        model.close()
    if args.hard_exit_after_result:
        logging.shutdown()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exit_code)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
