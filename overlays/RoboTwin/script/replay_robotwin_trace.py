#!/usr/bin/env python3
"""Replay RoboTwin failure-trace JSON files (seed + actions) for strict reproduction."""

from __future__ import annotations

import os

# Replay stored qpos actions via mplib TOPP only; skip Curobo JIT/planner init.
os.environ.setdefault("ROBOTWIN_SKIP_CUROBO_PLANNER", "1")

import argparse
import json
import logging
import sys
import traceback
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

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


@dataclass
class ReplayResult:
    file: str
    task_name: str
    task_config: str
    seed: int
    episode_idx: int
    instruction: str
    status: str
    reason: str
    replay_steps: int
    actions_len: int
    annotated_step: int | None
    init_joint_err: float | None = None
    init_left_endpose_pos_err: float | None = None
    init_left_endpose_rot_err: float | None = None
    init_right_endpose_pos_err: float | None = None
    init_right_endpose_rot_err: float | None = None
    init_left_gripper_err: float | None = None
    init_right_gripper_err: float | None = None
    eval_success_at_end: bool | None = None
    video_paths: dict[str, str] | None = None
    failure_detail: str | None = None
    traceback: str | None = None

    @classmethod
    def from_init_errors(cls, path: Path, data: dict, errors: InitialStateErrors, *, reason: str) -> ReplayResult:
        err = errors.to_dict()
        return cls(
            file=str(path),
            task_name=data["task_name"],
            task_config=data["task_config"],
            seed=int(data["seed"]),
            episode_idx=int(data.get("episode_idx", 0)),
            instruction=data["instruction"],
            status="state_mismatch",
            reason=reason,
            replay_steps=0,
            actions_len=len(data["actions"]),
            annotated_step=data.get("annotated_step"),
            init_joint_err=err["joint_err"],
            init_left_endpose_pos_err=err["left_endpose_pos_err"],
            init_left_endpose_rot_err=err["left_endpose_rot_err"],
            init_right_endpose_pos_err=err["right_endpose_pos_err"],
            init_right_endpose_rot_err=err["right_endpose_rot_err"],
            init_left_gripper_err=err["left_gripper_err"],
            init_right_gripper_err=err["right_gripper_err"],
            failure_detail="init_state_mismatch",
        )

    @classmethod
    def verify_ok(cls, path: Path, data: dict) -> ReplayResult:
        return cls(
            file=str(path),
            task_name=data["task_name"],
            task_config=data["task_config"],
            seed=int(data["seed"]),
            episode_idx=int(data.get("episode_idx", 0)),
            instruction=data["instruction"],
            status="verify_ok",
            reason="schema_and_task_files_ok",
            replay_steps=0,
            actions_len=len(data["actions"]),
            annotated_step=data.get("annotated_step"),
        )

    @classmethod
    def replay_crash(cls, path: Path, data: dict | None, exc: BaseException) -> ReplayResult:
        if data is None:
            return cls(
                file=str(path),
                task_name="",
                task_config="",
                seed=-1,
                episode_idx=-1,
                instruction="",
                status="replay_crash",
                reason=str(exc),
                replay_steps=0,
                actions_len=0,
                annotated_step=None,
                failure_detail=type(exc).__name__,
                traceback=traceback.format_exc(),
            )
        return cls(
            file=str(path),
            task_name=data.get("task_name", ""),
            task_config=data.get("task_config", ""),
            seed=int(data.get("seed", -1)),
            episode_idx=int(data.get("episode_idx", -1)),
            instruction=data.get("instruction", ""),
            status="replay_crash",
            reason=str(exc),
            replay_steps=0,
            actions_len=len(data.get("actions", [])),
            annotated_step=data.get("annotated_step"),
            failure_detail=type(exc).__name__,
            traceback=traceback.format_exc(),
        )


def _apply_init_errors(result: ReplayResult, errors: InitialStateErrors) -> ReplayResult:
    err = errors.to_dict()
    result.init_joint_err = err["joint_err"]
    result.init_left_endpose_pos_err = err["left_endpose_pos_err"]
    result.init_left_endpose_rot_err = err["left_endpose_rot_err"]
    result.init_right_endpose_pos_err = err["right_endpose_pos_err"]
    result.init_right_endpose_rot_err = err["right_endpose_rot_err"]
    result.init_left_gripper_err = err["left_gripper_err"]
    result.init_right_gripper_err = err["right_gripper_err"]
    return result


def replay_trace_json(
    path: Path,
    *,
    joint_tol: float,
    endpose_pos_tol: float,
    endpose_rot_tol: float,
    gripper_tol: float,
    skip_state_mismatch: bool,
    save_video: bool,
    output_dir: Path | None,
    video_fps: int,
    video_views: list[str],
) -> ReplayResult:
    data = load_trace_json(path)
    task_name = data["task_name"]
    task_config = data["task_config"]
    seed = int(data["seed"])
    episode_idx = int(data.get("episode_idx", 0))
    instruction = data["instruction"]
    actions = data["actions"]
    action_type = data.get("action_type", "qpos")

    video_save_dir = output_dir if save_video and output_dir is not None else None
    args = build_task_args(task_name, task_config, eval_video_save_dir=video_save_dir)
    task_env = class_decorator(task_name)
    task_env.setup_demo(now_ep_num=episode_idx, seed=seed, is_test=True, **args)
    task_env.set_instruction(instruction=instruction)

    obs = task_env.get_obs()
    init_errors = initial_state_error(obs, data["initial_state"])
    if init_errors.exceeds_tolerance(
        joint_tol=joint_tol,
        endpose_pos_tol=endpose_pos_tol,
        endpose_rot_tol=endpose_rot_tol,
        gripper_tol=gripper_tol,
    ):
        reason = (
            f"init state exceeds tol: joint={init_errors.joint_err:.6f}(tol {joint_tol}) "
            f"left_pos={init_errors.left_endpose_pos_err}(tol {endpose_pos_tol}) "
            f"left_rot={init_errors.left_endpose_rot_err}(tol {endpose_rot_tol}) "
            f"right_pos={init_errors.right_endpose_pos_err}(tol {endpose_pos_tol}) "
            f"right_rot={init_errors.right_endpose_rot_err}(tol {endpose_rot_tol}) "
            f"left_grip={init_errors.left_gripper_err}(tol {gripper_tol}) "
            f"right_grip={init_errors.right_gripper_err}(tol {gripper_tol})"
        )
        if skip_state_mismatch:
            logging.warning("[%s] %s (continuing due to --skip-state-mismatch-check)", path.name, reason)
        else:
            task_env.close_env()
            return ReplayResult.from_init_errors(path, data, init_errors, reason=reason)

    video_paths = None
    if save_video and output_dir is not None:
        video_paths = start_eval_video_recording(
            task_env,
            output_dir,
            sample_stem_from_path(path),
            args,
            video_fps=video_fps,
            video_views=video_views,
        )

    replay_steps = 0
    for action in actions:
        task_env.take_action(np.asarray(action, dtype=np.float64), action_type=action_type)
        replay_steps += 1

    eval_success_at_end = bool(task_env.eval_success)
    if save_video:
        task_env._del_eval_video_ffmpeg()
    task_env.close_env()

    result = ReplayResult(
        file=str(path),
        task_name=task_name,
        task_config=task_config,
        seed=seed,
        episode_idx=episode_idx,
        instruction=instruction,
        status="replay_ok",
        reason="replay_completed",
        replay_steps=replay_steps,
        actions_len=len(actions),
        annotated_step=data.get("annotated_step"),
        eval_success_at_end=eval_success_at_end,
        video_paths=video_paths,
    )
    return _apply_init_errors(result, init_errors)


def verify_trace(path: Path) -> ReplayResult:
    data = verify_trace_json(path)
    return ReplayResult.verify_ok(path, data)


def write_results_jsonl(path: Path, results: list[ReplayResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")


def build_summary(results: list[ReplayResult]) -> dict[str, Any]:
    status_counts = Counter(r.status for r in results)
    joint_errs = [r.init_joint_err for r in results if r.init_joint_err is not None]
    failed_files = [r.file for r in results if r.status != "replay_ok" and r.status != "verify_ok"]
    return {
        "total": len(results),
        "status_counts": dict(status_counts),
        "avg_init_joint_err": float(np.mean(joint_errs)) if joint_errs else None,
        "failed_files": failed_files,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay RoboTwin failure trace JSON files.")
    parser.add_argument("--json-path", type=str, default="", help="Single JSON file to replay")
    parser.add_argument(
        "--json-root",
        type=str,
        default="",
        help="Directory containing *.json failure traces",
    )
    parser.add_argument("--sample-file-list", type=str, default="", help="Text file with one JSON path per line")
    parser.add_argument("--output-dir", type=str, default="", help="Directory for results.jsonl / summary.json / videos")
    parser.add_argument("--run-name", type=str, default="", help="Subdirectory under output-dir (default: timestamp)")
    parser.add_argument("--verify-only", action="store_true", help="Validate JSON schema and task files only")
    parser.add_argument("--limit", type=int, default=-1, help="Max number of JSON files to process")
    parser.add_argument("--joint-tol", type=float, default=0.05)
    parser.add_argument("--endpose-pos-tol", type=float, default=0.02)
    parser.add_argument("--endpose-rot-tol", type=float, default=0.15)
    parser.add_argument("--gripper-tol", type=float, default=0.05)
    parser.add_argument("--skip-state-mismatch-check", action="store_true")
    parser.add_argument("--save-video", action="store_true", help="Record replay videos (default: off)")
    parser.add_argument("--video-fps", type=int, default=10)
    parser.add_argument(
        "--video-views",
        type=str,
        default=",".join(DEFAULT_VIDEO_VIEWS),
        help="Comma-separated camera names when --save-video is set",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    json_path = Path(args.json_path).expanduser().resolve() if args.json_path else None
    json_root = Path(args.json_root).expanduser().resolve() if args.json_root else None
    sample_file_list = Path(args.sample_file_list).expanduser().resolve() if args.sample_file_list else None

    paths = discover_trace_json_paths(
        json_path=json_path,
        json_root=json_root,
        sample_file_list=sample_file_list,
        limit=args.limit,
    )
    if not paths:
        print("No JSON files found.", file=sys.stderr)
        return 1

    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = None
    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve() / run_name
        output_dir.mkdir(parents=True, exist_ok=True)
    elif args.save_video and not args.verify_only:
        output_dir = Path("replay_logs") / run_name
        output_dir.mkdir(parents=True, exist_ok=True)

    log_handlers = [logging.StreamHandler(sys.stdout)]
    if output_dir is not None:
        log_handlers.append(logging.FileHandler(output_dir / "replay.log", encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=log_handlers,
        force=True,
    )

    video_views = [v.strip() for v in args.video_views.split(",") if v.strip()]
    results: list[ReplayResult] = []

    for path in paths:
        logging.info("Processing %s", path)
        try:
            if args.verify_only:
                result = verify_trace(path)
            else:
                result = replay_trace_json(
                    path,
                    joint_tol=args.joint_tol,
                    endpose_pos_tol=args.endpose_pos_tol,
                    endpose_rot_tol=args.endpose_rot_tol,
                    gripper_tol=args.gripper_tol,
                    skip_state_mismatch=args.skip_state_mismatch_check,
                    save_video=args.save_video,
                    output_dir=output_dir,
                    video_fps=args.video_fps,
                    video_views=video_views,
                )
            results.append(result)
            logging.info("  -> %s (%s)", result.status, result.reason)
        except Exception as exc:
            data = None
            try:
                data = load_trace_json(path)
            except Exception:
                pass
            result = ReplayResult.replay_crash(path, data, exc)
            results.append(result)
            logging.error("  -> replay_crash: %s", exc)

    if output_dir is not None:
        write_results_jsonl(output_dir / "results.jsonl", results)
        summary = build_summary(results)
        write_summary(output_dir / "summary.json", summary)
        logging.info("Wrote %s", output_dir / "results.jsonl")
        logging.info("Wrote %s", output_dir / "summary.json")
    else:
        print(json.dumps([asdict(r) for r in results], indent=2, ensure_ascii=False))

    ok_statuses = {"replay_ok"} if not args.verify_only else {"verify_ok"}
    if all(r.status in ok_statuses for r in results):
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
