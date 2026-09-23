#!/usr/bin/env python3
"""Replay a single RoboTwin OOD JSON and verify replay videos against reference eval videos."""

from __future__ import annotations

import os

# Replay stored qpos actions via mplib TOPP only; skip Curobo JIT/planner init.
os.environ.setdefault("ROBOTWIN_SKIP_CUROBO_PLANNER", "1")

import argparse
import json
import logging
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from replay_robotwin_trace import ReplayResult, replay_trace_json
from robotwin_trace_utils import (
    DEFAULT_VIDEO_VIEWS,
    load_ood_reference_videos,
    load_trace_json,
    sample_stem_from_path,
    setup_robotwin_paths,
)
from robotwin_video_compare import compare_camera_sets_exact

setup_robotwin_paths()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay and verify a RoboTwin OOD trace JSON.")
    parser.add_argument("--json-path", type=str, required=True, help="Path to OOD JSON file")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Output directory (default: replay_logs/ood_verify/<json_stem>)",
    )
    parser.add_argument("--save-video", action="store_true", default=True, help="Record replay videos")
    parser.add_argument("--no-save-video", action="store_false", dest="save_video")
    parser.add_argument("--compare-video", action="store_true", default=True, help="Compare replay vs reference")
    parser.add_argument("--no-compare-video", action="store_false", dest="compare_video")
    parser.add_argument("--video-fps", type=int, default=10)
    parser.add_argument(
        "--video-views",
        type=str,
        default=",".join(DEFAULT_VIDEO_VIEWS),
        help="Comma-separated camera names",
    )
    parser.add_argument("--joint-tol", type=float, default=0.05)
    parser.add_argument("--endpose-pos-tol", type=float, default=0.02)
    parser.add_argument("--endpose-rot-tol", type=float, default=0.15)
    parser.add_argument("--gripper-tol", type=float, default=0.05)
    parser.add_argument("--skip-state-mismatch-check", action="store_true")
    return parser.parse_args()


def build_report(
    *,
    json_path: Path,
    data: dict[str, Any],
    replay_result: ReplayResult,
    video_compare: dict[str, Any] | None,
    output_dir: Path,
) -> dict[str, Any]:
    expected_frames = len(data["actions"])
    step_ok = replay_result.replay_steps == expected_frames
    status = replay_result.status
    if status == "replay_ok" and video_compare is not None:
        if not step_ok:
            status = "replay_step_mismatch"
        elif not video_compare.get("all_cameras_match", False):
            status = "video_mismatch"
        else:
            status = "verify_ok"
    elif status == "replay_ok" and not step_ok:
        status = "replay_step_mismatch"

    return {
        "json_path": str(json_path),
        "output_dir": str(output_dir),
        "task_name": data.get("task_name"),
        "task_config": data.get("task_config"),
        "seed": data.get("seed"),
        "episode_idx": data.get("episode_idx"),
        "instruction": data.get("instruction"),
        "annotated_step": data.get("annotated_step"),
        "actions_len": len(data.get("actions") or []),
        "expected_frames": expected_frames,
        "error_type": data.get("error_type"),
        "s_stage": data.get("s_stage"),
        "d_stage": data.get("d_stage"),
        "status": status,
        "replay": asdict(replay_result),
        "video_compare": video_compare,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }


def main() -> int:
    args = parse_args()
    json_path = Path(args.json_path).expanduser().resolve()
    if not json_path.is_file():
        print(f"JSON not found: {json_path}", file=sys.stderr)
        return 1

    data = load_trace_json(json_path)
    stem = sample_stem_from_path(json_path)
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else Path("replay_logs") / "ood_verify" / stem
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(output_dir / "ood_verify.log", encoding="utf-8"),
        ],
        force=True,
    )

    video_views = [v.strip() for v in args.video_views.split(",") if v.strip()]
    logging.info("OOD JSON: %s", json_path)
    logging.info(
        "task=%s config=%s seed=%s ep=%s annotated_step=%s actions=%s",
        data["task_name"],
        data["task_config"],
        data["seed"],
        data.get("episode_idx"),
        data.get("annotated_step"),
        len(data["actions"]),
    )
    logging.info("instruction: %s", data["instruction"])

    try:
        ref_videos = load_ood_reference_videos(data, video_views)
        logging.info("reference videos: %s", {k: str(v) for k, v in ref_videos.items()})
    except (ValueError, FileNotFoundError) as exc:
        logging.error("reference video validation failed: %s", exc)
        report = {
            "json_path": str(json_path),
            "status": "reference_video_missing",
            "error": str(exc),
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        }
        with open(output_dir / "verify_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        return 2

    replay_result = replay_trace_json(
        json_path,
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
    logging.info("replay status=%s steps=%s", replay_result.status, replay_result.replay_steps)

    video_compare = None
    if args.compare_video:
        if replay_result.status != "replay_ok":
            logging.warning("skip video compare because replay status=%s", replay_result.status)
        elif not replay_result.video_paths:
            logging.warning("skip video compare because replay video_paths is empty")
        else:
            num_frames = replay_result.replay_steps
            replay_paths = {k: Path(v) for k, v in replay_result.video_paths.items()}
            video_compare = compare_camera_sets_exact(
                ref_videos,
                replay_paths,
                cameras=video_views,
                num_frames=num_frames,
            )
            for cam in video_compare["cameras"]:
                logging.info(
                    "  [%s] match=%s frames=%s reason=%s",
                    cam["camera"],
                    cam["match"],
                    cam["compared_frames"],
                    cam["reason"],
                )

    report = build_report(
        json_path=json_path,
        data=data,
        replay_result=replay_result,
        video_compare=video_compare,
        output_dir=output_dir,
    )
    report_path = output_dir / "verify_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logging.info("Wrote %s", report_path)
    logging.info("Final status: %s", report["status"])

    if report["status"] == "verify_ok":
        return 0
    if replay_result.status in {"replay_ok", "state_mismatch"}:
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
