#!/usr/bin/env python3
"""Thin wrapper around script/replay_robotwin_trace.py for backward compatibility."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROBOTWIN_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROBOTWIN_ROOT))
sys.path.insert(0, str(ROBOTWIN_ROOT / "script"))

from robotwin_trace_utils import setup_robotwin_paths
from replay_robotwin_trace import replay_trace_json, verify_trace

setup_robotwin_paths()


def main():
    parser = argparse.ArgumentParser(description="Replay RoboTwin failure trace JSON (wrapper).")
    parser.add_argument("json_path", type=str, help="Path to failure trace JSON")
    parser.add_argument("--verify-only", action="store_true", help="Validate JSON and task files only")
    parser.add_argument("--joint-tol", type=float, default=0.05)
    parser.add_argument("--endpose-pos-tol", type=float, default=0.02)
    parser.add_argument("--endpose-rot-tol", type=float, default=0.15)
    parser.add_argument("--gripper-tol", type=float, default=0.05)
    parser.add_argument("--skip-state-mismatch-check", action="store_true")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--screenshot-path", type=str, default="", help="Deprecated; ignored")
    args = parser.parse_args()

    if args.screenshot_path:
        print("warning: --screenshot-path is deprecated and ignored", file=sys.stderr)

    json_path = Path(args.json_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else None

    if args.verify_only:
        result = verify_trace(json_path)
        print("verify_ok")
        print(
            json.dumps(
                {
                    "task_name": result.task_name,
                    "task_config": result.task_config,
                    "seed": result.seed,
                    "episode_idx": result.episode_idx,
                    "actions_len": result.actions_len,
                },
                indent=2,
            )
        )
        return

    result = replay_trace_json(
        json_path,
        joint_tol=args.joint_tol,
        endpose_pos_tol=args.endpose_pos_tol,
        endpose_rot_tol=args.endpose_rot_tol,
        gripper_tol=args.gripper_tol,
        skip_state_mismatch=args.skip_state_mismatch_check,
        save_video=args.save_video,
        output_dir=output_dir,
        video_fps=10,
        video_views=["head_camera"],
    )
    print(json.dumps(result.__dict__, indent=2, default=str))
    if result.status != "replay_ok":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
