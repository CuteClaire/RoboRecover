#!/usr/bin/env python3
"""Lossless RoboTwin episode recorder for monitor/VLA training data.

The recorder wraps ``task_env.take_action``.  This keeps replay and inference
actions on the exact same observation/action timeline while leaving the task
environment implementation untouched.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


CAMERAS = ("head_camera", "left_camera", "right_camera")


def _json_value(value: Any) -> Any:
    """Convert numpy-rich RoboTwin observations into JSON-safe values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _state_from_obs(obs: dict[str, Any]) -> dict[str, Any]:
    state: dict[str, Any] = {}
    joint_action = obs.get("joint_action")
    if isinstance(joint_action, dict) and "vector" in joint_action:
        state["qpos"] = _json_value(np.asarray(joint_action["vector"], dtype=np.float64))
    if "endpose" in obs:
        state["ee_pose"] = _json_value(obs["endpose"])
    return state


@dataclass
class _ActionContext:
    phase: str = "replay"
    policy_action_raw: Any | None = None
    replay_source_action: Any | None = None
    infer_call_idx: int | None = None
    chunk_offset: int | None = None
    planner_status: str | None = None


@dataclass
class EpisodeDatasetRecorder:
    """Write one complete episode in a self-contained, atomic directory."""

    output_dir: Path
    metadata: dict[str, Any]
    image_format: str = "png"
    partial_dir: Path = field(init=False)
    final_dir: Path = field(init=False)
    frame_idx: int = field(default=0, init=False)
    transitions: list[dict[str, Any]] = field(default_factory=list, init=False)
    _context: _ActionContext = field(default_factory=_ActionContext, init=False)
    _original_take_action: Any | None = field(default=None, init=False)
    _task_env: Any | None = field(default=None, init=False)
    _finalized: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if self.image_format.lower() != "png":
            raise ValueError("Only lossless png is supported for dataset collection")
        self.output_dir = Path(self.output_dir)
        self.partial_dir = self.output_dir / "dataset_episode.partial"
        self.final_dir = self.output_dir / "dataset_episode"
        if self.final_dir.exists():
            raise FileExistsError(f"dataset episode already finalized: {self.final_dir}")
        if self.partial_dir.exists():
            shutil.rmtree(self.partial_dir)
        for camera in CAMERAS:
            (self.partial_dir / "frames" / camera).mkdir(parents=True, exist_ok=True)

    def attach(self, task_env: Any) -> None:
        if self._original_take_action is not None:
            raise RuntimeError("recorder is already attached")
        self._task_env = task_env
        self._original_take_action = task_env.take_action

        def wrapped_take_action(action: Any, action_type: str = "qpos") -> Any:
            return self._recorded_take_action(action, action_type=action_type)

        task_env.take_action = wrapped_take_action

    def set_action_context(
        self,
        *,
        phase: str,
        policy_action_raw: Any | None = None,
        replay_source_action: Any | None = None,
        infer_call_idx: int | None = None,
        chunk_offset: int | None = None,
        planner_status: str | None = None,
    ) -> None:
        self._context = _ActionContext(
            phase=phase,
            policy_action_raw=policy_action_raw,
            replay_source_action=replay_source_action,
            infer_call_idx=infer_call_idx,
            chunk_offset=chunk_offset,
            planner_status=planner_status,
        )

    def set_phase(self, phase: str) -> None:
        self._context.phase = phase

    def _save_images(self, observation: dict[str, Any], *, phase: str, terminal: bool) -> tuple[int, dict[str, str]]:
        try:
            from PIL import Image
        except ImportError as exc:  # pragma: no cover - installation error is environment-specific
            raise RuntimeError("Pillow is required for --save-dataset-episode") from exc

        images = observation.get("observation", {})
        frame = self.frame_idx
        relative_paths: dict[str, str] = {}
        for camera in CAMERAS:
            rgb = np.asarray(images[camera]["rgb"], dtype=np.uint8)
            if rgb.ndim != 3 or rgb.shape[-1] != 3:
                raise ValueError(f"invalid {camera} RGB shape: {rgb.shape}")
            relative = Path("frames") / camera / f"{frame:06d}.png"
            Image.fromarray(rgb, mode="RGB").save(self.partial_dir / relative, format="PNG", compress_level=1)
            relative_paths[camera] = relative.as_posix()
        self.frame_idx += 1
        return frame, relative_paths

    def _recorded_take_action(self, command_action: Any, *, action_type: str) -> Any:
        if self._task_env is None or self._original_take_action is None:
            raise RuntimeError("recorder is not attached")
        task_env = self._task_env
        context = self._context
        before = task_env.get_obs()
        frame_idx, image_paths = self._save_images(before, phase=context.phase, terminal=False)
        before_count = int(getattr(task_env, "take_action_cnt", 0))
        result = self._original_take_action(command_action, action_type=action_type)
        after = task_env.get_obs()
        after_count = int(getattr(task_env, "take_action_cnt", before_count))
        command = _json_value(np.asarray(command_action, dtype=np.float64))
        replay_source_action = context.replay_source_action
        if context.phase.startswith("replay") and replay_source_action is None:
            replay_source_action = command
        transition = {
            "global_action_idx": len(self.transitions),
            "phase": context.phase,
            "phase_action_idx": sum(1 for row in self.transitions if row["phase"] == context.phase),
            "frame_idx": frame_idx,
            "images": image_paths,
            "policy_action_raw": _json_value(context.policy_action_raw),
            "replay_source_action": _json_value(replay_source_action),
            "command_action": command,
            "command_action_type": action_type,
            "infer_call_idx": context.infer_call_idx,
            "chunk_offset": context.chunk_offset,
            "planner_status": context.planner_status,
            "qpos_before": _state_from_obs(before).get("qpos"),
            "qpos_after": _state_from_obs(after).get("qpos"),
            "ee_pose_before": _state_from_obs(before).get("ee_pose"),
            "ee_pose_after": _state_from_obs(after).get("ee_pose"),
            "take_action_count_before": before_count,
            "take_action_count_after": after_count,
            "action_executed": after_count > before_count,
            "eval_success_after": bool(getattr(task_env, "eval_success", False)),
        }
        self.transitions.append(transition)
        self._context = _ActionContext(phase=context.phase)
        return result

    def finalize(
        self,
        *,
        task_env: Any | None,
        status: str,
        success: bool | None,
        error: str | None = None,
    ) -> Path | None:
        if self._finalized:
            return self.final_dir if self.final_dir.exists() else None
        self._finalized = True
        try:
            terminal: dict[str, Any] | None = None
            if task_env is not None:
                try:
                    terminal = task_env.get_obs()
                except Exception as exc:  # keep a useful partial diagnostic
                    error = error or f"terminal_observation_failed: {type(exc).__name__}: {exc}"
            terminal_record: dict[str, Any] | None = None
            if terminal is not None:
                frame_idx, image_paths = self._save_images(terminal, phase=self._context.phase, terminal=True)
                terminal_record = {
                    "frame_idx": frame_idx,
                    "phase": self._context.phase,
                    "images": image_paths,
                    "qpos": _state_from_obs(terminal).get("qpos"),
                    "ee_pose": _state_from_obs(terminal).get("ee_pose"),
                }

            infer_label = 0 if bool(success) else 1
            for row in self.transitions:
                row["ood_label"] = 0 if row["phase"].startswith("replay") else infer_label
            if terminal_record is not None:
                terminal_record["ood_label"] = 0 if terminal_record["phase"].startswith("replay") else infer_label

            episode = {
                **_json_value(self.metadata),
                "image_format": self.image_format,
                "cameras": list(CAMERAS),
                "frame_shape_hwc": [240, 320, 3],
                "status": status,
                "success": success,
                "error": error,
                "action_count": len(self.transitions),
                "frame_count": self.frame_idx,
                "terminal": terminal_record,
            }
            with (self.partial_dir / "transitions.jsonl").open("w", encoding="utf-8") as handle:
                for row in self.transitions:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            with (self.partial_dir / "episode.json").open("w", encoding="utf-8") as handle:
                json.dump(episode, handle, ensure_ascii=False, indent=2)
            (self.partial_dir / "DONE").write_text("\n", encoding="utf-8")
            os.replace(self.partial_dir, self.final_dir)
            return self.final_dir
        except Exception:
            # Keep the .partial directory for a retry/debugger.  It intentionally has no DONE.
            raise

