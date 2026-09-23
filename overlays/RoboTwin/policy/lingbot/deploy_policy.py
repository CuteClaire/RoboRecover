from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


def _ensure_lingbot_path(lingbot_root: str | None) -> None:
    root = Path(lingbot_root or "/path/to/workspace/lingbot-vla").expanduser().resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


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


class LingBotClient:
    def __init__(self, host: str, port: int, max_actions_per_call: int | None):
        if int(port) <= 0:
            raise ValueError("model_server_port must be > 0 for LingBot source rollout")
        from deploy.websocket_client_policy import WebsocketClientPolicy

        self.client = WebsocketClientPolicy(host=host, port=int(port))
        self.max_actions_per_call = max_actions_per_call

    def reset_model(self) -> None:
        self.client.reset("robotwin")

    def infer_from_robotwin(self, task_env, observation: dict[str, Any]) -> np.ndarray:
        obs_images = observation.get("observation", {})
        payload = {
            "task": task_env.get_instruction(),
            "observation.images.cam_high": _rgb_uint8(obs_images["head_camera"]["rgb"], "head_camera"),
            "observation.images.cam_left_wrist": _rgb_uint8(obs_images["left_camera"]["rgb"], "left_camera"),
            "observation.images.cam_right_wrist": _rgb_uint8(obs_images["right_camera"]["rgb"], "right_camera"),
            "observation.state": np.asarray(observation["joint_action"]["vector"], dtype=np.float32),
        }
        actions = _lingbot_response_to_joint_actions(self.client.infer(payload))
        if self.max_actions_per_call is not None and self.max_actions_per_call > 0:
            actions = actions[: self.max_actions_per_call]
        return actions


def _optional_int(value, default: int | None = None) -> int | None:
    if value in (None, "", "None"):
        return default
    return int(value)


def get_model(usr_args):
    _ensure_lingbot_path(usr_args.get("lingbot_root"))
    host = usr_args.get("model_server_host", "127.0.0.1")
    port = int(usr_args.get("model_server_port", 0))
    max_actions = _optional_int(usr_args.get("max_actions_per_call", 50), default=50)
    os.environ.setdefault("ROBOTWIN_SKIP_CUROBO_PLANNER", "1")
    return LingBotClient(host, port, max_actions)


def _run_policy_step(task_env, model: LingBotClient, observation, trace=None):
    actions = model.infer_from_robotwin(task_env, observation)
    if trace is not None:
        trace.setdefault("action_type", "qpos")
        trace.setdefault("action_format", "robotwin_qpos_14d")
    for action in actions:
        if trace is not None:
            trace["actions"].append(np.asarray(action, dtype=np.float64).tolist())
        task_env.take_action(action)
        if task_env.eval_success or task_env.take_action_cnt >= task_env.step_lim:
            break


def eval(TASK_ENV, model, observation):
    _run_policy_step(TASK_ENV, model, observation)


def eval_with_trace(TASK_ENV, model, observation, trace):
    _run_policy_step(TASK_ENV, model, observation, trace=trace)


def reset_model(model):
    model.reset_model()
