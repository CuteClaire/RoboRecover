from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROBOTWIN_ROOT = Path(__file__).resolve().parents[2]
ROBOTWIN_SCRIPT = ROBOTWIN_ROOT / "script"
if str(ROBOTWIN_SCRIPT) not in sys.path:
    sys.path.insert(0, str(ROBOTWIN_SCRIPT))

from model_rpc import ModelRpcClient


class SmolVlaRpcPolicy:
    def __init__(self, host: str, port: int, timeout: float, max_actions_per_call: int | None):
        if int(port) <= 0:
            raise ValueError("model_server_port must be > 0 for SmolVLA source rollout")
        self.client = ModelRpcClient(host, int(port), timeout=float(timeout))
        self.max_actions_per_call = max_actions_per_call

    def reset_model(self) -> None:
        self.client.call("reset_model")

    def close(self) -> None:
        self.client.close()

    def infer_from_robotwin(self, task_env, observation: dict[str, Any]) -> np.ndarray:
        obs_images = observation.get("observation", {})
        payload = {
            "instruction": task_env.get_instruction(),
            "images": {
                "head_camera": obs_images["head_camera"]["rgb"],
                "left_camera": obs_images["left_camera"]["rgb"],
                "right_camera": obs_images["right_camera"]["rgb"],
            },
            "state": observation["joint_action"]["vector"],
        }
        actions = np.asarray(self.client.call("infer_from_robotwin", payload), dtype=np.float64)
        if actions.ndim == 1:
            actions = actions[None, :]
        if actions.ndim != 2 or actions.shape[1] != 14:
            raise ValueError(f"expected SmolVLA action chunk [T,14], got {actions.shape}")
        if self.max_actions_per_call is not None and self.max_actions_per_call > 0:
            actions = actions[: self.max_actions_per_call]
        return actions


def _optional_int(value, default: int | None = None) -> int | None:
    if value in (None, "", "None"):
        return default
    return int(value)


def get_model(usr_args):
    host = usr_args.get("model_server_host", "127.0.0.1")
    port = int(usr_args.get("model_server_port", 0))
    timeout = float(usr_args.get("model_server_timeout", 180.0))
    max_actions = _optional_int(usr_args.get("max_actions_per_call", 50), default=50)
    os.environ.setdefault("ROBOTWIN_SKIP_CUROBO_PLANNER", "1")
    return SmolVlaRpcPolicy(host, port, timeout, max_actions)


def _run_policy_step(task_env, model: SmolVlaRpcPolicy, observation, trace=None):
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
