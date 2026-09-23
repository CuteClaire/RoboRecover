"""Shared utilities for RoboTwin failure-trace eval and replay."""

from __future__ import annotations

import importlib
import json
import math
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
ROBOTWIN_ROOT = SCRIPT_DIR.parent

REQUIRED_TRACE_FIELDS = [
    "task_suite_name",
    "task_name",
    "task_config",
    "seed",
    "instruction",
    "actions",
    "initial_state",
]

DEFAULT_VIDEO_VIEWS = [
    "head_camera",
    "left_camera",
    "right_camera",
    "observer_camera",
    "world_camera1",
    "world_camera2",
]


def setup_robotwin_paths() -> None:
    os.chdir(ROBOTWIN_ROOT)
    for path in (ROBOTWIN_ROOT, ROBOTWIN_ROOT / "policy", ROBOTWIN_ROOT / "description" / "utils"):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


@dataclass
class InitialStateErrors:
    joint_err: float
    left_endpose_pos_err: float | None = None
    left_endpose_rot_err: float | None = None
    right_endpose_pos_err: float | None = None
    right_endpose_rot_err: float | None = None
    left_gripper_err: float | None = None
    right_gripper_err: float | None = None

    def exceeds_tolerance(
        self,
        *,
        joint_tol: float,
        endpose_pos_tol: float,
        endpose_rot_tol: float,
        gripper_tol: float,
    ) -> bool:
        if self.joint_err > joint_tol:
            return True
        for pos_err in (self.left_endpose_pos_err, self.right_endpose_pos_err):
            if pos_err is not None and pos_err > endpose_pos_tol:
                return True
        for rot_err in (self.left_endpose_rot_err, self.right_endpose_rot_err):
            if rot_err is not None and rot_err > endpose_rot_tol:
                return True
        for grip_err in (self.left_gripper_err, self.right_gripper_err):
            if grip_err is not None and grip_err > gripper_tol:
                return True
        return False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_trace_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    missing = [k for k in REQUIRED_TRACE_FIELDS if k not in data]
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    actions = data["actions"]
    if not isinstance(actions, list) or len(actions) == 0:
        raise ValueError("actions empty")
    step = data.get("annotated_step")
    if step is not None:
        step = int(step)
        if step < 0:
            raise ValueError(f"invalid annotated_step: {step}")
        if len(actions) > step + 1:
            data = dict(data)
            data["actions"] = actions[: step + 1]
    return data


def get_embodiment_config(robot_file: str) -> dict:
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        return yaml.load(f.read(), Loader=yaml.FullLoader)


def get_camera_config(camera_type: str) -> dict:
    from envs._GLOBAL_CONFIGS import CONFIGS_PATH

    camera_config_path = os.path.join(CONFIGS_PATH, "_camera_config.yml")
    if not os.path.isfile(camera_config_path):
        raise FileNotFoundError(camera_config_path)
    with open(camera_config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)
    if camera_type not in args:
        raise KeyError(f"camera {camera_type} is not defined")
    return args[camera_type]


def build_task_args(task_name: str, task_config: str, *, eval_video_save_dir: Path | None = None) -> dict:
    from envs._GLOBAL_CONFIGS import CONFIGS_PATH

    task_cfg_path = ROBOTWIN_ROOT / "task_config" / f"{task_config}.yml"
    with open(task_cfg_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args["task_name"] = task_name
    args["task_config"] = task_config
    args["eval_mode"] = True
    args["render_freq"] = 0
    if eval_video_save_dir is not None:
        args["eval_video_save_dir"] = eval_video_save_dir

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")
    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(name):
        robot_file = embodiment_types[name]["file_path"]
        if robot_file is None:
            raise RuntimeError("No embodiment files")
        return robot_file

    with open(os.path.join(CONFIGS_PATH, "_camera_config.yml"), "r", encoding="utf-8") as f:
        camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)

    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = camera_config[head_camera_type]["h"]
    args["head_camera_w"] = camera_config[head_camera_type]["w"]

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise RuntimeError("embodiment items should be 1 or 3")

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])
    return args


def class_decorator(task_name: str):
    envs_module = importlib.import_module(f"envs.{task_name}")
    env_class = getattr(envs_module, task_name)
    return env_class()


def quat_geodesic_error(q1: np.ndarray, q2: np.ndarray) -> float:
    q1 = np.asarray(q1, dtype=np.float64)
    q2 = np.asarray(q2, dtype=np.float64)
    q1 = q1 / max(np.linalg.norm(q1), 1e-12)
    q2 = q2 / max(np.linalg.norm(q2), 1e-12)
    dot = abs(float(np.dot(q1, q2)))
    dot = min(max(dot, -1.0), 1.0)
    return float(2.0 * math.acos(dot))


def endpose_errors(current: np.ndarray, expected: np.ndarray) -> tuple[float, float]:
    current = np.asarray(current, dtype=np.float64).reshape(-1)
    expected = np.asarray(expected, dtype=np.float64).reshape(-1)
    if current.shape[0] < 7 or expected.shape[0] < 7:
        raise ValueError(f"endpose must be 7D, got {current.shape} vs {expected.shape}")
    pos_err = float(np.linalg.norm(current[:3] - expected[:3]))
    rot_err = quat_geodesic_error(current[3:7], expected[3:7])
    return pos_err, rot_err


def initial_state_error(observation: dict, expected: dict) -> InitialStateErrors:
    current_joint = np.asarray(observation["joint_action"]["vector"], dtype=np.float64)
    target_joint = np.asarray(expected["joint_vector"], dtype=np.float64)
    if current_joint.shape != target_joint.shape:
        raise ValueError(f"joint_vector shape mismatch: {current_joint.shape} vs {target_joint.shape}")
    joint_err = float(np.linalg.norm(current_joint - target_joint))

    left_pos_err = left_rot_err = right_pos_err = right_rot_err = None
    left_gripper_err = right_gripper_err = None

    endpose_obs = observation.get("endpose", {})
    if "left_endpose" in expected:
        obs_pose = endpose_obs.get("left_endpose", expected["left_endpose"])
        left_pos_err, left_rot_err = endpose_errors(obs_pose, expected["left_endpose"])
    if "right_endpose" in expected:
        obs_pose = endpose_obs.get("right_endpose", expected["right_endpose"])
        right_pos_err, right_rot_err = endpose_errors(obs_pose, expected["right_endpose"])
    if "left_gripper" in expected:
        obs_grip = float(endpose_obs.get("left_gripper", expected["left_gripper"]))
        left_gripper_err = abs(obs_grip - float(expected["left_gripper"]))
    if "right_gripper" in expected:
        obs_grip = float(endpose_obs.get("right_gripper", expected["right_gripper"]))
        right_gripper_err = abs(obs_grip - float(expected["right_gripper"]))

    return InitialStateErrors(
        joint_err=joint_err,
        left_endpose_pos_err=left_pos_err,
        left_endpose_rot_err=left_rot_err,
        right_endpose_pos_err=right_pos_err,
        right_endpose_rot_err=right_rot_err,
        left_gripper_err=left_gripper_err,
        right_gripper_err=right_gripper_err,
    )


def verify_trace_json(path: Path) -> dict[str, Any]:
    data = load_trace_json(path)
    task_name = data["task_name"]
    task_config = data["task_config"]
    env_py = ROBOTWIN_ROOT / "envs" / f"{task_name}.py"
    task_cfg = ROBOTWIN_ROOT / "task_config" / f"{task_config}.yml"
    if not env_py.is_file():
        raise FileNotFoundError(env_py)
    if not task_cfg.is_file():
        raise FileNotFoundError(task_cfg)
    build_task_args(task_name, task_config)
    return data


def get_eval_video_camera_specs(task_args: dict, video_views: list[str] | None = None) -> list[tuple[str, str | None]]:
    all_specs: list[tuple[str, str | None]] = []
    camera_cfg = task_args["camera"]
    if camera_cfg.get("collect_head_camera", True):
        all_specs.append(("head_camera", camera_cfg["head_camera_type"]))
    if camera_cfg.get("collect_wrist_camera", True):
        wrist_type = camera_cfg.get("wrist_camera_type", camera_cfg["head_camera_type"])
        all_specs.append(("left_camera", wrist_type))
        all_specs.append(("right_camera", wrist_type))
    all_specs.extend(
        [
            ("observer_camera", None),
            ("world_camera1", None),
            ("world_camera2", None),
        ]
    )
    if video_views is None:
        return all_specs
    allowed = set(video_views)
    return [spec for spec in all_specs if spec[0] in allowed]


def get_video_size_for_camera(camera_name: str, camera_type: str | None) -> str:
    if camera_name == "observer_camera":
        return "320x240"
    if camera_name in ("world_camera1", "world_camera2"):
        return "640x480"
    camera_config = get_camera_config(camera_type)
    return f"{camera_config['w']}x{camera_config['h']}"


def start_eval_video_recording(
    task_env,
    save_dir: Path,
    sample_stem: str,
    task_args: dict,
    *,
    video_fps: int = 10,
    video_views: list[str] | None = None,
) -> dict[str, str]:
    video_dir = Path(save_dir) / "videos" / sample_stem
    video_dir.mkdir(parents=True, exist_ok=True)
    ffmpegs = {}
    video_paths: dict[str, str] = {}
    for camera_name, camera_type in get_eval_video_camera_specs(task_args, video_views):
        video_size = get_video_size_for_camera(camera_name, camera_type)
        out_path = video_dir / f"{camera_name}.mp4"
        ffmpeg = subprocess.Popen(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pixel_format",
                "rgb24",
                "-video_size",
                video_size,
                "-framerate",
                str(video_fps),
                "-i",
                "-",
                "-pix_fmt",
                "yuv420p",
                "-vcodec",
                "libx264",
                "-crf",
                "23",
                str(out_path),
            ],
            stdin=subprocess.PIPE,
        )
        ffmpegs[camera_name] = ffmpeg
        video_paths[camera_name] = str(out_path.resolve())
    task_env._set_eval_video_ffmpegs(ffmpegs)
    return video_paths


def discover_trace_json_paths(
    *,
    json_path: Path | None = None,
    json_root: Path | None = None,
    sample_file_list: Path | None = None,
    limit: int = -1,
) -> list[Path]:
    paths: list[Path] = []
    if json_path is not None:
        paths.append(json_path.expanduser().resolve())
    if sample_file_list is not None:
        with open(sample_file_list, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    paths.append(Path(line).expanduser().resolve())
    if json_root is not None:
        root = json_root.expanduser().resolve()
        paths.extend(
            sorted(
                path
                for path in root.rglob("*.json")
                if path.is_file() and path.name != OOD_INDEX_FILENAME
            )
        )
    # dedupe preserving order
    seen = set()
    unique_paths: list[Path] = []
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique_paths.append(path)
    if limit >= 0:
        unique_paths = unique_paths[:limit]
    return unique_paths


def sample_stem_from_path(path: Path) -> str:
    return path.stem


OOD_INDEX_FILENAME = "annotations_index.json"


def discover_ood_json_paths(ood_root: Path, *, limit: int = -1) -> list[Path]:
    root = ood_root.expanduser().resolve()
    paths = sorted(
        path
        for path in root.rglob("*.json")
        if path.is_file() and path.name != OOD_INDEX_FILENAME
    )
    if limit >= 0:
        paths = paths[:limit]
    return paths


def load_ood_reference_videos(data: dict[str, Any], cameras: list[str] | None = None) -> dict[str, Path]:
    video_paths = data.get("video_paths") or {}
    if not video_paths:
        raise ValueError("OOD JSON missing video_paths")
    views = cameras or DEFAULT_VIDEO_VIEWS
    resolved: dict[str, Path] = {}
    missing: list[str] = []
    for view in views:
        raw = video_paths.get(view)
        if not raw:
            missing.append(view)
            continue
        path = Path(raw).expanduser().resolve()
        if not path.is_file():
            missing.append(view)
            continue
        resolved[view] = path
    if missing:
        raise FileNotFoundError(f"reference videos missing or not found for views: {missing}")
    return resolved
