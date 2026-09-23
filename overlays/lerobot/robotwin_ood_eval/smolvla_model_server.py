#!/usr/bin/env python3
"""Persistent local SmolVLA model server for RoboTwin OOD replay eval."""

from __future__ import annotations

import argparse
import logging
import os
import socket
import sys
import threading
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
LEROBOT_ROOT = SCRIPT_DIR.parent
ROBOTWIN_ROOT = Path(os.environ["ROBOTWIN_ROOT"])
ROBOTWIN_SCRIPT = ROBOTWIN_ROOT / "script"
for path in (LEROBOT_ROOT / "src", ROBOTWIN_SCRIPT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from model_rpc import json_bytes_to_numpy, numpy_to_json_bytes


CAMERA_KEY_MAP = {
    "head_camera": "observation.images.cam_high",
    "left_camera": "observation.images.cam_left_wrist",
    "right_camera": "observation.images.cam_right_wrist",
}


def _to_chw_float(img: np.ndarray) -> torch.Tensor:
    arr = np.asarray(img, dtype=np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[-1] != 3:
        arr = arr[..., :3]
    return torch.from_numpy(arr).float().permute(2, 0, 1).contiguous() / 255.0


class SmolVlaRuntime:
    def __init__(
        self,
        model_path: str,
        device: str,
        action_chunk_size: int | None = None,
        inference_mode: str = "chunk",
    ):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        from lerobot.configs import PreTrainedConfig
        from lerobot.policies import get_policy_class, make_pre_post_processors

        cfg = PreTrainedConfig.from_pretrained(model_path)
        cfg.device = device
        policy_cls = get_policy_class(cfg.type)
        self.policy = policy_cls.from_pretrained(model_path, config=cfg)
        self.policy.to(cfg.device)
        self.policy.eval()

        device_override = {"device_processor": {"device": str(cfg.device)}}
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=cfg,
            pretrained_path=model_path,
            preprocessor_overrides=device_override,
            postprocessor_overrides=device_override,
        )
        self.device = torch.device(cfg.device)
        self.action_chunk_size = action_chunk_size
        self.inference_mode = inference_mode
        self.lock = threading.Lock()
        logging.info(
            "loaded SmolVLA policy type=%s device=%s mode=%s model=%s",
            cfg.type,
            cfg.device,
            self.inference_mode,
            model_path,
        )

    def reset_model(self) -> bool:
        with self.lock:
            self.policy.reset()
        return True

    def infer_from_robotwin(self, payload: dict[str, Any]) -> np.ndarray:
        observation = {
            "observation.state": torch.from_numpy(
                np.asarray(payload["state"], dtype=np.float32).reshape(-1)
            ),
            "task": str(payload.get("instruction") or ""),
        }
        images = payload.get("images") or {}
        for camera, feature_key in CAMERA_KEY_MAP.items():
            if camera not in images:
                continue
            observation[feature_key] = _to_chw_float(images[camera])

        with self.lock, torch.inference_mode():
            batch = self.preprocessor(observation)
            if self.inference_mode == "select_action":
                action = self.policy.select_action(batch)
            elif hasattr(self.policy, "predict_action_chunk"):
                action = self.policy.predict_action_chunk(batch)
            else:
                action = self.policy.select_action(batch).unsqueeze(1)
            action = self.postprocessor(action)

        if isinstance(action, torch.Tensor):
            arr = action.detach().cpu().numpy()
        else:
            arr = np.asarray(action)
        if arr.ndim == 1:
            arr = arr[None, :]
        elif arr.ndim == 3:
            arr = arr[0]
        if self.action_chunk_size is not None and self.action_chunk_size > 0:
            arr = arr[: self.action_chunk_size]
        if arr.ndim != 2 or arr.shape[1] != 14:
            raise ValueError(f"expected SmolVLA action chunk [T,14], got {arr.shape}")
        return arr.astype(np.float64, copy=False)


class RpcServer:
    def __init__(self, host: str, port: int, runtime: SmolVlaRuntime):
        self.host = host
        self.port = int(port)
        self.runtime = runtime
        self.sock: socket.socket | None = None
        self.running = False
        self.threads: list[threading.Thread] = []

    def start(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.host, self.port))
        sock.listen(16)
        self.sock = sock
        self.running = True
        logging.info("SmolVLA model server listening on %s:%d", self.host, self.port)
        try:
            while self.running:
                client, addr = sock.accept()
                thread = threading.Thread(target=self._handle_client, args=(client, addr), daemon=True)
                thread.start()
                self.threads.append(thread)
        finally:
            self.stop()

    def stop(self) -> None:
        self.running = False
        if self.sock is not None:
            try:
                self.sock.close()
            except Exception:
                pass
            self.sock = None

    def _recv_exact(self, client: socket.socket, size: int) -> bytes:
        chunks: list[bytes] = []
        remaining = size
        while remaining > 0:
            chunk = client.recv(min(remaining, 4096))
            if not chunk:
                raise ConnectionError("client disconnected")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def _handle_client(self, client: socket.socket, addr) -> None:
        logging.info("client connected: %s", addr)
        with client:
            while self.running:
                try:
                    req_len_raw = client.recv(4)
                    if not req_len_raw:
                        break
                    req_len = int.from_bytes(req_len_raw, "big")
                    req = json_bytes_to_numpy(self._recv_exact(client, req_len))
                    cmd = req.get("cmd")
                    payload = req.get("payload")
                    if not cmd:
                        raise ValueError("missing cmd")
                    method = getattr(self.runtime, cmd)
                    result = method(payload) if payload is not None else method()
                    resp = {"result": result}
                except Exception as exc:
                    logging.exception("request failed from %s", addr)
                    resp = {"error": str(exc), "traceback": traceback.format_exc()}
                    resp_bytes = numpy_to_json_bytes(resp)
                    client.sendall(len(resp_bytes).to_bytes(4, "big"))
                    client.sendall(resp_bytes)
                    break
                resp_bytes = numpy_to_json_bytes(resp)
                client.sendall(len(resp_bytes).to_bytes(4, "big"))
                client.sendall(resp_bytes)
        logging.info("client disconnected: %s", addr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Persistent SmolVLA RPC server")
    parser.add_argument("--model-path", default="/path/to/workspace/lerobot/checkpoints/smolvla_robotwin")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--action-chunk-size", type=int, default=50)
    parser.add_argument("--inference-mode", choices=["chunk", "select_action"], default="chunk")
    parser.add_argument("--log-path", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if args.log_path:
        handlers.append(logging.FileHandler(args.log_path, encoding="utf-8"))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=handlers, force=True)
    runtime = SmolVlaRuntime(args.model_path, args.device, args.action_chunk_size, args.inference_mode)
    server = RpcServer(args.host, args.port, runtime)
    try:
        server.start()
    except KeyboardInterrupt:
        logging.info("keyboard interrupt, stopping server")
    finally:
        server.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
