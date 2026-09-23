#!/usr/bin/env python3
"""Persistent local pi05 model server for OOD replay-then-infer eval."""

from __future__ import annotations

import argparse
import logging
import socket
import sys
import threading
import time
import traceback
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROBOTWIN_ROOT = SCRIPT_DIR.parent
for path in (ROBOTWIN_ROOT, ROBOTWIN_ROOT / "policy"):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from model_rpc import json_bytes_to_numpy, numpy_to_json_bytes
from policy.pi05.deploy_policy import get_model


class Pi05RpcServer:
    def __init__(self, host: str, port: int, model):
        self.host = host
        self.port = int(port)
        self.model = model
        self.sock: socket.socket | None = None
        self.running = False
        self.threads: list[threading.Thread] = []
        self.lock = threading.Lock()

    def start(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.host, self.port))
        sock.listen(16)
        self.sock = sock
        self.running = True
        logging.info("pi05 model server listening on %s:%d", self.host, self.port)
        try:
            while self.running:
                client, addr = sock.accept()
                t = threading.Thread(target=self._handle_client, args=(client, addr), daemon=True)
                t.start()
                self.threads.append(t)
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

    def _call_model(self, cmd: str, payload):
        with self.lock:
            method = getattr(self.model, cmd)
            if payload is None:
                return method()
            return method(payload)

    def _summarize_payload(self, payload) -> str:
        if payload is None:
            return "none"
        if not isinstance(payload, dict):
            return type(payload).__name__
        parts: list[str] = []
        images = payload.get("images")
        if isinstance(images, list):
            image_shapes = [getattr(image, "shape", None) for image in images]
            parts.append(f"images={image_shapes}")
        state = payload.get("state")
        if state is not None:
            parts.append(f"state={getattr(state, 'shape', None)}")
        if "pi0_step" in payload:
            parts.append(f"pi0_step={payload.get('pi0_step')}")
        if payload.get("instruction"):
            parts.append("instruction=yes")
        return " ".join(parts) if parts else ",".join(sorted(payload.keys()))

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
                    started = time.monotonic()
                    logging.info("request begin from %s cmd=%s %s", addr, cmd, self._summarize_payload(payload))
                    result = self._call_model(cmd, payload)
                    duration = time.monotonic() - started
                    result_shape = getattr(result, "shape", None)
                    logging.info("request done from %s cmd=%s duration=%.3fs result_shape=%s", addr, cmd, duration, result_shape)
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
                logging.info("response sent to %s cmd=%s bytes=%d", addr, cmd, len(resp_bytes))
        logging.info("client disconnected: %s", addr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Persistent pi05 model server for local eval")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--train-config-name", type=str, default="pi05_base_finetune_on_robotwin_clean_randomized_joint_training")
    parser.add_argument("--model-name", type=str, default="pi05_robotwin2")
    parser.add_argument("--checkpoint-id", type=str, default="final")
    parser.add_argument("--pi0-step", type=int, default=32)
    parser.add_argument("--log-path", type=str, default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    handlers = [logging.StreamHandler(sys.stdout)]
    if args.log_path:
        handlers.append(logging.FileHandler(args.log_path, encoding="utf-8"))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=handlers, force=True)
    model = get_model(
        {
            "train_config_name": args.train_config_name,
            "model_name": args.model_name,
            "checkpoint_id": args.checkpoint_id,
            "pi0_step": args.pi0_step,
        }
    )
    server = Pi05RpcServer(args.host, args.port, model)
    try:
        server.start()
    except KeyboardInterrupt:
        logging.info("keyboard interrupt, stopping server")
    finally:
        server.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
