"""Local TCP RPC helpers for a persistent pi05 model server."""

from __future__ import annotations

import base64
import json
import socket
import time
from typing import Any

import numpy as np


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return {
                "__numpy_array__": True,
                "data": base64.b64encode(obj.tobytes()).decode("ascii"),
                "dtype": str(obj.dtype),
                "shape": obj.shape,
            }
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def numpy_to_json_bytes(data: Any) -> bytes:
    return json.dumps(data, cls=NumpyEncoder).encode("utf-8")


def json_bytes_to_numpy(data: bytes) -> Any:
    def object_hook(dct):
        if "__numpy_array__" in dct:
            raw = base64.b64decode(dct["data"])
            return np.frombuffer(raw, dtype=dct["dtype"]).reshape(dct["shape"])
        return dct

    return json.loads(data.decode("utf-8"), object_hook=object_hook)


class ModelRpcClient:
    def __init__(self, host: str, port: int, *, timeout: float = 120.0):
        self.host = host
        self.port = int(port)
        self.timeout = float(timeout)
        self.sock: socket.socket | None = None
        self._call_count = 0
        self._call_elapsed_s = 0.0
        self._infer_call_count = 0
        self._infer_elapsed_s = 0.0
        self._last_call_elapsed_s = 0.0
        self._connect()

    def _connect(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)
        sock.connect((self.host, self.port))
        self.sock = sock

    def _recv_exact(self, size: int) -> bytes:
        assert self.sock is not None
        chunks: list[bytes] = []
        remaining = size
        while remaining > 0:
            chunk = self.sock.recv(min(remaining, 4096))
            if not chunk:
                raise ConnectionError("model server connection closed")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def call(self, cmd: str, payload: Any = None) -> Any:
        assert self.sock is not None
        req = {"cmd": cmd, "payload": payload}
        req_bytes = numpy_to_json_bytes(req)
        started = time.perf_counter()
        try:
            self.sock.sendall(len(req_bytes).to_bytes(4, "big"))
            self.sock.sendall(req_bytes)

            resp_len = int.from_bytes(self._recv_exact(4), "big")
            resp = json_bytes_to_numpy(self._recv_exact(resp_len))
            if isinstance(resp, dict) and resp.get("error"):
                raise RuntimeError(f"{resp['error']}\n{resp.get('traceback', '')}".strip())
            if not isinstance(resp, dict) or "result" not in resp:
                raise RuntimeError(f"invalid model server response: {resp!r}")
            return resp["result"]
        finally:
            elapsed = time.perf_counter() - started
            self._call_count += 1
            self._call_elapsed_s += elapsed
            self._last_call_elapsed_s = elapsed
            if cmd == "infer_from_robotwin":
                self._infer_call_count += 1
                self._infer_elapsed_s += elapsed

    def rpc_stats(self) -> dict[str, float | int]:
        """Return cumulative and inference-only timings for the current client."""
        return {
            "calls": self._call_count,
            "elapsed_s": self._call_elapsed_s,
            "infer_calls": self._infer_call_count,
            "infer_elapsed_s": self._infer_elapsed_s,
            "last_elapsed_s": self._last_call_elapsed_s,
        }

    def close(self) -> None:
        if self.sock is not None:
            try:
                self.sock.close()
            finally:
                self.sock = None

    def __enter__(self) -> "ModelRpcClient":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
