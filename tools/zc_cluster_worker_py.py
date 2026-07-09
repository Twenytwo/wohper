#!/usr/bin/env python3
"""Tiny Wohper cluster worker for Windows/Linux smoke tests.

Protocol-compatible with server::cluster:
- 4-byte little-endian frame length
- JSON payload with serde enum tag "type"
"""

from __future__ import annotations

import argparse
import json
import socket
import struct
from typing import Any


MAX_FRAME_BYTES = 256 * 1024 * 1024


def read_exact(conn: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = conn.recv(size - len(chunks))
        if not chunk:
            raise ConnectionError("connection closed while reading frame")
        chunks.extend(chunk)
    return bytes(chunks)


def read_frame(conn: socket.socket) -> dict[str, Any]:
    (length,) = struct.unpack("<I", read_exact(conn, 4))
    if length > MAX_FRAME_BYTES:
        raise ValueError(f"frame too large: {length} bytes")
    return json.loads(read_exact(conn, length).decode("utf-8"))


def write_frame(conn: socket.socket, message: dict[str, Any]) -> None:
    payload = json.dumps(message, separators=(",", ":")).encode("utf-8")
    conn.sendall(struct.pack("<I", len(payload)))
    conn.sendall(payload)


def handle(message: dict[str, Any]) -> dict[str, Any]:
    msg_type = message.get("type")
    request_id = str(message.get("request_id", "unknown"))
    if msg_type == "HiddenState":
        next_layer = int(message.get("next_layer", -1))
        hidden_size = int(message.get("hidden_size", 0))
        hidden_states = message.get("hidden_states") or []
        print(
            "hidden_state "
            f"request_id={request_id} "
            f"next_layer={next_layer} "
            f"hidden_size={hidden_size} "
            f"payload_f32={len(hidden_states)}",
            flush=True,
        )
        return {
            "type": "Ack",
            "request_id": request_id,
            "message": (
                f"python worker accepted hidden state for layer {next_layer} "
                f"({len(hidden_states)} f32)"
            ),
        }
    return {
        "type": "Error",
        "request_id": request_id,
        "message": f"unsupported message type: {msg_type}",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9000)
    args = parser.parse_args()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((args.host, args.port))
        server.listen()
        print(f"wohper python worker listening on {args.host}:{args.port}", flush=True)
        while True:
            conn, addr = server.accept()
            with conn:
                try:
                    message = read_frame(conn)
                    print(f"connection from {addr[0]}:{addr[1]}", flush=True)
                    write_frame(conn, handle(message))
                except Exception as exc:  # noqa: BLE001 - smoke-test server prints and keeps running.
                    print(f"worker error: {exc}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
