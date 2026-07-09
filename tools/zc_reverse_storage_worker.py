#!/usr/bin/env python3
"""Reverse storage relay worker connector."""

from __future__ import annotations

import argparse
import http.client
import json
import os
import socket
import struct
import time
from pathlib import Path
from urllib.parse import urlparse


CHUNK_BYTES = 1024 * 1024
MAX_HEADER_BYTES = 1024 * 1024


def read_exact(sock: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = sock.recv(size - len(chunks))
        if not chunk:
            raise ConnectionError("connection closed")
        chunks.extend(chunk)
    return bytes(chunks)


def read_json_frame(sock: socket.socket) -> dict[str, object]:
    (length,) = struct.unpack("<I", read_exact(sock, 4))
    if length > MAX_HEADER_BYTES:
        raise ValueError(f"frame too large: {length}")
    return json.loads(read_exact(sock, length).decode("utf-8"))


def write_json_frame(sock: socket.socket, payload: dict[str, object]) -> None:
    data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    sock.sendall(struct.pack("<I", len(data)))
    sock.sendall(data)


def expert_target_from_path(storage_root: str, request_path: str) -> Path:
    parsed = urlparse(request_path).path
    prefix = "/experts/"
    if not parsed.startswith(prefix):
        raise ValueError("expected /experts/<file>.zcblk")
    name = parsed[len(prefix) :]
    if "/" in name or "\\" in name or not name.endswith(".zcblk"):
        raise ValueError("invalid expert shard name")
    experts_dir = (Path(storage_root) / "experts").resolve()
    experts_dir.mkdir(parents=True, exist_ok=True)
    target = (experts_dir / name).resolve()
    try:
        target.relative_to(experts_dir)
    except ValueError as exc:
        raise ValueError("path escapes experts directory") from exc
    return target


def write_put_body_to_storage(sock: socket.socket, storage_root: str, path: str, body_len: int) -> tuple[int, dict[str, str], bytes]:
    target = expert_target_from_path(storage_root, path)
    temp_target = target.with_suffix(target.suffix + ".part")
    remaining = body_len
    written = 0
    with temp_target.open("wb") as handle:
        while remaining > 0:
            chunk = sock.recv(min(CHUNK_BYTES, remaining))
            if not chunk:
                temp_target.unlink(missing_ok=True)
                raise ConnectionError("master closed while streaming PUT body")
            handle.write(chunk)
            written += len(chunk)
            remaining -= len(chunk)
    temp_target.replace(target)
    body = json.dumps({"ok": True, "stored": target.name, "bytes": written}, indent=2).encode("utf-8")
    return (
        201,
        {
            "Content-Type": "application/json",
            "Content-Length": str(len(body)),
            "X-Wohper-Expert": target.name,
        },
        body,
    )


def serve_forever(master_host: str, master_port: int, storage_base: str, storage_root: str | None) -> None:
    with socket.create_connection((master_host, master_port), timeout=15) as sock:
        sock.settimeout(None)
        print("reverse worker connected; waiting for master requests", flush=True)
        while True:
            request = read_json_frame(sock)
            method = str(request.get("method", "GET"))
            path = str(request.get("path", "/health"))
            body_len = int(request.get("body_len", 0))
            print(f"reverse worker request: {method} {path}", flush=True)

            try:
                if method == "PUT" and storage_root:
                    status, headers, body = write_put_body_to_storage(sock, storage_root, path, body_len)
                    write_json_frame(sock, {"status": status, "headers": headers, "body_len": len(body)})
                    if body:
                        sock.sendall(body)
                    print(f"reverse worker stored: {status} {len(body)} bytes", flush=True)
                    continue

                base = urlparse(storage_base)
                conn = http.client.HTTPConnection(
                    base.hostname or "127.0.0.1", base.port or 9100, timeout=30
                )
                conn.request(method, path)
                response = conn.getresponse()
                body = response.read()
                headers = {key: value for key, value in response.getheaders()}
                headers["Content-Length"] = str(len(body))
                write_json_frame(
                    sock, {"status": response.status, "headers": headers, "body_len": len(body)}
                )
                if method != "HEAD" and body:
                    sock.sendall(body)
                conn.close()
                print(f"reverse worker response: {response.status} {len(body)} bytes", flush=True)
            except Exception as exc:  # noqa: BLE001
                body = f"worker storage error: {exc}".encode("utf-8")
                print(body.decode("utf-8"), flush=True)
                write_json_frame(
                    sock,
                    {
                        "status": 502,
                        "headers": {
                            "Content-Type": "text/plain",
                            "Content-Length": str(len(body)),
                        },
                        "body_len": len(body),
                    },
                )
                if method != "HEAD":
                    sock.sendall(body)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--master-host", required=True)
    parser.add_argument("--master-port", type=int, default=9200)
    parser.add_argument("--storage-base", default="http://127.0.0.1:9100")
    parser.add_argument("--storage-root", default=os.environ.get("ZC_WORKER_STORAGE_ROOT"))
    parser.add_argument("--retry-seconds", type=float, default=1.0)
    args = parser.parse_args()

    print(
        f"reverse worker connecting to {args.master_host}:{args.master_port}, "
        f"storage={args.storage_base}",
        flush=True,
    )
    while True:
        try:
            serve_forever(args.master_host, args.master_port, args.storage_base, args.storage_root)
        except KeyboardInterrupt:
            print("\nreverse worker stopped", flush=True)
            return 0
        except Exception as exc:  # noqa: BLE001
            print(f"reverse worker retry after error: {exc}", flush=True)
            time.sleep(args.retry_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
