#!/usr/bin/env python3
"""Reverse storage relay master.

Use when the worker can connect to the master, but the master cannot open TCP
connections to the worker. The worker dials this relay, and the relay exposes a
local HTTP endpoint for Wohper:

  Wohper -> http://127.0.0.1:9101/experts/layerN_expertM.zcblk
          -> reverse relay -> worker local storage server
"""

from __future__ import annotations

import argparse
import json
import socket
import struct
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


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


def read_json_frame(sock: socket.socket) -> dict[str, Any]:
    (length,) = struct.unpack("<I", read_exact(sock, 4))
    if length > MAX_HEADER_BYTES:
        raise ValueError(f"frame too large: {length}")
    return json.loads(read_exact(sock, length).decode("utf-8"))


def write_json_frame(sock: socket.socket, payload: dict[str, Any]) -> None:
    data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    sock.sendall(struct.pack("<I", len(data)))
    sock.sendall(data)


class WorkerConnection:
    def __init__(self) -> None:
        self.condition = threading.Condition()
        self.sock: socket.socket | None = None
        self.addr: tuple[str, int] | None = None
        self.busy = False

    def set(self, sock: socket.socket, addr: tuple[str, int]) -> None:
        with self.condition:
            if self.sock is not None:
                try:
                    self.sock.close()
                except OSError:
                    pass
            self.sock = sock
            self.addr = addr
            self.busy = False
            self.condition.notify_all()

    def request(self, payload: dict[str, Any]) -> tuple[dict[str, Any], socket.socket]:
        sock = self.begin_request(payload)
        try:
            response = read_json_frame(sock)
            return response, sock
        except Exception:
            self.drop(sock)
            raise

    def begin_request(self, payload: dict[str, Any]) -> socket.socket:
        with self.condition:
            if self.sock is None:
                self.condition.wait(timeout=30.0)
            if self.sock is None:
                raise TimeoutError("no reverse worker connected")
            while self.busy:
                self.condition.wait(timeout=30.0)
            if self.busy:
                raise TimeoutError("reverse worker is busy")
            sock = self.sock
            self.busy = True

        try:
            write_json_frame(sock, payload)
            return sock
        except Exception:
            self.drop(sock)
            raise

    def drop(self, sock: socket.socket) -> None:
        with self.condition:
            if self.sock is sock:
                self.sock = None
                self.addr = None
                self.busy = False
                self.condition.notify_all()
        try:
            sock.close()
        except OSError:
            pass

    def release(self, sock: socket.socket) -> None:
        with self.condition:
            if self.sock is sock:
                self.busy = False
                self.condition.notify_all()


class RelayHttpHandler(BaseHTTPRequestHandler):
    server: "RelayHttpServer"

    def do_GET(self) -> None:  # noqa: N802
        self._proxy("GET")

    def do_HEAD(self) -> None:  # noqa: N802
        self._proxy("HEAD")

    def do_PUT(self) -> None:  # noqa: N802
        self._proxy("PUT")

    def log_message(self, fmt: str, *args: object) -> None:
        print("%s - %s" % (self.address_string(), fmt % args), flush=True)

    def _proxy(self, method: str) -> None:
        last_error: Exception | None = None
        for _attempt in range(8):
            try:
                request_body_len = 0
                if method == "PUT":
                    request_body_len = int(self.headers.get("Content-Length", "0"))
                    worker = self.server.worker_connection.begin_request(
                        {"method": method, "path": self.path, "body_len": request_body_len}
                    )
                    remaining_request = request_body_len
                    while remaining_request > 0:
                        chunk = self.rfile.read(min(CHUNK_BYTES, remaining_request))
                        if not chunk:
                            raise ConnectionError("client closed while streaming PUT body")
                        worker.sendall(chunk)
                        remaining_request -= len(chunk)
                    response = read_json_frame(worker)
                else:
                    response, worker = self.server.worker_connection.request(
                        {"method": method, "path": self.path, "body_len": request_body_len}
                    )

                status = int(response.get("status", 502))
                headers = response.get("headers", {})
                body_len = int(response.get("body_len", 0))

                self.send_response(status)
                for name, value in headers.items():
                    if name.lower() in {"connection", "transfer-encoding"}:
                        continue
                    self.send_header(name, str(value))
                self.end_headers()

                if method == "HEAD":
                    return

                remaining = body_len
                while remaining > 0:
                    chunk = worker.recv(min(CHUNK_BYTES, remaining))
                    if not chunk:
                        raise ConnectionError("worker closed while streaming body")
                    remaining -= len(chunk)
                    self.wfile.write(chunk)
                return
            except Exception as exc:  # noqa: BLE001
                last_error = exc
            finally:
                if "worker" in locals():
                    self.server.worker_connection.release(worker)

        self.send_error(502, f"all reverse workers failed: {last_error}")


class RelayHttpServer(ThreadingHTTPServer):
    def __init__(self, addr: tuple[str, int], worker_connection: WorkerConnection):
        super().__init__(addr, RelayHttpHandler)
        self.worker_connection = worker_connection


def worker_accept_loop(host: str, port: int, worker_connection: WorkerConnection) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((host, port))
        listener.listen()
        print(f"reverse relay waiting for workers on {host}:{port}", flush=True)
        while True:
            sock, addr = listener.accept()
            print(f"reverse worker connected from {addr[0]}:{addr[1]}", flush=True)
            sock.settimeout(None)
            worker_connection.set(sock, addr)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-host", default="0.0.0.0")
    parser.add_argument("--worker-port", type=int, default=9200)
    parser.add_argument("--http-host", default="127.0.0.1")
    parser.add_argument("--http-port", type=int, default=9101)
    args = parser.parse_args()

    worker_connection = WorkerConnection()
    thread = threading.Thread(
        target=worker_accept_loop,
        args=(args.worker_host, args.worker_port, worker_connection),
        daemon=True,
    )
    thread.start()

    server = RelayHttpServer((args.http_host, args.http_port), worker_connection)
    print(f"reverse storage relay serving http://{args.http_host}:{args.http_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nreverse storage relay stopped", flush=True)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
