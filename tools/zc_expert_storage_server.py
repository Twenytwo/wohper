#!/usr/bin/env python3
"""Windows-friendly Wohper expert shard storage server.

Endpoints:
- GET /health
- GET /stats
- GET /experts/layer12_expert45.zcblk
- HEAD /experts/layer12_expert45.zcblk
- PUT /experts/layer12_expert45.zcblk

The server only serves files below <root>/experts and rejects path traversal.
It is intentionally tiny: good enough for LAN/offline cluster smoke tests and
for Wohper's HTTP cache-fill path.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import shutil
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse


CHUNK_BYTES = 1024 * 1024


class ExpertStorageHandler(BaseHTTPRequestHandler):
    server: "ExpertStorageServer"

    def do_GET(self) -> None:  # noqa: N802 - stdlib callback name
        if self.path == "/health":
            self._send_json(
                {
                    "ok": True,
                    "service": "wohper-expert-storage",
                    "root": str(self.server.root),
                    "experts_dir": str(self.server.experts_dir),
                }
            )
            return
        if self.path == "/stats":
            usage = shutil.disk_usage(self.server.root)
            self._send_json(
                {
                    "ok": True,
                    "service": "wohper-expert-storage",
                    "root": str(self.server.root),
                    "experts_dir": str(self.server.experts_dir),
                    "disk_total_bytes": usage.total,
                    "disk_used_bytes": usage.used,
                    "disk_free_bytes": usage.free,
                }
            )
            return
        self._serve_expert(send_body=True)

    def do_HEAD(self) -> None:  # noqa: N802 - stdlib callback name
        self._serve_expert(send_body=False)

    def do_PUT(self) -> None:  # noqa: N802 - stdlib callback name
        self._write_expert()

    def log_message(self, fmt: str, *args: object) -> None:
        print("%s - %s" % (self.address_string(), fmt % args), flush=True)

    def _serve_expert(self, send_body: bool) -> None:
        target = self._target_from_request()
        if target is None:
            return

        if not target.is_file():
            self.send_error(404, f"missing expert shard: {target.name}")
            return

        size = target.stat().st_size
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(target.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(size))
        self.send_header("X-Wohper-Expert", target.name)
        self.end_headers()

        if not send_body:
            return
        with target.open("rb") as handle:
            while True:
                chunk = handle.read(CHUNK_BYTES)
                if not chunk:
                    break
                self.wfile.write(chunk)

    def _write_expert(self) -> None:
        target = self._target_from_request()
        if target is None:
            return

        length_header = self.headers.get("Content-Length")
        if not length_header:
            self.send_error(411, "Content-Length required")
            return
        try:
            remaining = int(length_header)
        except ValueError:
            self.send_error(400, "invalid Content-Length")
            return
        if remaining < 0:
            self.send_error(400, "invalid Content-Length")
            return

        target.parent.mkdir(parents=True, exist_ok=True)
        temp_target = target.with_suffix(target.suffix + ".part")
        written = 0
        with temp_target.open("wb") as handle:
            while remaining > 0:
                chunk = self.rfile.read(min(CHUNK_BYTES, remaining))
                if not chunk:
                    temp_target.unlink(missing_ok=True)
                    self.send_error(400, "client closed before body completed")
                    return
                handle.write(chunk)
                written += len(chunk)
                remaining -= len(chunk)
        temp_target.replace(target)

        body = json.dumps(
            {
                "ok": True,
                "stored": target.name,
                "bytes": written,
                "path": str(target),
            },
            indent=2,
        ).encode("utf-8")
        self.send_response(201)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Wohper-Expert", target.name)
        self.end_headers()
        self.wfile.write(body)

    def _target_from_request(self) -> Path | None:
        path = urlparse(self.path).path
        prefix = "/experts/"
        if not path.startswith(prefix):
            self.send_error(404, "expected /experts/<file>.zcblk")
            return None

        name = unquote(path[len(prefix) :])
        if "/" in name or "\\" in name or not name.endswith(".zcblk"):
            self.send_error(400, "invalid expert shard name")
            return None

        target = (self.server.experts_dir / name).resolve()
        try:
            target.relative_to(self.server.experts_dir)
        except ValueError:
            self.send_error(403, "path escapes experts directory")
            return None
        return target

    def _send_json(self, payload: dict[str, object]) -> None:
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class ExpertStorageServer(ThreadingHTTPServer):
    def __init__(self, addr: tuple[str, int], root: Path):
        super().__init__(addr, ExpertStorageHandler)
        self.root = root.resolve()
        self.experts_dir = (self.root / "experts").resolve()
        self.experts_dir.mkdir(parents=True, exist_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Storage root, e.g. C:\\WohperStorage")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9100)
    args = parser.parse_args()

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "experts").mkdir(parents=True, exist_ok=True)

    server = ExpertStorageServer((args.host, args.port), root)
    print(
        f"wohper expert storage listening on http://{args.host}:{args.port} "
        f"root={server.root}",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nwohper expert storage stopped", flush=True)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    raise SystemExit(main())
