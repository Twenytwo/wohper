#!/usr/bin/env python3
"""Send a minimal PromptEnvelope to the Wohper Unix socket server."""

from __future__ import annotations

import argparse
import json
import socket
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke client for zc_infer_server Unix socket")
    parser.add_argument("--socket", type=Path, default=Path("/tmp/wohper-infer.sock"))
    parser.add_argument("--request-id", default="zc-smoke")
    parser.add_argument("--token-id", type=int, default=42)
    parser.add_argument(
        "--token-ids",
        default="",
        help="Optional comma/space separated prompt token ids; overrides --token-id",
    )
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--experts", default="0,1", help="Comma-separated expert route hint")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    experts = [int(value) for value in args.experts.split(",") if value.strip()]
    token_ids = (
        [int(value) for value in args.token_ids.replace(",", " ").split() if value.strip()]
        if args.token_ids.strip()
        else [args.token_id]
    )
    envelope = {
        "request_id": args.request_id,
        "objective": "wohper socket smoke",
        "token_ids": token_ids,
        "max_new_tokens": args.max_new_tokens,
        "route_hint": {"expert_ids": experts},
    }
    optional = {
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "seed": args.seed,
    }
    envelope.update({key: value for key, value in optional.items() if value is not None})

    payload = json.dumps(envelope, separators=(",", ":")).encode("utf-8") + b"\n"
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.connect(str(args.socket))
        client.sendall(payload)
        client.shutdown(socket.SHUT_WR)
        while True:
            chunk = client.recv(65536)
            if not chunk:
                break
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
