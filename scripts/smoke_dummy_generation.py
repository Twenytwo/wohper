#!/usr/bin/env python3
"""Send a tiny generation request to a running Wohper Unix socket server."""

from __future__ import annotations

import argparse
import json
import socket


def parse_int_list(value: str) -> list[int]:
    if not value.strip():
        return []
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", default="/tmp/wohper-infer.sock")
    parser.add_argument("--request-id", default="dummy-generation-smoke")
    parser.add_argument("--tokens", default="1,2,3")
    parser.add_argument("--experts", default="0,1")
    parser.add_argument("--max-new-tokens", type=int, default=2)
    parser.add_argument("--stop-token-id", type=int, default=154820)
    args = parser.parse_args()

    payload = {
        "request_id": args.request_id,
        "objective": "dummy ZCBLK01 generation smoke",
        "token_ids": parse_int_list(args.tokens),
        "max_new_tokens": args.max_new_tokens,
        "stop_token_ids": [args.stop_token_id],
        "route_hint": {"expert_ids": parse_int_list(args.experts)},
    }

    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.connect(args.socket)
    client.sendall((json.dumps(payload) + "\n").encode("utf-8"))

    with client.makefile("r", encoding="utf-8") as stream:
        for line in stream:
            print(line, end="")

    client.close()


if __name__ == "__main__":
    main()
