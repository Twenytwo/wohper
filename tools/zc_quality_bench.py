#!/usr/bin/env python3
"""Small repeatable quality/logit smoke benchmark for Wohper socket models."""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from pathlib import Path
from typing import Any

from tokenizer_chat_smoke import TokenizerSmoke, build_chat_prompt_from_messages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--prompts", type=Path, default=Path("config/quality_prompts.small.json"))
    parser.add_argument("--model-dir", type=Path, default=Path("models/zai-org/GLM-5.2"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--timeout-sec", type=float, default=600.0)
    parser.add_argument("--limit", type=int, default=8)
    return parser.parse_args()


def socket_events(socket_path: str, envelope: dict[str, Any], timeout: float) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    payload = json.dumps(envelope, separators=(",", ":")).encode("utf-8") + b"\n"
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(timeout)
        client.connect(socket_path)
        client.sendall(payload)
        client.shutdown(socket.SHUT_WR)
        with client.makefile("r", encoding="utf-8") as stream:
            for line in stream:
                line = line.strip()
                if line:
                    events.append(json.loads(line))
    return events


def load_prompts(path: Path, limit: int) -> tuple[list[dict[str, Any]], int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    prompts = data.get("prompts", [])
    if not isinstance(prompts, list):
        raise SystemExit("--prompts file must contain a prompts list")
    if limit <= 0 or limit > 32:
        raise SystemExit("--limit must be 1..32")
    default_max_new = int(data.get("max_new_tokens_default", 1) or 1)
    return prompts[:limit], default_max_new


def result_from_events(
    *,
    tokenizer: TokenizerSmoke,
    prompt_id: str,
    token_ids: list[int],
    events: list[dict[str, Any]],
    elapsed_ms: float,
) -> dict[str, Any]:
    generated = [int(event["token_id"]) for event in events if event.get("event") == "Token"]
    logits = [event.get("logit") for event in events if event.get("event") == "Token"]
    sources = [event.get("source") for event in events if event.get("event") == "Token"]
    finished = next((event for event in events if event.get("event") == "Finished"), None)
    try:
        decoded = tokenizer.decode(generated) if generated else ""
    except Exception as exc:
        decoded = f"<decode-error:{exc}>"
    return {
        "id": prompt_id,
        "input_tokens": len(token_ids),
        "generated_ids": generated,
        "generated_text": decoded,
        "token_logits": logits,
        "token_sources": sources,
        "finished": finished is not None,
        "stop_reason": finished.get("stop_reason") if finished else None,
        "elapsed_ms": round(elapsed_ms, 3),
        "events": events,
    }


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    prompts, default_max_new = load_prompts(args.prompts, args.limit)
    max_new_tokens = args.max_new_tokens if args.max_new_tokens is not None else default_max_new
    if max_new_tokens <= 0 or max_new_tokens > 8:
        raise SystemExit("--max-new-tokens must be 1..8 for this smoke benchmark")

    tokenizer = TokenizerSmoke.load(args.model_dir)
    results: list[dict[str, Any]] = []
    started = time.time()
    for index, prompt in enumerate(prompts):
        prompt_id = str(prompt.get("id") or f"prompt_{index}")
        messages = prompt.get("messages")
        if not isinstance(messages, list):
            raise SystemExit(f"prompt {prompt_id} missing messages list")
        prompt_text = build_chat_prompt_from_messages(messages, "")
        token_ids = tokenizer.encode(prompt_text)
        envelope = {
            "request_id": f"{args.model_name}-{prompt_id}",
            "objective": prompt_id,
            "token_ids": token_ids,
            "max_new_tokens": max_new_tokens,
            "route_hint": {"expert_ids": []},
            "stop_token_ids": [154820, 154827, 154829],
        }
        before = time.time()
        events = socket_events(args.socket, envelope, args.timeout_sec)
        elapsed_ms = (time.time() - before) * 1000.0
        results.append(
            result_from_events(
                tokenizer=tokenizer,
                prompt_id=prompt_id,
                token_ids=token_ids,
                events=events,
                elapsed_ms=elapsed_ms,
            )
        )
        print(
            f"{prompt_id}: ids={results[-1]['generated_ids']} "
            f"logits={results[-1]['token_logits']} text={results[-1]['generated_text']!r}",
            flush=True,
        )

    payload = {
        "format": "wohper-quality-bench-result",
        "version": 1,
        "model_name": args.model_name,
        "prompt_file": str(args.prompts),
        "max_new_tokens": max_new_tokens,
        "total_elapsed_ms": round((time.time() - started) * 1000.0, 3),
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
