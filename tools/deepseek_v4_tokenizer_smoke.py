#!/usr/bin/env python3
"""Dependency-free DeepSeek-V4 tokenizer/template smoke."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from render_deepseek_v4_prompt import render_messages
from tokenizer_chat_smoke import TokenizerSmoke, parse_ids


DEFAULT_MODEL_DIR = Path("models/deepseek-ai/DeepSeek-V4-Flash")
DEEPSEEK_SPECIALS = [
    "<｜begin▁of▁sentence｜>",
    "<｜end▁of▁sentence｜>",
    "<｜User｜>",
    "<｜Assistant｜>",
    "<|EOT|>",
    "<think>",
    "</think>",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DeepSeek-V4 tokenizer/template smoke")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--messages", type=Path, default=Path("config/deepseek_v4_flash.prompt_smoke.json"))
    parser.add_argument("--thinking-mode", choices=["chat", "thinking"], default="chat")
    parser.add_argument("--generated-ids", default="")
    parser.add_argument("--out", type=Path, default=Path("state/deepseek_v4_tokenizer_smoke_2026-07-05.json"))
    return parser.parse_args()


def load_messages(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    messages = payload.get("messages") if isinstance(payload, dict) else payload
    if not isinstance(messages, list):
        raise ValueError("messages payload must be a list or an object with messages")
    return messages


def robust_streaming_decode_report(tokenizer: TokenizerSmoke, ids: list[int]) -> dict[str, object]:
    def decode_pending(chars: list[str]) -> str:
        data = bytes(tokenizer.byte_decoder[char] for char in chars)
        return data.decode("utf-8", errors="strict")

    text = ""
    pending_chars: list[str] = []
    deltas: list[dict[str, object]] = []
    blockers: list[str] = []
    for token_id in ids:
        token = tokenizer.by_id.get(token_id)
        if token is None:
            blockers.append(f"unknown_token_{token_id}")
            deltas.append({"id": token_id, "delta": "", "text": text, "pending_bytes": len(pending_chars)})
            continue
        delta = ""
        if token in tokenizer.specials:
            if pending_chars:
                try:
                    delta += decode_pending(pending_chars)
                    pending_chars.clear()
                except UnicodeDecodeError:
                    blockers.append("invalid_pending_utf8_before_special")
            delta += token
        else:
            pending_chars.extend(token)
            try:
                delta += decode_pending(pending_chars)
                pending_chars.clear()
            except UnicodeDecodeError:
                delta = ""
        text += delta
        deltas.append({"id": token_id, "delta": delta, "text": text, "pending_bytes": len(pending_chars)})
    if pending_chars:
        try:
            delta = decode_pending(pending_chars)
            text += delta
            deltas.append({"id": None, "delta": delta, "text": text, "pending_bytes": 0})
            pending_chars.clear()
        except UnicodeDecodeError:
            blockers.append("trailing_incomplete_utf8")
    return {
        "token_ids": ids,
        "text": text,
        "deltas": deltas,
        "stable": "".join(str(item["delta"]) for item in deltas) == text,
        "unicode_ok": "\ufffd" not in text,
        "blockers": blockers,
    }


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    tokenizer = TokenizerSmoke.load(args.model_dir)
    messages = load_messages(args.messages)
    prompt = render_messages(messages, thinking_mode=args.thinking_mode)
    token_ids = tokenizer.encode(prompt)
    decoded = tokenizer.decode(token_ids)
    generated_ids = parse_ids(args.generated_ids)
    special_ids = {token: tokenizer.specials.get(token) for token in DEEPSEEK_SPECIALS}
    blockers = []
    if decoded != prompt:
        blockers.append("roundtrip_mismatch")
    for token in DEEPSEEK_SPECIALS:
        if special_ids.get(token) is None:
            blockers.append(f"missing_special_{token}")
    streaming = robust_streaming_decode_report(tokenizer, generated_ids) if generated_ids else None
    if streaming and not streaming["unicode_ok"]:
        blockers.append("streaming_unicode_replacement")
    if streaming and not streaming["stable"]:
        blockers.append("streaming_unstable_delta")
    if streaming:
        blockers.extend(str(item) for item in streaming.get("blockers", []))
    payload = {
        "format": "deepseek-v4-tokenizer-smoke",
        "version": 1,
        "status": "ready" if not blockers else "blocked",
        "blockers": blockers,
        "model_dir": str(args.model_dir),
        "messages": messages,
        "thinking_mode": args.thinking_mode,
        "prompt_text": prompt,
        "token_ids": token_ids,
        "token_count": len(token_ids),
        "roundtrip_ok": decoded == prompt,
        "special_ids": special_ids,
        "stop_token_ids": [
            special_ids["<｜end▁of▁sentence｜>"],
            special_ids["<|EOT|>"],
        ],
        "streaming": streaming,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items() if k not in {"prompt_text", "token_ids"}}, ensure_ascii=False, indent=2))
    return 0 if not blockers else 3


if __name__ == "__main__":
    raise SystemExit(main())
