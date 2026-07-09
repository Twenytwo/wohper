#!/usr/bin/env python3
"""Dependency-light DeepSeek-V4 prompt renderer.

This intentionally renders text only. Tokenization still requires the official
encoding files or a trusted adapter.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


BOS_TOKEN = "<｜begin▁of▁sentence｜>"
EOS_TOKEN = "<｜end▁of▁sentence｜>"
USER_TOKEN = "<｜User｜>"
ASSISTANT_TOKEN = "<｜Assistant｜>"
THINKING_START = "<think>"
THINKING_END = "</think>"


def render_messages(
    messages: list[dict[str, Any]],
    *,
    thinking_mode: str = "chat",
    add_bos: bool = True,
    drop_thinking: bool = True,
) -> str:
    if thinking_mode not in {"chat", "thinking"}:
        raise ValueError("thinking_mode must be chat or thinking")
    prompt = BOS_TOKEN if add_bos else ""
    last_user_idx = max(
        (idx for idx, msg in enumerate(messages) if msg.get("role") in {"user", "developer"}),
        default=-1,
    )
    for idx, msg in enumerate(messages):
        role = msg.get("role")
        content = msg.get("content") or ""
        if role == "system":
            prompt += content
        elif role in {"user", "developer"}:
            prompt += USER_TOKEN + content
        elif role == "assistant":
            reasoning = msg.get("reasoning_content") or ""
            thinking = ""
            if thinking_mode == "thinking" and (not drop_thinking or idx > last_user_idx):
                thinking = reasoning + THINKING_END
            prompt += thinking + content
            if not msg.get("wo_eos", False):
                prompt += EOS_TOKEN
        else:
            raise ValueError(f"unsupported DeepSeek-V4 prompt role: {role}")

        next_is_assistant = idx + 1 < len(messages) and messages[idx + 1].get("role") == "assistant"
        if role in {"user", "developer"} and not next_is_assistant:
            prompt += ASSISTANT_TOKEN
            prompt += THINKING_START if thinking_mode == "thinking" and idx >= last_user_idx else THINKING_END
    return prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a DeepSeek-V4 prompt without generic ChatML")
    parser.add_argument("--messages", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--thinking-mode", choices=["chat", "thinking"], default="chat")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = json.loads(args.messages.read_text(encoding="utf-8"))
    messages = payload["messages"] if isinstance(payload, dict) else payload
    rendered = render_messages(messages, thinking_mode=args.thinking_mode)
    report = {
        "format": "deepseek-v4-prompt-render",
        "version": 1,
        "status": "ready",
        "thinking_mode": args.thinking_mode,
        "char_count": len(rendered),
        "prompt": rendered,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "prompt"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
