#!/usr/bin/env python3
"""Validate Wohper small quality prompt files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate quality prompt set")
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data = json.loads(args.prompts.read_text(encoding="utf-8"))
    errors = []
    ids = set()
    prompts = data.get("prompts")
    if not isinstance(prompts, list) or not prompts:
        errors.append("prompts must be a non-empty list")
        prompts = []
    for index, prompt in enumerate(prompts):
        if not isinstance(prompt, dict):
            errors.append(f"prompt[{index}] must be object")
            continue
        prompt_id = str(prompt.get("id", "")).strip()
        if not prompt_id:
            errors.append(f"prompt[{index}] missing id")
        if prompt_id in ids:
            errors.append(f"duplicate id: {prompt_id}")
        ids.add(prompt_id)
        for key in ("category", "purpose", "system", "user"):
            if not str(prompt.get(key, "")).strip():
                errors.append(f"{prompt_id or index} missing {key}")
        max_new = int(prompt.get("max_new_tokens", data.get("max_new_tokens_default", 1)))
        if max_new < 1 or max_new > 4:
            errors.append(f"{prompt_id or index} unsafe max_new_tokens={max_new}")
    payload = {
        "format": "wohper-quality-prompt-validation",
        "version": 1,
        "prompt_file": str(args.prompts),
        "status": "passed" if not errors else "failed",
        "prompt_count": len(prompts),
        "ids": sorted(ids),
        "errors": errors,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"QUALITY_PROMPTS_STATUS={payload['status']}")
    print(f"PROMPT_COUNT={len(prompts)}")
    print(f"OUT={args.out}")
    return 0 if not errors else 8


if __name__ == "__main__":
    raise SystemExit(main())
