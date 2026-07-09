#!/usr/bin/env python3
"""Validate the DeepSeek-V4-Flash tokenizer/template contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DeepSeek-V4-Flash tokenizer contract check")
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("config/deepseek_v4_flash.tokenizer_contract.json"),
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models/deepseek-ai/DeepSeek-V4-Flash"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("state/deepseek_v4_flash_tokenizer_contract_2026-07-04.json"),
    )
    parser.add_argument("--contract-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    contract = load_json(args.contract)
    if contract is None:
        payload = {
            "format": "deepseek-v4-flash-tokenizer-contract",
            "version": 1,
            "status": "blocked_missing_contract",
            "contract": str(args.contract),
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 3

    policy = contract.get("policy", {})
    required_paths = list(contract.get("required_paths", []))
    missing_paths = [
        rel for rel in required_paths if args.model_dir.exists() and not (args.model_dir / rel).exists()
    ]
    status = "contract_ready"
    if not args.contract_only:
        if not args.model_dir.exists():
            status = "blocked_missing_model_dir"
        elif missing_paths:
            status = "blocked_missing_tokenizer_metadata"
        else:
            status = "ready_for_tokenizer_adapter"

    payload: dict[str, Any] = {
        "format": "deepseek-v4-flash-tokenizer-contract",
        "version": 1,
        "status": status,
        "contract": str(args.contract),
        "model_dir": str(args.model_dir),
        "contract_only": bool(args.contract_only),
        "model_family": contract.get("model_family"),
        "hf_repo": contract.get("hf_repo"),
        "policy": policy,
        "required_paths": required_paths,
        "missing_paths": missing_paths,
        "smoke_prompt_count": len(contract.get("smoke_prompts", [])),
        "blocked_until": contract.get("blocked_until", []),
        "adapter_decision": (
            "Do not use generic ChatML/Jinja fallback. Implement a "
            "DeepSeek-V4-Flash-specific encoding and prompt renderer."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if status in {"contract_ready", "ready_for_tokenizer_adapter"} else 3


if __name__ == "__main__":
    raise SystemExit(main())
