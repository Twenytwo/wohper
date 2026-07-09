#!/usr/bin/env python3
"""Decode/inspect token IDs using a local Hugging Face tokenizer.json.

This is intentionally dependency-free. If transformers/tokenizers are
installed, use tools/chat_interface.py for exact detokenization. This smoke tool
is for environments where only tokenizer.json is available.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect GLM token IDs without external deps")
    parser.add_argument("--model-dir", type=Path, default=Path("models/zai-org/GLM-5.2"))
    parser.add_argument("--ids", required=True, help="Comma/space separated token ids")
    parser.add_argument("--json", action="store_true", help="emit JSON")
    return parser.parse_args()


def parse_ids(value: str) -> list[int]:
    items = value.replace(",", " ").split()
    return [int(item) for item in items]


def load_vocab(tokenizer_json: Path) -> dict[int, str]:
    payload = json.loads(tokenizer_json.read_text(encoding="utf-8"))
    model = payload.get("model") or {}
    vocab = model.get("vocab")
    if not isinstance(vocab, dict):
        raise SystemExit(f"tokenizer vocab not found in {tokenizer_json}")
    by_id = {int(token_id): token for token, token_id in vocab.items()}
    for item in payload.get("added_tokens") or []:
        if not isinstance(item, dict):
            continue
        token_id = item.get("id")
        content = item.get("content")
        if token_id is not None and content is not None:
            by_id[int(token_id)] = str(content)
    return by_id


def rough_piece_to_text(piece: str) -> str:
    # Common readable substitutions for BPE/SentencePiece-ish tokenizer pieces.
    text = piece.replace("Ġ", " ").replace("▁", " ")
    text = text.replace("Ċ", "\n")
    return text


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    tokenizer_json = args.model_dir / "tokenizer.json"
    if not tokenizer_json.exists():
        raise SystemExit(f"missing tokenizer.json: {tokenizer_json}")

    vocab = load_vocab(tokenizer_json)
    rows = []
    for token_id in parse_ids(args.ids):
        piece = vocab.get(token_id)
        rows.append(
            {
                "id": token_id,
                "piece": piece,
                "rough_text": rough_piece_to_text(piece) if piece is not None else None,
                "known": piece is not None,
            }
        )

    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
    else:
        for row in rows:
            print(f"{row['id']}\t{row['piece']!r}\t{row['rough_text']!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
