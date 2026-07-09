#!/usr/bin/env python3
"""Smoke row-wise DeepSeek embed/head reads through dense_core.tensor_index.json."""

from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_core(index_path: Path, core_file: str) -> Path:
    core = Path(core_file)
    if core.is_absolute() or core.exists():
        return core
    return index_path.parent / core


def bf16_to_f32(raw: bytes) -> list[float]:
    values = []
    for (item,) in struct.iter_unpack("<H", raw):
        values.append(struct.unpack("<f", struct.pack("<I", item << 16))[0])
    return values


def read_row(core: Path, tensor: dict[str, Any], row: int) -> list[float]:
    row_count = int(tensor["row_count"])
    row_bytes = int(tensor["row_bytes"])
    if row < 0 or row >= row_count:
        raise ValueError(f"{tensor['name']}: row outside tensor")
    offset = int(tensor["absolute_data_offset"]) + row * row_bytes
    with core.open("rb") as handle:
        handle.seek(offset)
        raw = handle.read(row_bytes)
    if len(raw) != row_bytes:
        raise ValueError(f"{tensor['name']}: short row read")
    return bf16_to_f32(raw)


def summary(values: list[float]) -> dict[str, Any]:
    finite = [value for value in values if math.isfinite(value)]
    return {
        "count": len(values),
        "finite_count": len(finite),
        "nonzero_count": sum(1 for value in finite if value != 0.0),
        "min": min(finite) if finite else None,
        "max": max(finite) if finite else None,
        "l2": math.sqrt(sum(value * value for value in finite)) if finite else None,
        "sample": finite[:16],
    }


def dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def topk_window(core: Path, head: dict[str, Any], hidden: list[float], scan_rows: int, top_k: int) -> list[dict[str, Any]]:
    row_count = min(scan_rows, int(head["row_count"]))
    winners: list[tuple[float, int]] = []
    for token_id in range(row_count):
        score = dot(hidden, read_row(core, head, token_id))
        if not math.isfinite(score):
            raise ValueError(f"non-finite score at row {token_id}")
        winners.append((score, token_id))
        winners.sort(reverse=True)
        if len(winners) > top_k:
            winners.pop()
    return [{"token_id": token_id, "score": score} for score, token_id in winners]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DeepSeek row tensor index smoke")
    parser.add_argument("--index", type=Path, default=Path("models/wohper/DeepSeek-V4-Flash.RAW.L0-GLOBAL-E0E1/dense_core.tensor_index.json"))
    parser.add_argument("--out", type=Path, default=Path("state/deepseek_v4_row_index_smoke_l0_global_e0e1_2026-07-05.json"))
    parser.add_argument("--token-id", type=int, default=42)
    parser.add_argument("--scan-vocab", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    index = load_json(args.index)
    tensors = {tensor["name"]: tensor for tensor in index.get("tensors", [])}
    core = resolve_core(args.index, index["core_file"])
    embed = tensors["embed.weight"]
    head = tensors["head.weight"]
    hidden = read_row(core, embed, args.token_id)
    topk = topk_window(core, head, hidden, args.scan_vocab, args.top_k)
    blockers = []
    hidden_summary = summary(hidden)
    if hidden_summary["finite_count"] != len(hidden):
        blockers.append("embed_non_finite")
    if hidden_summary["nonzero_count"] == 0:
        blockers.append("embed_all_zero")
    if not topk:
        blockers.append("empty_topk")
    payload = {
        "format": "deepseek-v4-row-index-smoke",
        "version": 1,
        "status": "ready" if not blockers else "blocked",
        "blockers": blockers,
        "index": str(args.index),
        "core": str(core),
        "token_id": args.token_id,
        "scan_vocab": args.scan_vocab,
        "top_k": args.top_k,
        "embed_row_bytes": embed["row_bytes"],
        "head_row_bytes": head["row_bytes"],
        "embed_summary": hidden_summary,
        "bounded_lmhead_topk": topk,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not blockers else 3


if __name__ == "__main__":
    raise SystemExit(main())
