#!/usr/bin/env python3
"""Bounded DeepSeek router top6 smoke using tensor-level reads."""

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


def read_row(core: Path, tensor: dict[str, Any], row: int) -> bytes:
    row_bytes = int(tensor["row_bytes"])
    row_count = int(tensor["row_count"])
    if row < 0 or row >= row_count:
        raise ValueError(f"{tensor['name']}: row {row} outside {row_count}")
    offset = int(tensor["absolute_data_offset"]) + row * row_bytes
    with core.open("rb") as handle:
        handle.seek(offset)
        data = handle.read(row_bytes)
    if len(data) != row_bytes:
        raise ValueError(f"{tensor['name']}: short row read")
    return data


def decode_bf16(raw: bytes) -> list[float]:
    return [struct.unpack("<f", struct.pack("<I", item << 16))[0] for (item,) in struct.iter_unpack("<H", raw)]


def decode_i64(raw: bytes) -> list[int]:
    return [int(item[0]) for item in struct.iter_unpack("<q", raw)]


def dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def softplus(x: float) -> float:
    if x > 20.0:
        return x
    return math.log1p(math.exp(x))


def route_weight(logit: float) -> float:
    return math.sqrt(softplus(logit))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DeepSeek router top6 smoke")
    parser.add_argument("--index", type=Path, default=Path("models/wohper/DeepSeek-V4-Flash.RAW.L0-SPLIT-GLOBAL-E0E1/dense_core.tensor_index.json"))
    parser.add_argument("--out", type=Path, default=Path("state/deepseek_v4_router_top6_smoke_l0_split_global_e0e1_2026-07-05.json"))
    parser.add_argument("--token-id", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--route-scale", type=float, default=1.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    index = load_json(args.index)
    core = resolve_core(args.index, index["core_file"])
    tensors = {tensor["name"]: tensor for tensor in index.get("tensors", [])}
    embed = tensors["embed.weight"]
    gate = tensors["layers.0.ffn.gate.weight"]
    tid2eid = tensors["layers.0.ffn.gate.tid2eid"]

    hidden = decode_bf16(read_row(core, embed, args.token_id))
    rows = int(gate["row_count"])
    logits: list[tuple[int, float, float]] = []
    for expert_id in range(rows):
        weights = decode_bf16(read_row(core, gate, expert_id))
        logit = dot(hidden, weights)
        if not math.isfinite(logit):
            raise ValueError(f"non-finite router logit at expert {expert_id}")
        logits.append((expert_id, logit, route_weight(logit)))
    logits.sort(key=lambda item: item[2], reverse=True)
    selected = logits[: args.top_k]
    denom = sum(item[2] for item in selected)
    normalized = [
        {
            "expert_id": expert_id,
            "logit": logit,
            "weight_score": score,
            "route_weight": score / denom * args.route_scale if denom > 0 else args.route_scale / args.top_k,
        }
        for expert_id, logit, score in selected
    ]
    tid_route = decode_i64(read_row(core, tid2eid, args.token_id))
    blockers = []
    if len(normalized) != args.top_k:
        blockers.append("router_topk_wrong_size")
    if any(not math.isfinite(item["route_weight"]) for item in normalized):
        blockers.append("router_weight_non_finite")
    if not tid_route:
        blockers.append("tid2eid_empty")
    payload = {
        "format": "deepseek-v4-router-top6-smoke",
        "version": 1,
        "status": "ready" if not blockers else "blocked",
        "blockers": blockers,
        "index": str(args.index),
        "core": str(core),
        "token_id": args.token_id,
        "top_k": args.top_k,
        "router_rows": rows,
        "hidden_width": len(hidden),
        "computed_topk": normalized,
        "tid2eid_route": tid_route,
        "overlap_with_tid2eid": sorted(set(tid_route).intersection(item["expert_id"] for item in normalized)),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not blockers else 3


if __name__ == "__main__":
    raise SystemExit(main())
