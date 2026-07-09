#!/usr/bin/env python3
"""Smoke tensor-level reads for DeepSeek compute tensors.

This validates that router, attention FP8, shared expert FP8 and scale tensors
can be read directly from dense_core.tensor_index.json without loading the full
dense compute block.
"""

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


def read_tensor_prefix(core: Path, tensor: dict[str, Any], size: int) -> bytes:
    size = min(size, int(tensor["data_bytes"]))
    with core.open("rb") as handle:
        handle.seek(int(tensor["absolute_data_offset"]))
        data = handle.read(size)
    if len(data) != size:
        raise ValueError(f"{tensor['name']}: short tensor prefix read")
    return data


def read_tensor_row(core: Path, tensor: dict[str, Any], row: int) -> bytes:
    row_count = int(tensor["row_count"])
    row_bytes = int(tensor["row_bytes"])
    if row < 0 or row >= row_count:
        raise ValueError(f"{tensor['name']}: row outside tensor")
    offset = int(tensor["absolute_data_offset"]) + row * row_bytes
    with core.open("rb") as handle:
        handle.seek(offset)
        data = handle.read(row_bytes)
    if len(data) != row_bytes:
        raise ValueError(f"{tensor['name']}: short tensor row read")
    return data


def decode_bf16(raw: bytes) -> list[float]:
    return [struct.unpack("<f", struct.pack("<I", item << 16))[0] for (item,) in struct.iter_unpack("<H", raw)]


def decode_i64(raw: bytes) -> list[int]:
    return [int(item[0]) for item in struct.iter_unpack("<q", raw)]


def decode_ue8m0(byte: int) -> float:
    if byte == 0:
        return 0.0
    return float(2.0 ** (byte - 127))


def decode_fp8_e4m3(byte: int, scale: float = 1.0) -> float:
    sign = -1.0 if byte & 0x80 else 1.0
    exponent = (byte >> 3) & 0x0F
    mantissa = byte & 0x07
    if exponent == 0 and mantissa == 0:
        return 0.0
    if exponent == 0:
        value = (mantissa / 8.0) * (2.0 ** -6)
    else:
        value = (1.0 + mantissa / 8.0) * (2.0 ** (exponent - 7))
    return sign * value * scale


def finite_summary(values: list[float]) -> dict[str, Any]:
    finite = [value for value in values if math.isfinite(value)]
    return {
        "count": len(values),
        "finite_count": len(finite),
        "nonzero_count": sum(1 for value in finite if value != 0.0),
        "min": min(finite) if finite else None,
        "max": max(finite) if finite else None,
        "sample": finite[:16],
    }


def bf16_probe(core: Path, tensor: dict[str, Any], sample_bytes: int) -> dict[str, Any]:
    raw = read_tensor_prefix(core, tensor, sample_bytes)
    values = decode_bf16(raw[: len(raw) - (len(raw) % 2)])
    return {"tensor": tensor["name"], "summary": finite_summary(values)}


def i64_probe(core: Path, tensor: dict[str, Any], row: int) -> dict[str, Any]:
    raw = read_tensor_row(core, tensor, row)
    values = decode_i64(raw[: len(raw) - (len(raw) % 8)])
    return {
        "tensor": tensor["name"],
        "row": row,
        "count": len(values),
        "sample": values[:16],
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def fp8_with_scale_probe(core: Path, weight: dict[str, Any], scale: dict[str, Any], sample_values: int) -> dict[str, Any]:
    weight_raw = read_tensor_prefix(core, weight, sample_values)
    scale_raw = read_tensor_prefix(core, scale, min(sample_values, int(scale["data_bytes"])))
    scale_value = decode_ue8m0(scale_raw[0]) if scale_raw else 1.0
    values = [decode_fp8_e4m3(byte, scale_value) for byte in weight_raw[:sample_values]]
    return {
        "tensor": weight["name"],
        "scale_tensor": scale["name"],
        "first_scale": scale_value,
        "summary": finite_summary(values),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DeepSeek tensor-level compute smoke")
    parser.add_argument("--index", type=Path, default=Path("models/wohper/DeepSeek-V4-Flash.RAW.L0-SPLIT-GLOBAL-E0E1/dense_core.tensor_index.json"))
    parser.add_argument("--out", type=Path, default=Path("state/deepseek_v4_tensor_level_compute_smoke_l0_split_global_e0e1_2026-07-05.json"))
    parser.add_argument("--sample-values", type=int, default=64)
    parser.add_argument("--router-token-row", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    index = load_json(args.index)
    core = resolve_core(args.index, index["core_file"])
    tensors = {tensor["name"]: tensor for tensor in index.get("tensors", [])}
    blockers: list[str] = []
    warnings: list[str] = []

    def require(name: str) -> dict[str, Any]:
        tensor = tensors.get(name)
        if tensor is None:
            blockers.append(f"missing_{name}")
            return {}
        return tensor

    probes: dict[str, Any] = {}
    router = require("layers.0.ffn.gate.weight")
    router_map = require("layers.0.ffn.gate.tid2eid")
    if router:
        probes["router_weight"] = bf16_probe(core, router, args.sample_values * 2)
    if router_map:
        probes["router_tid2eid"] = i64_probe(core, router_map, args.router_token_row)

    for stem in ("wq_a", "wkv", "wo_a"):
        weight = require(f"layers.0.attn.{stem}.weight")
        scale = require(f"layers.0.attn.{stem}.scale")
        if weight and scale:
            probes[f"attention_{stem}"] = fp8_with_scale_probe(core, weight, scale, args.sample_values)

    for stem in ("w1", "w2", "w3"):
        weight = require(f"layers.0.ffn.shared_experts.{stem}.weight")
        scale = require(f"layers.0.ffn.shared_experts.{stem}.scale")
        if weight and scale:
            probes[f"shared_expert_{stem}"] = fp8_with_scale_probe(core, weight, scale, args.sample_values)

    for term in ("compressor", "indexer"):
        matching = [name for name in tensors if term in name and tensors[name].get("block_type") == 0]
        if not matching:
            warnings.append(f"{term}_not_present_in_current_l0_compute_slice")
        else:
            probes[term] = matching[:16]

    for key, probe in probes.items():
        if isinstance(probe, dict) and "summary" in probe:
            summary = probe["summary"]
            if summary["finite_count"] != summary["count"]:
                blockers.append(f"{key}_non_finite")
            if summary["nonzero_count"] == 0:
                blockers.append(f"{key}_all_zero")

    payload = {
        "format": "deepseek-v4-tensor-level-compute-smoke",
        "version": 1,
        "status": "ready" if not blockers else "blocked",
        "blockers": blockers,
        "warnings": warnings,
        "index": str(args.index),
        "core": str(core),
        "sample_values": args.sample_values,
        "probes": probes,
        "bytes_read_upper_bound": args.sample_values * 2 + args.sample_values * 6 + 48,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not blockers else 3


if __name__ == "__main__":
    raise SystemExit(main())
