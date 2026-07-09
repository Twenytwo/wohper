#!/usr/bin/env python3
"""Smoke-test DeepSeek routed expert dispatch over sharded FP4 expert blocks."""

from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path
from typing import Any

import convert_safetensors as zc


def read_block(path: Path, payload_bytes: int) -> bytes:
    with path.open("rb") as handle:
        data = handle.read(payload_bytes)
    if len(data) != payload_bytes:
        raise ValueError(f"{path}: short read")
    return data


def read_name(buffer: bytes, names_offset: int, name_offset: int) -> str:
    cursor = names_offset + name_offset
    (size,) = struct.unpack_from("<H", buffer, cursor)
    cursor += 2
    return buffer[cursor : cursor + size].decode("utf-8")


def parse_records(buffer: bytes) -> dict[str, dict[str, Any]]:
    magic, version, tensor_count, block_quant, flags, record_offset, names_offset = zc.BLOCK_HEADER_STRUCT.unpack_from(buffer, 0)
    if magic != zc.BLOCK_MAGIC:
        raise ValueError("invalid ZCBLK magic")
    if version != 1:
        raise ValueError(f"unsupported ZCBLK version {version}")
    records: dict[str, dict[str, Any]] = {}
    for index in range(tensor_count):
        cursor = record_offset + index * zc.TENSOR_RECORD_STRUCT.size
        (
            dtype_code,
            quant_format,
            rank,
            role_code,
            name_offset,
            shape_offset,
            data_offset,
            data_bytes,
            scale,
            zero_point,
        ) = zc.TENSOR_RECORD_STRUCT.unpack_from(buffer, cursor)
        name = read_name(buffer, names_offset, name_offset)
        records[name] = {
            "dtype_code": dtype_code,
            "quant_format": quant_format,
            "rank": rank,
            "role_code": role_code,
            "data_offset": data_offset,
            "data_bytes": data_bytes,
            "scale": scale,
            "zero_point": zero_point,
            "payload": memoryview(buffer)[data_offset : data_offset + data_bytes],
        }
    return records


def decode_ue8m0(byte: int) -> float:
    if byte == 0:
        return 0.0
    return float(2.0 ** (byte - 127))


def decode_fp4_e2m1(nibble: int, scale: float = 1.0) -> float:
    nibble &= 0x0F
    sign = -1.0 if nibble & 0x08 else 1.0
    exponent = (nibble >> 1) & 0x03
    mantissa = nibble & 0x01
    if exponent == 0 and mantissa == 0:
        return 0.0
    if exponent == 0:
        value = (mantissa / 2.0) * (2.0 ** -2)
    else:
        value = (1.0 + mantissa / 2.0) * (2.0 ** (exponent - 1))
    return sign * value * scale


def finite_summary(values: list[float]) -> dict[str, Any]:
    finite = [value for value in values if math.isfinite(value)]
    return {
        "count": len(values),
        "finite_count": len(finite),
        "min": min(finite) if finite else None,
        "max": max(finite) if finite else None,
        "nonzero_count": sum(1 for value in finite if value != 0.0),
        "sample": finite[:16],
    }


def find_tensor(records: dict[str, dict[str, Any]], suffix: str) -> tuple[str, dict[str, Any]]:
    matches = [(name, record) for name, record in records.items() if name.endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one tensor ending with {suffix!r}, found {len(matches)}")
    return matches[0]


def unpack_fp4(records: dict[str, dict[str, Any]], suffix: str, sample_values: int) -> dict[str, Any]:
    weight_name, weight = find_tensor(records, suffix)
    scale = records.get(weight_name.replace(".weight", ".scale"))
    scale_value = 1.0
    if scale is not None and scale["data_bytes"] > 0:
        scale_value = decode_ue8m0(int(scale["payload"][0]))
    values: list[float] = []
    for byte in weight["payload"][: (sample_values + 1) // 2]:
        values.append(decode_fp4_e2m1(int(byte) & 0x0F, scale_value))
        if len(values) < sample_values:
            values.append(decode_fp4_e2m1(int(byte) >> 4, scale_value))
    return {
        "weight": weight_name,
        "scale_tensor_found": scale is not None,
        "first_scale": scale_value,
        "summary": finite_summary(values),
    }


def load_router_experts(path: Path | None) -> list[int]:
    if path is None or not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [int(item["expert_id"]) for item in payload.get("computed_topk", [])]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DeepSeek expert dispatch FP4 smoke")
    parser.add_argument("--core", type=Path, default=Path("models/wohper/DeepSeek-V4-Flash.RAW.L0-SPLIT-GLOBAL-TOP6/dense_core.bin"))
    parser.add_argument("--shards", type=Path)
    parser.add_argument("--router-smoke", type=Path, default=Path("state/deepseek_v4_router_top6_smoke_l0_split_global_top6_2026-07-05.json"))
    parser.add_argument("--out", type=Path, default=Path("state/deepseek_v4_expert_dispatch_smoke_l0_split_global_top6_2026-07-05.json"))
    parser.add_argument("--sample-values", type=int, default=64)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    shard_index_path = args.shards or args.core.with_suffix(".shards.json")
    shard_index = json.loads(shard_index_path.read_text(encoding="utf-8"))
    routed_experts = load_router_experts(args.router_smoke)
    available_experts = {int(item["expert_id"]) for item in shard_index.get("experts", [])}
    blockers: list[str] = []
    missing_routed = sorted(set(routed_experts) - available_experts)
    if missing_routed:
        blockers.append("missing_routed_expert_shards")

    expert_results = []
    for item in shard_index.get("experts", []):
        expert_id = int(item["expert_id"])
        expert_path = args.core.parent / item["path"]
        payload_bytes = int(item["payload_bytes"])
        records = parse_records(read_block(expert_path, payload_bytes))
        weights = {
            "w1": unpack_fp4(records, ".w1.weight", args.sample_values),
            "w2": unpack_fp4(records, ".w2.weight", args.sample_values),
            "w3": unpack_fp4(records, ".w3.weight", args.sample_values),
        }
        for label, unpacked in weights.items():
            summary = unpacked["summary"]
            if summary["finite_count"] != args.sample_values:
                blockers.append(f"expert_{expert_id}_{label}_non_finite")
            if summary["nonzero_count"] == 0:
                blockers.append(f"expert_{expert_id}_{label}_all_zero")
            if not unpacked["scale_tensor_found"]:
                blockers.append(f"expert_{expert_id}_{label}_missing_scale")
        expert_results.append(
            {
                "layer_id": int(item["layer_id"]),
                "expert_id": expert_id,
                "path": str(expert_path),
                "payload_bytes": payload_bytes,
                "routed": expert_id in routed_experts,
                "weights": weights,
            }
        )

    payload = {
        "format": "deepseek-v4-expert-dispatch-smoke",
        "version": 1,
        "status": "ready" if not blockers else "blocked",
        "blockers": sorted(set(blockers)),
        "core": str(args.core),
        "shards": str(shard_index_path),
        "router_smoke": str(args.router_smoke) if args.router_smoke else None,
        "routed_experts": routed_experts,
        "available_experts": sorted(available_experts),
        "missing_routed_experts": missing_routed,
        "expert_results": expert_results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not blockers else 3


if __name__ == "__main__":
    raise SystemExit(main())
