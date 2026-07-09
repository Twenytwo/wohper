#!/usr/bin/env python3
"""Smoke-test DeepSeek FP8/FP4 unpack from Wohper raw ZCBLK blocks."""

from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path
from typing import Any

import convert_safetensors as zc


def read_block(path: Path, offset: int, payload_bytes: int) -> bytes:
    with path.open("rb") as handle:
        handle.seek(offset)
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
            "payload": memoryview(buffer)[data_offset : data_offset + data_bytes],
        }
    return records


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


def decode_ue8m0(byte: int) -> float:
    if byte == 0:
        return 0.0
    # E8M0 microscaling stores an unsigned exponent-only power-of-two scale.
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


def unpack_fp8(records: dict[str, dict[str, Any]], weight_name: str, sample_values: int) -> dict[str, Any]:
    weight = records[weight_name]
    scale = records.get(weight_name.replace(".weight", ".scale"))
    scale_value = 1.0
    if scale is not None and scale["data_bytes"] > 0:
        scale_value = decode_ue8m0(int(scale["payload"][0]))
    values = [
        decode_fp8_e4m3(int(byte), scale_value)
        for byte in weight["payload"][:sample_values]
    ]
    return {
        "weight": weight_name,
        "scale_tensor_found": scale is not None,
        "first_scale": scale_value,
        "summary": finite_summary(values),
    }


def unpack_fp4(records: dict[str, dict[str, Any]], weight_name: str, sample_values: int) -> dict[str, Any]:
    weight = records[weight_name]
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DeepSeek raw block unpack smoke")
    parser.add_argument("--core", type=Path, default=Path("models/wohper/DeepSeek-V4-Flash.RAW.L0-E0E1/dense_core.bin"))
    parser.add_argument("--expert", type=Path, default=Path("models/wohper/DeepSeek-V4-Flash.RAW.L0-E0E1/experts/layer0_expert0.zcblk"))
    parser.add_argument("--out", type=Path, default=Path("state/deepseek_v4_raw_unpack_smoke_2026-07-05.json"))
    parser.add_argument("--sample-values", type=int, default=64)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    # The first layer dense block starts at 4MiB in current Wohper MODEL files.
    header = zc.ENGINE_HEADER_STRUCT.unpack(read_block(args.core, 0, zc.ENGINE_HEADER_STRUCT.size))
    manifest_offset = int(header[4])
    manifest_header = zc.MANIFEST_HEADER_STRUCT.unpack(read_block(args.core, manifest_offset, zc.MANIFEST_HEADER_STRUCT.size))
    layer_desc_offset = int(manifest_header[4])
    layer = zc.LAYER_DESC_STRUCT.unpack(
        read_block(args.core, manifest_offset + layer_desc_offset, zc.LAYER_DESC_STRUCT.size)
    )
    dense_offset = int(layer[2])
    dense_payload = int(layer[4])
    dense_records = parse_records(read_block(args.core, dense_offset, dense_payload))
    expert_records = parse_records(read_block(args.expert, 0, args.expert.stat().st_size))
    fp8 = unpack_fp8(dense_records, "layers.0.attn.wq_a.weight", args.sample_values)
    fp4 = unpack_fp4(expert_records, "layers.0.ffn.experts.0.w1.weight", args.sample_values)
    blockers = []
    if fp8["summary"]["finite_count"] != args.sample_values:
        blockers.append("fp8_unpack_non_finite")
    if fp4["summary"]["finite_count"] != args.sample_values:
        blockers.append("fp4_unpack_non_finite")
    if fp8["summary"]["nonzero_count"] == 0:
        blockers.append("fp8_unpack_all_zero")
    if fp4["summary"]["nonzero_count"] == 0:
        blockers.append("fp4_unpack_all_zero")
    payload = {
        "format": "deepseek-v4-raw-unpack-smoke",
        "version": 1,
        "status": "ready" if not blockers else "blocked",
        "blockers": blockers,
        "core": str(args.core),
        "expert": str(args.expert),
        "fp8_dense": fp8,
        "fp4_expert": fp4,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not blockers else 3


if __name__ == "__main__":
    raise SystemExit(main())
