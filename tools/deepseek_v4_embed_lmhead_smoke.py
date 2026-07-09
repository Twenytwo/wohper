#!/usr/bin/env python3
"""Bounded DeepSeek embed + LM-head smoke on Wohper raw blocks."""

from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path
from typing import Any

import convert_safetensors as zc


def read_at(path: Path, offset: int, size: int) -> bytes:
    with path.open("rb") as handle:
        handle.seek(offset)
        data = handle.read(size)
    if len(data) != size:
        raise ValueError(f"{path}: short read at {offset} size {size}")
    return data


def read_name(path: Path, block_offset: int, payload_bytes: int, names_offset: int, name_offset: int) -> str:
    cursor = names_offset + name_offset
    if cursor + 2 > payload_bytes:
        raise ValueError("name offset outside block")
    (size,) = struct.unpack("<H", read_at(path, block_offset + cursor, 2))
    cursor += 2
    if cursor + size > payload_bytes:
        raise ValueError("truncated name blob")
    return read_at(path, block_offset + cursor, size).decode("utf-8")


def read_shape(path: Path, block_offset: int, payload_bytes: int, shape_offset: int) -> list[int]:
    if shape_offset + 4 > payload_bytes:
        raise ValueError("shape offset outside block")
    (rank,) = struct.unpack("<I", read_at(path, block_offset + shape_offset, 4))
    cursor = shape_offset + 4
    if cursor + rank * 8 > payload_bytes:
        raise ValueError("truncated shape blob")
    raw = read_at(path, block_offset + cursor, rank * 8)
    return [int(value) for value in struct.unpack("<" + "Q" * rank, raw)]


def dense_block_location(core: Path) -> tuple[int, int]:
    header = zc.ENGINE_HEADER_STRUCT.unpack(read_at(core, 0, zc.ENGINE_HEADER_STRUCT.size))
    manifest_offset = int(header[4])
    manifest_header = zc.MANIFEST_HEADER_STRUCT.unpack(
        read_at(core, manifest_offset, zc.MANIFEST_HEADER_STRUCT.size)
    )
    layer_desc_offset = int(manifest_header[4])
    layer = zc.LAYER_DESC_STRUCT.unpack(
        read_at(core, manifest_offset + layer_desc_offset, zc.LAYER_DESC_STRUCT.size)
    )
    return int(layer[2]), int(layer[4])


def find_tensor(core: Path, block_offset: int, payload_bytes: int, wanted: str) -> dict[str, Any]:
    header = read_at(core, block_offset, zc.BLOCK_HEADER_STRUCT.size)
    magic, version, tensor_count, block_quant, flags, record_offset, names_offset = zc.BLOCK_HEADER_STRUCT.unpack(header)
    if magic != zc.BLOCK_MAGIC:
        raise ValueError("invalid dense block magic")
    if version != 1:
        raise ValueError(f"unsupported dense block version {version}")
    records = read_at(core, block_offset + record_offset, tensor_count * zc.TENSOR_RECORD_STRUCT.size)
    for index in range(tensor_count):
        cursor = index * zc.TENSOR_RECORD_STRUCT.size
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
        ) = zc.TENSOR_RECORD_STRUCT.unpack_from(records, cursor)
        name = read_name(core, block_offset, payload_bytes, names_offset, name_offset)
        if name != wanted:
            continue
        shape = read_shape(core, block_offset, payload_bytes, shape_offset)
        if data_offset + data_bytes > payload_bytes:
            raise ValueError(f"{wanted}: tensor payload outside block")
        return {
            "name": name,
            "dtype_code": dtype_code,
            "quant_format": quant_format,
            "rank": rank,
            "role_code": role_code,
            "shape": shape,
            "data_offset": data_offset,
            "data_bytes": data_bytes,
        }
    raise KeyError(wanted)


def bf16_to_f32(raw: bytes) -> list[float]:
    if len(raw) % 2:
        raise ValueError("BF16 data must be 2-byte aligned")
    values = []
    for (item,) in struct.iter_unpack("<H", raw):
        values.append(struct.unpack("<f", struct.pack("<I", item << 16))[0])
    return values


def read_bf16_row(core: Path, block_offset: int, tensor: dict[str, Any], row: int, width: int) -> list[float]:
    shape = tensor["shape"]
    if len(shape) != 2:
        raise ValueError(f"{tensor['name']}: expected rank-2 shape")
    rows, cols = shape
    if row < 0 or row >= rows:
        raise ValueError(f"{tensor['name']}: row {row} outside {rows}")
    width = min(width, cols)
    byte_offset = block_offset + int(tensor["data_offset"]) + row * cols * 2
    return bf16_to_f32(read_at(core, byte_offset, width * 2))


def vector_summary(values: list[float]) -> dict[str, Any]:
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


def bounded_head_topk(
    core: Path,
    block_offset: int,
    head: dict[str, Any],
    hidden: list[float],
    scan_vocab: int,
    top_k: int,
) -> list[dict[str, float | int]]:
    rows = int(head["shape"][0])
    limit = min(scan_vocab, rows)
    winners: list[tuple[float, int]] = []
    for token_id in range(limit):
        row = read_bf16_row(core, block_offset, head, token_id, len(hidden))
        score = dot(hidden, row)
        if not math.isfinite(score):
            raise ValueError(f"non-finite head score at token {token_id}")
        winners.append((score, token_id))
        winners.sort(reverse=True)
        if len(winners) > top_k:
            winners.pop()
    return [{"token_id": token_id, "score": score} for score, token_id in winners]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DeepSeek embed/head bounded smoke")
    parser.add_argument("--core", type=Path, default=Path("models/wohper/DeepSeek-V4-Flash.RAW.L0-GLOBAL-E0E1/dense_core.bin"))
    parser.add_argument("--out", type=Path, default=Path("state/deepseek_v4_embed_lmhead_smoke_l0_global_e0e1_2026-07-05.json"))
    parser.add_argument("--token-id", type=int, default=42)
    parser.add_argument("--head-token-id", type=int, default=42)
    parser.add_argument("--width", type=int, default=4096)
    parser.add_argument("--scan-vocab", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    block_offset, payload_bytes = dense_block_location(args.core)
    embed = find_tensor(args.core, block_offset, payload_bytes, "embed.weight")
    head = find_tensor(args.core, block_offset, payload_bytes, "head.weight")
    blockers = []
    for tensor in (embed, head):
        if tensor["dtype_code"] != zc.DTYPE_CODES["bfloat16"]:
            blockers.append(f"{tensor['name']}_not_bf16")
        if tensor["quant_format"] != 2404:
            blockers.append(f"{tensor['name']}_unexpected_quant")
    embed_row = read_bf16_row(args.core, block_offset, embed, args.token_id, args.width)
    head_row = read_bf16_row(args.core, block_offset, head, args.head_token_id, args.width)
    embed_summary = vector_summary(embed_row)
    head_summary = vector_summary(head_row)
    dot_value = dot(embed_row, head_row)
    topk = bounded_head_topk(args.core, block_offset, head, embed_row, args.scan_vocab, args.top_k)
    if embed_summary["finite_count"] != len(embed_row):
        blockers.append("embed_non_finite")
    if head_summary["finite_count"] != len(head_row):
        blockers.append("head_non_finite")
    if embed_summary["nonzero_count"] == 0:
        blockers.append("embed_all_zero")
    if head_summary["nonzero_count"] == 0:
        blockers.append("head_all_zero")
    if not math.isfinite(dot_value):
        blockers.append("dot_non_finite")
    if not topk:
        blockers.append("empty_topk")
    payload = {
        "format": "deepseek-v4-embed-lmhead-smoke",
        "version": 1,
        "status": "ready" if not blockers else "blocked",
        "blockers": blockers,
        "core": str(args.core),
        "token_id": args.token_id,
        "head_token_id": args.head_token_id,
        "width": len(embed_row),
        "scan_vocab": args.scan_vocab,
        "top_k": args.top_k,
        "embed": {key: embed[key] for key in ("name", "dtype_code", "quant_format", "shape", "data_bytes")},
        "head": {key: head[key] for key in ("name", "dtype_code", "quant_format", "shape", "data_bytes")},
        "embed_summary": embed_summary,
        "head_summary": head_summary,
        "embed_head_dot": dot_value,
        "bounded_lmhead_topk": topk,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not blockers else 3


if __name__ == "__main__":
    raise SystemExit(main())
