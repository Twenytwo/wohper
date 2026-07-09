#!/usr/bin/env python3
"""Validate Wohper DeepSeek-V4 raw conversion artifacts."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path
from typing import Any

import convert_safetensors as zc


ENGINE_HEADER_STRUCT = zc.ENGINE_HEADER_STRUCT
MANIFEST_HEADER_STRUCT = zc.MANIFEST_HEADER_STRUCT
LAYER_DESC_STRUCT = zc.LAYER_DESC_STRUCT
EXPERT_DESC_STRUCT = zc.EXPERT_DESC_STRUCT
BLOCK_HEADER_STRUCT = zc.BLOCK_HEADER_STRUCT
TENSOR_RECORD_STRUCT = zc.TENSOR_RECORD_STRUCT


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_at(path: Path, offset: int, size: int) -> bytes:
    with path.open("rb") as handle:
        handle.seek(offset)
        data = handle.read(size)
    if len(data) != size:
        raise ValueError(f"{path}: short read at {offset} size {size}")
    return data


def checksum64_range(path: Path, offset: int, size: int) -> int:
    import hashlib

    remaining = size
    h = hashlib.blake2b(digest_size=8)
    with path.open("rb") as handle:
        handle.seek(offset)
        while remaining > 0:
            chunk = handle.read(min(8 * 1024 * 1024, remaining))
            if not chunk:
                raise ValueError(f"{path}: short checksum read at {offset} size {size}")
            h.update(chunk)
            remaining -= len(chunk)
    return int.from_bytes(h.digest(), "little")


def read_c_string_name_at(path: Path, block_offset: int, payload_bytes: int, names_offset: int, name_offset: int) -> str:
    cursor = names_offset + name_offset
    if cursor + 2 > payload_bytes:
        raise ValueError("name offset outside block")
    raw_size = read_at(path, block_offset + cursor, 2)
    (size,) = struct.unpack("<H", raw_size)
    cursor += 2
    if cursor + size > payload_bytes:
        raise ValueError("truncated name blob")
    return read_at(path, block_offset + cursor, size).decode("utf-8")


def parse_block(path: Path, offset: int, payload_bytes: int, sample_limit: int) -> dict[str, Any]:
    header_bytes = read_at(path, offset, BLOCK_HEADER_STRUCT.size)
    magic, version, tensor_count, quant_format, flags, record_offset, names_offset = BLOCK_HEADER_STRUCT.unpack(header_bytes)
    if magic != zc.BLOCK_MAGIC:
        raise ValueError(f"{path}: invalid block magic {magic!r}")
    if version != 1:
        raise ValueError(f"{path}: unsupported block version {version}")
    record_table_bytes = tensor_count * TENSOR_RECORD_STRUCT.size
    if record_offset + record_table_bytes > payload_bytes:
        raise ValueError(f"{path}: truncated record table")
    records_bytes = read_at(path, offset + record_offset, record_table_bytes)
    records = []
    total_data_bytes = 0
    quant_formats: dict[str, int] = {}
    dtype_codes: dict[str, int] = {}
    role_codes: dict[str, int] = {}
    for index in range(tensor_count):
        cursor = index * TENSOR_RECORD_STRUCT.size
        (
            dtype_original,
            record_quant_format,
            rank,
            role_code,
            name_offset,
            shape_offset,
            data_offset,
            data_bytes,
            scale,
            zero_point,
        ) = TENSOR_RECORD_STRUCT.unpack_from(records_bytes, cursor)
        if data_offset + data_bytes > payload_bytes:
            raise ValueError(f"{path}: tensor payload outside block")
        quant_formats[str(record_quant_format)] = quant_formats.get(str(record_quant_format), 0) + 1
        dtype_codes[str(dtype_original)] = dtype_codes.get(str(dtype_original), 0) + 1
        role_codes[str(role_code)] = role_codes.get(str(role_code), 0) + 1
        total_data_bytes += data_bytes
        if len(records) < sample_limit:
            name = read_c_string_name_at(path, offset, payload_bytes, names_offset, name_offset)
            records.append(
                {
                    "name": name,
                    "dtype_code": dtype_original,
                    "quant_format": record_quant_format,
                    "rank": rank,
                    "role_code": role_code,
                    "data_bytes": data_bytes,
                    "scale": scale,
                    "zero_point": zero_point,
                    "shape_offset": shape_offset,
                }
            )
    return {
        "tensor_count": tensor_count,
        "block_quant_format": quant_format,
        "flags": flags,
        "payload_bytes": payload_bytes,
        "total_tensor_data_bytes": total_data_bytes,
        "quant_format_counts": dict(sorted(quant_formats.items())),
        "dtype_code_counts": dict(sorted(dtype_codes.items())),
        "role_code_counts": dict(sorted(role_codes.items())),
        "record_sample": records,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate DeepSeek raw Wohper conversion")
    parser.add_argument("--core", type=Path, default=Path("models/wohper/DeepSeek-V4-Flash.RAW.L0-E0E1/dense_core.bin"))
    parser.add_argument("--shards", type=Path)
    parser.add_argument("--out", type=Path, default=Path("state/deepseek_v4_raw_conversion_validate_2026-07-05.json"))
    parser.add_argument("--sample-limit", type=int, default=12)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    shard_index_path = args.shards or args.core.with_suffix(".shards.json")
    shard_index = load_json(shard_index_path)
    header = ENGINE_HEADER_STRUCT.unpack(read_at(args.core, 0, ENGINE_HEADER_STRUCT.size))
    (
        magic,
        version,
        endian,
        file_size,
        manifest_offset,
        manifest_size,
        tokenizer_offset,
        tokenizer_size,
        router_metadata_offset,
        router_metadata_size,
        model_family,
        architecture,
        num_layers,
        hidden_size,
        num_attention_heads,
        num_kv_heads,
        experts_per_layer,
        active_experts_per_token,
        block_alignment,
        disk_quant_format,
        manifest_checksum,
        file_checksum,
    ) = header
    blockers: list[str] = []
    if magic != zc.MODEL_MAGIC:
        blockers.append("invalid_model_magic")
    if version != zc.FORMAT_VERSION:
        blockers.append("unsupported_model_version")
    if file_size != args.core.stat().st_size:
        blockers.append("file_size_header_mismatch")
    if block_alignment != zc.ALIGN_2MB:
        blockers.append("unexpected_block_alignment")
    if shard_index.get("format") != "wohper-sharded-experts":
        blockers.append("invalid_shard_index_format")

    manifest_header = MANIFEST_HEADER_STRUCT.unpack(read_at(args.core, manifest_offset, MANIFEST_HEADER_STRUCT.size))
    layer_count, expert_count, tensor_count, reserved, layer_desc_offset, expert_desc_offset, tensor_desc_offset = manifest_header
    layers = []
    dense_blocks = []
    for index in range(layer_count):
        raw = read_at(
            args.core,
            manifest_offset + layer_desc_offset + index * LAYER_DESC_STRUCT.size,
            LAYER_DESC_STRUCT.size,
        )
        fields = LAYER_DESC_STRUCT.unpack(raw)
        (
            layer_id,
            flags,
            dense_offset,
            dense_disk_bytes,
            dense_payload_bytes,
            dense_dequant_bytes,
            first_tensor_index,
            layer_tensor_count,
            first_expert_index,
            num_experts,
            quant_format,
            checksum_kind,
            checksum,
        ) = fields
        if dense_offset % zc.ALIGN_2MB != 0 or dense_disk_bytes % zc.ALIGN_2MB != 0:
            blockers.append(f"unaligned_dense_layer_{layer_id}")
        actual_dense_checksum = checksum64_range(args.core, dense_offset, dense_payload_bytes)
        if checksum and actual_dense_checksum != checksum:
            blockers.append(f"dense_checksum_mismatch_{layer_id}")
        layers.append(
            {
                "layer_id": layer_id,
                "dense_offset": dense_offset,
                "dense_disk_bytes": dense_disk_bytes,
                "dense_payload_bytes": dense_payload_bytes,
                "num_experts": num_experts,
                "quant_format": quant_format,
                "checksum": checksum,
                "actual_checksum": actual_dense_checksum,
            }
        )
        dense_blocks.append(parse_block(args.core, dense_offset, dense_payload_bytes, args.sample_limit))

    expert_shards = []
    for expert in shard_index.get("experts", []):
        expert_path = args.core.parent / expert["path"]
        if not expert_path.exists():
            blockers.append(f"missing_expert_{expert['layer_id']}_{expert['expert_id']}")
            continue
        if expert_path.stat().st_size != int(expert["disk_bytes"]):
            blockers.append(f"expert_size_mismatch_{expert['layer_id']}_{expert['expert_id']}")
        actual_expert_checksum = checksum64_range(expert_path, 0, int(expert["payload_bytes"]))
        if int(expert.get("checksum", 0)) and actual_expert_checksum != int(expert["checksum"]):
            blockers.append(f"expert_checksum_mismatch_{expert['layer_id']}_{expert['expert_id']}")
        block = parse_block(expert_path, 0, int(expert["payload_bytes"]), args.sample_limit)
        expert_shards.append(
            {
                "layer_id": expert["layer_id"],
                "expert_id": expert["expert_id"],
                "path": str(expert_path),
                "disk_bytes": expert["disk_bytes"],
                "payload_bytes": expert["payload_bytes"],
                "checksum": int(expert.get("checksum", 0)),
                "actual_checksum": actual_expert_checksum,
                "block": block,
            }
        )

    payload = {
        "format": "deepseek-v4-raw-conversion-validation",
        "version": 1,
        "status": "ready" if not blockers else "blocked",
        "core": str(args.core),
        "shard_index": str(shard_index_path),
        "blockers": blockers,
        "header": {
            "file_size": file_size,
            "model_family": model_family,
            "architecture": architecture,
            "num_layers": num_layers,
            "hidden_size": hidden_size,
            "experts_per_layer": experts_per_layer,
            "active_experts_per_token": active_experts_per_token,
            "disk_quant_format": disk_quant_format,
            "manifest_offset": manifest_offset,
            "manifest_size": manifest_size,
        },
        "manifest": {
            "layer_count": layer_count,
            "expert_count": expert_count,
            "tensor_count": tensor_count,
        },
        "layers": layers,
        "dense_blocks": dense_blocks,
        "expert_shards": expert_shards,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not blockers else 3


if __name__ == "__main__":
    raise SystemExit(main())
