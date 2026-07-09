#!/usr/bin/env python3
"""Build a row-wise tensor index for Wohper DeepSeek raw MODEL files.

The index contains absolute byte offsets for tensors inside dense ZCBLK blocks.
It is metadata-only: tensor payload bytes are never read, so it is safe for
multi-GB embed/head blocks on 16GB machines.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path
from typing import Any

import convert_safetensors as zc


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_at(path: Path, offset: int, size: int) -> bytes:
    with path.open("rb") as handle:
        handle.seek(offset)
        data = handle.read(size)
    if len(data) != size:
        raise ValueError(f"{path}: short read at {offset} size {size}")
    return data


def read_name(core: Path, block_offset: int, payload_bytes: int, names_offset: int, name_offset: int) -> str:
    cursor = names_offset + name_offset
    if cursor + 2 > payload_bytes:
        raise ValueError("name offset outside block")
    (size,) = struct.unpack("<H", read_at(core, block_offset + cursor, 2))
    cursor += 2
    if cursor + size > payload_bytes:
        raise ValueError("truncated name blob")
    return read_at(core, block_offset + cursor, size).decode("utf-8")


def read_shape(core: Path, block_offset: int, payload_bytes: int, shape_offset: int) -> list[int]:
    if shape_offset + 4 > payload_bytes:
        raise ValueError("shape offset outside block")
    (rank,) = struct.unpack("<I", read_at(core, block_offset + shape_offset, 4))
    cursor = shape_offset + 4
    if cursor + rank * 8 > payload_bytes:
        raise ValueError("truncated shape blob")
    if rank == 0:
        return []
    raw = read_at(core, block_offset + cursor, rank * 8)
    return [int(value) for value in struct.unpack("<" + "Q" * rank, raw)]


def parse_dense_block(core: Path, layer: dict[str, Any]) -> list[dict[str, Any]]:
    block_offset = int(layer["dense_offset"])
    payload_bytes = int(layer["dense_payload_bytes"])
    header = read_at(core, block_offset, zc.BLOCK_HEADER_STRUCT.size)
    magic, version, tensor_count, block_quant, flags, record_offset, names_offset = zc.BLOCK_HEADER_STRUCT.unpack(header)
    if magic != zc.BLOCK_MAGIC:
        raise ValueError(f"invalid ZCBLK magic at dense layer {layer['layer_id']}")
    if version != 1:
        raise ValueError(f"unsupported ZCBLK version {version}")
    record_bytes = read_at(core, block_offset + record_offset, tensor_count * zc.TENSOR_RECORD_STRUCT.size)
    tensors = []
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
        ) = zc.TENSOR_RECORD_STRUCT.unpack_from(record_bytes, cursor)
        shape = read_shape(core, block_offset, payload_bytes, shape_offset)
        name = read_name(core, block_offset, payload_bytes, names_offset, name_offset)
        if data_offset + data_bytes > payload_bytes:
            raise ValueError(f"{name}: data outside block")
        row_bytes = None
        row_count = None
        row_width = None
        if len(shape) == 2 and shape[0] > 0 and data_bytes % shape[0] == 0:
            row_count = shape[0]
            row_width = shape[1]
            row_bytes = data_bytes // shape[0]
        tensors.append(
            {
                "name": name,
                "layer_id": int(layer["layer_id"]),
                "block_type": int(layer.get("block_type", 0)),
                "dtype_code": int(dtype_code),
                "quant_format": int(quant_format),
                "role_code": int(role_code),
                "shape": shape,
                "data_offset": int(data_offset),
                "absolute_data_offset": int(block_offset + data_offset),
                "data_bytes": int(data_bytes),
                "row_count": row_count,
                "row_width": row_width,
                "row_bytes": row_bytes,
                "scale": float(scale),
                "zero_point": float(zero_point),
            }
        )
    return tensors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build DeepSeek row-wise tensor index")
    parser.add_argument("--core", type=Path, default=Path("models/wohper/DeepSeek-V4-Flash.RAW.L0-GLOBAL-E0E1/dense_core.bin"))
    parser.add_argument("--shards", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def display_path(path: Path, base: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve())).replace("\\", "/")
    except ValueError:
        return str(path)


def main() -> int:
    args = parse_args()
    shard_index_path = args.shards or args.core.with_suffix(".shards.json")
    out_path = args.out or args.core.with_suffix(".tensor_index.json")
    shard_index = load_json(shard_index_path)
    tensors: list[dict[str, Any]] = []
    for layer in shard_index.get("dense_layers", []):
        tensors.extend(parse_dense_block(args.core, layer))
    by_name = {tensor["name"]: tensor for tensor in tensors}
    required = ["embed.weight", "head.weight"]
    blockers = [f"missing_{name}" for name in required if name not in by_name]
    for name in required:
        tensor = by_name.get(name)
        if tensor is None:
            continue
        if tensor["dtype_code"] != zc.DTYPE_CODES["bfloat16"] or tensor["quant_format"] != 2404:
            blockers.append(f"{name}_not_deepseek_bf16")
        if tensor["row_bytes"] is None:
            blockers.append(f"{name}_not_row_addressable")
    payload = {
        "format": "wohper-row-tensor-index",
        "version": 1,
        "status": "ready" if not blockers else "blocked",
        "blockers": blockers,
        "core_file": display_path(args.core, out_path.parent),
        "shard_index": display_path(shard_index_path, out_path.parent),
        "tensor_count": len(tensors),
        "tensors": tensors,
        "metadata": {
            "model_family": shard_index.get("metadata", {}).get("model_family"),
            "source_format": shard_index.get("format"),
            "rowwise_safe": True,
            "payload_bytes_read": 0,
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.verbose:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        summary = {
            "format": payload["format"],
            "version": payload["version"],
            "status": payload["status"],
            "blockers": payload["blockers"],
            "out": str(out_path),
            "core_file": str(args.core),
            "tensor_count": len(tensors),
            "rowwise_tensors": sum(1 for tensor in tensors if tensor["row_bytes"] is not None),
            "embed": by_name.get("embed.weight"),
            "head": by_name.get("head.weight"),
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if not blockers else 3


if __name__ == "__main__":
    raise SystemExit(main())
