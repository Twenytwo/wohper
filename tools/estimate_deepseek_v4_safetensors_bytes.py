#!/usr/bin/env python3
"""Estimate DeepSeek-V4-Flash safetensors bytes by role from shard headers only."""

from __future__ import annotations

import argparse
import json
import struct
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from plan_deepseek_v4_flash_inventory import classify_tensor, infer_storage_precision


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_safetensors_header(path: Path) -> tuple[int, dict[str, Any]]:
    with path.open("rb") as handle:
        raw_len = handle.read(8)
        if len(raw_len) != 8:
            raise ValueError(f"{path}: truncated safetensors length prefix")
        (header_len,) = struct.unpack("<Q", raw_len)
        header_raw = handle.read(header_len)
        if len(header_raw) != header_len:
            raise ValueError(f"{path}: truncated safetensors header")
    header = json.loads(header_raw.decode("utf-8"))
    if not isinstance(header, dict):
        raise ValueError(f"{path}: safetensors header is not an object")
    return int(header_len), header


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate DeepSeek safetensors bytes by role")
    parser.add_argument("--model-dir", type=Path, default=Path("models/deepseek-ai/DeepSeek-V4-Flash"))
    parser.add_argument("--out", type=Path, default=Path("state/deepseek_v4_flash_safetensors_header_bytes_2026-07-04.json"))
    parser.add_argument("--max-mismatch-sample", type=int, default=32)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    index = load_json(args.model_dir / "model.safetensors.index.json")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise SystemExit("model.safetensors.index.json has no weight_map")

    declared_shards = sorted({str(value) for value in weight_map.values()})
    missing_shards = [name for name in declared_shards if not (args.model_dir / name).exists()]
    if missing_shards:
        payload = {
            "format": "deepseek-v4-flash-safetensors-header-bytes",
            "version": 1,
            "status": "blocked_missing_safetensors",
            "missing_shard_count": len(missing_shards),
            "missing_shard_sample": missing_shards[: args.max_mismatch_sample],
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 3

    role_bytes: Counter[str] = Counter()
    precision_bytes: Counter[str] = Counter()
    dtype_bytes: Counter[str] = Counter()
    role_counts: Counter[str] = Counter()
    precision_counts: Counter[str] = Counter()
    dtype_counts: Counter[str] = Counter()
    shard_payload_bytes: dict[str, int] = {}
    shard_header_bytes: dict[str, int] = {}
    shard_tensor_counts: dict[str, int] = {}
    shard_max_offset: dict[str, int] = {}
    tensors_seen: set[str] = set()
    mismatch_sample: list[dict[str, Any]] = []
    unknown_sample: list[str] = []
    layer_role_bytes: dict[int, Counter[str]] = defaultdict(Counter)

    expected_by_shard: dict[str, set[str]] = defaultdict(set)
    for tensor_name, shard_name in weight_map.items():
        expected_by_shard[str(shard_name)].add(str(tensor_name))

    for shard in declared_shards:
        shard_path = args.model_dir / shard
        header_len, header = read_safetensors_header(shard_path)
        shard_header_bytes[shard] = header_len + 8
        tensor_entries = {
            name: value
            for name, value in header.items()
            if name != "__metadata__"
        }
        shard_tensor_counts[shard] = len(tensor_entries)
        expected_names = expected_by_shard[shard]
        actual_names = set(tensor_entries)
        for name in sorted(expected_names - actual_names):
            if len(mismatch_sample) >= args.max_mismatch_sample:
                break
            mismatch_sample.append({"shard": shard, "tensor": name, "problem": "missing_from_header"})
        for name in sorted(actual_names - expected_names):
            if len(mismatch_sample) >= args.max_mismatch_sample:
                break
            mismatch_sample.append({"shard": shard, "tensor": name, "problem": "extra_in_header"})

        max_end = 0
        payload_bytes = 0
        for name, entry in tensor_entries.items():
            if not isinstance(entry, dict):
                continue
            offsets = entry.get("data_offsets")
            if not isinstance(offsets, list) or len(offsets) != 2:
                if len(mismatch_sample) < args.max_mismatch_sample:
                    mismatch_sample.append({"shard": shard, "tensor": name, "problem": "missing_data_offsets"})
                continue
            start = int(offsets[0])
            end = int(offsets[1])
            size = max(0, end - start)
            max_end = max(max_end, end)
            payload_bytes += size
            role = classify_tensor(name)
            precision = infer_storage_precision(role, name)
            dtype = str(entry.get("dtype", "unknown"))
            role_bytes[role] += size
            precision_bytes[precision] += size
            dtype_bytes[dtype] += size
            role_counts[role] += 1
            precision_counts[precision] += 1
            dtype_counts[dtype] += 1
            tensors_seen.add(name)
            if role == "unknown" and len(unknown_sample) < args.max_mismatch_sample:
                unknown_sample.append(name)
            layer_id = None
            parts = name.split(".")
            for idx, part in enumerate(parts):
                if part == "layers" and idx + 1 < len(parts):
                    try:
                        layer_id = int(parts[idx + 1])
                    except ValueError:
                        layer_id = None
                    break
            if layer_id is not None:
                layer_role_bytes[layer_id][role] += size
        shard_payload_bytes[shard] = payload_bytes
        shard_max_offset[shard] = max_end
        data_region_bytes = max(0, shard_path.stat().st_size - shard_header_bytes[shard])
        if max_end != data_region_bytes and len(mismatch_sample) < args.max_mismatch_sample:
            mismatch_sample.append(
                {
                    "shard": shard,
                    "problem": "data_region_size_mismatch",
                    "max_data_offset": max_end,
                    "data_region_bytes": data_region_bytes,
                }
            )

    missing_from_headers = sorted(set(weight_map) - tensors_seen)
    for name in missing_from_headers:
        if len(mismatch_sample) >= args.max_mismatch_sample:
            break
        mismatch_sample.append({"tensor": name, "problem": "declared_tensor_not_seen"})

    status = "ready"
    if mismatch_sample:
        status = "blocked_header_mismatch"
    elif unknown_sample:
        status = "blocked_unknown_tensor_roles"

    total_payload = sum(shard_payload_bytes.values())
    total_headers = sum(shard_header_bytes.values())
    payload = {
        "format": "deepseek-v4-flash-safetensors-header-bytes",
        "version": 1,
        "status": status,
        "model_dir": str(args.model_dir),
        "declared_shards": len(declared_shards),
        "present_shards": len(declared_shards) - len(missing_shards),
        "declared_total_size_bytes": int((index.get("metadata") or {}).get("total_size", 0) or 0),
        "file_bytes": sum((args.model_dir / shard).stat().st_size for shard in declared_shards),
        "header_bytes": total_headers,
        "payload_bytes_from_offsets": total_payload,
        "tensor_count_from_index": len(weight_map),
        "tensor_count_from_headers": len(tensors_seen),
        "role_counts": dict(sorted(role_counts.items())),
        "role_bytes": dict(sorted(role_bytes.items())),
        "precision_counts": dict(sorted(precision_counts.items())),
        "precision_bytes": dict(sorted(precision_bytes.items())),
        "dtype_bytes": dict(sorted(dtype_bytes.items())),
        "shard_payload_bytes_sample": dict(list(sorted(shard_payload_bytes.items()))[:8]),
        "shard_tensor_counts_sample": dict(list(sorted(shard_tensor_counts.items()))[:8]),
        "layer_role_bytes_sample": {
            str(layer): dict(sorted(counter.items()))
            for layer, counter in sorted(layer_role_bytes.items())[:8]
        },
        "unknown_tensor_sample": unknown_sample,
        "mismatch_sample": mismatch_sample,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if status == "ready" else 3


if __name__ == "__main__":
    raise SystemExit(main())
