#!/usr/bin/env python3
"""Stream-copy DeepSeek-V4-Flash quantized safetensors into Wohper raw blocks.

This converter is intentionally lossless for DeepSeek payloads: it does not
requantize FP8/FP4 tensors and it never loads whole tensors into RAM. It copies
the safetensors byte ranges into ZCBLK01 blocks with enough metadata for the
DeepSeek runtime decoder to interpret the original dtype/role.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import struct
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import convert_safetensors as zc
from plan_deepseek_v4_flash_inventory import classify_tensor


QUANT_DEEPSEEK_RAW_MIXED = 2400
QUANT_DEEPSEEK_FP8_E4M3 = 2401
QUANT_DEEPSEEK_FP4_E2M1_PACKED = 2402
QUANT_DEEPSEEK_UE8M0_SCALE = 2403
QUANT_DEEPSEEK_BF16_AUX = 2404
QUANT_DEEPSEEK_F32_AUX = 2405
QUANT_DEEPSEEK_I64_AUX = 2406

DTYPE_TO_ZC_NAME = {
    "BF16": "bfloat16",
    "F32": "float32",
    "I64": "int64",
    "I8": "int8",
}


@dataclass(frozen=True)
class TensorItem:
    name: str
    shard: str
    dtype: str
    shape: tuple[int, ...]
    data_start: int
    data_end: int
    header_len: int
    role: str
    layer_id: int | None
    expert_id: int | None

    @property
    def data_bytes(self) -> int:
        return self.data_end - self.data_start

    @property
    def absolute_start(self) -> int:
        return 8 + self.header_len + self.data_start

    @property
    def absolute_end(self) -> int:
        return 8 + self.header_len + self.data_end


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_layer_id(name: str) -> int | None:
    parts = name.split(".")
    for idx, part in enumerate(parts):
        if part == "layers" and idx + 1 < len(parts):
            try:
                return int(parts[idx + 1])
            except ValueError:
                return None
    return None


def parse_expert_id(name: str) -> int | None:
    match = re.search(r"(?:^|\.)ffn\.experts\.(\d+)\.", name)
    if match:
        return int(match.group(1))
    match = re.search(r"(?:^|\.)experts\.(\d+)\.", name)
    if match:
        return int(match.group(1))
    return None


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


def build_tensor_items(model_dir: Path) -> dict[str, TensorItem]:
    index = load_json(model_dir / "model.safetensors.index.json")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError("model.safetensors.index.json has no weight_map")
    by_shard: dict[str, list[str]] = defaultdict(list)
    for name, shard in weight_map.items():
        by_shard[str(shard)].append(str(name))

    items: dict[str, TensorItem] = {}
    for shard, names in sorted(by_shard.items()):
        shard_path = model_dir / shard
        if not shard_path.exists():
            raise FileNotFoundError(shard_path)
        header_len, header = read_safetensors_header(shard_path)
        for name in names:
            entry = header.get(name)
            if not isinstance(entry, dict):
                raise KeyError(f"{name} missing from {shard}")
            data_offsets = entry.get("data_offsets")
            if not isinstance(data_offsets, list) or len(data_offsets) != 2:
                raise ValueError(f"{name}: invalid safetensors data_offsets")
            role = classify_tensor(name)
            items[name] = TensorItem(
                name=name,
                shard=shard,
                dtype=str(entry.get("dtype", "unknown")).upper(),
                shape=tuple(int(dim) for dim in entry.get("shape", [])),
                data_start=int(data_offsets[0]),
                data_end=int(data_offsets[1]),
                header_len=header_len,
                role=role,
                layer_id=parse_layer_id(name),
                expert_id=parse_expert_id(name),
            )
    return items


def tensor_quant_format(item: TensorItem) -> int:
    lowered = item.name.lower()
    if lowered.endswith(".scale") or item.dtype == "F8_E8M0":
        return QUANT_DEEPSEEK_UE8M0_SCALE
    if item.role in {"expert", "shared_expert"} and item.dtype == "I8":
        return QUANT_DEEPSEEK_FP4_E2M1_PACKED
    if item.dtype == "F8_E4M3":
        return QUANT_DEEPSEEK_FP8_E4M3
    if item.dtype == "BF16":
        return QUANT_DEEPSEEK_BF16_AUX
    if item.dtype == "F32":
        return QUANT_DEEPSEEK_F32_AUX
    if item.dtype == "I64":
        return QUANT_DEEPSEEK_I64_AUX
    if item.dtype == "I8":
        return QUANT_DEEPSEEK_FP4_E2M1_PACKED
    return QUANT_DEEPSEEK_RAW_MIXED


def dtype_code(item: TensorItem) -> int:
    if item.dtype == "F8_E4M3":
        return 101
    if item.dtype == "F8_E8M0":
        return 102
    return zc.DTYPE_CODES.get(DTYPE_TO_ZC_NAME.get(item.dtype, item.dtype.lower()), 0)


def tensor_role_code(item: TensorItem) -> int:
    if item.role == "embed":
        return zc.TENSOR_ROLE_EMBED
    if item.role == "lm_head":
        return zc.TENSOR_ROLE_LM_HEAD
    if item.role == "norm":
        return zc.TENSOR_ROLE_NORM
    if item.role == "router":
        return zc.TENSOR_ROLE_ROUTER
    if item.role == "attention_q":
        return zc.TENSOR_ROLE_Q_PROJ
    if item.role == "attention_kv":
        return zc.TENSOR_ROLE_KV_PROJ
    if item.role == "attention_o":
        return zc.TENSOR_ROLE_O_PROJ
    if item.role == "shared_expert":
        return zc.TENSOR_ROLE_SHARED_EXPERT
    lower = item.name.lower()
    if ".w1." in lower:
        return zc.TENSOR_ROLE_GATE_PROJ
    if ".w2." in lower:
        return zc.TENSOR_ROLE_DOWN_PROJ
    if ".w3." in lower:
        return zc.TENSOR_ROLE_UP_PROJ
    return zc.TENSOR_ROLE_UNKNOWN


def estimate_block_payload(items: list[TensorItem]) -> int:
    names_bytes = sum(2 + len(item.name.encode("utf-8")) for item in items)
    shape_bytes = sum(4 + 8 * len(item.shape) for item in items)
    return (
        zc.BLOCK_HEADER_STRUCT.size
        + len(items) * zc.TENSOR_RECORD_STRUCT.size
        + names_bytes
        + sum(item.data_bytes for item in items)
        + shape_bytes
    )


def group_items(
    items: dict[str, TensorItem],
    selected_layers: list[int],
    selected_experts: list[int] | None,
    global_policy: str,
) -> tuple[dict[int, list[TensorItem]], dict[int, dict[int, list[TensorItem]]], list[TensorItem]]:
    dense_by_layer: dict[int, list[TensorItem]] = {layer: [] for layer in selected_layers}
    experts_by_layer: dict[int, dict[int, list[TensorItem]]] = {layer: defaultdict(list) for layer in selected_layers}
    global_items: list[TensorItem] = []
    selected_layer_set = set(selected_layers)
    selected_expert_set = set(selected_experts) if selected_experts is not None else None

    for item in sorted(items.values(), key=lambda value: value.name):
        if item.layer_id is None:
            if global_policy != "none":
                if global_policy == "embed_only" and item.role != "embed":
                    continue
                global_items.append(item)
            continue
        if item.layer_id not in selected_layer_set:
            continue
        if item.role == "expert" and item.expert_id is not None:
            if selected_expert_set is None or item.expert_id in selected_expert_set:
                experts_by_layer[item.layer_id][item.expert_id].append(item)
            continue
        dense_by_layer[item.layer_id].append(item)

    return dense_by_layer, {k: dict(v) for k, v in experts_by_layer.items()}, global_items


def parse_layer_range(value: str, total_layers: int) -> list[int]:
    if value == "all":
        return list(range(total_layers))
    parts = [part.strip() for part in value.split(",", 1)]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("--layer-range must be all or START,END")
    start, end = int(parts[0]), int(parts[1])
    if start < 0 or end <= start or start >= total_layers:
        raise argparse.ArgumentTypeError("invalid layer range")
    return list(range(start, min(end, total_layers)))


def parse_expert_ids(value: str) -> list[int] | None:
    if value == "all":
        return None
    ids = sorted({int(part.strip()) for part in value.split(",") if part.strip()})
    if not ids:
        raise argparse.ArgumentTypeError("--expert-ids cannot be empty")
    return ids


def write_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def copy_tensor_payload(model_dir: Path, item: TensorItem, out, chunk_bytes: int) -> None:
    with (model_dir / item.shard).open("rb") as src:
        src.seek(item.absolute_start)
        remaining = item.data_bytes
        while remaining:
            chunk = src.read(min(chunk_bytes, remaining))
            if not chunk:
                raise EOFError(f"{item.name}: unexpected EOF while copying payload")
            out.write(chunk)
            remaining -= len(chunk)


def write_raw_block_payload(
    model_dir: Path,
    items: list[TensorItem],
    block_tmp: Path,
    chunk_bytes: int,
) -> tuple[int, int, int]:
    names_blob = bytearray()
    name_offsets: list[int] = []
    for item in items:
        name_offsets.append(len(names_blob))
        names_blob += zc.pack_tensor_name(item.name)

    with block_tmp.open("wb") as out:
        out.write(b"\0" * zc.BLOCK_HEADER_STRUCT.size)
        record_table_offset = out.tell()
        out.write(b"\0" * (len(items) * zc.TENSOR_RECORD_STRUCT.size))
        names_offset = out.tell()
        out.write(names_blob)

        data_offsets: list[int] = []
        for item in items:
            data_offsets.append(out.tell())
            copy_tensor_payload(model_dir, item, out, chunk_bytes)

        out.seek(0)
        out.write(
            zc.BLOCK_HEADER_STRUCT.pack(
                zc.BLOCK_MAGIC,
                1,
                len(items),
                QUANT_DEEPSEEK_RAW_MIXED,
                0,
                record_table_offset,
                names_offset,
            )
        )

        out.seek(record_table_offset)
        for item, name_offset, data_offset in zip(items, name_offsets, data_offsets):
            shape_offset = zc.append_shape_blob(out, item.shape)
            out.write(
                zc.TENSOR_RECORD_STRUCT.pack(
                    min(dtype_code(item), 65535),
                    min(tensor_quant_format(item), 65535),
                    len(item.shape),
                    tensor_role_code(item),
                    name_offset,
                    shape_offset,
                    data_offset,
                    item.data_bytes,
                    1.0,
                    0.0,
                )
            )
        out.seek(0, os.SEEK_END)
        payload_size = out.tell()

    return payload_size, sum(item.data_bytes for item in items), zc.checksum64_file(block_tmp)


def plan_payload(
    dense_by_layer: dict[int, list[TensorItem]],
    experts_by_layer: dict[int, dict[int, list[TensorItem]]],
    global_items: list[TensorItem] | None = None,
) -> dict[str, Any]:
    dense_payload = 0
    dense_disk = 0
    expert_payload = 0
    expert_disk = 0
    expert_count = 0
    empty_dense_layers = []
    missing_experts = []
    for layer, refs in dense_by_layer.items():
        if not refs:
            empty_dense_layers.append(layer)
        payload = estimate_block_payload(refs)
        dense_payload += payload
        dense_disk += zc.align_up(payload)
    global_payload = 0
    global_disk = 0
    if global_items:
        global_payload = estimate_block_payload(global_items)
        global_disk = zc.align_up(global_payload)
        dense_payload += global_payload
        dense_disk += global_disk
    for layer, experts in experts_by_layer.items():
        for expert_id, refs in experts.items():
            if not refs:
                missing_experts.append({"layer_id": layer, "expert_id": expert_id})
            payload = estimate_block_payload(refs)
            expert_payload += payload
            expert_disk += zc.align_up(payload)
            expert_count += 1
    return {
        "dense_payload_estimate_bytes": dense_payload,
        "dense_disk_estimate_bytes": dense_disk,
        "expert_payload_estimate_bytes": expert_payload,
        "expert_disk_estimate_bytes": expert_disk,
        "total_output_disk_estimate_bytes": dense_disk + expert_disk,
        "dense_layer_count": len(dense_by_layer),
        "global_aux_payload_estimate_bytes": global_payload,
        "global_aux_disk_estimate_bytes": global_disk,
        "global_aux_tensor_count": len(global_items or []),
        "expert_shard_count": expert_count,
        "empty_dense_layers": empty_dense_layers,
        "missing_experts": missing_experts[:32],
    }


def preflight_disk(out: Path, required_bytes: int, min_free_after_gb: float) -> dict[str, Any]:
    root = out.parent if out.parent.exists() else Path(".")
    usage = shutil.disk_usage(root)
    min_free_after = int(min_free_after_gb * 1024**3)
    return {
        "filesystem_total_bytes": usage.total,
        "filesystem_free_bytes": usage.free,
        "required_output_bytes": required_bytes,
        "min_free_after_bytes": min_free_after,
        "feasible": usage.free - required_bytes >= min_free_after,
    }


def execute_conversion(
    args: argparse.Namespace,
    config: dict[str, Any],
    dense_by_layer: dict[int, list[TensorItem]],
    experts_by_layer: dict[int, dict[int, list[TensorItem]]],
    global_items: list[TensorItem],
    estimate: dict[str, Any],
) -> dict[str, Any]:
    core_path = args.out
    experts_dir = args.experts_dir or (core_path.parent / "experts")
    shard_index = zc.sharded_manifest_path(core_path, args.shard_index_out)
    selected_layers = list(dense_by_layer)
    selected_expert_total = sum(len(experts) for experts in experts_by_layer.values())
    manifest_layer_count = len(selected_layers) + (1 if global_items else 0)
    manifest_size = (
        zc.MANIFEST_HEADER_SIZE
        + manifest_layer_count * zc.LAYER_DESC_SIZE
        + selected_expert_total * zc.EXPERT_DESC_SIZE
    )

    core_path.parent.mkdir(parents=True, exist_ok=True)
    experts_dir.mkdir(parents=True, exist_ok=True)
    layer_plans: list[zc.LayerPlan] = []
    expert_plans: list[zc.ExpertPlan] = []
    expert_shards: list[zc.ExpertShardPlan] = []
    progress: dict[str, Any] = {
        "format": "deepseek-v4-stream-conversion-progress",
        "version": 1,
        "status": "running",
        "core_path": str(core_path),
        "experts_dir": str(experts_dir),
        "estimate": estimate,
        "layers_done": [],
        "experts_done": [],
        "errors": [],
    }
    write_report(args.progress, progress)

    with tempfile.TemporaryDirectory(prefix="zc_deepseek_convert_") as tmp_dir_text:
        tmp_dir = Path(tmp_dir_text)
        with core_path.open("wb") as out:
            manifest_offset = zc.reserve_header_and_manifest(out, manifest_size)
            if global_items:
                global_tmp = tmp_dir / "global_aux.zcblk"
                _, global_dequant, global_checksum = write_raw_block_payload(
                    args.model_dir,
                    global_items,
                    global_tmp,
                    args.chunk_mb * 1024 * 1024,
                )
                global_offset, global_disk, global_payload = zc.copy_aligned_block(out, global_tmp)
                global_tmp.unlink(missing_ok=True)
                layer_plans.append(
                    zc.LayerPlan(
                        layer_id=0,
                        dense_offset=global_offset,
                        dense_disk_bytes=global_disk,
                        dense_payload_bytes=global_payload,
                        dense_dequant_bytes=global_dequant,
                        first_expert_index=0,
                        num_experts=0,
                        checksum=global_checksum,
                        block_type=zc.LAYER_BLOCK_GLOBAL_AUX,
                    )
                )
                progress["layers_done"].append(
                    {
                        "layer_id": 0,
                        "block_type": "global_aux",
                        "dense_disk_bytes": global_disk,
                        "dense_payload_bytes": global_payload,
                        "checksum": global_checksum,
                    }
                )
                write_report(args.progress, progress)
            for layer_id in selected_layers:
                dense_refs = dense_by_layer[layer_id]
                dense_tmp = tmp_dir / f"dense_layer_{layer_id}.zcblk"
                _, dense_dequant, dense_checksum = write_raw_block_payload(
                    args.model_dir,
                    dense_refs,
                    dense_tmp,
                    args.chunk_mb * 1024 * 1024,
                )
                dense_offset, dense_disk, dense_payload = zc.copy_aligned_block(out, dense_tmp)
                dense_tmp.unlink(missing_ok=True)
                first_expert_index = len(expert_plans)

                for expert_id, refs in sorted(experts_by_layer[layer_id].items()):
                    expert_name = f"layer{layer_id}_expert{expert_id}.zcblk"
                    expert_path = experts_dir / expert_name
                    expert_tmp = tmp_dir / f"{expert_name}.tmp"
                    _, expert_dequant, expert_checksum = write_raw_block_payload(
                        args.model_dir,
                        refs,
                        expert_tmp,
                        args.chunk_mb * 1024 * 1024,
                    )
                    expert_disk, expert_payload = zc.write_aligned_file(expert_path, expert_tmp)
                    expert_tmp.unlink(missing_ok=True)
                    expert_plans.append(
                        zc.ExpertPlan(
                            layer_id=layer_id,
                            expert_id=expert_id,
                            disk_offset=0,
                            disk_bytes=expert_disk,
                            payload_bytes=expert_payload,
                            dequant_bytes=expert_dequant,
                            quant_format=QUANT_DEEPSEEK_RAW_MIXED,
                            route_rank_hint=expert_id,
                            checksum=expert_checksum,
                        )
                    )
                    try:
                        expert_rel = expert_path.relative_to(core_path.parent)
                    except ValueError:
                        expert_rel = expert_path
                    expert_shards.append(
                        zc.ExpertShardPlan(
                            layer_id=layer_id,
                            expert_id=expert_id,
                            path=str(expert_rel).replace("\\", "/"),
                            disk_bytes=expert_disk,
                            payload_bytes=expert_payload,
                            dequant_bytes=expert_dequant,
                            quant_format=QUANT_DEEPSEEK_RAW_MIXED,
                            checksum=expert_checksum,
                        )
                    )
                    progress["experts_done"].append(
                        {
                            "layer_id": layer_id,
                            "expert_id": expert_id,
                            "disk_bytes": expert_disk,
                            "payload_bytes": expert_payload,
                            "checksum": expert_checksum,
                        }
                    )
                    write_report(args.progress, progress)

                layer_plans.append(
                    zc.LayerPlan(
                        layer_id=layer_id,
                        dense_offset=dense_offset,
                        dense_disk_bytes=dense_disk,
                        dense_payload_bytes=dense_payload,
                        dense_dequant_bytes=dense_dequant,
                        first_expert_index=first_expert_index,
                        num_experts=len(experts_by_layer[layer_id]),
                        checksum=dense_checksum,
                    )
                )
                progress["layers_done"].append(
                    {
                        "layer_id": layer_id,
                        "dense_disk_bytes": dense_disk,
                        "dense_payload_bytes": dense_payload,
                        "checksum": dense_checksum,
                    }
                )
                write_report(args.progress, progress)

            file_size = out.tell()
            manifest_payload = zc.pack_manifest(layer_plans, expert_plans, QUANT_DEEPSEEK_RAW_MIXED)
            manifest_checksum = zc.checksum64_bytes(manifest_payload)
            header = zc.ENGINE_HEADER_STRUCT.pack(
                zc.MODEL_MAGIC,
                zc.FORMAT_VERSION,
                0,
                file_size,
                manifest_offset,
                len(manifest_payload),
                0,
                0,
                0,
                0,
                2,
                2,
                len(layer_plans),
                int(config.get("hidden_size", 4096)),
                int(config.get("num_attention_heads", 0) or 0),
                int(config.get("num_key_value_heads", 0) or 0),
                max((len(experts) for experts in experts_by_layer.values()), default=0),
                int(config.get("num_experts_per_tok", 6)),
                zc.ALIGN_2MB,
                QUANT_DEEPSEEK_RAW_MIXED,
                manifest_checksum,
                0,
            )
            out.seek(0)
            out.write(header)
            out.seek(manifest_offset)
            out.write(manifest_payload)

    zc.validate_output(core_path, layer_plans, [], manifest_offset)
    zc.write_sharded_index(
        shard_index,
        core_path=core_path,
        experts_dir=experts_dir,
        selected_layers=selected_layers,
        num_layers_total=int(config.get("num_hidden_layers", len(selected_layers))),
        experts_per_layer=max((len(experts) for experts in experts_by_layer.values()), default=0),
        layer_plans=layer_plans,
        expert_shards=expert_shards,
        quant_format=QUANT_DEEPSEEK_RAW_MIXED,
        metadata={
            "model_family": "deepseek_v4_flash",
            "raw_quant_container": True,
            "source_model_dir": str(args.model_dir),
            "created_at_unix": int(time.time()),
            "global_aux_separate": bool(global_items),
        },
    )
    progress["status"] = "ready"
    progress["core_file_bytes"] = core_path.stat().st_size
    progress["shard_index"] = str(shard_index)
    progress["expert_shards"] = len(expert_shards)
    write_report(args.progress, progress)
    return progress


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DeepSeek-V4 raw streaming converter")
    parser.add_argument("--model-dir", type=Path, default=Path("models/deepseek-ai/DeepSeek-V4-Flash"))
    parser.add_argument("--out", type=Path, default=Path("models/wohper/DeepSeek-V4-Flash.RAW/dense_core.bin"))
    parser.add_argument("--experts-dir", type=Path)
    parser.add_argument("--shard-index-out", type=Path)
    parser.add_argument("--report", type=Path, default=Path("state/deepseek_v4_stream_convert_plan_2026-07-05.json"))
    parser.add_argument("--progress", type=Path, default=Path("state/deepseek_v4_stream_convert_progress_2026-07-05.json"))
    parser.add_argument("--layer-range", default="0,1", help="all or START,END")
    parser.add_argument("--expert-ids", default="0,1", help="all or comma-separated expert ids")
    parser.add_argument("--global-policy", choices=("none", "embed_only", "all"), default="none")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--chunk-mb", type=int, default=8)
    parser.add_argument("--min-free-after-gb", type=float, default=250.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_json(args.model_dir / "config.json")
    total_layers = int(config.get("num_hidden_layers", 43))
    selected_layers = parse_layer_range(args.layer_range, total_layers)
    selected_experts = parse_expert_ids(args.expert_ids)
    items = build_tensor_items(args.model_dir)
    dense_by_layer, experts_by_layer, global_items = group_items(
        items,
        selected_layers,
        selected_experts,
        args.global_policy,
    )
    estimate = plan_payload(dense_by_layer, experts_by_layer, global_items)
    disk = preflight_disk(args.out, estimate["total_output_disk_estimate_bytes"], args.min_free_after_gb)
    role_counts = Counter(item.role for refs in dense_by_layer.values() for item in refs)
    expert_role_counts = Counter(
        item.role for experts in experts_by_layer.values() for refs in experts.values() for item in refs
    )
    payload = {
        "format": "deepseek-v4-stream-conversion-plan",
        "version": 1,
        "mode": "execute" if args.execute else "dry_run",
        "status": "dry_run_ready",
        "model_dir": str(args.model_dir),
        "out": str(args.out),
        "experts_dir": str(args.experts_dir or (args.out.parent / "experts")),
        "selected_layers": selected_layers,
        "selected_experts": "all" if selected_experts is None else selected_experts,
        "global_policy": args.global_policy,
        "global_tensor_count": len(global_items),
        "dense_role_counts": dict(sorted(role_counts.items())),
        "expert_role_counts": dict(sorted(expert_role_counts.items())),
        "estimate": estimate,
        "disk_preflight": disk,
        "warnings": [],
    }
    if not disk["feasible"]:
        payload["status"] = "blocked_low_disk"
        payload["warnings"].append("estimated output would violate min_free_after_gb")
    if estimate["empty_dense_layers"]:
        payload["status"] = "blocked_empty_dense_layers"
    write_report(args.report, payload)
    if not args.execute:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if payload["status"] == "dry_run_ready" else 3
    if payload["status"] != "dry_run_ready":
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 3
    progress = execute_conversion(args, config, dense_by_layer, experts_by_layer, global_items, estimate)
    payload["status"] = "ready"
    payload["execution"] = {
        "progress": str(args.progress),
        "core_file_bytes": progress.get("core_file_bytes"),
        "expert_shards": progress.get("expert_shards"),
        "shard_index": progress.get("shard_index"),
    }
    write_report(args.report, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
