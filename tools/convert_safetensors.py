#!/usr/bin/env python3
"""Convert Hugging Face safetensors MoE checkpoints into Wohper MODEL.bin.

This converter targets the current Rust reader in:

    engine/zc_infer_core/src/model_format.rs

It writes the exact EngineHeader, ManifestHeader, LayerBlockDescDisk and
ExpertBlockDescDisk layouts used by the runtime. Each dense block and each
expert block is independently 2MB-aligned for O_DIRECT + fixed-buffer reads.

The payload inside every block is self-describing:

    ZCBLK01\0 block header
    TensorRecord[n]
    quantized tensor payloads

The Rust hot path does not parse this inner block format yet; it is included so
the next AVX/dequant milestone can reconstruct tensor names, shapes, scales and
offsets without changing MODEL.bin's outer manifest.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import struct
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


ALIGN_2MB = 2 * 1024 * 1024
MODEL_MAGIC = b"ZCINF01\0"
BLOCK_MAGIC = b"ZCBLK01\0"
FORMAT_VERSION = 1

QUANT_INT8 = 8
QUANT_INT4 = 4
CHECKSUM_BLAKE2B64 = 1
LAYER_BLOCK_COMPUTE = 0
LAYER_BLOCK_GLOBAL_AUX = 1

# Keep synchronized with engine/zc_infer_core/src/model_format.rs.
ENGINE_HEADER_STRUCT = struct.Struct("<8sIIQQQQQQQIIIIIIIIIIQQ")
MANIFEST_HEADER_STRUCT = struct.Struct("<IIIIQQQ")
LAYER_DESC_STRUCT = struct.Struct("<IIQQQQIIIIIIQ")
EXPERT_DESC_STRUCT = struct.Struct("<IIQQQQIIQ")

ENGINE_HEADER_SIZE = ENGINE_HEADER_STRUCT.size
MANIFEST_HEADER_SIZE = MANIFEST_HEADER_STRUCT.size
LAYER_DESC_SIZE = LAYER_DESC_STRUCT.size
EXPERT_DESC_SIZE = EXPERT_DESC_STRUCT.size

# Inner block payload format, for future dequant/AVX kernels.
BLOCK_HEADER_STRUCT = struct.Struct("<8sIIIIQQ")
TENSOR_RECORD_STRUCT = struct.Struct("<HHIIQQQQff")

TENSOR_ROLE_UNKNOWN = 0
TENSOR_ROLE_QKV_PROJ = 1
TENSOR_ROLE_Q_PROJ = 2
TENSOR_ROLE_K_PROJ = 3
TENSOR_ROLE_V_PROJ = 4
TENSOR_ROLE_O_PROJ = 5
TENSOR_ROLE_GATE_PROJ = 6
TENSOR_ROLE_UP_PROJ = 7
TENSOR_ROLE_DOWN_PROJ = 8
TENSOR_ROLE_ROUTER = 9
TENSOR_ROLE_NORM = 10
TENSOR_ROLE_EMBED = 11
TENSOR_ROLE_LM_HEAD = 12
TENSOR_ROLE_SHARED_EXPERT = 13
TENSOR_ROLE_KV_PROJ = 14

TENSOR_ROLE_NAMES = {
    TENSOR_ROLE_UNKNOWN: "unknown",
    TENSOR_ROLE_QKV_PROJ: "qkv_proj",
    TENSOR_ROLE_Q_PROJ: "q_proj",
    TENSOR_ROLE_K_PROJ: "k_proj",
    TENSOR_ROLE_V_PROJ: "v_proj",
    TENSOR_ROLE_KV_PROJ: "kv_proj",
    TENSOR_ROLE_O_PROJ: "o_proj",
    TENSOR_ROLE_GATE_PROJ: "gate_proj",
    TENSOR_ROLE_UP_PROJ: "up_proj",
    TENSOR_ROLE_DOWN_PROJ: "down_proj",
    TENSOR_ROLE_ROUTER: "router",
    TENSOR_ROLE_NORM: "norm",
    TENSOR_ROLE_EMBED: "embed",
    TENSOR_ROLE_LM_HEAD: "lm_head",
    TENSOR_ROLE_SHARED_EXPERT: "shared_expert",
}

DTYPE_CODES = {
    "bool": 1,
    "uint8": 2,
    "int8": 3,
    "uint16": 4,
    "int16": 5,
    "uint32": 6,
    "int32": 7,
    "uint64": 8,
    "int64": 9,
    "float16": 10,
    "bfloat16": 11,
    "float32": 12,
    "float64": 13,
}


@dataclass(frozen=True)
class TensorRef:
    name: str
    filename: str
    layer_id: int | None
    expert_id: int | None
    role: str


@dataclass
class BlockTensorRecord:
    name: str
    dtype_original: str
    shape: tuple[int, ...]
    quant_format: int
    data_offset: int
    data_bytes: int
    runtime_bytes: int
    scale: float
    zero_point: float
    tensor_role: int


@dataclass
class LayerPlan:
    layer_id: int
    dense_offset: int
    dense_disk_bytes: int
    dense_payload_bytes: int
    dense_dequant_bytes: int
    first_expert_index: int
    num_experts: int
    checksum: int
    block_type: int = LAYER_BLOCK_COMPUTE


@dataclass
class ExpertPlan:
    layer_id: int
    expert_id: int
    disk_offset: int
    disk_bytes: int
    payload_bytes: int
    dequant_bytes: int
    quant_format: int
    route_rank_hint: int
    checksum: int


@dataclass
class ExpertShardPlan:
    layer_id: int
    expert_id: int
    path: str
    disk_bytes: int
    payload_bytes: int
    dequant_bytes: int
    quant_format: int
    checksum: int


@dataclass
class ConversionPlan:
    dense_by_layer: dict[int, list[TensorRef]] = field(default_factory=dict)
    experts_by_layer: dict[int, dict[int, list[TensorRef]]] = field(default_factory=dict)
    global_tensors: list[TensorRef] = field(default_factory=list)
    skipped_tensors: list[TensorRef] = field(default_factory=list)


def align_up(value: int, alignment: int = ALIGN_2MB) -> int:
    return (value + alignment - 1) // alignment * alignment


def checksum64_file(path: Path) -> int:
    import hashlib

    h = hashlib.blake2b(digest_size=8)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return int.from_bytes(h.digest(), "little")


def checksum64_bytes(data: bytes) -> int:
    import hashlib

    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "little")


def write_padding(handle, target_offset: int) -> None:
    current = handle.tell()
    if target_offset < current:
        raise ValueError(f"target offset {target_offset} is before current offset {current}")
    handle.write(b"\0" * (target_offset - current))


def parse_layer_id(name: str) -> int | None:
    patterns = [
        r"(?:^|\.)(?:model\.layers|layers|h|blocks|decoder\.layers|transformer\.layers)\.(\d+)\.",
        r"(?:^|/)layers\.(\d+)\.",
    ]
    for pattern in patterns:
        match = re.search(pattern, name)
        if match:
            return int(match.group(1))
    return None


def parse_expert_id(name: str) -> int | None:
    patterns = [
        r"(?:^|\.)(?:mlp\.experts|experts|local_experts)\.(\d+)\.",
        r"(?:^|\.)(?:block_sparse_moe\.experts|moe\.experts)\.(\d+)\.",
        r"(?:^|\.)(?:feed_forward\.experts|ffn\.experts|moe_block\.experts)\.(\d+)\.",
    ]
    for pattern in patterns:
        match = re.search(pattern, name)
        if match:
            return int(match.group(1))
    return None


def classify_tensor(name: str, filename: str) -> TensorRef:
    layer_id = parse_layer_id(name)
    expert_id = parse_expert_id(name)

    if layer_id is None:
        return TensorRef(name, filename, None, None, "global")
    if expert_id is not None:
        return TensorRef(name, filename, layer_id, expert_id, "expert")

    dense_markers = (
        "self_attn",
        "attention",
        ".attn.",
        "input_layernorm",
        "post_attention_layernorm",
        "pre_mlp_layernorm",
        "post_layernorm",
        "kv_a_layernorm",
        "q_a_layernorm",
        "indexer.",
        "router",
        "mlp.gate.weight",
        "mlp.gate.",
        "gate.weight",
        "gate_score",
        "e_score",
        "score_correction_bias",
        "shared_experts",
        "shared_expert",
        "routed_experts",
    )
    if any(marker in name for marker in dense_markers):
        return TensorRef(name, filename, layer_id, None, "dense")

    # Unknown layer-local tensors are packed into dense because they are safer to
    # read with the always-needed layer block than to hide as an expert.
    return TensorRef(name, filename, layer_id, None, "dense")


def tensor_role_code(name: str) -> int:
    lower = name.lower()
    tail = lower.rsplit(".", 2)[0] if lower.endswith(".weight") else lower

    if "embed_tokens" in lower or lower.endswith("wte.weight"):
        return TENSOR_ROLE_EMBED
    if lower.startswith("lm_head.") or ".lm_head." in lower:
        return TENSOR_ROLE_LM_HEAD
    if "layernorm" in lower or lower.endswith(".norm.weight") or ".norm." in lower:
        return TENSOR_ROLE_NORM
    if "shared_expert" in lower or "shared_experts" in lower:
        return TENSOR_ROLE_SHARED_EXPERT
    if (
        "router" in lower
        or "gate_score" in lower
        or "e_score" in lower
        or lower.endswith(".mlp.gate.weight")
        or ".mlp.gate." in lower
    ):
        return TENSOR_ROLE_ROUTER

    qkv_markers = (
        "qkv_proj",
        "query_key_value",
        "c_attn",
        "wqkv",
    )
    if any(marker in lower for marker in qkv_markers):
        return TENSOR_ROLE_QKV_PROJ
    if "kv_a_proj" in lower or "kv_b_proj" in lower or ".kv_proj" in lower:
        return TENSOR_ROLE_KV_PROJ
    if "q_a_proj" in lower or "q_b_proj" in lower or ".q_proj" in lower or tail.endswith(".q"):
        return TENSOR_ROLE_Q_PROJ
    if ".k_proj" in lower or tail.endswith(".k"):
        return TENSOR_ROLE_K_PROJ
    if ".v_proj" in lower or tail.endswith(".v"):
        return TENSOR_ROLE_V_PROJ
    if ".o_proj" in lower or "out_proj" in lower or "dense.weight" in lower:
        return TENSOR_ROLE_O_PROJ
    if "gate_proj" in lower or lower.endswith(".w1.weight"):
        return TENSOR_ROLE_GATE_PROJ
    if "up_proj" in lower or lower.endswith(".w3.weight"):
        return TENSOR_ROLE_UP_PROJ
    if "down_proj" in lower or lower.endswith(".w2.weight"):
        return TENSOR_ROLE_DOWN_PROJ

    return TENSOR_ROLE_UNKNOWN


def load_weight_map(model_dir: Path, index_path: Path | None) -> tuple[dict[str, str], dict[str, Any]]:
    if index_path is None:
        candidates = sorted(model_dir.glob("*.safetensors.index.json"))
        if not candidates:
            candidates = sorted(model_dir.glob("model.safetensors.index.json"))
        index_path = candidates[0] if candidates else None

    if index_path is not None and index_path.exists():
        data = json.loads(index_path.read_text(encoding="utf-8-sig"))
        weight_map = data.get("weight_map")
        if not isinstance(weight_map, dict):
            raise ValueError(f"{index_path} does not contain a valid weight_map")
        return {str(k): str(v) for k, v in weight_map.items()}, data.get("metadata", {})

    # Fallback for single-file checkpoints or local test fixtures. Requires the
    # safetensors package to list keys.
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise SystemExit(
            "safetensors is required when no model.safetensors.index.json is present"
        ) from exc

    weight_map: dict[str, str] = {}
    for shard in sorted(model_dir.glob("*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as sf:
            for key in sf.keys():
                weight_map[key] = shard.name
    if not weight_map:
        raise FileNotFoundError(f"no safetensors shards found in {model_dir}")
    return weight_map, {}


def load_hf_config(model_dir: Path, config_path: Path | None) -> dict[str, Any]:
    if config_path is None:
        config_path = model_dir / "config.json"
    if not config_path.exists():
        return {}
    return json.loads(config_path.read_text(encoding="utf-8-sig"))


def first_int(config: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = config.get(key)
        if isinstance(value, int):
            return value
    return None


def build_conversion_plan(weight_map: dict[str, str]) -> ConversionPlan:
    plan = ConversionPlan()
    for name, filename in sorted(weight_map.items()):
        ref = classify_tensor(name, filename)
        if ref.role == "dense" and ref.layer_id is not None:
            plan.dense_by_layer.setdefault(ref.layer_id, []).append(ref)
        elif ref.role == "expert" and ref.layer_id is not None and ref.expert_id is not None:
            plan.experts_by_layer.setdefault(ref.layer_id, {}).setdefault(ref.expert_id, []).append(ref)
        elif ref.role == "global":
            plan.global_tensors.append(ref)
        else:
            plan.skipped_tensors.append(ref)
    return plan


def import_tensor_stack():
    try:
        import numpy as np
        import torch
        from safetensors import safe_open
    except ImportError as exc:
        raise SystemExit(
            "convert_safetensors.py requires: pip install safetensors torch numpy"
        ) from exc
    return np, torch, safe_open


class TensorReader:
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.np, self.torch, self.safe_open = import_tensor_stack()
        self._current_name: str | None = None
        self._current_handle = None

    def close(self) -> None:
        self._current_handle = None
        self._current_name = None

    def tensor(self, ref: TensorRef):
        if self._current_name != ref.filename:
            self.close()
            shard = self.model_dir / ref.filename
            self._current_handle = self.safe_open(shard, framework="pt", device="cpu")
            self._current_name = ref.filename
        return self._current_handle.get_tensor(ref.name)


def tensor_runtime_bytes(tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def dtype_name(tensor) -> str:
    text = str(tensor.dtype).replace("torch.", "")
    return text


def quantize_tensor(tensor, quant_format: int) -> tuple[bytes, float, float]:
    # Converter-time quantization is intentionally simple and deterministic.
    # Future kernels can replace this with grouped AWQ/GPTQ-style quantization.
    import torch

    data = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous().view(-1)
    if data.numel() == 0:
        return b"", 1.0, 0.0

    max_abs = float(torch.max(torch.abs(data)).item())
    if not math.isfinite(max_abs) or max_abs == 0.0:
        max_abs = 1.0

    if quant_format == QUANT_INT8:
        scale = max_abs / 127.0
        q = torch.clamp(torch.round(data / scale), -127, 127).to(torch.int8)
        return q.numpy().tobytes(order="C"), float(scale), 0.0

    if quant_format == QUANT_INT4:
        scale = max_abs / 7.0
        q = torch.clamp(torch.round(data / scale), -8, 7).to(torch.int16)
        q = (q + 8).to(torch.uint8)
        if q.numel() % 2:
            q = torch.cat([q, torch.zeros(1, dtype=torch.uint8)])
        low = q[0::2]
        high = q[1::2] << 4
        packed = (low | high).contiguous()
        return packed.numpy().tobytes(order="C"), float(scale), 8.0

    raise ValueError(f"unsupported quant format: {quant_format}")


def pack_tensor_name(name: str) -> bytes:
    raw = name.encode("utf-8")
    if len(raw) > 65535:
        raise ValueError(f"tensor name is too long: {name}")
    return struct.pack("<H", len(raw)) + raw


def write_block_payload(
    refs: list[TensorRef],
    reader: TensorReader,
    block_tmp: Path,
    quant_format: int,
) -> tuple[int, int, int]:
    records: list[BlockTensorRecord] = []
    names_blob = bytearray()

    with block_tmp.open("wb") as out:
        out.write(b"\0" * BLOCK_HEADER_STRUCT.size)

        # Reserve worst-case record table first. Names are variable-size and are
        # emitted before tensor bytes so the block remains self-contained.
        record_table_offset = out.tell()
        out.write(b"\0" * (len(refs) * TENSOR_RECORD_STRUCT.size))

        name_offsets: list[int] = []
        for ref in refs:
            name_offsets.append(len(names_blob))
            names_blob += pack_tensor_name(ref.name)
        names_offset = out.tell()
        out.write(names_blob)

        for ref, name_offset in zip(refs, name_offsets):
            tensor = reader.tensor(ref)
            original_dtype = dtype_name(tensor)
            shape = tuple(int(dim) for dim in tensor.shape)
            runtime_bytes = tensor_runtime_bytes(tensor)
            quantized, scale, zero_point = quantize_tensor(tensor, quant_format)
            data_offset = out.tell()
            out.write(quantized)

            records.append(
                BlockTensorRecord(
                    name=ref.name,
                    dtype_original=original_dtype,
                    shape=shape,
                    quant_format=quant_format,
                    data_offset=data_offset,
                    data_bytes=len(quantized),
                    runtime_bytes=runtime_bytes,
                    scale=scale,
                    zero_point=zero_point,
                    tensor_role=tensor_role_code(ref.name),
                )
            )

        out.seek(0)
        out.write(
            BLOCK_HEADER_STRUCT.pack(
                BLOCK_MAGIC,
                1,
                len(records),
                quant_format,
                0,
                record_table_offset,
                names_offset,
            )
        )

        out.seek(record_table_offset)
        for record, name_offset in zip(records, name_offsets):
            shape_offset = append_shape_blob(out, record.shape)
            out.write(
                TENSOR_RECORD_STRUCT.pack(
                    min(DTYPE_CODES.get(record.dtype_original, 0), 65535),
                    record.quant_format,
                    len(record.shape),
                    record.tensor_role,
                    name_offset,
                    shape_offset,
                    record.data_offset,
                    record.data_bytes,
                    record.scale,
                    record.zero_point,
                )
            )

        out.seek(0, os.SEEK_END)
        payload_size = out.tell()

    return payload_size, sum(record.runtime_bytes for record in records), checksum64_file(block_tmp)


def append_shape_blob(handle, shape: tuple[int, ...]) -> int:
    # Shape blobs are appended after the tensor data. The record stores an offset
    # to a compact [rank:u32, dims:u64*rank] payload. This keeps the fixed record
    # size small while supporting arbitrary tensor ranks.
    current = handle.tell()
    end = handle.seek(0, os.SEEK_END)
    handle.write(struct.pack("<I", len(shape)))
    for dim in shape:
        handle.write(struct.pack("<Q", dim))
    handle.seek(current)
    return end


def copy_aligned_block(out, block_tmp: Path) -> tuple[int, int, int]:
    offset = align_up(out.tell())
    write_padding(out, offset)
    payload_bytes = block_tmp.stat().st_size
    with block_tmp.open("rb") as src:
        shutil.copyfileobj(src, out, length=8 * 1024 * 1024)
    disk_bytes = align_up(payload_bytes)
    out.write(b"\0" * (disk_bytes - payload_bytes))
    return offset, disk_bytes, payload_bytes


def write_aligned_file(out_path: Path, block_tmp: Path) -> tuple[int, int]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload_bytes = block_tmp.stat().st_size
    disk_bytes = align_up(payload_bytes)
    with out_path.open("wb") as out:
        with block_tmp.open("rb") as src:
            shutil.copyfileobj(src, out, length=8 * 1024 * 1024)
        out.write(b"\0" * (disk_bytes - payload_bytes))
    return disk_bytes, payload_bytes


def reserve_header_and_manifest(out, manifest_size: int) -> int:
    out.write(b"\0" * ENGINE_HEADER_SIZE)
    manifest_offset = align_up(out.tell())
    write_padding(out, manifest_offset)
    out.write(b"\0" * manifest_size)
    return manifest_offset


def pack_manifest(layer_plans: list[LayerPlan], expert_plans: list[ExpertPlan], quant_format: int) -> bytes:
    layer_count = len(layer_plans)
    expert_count = len(expert_plans)
    layer_desc_offset = MANIFEST_HEADER_SIZE
    expert_desc_offset = layer_desc_offset + layer_count * LAYER_DESC_SIZE
    tensor_desc_offset = expert_desc_offset + expert_count * EXPERT_DESC_SIZE

    payload = bytearray()
    payload += MANIFEST_HEADER_STRUCT.pack(
        layer_count,
        expert_count,
        0,
        0,
        layer_desc_offset,
        expert_desc_offset,
        tensor_desc_offset,
    )

    for layer in layer_plans:
        payload += LAYER_DESC_STRUCT.pack(
            layer.layer_id,
            layer.block_type,
            layer.dense_offset,
            layer.dense_disk_bytes,
            layer.dense_payload_bytes,
            layer.dense_dequant_bytes,
            0,
            0,
            layer.first_expert_index,
            layer.num_experts,
            quant_format,
            CHECKSUM_BLAKE2B64,
            layer.checksum,
        )

    for expert in expert_plans:
        payload += EXPERT_DESC_STRUCT.pack(
            expert.layer_id,
            expert.expert_id,
            expert.disk_offset,
            expert.disk_bytes,
            expert.payload_bytes,
            expert.dequant_bytes,
            expert.quant_format,
            expert.route_rank_hint,
            expert.checksum,
        )

    return bytes(payload)


def write_debug_index(
    path: Path,
    plan: ConversionPlan,
    layer_plans: list[LayerPlan],
    expert_plans: list[ExpertPlan],
) -> None:
    role_counts: dict[str, int] = {}
    for refs in plan.dense_by_layer.values():
        for ref in refs:
            role = TENSOR_ROLE_NAMES.get(tensor_role_code(ref.name), "unknown")
            role_counts[role] = role_counts.get(role, 0) + 1
    for experts in plan.experts_by_layer.values():
        for refs in experts.values():
            for ref in refs:
                role = TENSOR_ROLE_NAMES.get(tensor_role_code(ref.name), "unknown")
                role_counts[role] = role_counts.get(role, 0) + 1
    for ref in plan.global_tensors:
        role = TENSOR_ROLE_NAMES.get(tensor_role_code(ref.name), "unknown")
        role_counts[role] = role_counts.get(role, 0) + 1

    data = {
        "tensor_role_encoding": TENSOR_ROLE_NAMES,
        "tensor_role_counts": dict(sorted(role_counts.items())),
        "layers": [
            {
                "layer_id": layer.layer_id,
                "dense_offset": layer.dense_offset,
                "dense_disk_bytes": layer.dense_disk_bytes,
                "num_experts": layer.num_experts,
            }
            for layer in layer_plans
        ],
        "experts": [
            {
                "layer_id": expert.layer_id,
                "expert_id": expert.expert_id,
                "disk_offset": expert.disk_offset,
                "disk_bytes": expert.disk_bytes,
                "payload_bytes": expert.payload_bytes,
            }
            for expert in expert_plans
        ],
        "global_tensors": [ref.name for ref in plan.global_tensors],
        "skipped_tensors": [ref.name for ref in plan.skipped_tensors],
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def parse_layer_range(value: str | None) -> tuple[int, int] | None:
    if not value:
        return None
    parts = [part.strip() for part in value.split(",", 1)]
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise argparse.ArgumentTypeError("--layer-range must be START,END")
    start, end = int(parts[0]), int(parts[1])
    if start < 0 or end <= start:
        raise argparse.ArgumentTypeError("--layer-range requires 0 <= START < END")
    return start, end


def layers_for_range(num_layers: int, layer_range: tuple[int, int] | None) -> list[int]:
    if layer_range is None:
        return list(range(num_layers))
    start, end = layer_range
    if start >= num_layers:
        raise ValueError(f"layer-range start {start} is outside model layer count {num_layers}")
    return list(range(start, min(end, num_layers)))


def sharded_manifest_path(core_path: Path, explicit: Path | None) -> Path:
    return explicit or core_path.with_suffix(".shards.json")


def write_sharded_index(
    path: Path,
    *,
    core_path: Path,
    experts_dir: Path,
    selected_layers: list[int],
    num_layers_total: int,
    experts_per_layer: int,
    layer_plans: list[LayerPlan],
    expert_shards: list[ExpertShardPlan],
    quant_format: int,
    metadata: dict[str, Any],
) -> None:
    data = {
        "format": "wohper-sharded-experts",
        "version": 1,
        "core_file": str(core_path),
        "experts_dir": str(experts_dir),
        "layer_range": [selected_layers[0], selected_layers[-1] + 1] if selected_layers else [0, 0],
        "num_layers_total": num_layers_total,
        "layers_in_core": selected_layers,
        "experts_per_layer": experts_per_layer,
        "alignment": ALIGN_2MB,
        "quant_format": quant_format,
        "dense_layers": [layer.__dict__ for layer in layer_plans],
        "experts": [expert.__dict__ for expert in expert_shards],
        "remote_fetch": {
            "enabled": False,
            "endpoint_template": "",
            "path_template": "experts/layer{layer_id}_expert{expert_id}.zcblk",
        },
        "cache": {
            "default_dir": "cache/experts",
            "default_max_bytes": 100 * 1024**3,
        },
        "metadata": metadata,
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def infer_arch_config(plan: ConversionPlan, explicit_layers: int | None, explicit_experts: int | None) -> tuple[int, int]:
    if explicit_layers is not None:
        num_layers = explicit_layers
    else:
        all_layers = set(plan.dense_by_layer) | set(plan.experts_by_layer)
        if not all_layers:
            raise ValueError("could not infer layer count from tensor names; pass --num-layers")
        num_layers = max(all_layers) + 1

    if explicit_experts is not None:
        experts_per_layer = explicit_experts
    else:
        counts = [len(experts) for experts in plan.experts_by_layer.values()]
        experts_per_layer = max(counts) if counts else 0
    if explicit_experts is None and experts_per_layer <= 0:
        raise ValueError("could not infer experts_per_layer; pass --experts-per-layer")
    if experts_per_layer < 0:
        raise ValueError("experts_per_layer must be >= 0")
    return num_layers, experts_per_layer


def print_plan_summary(
    plan: ConversionPlan,
    num_layers: int,
    experts_per_layer: int,
    config: dict[str, Any],
) -> None:
    dense_tensors = sum(len(items) for items in plan.dense_by_layer.values())
    expert_tensors = sum(
        len(items)
        for experts in plan.experts_by_layer.values()
        for items in experts.values()
    )
    print("conversion_plan=ok")
    print(f"layers={num_layers}")
    print(f"experts_per_layer={experts_per_layer}")
    print(f"dense_layers_detected={len(plan.dense_by_layer)}")
    print(f"expert_layers_detected={len(plan.experts_by_layer)}")
    print(f"dense_tensors={dense_tensors}")
    print(f"expert_tensors={expert_tensors}")
    print(f"global_tensors={len(plan.global_tensors)}")
    print(f"skipped_tensors={len(plan.skipped_tensors)}")
    if config:
        print("config_keys=" + ",".join(sorted(config.keys())[:32]))


def convert(args: argparse.Namespace) -> None:
    quant_format = QUANT_INT8 if args.quant == "int8" else QUANT_INT4
    if args.sharded_experts and args.out == Path("MODEL.bin"):
        args.out = Path("dense_core.bin")
    model_dir = args.model_dir.resolve()
    weight_map, metadata = load_weight_map(model_dir, args.index)
    config = load_hf_config(model_dir, args.config)
    plan = build_conversion_plan(weight_map)
    num_layers, experts_per_layer = infer_arch_config(
        plan,
        args.num_layers
        if args.num_layers is not None
        else first_int(config, "num_hidden_layers", "n_layer", "num_layers"),
        args.experts_per_layer
        if args.experts_per_layer is not None
        else first_int(
            config,
            "n_routed_experts",
            "num_experts",
            "num_local_experts",
            "moe_num_experts",
            "experts_per_layer",
        ),
    )
    args.hidden_size = args.hidden_size or first_int(config, "hidden_size", "n_embd", "dim") or 0
    args.heads = args.heads or first_int(config, "num_attention_heads", "n_head") or 0
    args.kv_heads = args.kv_heads or first_int(
        config,
        "num_key_value_heads",
        "multi_query_group_num",
        "num_kv_heads",
    ) or 0

    if args.plan_only:
        print_plan_summary(plan, num_layers, experts_per_layer, config)
        return

    selected_layers = layers_for_range(num_layers, args.layer_range)
    if args.sharded_experts:
        convert_sharded_experts(
            args,
            model_dir,
            plan,
            metadata,
            selected_layers,
            num_layers,
            experts_per_layer,
            quant_format,
        )
        return

    if args.layer_range is not None:
        print(
            "warning: --layer-range without --sharded-experts still writes one MODEL.bin "
            "containing only the selected layer range",
            file=sys.stderr,
        )
    convert_single_model(
        args,
        model_dir,
        plan,
        metadata,
        selected_layers,
        num_layers,
        experts_per_layer,
        quant_format,
    )


def convert_single_model(
    args: argparse.Namespace,
    model_dir: Path,
    plan: ConversionPlan,
    metadata: dict[str, Any],
    selected_layers: list[int],
    num_layers_total: int,
    experts_per_layer: int,
    quant_format: int,
) -> None:
    total_experts = len(selected_layers) * experts_per_layer
    manifest_size = (
        MANIFEST_HEADER_SIZE
        + len(selected_layers) * LAYER_DESC_SIZE
        + total_experts * EXPERT_DESC_SIZE
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    layer_plans: list[LayerPlan] = []
    expert_plans: list[ExpertPlan] = []

    reader = TensorReader(model_dir)
    with tempfile.TemporaryDirectory(prefix="zc_convert_") as tmp_dir_text:
        tmp_dir = Path(tmp_dir_text)
        with args.out.open("wb") as out:
            manifest_offset = reserve_header_and_manifest(out, manifest_size)

            for layer_id in selected_layers:
                dense_refs = list(plan.dense_by_layer.get(layer_id, []))
                if layer_id == 0 and args.pack_global_into_layer0:
                    dense_refs = list(plan.global_tensors) + dense_refs
                if not dense_refs:
                    raise ValueError(f"layer {layer_id} has no dense tensors")

                dense_tmp = tmp_dir / f"dense_layer_{layer_id}.bin"
                _, dense_dequant, dense_checksum = write_block_payload(
                    dense_refs,
                    reader,
                    dense_tmp,
                    quant_format,
                )
                dense_offset, dense_disk, dense_payload = copy_aligned_block(out, dense_tmp)

                first_expert_index = len(expert_plans)
                for expert_id in range(experts_per_layer):
                    refs = plan.experts_by_layer.get(layer_id, {}).get(expert_id, [])
                    if not refs:
                        if args.allow_missing_experts:
                            refs = []
                        else:
                            raise ValueError(f"missing tensors for layer {layer_id} expert {expert_id}")

                    expert_tmp = tmp_dir / f"expert_layer_{layer_id}_{expert_id}.bin"
                    _, expert_dequant, expert_checksum = write_block_payload(
                        refs,
                        reader,
                        expert_tmp,
                        quant_format,
                    )
                    expert_offset, expert_disk, expert_payload = copy_aligned_block(out, expert_tmp)
                    expert_plans.append(
                        ExpertPlan(
                            layer_id=layer_id,
                            expert_id=expert_id,
                            disk_offset=expert_offset,
                            disk_bytes=expert_disk,
                            payload_bytes=expert_payload,
                            dequant_bytes=expert_dequant,
                            quant_format=quant_format,
                            route_rank_hint=expert_id,
                            checksum=expert_checksum,
                        )
                    )

                layer_plans.append(
                    LayerPlan(
                        layer_id=layer_id,
                        dense_offset=dense_offset,
                        dense_disk_bytes=dense_disk,
                        dense_payload_bytes=dense_payload,
                        dense_dequant_bytes=dense_dequant,
                        first_expert_index=first_expert_index,
                        num_experts=experts_per_layer,
                        checksum=dense_checksum,
                    )
                )

            file_size = out.tell()
            manifest_payload = pack_manifest(layer_plans, expert_plans, quant_format)
            manifest_checksum = checksum64_bytes(manifest_payload)

            header = ENGINE_HEADER_STRUCT.pack(
                MODEL_MAGIC,
                FORMAT_VERSION,
                0,
                file_size,
                manifest_offset,
                len(manifest_payload),
                0,
                0,
                0,
                0,
                args.model_family,
                1,
                len(selected_layers),
                args.hidden_size,
                args.heads,
                args.kv_heads,
                experts_per_layer,
                args.active_experts,
                ALIGN_2MB,
                quant_format,
                manifest_checksum,
                0,
            )

            out.seek(0)
            out.write(header)
            out.seek(manifest_offset)
            out.write(manifest_payload)

    reader.close()

    validate_output(args.out, layer_plans, expert_plans, manifest_offset)
    debug_path = args.index_out or args.out.with_suffix(".index.json")
    write_debug_index(debug_path, plan, layer_plans, expert_plans)
    print_summary(args.out, layer_plans, expert_plans, manifest_offset, manifest_size, metadata)


def convert_sharded_experts(
    args: argparse.Namespace,
    model_dir: Path,
    plan: ConversionPlan,
    metadata: dict[str, Any],
    selected_layers: list[int],
    num_layers_total: int,
    experts_per_layer: int,
    quant_format: int,
) -> None:
    if not selected_layers:
        raise ValueError("no layers selected for sharded conversion")

    core_path = args.out
    experts_dir = args.experts_dir or (core_path.parent / "experts")
    shard_index = sharded_manifest_path(core_path, args.shard_index_out)
    total_experts = len(selected_layers) * experts_per_layer
    manifest_size = (
        MANIFEST_HEADER_SIZE
        + len(selected_layers) * LAYER_DESC_SIZE
        + total_experts * EXPERT_DESC_SIZE
    )

    core_path.parent.mkdir(parents=True, exist_ok=True)
    experts_dir.mkdir(parents=True, exist_ok=True)
    layer_plans: list[LayerPlan] = []
    expert_plans: list[ExpertPlan] = []
    expert_shards: list[ExpertShardPlan] = []

    reader = TensorReader(model_dir)
    with tempfile.TemporaryDirectory(prefix="zc_convert_") as tmp_dir_text:
        tmp_dir = Path(tmp_dir_text)
        with core_path.open("wb") as out:
            manifest_offset = reserve_header_and_manifest(out, manifest_size)

            for layer_id in selected_layers:
                dense_refs = list(plan.dense_by_layer.get(layer_id, []))
                if layer_id == 0 and args.pack_global_into_layer0:
                    dense_refs = list(plan.global_tensors) + dense_refs
                if not dense_refs:
                    raise ValueError(f"layer {layer_id} has no dense tensors")

                dense_tmp = tmp_dir / f"dense_layer_{layer_id}.bin"
                _, dense_dequant, dense_checksum = write_block_payload(
                    dense_refs,
                    reader,
                    dense_tmp,
                    quant_format,
                )
                dense_offset, dense_disk, dense_payload = copy_aligned_block(out, dense_tmp)

                first_expert_index = len(expert_plans)
                for expert_id in range(experts_per_layer):
                    refs = plan.experts_by_layer.get(layer_id, {}).get(expert_id, [])
                    if not refs:
                        if args.allow_missing_experts:
                            refs = []
                        else:
                            raise ValueError(f"missing tensors for layer {layer_id} expert {expert_id}")

                    expert_tmp = tmp_dir / f"expert_layer_{layer_id}_{expert_id}.bin"
                    _, expert_dequant, expert_checksum = write_block_payload(
                        refs,
                        reader,
                        expert_tmp,
                        quant_format,
                    )
                    expert_path = experts_dir / f"layer{layer_id}_expert{expert_id}.zcblk"
                    try:
                        expert_rel = expert_path.relative_to(core_path.parent)
                    except ValueError:
                        expert_rel = expert_path
                    expert_disk, expert_payload = write_aligned_file(expert_path, expert_tmp)

                    # External experts deliberately use disk_offset=0 because the
                    # Rust sidecar shard map resolves the path at runtime.
                    expert_plans.append(
                        ExpertPlan(
                            layer_id=layer_id,
                            expert_id=expert_id,
                            disk_offset=0,
                            disk_bytes=expert_disk,
                            payload_bytes=expert_payload,
                            dequant_bytes=expert_dequant,
                            quant_format=quant_format,
                            route_rank_hint=expert_id,
                            checksum=expert_checksum,
                        )
                    )
                    expert_shards.append(
                        ExpertShardPlan(
                            layer_id=layer_id,
                            expert_id=expert_id,
                            path=str(expert_rel).replace("\\", "/"),
                            disk_bytes=expert_disk,
                            payload_bytes=expert_payload,
                            dequant_bytes=expert_dequant,
                            quant_format=quant_format,
                            checksum=expert_checksum,
                        )
                    )

                layer_plans.append(
                    LayerPlan(
                        layer_id=layer_id,
                        dense_offset=dense_offset,
                        dense_disk_bytes=dense_disk,
                        dense_payload_bytes=dense_payload,
                        dense_dequant_bytes=dense_dequant,
                        first_expert_index=first_expert_index,
                        num_experts=experts_per_layer,
                        checksum=dense_checksum,
                    )
                )

            file_size = out.tell()
            manifest_payload = pack_manifest(layer_plans, expert_plans, quant_format)
            manifest_checksum = checksum64_bytes(manifest_payload)

            header = ENGINE_HEADER_STRUCT.pack(
                MODEL_MAGIC,
                FORMAT_VERSION,
                0,
                file_size,
                manifest_offset,
                len(manifest_payload),
                0,
                0,
                0,
                0,
                args.model_family,
                2,
                len(selected_layers),
                args.hidden_size,
                args.heads,
                args.kv_heads,
                experts_per_layer,
                args.active_experts,
                ALIGN_2MB,
                quant_format,
                manifest_checksum,
                0,
            )

            out.seek(0)
            out.write(header)
            out.seek(manifest_offset)
            out.write(manifest_payload)

    reader.close()

    validate_output(core_path, layer_plans, [], manifest_offset)
    debug_path = args.index_out or core_path.with_suffix(".index.json")
    write_debug_index(debug_path, plan, layer_plans, expert_plans)
    write_sharded_index(
        shard_index,
        core_path=core_path,
        experts_dir=experts_dir,
        selected_layers=selected_layers,
        num_layers_total=num_layers_total,
        experts_per_layer=experts_per_layer,
        layer_plans=layer_plans,
        expert_shards=expert_shards,
        quant_format=quant_format,
        metadata=metadata,
    )
    print_sharded_summary(core_path, experts_dir, shard_index, layer_plans, expert_shards, metadata)


def validate_output(
    out_path: Path,
    layers: list[LayerPlan],
    experts: list[ExpertPlan],
    manifest_offset: int,
) -> None:
    if manifest_offset % ALIGN_2MB:
        raise AssertionError("manifest_offset is not 2MB-aligned")
    for layer in layers:
        if layer.dense_offset % ALIGN_2MB or layer.dense_disk_bytes % ALIGN_2MB:
            raise AssertionError(f"layer {layer.layer_id} dense block is not 2MB-aligned")
    for expert in experts:
        if expert.disk_offset % ALIGN_2MB or expert.disk_bytes % ALIGN_2MB:
            raise AssertionError(
                f"layer {expert.layer_id} expert {expert.expert_id} block is not 2MB-aligned"
            )
    if out_path.stat().st_size % ALIGN_2MB:
        raise AssertionError("MODEL.bin file size is not 2MB-aligned")


def print_summary(
    out_path: Path,
    layers: list[LayerPlan],
    experts: list[ExpertPlan],
    manifest_offset: int,
    manifest_size: int,
    metadata: dict[str, Any],
) -> None:
    size = out_path.stat().st_size
    print(f"MODEL.bin: {out_path}")
    print(f"size: {size:,} bytes ({size / (1024 ** 3):.3f} GiB)")
    print(f"alignment: {ALIGN_2MB:,} bytes")
    print(f"manifest_offset: {manifest_offset:,}")
    print(f"manifest_size: {manifest_size:,}")
    print(f"layers: {len(layers)}")
    print(f"experts: {len(experts)}")
    print(f"first_dense_offset: {layers[0].dense_offset:,}")
    print(f"first_expert_offset: {experts[0].disk_offset:,}")
    if metadata:
        print(f"hf_metadata_keys: {','.join(sorted(metadata.keys()))}")


def print_sharded_summary(
    core_path: Path,
    experts_dir: Path,
    shard_index: Path,
    layers: list[LayerPlan],
    expert_shards: list[ExpertShardPlan],
    metadata: dict[str, Any],
) -> None:
    core_size = core_path.stat().st_size
    expert_bytes = sum(shard.disk_bytes for shard in expert_shards)
    print(f"dense_core.bin: {core_path}")
    print(f"core_size: {core_size:,} bytes ({core_size / (1024 ** 3):.3f} GiB)")
    print(f"experts_dir: {experts_dir}")
    print(f"expert_files: {len(expert_shards)}")
    print(f"expert_disk_total: {expert_bytes:,} bytes ({expert_bytes / (1024 ** 3):.3f} GiB)")
    print(f"shard_index: {shard_index}")
    print(f"layers: {len(layers)}")
    print(f"first_dense_offset: {layers[0].dense_offset:,}")
    if expert_shards:
        print(f"first_expert_file: {expert_shards[0].path}")
    if metadata:
        print(f"hf_metadata_keys: {','.join(sorted(metadata.keys()))}")


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert HF safetensors MoE checkpoints into Wohper MODEL.bin"
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--index", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("MODEL.bin"))
    parser.add_argument("--index-out", type=Path, default=None)
    parser.add_argument("--shard-index-out", type=Path, default=None)
    parser.add_argument("--experts-dir", type=Path, default=None)
    parser.add_argument("--quant", choices=("int4", "int8"), default="int4")
    parser.add_argument(
        "--sharded-experts",
        action="store_true",
        help="Write dense_core.bin plus experts/layerN_expertM.zcblk files instead of one monolithic MODEL.bin.",
    )
    parser.add_argument(
        "--layer-range",
        type=parse_layer_range,
        default=None,
        metavar="START,END",
        help="Convert only layers START..END-1 for cluster/pipeline sharding.",
    )
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--experts-per-layer", type=int, default=None)
    parser.add_argument("--active-experts", type=int, default=2)
    parser.add_argument("--hidden-size", type=int, default=0)
    parser.add_argument("--heads", type=int, default=0)
    parser.add_argument("--kv-heads", type=int, default=0)
    parser.add_argument("--model-family", type=int, default=1)
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Only parse index/config and print grouping summary; does not import torch or read tensor data.",
    )
    parser.add_argument(
        "--pack-global-into-layer0",
        action="store_true",
        help="Pack embeddings/output/router globals into layer 0 dense block for first-pass conversion.",
    )
    parser.add_argument(
        "--allow-missing-experts",
        action="store_true",
        help="Emit empty expert blocks when a layer/expert id has no matching tensors.",
    )
    return parser.parse_args(list(argv))


def main(argv: Iterable[str] = sys.argv[1:]) -> int:
    args = parse_args(argv)
    convert(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
