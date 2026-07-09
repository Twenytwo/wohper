#!/usr/bin/env python3
"""Run a bounded single-token DeepSeek-V4 L0 math smoke over raw Wohper shards.

This is a safety/runtime smoke, not a Hugging Face parity trace. It exercises
real converted tensors with chunked reads/dequantization and uses the exact
single-key attention reduction implied by the local DeepSeek reference.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import time
from pathlib import Path
from typing import Any

import numpy as np

import convert_safetensors as zc


BLOCK = 128
_FP8_LUT_CACHE: tuple[np.ndarray, np.ndarray] | None = None
_FP4_LUT_CACHE: np.ndarray | None = None


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_core(index_path: Path, core_file: str) -> Path:
    core = Path(core_file)
    if core.is_absolute() or core.exists():
        return core
    return index_path.parent / core


def read_at(path: Path, offset: int, size: int) -> bytes:
    with path.open("rb") as handle:
        handle.seek(offset)
        data = handle.read(size)
    if len(data) != size:
        raise ValueError(f"{path}: short read at {offset} size {size}")
    return data


def bf16_to_f32(raw: bytes) -> np.ndarray:
    u32 = np.frombuffer(raw, dtype="<u2").astype("<u4") << 16
    return u32.view("<f4").astype(np.float32, copy=False)


def f32_from_raw(raw: bytes) -> np.ndarray:
    return np.frombuffer(raw, dtype="<f4").astype(np.float32, copy=False)


def i64_from_raw(raw: bytes) -> list[int]:
    return [int(item[0]) for item in struct.iter_unpack("<q", raw)]


def read_tensor(core: Path, tensor: dict[str, Any]) -> bytes:
    return read_at(core, int(tensor["absolute_data_offset"]), int(tensor["data_bytes"]))


def read_row(core: Path, tensor: dict[str, Any], row: int) -> bytes:
    row_bytes = int(tensor["row_bytes"])
    row_count = int(tensor["row_count"])
    if row < 0 or row >= row_count:
        raise ValueError(f"{tensor['name']}: row {row} outside {row_count}")
    offset = int(tensor["absolute_data_offset"]) + row * row_bytes
    return read_at(core, offset, row_bytes)


def decode_ue8m0_array(raw: bytes) -> np.ndarray:
    items = np.frombuffer(raw, dtype=np.uint8)
    out = np.zeros(items.shape, dtype=np.float32)
    nonzero = items != 0
    out[nonzero] = np.exp2(items[nonzero].astype(np.int16) - 127).astype(np.float32)
    return out


def decode_fp8_e4m3_array(raw: np.ndarray) -> np.ndarray:
    b = raw.astype(np.uint8, copy=False)
    sign = np.where((b & 0x80) != 0, -1.0, 1.0).astype(np.float32)
    exponent = ((b >> 3) & 0x0F).astype(np.int16)
    mantissa = (b & 0x07).astype(np.float32)
    subnormal = (mantissa / 8.0) * np.float32(2.0**-6)
    normal = (1.0 + mantissa / 8.0) * np.exp2(exponent - 7).astype(np.float32)
    values = np.where(exponent == 0, subnormal, normal).astype(np.float32)
    values[(exponent == 0) & (mantissa == 0.0)] = 0.0
    return sign * values


def fp8_lut() -> tuple[np.ndarray, np.ndarray]:
    global _FP8_LUT_CACHE
    if _FP8_LUT_CACHE is None:
        values = decode_fp8_e4m3_array(np.arange(256, dtype=np.uint8))
        order = np.argsort(values)
        _FP8_LUT_CACHE = values[order], order.astype(np.uint8)
    return _FP8_LUT_CACHE


def quant_dequant_fp8_activation(x: np.ndarray, block_size: int = BLOCK, round_power2: bool = True) -> np.ndarray:
    original_shape = x.shape
    rows = x.reshape((-1, original_shape[-1])).astype(np.float32, copy=False)
    if rows.shape[1] % block_size != 0:
        raise ValueError(f"activation width {rows.shape[1]} must be divisible by {block_size}")
    lut_values, _ = fp8_lut()
    out = np.empty_like(rows, dtype=np.float32)
    for r in range(rows.shape[0]):
        for c0 in range(0, rows.shape[1], block_size):
            c1 = c0 + block_size
            block = rows[r, c0:c1]
            amax = max(float(np.max(np.abs(block))), 1e-4)
            scale = amax / 448.0
            if round_power2:
                scale = float(2.0 ** math.ceil(math.log2(scale)))
            scaled = np.clip(block / np.float32(scale), -448.0, 448.0)
            idx = np.searchsorted(lut_values, scaled)
            idx0 = np.clip(idx - 1, 0, lut_values.size - 1)
            idx1 = np.clip(idx, 0, lut_values.size - 1)
            choose_hi = np.abs(lut_values[idx1] - scaled) < np.abs(lut_values[idx0] - scaled)
            nearest = np.where(choose_hi, lut_values[idx1], lut_values[idx0])
            out[r, c0:c1] = nearest.astype(np.float32) * np.float32(scale)
    return out.reshape(original_shape)


def quant_dequant_kv_nonrope(x: np.ndarray, rope_head_dim: int = 64) -> np.ndarray:
    out = x.astype(np.float32, copy=True)
    if out.shape[-1] <= rope_head_dim:
        return out
    out[..., :-rope_head_dim] = quant_dequant_fp8_activation(out[..., :-rope_head_dim], block_size=64)
    return out


def decode_fp4_e2m1_array(nibbles: np.ndarray) -> np.ndarray:
    global _FP4_LUT_CACHE
    if _FP4_LUT_CACHE is None:
        n = np.arange(16, dtype=np.uint8)
        sign = np.where((n & 0x08) != 0, -1.0, 1.0).astype(np.float32)
        exponent = ((n >> 1) & 0x03).astype(np.int16)
        mantissa = (n & 0x01).astype(np.float32)
        subnormal = (mantissa / 2.0) * np.float32(2.0**-2)
        normal = (1.0 + mantissa / 2.0) * np.exp2(exponent - 1).astype(np.float32)
        values = np.where(exponent == 0, subnormal, normal).astype(np.float32)
        values[(exponent == 0) & (mantissa == 0.0)] = 0.0
        _FP4_LUT_CACHE = (sign * values).astype(np.float32)
    return _FP4_LUT_CACHE[nibbles & 0x0F]


def quant_dequant_fp4_activation(x: np.ndarray, block_size: int = 32) -> np.ndarray:
    original_shape = x.shape
    rows = x.reshape((-1, original_shape[-1])).astype(np.float32, copy=False)
    if rows.shape[1] % block_size != 0:
        raise ValueError(f"activation width {rows.shape[1]} must be divisible by {block_size}")
    fp4_values = decode_fp4_e2m1_array(np.arange(16, dtype=np.uint8))
    order = np.argsort(fp4_values)
    lut = fp4_values[order]
    out = np.empty_like(rows, dtype=np.float32)
    for r in range(rows.shape[0]):
        for c0 in range(0, rows.shape[1], block_size):
            c1 = c0 + block_size
            block = rows[r, c0:c1]
            amax = max(float(np.max(np.abs(block))), 6.0 * (2.0 ** -126))
            scale = float(2.0 ** math.ceil(math.log2(amax / 6.0)))
            scaled = np.clip(block / np.float32(scale), -6.0, 6.0)
            idx = np.searchsorted(lut, scaled)
            idx0 = np.clip(idx - 1, 0, lut.size - 1)
            idx1 = np.clip(idx, 0, lut.size - 1)
            choose_hi = np.abs(lut[idx1] - scaled) < np.abs(lut[idx0] - scaled)
            nearest = np.where(choose_hi, lut[idx1], lut[idx0])
            out[r, c0:c1] = nearest.astype(np.float32) * np.float32(scale)
    return out.reshape(original_shape)


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    scale = np.float32(1.0 / math.sqrt(float(np.mean(x.astype(np.float32) ** 2)) + eps))
    return (x * scale * weight).astype(np.float32)


def silu(x: np.ndarray) -> np.ndarray:
    return (x / (1.0 + np.exp(-x))).astype(np.float32)


def softplus(x: np.ndarray) -> np.ndarray:
    return np.where(x > 20.0, x, np.log1p(np.exp(x))).astype(np.float32)


def summary(values: np.ndarray, sample: int = 8) -> dict[str, Any]:
    finite = values[np.isfinite(values)]
    return {
        "count": int(values.size),
        "finite_count": int(finite.size),
        "nonzero_count": int(np.count_nonzero(finite)) if finite.size else 0,
        "min": float(np.min(finite)) if finite.size else None,
        "max": float(np.max(finite)) if finite.size else None,
        "l2": float(np.linalg.norm(finite)) if finite.size else None,
        "sample": [float(v) for v in finite[:sample]],
    }


def fp8_matvec(core: Path, tensors: dict[str, dict[str, Any]], name: str, x: np.ndarray) -> tuple[np.ndarray, int]:
    weight = tensors[name]
    scale = tensors[name.replace(".weight", ".scale")]
    rows, cols = [int(v) for v in weight["shape"]]
    if x.size != cols:
        raise ValueError(f"{name}: input width {x.size} != {cols}")
    x = quant_dequant_fp8_activation(x)
    scales = decode_ue8m0_array(read_tensor(core, scale)).reshape((math.ceil(rows / BLOCK), math.ceil(cols / BLOCK)))
    out = np.empty(rows, dtype=np.float32)
    bytes_read = int(scale["data_bytes"])
    base = int(weight["absolute_data_offset"])
    for r0 in range(0, rows, BLOCK):
        r1 = min(r0 + BLOCK, rows)
        row_count = r1 - r0
        raw = read_at(core, base + r0 * cols, row_count * cols)
        bytes_read += len(raw)
        raw_rows = np.frombuffer(raw, dtype=np.uint8).reshape((row_count, cols))
        acc = np.zeros(row_count, dtype=np.float32)
        scale_row = r0 // BLOCK
        for c0 in range(0, cols, BLOCK):
            c1 = min(c0 + BLOCK, cols)
            vals = decode_fp8_e4m3_array(raw_rows[:, c0:c1])
            vals *= scales[scale_row, c0 // BLOCK]
            acc += vals @ x[c0:c1]
        out[r0:r1] = acc
    return out, bytes_read


def fp8_grouped_wo_a(
    core: Path,
    tensors: dict[str, dict[str, Any]],
    name: str,
    grouped_x: np.ndarray,
) -> tuple[np.ndarray, int]:
    weight = tensors[name]
    scale = tensors[name.replace(".weight", ".scale")]
    rows, cols = [int(v) for v in weight["shape"]]
    groups = int(grouped_x.shape[0])
    group_cols = int(grouped_x.shape[1])
    if groups <= 0 or rows % groups != 0:
        raise ValueError(f"{name}: rows {rows} are incompatible with {groups} groups")
    if group_cols != cols:
        raise ValueError(f"{name}: grouped input width {group_cols} != {cols}")
    rows_per_group = rows // groups
    scales = decode_ue8m0_array(read_tensor(core, scale)).reshape((math.ceil(rows / BLOCK), math.ceil(cols / BLOCK)))
    out = np.empty(rows, dtype=np.float32)
    bytes_read = int(scale["data_bytes"])
    base_offset = int(weight["absolute_data_offset"])
    for r0 in range(0, rows, BLOCK):
        r1 = min(r0 + BLOCK, rows)
        row_count = r1 - r0
        group_id = r0 // rows_per_group
        raw = read_at(core, base_offset + r0 * cols, row_count * cols)
        bytes_read += len(raw)
        raw_rows = np.frombuffer(raw, dtype=np.uint8).reshape((row_count, cols))
        acc = np.zeros(row_count, dtype=np.float32)
        scale_row = r0 // BLOCK
        x = grouped_x[group_id]
        for c0 in range(0, cols, BLOCK):
            c1 = min(c0 + BLOCK, cols)
            vals = decode_fp8_e4m3_array(raw_rows[:, c0:c1])
            vals *= scales[scale_row, c0 // BLOCK]
            acc += vals @ x[c0:c1]
        out[r0:r1] = acc
    return out, bytes_read


def single_token_attention_out(
    core: Path,
    tensors: dict[str, dict[str, Any]],
    prefix: str,
    q_full: np.ndarray,
    kv_normed: np.ndarray,
) -> tuple[np.ndarray, int, dict[str, Any]]:
    q_heads = q_full.reshape((64, 512)).astype(np.float32, copy=True)
    q_rms = np.sqrt(np.mean(q_heads * q_heads, axis=1, keepdims=True) + np.float32(1e-6))
    q_heads = (q_heads / q_rms).astype(np.float32)
    kv_normed = quant_dequant_kv_nonrope(kv_normed)
    sink_name = f"{prefix}.attn_sink"
    sink = f32_from_raw(read_tensor(core, tensors[sink_name]))
    scores = (q_heads @ kv_normed.astype(np.float32)) * np.float32(512.0 ** -0.5)
    alpha = (1.0 / (1.0 + np.exp(np.clip(sink - scores, -80.0, 80.0)))).astype(np.float32)
    o_heads = (alpha[:, None] * kv_normed[None, :]).astype(np.float32)
    grouped = o_heads.reshape((8, 4096))
    wo_a, b1 = fp8_grouped_wo_a(core, tensors, f"{prefix}.wo_a.weight", grouped)
    attn_out, b2 = fp8_matvec(core, tensors, f"{prefix}.wo_b.weight", wo_a)
    report = {
        "mode": "single_key_sparse_attention_with_attn_sink",
        "q_heads": summary(q_heads),
        "sink": summary(sink),
        "scores": summary(scores),
        "alpha": summary(alpha),
        "wo_a": summary(wo_a),
    }
    return attn_out, int(tensors[sink_name]["data_bytes"]) + b1 + b2, report


def read_name(buffer: bytes, names_offset: int, name_offset: int) -> str:
    cursor = names_offset + name_offset
    (size,) = struct.unpack_from("<H", buffer, cursor)
    cursor += 2
    return buffer[cursor : cursor + size].decode("utf-8")


def read_shape(buffer: bytes, shape_offset: int) -> list[int]:
    (rank,) = struct.unpack_from("<I", buffer, shape_offset)
    cursor = shape_offset + 4
    if rank == 0:
        return []
    return [int(v) for v in struct.unpack_from("<" + "Q" * rank, buffer, cursor)]


def parse_expert_block(path: Path, payload_bytes: int) -> dict[str, dict[str, Any]]:
    buffer = read_at(path, 0, payload_bytes)
    magic, version, tensor_count, block_quant, flags, record_offset, names_offset = zc.BLOCK_HEADER_STRUCT.unpack_from(buffer, 0)
    if magic != zc.BLOCK_MAGIC:
        raise ValueError(f"{path}: invalid ZCBLK magic")
    if version != 1:
        raise ValueError(f"{path}: unsupported ZCBLK version {version}")
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
            "shape": read_shape(buffer, shape_offset),
            "data_offset": data_offset,
            "data_bytes": data_bytes,
            "payload": memoryview(buffer)[data_offset : data_offset + data_bytes],
        }
    return records


def find_suffix(records: dict[str, dict[str, Any]], suffix: str) -> tuple[str, dict[str, Any]]:
    matches = [(name, record) for name, record in records.items() if name.endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(f"expected one tensor ending with {suffix}, found {len(matches)}")
    return matches[0]


def fp4_matvec_from_records(records: dict[str, dict[str, Any]], suffix: str, shape: tuple[int, int], x: np.ndarray) -> np.ndarray:
    rows, cols = shape
    if x.size != cols:
        raise ValueError(f"{suffix}: input width {x.size} != {cols}")
    x = quant_dequant_fp8_activation(x)
    weight_name, weight = find_suffix(records, suffix)
    scale = records.get(weight_name.replace(".weight", ".scale"))
    if scale is None:
        raise ValueError(f"{weight_name}: missing scale tensor")
    packed_shape = [int(v) for v in weight["shape"]]
    scale_shape = [int(v) for v in scale["shape"]]
    if packed_shape != [rows, (cols + 1) // 2]:
        raise ValueError(f"{weight_name}: packed shape {packed_shape} does not match logical {(rows, cols)}")
    if len(scale_shape) != 2 or scale_shape[0] != rows:
        raise ValueError(f"{weight_name}: unsupported scale shape {scale_shape}")
    group_cols = cols // scale_shape[1]
    if group_cols <= 0 or cols % scale_shape[1] != 0:
        raise ValueError(f"{weight_name}: scale shape {scale_shape} is incompatible with {cols} columns")
    scales = decode_ue8m0_array(bytes(scale["payload"])).reshape((scale_shape[0], scale_shape[1]))
    row_bytes = packed_shape[1]
    raw = np.frombuffer(weight["payload"], dtype=np.uint8).reshape((rows, row_bytes))
    out = np.empty(rows, dtype=np.float32)
    for r0 in range(0, rows, BLOCK):
        r1 = min(r0 + BLOCK, rows)
        row_count = r1 - r0
        packed = raw[r0:r1]
        unpacked = np.empty((row_count, cols), dtype=np.uint8)
        unpacked[:, 0::2] = packed[:, : (cols + 1) // 2] & 0x0F
        unpacked[:, 1::2] = packed[:, : cols // 2] >> 4
        acc = np.zeros(row_count, dtype=np.float32)
        for c0 in range(0, cols, group_cols):
            c1 = min(c0 + BLOCK, cols)
            c1 = min(c0 + group_cols, cols)
            vals = decode_fp4_e2m1_array(unpacked[:, c0:c1])
            vals *= scales[r0:r1, c0 // group_cols][:, None]
            acc += vals @ x[c0:c1]
        out[r0:r1] = acc
    return out


def swiglu_hidden(gate: np.ndarray, up: np.ndarray, limit: float = 10.0) -> np.ndarray:
    if limit > 0.0:
        up = np.clip(up, -limit, limit)
        gate = np.minimum(gate, limit)
    return (silu(gate) * up).astype(np.float32)


def fp4_expert_forward(records: dict[str, dict[str, Any]], x: np.ndarray) -> np.ndarray:
    gate = fp4_matvec_from_records(records, ".w1.weight", (2048, 4096), x)
    up = fp4_matvec_from_records(records, ".w3.weight", (2048, 4096), x)
    hidden = swiglu_hidden(gate, up)
    return fp4_matvec_from_records(records, ".w2.weight", (4096, 2048), hidden)


def fp8_shared_forward(core: Path, tensors: dict[str, dict[str, Any]], x: np.ndarray) -> tuple[np.ndarray, int]:
    gate, b1 = fp8_matvec(core, tensors, "layers.0.ffn.shared_experts.w1.weight", x)
    up, b3 = fp8_matvec(core, tensors, "layers.0.ffn.shared_experts.w3.weight", x)
    hidden = swiglu_hidden(gate, up)
    out, b2 = fp8_matvec(core, tensors, "layers.0.ffn.shared_experts.w2.weight", hidden)
    return out, b1 + b2 + b3


def router_topk(core: Path, tensors: dict[str, dict[str, Any]], x: np.ndarray, top_k: int, route_scale: float) -> list[dict[str, float | int]]:
    gate = tensors["layers.0.ffn.gate.weight"]
    rows = int(gate["row_count"])
    scored: list[tuple[int, float, float]] = []
    for expert_id in range(rows):
        weights = bf16_to_f32(read_row(core, gate, expert_id))
        logit = float(np.dot(x, weights))
        score = float(math.sqrt(math.log1p(math.exp(logit))) if logit <= 20.0 else math.sqrt(logit))
        scored.append((expert_id, logit, score))
    scored.sort(key=lambda item: item[2], reverse=True)
    picked = scored[:top_k]
    denom = sum(item[2] for item in picked)
    return [
        {
            "expert_id": expert_id,
            "logit": logit,
            "weight_score": score,
            "route_weight": score / denom * route_scale if denom > 0 else route_scale / top_k,
        }
        for expert_id, logit, score in picked
    ]


def bounded_lm_head(core: Path, tensors: dict[str, dict[str, Any]], hidden: np.ndarray, scan_vocab: int, top_k: int) -> list[dict[str, float | int]]:
    head = tensors["head.weight"]
    limit = min(scan_vocab, int(head["row_count"]))
    winners: list[tuple[float, int]] = []
    for token_id in range(limit):
        row = bf16_to_f32(read_row(core, head, token_id))
        score = float(np.dot(hidden, row))
        if not math.isfinite(score):
            raise ValueError(f"non-finite LM-head score at token {token_id}")
        winners.append((score, token_id))
        winners.sort(reverse=True)
        if len(winners) > top_k:
            winners.pop()
    return [{"token_id": token_id, "score": score} for score, token_id in winners]


def lm_head_topk_chunked(
    core: Path,
    tensor: dict[str, Any],
    hidden: np.ndarray,
    scan_vocab: int,
    top_k: int,
    chunk_rows: int = 1024,
) -> tuple[list[dict[str, float | int]], int]:
    row_count = int(tensor["row_count"])
    row_width = int(tensor["row_width"])
    row_bytes = int(tensor["row_bytes"])
    limit = row_count if scan_vocab <= 0 else min(scan_vocab, row_count)
    if hidden.size != row_width:
        raise ValueError(f"{tensor['name']}: hidden width {hidden.size} != {row_width}")
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    winners_scores = np.empty(0, dtype=np.float32)
    winners_ids = np.empty(0, dtype=np.int64)
    bytes_read = 0
    base_offset = int(tensor["absolute_data_offset"])
    hidden_f32 = hidden.astype(np.float32, copy=False)
    for start in range(0, limit, chunk_rows):
        rows = min(chunk_rows, limit - start)
        raw = read_at(core, base_offset + start * row_bytes, rows * row_bytes)
        bytes_read += len(raw)
        weights = bf16_to_f32(raw).reshape((rows, row_width))
        scores = weights @ hidden_f32
        token_ids = np.arange(start, start + rows, dtype=np.int64)
        if winners_scores.size:
            scores = np.concatenate([winners_scores, scores])
            token_ids = np.concatenate([winners_ids, token_ids])
        keep = min(top_k, scores.size)
        idx = np.argpartition(scores, -keep)[-keep:]
        winners_scores = scores[idx].astype(np.float32, copy=False)
        winners_ids = token_ids[idx]
    order = np.argsort(winners_scores)[::-1]
    return [
        {"token_id": int(winners_ids[i]), "score": float(winners_scores[i])}
        for i in order[:top_k]
    ], bytes_read


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DeepSeek single-token L0 math smoke")
    parser.add_argument("--index", type=Path, default=Path("models/wohper/DeepSeek-V4-Flash.RAW.L0-SPLIT-GLOBAL-TOP6/dense_core.tensor_index.json"))
    parser.add_argument("--shards", type=Path)
    parser.add_argument("--out", type=Path, default=Path("state/deepseek_v4_single_token_l0_math_smoke_2026-07-05.json"))
    parser.add_argument("--token-id", type=int, default=42)
    parser.add_argument("--scan-vocab", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    started = time.perf_counter()
    args = parse_args()
    index = load_json(args.index)
    core = resolve_core(args.index, index["core_file"])
    shards_path = args.shards or core.with_suffix(".shards.json")
    shards = load_json(shards_path)
    tensors = {tensor["name"]: tensor for tensor in index.get("tensors", [])}
    bytes_read_upper_bound = 0

    hidden = bf16_to_f32(read_row(core, tensors["embed.weight"], args.token_id))
    bytes_read_upper_bound += int(tensors["embed.weight"]["row_bytes"])
    attn_norm = bf16_to_f32(read_tensor(core, tensors["layers.0.attn_norm.weight"]))
    ffn_norm = bf16_to_f32(read_tensor(core, tensors["layers.0.ffn_norm.weight"]))
    bytes_read_upper_bound += int(tensors["layers.0.attn_norm.weight"]["data_bytes"])
    bytes_read_upper_bound += int(tensors["layers.0.ffn_norm.weight"]["data_bytes"])

    x_attn = rms_norm(hidden, attn_norm)
    q_a, b = fp8_matvec(core, tensors, "layers.0.attn.wq_a.weight", x_attn)
    bytes_read_upper_bound += b
    q_norm = bf16_to_f32(read_tensor(core, tensors["layers.0.attn.q_norm.weight"]))
    bytes_read_upper_bound += int(tensors["layers.0.attn.q_norm.weight"]["data_bytes"])
    q_full, b = fp8_matvec(core, tensors, "layers.0.attn.wq_b.weight", rms_norm(q_a, q_norm))
    bytes_read_upper_bound += b
    kv, b = fp8_matvec(core, tensors, "layers.0.attn.wkv.weight", x_attn)
    bytes_read_upper_bound += b
    kv_norm = bf16_to_f32(read_tensor(core, tensors["layers.0.attn.kv_norm.weight"]))
    bytes_read_upper_bound += int(tensors["layers.0.attn.kv_norm.weight"]["data_bytes"])
    kv_normed = rms_norm(kv, kv_norm)

    attn_out, b, attn_report = single_token_attention_out(core, tensors, "layers.0.attn", q_full, kv_normed)
    bytes_read_upper_bound += b
    after_attn = hidden + attn_out

    x_ffn = rms_norm(after_attn, ffn_norm)
    routes = router_topk(core, tensors, x_ffn, top_k=6, route_scale=1.5)
    bytes_read_upper_bound += int(tensors["layers.0.ffn.gate.weight"]["data_bytes"])
    tid_route = i64_from_raw(read_row(core, tensors["layers.0.ffn.gate.tid2eid"], args.token_id))
    bytes_read_upper_bound += int(tensors["layers.0.ffn.gate.tid2eid"]["row_bytes"])
    shared_out, b = fp8_shared_forward(core, tensors, x_ffn)
    bytes_read_upper_bound += b

    expert_by_id = {int(item["expert_id"]): item for item in shards.get("experts", [])}
    routed_out = np.zeros_like(shared_out)
    missing_experts = []
    expert_summaries = []
    for route in routes:
        expert_id = int(route["expert_id"])
        item = expert_by_id.get(expert_id)
        if item is None:
            missing_experts.append(expert_id)
            continue
        expert_path = core.parent / item["path"]
        bytes_read_upper_bound += int(item["payload_bytes"])
        records = parse_expert_block(expert_path, int(item["payload_bytes"]))
        out = fp4_expert_forward(records, x_ffn)
        routed_out += np.float32(route["route_weight"]) * out
        expert_summaries.append({"expert_id": expert_id, "route_weight": float(route["route_weight"]), "out": summary(out)})

    ffn_out = shared_out + routed_out
    final_hidden = after_attn + ffn_out
    topk = bounded_lm_head(core, tensors, final_hidden, args.scan_vocab, args.top_k)
    bytes_read_upper_bound += min(args.scan_vocab, int(tensors["head.weight"]["row_count"])) * int(tensors["head.weight"]["row_bytes"])

    hc_attn_scale = f32_from_raw(read_tensor(core, tensors["layers.0.hc_attn_scale"]))
    hc_ffn_scale = f32_from_raw(read_tensor(core, tensors["layers.0.hc_ffn_scale"]))
    bytes_read_upper_bound += int(tensors["layers.0.hc_attn_scale"]["data_bytes"])
    bytes_read_upper_bound += int(tensors["layers.0.hc_ffn_scale"]["data_bytes"])

    blockers = []
    for label, vector in {
        "hidden": hidden,
        "q_a": q_a,
        "q_full": q_full,
        "kv": kv,
        "kv_normed": kv_normed,
        "attn_out": attn_out,
        "shared_out": shared_out,
        "routed_out": routed_out,
        "final_hidden": final_hidden,
    }.items():
        if int(np.isfinite(vector).sum()) != int(vector.size):
            blockers.append(f"{label}_non_finite")
        if int(np.count_nonzero(vector)) == 0:
            blockers.append(f"{label}_all_zero")
    if missing_experts:
        blockers.append("missing_routed_expert")
    if not topk:
        blockers.append("empty_lmhead_topk")

    payload = {
        "format": "deepseek-v4-single-token-l0-math-smoke",
        "version": 1,
        "status": "ready" if not blockers else "blocked",
        "blockers": blockers,
        "semantic_warnings": [
            "attention_is_single_key_sparse_path_without_long_context_compressor_indexer",
            "mHC_scales_are_read_and_summarized_but_not_applied_to_residual_path",
            "lm_head_is_bounded_to_scan_vocab",
        ],
        "index": str(args.index),
        "core": str(core),
        "shards": str(shards_path),
        "token_id": args.token_id,
        "scan_vocab": args.scan_vocab,
        "bytes_read_upper_bound": bytes_read_upper_bound,
        "elapsed_seconds": time.perf_counter() - started,
        "routes": routes,
        "tid2eid_route": tid_route,
        "missing_experts": missing_experts,
        "vectors": {
            "hidden": summary(hidden),
            "q_a": summary(q_a),
            "q_full": summary(q_full),
            "kv": summary(kv),
            "kv_normed": summary(kv_normed),
            "attn_out": summary(attn_out),
            "attn": attn_report,
            "shared_out": summary(shared_out),
            "routed_out": summary(routed_out),
            "final_hidden": summary(final_hidden),
            "hc_attn_scale": summary(hc_attn_scale),
            "hc_ffn_scale": summary(hc_ffn_scale),
        },
        "expert_summaries": expert_summaries,
        "bounded_lmhead_topk": topk,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not blockers else 3


if __name__ == "__main__":
    raise SystemExit(main())
