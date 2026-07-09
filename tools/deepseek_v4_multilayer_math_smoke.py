#!/usr/bin/env python3
"""Bounded multi-layer DeepSeek-V4 math smoke over raw Wohper shards."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

import deepseek_v4_single_token_l0_math_smoke as base

HC_MULT = 4
HC_EPS = 1e-6
HC_SINKHORN_ITERS = 20
HEADS = 64
HEAD_DIM = 512
ROPE_HEAD_DIM = 64
WINDOW_SIZE = 128
ROPE_THETA = 10000.0
COMPRESS_ROPE_THETA = 160000.0
COMPRESS_RATIOS = [0, 0, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 0]
YARN_ORIGINAL_SEQ_LEN = 65536
YARN_FACTOR = 16.0
YARN_BETA_FAST = 32
YARN_BETA_SLOW = 1
_ROPE_FREQ_CACHE: dict[tuple[float, int, float, int, int], np.ndarray] = {}


def lname(layer_id: int, suffix: str) -> str:
    return f"layers.{layer_id}.{suffix}"


def sigmoid(x: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-x))).astype(np.float32)


def hc_split_sinkhorn(
    mixes: np.ndarray,
    hc_scale: np.ndarray,
    hc_base: np.ndarray,
    hc_mult: int = HC_MULT,
    sinkhorn_iters: int = HC_SINKHORN_ITERS,
    eps: float = HC_EPS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pre = sigmoid(mixes[:hc_mult] * hc_scale[0] + hc_base[:hc_mult]) + eps
    post = 2.0 * sigmoid(mixes[hc_mult : 2 * hc_mult] * hc_scale[1] + hc_base[hc_mult : 2 * hc_mult])
    raw = (
        mixes[2 * hc_mult :].reshape(hc_mult, hc_mult) * hc_scale[2]
        + hc_base[2 * hc_mult :].reshape(hc_mult, hc_mult)
    )
    raw -= np.max(raw, axis=1, keepdims=True)
    comb = np.exp(raw).astype(np.float32)
    comb = comb / np.sum(comb, axis=1, keepdims=True) + eps
    comb = comb / (np.sum(comb, axis=0, keepdims=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (np.sum(comb, axis=1, keepdims=True) + eps)
        comb = comb / (np.sum(comb, axis=0, keepdims=True) + eps)
    return pre.astype(np.float32), post.astype(np.float32), comb.astype(np.float32)


def hc_pre(
    core: Path,
    tensors: dict[str, dict[str, Any]],
    layer_id: int,
    stage: str,
    x_hc: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, dict[str, Any]]:
    fn_name = lname(layer_id, f"hc_{stage}_fn")
    scale_name = lname(layer_id, f"hc_{stage}_scale")
    base_name = lname(layer_id, f"hc_{stage}_base")
    hc_fn = base.f32_from_raw(base.read_tensor(core, tensors[fn_name])).reshape((24, HC_MULT * 4096))
    hc_scale = base.f32_from_raw(base.read_tensor(core, tensors[scale_name]))
    hc_base = base.f32_from_raw(base.read_tensor(core, tensors[base_name]))
    x_flat = x_hc.reshape(HC_MULT * 4096).astype(np.float32, copy=False)
    rsqrt = np.float32(1.0 / math.sqrt(float(np.mean(x_flat * x_flat)) + HC_EPS))
    mixes = (hc_fn @ x_flat * rsqrt).astype(np.float32)
    pre, post, comb = hc_split_sinkhorn(mixes, hc_scale, hc_base)
    y = np.sum(pre[:, None] * x_hc, axis=0).astype(np.float32)
    bytes_read = (
        int(tensors[fn_name]["data_bytes"])
        + int(tensors[scale_name]["data_bytes"])
        + int(tensors[base_name]["data_bytes"])
    )
    report = {
        "pre": [float(v) for v in pre],
        "post": [float(v) for v in post],
        "comb_row_sums": [float(v) for v in np.sum(comb, axis=1)],
        "comb_col_sums": [float(v) for v in np.sum(comb, axis=0)],
        "input_l2": float(np.linalg.norm(x_hc)),
        "mixed_l2": float(np.linalg.norm(y)),
    }
    return y, post, comb, bytes_read, report


def hc_post(x: np.ndarray, residual: np.ndarray, post: np.ndarray, comb: np.ndarray) -> np.ndarray:
    return (post[:, None] * x[None, :] + np.einsum("jk,kd->jd", comb, residual)).astype(np.float32)


def hc_head(
    core: Path,
    tensors: dict[str, dict[str, Any]],
    x_hc: np.ndarray,
) -> tuple[np.ndarray, int, dict[str, Any]]:
    if not {"hc_head_fn", "hc_head_scale", "hc_head_base", "norm.weight"}.issubset(tensors):
        collapsed = np.mean(x_hc, axis=0).astype(np.float32)
        return collapsed, 0, {"mode": "mean_fallback_missing_hc_head"}
    hc_fn = base.f32_from_raw(base.read_tensor(core, tensors["hc_head_fn"])).reshape((HC_MULT, HC_MULT * 4096))
    hc_scale = base.f32_from_raw(base.read_tensor(core, tensors["hc_head_scale"]))
    hc_base = base.f32_from_raw(base.read_tensor(core, tensors["hc_head_base"]))
    norm = base.bf16_to_f32(base.read_tensor(core, tensors["norm.weight"]))
    x_flat = x_hc.reshape(HC_MULT * 4096).astype(np.float32, copy=False)
    rsqrt = np.float32(1.0 / math.sqrt(float(np.mean(x_flat * x_flat)) + HC_EPS))
    mixes = (hc_fn @ x_flat * rsqrt).astype(np.float32)
    pre = sigmoid(mixes * hc_scale[0] + hc_base) + HC_EPS
    collapsed = np.sum(pre[:, None] * x_hc, axis=0).astype(np.float32)
    collapsed = base.rms_norm(collapsed, norm)
    bytes_read = (
        int(tensors["hc_head_fn"]["data_bytes"])
        + int(tensors["hc_head_scale"]["data_bytes"])
        + int(tensors["hc_head_base"]["data_bytes"])
        + int(tensors["norm.weight"]["data_bytes"])
    )
    return collapsed, bytes_read, {"mode": "hc_head", "pre": [float(v) for v in pre], "collapsed": base.summary(collapsed)}


def parse_extra_token_ids(raw: str | None) -> list[int]:
    if not raw:
        return []
    return sorted({int(item.strip()) for item in raw.split(",") if item.strip()})


def parse_context_token_ids(raw: str | None) -> list[int]:
    if not raw:
        return []
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def rope_theta_for_layer(layer_id: int) -> float:
    ratio = COMPRESS_RATIOS[layer_id] if layer_id < len(COMPRESS_RATIOS) else 0
    return COMPRESS_ROPE_THETA if ratio else ROPE_THETA


def rope_original_seq_len_for_layer(layer_id: int) -> int:
    ratio = COMPRESS_RATIOS[layer_id] if layer_id < len(COMPRESS_RATIOS) else 0
    return YARN_ORIGINAL_SEQ_LEN if ratio else 0


def _correction_dim(num_rotations: int, dim: int, base: float, max_seq_len: int) -> float:
    return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))


def _correction_range(low_rot: int, high_rot: int, dim: int, base: float, max_seq_len: int) -> tuple[int, int]:
    low = math.floor(_correction_dim(low_rot, dim, base, max_seq_len))
    high = math.ceil(_correction_dim(high_rot, dim, base, max_seq_len))
    return max(low, 0), min(high, dim - 1)


def _linear_ramp_factor(low: int, high: int, size: int) -> np.ndarray:
    if low == high:
        high += 0.001
    ramp = (np.arange(size, dtype=np.float32) - np.float32(low)) / np.float32(high - low)
    return np.clip(ramp, 0.0, 1.0).astype(np.float32)


def rope_freqs(theta: float, original_seq_len: int = 0) -> np.ndarray:
    key = (float(theta), int(original_seq_len), float(YARN_FACTOR), int(YARN_BETA_FAST), int(YARN_BETA_SLOW))
    cached = _ROPE_FREQ_CACHE.get(key)
    if cached is not None:
        return cached
    freqs = 1.0 / (theta ** (np.arange(0, ROPE_HEAD_DIM, 2, dtype=np.float32) / ROPE_HEAD_DIM))
    if original_seq_len > 0:
        low, high = _correction_range(YARN_BETA_FAST, YARN_BETA_SLOW, ROPE_HEAD_DIM, theta, original_seq_len)
        smooth = 1.0 - _linear_ramp_factor(low, high, ROPE_HEAD_DIM // 2)
        freqs = freqs / np.float32(YARN_FACTOR) * (1.0 - smooth) + freqs * smooth
    _ROPE_FREQ_CACHE[key] = freqs.astype(np.float32)
    return _ROPE_FREQ_CACHE[key]


def apply_rope_tail(
    values: np.ndarray,
    position: int,
    theta: float,
    inverse: bool = False,
    original_seq_len: int = 0,
) -> np.ndarray:
    out = values.astype(np.float32, copy=True)
    tail = out[..., -ROPE_HEAD_DIM:]
    angles = np.float32(position) * rope_freqs(theta, original_seq_len)
    if inverse:
        angles = -angles
    cos = np.cos(angles).astype(np.float32)
    sin = np.sin(angles).astype(np.float32)
    even = tail[..., 0::2].copy()
    odd = tail[..., 1::2].copy()
    tail[..., 0::2] = even * cos - odd * sin
    tail[..., 1::2] = even * sin + odd * cos
    return out


def hadamard_transform_last_dim(values: np.ndarray) -> np.ndarray:
    dim = int(values.shape[-1])
    if dim <= 0 or dim & (dim - 1):
        raise ValueError(f"Hadamard dimension must be a power of two, got {dim}")
    out = values.astype(np.float32, copy=True)
    width = 1
    while width < dim:
        reshaped = out.reshape((-1, dim // (2 * width), 2, width))
        left = reshaped[:, :, 0, :].copy()
        right = reshaped[:, :, 1, :].copy()
        reshaped[:, :, 0, :] = left + right
        reshaped[:, :, 1, :] = left - right
        width *= 2
    out *= np.float32(dim ** -0.5)
    return out.reshape(values.shape)


def q_heads_from_full(q_full: np.ndarray, position: int, theta: float, original_seq_len: int) -> np.ndarray:
    q_heads = q_full.reshape((HEADS, HEAD_DIM)).astype(np.float32, copy=True)
    q_heads *= (1.0 / np.sqrt(np.mean(q_heads * q_heads, axis=1, keepdims=True) + np.float32(HC_EPS))).astype(np.float32)
    return apply_rope_tail(q_heads, position, theta, original_seq_len=original_seq_len)


def sparse_attention_from_cache(
    core: Path,
    tensors: dict[str, dict[str, Any]],
    prefix: str,
    q_full: np.ndarray,
    kv_cache: np.ndarray,
    position: int,
    theta: float,
    original_seq_len: int,
    fp8_cache: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, int, dict[str, Any]]:
    q_heads = q_heads_from_full(q_full, position, theta, original_seq_len)
    sink_name = f"{prefix}.attn_sink"
    sink = base.f32_from_raw(base.read_tensor(core, tensors[sink_name]))
    scores = (q_heads @ kv_cache.T) * np.float32(HEAD_DIM ** -0.5)
    row_max = np.maximum(np.max(scores, axis=1), sink)
    exp_scores = np.exp(scores - row_max[:, None]).astype(np.float32)
    exp_sink = np.exp(sink - row_max).astype(np.float32)
    denom = np.sum(exp_scores, axis=1) + exp_sink
    weights = exp_scores / denom[:, None]
    o_heads = (weights @ kv_cache).astype(np.float32)
    o_heads = apply_rope_tail(o_heads, position, theta, inverse=True, original_seq_len=original_seq_len)
    grouped = o_heads.reshape((8, 4096))
    if fp8_cache is None:
        wo_a, b1 = base.fp8_grouped_wo_a(core, tensors, f"{prefix}.wo_a.weight", grouped)
        attn_out, b2 = base.fp8_matvec(core, tensors, f"{prefix}.wo_b.weight", wo_a)
    else:
        wo_a, b1 = fp8_grouped_wo_a_cached(core, tensors, f"{prefix}.wo_a.weight", grouped, fp8_cache)
        attn_out, b2 = fp8_matvec_cached(core, tensors, f"{prefix}.wo_b.weight", wo_a, fp8_cache)
    report = {
        "mode": "sliding_window_sparse_attention_prefill",
        "position": position,
        "kv_count": int(kv_cache.shape[0]),
        "q_heads": base.summary(q_heads),
        "sink": base.summary(sink),
        "scores": base.summary(scores.reshape(-1)),
        "weights": base.summary(weights.reshape(-1)),
        "wo_a": base.summary(wo_a),
    }
    return attn_out, int(tensors[sink_name]["data_bytes"]) + b1 + b2, report


def lm_head_candidates(
    core: Path,
    tensors: dict[str, dict[str, Any]],
    hidden: np.ndarray,
    scan_vocab: int,
    top_k: int,
    extra_token_ids: list[int],
    chunk_rows: int,
) -> tuple[list[dict[str, float | int | str]], int]:
    head = tensors["head.weight"]
    row_count = int(head["row_count"])
    candidates: dict[int, dict[str, float | int | str]] = {}
    topk_items, bytes_read = base.lm_head_topk_chunked(core, head, hidden, scan_vocab, top_k, chunk_rows)
    for item in topk_items:
        token_id = int(item["token_id"])
        candidates[token_id] = {"token_id": token_id, "score": float(item["score"]), "source": "scan"}
    for token_id in extra_token_ids:
        if token_id < 0 or token_id >= row_count or token_id in candidates:
            continue
        weights = base.bf16_to_f32(base.read_row(core, head, token_id))
        bytes_read += int(head["row_bytes"])
        candidates[token_id] = {
            "token_id": token_id,
            "score": float(np.dot(hidden, weights)),
            "source": "extra",
        }
    merged = sorted(candidates.values(), key=lambda item: float(item["score"]), reverse=True)
    return merged[:top_k], bytes_read


def fp8_shared_forward(core: Path, tensors: dict[str, dict[str, Any]], layer_id: int, x: np.ndarray) -> tuple[np.ndarray, int]:
    gate, b1 = base.fp8_matvec(core, tensors, lname(layer_id, "ffn.shared_experts.w1.weight"), x)
    up, b3 = base.fp8_matvec(core, tensors, lname(layer_id, "ffn.shared_experts.w3.weight"), x)
    hidden = base.swiglu_hidden(gate, up)
    out, b2 = base.fp8_matvec(core, tensors, lname(layer_id, "ffn.shared_experts.w2.weight"), hidden)
    return out, b1 + b2 + b3


def decode_fp8_weight_cached(
    core: Path,
    tensors: dict[str, dict[str, Any]],
    name: str,
    cache: dict[str, np.ndarray],
) -> tuple[np.ndarray, int]:
    cached = cache.get(name)
    if cached is not None:
        return cached, 0
    weight = tensors[name]
    scale = tensors[name.replace(".weight", ".scale")]
    rows, cols = [int(v) for v in weight["shape"]]
    scales = base.decode_ue8m0_array(base.read_tensor(core, scale)).reshape((math.ceil(rows / base.BLOCK), math.ceil(cols / base.BLOCK)))
    raw = np.frombuffer(base.read_tensor(core, weight), dtype=np.uint8).reshape((rows, cols))
    decoded = np.empty((rows, cols), dtype=np.float32)
    for r0 in range(0, rows, base.BLOCK):
        r1 = min(r0 + base.BLOCK, rows)
        scale_row = r0 // base.BLOCK
        for c0 in range(0, cols, base.BLOCK):
            c1 = min(c0 + base.BLOCK, cols)
            vals = base.decode_fp8_e4m3_array(raw[r0:r1, c0:c1])
            decoded[r0:r1, c0:c1] = vals * scales[scale_row, c0 // base.BLOCK]
    cache[name] = decoded
    return decoded, int(weight["data_bytes"]) + int(scale["data_bytes"])


def fp8_matvec_cached(
    core: Path,
    tensors: dict[str, dict[str, Any]],
    name: str,
    x: np.ndarray,
    cache: dict[str, np.ndarray],
) -> tuple[np.ndarray, int]:
    weight, bytes_read = decode_fp8_weight_cached(core, tensors, name, cache)
    if x.size != weight.shape[1]:
        raise ValueError(f"{name}: input width {x.size} != {weight.shape[1]}")
    xq = base.quant_dequant_fp8_activation(x)
    return (weight @ xq.astype(np.float32, copy=False)).astype(np.float32), bytes_read


def fp8_matmat_cached(
    core: Path,
    tensors: dict[str, dict[str, Any]],
    name: str,
    x: np.ndarray,
    cache: dict[str, np.ndarray],
) -> tuple[np.ndarray, int]:
    weight, bytes_read = decode_fp8_weight_cached(core, tensors, name, cache)
    if x.shape[1] != weight.shape[1]:
        raise ValueError(f"{name}: input width {x.shape[1]} != {weight.shape[1]}")
    xq = base.quant_dequant_fp8_activation(x)
    return (xq.astype(np.float32, copy=False) @ weight.T).astype(np.float32), bytes_read


def dense_weight_cached(
    core: Path,
    tensors: dict[str, dict[str, Any]],
    name: str,
    cache: dict[str, np.ndarray],
) -> tuple[np.ndarray, int]:
    cached = cache.get(name)
    if cached is not None:
        return cached, 0
    tensor = tensors[name]
    shape = [int(v) for v in tensor["shape"]]
    raw = base.read_tensor(core, tensor)
    dtype_code = int(tensor.get("dtype_code", -1))
    if dtype_code == 11:
        decoded = base.bf16_to_f32(raw).reshape(shape)
    elif dtype_code == 12:
        decoded = base.f32_from_raw(raw).reshape(shape).astype(np.float32, copy=False)
    else:
        raise ValueError(f"{name}: unsupported dense dtype_code {dtype_code}")
    cache[name] = decoded
    return decoded, int(tensor["data_bytes"])


def dense_matmat_cached(
    core: Path,
    tensors: dict[str, dict[str, Any]],
    name: str,
    x: np.ndarray,
    cache: dict[str, np.ndarray],
) -> tuple[np.ndarray, int]:
    weight, bytes_read = dense_weight_cached(core, tensors, name, cache)
    if x.shape[1] != weight.shape[1]:
        raise ValueError(f"{name}: input width {x.shape[1]} != {weight.shape[1]}")
    return (x.astype(np.float32, copy=False) @ weight.T).astype(np.float32), bytes_read


def rms_norm_rows(x: np.ndarray, weight: np.ndarray, eps: float = HC_EPS) -> np.ndarray:
    scale = (1.0 / np.sqrt(np.mean(x.astype(np.float32) ** 2, axis=1, keepdims=True) + np.float32(eps))).astype(np.float32)
    return (x * scale * weight[None, :]).astype(np.float32)


def fp8_grouped_wo_a_cached(
    core: Path,
    tensors: dict[str, dict[str, Any]],
    name: str,
    grouped_x: np.ndarray,
    cache: dict[str, np.ndarray],
) -> tuple[np.ndarray, int]:
    weight, bytes_read = decode_fp8_weight_cached(core, tensors, name, cache)
    groups = int(grouped_x.shape[0])
    rows_per_group = weight.shape[0] // groups
    out = np.empty(weight.shape[0], dtype=np.float32)
    for group_id in range(groups):
        r0 = group_id * rows_per_group
        r1 = r0 + rows_per_group
        xq = base.quant_dequant_fp8_activation(grouped_x[group_id])
        out[r0:r1] = weight[r0:r1] @ xq.astype(np.float32, copy=False)
    return out, bytes_read


def fp8_shared_forward_cached(
    core: Path,
    tensors: dict[str, dict[str, Any]],
    layer_id: int,
    x: np.ndarray,
    cache: dict[str, np.ndarray],
) -> tuple[np.ndarray, int]:
    gate, b1 = fp8_matvec_cached(core, tensors, lname(layer_id, "ffn.shared_experts.w1.weight"), x, cache)
    up, b3 = fp8_matvec_cached(core, tensors, lname(layer_id, "ffn.shared_experts.w3.weight"), x, cache)
    hidden = base.swiglu_hidden(gate, up)
    out, b2 = fp8_matvec_cached(core, tensors, lname(layer_id, "ffn.shared_experts.w2.weight"), hidden, cache)
    return out, b1 + b2 + b3


def softmax_axis1(values: np.ndarray) -> np.ndarray:
    max_values = np.max(values, axis=1, keepdims=True)
    shifted = np.exp(values - max_values).astype(np.float32)
    shifted = np.where(np.isfinite(values), shifted, 0.0)
    denom = np.sum(shifted, axis=1, keepdims=True)
    return shifted / np.maximum(denom, np.float32(1.0e-20))


def compressor_prefill_kv(
    core: Path,
    tensors: dict[str, dict[str, Any]],
    layer_id: int,
    x_attn_seq: np.ndarray,
    ratio: int,
    theta: float,
    original_seq_len: int,
    dense_cache: dict[str, np.ndarray],
    prefix_suffix: str = "attn.compressor",
    head_dim: int = HEAD_DIM,
    rotate: bool = False,
) -> tuple[np.ndarray, int, dict[str, Any]]:
    prefix = lname(layer_id, prefix_suffix)
    required = [
        f"{prefix}.ape",
        f"{prefix}.wkv.weight",
        f"{prefix}.wgate.weight",
        f"{prefix}.norm.weight",
    ]
    if ratio <= 0 or any(name not in tensors for name in required):
        return np.empty((0, HEAD_DIM), dtype=np.float32), 0, {"mode": "disabled_or_missing"}
    seq_len = int(x_attn_seq.shape[0])
    cutoff = seq_len - (seq_len % ratio)
    if cutoff <= 0:
        return np.empty((0, HEAD_DIM), dtype=np.float32), 0, {"mode": "no_complete_compression_block", "ratio": ratio}

    bytes_read = 0
    kv_full, b = dense_matmat_cached(core, tensors, f"{prefix}.wkv.weight", x_attn_seq, dense_cache)
    bytes_read += b
    score_full, b = dense_matmat_cached(core, tensors, f"{prefix}.wgate.weight", x_attn_seq, dense_cache)
    bytes_read += b
    ape = base.f32_from_raw(base.read_tensor(core, tensors[f"{prefix}.ape"])).reshape((ratio, -1))
    norm = base.bf16_to_f32(base.read_tensor(core, tensors[f"{prefix}.norm.weight"]))
    bytes_read += int(tensors[f"{prefix}.ape"]["data_bytes"])
    bytes_read += int(tensors[f"{prefix}.norm.weight"]["data_bytes"])

    coff = 2 if ratio == 4 else 1
    kv_blocks = kv_full[:cutoff].reshape((-1, ratio, coff * head_dim))
    score_blocks = score_full[:cutoff].reshape((-1, ratio, coff * head_dim)) + ape[None, :, :]
    if coff == 2:
        block_count = int(kv_blocks.shape[0])
        kv_pool = np.zeros((block_count, 2 * ratio, head_dim), dtype=np.float32)
        score_pool = np.full((block_count, 2 * ratio, head_dim), -np.inf, dtype=np.float32)
        kv_pool[:, ratio:, :] = kv_blocks[:, :, head_dim:]
        score_pool[:, ratio:, :] = score_blocks[:, :, head_dim:]
        if block_count > 1:
            kv_pool[1:, :ratio, :] = kv_blocks[:-1, :, :head_dim]
            score_pool[1:, :ratio, :] = score_blocks[:-1, :, :head_dim]
    else:
        kv_pool = kv_blocks
        score_pool = score_blocks
    weights = softmax_axis1(score_pool)
    compressed = np.sum(kv_pool * weights, axis=1).astype(np.float32)
    compressed = rms_norm_rows(compressed, norm)
    for idx, position in enumerate(range(0, cutoff, ratio)):
        compressed[idx] = apply_rope_tail(compressed[idx], position, theta, original_seq_len=original_seq_len)
    if rotate:
        compressed = hadamard_transform_last_dim(compressed)
        compressed = base.quant_dequant_fp4_activation(compressed, block_size=32)
    else:
        compressed = base.quant_dequant_kv_nonrope(compressed, ROPE_HEAD_DIM)
    report = {
        "mode": "short_context_compressor_prefill",
        "prefix": prefix_suffix,
        "ratio": ratio,
        "head_dim": head_dim,
        "compressed_count": int(compressed.shape[0]),
        "cutoff": int(cutoff),
        "overlap": bool(coff == 2),
        "rotate_fp4": bool(rotate),
        "kv": base.summary(compressed.reshape(-1)),
        "kv_full_sample_per_pos": [kv_full[p][:4].tolist() for p in range(min(cutoff, 4))],
        "score_full_sample_per_pos": [score_full[p][:4].tolist() for p in range(min(cutoff, 4))],
    }
    return compressed, bytes_read, report


def indexer_prefill_topk_indices(
    core: Path,
    tensors: dict[str, dict[str, Any]],
    layer_id: int,
    x_attn_seq: np.ndarray,
    q_normed_seq: np.ndarray,
    ratio: int,
    theta: float,
    original_seq_len: int,
    compressed_count: int,
    dense_cache: dict[str, np.ndarray],
    fp8_cache: dict[str, np.ndarray],
) -> tuple[list[np.ndarray] | None, int, dict[str, Any]]:
    if ratio != 4:
        return None, 0, {"mode": "not_applicable", "reason": "indexer_only_for_ratio_4"}
    if compressed_count <= 512:
        return None, 0, {
            "mode": "not_needed_all_compressed_blocks_are_selected",
            "compressed_count": compressed_count,
            "index_topk": 512,
        }
    prefix = lname(layer_id, "attn.indexer")
    required = [
        f"{prefix}.wq_b.weight",
        f"{prefix}.weights_proj.weight",
        f"{prefix}.compressor.ape",
        f"{prefix}.compressor.norm.weight",
        f"{prefix}.compressor.wgate.weight",
        f"{prefix}.compressor.wkv.weight",
    ]
    missing = [name for name in required if name not in tensors]
    if missing:
        return None, 0, {"mode": "missing_indexer_tensors", "missing": missing}

    bytes_read = 0
    idx_kv, b, idx_compressor_report = compressor_prefill_kv(
        core,
        tensors,
        layer_id,
        x_attn_seq,
        ratio,
        theta,
        original_seq_len,
        dense_cache,
        prefix_suffix="attn.indexer.compressor",
        head_dim=128,
        rotate=True,
    )
    bytes_read += b
    q_full, b = fp8_matmat_cached(core, tensors, f"{prefix}.wq_b.weight", q_normed_seq, fp8_cache)
    bytes_read += b
    q = q_full.reshape((int(q_normed_seq.shape[0]), HEADS, 128)).astype(np.float32, copy=True)
    for pos in range(q.shape[0]):
        q[pos] = apply_rope_tail(q[pos], pos, theta, original_seq_len=original_seq_len)
    q = hadamard_transform_last_dim(q)
    q = base.quant_dequant_fp4_activation(q, block_size=32)
    weights, b = dense_matmat_cached(core, tensors, f"{prefix}.weights_proj.weight", x_attn_seq, dense_cache)
    bytes_read += b
    weights = weights.astype(np.float32, copy=False) * np.float32((128 ** -0.5) * (HEADS ** -0.5))

    selected: list[np.ndarray] = []
    topk = 512
    for pos in range(q.shape[0]):
        available = min((pos + 1) // ratio, int(idx_kv.shape[0]))
        if available <= 0:
            selected.append(np.empty(0, dtype=np.int64))
            continue
        scores = np.maximum(q[pos] @ idx_kv[:available].T, 0.0)
        scores = np.sum(scores * weights[pos, :, None], axis=0)
        keep = min(topk, available)
        if keep < available:
            ids = np.argpartition(scores, -keep)[-keep:]
            ids = ids[np.argsort(scores[ids])[::-1]]
        else:
            ids = np.arange(available, dtype=np.int64)
        selected.append(ids.astype(np.int64, copy=False))

    counts = [int(item.size) for item in selected]
    report = {
        "mode": "learned_indexer_prefill_topk",
        "compressed_count": int(idx_kv.shape[0]),
        "index_topk": topk,
        "selected_min": min(counts) if counts else 0,
        "selected_max": max(counts) if counts else 0,
        "selected_last": counts[-1] if counts else 0,
        "compressor": idx_compressor_report,
    }
    return selected, bytes_read, report


def router_topk(
    core: Path,
    tensors: dict[str, dict[str, Any]],
    layer_id: int,
    x: np.ndarray,
    token_id: int,
    top_k: int,
    route_scale: float,
) -> tuple[list[dict[str, float | int | str]], list[int], int]:
    gate = tensors[lname(layer_id, "ffn.gate.weight")]
    rows = int(gate["row_count"])
    bytes_read = 0
    score_by_expert: dict[int, tuple[float, float]] = {}
    selection_scores: list[tuple[int, float]] = []
    bias_name = lname(layer_id, "ffn.gate.bias")
    bias = base.f32_from_raw(base.read_tensor(core, tensors[bias_name])) if bias_name in tensors else None
    if bias_name in tensors:
        bytes_read += int(tensors[bias_name]["data_bytes"])
    for expert_id in range(rows):
        raw = base.read_row(core, gate, expert_id)
        bytes_read += len(raw)
        weights = base.bf16_to_f32(raw)
        logit = float(np.dot(x, weights))
        score = float(base.softplus(np.array([logit], dtype=np.float32))[0] ** 0.5)
        score_by_expert[expert_id] = (logit, score)
        select_score = score + (float(bias[expert_id]) if bias is not None else 0.0)
        selection_scores.append((expert_id, select_score))

    tid_name = lname(layer_id, "ffn.gate.tid2eid")
    if tid_name in tensors:
        picked_ids = base.i64_from_raw(base.read_row(core, tensors[tid_name], token_id))[:top_k]
        bytes_read += int(tensors[tid_name]["row_bytes"])
        router_mode = "hash_tid2eid"
    else:
        selection_scores.sort(key=lambda item: item[1], reverse=True)
        picked_ids = [expert_id for expert_id, _ in selection_scores[:top_k]]
        router_mode = "score_bias_topk" if bias is not None else "score_topk"

    denom = sum(score_by_expert[expert_id][1] for expert_id in picked_ids)
    routes = [
        {
            "expert_id": expert_id,
            "logit": score_by_expert[expert_id][0],
            "router_mode": router_mode,
            "weight_score": score,
            "route_weight": score / denom * route_scale if denom > 0 else route_scale / top_k,
        }
        for expert_id in picked_ids
        for score in [score_by_expert[expert_id][1]]
    ]
    return routes, picked_ids, bytes_read


def layer_forward(
    core: Path,
    tensors: dict[str, dict[str, Any]],
    expert_by_key: dict[tuple[int, int], dict[str, Any]],
    hidden_hc: np.ndarray,
    token_id: int,
    layer_id: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    bytes_read = 0
    missing_experts: list[int] = []

    attn_norm = base.bf16_to_f32(base.read_tensor(core, tensors[lname(layer_id, "attn_norm.weight")]))
    ffn_norm = base.bf16_to_f32(base.read_tensor(core, tensors[lname(layer_id, "ffn_norm.weight")]))
    bytes_read += int(tensors[lname(layer_id, "attn_norm.weight")]["data_bytes"])
    bytes_read += int(tensors[lname(layer_id, "ffn_norm.weight")]["data_bytes"])

    attn_residual = hidden_hc
    attn_in, attn_post, attn_comb, b, attn_hc_report = hc_pre(core, tensors, layer_id, "attn", attn_residual)
    bytes_read += b
    x_attn = base.rms_norm(attn_in, attn_norm)
    q_a, b = base.fp8_matvec(core, tensors, lname(layer_id, "attn.wq_a.weight"), x_attn)
    bytes_read += b
    q_norm = base.bf16_to_f32(base.read_tensor(core, tensors[lname(layer_id, "attn.q_norm.weight")]))
    bytes_read += int(tensors[lname(layer_id, "attn.q_norm.weight")]["data_bytes"])
    q_full, b = base.fp8_matvec(core, tensors, lname(layer_id, "attn.wq_b.weight"), base.rms_norm(q_a, q_norm))
    bytes_read += b
    kv, b = base.fp8_matvec(core, tensors, lname(layer_id, "attn.wkv.weight"), x_attn)
    bytes_read += b
    kv_norm = base.bf16_to_f32(base.read_tensor(core, tensors[lname(layer_id, "attn.kv_norm.weight")]))
    bytes_read += int(tensors[lname(layer_id, "attn.kv_norm.weight")]["data_bytes"])
    kv_normed = base.rms_norm(kv, kv_norm)

    attn_out, b, attn_report = base.single_token_attention_out(
        core,
        tensors,
        lname(layer_id, "attn"),
        q_full,
        kv_normed,
    )
    bytes_read += b
    after_attn = hc_post(attn_out, attn_residual, attn_post, attn_comb)

    ffn_residual = after_attn
    ffn_in, ffn_post, ffn_comb, b, ffn_hc_report = hc_pre(core, tensors, layer_id, "ffn", ffn_residual)
    bytes_read += b
    x_ffn = base.rms_norm(ffn_in, ffn_norm)
    routes, tid_route, b = router_topk(core, tensors, layer_id, x_ffn, token_id, top_k=6, route_scale=1.5)
    bytes_read += b
    shared_out, b = fp8_shared_forward(core, tensors, layer_id, x_ffn)
    bytes_read += b

    routed_out = np.zeros_like(shared_out)
    expert_summaries = []
    for route in routes:
        expert_id = int(route["expert_id"])
        item = expert_by_key.get((layer_id, expert_id))
        if item is None:
            missing_experts.append(expert_id)
            continue
        expert_path = core.parent / item["path"]
        bytes_read += int(item["payload_bytes"])
        records = base.parse_expert_block(expert_path, int(item["payload_bytes"]))
        out = base.fp4_expert_forward(records, x_ffn)
        routed_out += np.float32(route["route_weight"]) * out
        expert_summaries.append({"expert_id": expert_id, "route_weight": float(route["route_weight"]), "out": base.summary(out)})

    ffn_out = shared_out + routed_out
    final_hidden = hc_post(ffn_out, ffn_residual, ffn_post, ffn_comb)
    vectors = {
        "attn_hc_in": base.summary(attn_in),
        "q_a": base.summary(q_a),
        "q_full": base.summary(q_full),
        "kv": base.summary(kv),
        "kv_normed": base.summary(kv_normed),
        "attn_out": base.summary(attn_out),
        "attn": attn_report,
        "ffn_hc_in": base.summary(ffn_in),
        "shared_out": base.summary(shared_out),
        "routed_out": base.summary(routed_out),
        "final_hidden_hc": base.summary(final_hidden.reshape(-1)),
    }
    report = {
        "layer_id": layer_id,
        "bytes_read_upper_bound": bytes_read,
        "routes": routes,
        "tid2eid_route": tid_route,
        "missing_experts": missing_experts,
        "expert_summaries": expert_summaries,
        "hc": {"attn": attn_hc_report, "ffn": ffn_hc_report},
        "vectors": vectors,
    }
    return final_hidden, report


def layer_forward_sequence(
    core: Path,
    tensors: dict[str, dict[str, Any]],
    expert_by_key: dict[tuple[int, int], dict[str, Any]],
    hidden_seq: np.ndarray,
    token_ids: list[int],
    layer_id: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    bytes_read = 0
    layer_started = time.perf_counter()
    timings: dict[str, float] = {}
    missing_experts: set[int] = set()
    seq_len = int(hidden_seq.shape[0])
    theta = rope_theta_for_layer(layer_id)
    original_seq_len = rope_original_seq_len_for_layer(layer_id)
    fp8_cache: dict[str, np.ndarray] = {}
    dense_cache: dict[str, np.ndarray] = {}
    expert_cache: dict[tuple[int, int], dict[str, dict[str, Any]]] = {}
    compress_ratio = COMPRESS_RATIOS[layer_id] if layer_id < len(COMPRESS_RATIOS) else 0

    stage_started = time.perf_counter()
    attn_norm = base.bf16_to_f32(base.read_tensor(core, tensors[lname(layer_id, "attn_norm.weight")]))
    ffn_norm = base.bf16_to_f32(base.read_tensor(core, tensors[lname(layer_id, "ffn_norm.weight")]))
    q_norm = base.bf16_to_f32(base.read_tensor(core, tensors[lname(layer_id, "attn.q_norm.weight")]))
    kv_norm = base.bf16_to_f32(base.read_tensor(core, tensors[lname(layer_id, "attn.kv_norm.weight")]))
    bytes_read += int(tensors[lname(layer_id, "attn_norm.weight")]["data_bytes"])
    bytes_read += int(tensors[lname(layer_id, "ffn_norm.weight")]["data_bytes"])
    bytes_read += int(tensors[lname(layer_id, "attn.q_norm.weight")]["data_bytes"])
    bytes_read += int(tensors[lname(layer_id, "attn.kv_norm.weight")]["data_bytes"])
    timings["norm_load_seconds"] = time.perf_counter() - stage_started

    stage_started = time.perf_counter()
    after_attn_seq = np.empty_like(hidden_seq)
    attn_residuals: list[np.ndarray] = []
    attn_posts: list[np.ndarray] = []
    attn_combs: list[np.ndarray] = []
    attn_inputs: list[np.ndarray] = []
    attn_hc_reports: list[dict[str, Any]] = []
    for pos in range(seq_len):
        attn_residual = hidden_seq[pos]
        attn_in, attn_post, attn_comb, b, attn_hc_report = hc_pre(core, tensors, layer_id, "attn", attn_residual)
        bytes_read += b
        attn_residuals.append(attn_residual)
        attn_posts.append(attn_post)
        attn_combs.append(attn_comb)
        attn_inputs.append(base.rms_norm(attn_in, attn_norm))
        attn_hc_reports.append(attn_hc_report)
    timings["attn_hc_pre_seconds"] = time.perf_counter() - stage_started

    x_attn_seq = np.stack(attn_inputs, axis=0)
    stage_started = time.perf_counter()
    q_a_seq, b = fp8_matmat_cached(core, tensors, lname(layer_id, "attn.wq_a.weight"), x_attn_seq, fp8_cache)
    bytes_read += b
    q_normed_seq = rms_norm_rows(q_a_seq, q_norm)
    q_full_seq, b = fp8_matmat_cached(core, tensors, lname(layer_id, "attn.wq_b.weight"), q_normed_seq, fp8_cache)
    bytes_read += b
    kv_seq, b = fp8_matmat_cached(core, tensors, lname(layer_id, "attn.wkv.weight"), x_attn_seq, fp8_cache)
    bytes_read += b
    timings["attn_qkv_project_seconds"] = time.perf_counter() - stage_started
    stage_started = time.perf_counter()
    compressed_kv, b, compressor_report = compressor_prefill_kv(
        core,
        tensors,
        layer_id,
        x_attn_seq,
        compress_ratio,
        theta,
        original_seq_len,
        dense_cache,
    )
    bytes_read += b
    timings["attn_compressor_prefill_seconds"] = time.perf_counter() - stage_started
    if compress_ratio == 4 and int(compressed_kv.shape[0]) > 512:
        stage_started = time.perf_counter()
        indexer_topk, b, indexer_report = indexer_prefill_topk_indices(
            core,
            tensors,
            layer_id,
            x_attn_seq,
            q_normed_seq,
            compress_ratio,
            theta,
            original_seq_len,
            int(compressed_kv.shape[0]),
            dense_cache,
            fp8_cache,
        )
        bytes_read += b
        timings["attn_indexer_prefill_seconds"] = time.perf_counter() - stage_started
    else:
        indexer_topk = None
        indexer_report = {
            "mode": "not_needed_all_compressed_blocks_are_selected" if compress_ratio == 4 else "not_applicable",
            "compressed_count": int(compressed_kv.shape[0]),
            "index_topk": 512,
        }
        timings["attn_indexer_prefill_seconds"] = 0.0

    stage_started = time.perf_counter()
    kv_cache: list[np.ndarray] = []
    last_attn_report: dict[str, Any] = {}
    last_attn_vectors: dict[str, Any] = {}
    last_attn_hc_report: dict[str, Any] = {}
    for pos in range(seq_len):
        attn_residual = attn_residuals[pos]
        attn_post = attn_posts[pos]
        attn_comb = attn_combs[pos]
        q_a = q_a_seq[pos]
        q_full = q_full_seq[pos]
        kv = kv_seq[pos]
        kv_normed = base.rms_norm(kv, kv_norm)
        kv_rot = apply_rope_tail(kv_normed, pos, theta, original_seq_len=original_seq_len)
        kv_cache.append(base.quant_dequant_kv_nonrope(kv_rot, ROPE_HEAD_DIM))
        window_parts = [np.stack(kv_cache[max(0, len(kv_cache) - WINDOW_SIZE) :], axis=0)]
        if compressed_kv.size:
            if indexer_topk is not None:
                selected_compressed = indexer_topk[pos]
                if selected_compressed.size:
                    window_parts.append(compressed_kv[selected_compressed])
            else:
                available_compressed = min((pos + 1) // max(1, compress_ratio), int(compressed_kv.shape[0]), 512)
                if available_compressed > 0:
                    window_parts.append(compressed_kv[:available_compressed])
        window = np.concatenate(window_parts, axis=0)
        attn_out, b, attn_report = sparse_attention_from_cache(
            core,
            tensors,
            lname(layer_id, "attn"),
            q_full,
            window,
            pos,
            theta,
            original_seq_len,
            fp8_cache,
        )
        bytes_read += b
        after_attn_seq[pos] = hc_post(attn_out, attn_residual, attn_post, attn_comb)
        if pos == seq_len - 1:
            last_attn_report = attn_report
            last_attn_hc_report = attn_hc_reports[pos]
            last_attn_vectors = {
                "attn_hc_in": base.summary(attn_inputs[pos]),
                "q_a": base.summary(q_a),
                "q_full": base.summary(q_full),
                "kv": base.summary(kv),
                "kv_normed": base.summary(kv_normed),
                "attn_out": base.summary(attn_out),
                "attn": attn_report,
            }
    timings["attn_sparse_loop_seconds"] = time.perf_counter() - stage_started

    stage_started = time.perf_counter()
    final_seq = np.empty_like(hidden_seq)
    ffn_residuals: list[np.ndarray] = []
    ffn_posts: list[np.ndarray] = []
    ffn_combs: list[np.ndarray] = []
    ffn_inputs: list[np.ndarray] = []
    ffn_hc_reports: list[dict[str, Any]] = []
    for pos in range(seq_len):
        ffn_residual = after_attn_seq[pos]
        ffn_in, ffn_post, ffn_comb, b, ffn_hc_report = hc_pre(core, tensors, layer_id, "ffn", ffn_residual)
        bytes_read += b
        ffn_residuals.append(ffn_residual)
        ffn_posts.append(ffn_post)
        ffn_combs.append(ffn_comb)
        ffn_inputs.append(base.rms_norm(ffn_in, ffn_norm))
        ffn_hc_reports.append(ffn_hc_report)
    timings["ffn_hc_pre_seconds"] = time.perf_counter() - stage_started
    x_ffn_seq = np.stack(ffn_inputs, axis=0)
    stage_started = time.perf_counter()
    shared_gate_seq, b = fp8_matmat_cached(core, tensors, lname(layer_id, "ffn.shared_experts.w1.weight"), x_ffn_seq, fp8_cache)
    bytes_read += b
    shared_up_seq, b = fp8_matmat_cached(core, tensors, lname(layer_id, "ffn.shared_experts.w3.weight"), x_ffn_seq, fp8_cache)
    bytes_read += b
    shared_hidden_seq = base.swiglu_hidden(shared_gate_seq, shared_up_seq)
    shared_out_seq, b = fp8_matmat_cached(core, tensors, lname(layer_id, "ffn.shared_experts.w2.weight"), shared_hidden_seq, fp8_cache)
    bytes_read += b
    timings["ffn_shared_experts_seconds"] = time.perf_counter() - stage_started

    stage_started = time.perf_counter()
    last_ffn_hc_report: dict[str, Any] = {}
    last_routes: list[dict[str, float | int | str]] = []
    last_tid_route: list[int] = []
    last_expert_summaries: list[dict[str, Any]] = []
    last_ffn_vectors: dict[str, Any] = {}
    router_seconds = 0.0
    expert_parse_seconds = 0.0
    expert_forward_seconds = 0.0
    for pos in range(seq_len):
        ffn_residual = ffn_residuals[pos]
        ffn_post = ffn_posts[pos]
        ffn_comb = ffn_combs[pos]
        x_ffn = x_ffn_seq[pos]
        router_started = time.perf_counter()
        routes, tid_route, b = router_topk(core, tensors, layer_id, x_ffn, token_ids[pos], top_k=6, route_scale=1.5)
        router_seconds += time.perf_counter() - router_started
        bytes_read += b
        shared_out = shared_out_seq[pos]
        routed_out = np.zeros_like(shared_out)
        expert_summaries = []
        for route in routes:
            expert_id = int(route["expert_id"])
            item = expert_by_key.get((layer_id, expert_id))
            if item is None:
                missing_experts.add(expert_id)
                continue
            expert_path = core.parent / item["path"]
            cache_key = (layer_id, expert_id)
            records = expert_cache.get(cache_key)
            if records is None:
                bytes_read += int(item["payload_bytes"])
                parse_started = time.perf_counter()
                records = base.parse_expert_block(expert_path, int(item["payload_bytes"]))
                expert_parse_seconds += time.perf_counter() - parse_started
                expert_cache[cache_key] = records
            expert_started = time.perf_counter()
            out = base.fp4_expert_forward(records, x_ffn)
            expert_forward_seconds += time.perf_counter() - expert_started
            routed_out += np.float32(route["route_weight"]) * out
            if pos == seq_len - 1:
                expert_summaries.append({"expert_id": expert_id, "route_weight": float(route["route_weight"]), "out": base.summary(out)})
        ffn_out = shared_out + routed_out
        final_seq[pos] = hc_post(ffn_out, ffn_residual, ffn_post, ffn_comb)
        if pos == seq_len - 1:
            last_ffn_hc_report = ffn_hc_reports[pos]
            last_routes = routes
            last_tid_route = tid_route
            last_expert_summaries = expert_summaries
            last_ffn_vectors = {
                "ffn_hc_in": base.summary(ffn_inputs[pos]),
                "shared_out": base.summary(shared_out),
                "routed_out": base.summary(routed_out),
                "final_hidden_hc": base.summary(final_seq[pos].reshape(-1)),
            }
    timings["ffn_router_expert_loop_seconds"] = time.perf_counter() - stage_started
    timings["ffn_router_score_seconds"] = router_seconds
    timings["ffn_expert_parse_seconds"] = expert_parse_seconds
    timings["ffn_expert_forward_seconds"] = expert_forward_seconds
    timings["layer_total_seconds"] = time.perf_counter() - layer_started

    report = {
        "layer_id": layer_id,
        "mode": "context_prefill_sliding_window",
        "sequence_length": seq_len,
        "compressor": compressor_report,
        "indexer": indexer_report,
        "rope": {
            "theta": theta,
            "original_seq_len": original_seq_len,
            "yarn_factor": YARN_FACTOR if original_seq_len else 0,
            "kv_nonrope_act_quant_block": 64,
        },
        "bytes_read_upper_bound": bytes_read,
        "routes": last_routes,
        "tid2eid_route": last_tid_route,
        "missing_experts": sorted(missing_experts),
        "expert_summaries": last_expert_summaries,
        "timings_seconds": timings,
        "hc": {"attn": last_attn_hc_report, "ffn": last_ffn_hc_report},
        "vectors": {**last_attn_vectors, **last_ffn_vectors},
    }
    return final_seq, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DeepSeek multi-layer math smoke")
    parser.add_argument("--index", type=Path, default=Path("models/wohper/DeepSeek-V4-Flash.RAW.L0-L2-SPLIT-GLOBAL-CATALOG22/dense_core.tensor_index.json"))
    parser.add_argument("--shards", type=Path)
    parser.add_argument("--out", type=Path, default=Path("state/deepseek_v4_multilayer_math_smoke_2026-07-05.json"))
    parser.add_argument("--token-id", type=int, default=42)
    parser.add_argument("--context-token-ids", default="", help="Comma-separated prompt/context token ids; logits are taken from the final position.")
    parser.add_argument("--layer-count", type=int, default=2)
    parser.add_argument("--scan-vocab", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--extra-token-ids", default="")
    parser.add_argument("--lmhead-chunk-rows", type=int, default=1024)
    parser.add_argument("--skip-lmhead-when-blocked", action="store_true")
    parser.add_argument("--compact-output", action="store_true", help="Omit large vector summaries from the report.")
    return parser.parse_args()


def compact_layer_report(layer: dict[str, Any]) -> dict[str, Any]:
    keep = {
        "layer_id",
        "mode",
        "sequence_length",
        "compressor",
        "indexer",
        "rope",
        "bytes_read_upper_bound",
        "routes",
        "tid2eid_route",
        "missing_experts",
        "timings_seconds",
    }
    return {key: layer[key] for key in keep if key in layer}


def main() -> int:
    started = time.perf_counter()
    args = parse_args()
    index = base.load_json(args.index)
    core = base.resolve_core(args.index, index["core_file"])
    shards_path = args.shards or core.with_suffix(".shards.json")
    shards = base.load_json(shards_path)
    tensors = {tensor["name"]: tensor for tensor in index.get("tensors", [])}
    expert_by_key = {
        (int(item["layer_id"]), int(item["expert_id"])): item
        for item in shards.get("experts", [])
    }

    bytes_read_upper_bound = 0
    context_token_ids = parse_context_token_ids(args.context_token_ids)
    token_ids = context_token_ids or [args.token_id]
    hidden_items = []
    embedding_started = time.perf_counter()
    for token_id in token_ids:
        embed = base.bf16_to_f32(base.read_row(core, tensors["embed.weight"], token_id))
        hidden_items.append(np.repeat(embed[None, :], HC_MULT, axis=0).astype(np.float32))
        bytes_read_upper_bound += int(tensors["embed.weight"]["row_bytes"])
    embedding_seconds = time.perf_counter() - embedding_started
    if context_token_ids:
        hidden_seq = np.stack(hidden_items, axis=0)
        hidden = hidden_seq[-1]
    else:
        hidden_seq = None
        hidden = hidden_items[0]
    layer_reports = []
    blockers = []
    layer_wall_seconds = []
    for layer_id in range(args.layer_count):
        required = [
            lname(layer_id, "attn_norm.weight"),
            lname(layer_id, "ffn_norm.weight"),
            lname(layer_id, "attn.wq_a.weight"),
            lname(layer_id, "attn.wq_b.weight"),
            lname(layer_id, "attn.wkv.weight"),
            lname(layer_id, "attn.wo_a.weight"),
            lname(layer_id, "attn.wo_b.weight"),
            lname(layer_id, "ffn.gate.weight"),
            lname(layer_id, "ffn.shared_experts.w1.weight"),
            lname(layer_id, "ffn.shared_experts.w2.weight"),
            lname(layer_id, "ffn.shared_experts.w3.weight"),
        ]
        missing_tensors = [name for name in required if name not in tensors]
        if missing_tensors:
            blockers.append(f"missing_layer_{layer_id}_tensors")
            layer_reports.append({"layer_id": layer_id, "missing_tensors": missing_tensors})
            break
        layer_started = time.perf_counter()
        if hidden_seq is not None:
            hidden_seq, layer_report = layer_forward_sequence(core, tensors, expert_by_key, hidden_seq, token_ids, layer_id)
            hidden = hidden_seq[-1]
        else:
            hidden, layer_report = layer_forward(core, tensors, expert_by_key, hidden, args.token_id, layer_id)
        layer_elapsed = time.perf_counter() - layer_started
        layer_wall_seconds.append({"layer_id": layer_id, "seconds": layer_elapsed})
        timings = dict(layer_report.get("timings_seconds", {}))
        timings["main_loop_layer_wall_seconds"] = layer_elapsed
        layer_report["timings_seconds"] = timings
        bytes_read_upper_bound += int(layer_report["bytes_read_upper_bound"])
        layer_reports.append(layer_report)
        if layer_report["missing_experts"]:
            blockers.append(f"missing_layer_{layer_id}_routed_expert")

    extra_token_ids = parse_extra_token_ids(args.extra_token_ids)
    topk: list[dict[str, float | int | str]] = []
    hc_head_seconds = 0.0
    lmhead_seconds = 0.0
    if blockers and args.skip_lmhead_when_blocked:
        lm_hidden = np.zeros(4096, dtype=np.float32)
        hc_head_report = {"mode": "skipped_because_blocked"}
    else:
        head_started = time.perf_counter()
        lm_hidden, b, hc_head_report = hc_head(core, tensors, hidden)
        hc_head_seconds = time.perf_counter() - head_started
        bytes_read_upper_bound += b
        lmhead_started = time.perf_counter()
        topk, b = lm_head_candidates(
            core,
            tensors,
            lm_hidden,
            args.scan_vocab,
            args.top_k,
            extra_token_ids,
            args.lmhead_chunk_rows,
        )
        lmhead_seconds = time.perf_counter() - lmhead_started
        bytes_read_upper_bound += b
        if not topk:
            blockers.append("empty_lmhead_topk")
    if int(np.isfinite(hidden).sum()) != int(hidden.size):
        blockers.append("final_hidden_non_finite")
    if int(np.count_nonzero(hidden)) == 0:
        blockers.append("final_hidden_all_zero")

    if args.compact_output:
        layer_reports = [compact_layer_report(layer) for layer in layer_reports]
        hc_head_report = {"mode": hc_head_report.get("mode", "compact")}

    payload = {
        "format": "deepseek-v4-multilayer-math-smoke",
        "version": 1,
        "status": "ready" if not blockers else "blocked",
        "blockers": sorted(set(blockers)),
        "semantic_warnings": [
            "attention_prefill_uses_sliding_window_compressor_and_bounded_ratio4_indexer_topk",
            "mHC_numpy_path_enabled_for_validation_smoke",
            "lm_head_is_bounded_to_scan_vocab_plus_explicit_extra_token_ids",
        ],
        "index": str(args.index),
        "core": str(core),
        "shards": str(shards_path),
        "token_id": args.token_id,
        "context_token_ids": context_token_ids,
        "context_token_count": len(token_ids),
        "layer_count": args.layer_count,
        "scan_vocab": args.scan_vocab,
        "lmhead_chunk_rows": args.lmhead_chunk_rows,
        "skip_lmhead_when_blocked": args.skip_lmhead_when_blocked,
        "compact_output": args.compact_output,
        "extra_token_ids": extra_token_ids,
        "bytes_read_upper_bound": bytes_read_upper_bound,
        "elapsed_seconds": time.perf_counter() - started,
        "profile": {
            "embedding_seconds": embedding_seconds,
            "layer_wall_seconds": layer_wall_seconds,
            "hc_head_seconds": hc_head_seconds,
            "lmhead_seconds": lmhead_seconds,
        },
        "final_hidden": base.summary(lm_hidden) if not args.compact_output else {"omitted": True},
        "final_hidden_hc": base.summary(hidden.reshape(-1)) if not args.compact_output else {"omitted": True},
        "hc_head": hc_head_report,
        "layers": layer_reports,
        "bounded_lmhead_topk": topk,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not blockers else 3


if __name__ == "__main__":
    raise SystemExit(main())
