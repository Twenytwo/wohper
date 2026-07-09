#!/usr/bin/env python3
"""Generate a fake MODEL.bin matching zc_infer_core's Rust disk structs.

The file is intentionally shaped like a MoE model:

- one EngineHeader;
- one ManifestHeader;
- LayerBlockDescDisk[num_layers];
- ExpertBlockDescDisk[num_layers * experts_per_layer];
- 2MB-aligned dense blocks;
- 2MB-aligned expert blocks.

It is not a valid neural network. It is a deterministic I/O target for testing:

- manifest parsing;
- 2MB alignment validation;
- O_DIRECT reads;
- io_uring fixed-buffer reads;
- conditional expert fetch scheduling.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import random
import struct
from dataclasses import dataclass
from pathlib import Path


ALIGN_2MB = 2 * 1024 * 1024
MODEL_MAGIC = b"ZCINF01\0"
FORMAT_VERSION = 1

# Keep these byte layouts synchronized with engine/zc_infer_core/src/model_format.rs.
ENGINE_HEADER_STRUCT = struct.Struct("<8sIIQQQQQQQIIIIIIIIIIQQ")
MANIFEST_HEADER_STRUCT = struct.Struct("<IIIIQQQ")
LAYER_DESC_STRUCT = struct.Struct("<IIQQQQIIIIIIQ")
EXPERT_DESC_STRUCT = struct.Struct("<IIQQQQIIQ")

ENGINE_HEADER_SIZE = ENGINE_HEADER_STRUCT.size  # 128
MANIFEST_HEADER_SIZE = MANIFEST_HEADER_STRUCT.size  # 40
LAYER_DESC_SIZE = LAYER_DESC_STRUCT.size  # 72
EXPERT_DESC_SIZE = EXPERT_DESC_STRUCT.size  # 56


@dataclass(frozen=True)
class LayerPlan:
    layer_id: int
    dense_offset: int
    dense_disk_bytes: int
    dense_payload_bytes: int
    dense_dequant_bytes: int
    first_expert_index: int
    num_experts: int
    checksum: int


@dataclass(frozen=True)
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


def align_up(value: int, alignment: int = ALIGN_2MB) -> int:
    return (value + alignment - 1) // alignment * alignment


def checksum64(data: bytes) -> int:
    digest = hashlib.blake2b(data, digest_size=8).digest()
    return int.from_bytes(digest, "little")


def fake_block_payload(size: int, *, seed: int, randomize: bool) -> bytes:
    if size <= 0:
        return b""
    if not randomize:
        # Deterministic non-zero-ish pattern without expensive random generation.
        pattern = seed.to_bytes(8, "little", signed=False)
        return (pattern * ((size + len(pattern) - 1) // len(pattern)))[:size]

    rng = random.Random(seed)
    chunk = bytearray(min(size, 1024 * 1024))
    out = bytearray()
    remaining = size
    while remaining:
        n = min(remaining, len(chunk))
        for i in range(n):
            chunk[i] = rng.getrandbits(8)
        out += chunk[:n]
        remaining -= n
    return bytes(out)


def write_padding(file, target_offset: int) -> None:
    current = file.tell()
    if target_offset < current:
        raise ValueError(f"target offset {target_offset} is before current offset {current}")
    file.write(b"\0" * (target_offset - current))


def write_aligned_block(file, payload: bytes) -> tuple[int, int, int, int]:
    offset = align_up(file.tell())
    write_padding(file, offset)
    file.write(payload)
    payload_size = len(payload)
    disk_size = align_up(payload_size)
    file.write(b"\0" * (disk_size - payload_size))
    return offset, disk_size, payload_size, checksum64(payload)


def reserve_header_and_manifest(file, manifest_size: int) -> int:
    file.write(b"\0" * ENGINE_HEADER_SIZE)
    manifest_offset = align_up(file.tell())
    write_padding(file, manifest_offset)
    file.write(b"\0" * manifest_size)
    return manifest_offset


def build_model(
    out_path: Path,
    *,
    layers: int,
    experts_per_layer: int,
    dense_payload_bytes: int,
    expert_payload_bytes: int,
    hidden_size: int,
    heads: int,
    kv_heads: int,
    active_experts: int,
    randomize: bool,
) -> None:
    if layers <= 0:
        raise ValueError("--layers must be > 0")
    if experts_per_layer <= 0:
        raise ValueError("--experts-per-layer must be > 0")
    if active_experts <= 0 or active_experts > experts_per_layer:
        raise ValueError("--active-experts must be in 1..experts_per_layer")

    total_experts = layers * experts_per_layer
    manifest_size = (
        MANIFEST_HEADER_SIZE
        + layers * LAYER_DESC_SIZE
        + total_experts * EXPERT_DESC_SIZE
    )

    layer_plans: list[LayerPlan] = []
    expert_plans: list[ExpertPlan] = []

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as file:
        manifest_offset = reserve_header_and_manifest(file, manifest_size)

        for layer_id in range(layers):
            dense_seed = 0xD300_0000 + layer_id
            dense_payload = fake_block_payload(
                dense_payload_bytes,
                seed=dense_seed,
                randomize=randomize,
            )
            dense_offset, dense_disk, dense_payload_size, dense_checksum = write_aligned_block(
                file,
                dense_payload,
            )

            first_expert_index = len(expert_plans)
            for expert_id in range(experts_per_layer):
                expert_seed = 0xE000_0000 + layer_id * 4096 + expert_id
                expert_payload = fake_block_payload(
                    expert_payload_bytes,
                    seed=expert_seed,
                    randomize=randomize,
                )
                expert_offset, expert_disk, expert_payload_size, expert_checksum = (
                    write_aligned_block(file, expert_payload)
                )
                expert_plans.append(
                    ExpertPlan(
                        layer_id=layer_id,
                        expert_id=expert_id,
                        disk_offset=expert_offset,
                        disk_bytes=expert_disk,
                        payload_bytes=expert_payload_size,
                        dequant_bytes=expert_payload_size * 4,
                        quant_format=2,  # fake q2_moe
                        route_rank_hint=expert_id,
                        checksum=expert_checksum,
                    )
                )

            layer_plans.append(
                LayerPlan(
                    layer_id=layer_id,
                    dense_offset=dense_offset,
                    dense_disk_bytes=dense_disk,
                    dense_payload_bytes=dense_payload_size,
                    dense_dequant_bytes=dense_payload_size * 4,
                    first_expert_index=first_expert_index,
                    num_experts=experts_per_layer,
                    checksum=dense_checksum,
                )
            )

        file_size = file.tell()

        manifest_payload = pack_manifest(layer_plans, expert_plans)
        if len(manifest_payload) != manifest_size:
            raise AssertionError(
                f"manifest size mismatch: got {len(manifest_payload)}, expected {manifest_size}"
            )

        manifest_checksum = checksum64(manifest_payload)
        file_checksum = 0  # Placeholder: full-file checksum would require a second pass.
        header = ENGINE_HEADER_STRUCT.pack(
            MODEL_MAGIC,
            FORMAT_VERSION,
            0,  # little endian
            file_size,
            manifest_offset,
            manifest_size,
            0,  # tokenizer_offset, omitted in fake file
            0,  # tokenizer_size
            0,  # router_metadata_offset, omitted in fake file
            0,  # router_metadata_size
            1,  # model_family fake GLM-like
            1,  # architecture MoE
            layers,
            hidden_size,
            heads,
            kv_heads,
            experts_per_layer,
            active_experts,
            ALIGN_2MB,
            2,  # fake q2_moe disk format
            manifest_checksum,
            file_checksum,
        )

        file.seek(0)
        file.write(header)
        file.seek(manifest_offset)
        file.write(manifest_payload)

    validate_generated_file(out_path, layer_plans, expert_plans, manifest_offset)
    print_summary(out_path, layer_plans, expert_plans, manifest_offset, manifest_size)


def pack_manifest(layer_plans: list[LayerPlan], expert_plans: list[ExpertPlan]) -> bytes:
    layer_count = len(layer_plans)
    expert_count = len(expert_plans)
    layer_desc_offset = MANIFEST_HEADER_SIZE
    expert_desc_offset = layer_desc_offset + layer_count * LAYER_DESC_SIZE
    tensor_desc_offset = expert_desc_offset + expert_count * EXPERT_DESC_SIZE

    payload = bytearray()
    payload += MANIFEST_HEADER_STRUCT.pack(
        layer_count,
        expert_count,
        0,  # tensor_count
        0,  # reserved
        layer_desc_offset,
        expert_desc_offset,
        tensor_desc_offset,
    )

    for layer in layer_plans:
        payload += LAYER_DESC_STRUCT.pack(
            layer.layer_id,
            0,  # flags
            layer.dense_offset,
            layer.dense_disk_bytes,
            layer.dense_payload_bytes,
            layer.dense_dequant_bytes,
            0,  # tensor_count
            0,  # first_tensor_index
            layer.first_expert_index,
            layer.num_experts,
            2,  # fake q2_moe
            1,  # checksum_kind blake2b64
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


def validate_generated_file(
    out_path: Path,
    layers: list[LayerPlan],
    experts: list[ExpertPlan],
    manifest_offset: int,
) -> None:
    assert manifest_offset % ALIGN_2MB == 0
    for layer in layers:
        assert layer.dense_offset % ALIGN_2MB == 0
        assert layer.dense_disk_bytes % ALIGN_2MB == 0
    for expert in experts:
        assert expert.disk_offset % ALIGN_2MB == 0
        assert expert.disk_bytes % ALIGN_2MB == 0
    assert out_path.stat().st_size % ALIGN_2MB == 0


def print_summary(
    out_path: Path,
    layers: list[LayerPlan],
    experts: list[ExpertPlan],
    manifest_offset: int,
    manifest_size: int,
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


def parse_size(text: str) -> int:
    raw = text.strip().lower().replace("_", "")
    units = {
        "k": 1024,
        "kb": 1024,
        "m": 1024**2,
        "mb": 1024**2,
        "g": 1024**3,
        "gb": 1024**3,
    }
    for suffix, mul in sorted(units.items(), key=lambda item: -len(item[0])):
        if raw.endswith(suffix):
            return int(float(raw[: -len(suffix)]) * mul)
    return int(raw)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("MODEL.bin"))
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--experts-per-layer", type=int, default=8)
    parser.add_argument("--active-experts", type=int, default=2)
    parser.add_argument("--dense-size", type=parse_size, default=parse_size("8mb"))
    parser.add_argument("--expert-size", type=parse_size, default=parse_size("16mb"))
    parser.add_argument("--hidden-size", type=int, default=8192)
    parser.add_argument("--heads", type=int, default=64)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument(
        "--random",
        action="store_true",
        help="Fill blocks with deterministic pseudo-random bytes. Default is a faster repeating pattern.",
    )
    args = parser.parse_args()

    build_model(
        args.out,
        layers=args.layers,
        experts_per_layer=args.experts_per_layer,
        dense_payload_bytes=args.dense_size,
        expert_payload_bytes=args.expert_size,
        hidden_size=args.hidden_size,
        heads=args.heads,
        kv_heads=args.kv_heads,
        active_experts=args.active_experts,
        randomize=args.random,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
