#!/usr/bin/env python3
"""Generate a small ZCBLK01-backed Wohper MODEL.bin for public smoke tests.

The large benchmark generator in tools/make_fake_model.py is useful for raw
NVMe stress tests, but it writes opaque payload bytes. This script writes real
ZCBLK01 tensor blocks so the runtime parser and fused INT4 GEMV path can be
exercised without downloading GLM-5.2.

Default output is roughly 54 MB:

    2 MB header gap + 2 MB manifest gap + 25 x 2 MB blocks

Use it from the repository root:

    python3 scripts/generate_dummy_model.py --out projects/MODEL.dummy.zcblk01.bin
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from dataclasses import dataclass
from pathlib import Path


ALIGN_2MB = 2 * 1024 * 1024
MODEL_MAGIC = b"ZCINF01\0"
BLOCK_MAGIC = b"ZCBLK01\0"
FORMAT_VERSION = 1
QUANT_INT4 = 4
CHECKSUM_BLAKE2B64 = 1

ENGINE_HEADER_STRUCT = struct.Struct("<8sIIQQQQQQQIIIIIIIIIIQQ")
MANIFEST_HEADER_STRUCT = struct.Struct("<IIIIQQQ")
LAYER_DESC_STRUCT = struct.Struct("<IIQQQQIIIIIIQ")
EXPERT_DESC_STRUCT = struct.Struct("<IIQQQQIIQ")
BLOCK_HEADER_STRUCT = struct.Struct("<8sIIIIQQ")
TENSOR_RECORD_STRUCT = struct.Struct("<HHIIQQQQff")

DTYPE_FLOAT32 = 12

TENSOR_ROLE_QKV_PROJ = 1
TENSOR_ROLE_O_PROJ = 5
TENSOR_ROLE_GATE_PROJ = 6
TENSOR_ROLE_UP_PROJ = 7
TENSOR_ROLE_DOWN_PROJ = 8


@dataclass(frozen=True)
class DummyTensor:
    name: str
    role: int
    shape: tuple[int, ...]
    seed: int


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


def align_up(value: int, alignment: int = ALIGN_2MB) -> int:
    return (value + alignment - 1) // alignment * alignment


def checksum64(data: bytes) -> int:
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "little")


def checksum64_file(path: Path) -> int:
    h = hashlib.blake2b(digest_size=8)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return int.from_bytes(h.digest(), "little")


def write_padding(handle, target_offset: int) -> None:
    current = handle.tell()
    if target_offset < current:
        raise ValueError(f"target offset {target_offset} is before current offset {current}")
    handle.write(b"\0" * (target_offset - current))


def pack_tensor_name(name: str) -> bytes:
    raw = name.encode("utf-8")
    if len(raw) > 65535:
        raise ValueError(f"tensor name too long: {name}")
    return struct.pack("<H", len(raw)) + raw


def pack_shape(shape: tuple[int, ...]) -> bytes:
    payload = bytearray(struct.pack("<I", len(shape)))
    for dim in shape:
        payload += struct.pack("<Q", dim)
    return bytes(payload)


def element_count(shape: tuple[int, ...]) -> int:
    total = 1
    for dim in shape:
        total *= dim
    return total


def deterministic_int4_payload(elements: int, seed: int) -> bytes:
    """Pack two unsigned INT4 values per byte, low nibble first."""

    packed = bytearray((elements + 1) // 2)
    for index in range(elements):
        # Tiny deterministic pattern centered around zero_point=8.
        nibble = (seed + index * 7 + (index >> 3)) & 0x0F
        if index & 1:
            packed[index // 2] |= nibble << 4
        else:
            packed[index // 2] |= nibble
    return bytes(packed)


def build_zcblk01(tensors: list[DummyTensor]) -> tuple[bytes, int, int]:
    """Return (payload, dequant_runtime_bytes, checksum)."""

    block = bytearray(BLOCK_HEADER_STRUCT.size)
    record_table_offset = len(block)
    block += b"\0" * (len(tensors) * TENSOR_RECORD_STRUCT.size)

    names_offset = len(block)
    name_offsets: list[int] = []
    for tensor in tensors:
        name_offsets.append(len(block) - names_offset)
        block += pack_tensor_name(tensor.name)

    data_offsets: list[int] = []
    data_bytes: list[int] = []
    runtime_bytes = 0
    for tensor in tensors:
        data_offsets.append(len(block))
        values = element_count(tensor.shape)
        data = deterministic_int4_payload(values, tensor.seed)
        block += data
        data_bytes.append(len(data))
        runtime_bytes += values * 4

    shape_offsets: list[int] = []
    for tensor in tensors:
        shape_offsets.append(len(block))
        block += pack_shape(tensor.shape)

    block[0:BLOCK_HEADER_STRUCT.size] = BLOCK_HEADER_STRUCT.pack(
        BLOCK_MAGIC,
        1,
        len(tensors),
        QUANT_INT4,
        0,
        record_table_offset,
        names_offset,
    )

    cursor = record_table_offset
    for tensor, name_offset, data_offset, size, shape_offset in zip(
        tensors,
        name_offsets,
        data_offsets,
        data_bytes,
        shape_offsets,
    ):
        block[cursor : cursor + TENSOR_RECORD_STRUCT.size] = TENSOR_RECORD_STRUCT.pack(
            DTYPE_FLOAT32,
            QUANT_INT4,
            len(tensor.shape),
            tensor.role,
            name_offset,
            shape_offset,
            data_offset,
            size,
            1.0 / 8.0,
            8.0,
        )
        cursor += TENSOR_RECORD_STRUCT.size

    payload = bytes(block)
    return payload, runtime_bytes, checksum64(payload)


def dense_tensors(layer_id: int, hidden_size: int) -> list[DummyTensor]:
    shape = (hidden_size, hidden_size)
    return [
        DummyTensor(
            f"model.layers.{layer_id}.self_attn.qkv_proj.weight",
            TENSOR_ROLE_QKV_PROJ,
            shape,
            seed=layer_id * 17 + 1,
        ),
        DummyTensor(
            f"model.layers.{layer_id}.self_attn.o_proj.weight",
            TENSOR_ROLE_O_PROJ,
            shape,
            seed=layer_id * 17 + 2,
        ),
    ]


def expert_tensors(layer_id: int, expert_id: int, hidden_size: int) -> list[DummyTensor]:
    shape = (hidden_size, hidden_size)
    prefix = f"model.layers.{layer_id}.mlp.experts.{expert_id}"
    seed = layer_id * 257 + expert_id * 11
    return [
        DummyTensor(f"{prefix}.gate_proj.weight", TENSOR_ROLE_GATE_PROJ, shape, seed + 1),
        DummyTensor(f"{prefix}.up_proj.weight", TENSOR_ROLE_UP_PROJ, shape, seed + 2),
        DummyTensor(f"{prefix}.down_proj.weight", TENSOR_ROLE_DOWN_PROJ, shape, seed + 3),
    ]


def write_aligned_payload(handle, payload: bytes) -> tuple[int, int, int]:
    offset = align_up(handle.tell())
    write_padding(handle, offset)
    handle.write(payload)
    disk_bytes = align_up(len(payload))
    handle.write(b"\0" * (disk_bytes - len(payload)))
    return offset, disk_bytes, len(payload)


def pack_manifest(layer_plans: list[LayerPlan], expert_plans: list[ExpertPlan]) -> bytes:
    layer_desc_offset = MANIFEST_HEADER_STRUCT.size
    expert_desc_offset = layer_desc_offset + len(layer_plans) * LAYER_DESC_STRUCT.size
    tensor_desc_offset = expert_desc_offset + len(expert_plans) * EXPERT_DESC_STRUCT.size

    payload = bytearray()
    payload += MANIFEST_HEADER_STRUCT.pack(
        len(layer_plans),
        len(expert_plans),
        0,
        0,
        layer_desc_offset,
        expert_desc_offset,
        tensor_desc_offset,
    )

    for layer in layer_plans:
        payload += LAYER_DESC_STRUCT.pack(
            layer.layer_id,
            0,
            layer.dense_offset,
            layer.dense_disk_bytes,
            layer.dense_payload_bytes,
            layer.dense_dequant_bytes,
            0,
            0,
            layer.first_expert_index,
            layer.num_experts,
            QUANT_INT4,
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


def validate_alignment(path: Path, layers: list[LayerPlan], experts: list[ExpertPlan]) -> None:
    if path.stat().st_size % ALIGN_2MB:
        raise AssertionError("file size is not 2MB aligned")
    for layer in layers:
        if layer.dense_offset % ALIGN_2MB or layer.dense_disk_bytes % ALIGN_2MB:
            raise AssertionError(f"layer {layer.layer_id} dense block is not aligned")
    for expert in experts:
        if expert.disk_offset % ALIGN_2MB or expert.disk_bytes % ALIGN_2MB:
            raise AssertionError(f"expert {expert.layer_id}/{expert.expert_id} is not aligned")


def write_index(
    out_path: Path,
    index_path: Path,
    manifest_offset: int,
    layer_plans: list[LayerPlan],
    expert_plans: list[ExpertPlan],
) -> None:
    data = {
        "kind": "wohper_dummy_zcblk01",
        "model": str(out_path),
        "file_size": out_path.stat().st_size,
        "alignment": ALIGN_2MB,
        "manifest_offset": manifest_offset,
        "layers": [layer.__dict__ for layer in layer_plans],
        "experts": [expert.__dict__ for expert in expert_plans],
        "tensor_role_encoding": {
            "qkv_proj": TENSOR_ROLE_QKV_PROJ,
            "o_proj": TENSOR_ROLE_O_PROJ,
            "gate_proj": TENSOR_ROLE_GATE_PROJ,
            "up_proj": TENSOR_ROLE_UP_PROJ,
            "down_proj": TENSOR_ROLE_DOWN_PROJ,
        },
    }
    index_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def generate(args: argparse.Namespace) -> None:
    if args.layers <= 0:
        raise ValueError("--layers must be positive")
    if args.experts_per_layer <= 0:
        raise ValueError("--experts-per-layer must be positive")
    if args.active_experts <= 0 or args.active_experts > args.experts_per_layer:
        raise ValueError("--active-experts must be in the range 1..experts-per-layer")
    if args.hidden_size <= 0:
        raise ValueError("--hidden-size must be positive")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    layer_plans: list[LayerPlan] = []
    expert_plans: list[ExpertPlan] = []
    manifest_offset = ALIGN_2MB
    manifest_reserve = ALIGN_2MB

    with args.out.open("wb") as handle:
        handle.write(b"\0" * ENGINE_HEADER_STRUCT.size)
        write_padding(handle, manifest_offset)
        handle.write(b"\0" * manifest_reserve)

        for layer_id in range(args.layers):
            dense_payload, dense_dequant, dense_checksum = build_zcblk01(
                dense_tensors(layer_id, args.hidden_size)
            )
            dense_offset, dense_disk, dense_payload_size = write_aligned_payload(
                handle, dense_payload
            )

            first_expert_index = len(expert_plans)
            for expert_id in range(args.experts_per_layer):
                expert_payload, expert_dequant, expert_checksum = build_zcblk01(
                    expert_tensors(layer_id, expert_id, args.hidden_size)
                )
                expert_offset, expert_disk, expert_payload_size = write_aligned_payload(
                    handle, expert_payload
                )
                expert_plans.append(
                    ExpertPlan(
                        layer_id=layer_id,
                        expert_id=expert_id,
                        disk_offset=expert_offset,
                        disk_bytes=expert_disk,
                        payload_bytes=expert_payload_size,
                        dequant_bytes=expert_dequant,
                        quant_format=QUANT_INT4,
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
                    dense_dequant_bytes=dense_dequant,
                    first_expert_index=first_expert_index,
                    num_experts=args.experts_per_layer,
                    checksum=dense_checksum,
                )
            )

        file_size = handle.tell()
        manifest_payload = pack_manifest(layer_plans, expert_plans)
        if len(manifest_payload) > manifest_reserve:
            raise ValueError("manifest exceeded the 2MB dummy reserve")

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
            1,
            1,
            args.layers,
            args.hidden_size,
            args.heads,
            args.kv_heads,
            args.experts_per_layer,
            args.active_experts,
            ALIGN_2MB,
            QUANT_INT4,
            checksum64(manifest_payload),
            0,
        )

        handle.seek(0)
        handle.write(header)
        handle.seek(manifest_offset)
        handle.write(manifest_payload)

    validate_alignment(args.out, layer_plans, expert_plans)
    index_path = args.index_out or args.out.with_suffix(".index.json")
    write_index(args.out, index_path, manifest_offset, layer_plans, expert_plans)

    size_mb = args.out.stat().st_size / (1024 * 1024)
    print(f"wrote {args.out}")
    print(f"size_mb={size_mb:.2f} layers={args.layers} experts={len(expert_plans)}")
    print(f"manifest_offset={manifest_offset} file_mod_2mb={args.out.stat().st_size % ALIGN_2MB}")
    print(f"index={index_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("projects/MODEL.dummy.zcblk01.bin"),
        help="output MODEL.bin path",
    )
    parser.add_argument("--index-out", type=Path, default=None, help="optional JSON index path")
    parser.add_argument("--layers", type=int, default=5, help="dummy layer count")
    parser.add_argument("--experts-per-layer", type=int, default=4, help="dummy experts per layer")
    parser.add_argument("--active-experts", type=int, default=2, help="active experts per token")
    parser.add_argument("--hidden-size", type=int, default=128, help="dummy hidden size")
    parser.add_argument("--heads", type=int, default=8, help="dummy attention heads")
    parser.add_argument("--kv-heads", type=int, default=4, help="dummy KV heads")
    return parser.parse_args()


def main() -> None:
    generate(parse_args())


if __name__ == "__main__":
    main()
