#!/usr/bin/env python3
"""Merge small Wohper MODEL.bin smoke slices.

This is intentionally narrow: it builds a validation model composed of:

  layer 0: global tensors (embed_tokens, final norm, lm_head)
  one or more compute layers: real GLM-DSA attention/router + selected MoE experts

It avoids reconverting tensors and copies already aligned dense blocks.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import convert_safetensors as zc


@dataclass
class SourceLayer:
    layer_id: int
    block_type: int
    dense_offset: int
    dense_disk_bytes: int
    dense_payload_bytes: int
    dense_dequant_bytes: int
    first_expert_index: int
    num_experts: int
    checksum: int


@dataclass
class SourceExpert:
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
class SourceModel:
    path: Path
    header: tuple
    layers: list[SourceLayer]
    experts: list[SourceExpert]
    shards: dict[tuple[int, int], dict]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge Wohper smoke MODEL.bin slices")
    parser.add_argument(
        "--global-core",
        type=Path,
        action="append",
        required=True,
        help="Global core to merge. Can be passed multiple times for vocab row shards.",
    )
    parser.add_argument(
        "--layer-core",
        type=Path,
        action="append",
        required=True,
        help="Compute core to merge. Can be passed multiple times.",
    )
    parser.add_argument(
        "--compute-layer",
        type=int,
        action="append",
        default=None,
        help="Physical compute layer to include. Defaults to all layers from --layer-core except layer 0.",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--index-out", type=Path, default=None)
    parser.add_argument("--shard-index-out", type=Path, default=None)
    parser.add_argument(
        "--experts-out-dir",
        type=Path,
        default=None,
        help="Optional directory where referenced expert shard files are copied.",
    )
    parser.add_argument(
        "--remote-fetch-endpoint",
        default=None,
        help="Optional expert fetch endpoint to persist in the shard index.",
    )
    return parser.parse_args()


def read_source(path: Path) -> SourceModel:
    with path.open("rb") as handle:
        header = zc.ENGINE_HEADER_STRUCT.unpack(handle.read(zc.ENGINE_HEADER_SIZE))
        if header[0] != zc.MODEL_MAGIC:
            raise ValueError(f"{path} is not a Wohper core file")
        manifest_offset = header[4]
        handle.seek(manifest_offset)
        manifest_header = zc.MANIFEST_HEADER_STRUCT.unpack(
            handle.read(zc.MANIFEST_HEADER_SIZE)
        )
        layer_count, expert_count, _, _, layer_desc_offset, expert_desc_offset, _ = manifest_header

        layers: list[SourceLayer] = []
        handle.seek(manifest_offset + layer_desc_offset)
        for _ in range(layer_count):
            fields = zc.LAYER_DESC_STRUCT.unpack(handle.read(zc.LAYER_DESC_SIZE))
            layers.append(
                SourceLayer(
                    layer_id=fields[0],
                    block_type=fields[1],
                    dense_offset=fields[2],
                    dense_disk_bytes=fields[3],
                    dense_payload_bytes=fields[4],
                    dense_dequant_bytes=fields[5],
                    first_expert_index=fields[8],
                    num_experts=fields[9],
                    checksum=fields[12],
                )
            )

        experts: list[SourceExpert] = []
        handle.seek(manifest_offset + expert_desc_offset)
        for _ in range(expert_count):
            fields = zc.EXPERT_DESC_STRUCT.unpack(handle.read(zc.EXPERT_DESC_SIZE))
            experts.append(
                SourceExpert(
                    layer_id=fields[0],
                    expert_id=fields[1],
                    disk_offset=fields[2],
                    disk_bytes=fields[3],
                    payload_bytes=fields[4],
                    dequant_bytes=fields[5],
                    quant_format=fields[6],
                    route_rank_hint=fields[7],
                    checksum=fields[8],
                )
            )

    return SourceModel(path=path, header=header, layers=layers, experts=experts, shards=read_shards(path))


def read_shards(path: Path) -> dict[tuple[int, int], dict]:
    shard_path = path.with_suffix(".shards.json")
    if not shard_path.exists():
        return {}
    payload = json.loads(shard_path.read_text(encoding="utf-8"))
    shards: dict[tuple[int, int], dict] = {}
    for item in payload.get("experts", []):
        shards[(int(item["layer_id"]), int(item["expert_id"]))] = item
    return shards


def copy_dense_block(src: SourceModel, layer: SourceLayer, out) -> tuple[int, int, int]:
    offset = zc.align_up(out.tell())
    zc.write_padding(out, offset)
    with src.path.open("rb") as handle:
        handle.seek(layer.dense_offset)
        remaining = layer.dense_disk_bytes
        while remaining:
            chunk = handle.read(min(8 * 1024 * 1024, remaining))
            if not chunk:
                raise EOFError(f"truncated dense block in {src.path}")
            out.write(chunk)
            remaining -= len(chunk)
    return offset, layer.dense_disk_bytes, layer.dense_payload_bytes


def choose_single_layer(src: SourceModel, wanted_layer_id: int | None = None) -> SourceLayer:
    if wanted_layer_id is not None:
        for layer in src.layers:
            if layer.layer_id == wanted_layer_id:
                return layer
        raise ValueError(f"{src.path} does not contain layer {wanted_layer_id}")
    if len(src.layers) != 1:
        raise ValueError(f"{src.path} should contain exactly one layer")
    return src.layers[0]


def choose_compute_layers(sources: list[SourceModel], wanted: list[int] | None) -> list[tuple[SourceModel, SourceLayer]]:
    selected: list[tuple[SourceModel, SourceLayer]] = []
    wanted_set = set(wanted or [])
    for src in sources:
        for layer in src.layers:
            if layer.layer_id == 0 and not wanted_set:
                continue
            if wanted_set and layer.layer_id not in wanted_set:
                continue
            selected.append((src, layer))
    if wanted_set:
        found = {layer.layer_id for _, layer in selected}
        missing = sorted(wanted_set - found)
        if missing:
            raise ValueError(f"missing requested compute layers: {missing}")
    if not selected:
        raise ValueError("no compute layers selected")
    selected.sort(key=lambda item: item[1].layer_id)
    return selected


def experts_for_layer(src: SourceModel, layer: SourceLayer) -> list[SourceExpert]:
    start = layer.first_expert_index
    end = start + layer.num_experts
    return src.experts[start:end]


def main() -> int:
    args = parse_args()
    global_sources = [read_source(path) for path in args.global_core]
    header_src = global_sources[0]
    layer_sources = [read_source(path) for path in args.layer_core]
    global_layers = [(src, choose_single_layer(src, 0)) for src in global_sources]
    compute_layers = choose_compute_layers(layer_sources, args.compute_layer)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    expert_count = sum(layer.num_experts for _, layer in compute_layers)
    manifest_size = (
        zc.MANIFEST_HEADER_SIZE
        + (len(global_layers) + len(compute_layers)) * zc.LAYER_DESC_SIZE
        + expert_count * zc.EXPERT_DESC_SIZE
    )
    layer_plans: list[zc.LayerPlan] = []
    expert_plans: list[zc.ExpertPlan] = []
    expert_sources: list[tuple[SourceModel, SourceExpert]] = []

    with args.out.open("wb") as out:
        manifest_offset = zc.reserve_header_and_manifest(out, manifest_size)

        for global_src, global_layer in global_layers:
            global_offset, global_disk, global_payload = copy_dense_block(global_src, global_layer, out)
            layer_plans.append(
                zc.LayerPlan(
                    layer_id=global_layer.layer_id,
                    dense_offset=global_offset,
                    dense_disk_bytes=global_disk,
                    dense_payload_bytes=global_payload,
                    dense_dequant_bytes=global_layer.dense_dequant_bytes,
                    first_expert_index=0,
                    num_experts=0,
                    checksum=global_layer.checksum,
                    block_type=zc.LAYER_BLOCK_GLOBAL_AUX,
                )
            )

        for layer_src, compute_layer in compute_layers:
            first_expert_index = len(expert_plans)
            layer_offset, layer_disk, layer_payload = copy_dense_block(layer_src, compute_layer, out)
            layer_experts = experts_for_layer(layer_src, compute_layer)
            for expert in layer_experts:
                expert_sources.append((layer_src, expert))
                expert_plans.append(
                    zc.ExpertPlan(
                        layer_id=expert.layer_id,
                        expert_id=expert.expert_id,
                        disk_offset=expert.disk_offset,
                        disk_bytes=expert.disk_bytes,
                        payload_bytes=expert.payload_bytes,
                        dequant_bytes=expert.dequant_bytes,
                        quant_format=expert.quant_format,
                        route_rank_hint=expert.route_rank_hint,
                        checksum=expert.checksum,
                    )
                )
            layer_plans.append(
                zc.LayerPlan(
                    layer_id=compute_layer.layer_id,
                    dense_offset=layer_offset,
                    dense_disk_bytes=layer_disk,
                    dense_payload_bytes=layer_payload,
                    dense_dequant_bytes=compute_layer.dense_dequant_bytes,
                    first_expert_index=first_expert_index,
                    num_experts=len(layer_experts),
                    checksum=compute_layer.checksum,
                    block_type=zc.LAYER_BLOCK_COMPUTE,
                )
            )

        manifest_payload = zc.pack_manifest(layer_plans, expert_plans, zc.QUANT_INT4)
        manifest_checksum = zc.checksum64_bytes(manifest_payload)
        file_size = zc.align_up(out.tell())
        zc.write_padding(out, file_size)
        header = zc.ENGINE_HEADER_STRUCT.pack(
            zc.MODEL_MAGIC,
            zc.FORMAT_VERSION,
            1,
            file_size,
            manifest_offset,
            len(manifest_payload),
            0,
            0,
            0,
            0,
            header_src.header[10],
            header_src.header[11],
            len(layer_plans),
            header_src.header[13],
            header_src.header[14],
            header_src.header[15],
            max((layer.num_experts for _, layer in compute_layers), default=0),
            max((src.header[17] for src, _ in compute_layers), default=0),
            zc.ALIGN_2MB,
            zc.QUANT_INT4,
            manifest_checksum,
            0,
        )
        out.seek(0)
        out.write(header)
        out.seek(manifest_offset)
        out.write(manifest_payload)

    if args.experts_out_dir:
        copy_expert_shards(args.experts_out_dir, expert_sources)
    write_shard_index(args, expert_sources)
    write_debug_index(args, layer_plans, expert_plans)
    print(f"merged_core={args.out}")
    print(f"size={args.out.stat().st_size}")
    print("layers=" + ",".join(str(layer.layer_id) for layer in layer_plans))
    print(f"experts={len(expert_plans)}")
    return 0


def copy_expert_shards(target_dir: Path, expert_sources: list[tuple[SourceModel, SourceExpert]]) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for src, expert in expert_sources:
        source_path = resolve_expert_path(src, expert)
        if not source_path or not source_path.exists():
            continue
        target_path = target_dir / f"layer{expert.layer_id}_expert{expert.expert_id}.zcblk"
        if target_path.resolve() != source_path.resolve():
            target_path.write_bytes(source_path.read_bytes())


def resolve_expert_path(src: SourceModel, expert: SourceExpert) -> Path | None:
    shard = src.shards.get((expert.layer_id, expert.expert_id))
    if shard:
        return src.path.parent / shard["path"]
    candidate = src.path.parent / "experts" / f"layer{expert.layer_id}_expert{expert.expert_id}.zcblk"
    if candidate.exists():
        return candidate
    return None


def write_shard_index(args: argparse.Namespace, expert_sources: list[tuple[SourceModel, SourceExpert]]) -> None:
    shard_out = args.shard_index_out or args.out.with_suffix(".shards.json")
    experts = []
    for src, expert in expert_sources:
        source = src.shards.get((expert.layer_id, expert.expert_id))
        default_path = f"experts/layer{expert.layer_id}_expert{expert.expert_id}.zcblk"
        path = default_path if args.experts_out_dir else (source["path"] if source else default_path)
        experts.append(
            {
                "layer_id": expert.layer_id,
                "expert_id": expert.expert_id,
                "path": path,
                "disk_bytes": expert.disk_bytes,
                "payload_bytes": expert.payload_bytes,
                "dequant_bytes": expert.dequant_bytes,
                "quant_format": expert.quant_format,
                "checksum": expert.checksum,
            }
        )
    payload = {"format": "wohper-sharded-experts", "version": 1, "experts": experts}
    if args.remote_fetch_endpoint:
        payload["remote_fetch"] = {
            "enabled": True,
            "endpoint_template": args.remote_fetch_endpoint.rstrip("/"),
            "path_template": "experts/layer{layer_id}_expert{expert_id}.zcblk",
        }
    Path(shard_out).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_debug_index(args: argparse.Namespace, layers: list[zc.LayerPlan], experts: list[zc.ExpertPlan]) -> None:
    if not args.index_out:
        return
    payload = {
        "format": "wohper-merged-smoke-index",
        "layers": [layer.__dict__ for layer in layers],
        "experts": [expert.__dict__ for expert in experts],
    }
    Path(args.index_out).write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
