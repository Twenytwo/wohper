#!/usr/bin/env python3
"""Build a runtime MODEL from an existing dense core and targeted expert shards."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import convert_safetensors as zc
import merge_smoke_models as merge


@dataclass
class ShardExpert:
    layer_id: int
    expert_id: int
    source_path: Path
    runtime_path: str
    disk_bytes: int
    payload_bytes: int
    dequant_bytes: int
    quant_format: int
    checksum: int


def relpath_for_runtime(source: Path, runtime_base: Path) -> str:
    try:
        rel = os.path.relpath(source, runtime_base)
    except ValueError:
        return str(source)
    return rel.replace("\\", "/")


def load_expert_shards(index_path: Path, runtime_base: Path) -> dict[tuple[int, int], ShardExpert]:
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    index_base = index_path.parent
    shards: dict[tuple[int, int], ShardExpert] = {}
    for item in payload.get("experts", []):
        raw_path = Path(str(item["path"]))
        source_path = raw_path if raw_path.is_absolute() else index_base / raw_path
        layer_id = int(item["layer_id"])
        expert_id = int(item["expert_id"])
        shards[(layer_id, expert_id)] = ShardExpert(
            layer_id=layer_id,
            expert_id=expert_id,
            source_path=source_path,
            runtime_path=relpath_for_runtime(source_path, runtime_base),
            disk_bytes=int(item["disk_bytes"]),
            payload_bytes=int(item["payload_bytes"]),
            dequant_bytes=int(item.get("dequant_bytes", 0)),
            quant_format=int(item.get("quant_format", zc.QUANT_INT4)),
            checksum=int(item.get("checksum", 0)),
        )
    return shards


def selected_compute_layers(base: merge.SourceModel, wanted: list[int] | None) -> list[merge.SourceLayer]:
    wanted_set = set(wanted or [])
    layers = []
    for layer in base.layers:
        if layer.layer_id == 0:
            continue
        if wanted_set and layer.layer_id not in wanted_set:
            continue
        layers.append(layer)
    if wanted_set:
        found = {layer.layer_id for layer in layers}
        missing = sorted(wanted_set - found)
        if missing:
            raise ValueError(f"base core missing requested compute layers: {missing}")
    if not layers:
        raise ValueError("no compute layers selected")
    return layers


def parse_compute_layers(values: list[int] | None) -> list[int] | None:
    if not values:
        return None
    return sorted(set(values))


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retarget a dense core to a targeted expert shard plan")
    parser.add_argument("--base-core", type=Path, required=True)
    parser.add_argument("--expert-shard-index", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--index-out", type=Path, default=None)
    parser.add_argument("--shard-index-out", type=Path, default=None)
    parser.add_argument("--compute-layer", type=int, action="append", default=None)
    parser.add_argument("--active-experts", type=int, default=2)
    return parser.parse_args(list(argv))


def main(argv: Iterable[str] = sys.argv[1:]) -> int:
    args = parse_args(argv)
    base = merge.read_source(args.base_core)
    runtime_base = args.out.parent
    expert_shards = load_expert_shards(args.expert_shard_index, runtime_base)
    compute_layers = selected_compute_layers(base, parse_compute_layers(args.compute_layer))
    global_layers = [layer for layer in base.layers if layer.layer_id == 0]
    if not global_layers:
        raise ValueError("base core has no global layer 0 blocks")

    missing: list[tuple[int, int]] = []
    for layer in compute_layers:
        planned = sorted(expert_id for (layer_id, expert_id) in expert_shards if layer_id == layer.layer_id)
        if not planned:
            missing.append((layer.layer_id, -1))
    if missing:
        raise ValueError(f"expert plan missing layers: {missing[:12]}")

    layer_plans: list[zc.LayerPlan] = []
    expert_plans: list[zc.ExpertPlan] = []
    args.out.parent.mkdir(parents=True, exist_ok=True)
    manifest_expert_count = sum(
        1
        for layer in compute_layers
        for key in expert_shards
        if key[0] == layer.layer_id
    )
    manifest_size = (
        zc.MANIFEST_HEADER_SIZE
        + (len(global_layers) + len(compute_layers)) * zc.LAYER_DESC_SIZE
        + manifest_expert_count * zc.EXPERT_DESC_SIZE
    )

    with args.out.open("wb") as out:
        manifest_offset = zc.reserve_header_and_manifest(out, manifest_size)

        for global_layer in global_layers:
            offset, disk, payload = merge.copy_dense_block(base, global_layer, out)
            layer_plans.append(
                zc.LayerPlan(
                    layer_id=global_layer.layer_id,
                    dense_offset=offset,
                    dense_disk_bytes=disk,
                    dense_payload_bytes=payload,
                    dense_dequant_bytes=global_layer.dense_dequant_bytes,
                    first_expert_index=0,
                    num_experts=0,
                    checksum=global_layer.checksum,
                    block_type=zc.LAYER_BLOCK_GLOBAL_AUX,
                )
            )

        for layer in compute_layers:
            first_expert_index = len(expert_plans)
            offset, disk, payload = merge.copy_dense_block(base, layer, out)
            layer_experts = [
                shard
                for (layer_id, _), shard in sorted(expert_shards.items())
                if layer_id == layer.layer_id
            ]
            for shard in layer_experts:
                expert_plans.append(
                    zc.ExpertPlan(
                        layer_id=shard.layer_id,
                        expert_id=shard.expert_id,
                        disk_offset=0,
                        disk_bytes=shard.disk_bytes,
                        payload_bytes=shard.payload_bytes,
                        dequant_bytes=shard.dequant_bytes,
                        quant_format=shard.quant_format,
                        route_rank_hint=shard.expert_id,
                        checksum=shard.checksum,
                    )
                )
            layer_plans.append(
                zc.LayerPlan(
                    layer_id=layer.layer_id,
                    dense_offset=offset,
                    dense_disk_bytes=disk,
                    dense_payload_bytes=payload,
                    dense_dequant_bytes=layer.dense_dequant_bytes,
                    first_expert_index=first_expert_index,
                    num_experts=len(layer_experts),
                    checksum=layer.checksum,
                )
            )

        file_size = out.tell()
        manifest_payload = zc.pack_manifest(layer_plans, expert_plans, zc.QUANT_INT4)
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
            base.header[10],
            base.header[11],
            len(layer_plans),
            base.header[13],
            base.header[14],
            base.header[15],
            base.header[16],
            args.active_experts,
            zc.ALIGN_2MB,
            zc.QUANT_INT4,
            manifest_checksum,
            0,
        )
        out.seek(0)
        out.write(header)
        out.seek(manifest_offset)
        out.write(manifest_payload)

    shard_index = args.shard_index_out or args.out.with_suffix(".shards.json")
    shard_entries = []
    for expert in expert_plans:
        shard = expert_shards[(expert.layer_id, expert.expert_id)]
        shard_entries.append(
            zc.ExpertShardPlan(
                layer_id=shard.layer_id,
                expert_id=shard.expert_id,
                path=shard.runtime_path,
                disk_bytes=shard.disk_bytes,
                payload_bytes=shard.payload_bytes,
                dequant_bytes=shard.dequant_bytes,
                quant_format=shard.quant_format,
                checksum=shard.checksum,
            )
        )
    zc.write_sharded_index(
        shard_index,
        core_path=args.out,
        experts_dir=runtime_base,
        selected_layers=[layer.layer_id for layer in compute_layers],
        num_layers_total=base.header[12],
        experts_per_layer=max((layer.num_experts for layer in layer_plans), default=0),
        layer_plans=layer_plans,
        expert_shards=shard_entries,
        quant_format=zc.QUANT_INT4,
        metadata={"source": "retarget_expert_plan_model", "expert_shard_index": str(args.expert_shard_index)},
    )
    zc.write_debug_index(args.index_out or args.out.with_suffix(".index.json"), zc.ConversionPlan(), layer_plans, expert_plans)
    print(f"retargeted_model={args.out}")
    print(f"layers={len(layer_plans)}")
    print(f"compute_layers={len(compute_layers)}")
    print(f"experts={len(expert_plans)}")
    print(f"shard_index={shard_index}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
