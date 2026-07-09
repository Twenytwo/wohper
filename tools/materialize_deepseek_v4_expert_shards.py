#!/usr/bin/env python3
"""Materialize additional DeepSeek-V4 expert shards into an existing RAW slice."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import convert_safetensors as zc
import stream_convert_deepseek_v4 as ds


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_ids(value: str) -> list[int]:
    return sorted({int(part.strip()) for part in value.split(",") if part.strip()})


def expert_key(item: dict[str, Any]) -> tuple[int, int]:
    return int(item["layer_id"]), int(item["expert_id"])


class CatalogLockError(RuntimeError):
    pass


def acquire_catalog_lock(lock_path: Path, wait_seconds: float) -> int:
    deadline = time.monotonic() + max(0.0, wait_seconds)
    payload = {
        "created_at_unix": int(time.time()),
        "pid": os.getpid(),
        "purpose": "deepseek_v4_expert_catalog_materialization",
    }
    encoded = (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8")
    while True:
        try:
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, encoded)
            return fd
        except FileExistsError as exc:
            if time.monotonic() >= deadline:
                raise CatalogLockError(f"catalog lock exists: {lock_path}") from exc
            time.sleep(1.0)


def release_catalog_lock(lock_path: Path, fd: int | None) -> None:
    if fd is not None:
        os.close(fd)
    try:
        lock_path.unlink()
    except FileNotFoundError:
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize DeepSeek expert shards incrementally")
    parser.add_argument("--model-dir", type=Path, default=Path("models/deepseek-ai/DeepSeek-V4-Flash"))
    parser.add_argument("--core", type=Path, default=Path("models/wohper/DeepSeek-V4-Flash.RAW.L0-SPLIT-GLOBAL-TOP6/dense_core.bin"))
    parser.add_argument("--shards", type=Path)
    parser.add_argument("--layer-id", type=int, default=0)
    parser.add_argument("--expert-ids", required=True)
    parser.add_argument("--chunk-mb", type=int, default=16)
    parser.add_argument("--min-free-after-gb", type=float, default=250.0)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--wait-lock-seconds", type=float, default=0.0)
    parser.add_argument("--out", type=Path, default=Path("state/deepseek_v4_materialize_expert_shards_2026-07-05.json"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    shard_index_path = args.shards or args.core.with_suffix(".shards.json")
    lock_path = Path(str(shard_index_path) + ".lock")
    lock_fd: int | None = None
    if args.execute:
        try:
            lock_fd = acquire_catalog_lock(lock_path, args.wait_lock_seconds)
        except CatalogLockError as exc:
            payload = {
                "format": "deepseek-v4-materialize-expert-shards",
                "version": 1,
                "status": "blocked",
                "blockers": ["catalog_lock_exists"],
                "model_dir": str(args.model_dir),
                "core": str(args.core),
                "shards": str(shard_index_path),
                "catalog_lock": str(lock_path),
                "wait_lock_seconds": args.wait_lock_seconds,
                "execute": bool(args.execute),
                "error": str(exc),
            }
            write_json(args.out, payload)
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 4

    try:
        shard_index = load_json(shard_index_path)
        experts_dir = args.core.parent / "experts"
        requested = parse_ids(args.expert_ids)
        existing = {expert_key(item): item for item in shard_index.get("experts", [])}
        to_write = [
            expert_id
            for expert_id in requested
            if args.overwrite or (args.layer_id, expert_id) not in existing
        ]

        items = ds.build_tensor_items(args.model_dir)
        grouped: dict[int, list[ds.TensorItem]] = {}
        missing = []
        for expert_id in to_write:
            refs = [
                item
                for item in items.values()
                if item.layer_id == args.layer_id and item.expert_id == expert_id and item.role == "expert"
            ]
            if not refs:
                missing.append(expert_id)
                continue
            grouped[expert_id] = sorted(refs, key=lambda item: item.name)

        required_bytes = sum(zc.align_up(ds.estimate_block_payload(refs)) for refs in grouped.values())
        free_bytes = shutil.disk_usage(args.core.parent).free
        min_free_after = int(args.min_free_after_gb * 1024**3)
        blockers = []
        if missing:
            blockers.append("missing_source_expert")
        if free_bytes - required_bytes < min_free_after:
            blockers.append("blocked_low_disk")

        payload: dict[str, Any] = {
            "format": "deepseek-v4-materialize-expert-shards",
            "version": 1,
            "status": "dry_run_ready" if not blockers else "blocked",
            "blockers": blockers,
            "model_dir": str(args.model_dir),
            "core": str(args.core),
            "shards": str(shard_index_path),
            "catalog_lock": str(lock_path),
            "lock_acquired": bool(lock_fd is not None),
            "wait_lock_seconds": args.wait_lock_seconds,
            "layer_id": args.layer_id,
            "requested_expert_ids": requested,
            "existing_expert_ids": sorted(int(item["expert_id"]) for item in existing.values() if int(item["layer_id"]) == args.layer_id),
            "to_write_expert_ids": sorted(grouped),
            "missing_source_expert_ids": missing,
            "required_output_bytes": required_bytes,
            "free_before_bytes": free_bytes,
            "min_free_after_bytes": min_free_after,
            "execute": bool(args.execute),
        }
        if not args.execute or blockers:
            write_json(args.out, payload)
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0 if not blockers else 3

        experts_dir.mkdir(parents=True, exist_ok=True)
        written = []
        with tempfile.TemporaryDirectory(prefix="zc_ds4_experts_") as tmp_name:
            tmp_dir = Path(tmp_name)
            for expert_id, refs in sorted(grouped.items()):
                tmp_path = tmp_dir / f"layer{args.layer_id}_expert{expert_id}.zcblk"
                _, dequant_bytes, checksum = ds.write_raw_block_payload(
                    args.model_dir,
                    refs,
                    tmp_path,
                    max(1, args.chunk_mb) * 1024 * 1024,
                )
                expert_name = f"layer{args.layer_id}_expert{expert_id}.zcblk"
                expert_path = experts_dir / expert_name
                disk_bytes, payload_bytes = zc.write_aligned_file(expert_path, tmp_path)
                record = {
                    "layer_id": args.layer_id,
                    "expert_id": expert_id,
                    "path": f"experts/{expert_name}",
                    "disk_bytes": disk_bytes,
                    "payload_bytes": payload_bytes,
                    "dequant_bytes": dequant_bytes,
                    "quant_format": ds.QUANT_DEEPSEEK_RAW_MIXED,
                    "checksum": checksum,
                }
                existing[(args.layer_id, expert_id)] = record
                written.append(record)

        experts = sorted(existing.values(), key=lambda item: (int(item["layer_id"]), int(item["expert_id"])))
        shard_index["experts"] = experts
        per_layer_counts: dict[int, int] = {}
        for item in experts:
            layer_id = int(item["layer_id"])
            per_layer_counts[layer_id] = per_layer_counts.get(layer_id, 0) + 1
        shard_index["experts_per_layer"] = max(per_layer_counts.values(), default=0)
        metadata = dict(shard_index.get("metadata", {}))
        metadata["expert_catalog_extended_at_unix"] = int(time.time())
        metadata["expert_catalog_extends_core_manifest"] = True
        shard_index["metadata"] = metadata
        write_json(shard_index_path, shard_index)

        payload["status"] = "ready"
        payload["written"] = written
        payload["free_after_bytes"] = shutil.disk_usage(args.core.parent).free
        payload["catalog_expert_count"] = len(experts)
        write_json(args.out, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    finally:
        release_catalog_lock(lock_path, lock_fd)


if __name__ == "__main__":
    raise SystemExit(main())
