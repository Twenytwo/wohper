#!/usr/bin/env python3
"""Reuse materialized DeepSeek expert shards from one RAW slice in another."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_layers(raw: str) -> set[int] | None:
    if raw == "all":
        return None
    selected: set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = [int(v) for v in part.split("-", 1)]
            selected.update(range(start, end + 1))
        else:
            selected.add(int(part))
    return selected


def acquire_lock(lock_path: Path, wait_seconds: float) -> int:
    deadline = time.monotonic() + max(0.0, wait_seconds)
    payload = {
        "created_at_unix": int(time.time()),
        "pid": os.getpid(),
        "purpose": "deepseek_v4_reuse_expert_catalog",
    }
    encoded = (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8")
    while True:
        try:
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, encoded)
            return fd
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise RuntimeError(f"catalog lock exists: {lock_path}")
            time.sleep(1.0)


def release_lock(lock_path: Path, fd: int | None) -> None:
    if fd is not None:
        os.close(fd)
    try:
        lock_path.unlink()
    except FileNotFoundError:
        pass


def expert_key(item: dict[str, Any]) -> tuple[int, int]:
    return int(item["layer_id"]), int(item["expert_id"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reuse DeepSeek expert shards between RAW slices")
    parser.add_argument("--source-shards", type=Path, required=True)
    parser.add_argument("--target-shards", type=Path, required=True)
    parser.add_argument("--layers", default="all", help="all, comma list, or inclusive ranges like 0-15")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--copy-fallback", action="store_true", help="copy if hardlink fails; disabled by default to protect disk")
    parser.add_argument("--wait-lock-seconds", type=float, default=0.0)
    parser.add_argument("--out", type=Path, default=Path("state/deepseek_v4_reuse_expert_catalog_2026-07-05.json"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected_layers = parse_layers(args.layers)
    source_index = load_json(args.source_shards)
    target_lock = Path(str(args.target_shards) + ".lock")
    lock_fd: int | None = None
    if args.execute:
        try:
            lock_fd = acquire_lock(target_lock, args.wait_lock_seconds)
        except RuntimeError as exc:
            payload = {
                "format": "deepseek-v4-reuse-expert-catalog",
                "version": 1,
                "status": "blocked",
                "blockers": ["catalog_lock_exists"],
                "error": str(exc),
                "target_lock": str(target_lock),
            }
            write_json(args.out, payload)
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 4

    try:
        target_index = load_json(args.target_shards)
        source_root = args.source_shards.parent
        target_root = args.target_shards.parent
        target_experts_dir = target_root / "experts"
        existing = {expert_key(item): item for item in target_index.get("experts", [])}
        reusable = []
        missing_files = []
        for item in source_index.get("experts", []):
            layer_id, expert_id = expert_key(item)
            if selected_layers is not None and layer_id not in selected_layers:
                continue
            if (layer_id, expert_id) in existing:
                continue
            rel_path = Path(str(item["path"]))
            source_path = source_root / rel_path
            target_path = target_root / rel_path
            if not source_path.exists():
                missing_files.append(str(source_path))
                continue
            reusable.append((item, source_path, target_path))

        blockers = []
        if missing_files:
            blockers.append("missing_source_expert_file")

        payload: dict[str, Any] = {
            "format": "deepseek-v4-reuse-expert-catalog",
            "version": 1,
            "status": "dry_run_ready" if not blockers else "blocked",
            "blockers": blockers,
            "source_shards": str(args.source_shards),
            "target_shards": str(args.target_shards),
            "target_lock": str(target_lock),
            "execute": bool(args.execute),
            "copy_fallback": bool(args.copy_fallback),
            "selected_layers": "all" if selected_layers is None else sorted(selected_layers),
            "existing_target_experts": len(existing),
            "reusable_experts": len(reusable),
            "missing_source_files": missing_files[:16],
        }
        if not args.execute or blockers:
            write_json(args.out, payload)
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0 if not blockers else 3

        target_experts_dir.mkdir(parents=True, exist_ok=True)
        linked = 0
        copied = 0
        for item, source_path, target_path in reusable:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            if not target_path.exists():
                try:
                    os.link(source_path, target_path)
                    linked += 1
                except OSError:
                    if not args.copy_fallback:
                        raise
                    shutil.copy2(source_path, target_path)
                    copied += 1
            record = dict(item)
            record["path"] = f"experts/{target_path.name}"
            existing[expert_key(record)] = record

        experts = sorted(existing.values(), key=lambda item: (int(item["layer_id"]), int(item["expert_id"])))
        target_index["experts"] = experts
        per_layer_counts: dict[int, int] = {}
        for item in experts:
            layer_id = int(item["layer_id"])
            per_layer_counts[layer_id] = per_layer_counts.get(layer_id, 0) + 1
        target_index["experts_per_layer"] = max(per_layer_counts.values(), default=0)
        metadata = dict(target_index.get("metadata", {}))
        metadata["expert_catalog_reused_at_unix"] = int(time.time())
        metadata["expert_catalog_reuse_source"] = str(args.source_shards)
        target_index["metadata"] = metadata
        write_json(args.target_shards, target_index)

        payload["status"] = "ready"
        payload["linked"] = linked
        payload["copied"] = copied
        payload["catalog_expert_count"] = len(experts)
        write_json(args.out, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    finally:
        release_lock(target_lock, lock_fd)


if __name__ == "__main__":
    raise SystemExit(main())
