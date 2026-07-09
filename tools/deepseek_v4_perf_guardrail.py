#!/usr/bin/env python3
"""DeepSeek-V4 local 16GB/1TB performance and safety guardrail report."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def dir_size(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for item in path.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
    return total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DeepSeek-V4 16GB/1TB guardrail")
    parser.add_argument("--core", type=Path, default=Path("models/wohper/DeepSeek-V4-Flash.RAW.L0-SPLIT-GLOBAL-TOP6/dense_core.bin"))
    parser.add_argument("--single-token", type=Path, default=Path("state/deepseek_v4_single_token_l0_math_smoke_scan256_2026-07-05.json"))
    parser.add_argument("--chat", type=Path, default=Path("state/deepseek_v4_bounded_chat_smoke_2026-07-05.json"))
    parser.add_argument("--target-ram-gb", type=float, default=16.0)
    parser.add_argument("--target-ssd-gb", type=float, default=1000.0)
    parser.add_argument("--min-free-after-gb", type=float, default=250.0)
    parser.add_argument("--max-single-token-read-mb", type=float, default=512.0)
    parser.add_argument("--max-chat-token-seconds", type=float, default=30.0)
    parser.add_argument("--profile-summary", type=Path)
    parser.add_argument("--out", type=Path, default=Path("state/deepseek_v4_perf_guardrail_16gb_2026-07-05.json"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    shard_index_path = args.core.with_suffix(".shards.json")
    shard_index = load_json(shard_index_path)
    single = load_json(args.single_token)
    chat = load_json(args.chat)
    profile_summary = load_json(args.profile_summary) if args.profile_summary else {}
    disk = shutil.disk_usage(args.core.parent)
    experts_dir = args.core.parent / "experts"
    expert_disk_bytes = dir_size(experts_dir)
    dense_bytes = args.core.stat().st_size if args.core.exists() else 0
    row_bytes = 8192
    vocab_size = 129280
    full_lmhead_scan_bytes = row_bytes * vocab_size

    blockers = []
    warnings = []
    if single.get("status") != "ready":
        blockers.append("single_token_smoke_not_ready")
    if chat.get("status") != "ready":
        blockers.append("bounded_chat_smoke_not_ready")
    if disk.free < int(args.min_free_after_gb * 1024**3):
        blockers.append("disk_below_min_free_after")
    if int(single.get("bytes_read_upper_bound", 0)) > int(args.max_single_token_read_mb * 1024**2):
        blockers.append("single_token_read_budget_exceeded")
    chat_steps = chat.get("steps", [])
    chat_elapsed = None
    if chat_steps:
        chat_elapsed = chat_steps[-1].get("elapsed_seconds")
    if chat_elapsed is not None and float(chat_elapsed) > args.max_chat_token_seconds:
        blockers.append("chat_token_latency_budget_exceeded")
    if not shard_index.get("metadata", {}).get("global_aux_separate"):
        blockers.append("global_aux_not_separate")
    if full_lmhead_scan_bytes > 512 * 1024**2:
        warnings.append("full_vocab_lmhead_scan_is_about_1gb_per_token_without_topk_index_or_sharding")
    if shard_index.get("metadata", {}).get("expert_catalog_extends_core_manifest"):
        warnings.append("expert_catalog_extends_core_manifest_for_incremental_smoke")
    if chat.get("generated_token_ids") == [0, 0]:
        warnings.append("bounded_chat_quality_repeats_bos")

    payload = {
        "format": "deepseek-v4-perf-guardrail",
        "version": 1,
        "status": "ready" if not blockers else "blocked",
        "blockers": blockers,
        "warnings": warnings,
        "target": {
            "ram_gb": args.target_ram_gb,
            "ssd_gb": args.target_ssd_gb,
            "min_free_after_gb": args.min_free_after_gb,
        },
        "artifacts": {
            "core": str(args.core),
            "dense_core_bytes": dense_bytes,
            "experts_dir": str(experts_dir),
            "expert_disk_bytes": expert_disk_bytes,
            "catalog_expert_count": len(shard_index.get("experts", [])),
            "slice_total_bytes": dense_bytes + expert_disk_bytes,
            "global_aux_separate": bool(shard_index.get("metadata", {}).get("global_aux_separate")),
        },
        "disk": {
            "total_bytes": disk.total,
            "used_bytes": disk.used,
            "free_bytes": disk.free,
            "free_gib": disk.free / 1024**3,
        },
        "single_token": {
            "status": single.get("status"),
            "elapsed_seconds": single.get("elapsed_seconds"),
            "bytes_read_upper_bound": single.get("bytes_read_upper_bound"),
            "scan_vocab": single.get("scan_vocab"),
        },
        "chat": {
            "status": chat.get("status"),
            "generated_token_ids": chat.get("generated_token_ids"),
            "stop_reason": chat.get("stop_reason"),
            "materialization_count": len(chat.get("materializations", [])),
            "last_step_elapsed_seconds": chat_elapsed,
            "max_chat_token_seconds": args.max_chat_token_seconds,
        },
        "profile_summary": profile_summary,
        "lm_head": {
            "row_bytes": row_bytes,
            "vocab_size": vocab_size,
            "full_scan_bytes_per_token": full_lmhead_scan_bytes,
            "full_scan_gib_per_token": full_lmhead_scan_bytes / 1024**3,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not blockers else 3


if __name__ == "__main__":
    raise SystemExit(main())
