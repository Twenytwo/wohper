#!/usr/bin/env python3
"""Gate DeepSeek-V4 runtime smokes without pretending weights exist."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DeepSeek-V4 runtime smoke gate")
    parser.add_argument("--readiness", type=Path, default=Path("state/deepseek_v4_flash_readiness_2026-07-04.json"))
    parser.add_argument("--inventory", type=Path, default=Path("state/deepseek_v4_flash_inventory_metadata_2026-07-04.json"))
    parser.add_argument("--quality-prompts", type=Path, default=Path("config/deepseek_v4_quality_prompts.small.json"))
    parser.add_argument("--out", type=Path, default=Path("state/deepseek_v4_flash_runtime_smoke_gate_2026-07-04.json"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    readiness = load_json(args.readiness)
    inventory = load_json(args.inventory)
    quality = load_json(args.quality_prompts)
    missing_shards = int(readiness.get("missing_declared_count", 0) or inventory.get("missing_shard_count", 0))
    metadata_ready = readiness.get("has_index") and readiness.get("config_checks", {}).get("has_config")
    role_map_ready = inventory.get("status") in {
        "ready_for_metadata_converter_dry_run",
        "ready_for_converter_dry_run",
    }
    payload_ready = missing_shards == 0 and readiness.get("present_shards", 0) >= readiness.get("expected_shards", 46)

    status = "ready_for_runtime_smoke" if metadata_ready and role_map_ready and payload_ready else "expected_blocked_missing_weights"
    blockers = []
    if not metadata_ready:
        blockers.append("metadata not ready")
    if not role_map_ready:
        blockers.append("role map not ready")
    if not payload_ready:
        blockers.append(f"missing safetensors shards: {missing_shards}")

    payload = {
        "format": "deepseek-v4-runtime-smoke-gate",
        "version": 1,
        "status": status,
        "metadata_ready": bool(metadata_ready),
        "role_map_ready": bool(role_map_ready),
        "payload_ready": bool(payload_ready),
        "expected_shards": readiness.get("expected_shards"),
        "present_shards": readiness.get("present_shards"),
        "missing_shards": missing_shards,
        "quality_prompt_count": len(quality.get("prompts", [])),
        "smokes": {
            "single_token_logit": "ready" if payload_ready else "expected_blocked",
            "bounded_multitoken_chat": "ready" if payload_ready else "expected_blocked",
            "quality_mini_benchmark": "ready" if payload_ready else "expected_blocked",
        },
        "blockers": blockers,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if status == "ready_for_runtime_smoke" else 3


if __name__ == "__main__":
    raise SystemExit(main())
