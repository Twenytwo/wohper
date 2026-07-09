#!/usr/bin/env python3
"""Data-driven TOP8 expansion gate."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate whether TOP8 expansion is justified")
    parser.add_argument("--quality-state", type=Path, required=True)
    parser.add_argument("--top4-catalog", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-free-gb-after-estimate", type=int, default=20)
    return parser.parse_args()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    args = parse_args()
    quality_exists = args.quality_state.exists()
    quality = load(args.quality_state) if quality_exists else {}
    catalog = load(args.top4_catalog)
    free_bytes = shutil.disk_usage(Path(".")).free
    top4_bytes = int(catalog.get("total_bytes", 0))
    top8_increment_estimate = top4_bytes
    free_after = free_bytes - top8_increment_estimate
    quality_passed = quality_exists and quality.get("current_quality_status") == "passed"
    disk_passed = free_after >= args.min_free_gb_after_estimate * 1024**3
    decision = "allow_top8" if quality_passed and disk_passed else "defer_top8"
    reasons = []
    if not quality_passed:
        reasons.append("quality_state_missing" if not quality_exists else "quality_gate_not_passed")
    if not disk_passed:
        reasons.append("disk_after_top8_estimate_below_guardrail")
    payload = {
        "format": "wohper-top8-data-driven-gate",
        "version": 1,
        "decision": decision,
        "reasons": reasons,
        "quality_state": str(args.quality_state),
        "quality_state_exists": quality_exists,
        "quality_passed": quality_passed,
        "top4_catalog": str(args.top4_catalog),
        "top4_bytes": top4_bytes,
        "top8_increment_estimate_bytes": top8_increment_estimate,
        "filesystem_free_bytes": free_bytes,
        "free_after_estimate_bytes": free_after,
        "min_free_after_estimate_gb": args.min_free_gb_after_estimate,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"TOP8_DECISION={decision}")
    print(f"REASONS={','.join(reasons)}")
    print(f"OUT={args.out}")
    return 0 if decision == "allow_top8" else 7


if __name__ == "__main__":
    raise SystemExit(main())
