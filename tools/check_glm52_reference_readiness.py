#!/usr/bin/env python3
"""Check whether local GLM-5.2 files are sufficient for numeric parity traces."""

from __future__ import annotations

import argparse
import json
import shutil
import re
from pathlib import Path
from typing import Any


SHARD_RE = re.compile(r"model-\d{5}-of-(\d{5})\.safetensors$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GLM-5.2 reference parity readiness check")
    parser.add_argument("--model-dir", type=Path, default=Path("models/zai-org/GLM-5.2"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-shards", type=int, default=282)
    return parser.parse_args()


def load_index(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    args = parse_args()
    model_dir = args.model_dir
    index_path = model_dir / "model.safetensors.index.json"
    index = load_index(index_path)
    shards = sorted(model_dir.glob("*.safetensors"))
    shard_names = {path.name for path in shards}
    declared_shards = set()
    if index and isinstance(index.get("weight_map"), dict):
        declared_shards = {str(name) for name in index["weight_map"].values()}
    metadata = index.get("metadata", {}) if index else {}
    total_size = int(metadata.get("total_size", 0) or 0)
    present_size = sum(path.stat().st_size for path in shards)
    free_bytes = shutil.disk_usage(model_dir if model_dir.exists() else Path(".")).free
    remaining_bytes = max(0, total_size - present_size)
    expected_from_name = None
    for path in shards:
        match = SHARD_RE.match(path.name)
        if match:
            expected_from_name = int(match.group(1))
            break
    expected_shards = expected_from_name or args.expected_shards
    missing_declared = sorted(declared_shards - shard_names)
    extra_shards = sorted(shard_names - declared_shards) if declared_shards else []
    ready = (
        model_dir.exists()
        and index is not None
        and len(shards) >= expected_shards
        and not missing_declared
    )
    payload = {
        "format": "glm52-reference-readiness",
        "version": 1,
        "model_dir": str(model_dir),
        "status": "ready" if ready else "blocked",
        "expected_shards": expected_shards,
        "present_shards": len(shards),
        "has_index": index is not None,
        "declared_shards": len(declared_shards),
        "declared_total_size_bytes": total_size,
        "present_safetensors_bytes": present_size,
        "remaining_safetensors_bytes": remaining_bytes,
        "filesystem_free_bytes": free_bytes,
        "storage_feasible": total_size == 0 or remaining_bytes < free_bytes,
        "missing_declared_count": len(missing_declared),
        "missing_declared_sample": missing_declared[:16],
        "extra_shard_count": len(extra_shards),
        "extra_shard_sample": extra_shards[:16],
        "requirements": [
            "model.safetensors.index.json must exist",
            "all declared safetensors shards must exist locally",
            "numeric trace must run local-files-only against the same prompt tokens",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"READINESS_STATUS={payload['status']}")
    print(f"PRESENT_SHARDS={len(shards)}")
    print(f"EXPECTED_SHARDS={expected_shards}")
    print(f"HAS_INDEX={index is not None}")
    print(f"MISSING_DECLARED={len(missing_declared)}")
    print(f"DECLARED_TOTAL_SIZE_BYTES={total_size}")
    print(f"REMAINING_SAFETENSORS_BYTES={remaining_bytes}")
    print(f"FILESYSTEM_FREE_BYTES={free_bytes}")
    print(f"STORAGE_FEASIBLE={payload['storage_feasible']}")
    print(f"OUT={args.out}")
    return 0 if ready else 3


if __name__ == "__main__":
    raise SystemExit(main())
