#!/usr/bin/env python3
"""DeepSeek-V4-Flash contract and local checkpoint readiness checker."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any


SHARD_RE = re.compile(r"model-\d{5}-of-(\d{5})\.safetensors$")


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def path_exists(root: Path, rel: str) -> bool:
    return (root / rel).exists()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DeepSeek-V4-Flash readiness check")
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("config/deepseek_v4_flash.contract.json"),
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models/deepseek-ai/DeepSeek-V4-Flash"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("state/deepseek_v4_flash_readiness_2026-07-04.json"),
    )
    parser.add_argument(
        "--contract-only",
        action="store_true",
        help="validate the Wohper contract without requiring local model weights",
    )
    return parser.parse_args()


def collect_shards(model_dir: Path) -> tuple[list[Path], int | None]:
    shards = sorted(model_dir.glob("*.safetensors")) if model_dir.exists() else []
    expected_from_name = None
    for path in shards:
        match = SHARD_RE.match(path.name)
        if match:
            expected_from_name = int(match.group(1))
            break
    return shards, expected_from_name


def declared_shards_from_index(index: dict[str, Any] | None) -> set[str]:
    if not index or not isinstance(index.get("weight_map"), dict):
        return set()
    return {str(name) for name in index["weight_map"].values()}


def compute_status(
    *,
    contract: dict[str, Any] | None,
    contract_only: bool,
    model_dir: Path,
    missing_contract_keys: list[str],
    missing_metadata: list[str],
    missing_declared: list[str],
    expected_shards: int,
    present_shards: int,
) -> str:
    if contract is None or missing_contract_keys:
        return "blocked_bad_contract"
    if contract_only:
        return "contract_ready"
    if not model_dir.exists():
        return "blocked_missing_model_dir"
    if missing_metadata:
        return "blocked_missing_metadata"
    if present_shards < expected_shards:
        return "blocked_missing_safetensors"
    if missing_declared:
        return "blocked_missing_declared_shards"
    return "ready_for_converter_dry_run"


def main() -> int:
    args = parse_args()
    contract = load_json(args.contract)
    required_contract_keys = [
        "model_family",
        "hf_repo",
        "official_public_facts",
        "quickstart_policy",
        "required_metadata_paths",
        "required_runtime_adapters",
        "deepspec_policy",
    ]
    missing_contract_keys = [
        key for key in required_contract_keys if not contract or key not in contract
    ]
    facts = contract.get("official_public_facts", {}) if contract else {}
    policy = contract.get("quickstart_policy", {}) if contract else {}

    expected_shards = int(facts.get("expected_safetensors_shards", 46) or 46)
    expected_model_bytes = int(float(facts.get("hf_repo_size_gb", 160)) * 1024**3)
    min_free_after_bytes = int(float(policy.get("min_free_after_download_gb", 250)) * 1024**3)
    staging_bytes = int(expected_model_bytes * float(policy.get("temp_staging_multiplier", 0.1)))
    required_free_for_download = expected_model_bytes + min_free_after_bytes + staging_bytes

    model_dir = args.model_dir
    index = load_json(model_dir / "model.safetensors.index.json")
    shards, expected_from_name = collect_shards(model_dir)
    if expected_from_name:
        expected_shards = expected_from_name

    declared_shards = declared_shards_from_index(index)
    shard_names = {path.name for path in shards}
    missing_declared = sorted(declared_shards - shard_names)
    extra_shards = sorted(shard_names - declared_shards) if declared_shards else []
    present_bytes = sum(path.stat().st_size for path in shards)
    declared_total_size = int((index or {}).get("metadata", {}).get("total_size", 0) or 0)

    required_metadata = list(contract.get("required_metadata_paths", [])) if contract else []
    missing_metadata = [
        rel for rel in required_metadata if model_dir.exists() and not path_exists(model_dir, rel)
    ]

    disk_root = model_dir if model_dir.exists() else Path(".")
    disk_usage = shutil.disk_usage(disk_root)
    total_bytes = disk_usage.total
    free_bytes = disk_usage.free
    target_ssd_class_bytes = int(float(policy.get("min_target_ssd_gb", 1000)) * 1_000_000_000)
    target_ssd_class_ok = total_bytes >= target_ssd_class_bytes
    download_storage_feasible = free_bytes >= required_free_for_download and target_ssd_class_ok
    config = load_json(model_dir / "config.json")
    config_checks = {
        "has_config": config is not None,
        "model_type": (config or {}).get("model_type"),
        "model_type_matches_tag": (config or {}).get("model_type") in {None, facts.get("model_type_tag")},
    }

    status = compute_status(
        contract=contract,
        contract_only=args.contract_only,
        model_dir=model_dir,
        missing_contract_keys=missing_contract_keys,
        missing_metadata=missing_metadata,
        missing_declared=missing_declared,
        expected_shards=expected_shards,
        present_shards=len(shards),
    )
    payload = {
        "format": "deepseek-v4-flash-readiness",
        "version": 1,
        "status": status,
        "contract": str(args.contract),
        "model_dir": str(model_dir),
        "contract_only": bool(args.contract_only),
        "missing_contract_keys": missing_contract_keys,
        "model_family": contract.get("model_family") if contract else None,
        "hf_repo": contract.get("hf_repo") if contract else None,
        "official_public_facts": facts,
        "quickstart_policy": policy,
        "safe_for_16gb_quickstart": False,
        "expected_shards": expected_shards,
        "present_shards": len(shards),
        "has_index": index is not None,
        "declared_shards": len(declared_shards),
        "declared_total_size_bytes": declared_total_size,
        "expected_model_bytes_from_contract": expected_model_bytes,
        "present_safetensors_bytes": present_bytes,
        "missing_metadata": missing_metadata,
        "missing_declared_count": len(missing_declared),
        "missing_declared_sample": missing_declared[:16],
        "extra_shard_count": len(extra_shards),
        "extra_shard_sample": extra_shards[:16],
        "filesystem_free_bytes": free_bytes,
        "filesystem_total_bytes": total_bytes,
        "target_ssd_class_bytes": target_ssd_class_bytes,
        "target_ssd_class_ok": target_ssd_class_ok,
        "required_free_for_safe_download_bytes": required_free_for_download,
        "download_storage_feasible_here": download_storage_feasible,
        "config_checks": config_checks,
        "required_runtime_adapters": contract.get("required_runtime_adapters", []) if contract else [],
        "deepspec_policy": contract.get("deepspec_policy", {}) if contract else {},
        "next_gate": (
            "safetensors_inventory_and_converter_dry_run"
            if status == "ready_for_converter_dry_run"
            else "metadata_download_or_contract_review"
        ),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if status in {"contract_ready", "ready_for_converter_dry_run"} else 3


if __name__ == "__main__":
    raise SystemExit(main())
