#!/usr/bin/env python3
"""DeepSeek-V4-Flash converter dry-run.

This does not read safetensors payloads. It consumes metadata/index reports and
emits the role mapping and disk plan required before conversion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DeepSeek-V4-Flash converter dry-run")
    parser.add_argument("--model-dir", type=Path, default=Path("models/deepseek-ai/DeepSeek-V4-Flash"))
    parser.add_argument(
        "--inventory",
        type=Path,
        default=Path("state/deepseek_v4_flash_inventory_metadata_2026-07-04.json"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("state/deepseek_v4_flash_converter_dry_run_2026-07-04.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    inventory = load_json(args.inventory)
    config = load_json(args.model_dir / "config.json")
    index = load_json(args.model_dir / "model.safetensors.index.json")

    role_counts = dict(inventory.get("role_counts", {}))
    precision_counts = dict(inventory.get("precision_counts", {}))
    missing_shard_count = int(inventory.get("missing_shard_count", 0))
    unknown_count = int(role_counts.get("unknown", 0))
    declared_total = int(inventory.get("declared_total_size_bytes", 0))

    expert_tensor_count = int(role_counts.get("expert", 0) + role_counts.get("shared_expert", 0))
    dense_tensor_count = int(
        role_counts.get("attention_q", 0)
        + role_counts.get("attention_kv", 0)
        + role_counts.get("attention_o", 0)
        + role_counts.get("compressor", 0)
        + role_counts.get("indexer", 0)
        + role_counts.get("router", 0)
        + role_counts.get("embed", 0)
        + role_counts.get("lm_head", 0)
        + role_counts.get("mtp", 0)
    )
    aux_tensor_count = int(
        role_counts.get("norm", 0)
        + role_counts.get("attention_sink", 0)
        + role_counts.get("mhc_attention", 0)
        + role_counts.get("mhc_ffn", 0)
        + role_counts.get("mhc_head", 0)
    )
    mapped = expert_tensor_count + dense_tensor_count + aux_tensor_count
    tensor_count = int(inventory.get("tensor_count", 0))

    status = "ready_for_payload_converter_dry_run" if missing_shard_count == 0 else "ready_for_metadata_converter_dry_run"
    blockers: list[str] = []
    if unknown_count:
        status = "blocked_unknown_tensor_roles"
        blockers.append("all tensor roles must be classified")
    if mapped != tensor_count:
        status = "blocked_mapping_gap"
        blockers.append(f"mapped tensor count {mapped} != tensor_count {tensor_count}")
    if missing_shard_count:
        blockers.append("payload conversion blocked until safetensors shards are present")

    payload = {
        "format": "deepseek-v4-flash-converter-dry-run",
        "version": 1,
        "status": status,
        "model_dir": str(args.model_dir),
        "inventory": str(args.inventory),
        "model_config": {
            "model_type": config.get("model_type"),
            "num_hidden_layers": config.get("num_hidden_layers"),
            "hidden_size": config.get("hidden_size"),
            "vocab_size": config.get("vocab_size"),
            "n_routed_experts": config.get("n_routed_experts"),
            "num_experts_per_tok": config.get("num_experts_per_tok"),
            "expert_dtype": config.get("expert_dtype"),
            "quant_method": (config.get("quantization_config") or {}).get("quant_method"),
            "fp8_fmt": (config.get("quantization_config") or {}).get("fmt"),
            "scale_fmt": (config.get("quantization_config") or {}).get("scale_fmt"),
            "sliding_window": config.get("sliding_window"),
            "index_topk": config.get("index_topk"),
            "num_nextn_predict_layers": config.get("num_nextn_predict_layers"),
        },
        "source_index": {
            "declared_total_size_bytes": declared_total,
            "declared_shards": len(set(index.get("weight_map", {}).values())),
            "missing_shard_count": missing_shard_count,
        },
        "mapping": {
            "tensor_count": tensor_count,
            "mapped_tensor_count": mapped,
            "expert_tensor_count": expert_tensor_count,
            "dense_tensor_count": dense_tensor_count,
            "aux_tensor_count": aux_tensor_count,
            "role_counts": role_counts,
            "precision_counts": precision_counts,
        },
        "disk_plan": {
            "source_checkpoint_bytes": declared_total,
            "metadata_only_bytes_required": 0,
            "payload_conversion_requires_source_shards": missing_shard_count == 0,
            "recommended_min_free_after_conversion_bytes": 250 * 1024**3,
            "estimated_wohper_payload_upper_bound_bytes": declared_total,
            "notes": [
                "use the safetensors header byte estimator to split bytes by tensor role before payload conversion",
                "payload conversion must stream shards; do not duplicate full checkpoint by default",
                "expert shards should be externalized for fetch/cache workflow",
            ],
        },
        "converter_outputs": [
            "dense_core.bin",
            "dense_core.shards.json",
            "experts/layer-*/expert-*.zcblk",
            "tokenizer/encoding metadata copy",
            "conversion_ledger.json",
        ],
        "blockers": blockers,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if status in {"ready_for_metadata_converter_dry_run", "ready_for_payload_converter_dry_run"} else 3


if __name__ == "__main__":
    raise SystemExit(main())
