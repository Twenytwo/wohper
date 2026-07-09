#!/usr/bin/env python3
"""Plan DeepSeek-V4-Flash safetensors inventory before any conversion."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def classify_tensor(name: str) -> str:
    lowered = name.lower()
    if lowered == "embed.weight" or "embed_tokens" in lowered or "word_embeddings" in lowered:
        return "embed"
    if lowered == "head.weight" or "lm_head" in lowered:
        return "lm_head"
    if lowered == "norm.weight":
        return "norm"
    if lowered.startswith("mtp."):
        return "mtp"
    if lowered.startswith("hc_head_"):
        return "mhc_head"
    if ".hc_attn_" in lowered:
        return "mhc_attention"
    if ".hc_ffn_" in lowered:
        return "mhc_ffn"
    if lowered.endswith(".norm.weight") or "_norm.weight" in lowered or ".input_layernorm." in lowered or ".post_attention_layernorm." in lowered:
        return "norm"
    if ".compressor." in lowered:
        return "compressor"
    if ".indexer." in lowered:
        return "indexer"
    if ".attn." in lowered or ".self_attn." in lowered or ".attention." in lowered:
        if "attn_sink" in lowered:
            return "attention_sink"
        if ".q_norm." in lowered:
            return "norm"
        if ".kv_norm." in lowered:
            return "norm"
        if ".wq_a." in lowered or ".wq_b." in lowered or ".q_proj" in lowered:
            return "attention_q"
        if ".wkv." in lowered or ".k_proj" in lowered or ".v_proj" in lowered:
            return "attention_kv"
        if ".wo_a." in lowered or ".wo_b." in lowered or ".o_proj" in lowered or ".out_proj" in lowered:
            return "attention_o"
        return "attention_other"
    if ".experts." in lowered:
        return "expert"
    if "shared_expert" in lowered or "shared_experts" in lowered:
        return "shared_expert"
    if ".gate." in lowered or "router" in lowered or "gate.weight" in lowered or ".tid2eid" in lowered:
        return "router"
    if ".mlp." in lowered:
        return "dense_mlp"
    return "unknown"


def infer_storage_precision(role: str, name: str) -> str:
    lowered = name.lower()
    if lowered.endswith(".scale"):
        return "scale"
    if role in {"expert", "shared_expert"}:
        return "fp4_expert"
    if role in {
        "attention_q",
        "attention_kv",
        "attention_o",
        "router",
        "compressor",
        "indexer",
        "dense_mlp",
        "mtp",
        "embed",
        "lm_head",
    }:
        return "fp8_or_bf16_dense"
    if role.startswith("mhc") or role in {"norm", "attention_sink", "mtp"}:
        return "bf16_or_fp32_aux"
    return "unknown"


def layer_id_from_name(name: str) -> int | None:
    parts = name.split(".")
    for idx, part in enumerate(parts):
        if part == "layers" and idx + 1 < len(parts):
            try:
                return int(parts[idx + 1])
            except ValueError:
                return None
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DeepSeek-V4-Flash inventory dry-run")
    parser.add_argument("--model-dir", type=Path, default=Path("models/deepseek-ai/DeepSeek-V4-Flash"))
    parser.add_argument("--contract", type=Path, default=Path("config/deepseek_v4_flash.contract.json"))
    parser.add_argument("--out", type=Path, default=Path("state/deepseek_v4_flash_inventory_2026-07-04.json"))
    parser.add_argument(
        "--max-unknown-sample",
        type=int,
        default=32,
        help="number of unknown tensor names to keep in the report",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="allow missing safetensors shards and produce a converter plan from index metadata",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    contract = load_json(args.contract)
    index = load_json(args.model_dir / "model.safetensors.index.json")
    config = load_json(args.model_dir / "config.json")

    if not contract:
        payload = {
            "format": "deepseek-v4-flash-inventory-plan",
            "version": 1,
            "status": "blocked_missing_contract",
            "contract": str(args.contract),
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 3

    if not index or not isinstance(index.get("weight_map"), dict):
        payload = {
            "format": "deepseek-v4-flash-inventory-plan",
            "version": 1,
            "status": "blocked_missing_index",
            "model_dir": str(args.model_dir),
            "required_file": "model.safetensors.index.json",
            "next_gate": "download_metadata_without_weights",
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 3

    weight_map = index["weight_map"]
    declared_shards = sorted({str(shard) for shard in weight_map.values()})
    present_shards = {path.name for path in args.model_dir.glob("*.safetensors")}
    missing_shards = sorted(set(declared_shards) - present_shards)
    roles = Counter()
    precision_counts = Counter()
    role_shards: dict[str, set[str]] = defaultdict(set)
    layers: dict[int, Counter[str]] = defaultdict(Counter)
    unknown_sample: list[str] = []

    for tensor_name in sorted(weight_map):
        role = classify_tensor(str(tensor_name))
        precision = infer_storage_precision(role, str(tensor_name))
        roles[role] += 1
        precision_counts[precision] += 1
        role_shards[role].add(str(weight_map[tensor_name]))
        layer_id = layer_id_from_name(str(tensor_name))
        if layer_id is not None:
            layers[layer_id][role] += 1
        if role == "unknown" and len(unknown_sample) < args.max_unknown_sample:
            unknown_sample.append(str(tensor_name))

    metadata = index.get("metadata", {})
    declared_total_size = int(metadata.get("total_size", 0) or 0)
    status = "ready_for_converter_dry_run"
    if missing_shards and not args.metadata_only:
        status = "blocked_missing_safetensors"
    elif roles["unknown"]:
        status = "blocked_unknown_tensor_roles"
    elif args.metadata_only:
        status = "ready_for_metadata_converter_dry_run"

    payload = {
        "format": "deepseek-v4-flash-inventory-plan",
        "version": 1,
        "status": status,
        "model_dir": str(args.model_dir),
        "model_family": contract.get("model_family"),
        "hf_repo": contract.get("hf_repo"),
        "has_config": config is not None,
        "config_model_type": (config or {}).get("model_type"),
        "tensor_count": len(weight_map),
        "declared_shard_count": len(declared_shards),
        "present_shard_count": len(present_shards),
        "missing_shard_count": len(missing_shards),
        "missing_shard_sample": missing_shards[:16],
        "declared_total_size_bytes": declared_total_size,
        "role_counts": dict(sorted(roles.items())),
        "precision_counts": dict(sorted(precision_counts.items())),
        "role_shard_counts": {
            role: len(shards)
            for role, shards in sorted(role_shards.items())
        },
        "layer_count_with_tensors": len(layers),
        "layer_role_sample": {
            str(layer): dict(sorted(counter.items()))
            for layer, counter in sorted(layers.items())[:8]
        },
        "unknown_tensor_sample": unknown_sample,
        "converter_dry_run_requirements": [
            "all declared safetensors shards present",
            "all tensor names classified or explicitly ignored",
            "FP4 expert tensors separated from FP8 dense tensors",
            "router/expert/shared-expert tensors mapped by layer",
            "embed and LM-head tensors mapped exactly",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if status in {"ready_for_converter_dry_run", "ready_for_metadata_converter_dry_run"} else 3


if __name__ == "__main__":
    raise SystemExit(main())
