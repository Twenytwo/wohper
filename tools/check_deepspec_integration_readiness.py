#!/usr/bin/env python3
"""Check how DeepSpec is integrated without importing heavy ML dependencies."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


REQUIRED_FILES = [
    "README.md",
    "LICENSE",
    "NOTICE",
    "train.py",
    "eval.py",
    "requirements.txt",
    "scripts/data/README.md",
    "deepspec/eval/base_evaluator.py",
    "deepspec/utils/sampling.py",
    "deepspec/eval/dspark/draft_ops.py",
]


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def find_target_models(repo: Path) -> list[str]:
    models: set[str] = set()
    for config_path in (repo / "config").glob("*/*.py"):
        text = read_text(config_path)
        for match in re.finditer(r'target_model_name_or_path\s*=\s*"([^"]+)"', text):
            models.add(match.group(1))
    return sorted(models)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="vendor/deepspec")
    parser.add_argument(
        "--out",
        default="state/deepspec_integration_readiness_2026-07-04.json",
    )
    args = parser.parse_args()

    repo = Path(args.repo)
    out = Path(args.out)

    missing = [rel for rel in REQUIRED_FILES if not (repo / rel).exists()]
    readme = read_text(repo / "README.md")
    data_readme = read_text(repo / "scripts/data/README.md")
    train_py = read_text(repo / "train.py")
    eval_py = read_text(repo / "eval.py")
    evaluator = read_text(repo / "deepspec/eval/base_evaluator.py")

    algorithms = sorted(
        path.name for path in (repo / "config").iterdir() if path.is_dir()
    ) if (repo / "config").exists() else []

    danger_markers = {
        "target_cache_38tb_warning": "38 TB" in data_readme,
        "multi_gpu_default": "eight visible GPUs" in data_readme or "8 GPUs" in readme,
        "cuda_spawn_training": "torch.cuda.device_count()" in train_py,
        "cuda_spawn_eval": "torch.cuda.device_count()" in eval_py,
        "requires_external_openai_endpoint_for_regen": "/v1" in data_readme
        and "OpenAI-compatible" in data_readme,
    }

    reusable_contracts = {
        "draft_proposal": "class DraftProposal" in evaluator,
        "verification_result": "class VerificationResult" in evaluator,
        "verify_draft_tokens": "def verify_draft_tokens" in evaluator,
        "generate_decoding_sample": "def generate_decoding_sample" in evaluator,
        "residual_sampling": "sample_residual" in evaluator,
    }

    deepseek_v4_mentions = [
        str(path.relative_to(repo))
        for path in repo.rglob("*")
        if path.is_file()
        and path.suffix in {".py", ".md", ".txt"}
        and "DeepSeek-V4" in read_text(path)
    ]

    status = (
        "ready_as_reference_not_runtime"
        if repo.exists() and not missing and all(reusable_contracts.values())
        else "blocked"
    )

    report = {
        "repo": str(repo),
        "status": status,
        "missing_required_files": missing,
        "algorithms": algorithms,
        "target_models_in_configs": find_target_models(repo),
        "deepseek_v4_flash_native_support": bool(deepseek_v4_mentions),
        "deepseek_v4_mentions": deepseek_v4_mentions,
        "safe_for_16gb_quickstart": False,
        "danger_markers": danger_markers,
        "reusable_contracts": reusable_contracts,
        "integration_decision": (
            "Use DeepSpec as an advanced speculative-decoding reference and "
            "evaluation contract. Do not run its default data/cache/training "
            "pipeline in the Wohper 16GB/1TB quick start."
        ),
        "required_next_adapters": [
            "deepseek_v4_flash_metadata",
            "deepseek_v4_flash_tokenizer_template",
            "deepseek_v4_flash_safetensors_converter",
            "deepseek_v4_flash_fp4_fp8_unpack",
            "deepseek_v4_flash_moe_router",
            "deepseek_v4_flash_attention_csa_hca",
            "wohper_speculative_decoding_contract",
        ],
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if status != "blocked" else 1


if __name__ == "__main__":
    raise SystemExit(main())
