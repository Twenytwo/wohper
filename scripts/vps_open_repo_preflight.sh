#!/usr/bin/env bash
set -euo pipefail

cd /home/deploy/hermes-agent/loop_workspace

min_free_gb="${1:-20}"
out="${2:-state/zc_open_repo_preflight_2026-07-04.json}"

if ! [[ "$min_free_gb" =~ ^[0-9]+$ ]] || [ "$min_free_gb" -lt 1 ] || [ "$min_free_gb" -gt 200 ]; then
  echo "refusing unsafe min_free_gb=$min_free_gb; expected 1..200" >&2
  exit 2
fi

required_files=(
  "PROJECT_NOTEBOOK.md"
  "README.md"
  "docs/release-readiness.md"
  "docs/expert-storage-workflow.md"
  "docs/test-matrix.md"
  "docs/model-quality-prompt-set.md"
  "docs/deepspec-integration-plan.md"
  "docs/deepseek-v4-flash-runtime-plan.md"
  "config/deepseek_v4_flash.contract.json"
  "config/deepseek_v4_flash.tokenizer_contract.json"
  "scripts/vps_quality_bench.sh"
  "scripts/vps_sampling_top4_smoke.sh"
  "scripts/vps_reference_readiness.sh"
  "scripts/vps_indexer_long_context_guard.sh"
  "scripts/vps_deepseek_v4_flash_contract.sh"
  "scripts/vps_deepseek_v4_flash_inventory_gate.sh"
  "scripts/vps_deepseek_v4_flash_inventory_metadata.sh"
  "scripts/vps_deepseek_v4_flash_metadata_sync.sh"
  "scripts/vps_deepseek_v4_flash_tokenizer_contract.sh"
  "scripts/vps_deepseek_v4_flash_converter_dry_run.sh"
  "scripts/vps_deepseek_v4_flash_runtime_smoke_gate.sh"
  "scripts/vps_chat_stability_smoke.sh"
  "scripts/vps_expert_storage_preflight.sh"
  "tools/check_glm52_reference_readiness.py"
  "tools/check_glm52_indexer_long_context.py"
  "tools/check_deepseek_v4_flash_readiness.py"
  "tools/plan_deepseek_v4_flash_inventory.py"
  "tools/download_hf_metadata_only.py"
  "tools/check_deepseek_v4_flash_tokenizer_contract.py"
  "tools/deepseek_v4_converter_dry_run.py"
  "tools/deepseek_v4_runtime_smoke_gate.py"
  "tools/summarize_zc_perf_log.py"
  "config/quality_prompts.small.json"
  "config/deepseek_v4_quality_prompts.small.json"
)

missing=()
for file in "${required_files[@]}"; do
  [ -e "$file" ] || missing+=("$file")
done

free_kb="$(df -Pk . | awk 'NR==2 {print $4}')"
free_gb="$((free_kb / 1024 / 1024))"
status="ready"
if [ "${#missing[@]}" -gt 0 ]; then
  status="blocked_missing_files"
elif [ "$free_gb" -lt "$min_free_gb" ]; then
  status="blocked_low_disk"
fi

mkdir -p state
python3 - "$out" "$status" "$free_gb" "$min_free_gb" "${missing[@]}" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
status = sys.argv[2]
free_gb = int(sys.argv[3])
min_free_gb = int(sys.argv[4])
missing = sys.argv[5:]
payload = {
    "format": "wohper-open-repo-preflight",
    "version": 1,
    "status": status,
    "free_gb": free_gb,
    "min_free_gb": min_free_gb,
    "missing_files": missing,
    "guardrails": [
        "hash deploy manifest required for VPS sync",
        "bounded benchmark prompt_limit",
        "bounded IO buffer budgets",
        "reference readiness fails without checkpoint shards",
        "long-context indexer fails above 2048 until implemented",
        "run-specific cache directories",
    ],
}
out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(json.dumps(payload, indent=2))
PY

[ "$status" = "ready" ]
