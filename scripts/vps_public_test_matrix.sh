#!/usr/bin/env bash
set -euo pipefail

cd /home/deploy/hermes-agent/loop_workspace

out="${1:-state/zc_public_test_matrix_2026-07-04.json}"
mkdir -p state logs

run_pass() {
  local name="$1"
  shift
  echo "RUN $name"
  "$@" >/tmp/zc_matrix_${name}.log 2>&1
  echo "PASS $name"
}

run_expected_blocked() {
  local name="$1"
  shift
  echo "RUN_EXPECTED_BLOCKED $name"
  if "$@" >/tmp/zc_matrix_${name}.log 2>&1; then
    echo "expected blocked but passed: $name" >&2
    cat /tmp/zc_matrix_${name}.log >&2
    return 1
  fi
  echo "EXPECTED_BLOCKED $name"
}

run_pass open_repo_preflight bash scripts/vps_open_repo_preflight.sh 20 state/zc_open_repo_preflight_2026-07-04.json
run_pass expert_storage_preflight bash scripts/vps_expert_storage_preflight.sh GLM-5.2.INT4.GLOBAL-FULL-L3-L78-TOP4-AE2-SMOKE 20 state/zc_expert_storage_preflight_2026-07-04.json
run_expected_blocked reference_readiness bash scripts/vps_reference_readiness.sh models/zai-org/GLM-5.2 state/glm52_reference_readiness_2026-07-04.json
run_expected_blocked indexer_long_context bash scripts/vps_indexer_long_context_guard.sh 2049 state/glm52_indexer_long_context_guard_2026-07-04.json
run_pass chat_stability bash scripts/vps_chat_stability_smoke.sh state/zc_chat_stability_smoke_2026-07-04.json
run_pass perf_summary bash scripts/vps_perf_profile_top4_summary.sh
run_pass quality_prompts_v2 bash scripts/vps_validate_quality_prompts.sh config/quality_prompts.v2.small.json state/zc_quality_prompts_v2_validation_2026-07-04.json
run_pass transformer_math_audit bash scripts/vps_transformer_math_audit.sh logs/zc_sampling_top4_smoke_2026-07-04.server.log state/zc_transformer_math_audit_2026-07-04.json
run_pass deepseek_v4_flash_contract bash scripts/vps_deepseek_v4_flash_contract.sh state/deepseek_v4_flash_contract_2026-07-04.json
run_pass deepseek_v4_flash_tokenizer_contract bash scripts/vps_deepseek_v4_flash_tokenizer_contract.sh state/deepseek_v4_flash_tokenizer_contract_2026-07-04.json
run_pass deepseek_v4_flash_inventory_metadata bash scripts/vps_deepseek_v4_flash_inventory_metadata.sh models/deepseek-ai/DeepSeek-V4-Flash state/deepseek_v4_flash_inventory_metadata_2026-07-04.json
run_pass deepseek_v4_flash_converter_dry_run bash scripts/vps_deepseek_v4_flash_converter_dry_run.sh models/deepseek-ai/DeepSeek-V4-Flash state/deepseek_v4_flash_inventory_metadata_2026-07-04.json state/deepseek_v4_flash_converter_dry_run_2026-07-04.json
run_expected_blocked deepseek_v4_flash_inventory bash scripts/vps_deepseek_v4_flash_inventory_gate.sh models/deepseek-ai/DeepSeek-V4-Flash state/deepseek_v4_flash_inventory_2026-07-04.json
run_expected_blocked deepseek_v4_flash_runtime_smoke bash scripts/vps_deepseek_v4_flash_runtime_smoke_gate.sh state/deepseek_v4_flash_runtime_smoke_gate_2026-07-04.json
run_expected_blocked top8_gate bash scripts/vps_top8_decision_gate.sh state/zc_model_quality_prompt_set_2026-07-04.json state/zc_top4_expert_catalog_2026-07-04.json state/zc_top8_data_driven_gate_2026-07-04.json

python3 - "$out" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
payload = {
    "format": "wohper-public-test-matrix",
    "version": 1,
    "status": "passed",
    "passed": [
        "open_repo_preflight",
        "expert_storage_preflight",
        "chat_stability",
        "perf_summary",
        "quality_prompts_v2",
        "transformer_math_audit",
        "deepseek_v4_flash_contract",
        "deepseek_v4_flash_tokenizer_contract",
        "deepseek_v4_flash_inventory_metadata",
        "deepseek_v4_flash_converter_dry_run",
    ],
    "expected_blocked": [
        "reference_readiness",
        "indexer_long_context",
        "deepseek_v4_flash_inventory",
        "deepseek_v4_flash_runtime_smoke",
        "top8_gate",
    ],
}
out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(json.dumps(payload, indent=2))
PY
