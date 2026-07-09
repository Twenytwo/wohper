#!/usr/bin/env bash
set -euo pipefail

cd /home/deploy/hermes-agent/loop_workspace

quality_state="${1:-state/zc_model_quality_prompt_set_2026-07-04.json}"
top4_catalog="${2:-state/zc_top4_expert_catalog_2026-07-04.json}"
out="${3:-state/zc_top8_data_driven_gate_2026-07-04.json}"

PYTHONDONTWRITEBYTECODE=1 python3 tools/top8_decision_gate.py \
  --quality-state "$quality_state" \
  --top4-catalog "$top4_catalog" \
  --out "$out" || status=$?

status="${status:-0}"
python3 -m json.tool "$out"
exit "$status"
