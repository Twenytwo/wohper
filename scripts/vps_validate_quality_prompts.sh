#!/usr/bin/env bash
set -euo pipefail

cd /home/deploy/hermes-agent/loop_workspace

prompts="${1:-config/quality_prompts.v2.small.json}"
out="${2:-state/zc_quality_prompts_v2_validation_2026-07-04.json}"

PYTHONDONTWRITEBYTECODE=1 python3 tools/validate_quality_prompts.py \
  --prompts "$prompts" \
  --out "$out"

python3 -m json.tool "$out"
