#!/usr/bin/env bash
set -euo pipefail

cd /home/deploy/hermes-agent/loop_workspace

model_dir="${1:-models/zai-org/GLM-5.2}"
out="${2:-state/glm52_reference_readiness_2026-07-04.json}"

PYTHONDONTWRITEBYTECODE=1 python3 tools/check_glm52_reference_readiness.py \
  --model-dir "$model_dir" \
  --out "$out" || status=$?

status="${status:-0}"
python3 -m json.tool "$out"
exit "$status"
