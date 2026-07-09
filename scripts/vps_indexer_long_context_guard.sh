#!/usr/bin/env bash
set -euo pipefail

cd /home/deploy/hermes-agent/loop_workspace

context_len="${1:-2049}"
out="${2:-state/glm52_indexer_long_context_guard_2026-07-04.json}"

PYTHONDONTWRITEBYTECODE=1 python3 tools/check_glm52_indexer_long_context.py \
  --context-len "$context_len" \
  --out "$out" || status=$?

status="${status:-0}"
python3 -m json.tool "$out"
exit "$status"
