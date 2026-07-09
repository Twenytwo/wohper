#!/usr/bin/env bash
set -euo pipefail

cd /home/deploy/hermes-agent/loop_workspace

log="${1:-logs/zc_sampling_top4_smoke_2026-07-04.server.log}"
out="${2:-state/zc_transformer_math_audit_2026-07-04.json}"

PYTHONDONTWRITEBYTECODE=1 python3 tools/audit_transformer_math_log.py \
  --log "$log" \
  --out "$out"

python3 -m json.tool "$out"
