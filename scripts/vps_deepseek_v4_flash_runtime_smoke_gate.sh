#!/usr/bin/env bash
set -euo pipefail

cd /home/deploy/hermes-agent/loop_workspace

out="${1:-state/deepseek_v4_flash_runtime_smoke_gate_2026-07-04.json}"

PYTHONDONTWRITEBYTECODE=1 python3 tools/deepseek_v4_runtime_smoke_gate.py \
  --readiness state/deepseek_v4_flash_readiness_2026-07-04.json \
  --inventory state/deepseek_v4_flash_inventory_metadata_2026-07-04.json \
  --quality-prompts config/deepseek_v4_quality_prompts.small.json \
  --out "$out"
