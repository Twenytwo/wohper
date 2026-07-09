#!/usr/bin/env bash
set -euo pipefail

cd /home/deploy/hermes-agent/loop_workspace

model_dir="${1:-models/deepseek-ai/DeepSeek-V4-Flash}"
out="${2:-state/deepseek_v4_flash_inventory_2026-07-04.json}"

PYTHONDONTWRITEBYTECODE=1 python3 tools/plan_deepseek_v4_flash_inventory.py \
  --model-dir "$model_dir" \
  --contract config/deepseek_v4_flash.contract.json \
  --out "$out"
