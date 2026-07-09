#!/usr/bin/env bash
set -euo pipefail

cd /home/deploy/hermes-agent/loop_workspace

model_dir="${1:-models/deepseek-ai/DeepSeek-V4-Flash}"
inventory="${2:-state/deepseek_v4_flash_inventory_metadata_2026-07-04.json}"
out="${3:-state/deepseek_v4_flash_converter_dry_run_2026-07-04.json}"

PYTHONDONTWRITEBYTECODE=1 python3 tools/deepseek_v4_converter_dry_run.py \
  --model-dir "$model_dir" \
  --inventory "$inventory" \
  --out "$out"
