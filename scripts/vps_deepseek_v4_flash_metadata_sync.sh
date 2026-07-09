#!/usr/bin/env bash
set -euo pipefail

cd /home/deploy/hermes-agent/loop_workspace

model_dir="${1:-models/deepseek-ai/DeepSeek-V4-Flash}"
report="${2:-state/deepseek_v4_flash_metadata_sync_2026-07-04.json}"

PYTHONDONTWRITEBYTECODE=1 python3 tools/download_hf_metadata_only.py \
  --repo-id deepseek-ai/DeepSeek-V4-Flash \
  --revision main \
  --out-dir "$model_dir" \
  --report "$report" \
  --max-file-bytes $((32 * 1024 * 1024)) \
  --max-total-bytes $((256 * 1024 * 1024))
