#!/usr/bin/env bash
set -euo pipefail

cd /home/deploy/hermes-agent/loop_workspace

out="${1:-state/deepseek_v4_flash_tokenizer_contract_2026-07-04.json}"

PYTHONDONTWRITEBYTECODE=1 python3 tools/check_deepseek_v4_flash_tokenizer_contract.py \
  --contract config/deepseek_v4_flash.tokenizer_contract.json \
  --model-dir models/deepseek-ai/DeepSeek-V4-Flash \
  --out "$out" \
  --contract-only
