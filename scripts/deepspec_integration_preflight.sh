#!/usr/bin/env bash
set -euo pipefail

repo="${1:-vendor/deepspec}"
out="${2:-state/deepspec_integration_readiness_2026-07-04.json}"

PYTHONDONTWRITEBYTECODE=1 python3 tools/check_deepspec_integration_readiness.py \
  --repo "$repo" \
  --out "$out"
