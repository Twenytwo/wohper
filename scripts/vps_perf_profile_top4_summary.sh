#!/usr/bin/env bash
set -euo pipefail

cd /home/deploy/hermes-agent/loop_workspace

server_log="${1:-logs/zc_quality_bench_GLM-5.2.INT4.GLOBAL-FULL-L3-L78-TOP4-AE2-SMOKE_top4_smoke2.server.log}"
bench_json="${2:-state/zc_quality_bench_GLM-5.2.INT4.GLOBAL-FULL-L3-L78-TOP4-AE2-SMOKE_top4_smoke2.json}"
out="${3:-state/zc_perf_profile_top4_summary_2026-07-04.json}"

PYTHONDONTWRITEBYTECODE=1 python3 tools/summarize_zc_perf_log.py \
  --server-log "$server_log" \
  --bench-json "$bench_json" \
  --out "$out"

python3 -m json.tool "$out"
