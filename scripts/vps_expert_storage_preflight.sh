#!/usr/bin/env bash
set -euo pipefail

cd /home/deploy/hermes-agent/loop_workspace

model_name="${1:-GLM-5.2.INT4.GLOBAL-FULL-L3-L78-TOP4-AE2-SMOKE}"
min_free_gb="${2:-20}"
out="${3:-state/zc_expert_storage_preflight_2026-07-04.json}"
model_dir="models/wohper/${model_name}"
dense_core="${model_dir}/dense_core.bin"
expert_dir="${model_dir}/experts"

if ! [[ "$min_free_gb" =~ ^[0-9]+$ ]] || [ "$min_free_gb" -lt 1 ] || [ "$min_free_gb" -gt 200 ]; then
  echo "refusing unsafe min_free_gb=$min_free_gb; expected 1..200" >&2
  exit 2
fi

mkdir -p state

free_kb="$(df -Pk . | awk 'NR==2 {print $4}')"
free_gb="$((free_kb / 1024 / 1024))"
dense_bytes=0
[ -f "$dense_core" ] && dense_bytes="$(stat -c %s "$dense_core")"
expert_files=0
expert_bytes=0
if [ -d "$expert_dir" ]; then
  expert_files="$(find "$expert_dir" -type f | wc -l | tr -d ' ')"
  expert_bytes="$(find "$expert_dir" -type f -printf '%s\n' | awk '{s+=$1} END {print s+0}')"
fi
status="ready"
if [ ! -f "$dense_core" ]; then
  status="blocked_missing_dense_core"
elif [ "$free_gb" -lt "$min_free_gb" ]; then
  status="blocked_low_disk"
fi

cat > "$out" <<JSON
{
  "format": "wohper-expert-storage-preflight",
  "version": 1,
  "model_name": "$model_name",
  "model_dir": "$model_dir",
  "dense_core": "$dense_core",
  "dense_core_exists": $([ -f "$dense_core" ] && echo true || echo false),
  "dense_core_bytes": $dense_bytes,
  "expert_dir": "$expert_dir",
  "expert_dir_exists": $([ -d "$expert_dir" ] && echo true || echo false),
  "expert_files": $expert_files,
  "expert_bytes": $expert_bytes,
  "cache_policy": "run-specific cache/<name>, capped by script",
  "remote_endpoint_default": "http://127.0.0.1:9101",
  "free_gb": $free_gb,
  "min_free_gb": $min_free_gb,
  "status": "$status"
}
JSON

python3 -m json.tool "$out"
[ "$status" = "ready" ]
