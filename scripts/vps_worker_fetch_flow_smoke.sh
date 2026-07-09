#!/usr/bin/env bash
set -euo pipefail

cd /home/deploy/hermes-agent/loop_workspace

model_name="${1:-GLM-5.2.INT4.GLOBAL-32K-L3-L15-E0E3-AE2-SMOKE}"
layer_id="${2:-3}"
expert_id="${3:-0}"
endpoint="${4:-http://127.0.0.1:9101}"
cache="cache/worker-fetch-${model_name}-L${layer_id}-E${expert_id}-$(date +%s)"
model="models/wohper/${model_name}/dense_core.bin"
fetch_bin="./engine/zc_infer_core/target-user/release/zc_remote_fetch_smoke"

if [ ! -f "$model" ]; then
  echo "missing model: $model" >&2
  exit 2
fi
if [ ! -x "$fetch_bin" ]; then
  echo "missing executable: $fetch_bin" >&2
  exit 3
fi

python3 - "$endpoint" <<'PY'
import json
import sys
import urllib.request

endpoint = sys.argv[1].rstrip("/")
with urllib.request.urlopen(endpoint + "/stats", timeout=10) as response:
    payload = json.loads(response.read().decode("utf-8"))
print("worker_stats_ok free_bytes={}".format(payload.get("disk_free_bytes", 0)))
PY

rm -rf "$cache"
mkdir -p "$cache"

"$fetch_bin" \
  --model "$model" \
  --endpoint "$endpoint" \
  --cache-dir "$cache" \
  --layer-id "$layer_id" \
  --expert-id "$expert_id" \
  --max-cache-bytes 2147483648

echo "CACHE=$cache"
find "$cache" -maxdepth 1 -type f -printf "%f %s\n" | sort
