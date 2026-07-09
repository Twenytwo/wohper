#!/usr/bin/env bash
set -euo pipefail

cd /home/deploy/hermes-agent/loop_workspace

socket="/tmp/wohper-l3r-router.sock"
cache="cache/glm52-l3r-e0e1-router-smoke-$(date +%s)"
log="logs/zc_infer_server_l3r_router.log"
pid_file="state/zc_infer_server_l3r_router.pid"

rm -f "$socket"
mkdir -p "$cache" logs state

./engine/zc_infer_core/target-user/release/zc_infer_server \
  --model models/wohper/GLM-5.2.INT4.L3R-E0E1/dense_core.bin \
  --socket "$socket" \
  --active-experts 1 \
  --io-buffer-count 8 \
  --io-buffer-mb 32 \
  --expert-cache-dir "$cache" \
  --expert-cache-gb 1 \
  --expert-remote-endpoint http://127.0.0.1:9101 \
  > "$log" 2>&1 &

pid=$!
echo "$pid" > "$pid_file"
trap 'kill "$pid" 2>/dev/null || true' EXIT

for _ in $(seq 1 50); do
  if [ -S "$socket" ]; then
    break
  fi
  sleep 0.1
done

PYTHONDONTWRITEBYTECODE=1 timeout 30s python3 tools/zc_socket_smoke_client.py \
  --socket "$socket" \
  --request-id l3r-router-smoke \
  --token-id 42 \
  --max-new-tokens 1 \
  --experts ""

client_exit=$?
echo "CLIENT_EXIT=$client_exit"
echo "CACHE=$cache"
find "$cache" -maxdepth 1 -type f -printf "%f %s\n" | sort
echo "LOG"
tail -100 "$log"
exit "$client_exit"
