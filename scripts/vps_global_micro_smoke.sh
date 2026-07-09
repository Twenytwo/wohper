#!/usr/bin/env bash
set -euo pipefail

cd /home/deploy/hermes-agent/loop_workspace

socket="/tmp/wohper-global-micro.sock"
log="logs/zc_infer_server_global_micro.log"
pid_file="state/zc_infer_server_global_micro.pid"
model="models/wohper/GLM-5.2.INT4.GLOBAL-MICRO/dense_core.bin"

if [ ! -f "$model" ]; then
  echo "missing model: $model" >&2
  exit 2
fi

rm -f "$socket"
mkdir -p logs state

./engine/zc_infer_core/target-user/release/zc_infer_server \
  --model "$model" \
  --socket "$socket" \
  --active-experts 0 \
  --io-buffer-count 2 \
  --io-buffer-mb 64 \
  --expert-cache-dir cache/global-micro-empty \
  --expert-cache-gb 1 \
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

PYTHONDONTWRITEBYTECODE=1 timeout 60s python3 tools/zc_socket_smoke_client.py \
  --socket "$socket" \
  --request-id global-micro-smoke \
  --token-id 42 \
  --max-new-tokens 1 \
  --experts ""

client_exit=$?
echo "CLIENT_EXIT=$client_exit"
echo "LOG"
tail -120 "$log"

if ! grep -q "embedding source=embed_tokens" "$log"; then
  echo "expected embedding source=embed_tokens in log" >&2
  exit 3
fi
if ! grep -q "sampling source=lm_head" "$log"; then
  echo "expected sampling source=lm_head in log" >&2
  exit 4
fi

exit "$client_exit"
