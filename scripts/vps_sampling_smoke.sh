#!/usr/bin/env bash
set -euo pipefail

cd /home/deploy/hermes-agent/loop_workspace

model="${1:-projects/MODEL.dummy.zcblk01.bin}"
socket="${2:-/tmp/zc-sampling-smoke.sock}"
out="${3:-state/zc_sampling_smoke_2026-07-04.ndjson}"
server_log="${4:-logs/zc_sampling_smoke_2026-07-04.server.log}"

if [ ! -f "$model" ]; then
  echo "missing model: $model" >&2
  exit 2
fi

rm -f "$socket" "$out"
mkdir -p logs state

./engine/zc_infer_core/target-user/release/zc_infer_server \
  --model "$model" \
  --socket "$socket" \
  --active-experts 2 \
  --pipeline-depth 2 \
  --io-buffer-count 6 \
  --io-buffer-mb 16 \
  --runtime-buffer-mb 64 \
  > "$server_log" 2>&1 &

pid=$!
cleanup() {
  kill "$pid" >/dev/null 2>&1 || true
  wait "$pid" >/dev/null 2>&1 || true
}
trap cleanup EXIT

for _ in $(seq 1 80); do
  [ -S "$socket" ] && break
  sleep 0.1
done

python3 tools/zc_socket_smoke_client.py \
  --socket "$socket" \
  --request-id sampling-smoke-2026-07-04 \
  --token-id 42 \
  --max-new-tokens 1 \
  --experts 0,1 \
  --temperature 0.8 \
  --top-k 3 \
  --top-p 0.9 \
  --repetition-penalty 1.1 \
  --seed 123 \
  > "$out"

grep -q '"source":"lm_head_topk_sample"' "$out"
grep -q '"logit":' "$out"

echo "SAMPLING_SMOKE_OUT=$out"
echo "SERVER_LOG=$server_log"
cat "$out"
tail -40 "$server_log"
