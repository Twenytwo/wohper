#!/usr/bin/env bash
set -euo pipefail

cd /home/deploy/hermes-agent/loop_workspace

model_name="${1:-GLM-5.2.INT4.GLOBAL-FULL-L3-L78-TOP4-AE2-SMOKE}"
token_id="${2:-154826}"
timeout_sec="${3:-900}"
io_buffer_mb="${4:-256}"
io_buffer_count="${5:-3}"
max_new_tokens="${6:-1}"

if ! [[ "$timeout_sec" =~ ^[0-9]+$ ]] || [ "$timeout_sec" -lt 60 ] || [ "$timeout_sec" -gt 1800 ]; then
  echo "refusing unsafe timeout_sec=$timeout_sec; expected 60..1800" >&2
  exit 2
fi
if ! [[ "$io_buffer_mb" =~ ^[0-9]+$ ]] || [ "$io_buffer_mb" -lt 1 ] || [ "$io_buffer_mb" -gt 512 ]; then
  echo "refusing unsafe io_buffer_mb=$io_buffer_mb; expected 1..512" >&2
  exit 2
fi
if ! [[ "$io_buffer_count" =~ ^[0-9]+$ ]] || [ "$io_buffer_count" -lt 1 ] || [ "$io_buffer_count" -gt 8 ]; then
  echo "refusing unsafe io_buffer_count=$io_buffer_count; expected 1..8" >&2
  exit 2
fi
if [ $((io_buffer_mb * io_buffer_count)) -gt 1024 ]; then
  echo "refusing unsafe total io buffer budget $((io_buffer_mb * io_buffer_count))MB; max 1024MB" >&2
  exit 2
fi
if ! [[ "$max_new_tokens" =~ ^[0-9]+$ ]] || [ "$max_new_tokens" -lt 1 ] || [ "$max_new_tokens" -gt 4 ]; then
  echo "refusing unsafe max_new_tokens=$max_new_tokens; expected 1..4" >&2
  exit 2
fi

socket="/tmp/zc-sampling-top4-${model_name//[^A-Za-z0-9]/-}.sock"
cache="cache/sampling-${model_name}-$(date +%s)"
out="state/zc_sampling_top4_smoke_2026-07-04.ndjson"
server_log="logs/zc_sampling_top4_smoke_2026-07-04.server.log"
model="models/wohper/${model_name}/dense_core.bin"

if [ ! -f "$model" ]; then
  echo "missing model: $model" >&2
  exit 2
fi

rm -f "$socket" "$out"
mkdir -p "$cache" logs state

./engine/zc_infer_core/target-user/release/zc_infer_server \
  --model "$model" \
  --socket "$socket" \
  --active-experts 2 \
  --io-buffer-count "$io_buffer_count" \
  --io-buffer-mb "$io_buffer_mb" \
  --expert-cache-dir "$cache" \
  --expert-cache-gb 2 \
  --expert-remote-endpoint http://127.0.0.1:9101 \
  > "$server_log" 2>&1 &

pid=$!
cleanup() {
  kill "$pid" >/dev/null 2>&1 || true
  wait "$pid" >/dev/null 2>&1 || true
}
trap cleanup EXIT

for _ in $(seq 1 100); do
  [ -S "$socket" ] && break
  sleep 0.1
done

PYTHONDONTWRITEBYTECODE=1 timeout "${timeout_sec}s" python3 tools/zc_socket_smoke_client.py \
  --socket "$socket" \
  --request-id sampling-top4-smoke-2026-07-04 \
  --token-id "$token_id" \
  --max-new-tokens "$max_new_tokens" \
  --experts "" \
  --temperature 0.8 \
  --top-k 3 \
  --top-p 0.9 \
  --repetition-penalty 1.1 \
  --seed 123 \
  > "$out"

grep -q '"source":"lm_head_topk_sample"' "$out"
grep -q '"logit":' "$out"

echo "SAMPLING_TOP4_OUT=$out"
echo "SERVER_LOG=$server_log"
echo "CACHE=$cache"
cat "$out"
tail -80 "$server_log"
