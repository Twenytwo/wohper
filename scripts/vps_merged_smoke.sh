#!/usr/bin/env bash
set -euo pipefail

cd /home/deploy/hermes-agent/loop_workspace

model_name="${1:-GLM-5.2.INT4.GLOBAL-L3-L5-SMOKE}"
expected_layers="${2:-3,4}"
active_experts="${3:-1}"
io_buffer_mb="${4:-128}"
io_buffer_count="${5:-4}"
input_token_id="${6:-42}"

if ! [[ "$io_buffer_mb" =~ ^[0-9]+$ ]] || ! [[ "$io_buffer_count" =~ ^[0-9]+$ ]]; then
  echo "io_buffer_mb and io_buffer_count must be positive integers" >&2
  exit 2
fi

if [ "$io_buffer_mb" -lt 1 ] || [ "$io_buffer_mb" -gt 512 ]; then
  echo "refusing unsafe io_buffer_mb=$io_buffer_mb; expected 1..512" >&2
  exit 2
fi

if [ "$io_buffer_count" -lt 1 ] || [ "$io_buffer_count" -gt 8 ]; then
  echo "refusing unsafe io_buffer_count=$io_buffer_count; expected 1..8" >&2
  exit 2
fi

total_io_buffer_mb=$((io_buffer_mb * io_buffer_count))
if [ "$total_io_buffer_mb" -gt 1024 ]; then
  echo "refusing unsafe total io buffer budget ${total_io_buffer_mb}MB; max 1024MB" >&2
  exit 2
fi

socket="/tmp/wohper-${model_name//[^A-Za-z0-9]/-}.sock"
cache="cache/${model_name}-$(date +%s)"
log="logs/zc_infer_server_${model_name}.log"
pid_file="state/zc_infer_server_${model_name}.pid"
model="models/wohper/${model_name}/dense_core.bin"

if [ ! -f "$model" ]; then
  echo "missing model: $model" >&2
  exit 2
fi

rm -f "$socket"
mkdir -p "$cache" logs state

./engine/zc_infer_core/target-user/release/zc_infer_server \
  --model "$model" \
  --socket "$socket" \
  --active-experts "$active_experts" \
  --io-buffer-count "$io_buffer_count" \
  --io-buffer-mb "$io_buffer_mb" \
  --expert-cache-dir "$cache" \
  --expert-cache-gb 2 \
  --expert-remote-endpoint http://127.0.0.1:9101 \
  > "$log" 2>&1 &

pid=$!
echo "$pid" > "$pid_file"
trap 'kill "$pid" 2>/dev/null || true' EXIT

for _ in $(seq 1 80); do
  if [ -S "$socket" ]; then
    break
  fi
  sleep 0.1
done

PYTHONDONTWRITEBYTECODE=1 timeout 180s python3 tools/zc_socket_smoke_client.py \
  --socket "$socket" \
  --request-id "${model_name}-smoke" \
  --token-id "$input_token_id" \
  --max-new-tokens 1 \
  --experts ""

client_exit=$?
echo "CLIENT_EXIT=$client_exit"
echo "CACHE=$cache"
find "$cache" -maxdepth 1 -type f -printf "%f %s\n" | sort || true
echo "LOG"
tail -220 "$log"

grep -q "embedding source=embed_tokens" "$log"
grep -q "sampling source=lm_head" "$log"
IFS=',' read -r -a layers <<< "$expected_layers"
for layer in "${layers[@]}"; do
  grep -q "router layer=${layer} source=router" "$log"
  grep -q "compute layer=${layer}" "$log"
done

exit "$client_exit"
