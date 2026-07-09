#!/usr/bin/env bash
set -euo pipefail

cd /home/deploy/hermes-agent/loop_workspace

model_name="${1:?model name required}"
bench_name="${2:-quality}"
active_experts="${3:-2}"
io_buffer_mb="${4:-256}"
io_buffer_count="${5:-3}"
max_new_tokens="${6:-1}"
timeout_sec="${7:-900}"
prompt_limit="${8:-2}"

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
if [ $((io_buffer_mb * io_buffer_count)) -gt 1024 ]; then
  echo "refusing unsafe total io buffer budget $((io_buffer_mb * io_buffer_count))MB; max 1024MB" >&2
  exit 2
fi
if ! [[ "$max_new_tokens" =~ ^[0-9]+$ ]] || [ "$max_new_tokens" -lt 1 ] || [ "$max_new_tokens" -gt 8 ]; then
  echo "refusing unsafe max_new_tokens=$max_new_tokens; expected 1..8" >&2
  exit 2
fi
if ! [[ "$timeout_sec" =~ ^[0-9]+$ ]] || [ "$timeout_sec" -lt 60 ] || [ "$timeout_sec" -gt 1800 ]; then
  echo "refusing unsafe timeout_sec=$timeout_sec; expected 60..1800" >&2
  exit 2
fi
if ! [[ "$prompt_limit" =~ ^[0-9]+$ ]] || [ "$prompt_limit" -lt 1 ] || [ "$prompt_limit" -gt 8 ]; then
  echo "refusing unsafe prompt_limit=$prompt_limit; expected 1..8" >&2
  exit 2
fi

socket="/tmp/wohper-bench-${model_name//[^A-Za-z0-9]/-}.sock"
cache="cache/bench-${model_name}-${bench_name}-$(date +%s)"
server_log="logs/zc_quality_bench_${model_name}_${bench_name}.server.log"
bench_json="state/zc_quality_bench_${model_name}_${bench_name}.json"
bench_stdout="logs/zc_quality_bench_${model_name}_${bench_name}.stdout.log"
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
  > "$server_log" 2>&1 &

pid=$!
trap 'kill "$pid" 2>/dev/null || true' EXIT

for _ in $(seq 1 100); do
  [ -S "$socket" ] && break
  sleep 0.1
done

PYTHONDONTWRITEBYTECODE=1 timeout "${timeout_sec}s" python3 tools/zc_quality_bench.py \
  --socket "$socket" \
  --model-name "$model_name" \
  --prompts config/quality_prompts.small.json \
  --model-dir models/zai-org/GLM-5.2 \
  --out "$bench_json" \
  --max-new-tokens "$max_new_tokens" \
  --limit "$prompt_limit" \
  --timeout-sec "$timeout_sec" \
  | tee "$bench_stdout"

echo "BENCH_JSON=$bench_json"
echo "BENCH_STDOUT=$bench_stdout"
echo "SERVER_LOG=$server_log"
echo "CACHE=$cache"
tail -80 "$server_log"
