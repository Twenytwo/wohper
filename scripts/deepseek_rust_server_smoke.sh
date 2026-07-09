#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODEL="${ZC_DEEPSEEK_MODEL:-${ROOT}/models/wohper/DeepSeek-V4-Flash.RAW.L0-L43-SPLIT-GLOBAL-CATALOGSEED/dense_core.bin}"
SOCKET="${ZC_SOCKET:-/tmp/wohper-deepseek-smoke.sock}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CARGO_BIN="${CARGO_BIN:-cargo}"
CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-${ROOT}/engine/zc_infer_core/target}"
SERVER_BIN="${CARGO_TARGET_DIR}/release/zc_infer_server"

TOKENS="${ZC_TOKENS:-42}"
EXPERTS="${ZC_EXPERTS-0,1}"
MAX_NEW_TOKENS="${ZC_MAX_NEW_TOKENS:-1}"
ACTIVE_EXPERTS="${ZC_ACTIVE_EXPERTS:-2}"
PIPELINE_DEPTH="${ZC_PIPELINE_DEPTH:-1}"
IO_BUFFER_COUNT="${ZC_IO_BUFFER_COUNT:-4}"
IO_BUFFER_MB="${ZC_IO_BUFFER_MB:-192}"
RUNTIME_BUFFER_MB="${ZC_RUNTIME_BUFFER_MB:-0}"
SERVER_TIMEOUT="${ZC_SERVER_TIMEOUT:-900s}"
CLIENT_TIMEOUT="${ZC_CLIENT_TIMEOUT:-480s}"
LOCAL_LAYER_START="${ZC_LOCAL_LAYER_START:-0}"
LOCAL_LAYER_END="${ZC_LOCAL_LAYER_END:-}"
export ZC_SCHEDULER_WAIT_MS="${ZC_SCHEDULER_WAIT_MS:-300000}"

log() {
  printf '[deepseek-rust-smoke] %s\n' "$*"
}

if [ ! -f "${MODEL}" ]; then
  echo "missing model: ${MODEL}" >&2
  exit 2
fi

if [ "${ZC_FORCE_BUILD:-0}" = "1" ] || [ ! -x "${SERVER_BIN}" ]; then
  log "building zc_infer_server"
  "${CARGO_BIN}" build --release --manifest-path "${ROOT}/engine/zc_infer_core/Cargo.toml" --bin zc_infer_server
fi

rm -f "${SOCKET}"

cleanup() {
  if [ -n "${SERVER_PID:-}" ]; then
    kill "${SERVER_PID}" >/dev/null 2>&1 || true
    wait "${SERVER_PID}" >/dev/null 2>&1 || true
  fi
  rm -f "${SOCKET}"
}
trap cleanup EXIT

log "model=${MODEL}"
log "socket=${SOCKET}"
log "tokens=${TOKENS} experts=${EXPERTS} max_new_tokens=${MAX_NEW_TOKENS}"
log "io_buffer_count=${IO_BUFFER_COUNT} io_buffer_mb=${IO_BUFFER_MB}"
CLIENT_OUT="$(mktemp)"
SERVER_ARGS=(
  --model "${MODEL}"
  --socket "${SOCKET}"
  --active-experts "${ACTIVE_EXPERTS}"
  --pipeline-depth "${PIPELINE_DEPTH}"
  --io-buffer-count "${IO_BUFFER_COUNT}"
  --io-buffer-mb "${IO_BUFFER_MB}"
  --runtime-buffer-mb "${RUNTIME_BUFFER_MB}"
  --local-layer-start "${LOCAL_LAYER_START}"
)
if [ -n "${LOCAL_LAYER_END}" ]; then
  SERVER_ARGS+=(--local-layer-end "${LOCAL_LAYER_END}")
fi

timeout --kill-after=5s "${SERVER_TIMEOUT}" "${SERVER_BIN}" "${SERVER_ARGS[@]}" &
SERVER_PID=$!

for _ in $(seq 1 200); do
  if ! kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
    echo "server exited before socket became ready" >&2
    exit 4
  fi
  [ -S "${SOCKET}" ] && break
  sleep 0.1
done

if [ ! -S "${SOCKET}" ]; then
  echo "server socket did not appear: ${SOCKET}" >&2
  exit 3
fi

timeout --kill-after=5s "${CLIENT_TIMEOUT}" "${PYTHON_BIN}" "${ROOT}/tools/zc_socket_smoke_client.py" \
  --socket "${SOCKET}" \
  --request-id "deepseek-rust-l0-l43-smoke" \
  --token-ids "${TOKENS}" \
  --experts "${EXPERTS}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --temperature 0 \
  --top-k 1 \
  --top-p 1 \
  --repetition-penalty 1 \
  --seed 1 | tee "${CLIENT_OUT}"

if ! grep -q '"event":"Finished"' "${CLIENT_OUT}"; then
  echo "client stream did not contain Finished event" >&2
  exit 5
fi

log "smoke complete"
