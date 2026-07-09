#!/usr/bin/env bash
# Long-running DeepSeek-V4-Flash server (E2): starts zc_infer_server on a
# Unix socket and keeps it alive. The generation runtime is persistent, so
# the RAM caches (dense blocks, lm_head) stay warm across chat turns.
# Meant to run inside the zc-infer-dev container (detached):
#   docker run -d --name zc-chat ... zc-infer-dev bash scripts/deepseek_chat_server.sh
# Then per turn:
#   docker exec zc-chat python3 tools/zc_socket_smoke_client.py --socket ... --token-ids ...
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODEL="${ZC_DEEPSEEK_MODEL:-${ROOT}/models/wohper/DeepSeek-V4-Flash.RAW.L0-L43-SPLIT-GLOBAL-CATALOGSEED/dense_core.bin}"
SOCKET="${ZC_SOCKET:-/tmp/wohper-chat.sock}"
CARGO_BIN="${CARGO_BIN:-cargo}"
CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-${ROOT}/engine/zc_infer_core/target}"
SERVER_BIN="${CARGO_TARGET_DIR}/release/zc_infer_server"

ACTIVE_EXPERTS="${ZC_ACTIVE_EXPERTS:-6}"
PIPELINE_DEPTH="${ZC_PIPELINE_DEPTH:-1}"
IO_BUFFER_COUNT="${ZC_IO_BUFFER_COUNT:-10}"
IO_BUFFER_MB="${ZC_IO_BUFFER_MB:-192}"
RUNTIME_BUFFER_MB="${ZC_RUNTIME_BUFFER_MB:-0}"
LOCAL_LAYER_START="${ZC_LOCAL_LAYER_START:-0}"
LOCAL_LAYER_END="${ZC_LOCAL_LAYER_END:-}"
export ZC_SCHEDULER_WAIT_MS="${ZC_SCHEDULER_WAIT_MS:-300000}"
# Demo cache defaults (override via env)
export ZC_DENSE_CACHE_MB="${ZC_DENSE_CACHE_MB:-6500}"
export ZC_LMHEAD_CACHE="${ZC_LMHEAD_CACHE:-1}"
export ZC_SIDECAR_EXPERTS="${ZC_SIDECAR_EXPERTS:-all}"
# Expert RAM cache OFF by default on 12GB WSL2: one token round touches
# ~258 experts (~3.4GB) - a smaller LRU thrashes (0 hits) and the extra
# RAM pressure evicts the OS page cache that expert reads rely on.
# Enable (e.g. 4000+) only with 24GB+ available to the container.
export ZC_EXPERT_RAM_CACHE_MB="${ZC_EXPERT_RAM_CACHE_MB:-0}"

if [ ! -f "${MODEL}" ]; then
  echo "missing model: ${MODEL}" >&2
  exit 2
fi

if [ "${ZC_FORCE_BUILD:-0}" = "1" ] || [ ! -x "${SERVER_BIN}" ]; then
  echo "[chat-server] building zc_infer_server"
  "${CARGO_BIN}" build --release --manifest-path "${ROOT}/engine/zc_infer_core/Cargo.toml" --bin zc_infer_server
fi

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
# V2: expert cache on the fast ext4 volume (the default cache/experts is
# on the slow Windows bind mount and would negate the model migration).
if [ -n "${ZC_EXPERT_CACHE_DIR:-}" ]; then
  SERVER_ARGS+=(--expert-cache-dir "${ZC_EXPERT_CACHE_DIR}")
fi
if [ -n "${ZC_EXPERT_CACHE_GB:-}" ]; then
  SERVER_ARGS+=(--expert-cache-gb "${ZC_EXPERT_CACHE_GB}")
fi

echo "[chat-server] model=${MODEL}"
echo "[chat-server] socket=${SOCKET} dense_cache_mb=${ZC_DENSE_CACHE_MB} lmhead_cache=${ZC_LMHEAD_CACHE}"
exec "${SERVER_BIN}" "${SERVER_ARGS[@]}"
