#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
CARGO_BIN="${CARGO_BIN:-cargo}"
CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-${ROOT}/engine/zc_infer_core/target}"
if [ -z "${ZC_HF_MODEL_DIR:-}" ]; then
  if [ -d /mnt/nvme ]; then
    MODEL_DIR="/mnt/nvme/models/zai-org/GLM-5.2"
  else
    MODEL_DIR="${ROOT}/models/zai-org/GLM-5.2"
  fi
else
  MODEL_DIR="${ZC_HF_MODEL_DIR}"
fi

if [ -z "${ZC_MODEL_OUT:-}" ]; then
  if [ -d /mnt/nvme ]; then
    REAL_MODEL_OUT="/mnt/nvme/models/wohper/GLM-5.2.INT4.MODEL.bin"
  else
    REAL_MODEL_OUT="${ROOT}/models/wohper/GLM-5.2.INT4.MODEL.bin"
  fi
else
  REAL_MODEL_OUT="${ZC_MODEL_OUT}"
fi
DUMMY_MODEL="${ZC_DUMMY_MODEL:-${ROOT}/projects/MODEL.dummy.zcblk01.bin}"
SOCKET="${ZC_SOCKET:-/tmp/wohper-infer.sock}"
SERVER_BIN="${CARGO_TARGET_DIR}/release/zc_infer_server"
IO_BENCH_BIN="${CARGO_TARGET_DIR}/release/io_bench"

usage() {
  cat <<'EOF'
Wohper setup helper

Usage:
  scripts/setup_wohper.sh check
  scripts/setup_wohper.sh smoke
  scripts/setup_wohper.sh real

Modes:
  check   Validate local Python/Cargo/kernel/memlock environment.
  smoke   Build Rust, generate the 54MB ZCBLK01 dummy model, run io_bench,
          then run a temporary socket generation smoke.
  real    Prepare real GLM-5.2 conversion from an already downloaded local
          Hugging Face directory and build the release server.

Environment overrides:
  PYTHON_BIN=python3
  CARGO_BIN=cargo
  CARGO_TARGET_DIR=engine/zc_infer_core/target
  ZC_HF_MODEL_DIR=/mnt/nvme/models/zai-org/GLM-5.2
  ZC_MODEL_OUT=/mnt/nvme/models/wohper/GLM-5.2.INT4.MODEL.bin
  ZC_DUMMY_MODEL=projects/MODEL.dummy.zcblk01.bin
  ZC_SOCKET=/tmp/wohper-infer.sock

This script does not write system sysctl/limits files. It prints exact tuning
suggestions when the host is not configured for fixed-buffer io_uring tests.
EOF
}

log() {
  printf '[wohper] %s\n' "$*"
}

warn() {
  printf '[wohper][warn] %s\n' "$*" >&2
}

have() {
  command -v "$1" >/dev/null 2>&1
}

check_command() {
  local cmd="$1"
  if have "$cmd"; then
    log "found ${cmd}: $(command -v "$cmd")"
  else
    warn "missing command: ${cmd}"
    return 1
  fi
}

check_kernel() {
  if [ "$(uname -s)" != "Linux" ]; then
    warn "Wohper runtime is Linux-first; current kernel is $(uname -s)"
    return 0
  fi

  if [ -r /proc/sys/kernel/io_uring_disabled ]; then
    local io_uring_disabled
    io_uring_disabled="$(cat /proc/sys/kernel/io_uring_disabled)"
    log "kernel.io_uring_disabled=${io_uring_disabled}"
    if [ "${io_uring_disabled}" != "0" ]; then
      warn "io_uring is disabled; enable with: sudo sysctl kernel.io_uring_disabled=0"
    fi
  fi

  if [ -r /proc/sys/vm/swappiness ]; then
    local swappiness
    swappiness="$(cat /proc/sys/vm/swappiness)"
    log "vm.swappiness=${swappiness}"
    if [ "${swappiness}" != "1" ]; then
      warn "recommended: sudo sysctl vm.swappiness=1"
    fi
  fi

  local memlock
  memlock="$(ulimit -l || true)"
  log "ulimit -l=${memlock}"
  if [ "${memlock}" != "unlimited" ]; then
    warn "fixed buffers may need more memlock. Suggested limits file:"
    cat >&2 <<'EOF'
sudo tee /etc/security/limits.d/99-zc-infer.conf >/dev/null <<'LIMITS'
* soft memlock unlimited
* hard memlock unlimited
LIMITS
EOF
  fi
}

check_python_module() {
  local module="$1"
  if "${PYTHON_BIN}" -c "import ${module}" >/dev/null 2>&1; then
    log "python module ok: ${module}"
  else
    warn "python module missing: ${module}"
    return 1
  fi
}

check_env() {
  log "workspace=${ROOT}"
  check_command "${PYTHON_BIN}" || true
  check_command "${CARGO_BIN}" || true
  check_kernel
  if have "${PYTHON_BIN}"; then
    "${PYTHON_BIN}" --version || true
  fi
  if have "${CARGO_BIN}"; then
    "${CARGO_BIN}" --version || true
  elif [ -x "${SERVER_BIN}" ] && [ -x "${IO_BENCH_BIN}" ]; then
    log "cargo missing, but existing release binaries are present"
  fi
  log "for chat bridge dependencies, use a local venv and install:"
  log "  ${PYTHON_BIN} -m pip install transformers accelerate safetensors torch sentencepiece protobuf huggingface_hub"
}

build_release() {
  if have "${CARGO_BIN}"; then
    log "building Rust release binaries"
    "${CARGO_BIN}" build --release --manifest-path "${ROOT}/engine/zc_infer_core/Cargo.toml" --bins
    return
  fi

  if [ -x "${SERVER_BIN}" ] && [ -x "${IO_BENCH_BIN}" ]; then
    log "cargo missing; reusing existing release binaries"
    return
  fi

  warn "cargo is missing and release binaries are not present"
  warn "install Rust/Cargo, or build with the Docker dev image from README.md"
  return 2
}

run_smoke() {
  check_env
  mkdir -p "${ROOT}/projects"

  log "generating dummy ZCBLK01 MODEL.bin"
  "${PYTHON_BIN}" "${ROOT}/scripts/generate_dummy_model.py" --out "${DUMMY_MODEL}"

  build_release

  log "running io_bench smoke"
  "${IO_BENCH_BIN}" \
    --model "${DUMMY_MODEL}" \
    --rounds 2 \
    --active-experts 2 \
    --compressed-buffer-mb 16 \
    --runtime-buffer-mb 0 \
    --io-buffer-count 6 \
    --pipeline-depth 2

  log "running temporary Unix socket generation smoke"
  rm -f "${SOCKET}"
  "${SERVER_BIN}" \
    --model "${DUMMY_MODEL}" \
    --socket "${SOCKET}" \
    --active-experts 2 \
    --pipeline-depth 2 \
    --io-buffer-count 6 \
    --io-buffer-mb 16 \
    --runtime-buffer-mb 0 \
    --stop-token-id 154820 &
  local server_pid=$!

  cleanup() {
    kill "${server_pid}" >/dev/null 2>&1 || true
    wait "${server_pid}" >/dev/null 2>&1 || true
  }
  trap cleanup EXIT

  for _ in $(seq 1 50); do
    [ -S "${SOCKET}" ] && break
    sleep 0.1
  done

  "${PYTHON_BIN}" "${ROOT}/scripts/smoke_dummy_generation.py" \
    --socket "${SOCKET}" \
    --tokens 1,2,3 \
    --experts 0,1 \
    --max-new-tokens 2

  cleanup
  trap - EXIT
  log "smoke complete"
}

ensure_model_dir() {
  if [ ! -d "${MODEL_DIR}" ]; then
    warn "model directory does not exist: ${MODEL_DIR}"
    warn "download with:"
    cat >&2 <<EOF
python3 -m pip install -U "huggingface_hub[cli]"
hf download zai-org/GLM-5.2 --local-dir "${MODEL_DIR}" --local-dir-use-symlinks False
EOF
    return 1
  fi
}

run_real_setup() {
  check_env
  check_python_module safetensors || true
  check_python_module torch || true
  check_python_module numpy || true
  check_python_module transformers || true

  ensure_model_dir

  if ! ls "${MODEL_DIR}"/*.safetensors >/dev/null 2>&1; then
    warn "no safetensors shards found in ${MODEL_DIR}"
    warn "real conversion requires the full local GLM-5.2 checkpoint"
    return 2
  fi

  mkdir -p "$(dirname "${REAL_MODEL_OUT}")"
  log "planning GLM-5.2 tensor map"
  "${PYTHON_BIN}" "${ROOT}/tools/convert_safetensors.py" \
    --model-dir "${MODEL_DIR}" \
    --plan-only

  log "converting GLM-5.2 to Wohper INT4 MODEL.bin"
  "${PYTHON_BIN}" "${ROOT}/tools/convert_safetensors.py" \
    --model-dir "${MODEL_DIR}" \
    --out "${REAL_MODEL_OUT}" \
    --quant int4 \
    --pack-global-into-layer0

  build_release

  log "real setup complete"
  log "start server:"
  cat <<EOF
${SERVER_BIN} \\
  --model "${REAL_MODEL_OUT}" \\
  --socket "${SOCKET}" \\
  --active-experts 8 \\
  --pipeline-depth 4 \\
  --io-buffer-count 12 \\
  --io-buffer-mb 128 \\
  --runtime-buffer-mb 0 \\
  --stop-token-id 154820
EOF
  log "chat bridge:"
  cat <<EOF
${PYTHON_BIN} "${ROOT}/tools/chat_interface.py" \\
  --backend hybrid \\
  --model-dir "${MODEL_DIR}" \\
  --socket "${SOCKET}"
EOF
}

if [ -z "${MODE}" ]; then
  usage
  exit 0
fi

case "${MODE}" in
  check)
    check_env
    ;;
  smoke)
    run_smoke
    ;;
  real)
    run_real_setup
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
