#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${MODEL:-${ROOT}/projects/MODEL.fake.2g.bin}"
LAYERS="${LAYERS:-16}"
EXPERTS_PER_LAYER="${EXPERTS_PER_LAYER:-8}"
ACTIVE_EXPERTS="${ACTIVE_EXPERTS:-2}"
DENSE_SIZE="${DENSE_SIZE:-8mb}"
EXPERT_SIZE="${EXPERT_SIZE:-14mb}"
ROUNDS="${ROUNDS:-2}"
COMPRESSED_BUFFER_MB="${COMPRESSED_BUFFER_MB:-256}"
RUNTIME_BUFFER_MB="${RUNTIME_BUFFER_MB:-0}"
IO_BUFFER_COUNT="${IO_BUFFER_COUNT:-6}"
PIPELINE_DEPTH="${PIPELINE_DEPTH:-4}"
PRINT_PLAN_LAYERS="${PRINT_PLAN_LAYERS:-8}"

cd "${ROOT}"

if [ -f "${HOME}/.cargo/env" ]; then
  # rustup installs cargo here for non-root users; source it for non-login shells.
  # shellcheck disable=SC1091
  . "${HOME}/.cargo/env"
fi

if [ -x /usr/local/cargo/bin/cargo ]; then
  export PATH="/usr/local/cargo/bin:${PATH:-/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin}"
fi

if ! command -v cargo >/dev/null 2>&1; then
  echo "cargo is required on the Linux test machine." >&2
  exit 2
fi

mkdir -p projects logs

echo "[1/3] Building zc_infer_core release benchmark"
cargo build --release --manifest-path engine/zc_infer_core/Cargo.toml --bin io_bench

if [ ! -f "${MODEL}" ]; then
  echo "[2/3] Generating fake MODEL.bin at ${MODEL}"
  python3 tools/make_fake_model.py \
    --out "${MODEL}" \
    --layers "${LAYERS}" \
    --experts-per-layer "${EXPERTS_PER_LAYER}" \
    --active-experts "${ACTIVE_EXPERTS}" \
    --dense-size "${DENSE_SIZE}" \
    --expert-size "${EXPERT_SIZE}"
else
  echo "[2/3] Reusing existing fake model ${MODEL}"
fi

echo "[3/3] Running io_bench"
"${ROOT}/engine/zc_infer_core/target/release/io_bench" \
  --model "${MODEL}" \
  --rounds "${ROUNDS}" \
  --active-experts "${ACTIVE_EXPERTS}" \
  --compressed-buffer-mb "${COMPRESSED_BUFFER_MB}" \
  --runtime-buffer-mb "${RUNTIME_BUFFER_MB}" \
  --io-buffer-count "${IO_BUFFER_COUNT}" \
  --pipeline-depth "${PIPELINE_DEPTH}" \
  --print-plan-layers "${PRINT_PLAN_LAYERS}" | tee "logs/zc_core_io_bench.$(date -u +%Y%m%dT%H%M%SZ).log"
