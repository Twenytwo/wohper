#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${MODEL:-${ROOT}/projects/MODEL.fake.2g.bin}"
SOCKET="${SOCKET:-/tmp/wohper-infer.sock}"
ACTIVE_EXPERTS="${ACTIVE_EXPERTS:-2}"
PIPELINE_DEPTH="${PIPELINE_DEPTH:-4}"

SERVER="${ROOT}/engine/zc_infer_core/target/release/zc_infer_server"

if [ ! -x "${SERVER}" ]; then
  cargo build --release --manifest-path "${ROOT}/engine/zc_infer_core/Cargo.toml" --bin zc_infer_server
fi

"${SERVER}" \
  --model "${MODEL}" \
  --socket "${SOCKET}" \
  --active-experts "${ACTIVE_EXPERTS}" \
  --pipeline-depth "${PIPELINE_DEPTH}" &
SERVER_PID=$!

cleanup() {
  kill "${SERVER_PID}" >/dev/null 2>&1 || true
  wait "${SERVER_PID}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

for _ in $(seq 1 50); do
  if [ -S "${SOCKET}" ]; then
    break
  fi
  sleep 0.1
done

python3 - "$SOCKET" <<'PY'
import json
import socket
import sys

socket_path = sys.argv[1]
payload = {
    "request_id": "smoke-001",
    "objective": "Plan first Wohper offline inference loop",
    "compact_context": "cave: build app. constraints: no SSD writes.",
    "constraints": ["read_only_weights", "short_context"],
    "tools_allowed": ["scheduler", "direct_io"],
    "max_new_tokens": 128,
    "temperature": 0.2,
    "route_hint": {
        "warm_layers": [0, 1, 2, 3],
        "expert_ids": [1, 4],
    },
}

client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
client.connect(socket_path)
client.sendall(json.dumps(payload).encode("utf-8") + b"\n")
chunks = []
while True:
    chunk = client.recv(65536)
    if not chunk:
        break
    chunks.append(chunk)
client.close()
print(b"".join(chunks).decode("utf-8").strip())
PY
