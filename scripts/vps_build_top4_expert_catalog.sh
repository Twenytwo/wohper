#!/usr/bin/env bash
set -euo pipefail

cd /home/deploy/hermes-agent/loop_workspace

expert_root="${1:-models/wohper/GLM-5.2.INT4.L3-L78-TOP4-EXPERTS-ONLY}"
out="${2:-state/zc_top4_expert_catalog_2026-07-04.json}"

PYTHONDONTWRITEBYTECODE=1 python3 tools/build_expert_catalog.py \
  --expert-root "$expert_root" \
  --out "$out" \
  --hash

python3 - "$out" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
data = json.loads(path.read_text(encoding="utf-8"))
assert data["file_count"] > 0, "empty expert catalog"
assert data["total_bytes"] > 0, "empty expert bytes"
assert all("sha256" in entry for entry in data["entries"]), "missing sha256"
print("EXPERT_CATALOG_OK")
print(f"file_count={data['file_count']}")
print(f"total_bytes={data['total_bytes']}")
PY
