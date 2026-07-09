#!/usr/bin/env bash
set -euo pipefail

cd /home/deploy/hermes-agent/loop_workspace

model_dir="${1:-models/zai-org/GLM-5.2}"
out="${2:-state/glm52_reference_trace_short_prompt_2026-07-03.json}"
prompt="${3:-Ciao}"

PYTHONDONTWRITEBYTECODE=1 .venv-wohper-uv/bin/python tools/glm52_reference_trace.py \
  --model-dir "$model_dir" \
  --out "$out" \
  --prompt "$prompt" \
  --numeric \
  --local-files-only

.venv-wohper-uv/bin/python - <<PY
import json
from pathlib import Path
path = Path("$out")
data = json.loads(path.read_text(encoding="utf-8"))
print("REFERENCE_TRACE_SUMMARY")
print("model_type=", data.get("model_type"))
print("token_count=", len(data.get("input_ids", [])))
print("input_ids=", data.get("input_ids"))
print("numeric_status=", (data.get("numeric_trace") or {}).get("status"))
print("numeric_error=", (data.get("numeric_trace") or {}).get("error_type"))
PY
