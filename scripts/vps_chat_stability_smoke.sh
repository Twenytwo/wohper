#!/usr/bin/env bash
set -euo pipefail

cd /home/deploy/hermes-agent/loop_workspace

out="${1:-state/zc_chat_stability_smoke_2026-07-04.json}"
messages_b64="W3sicm9sZSI6InN5c3RlbSIsImNvbnRlbnQiOiJSaXNwb25kaSBpbiBpdGFsaWFubywgaW4gbW9kbyBjb25jaXNvLiJ9LHsicm9sZSI6InVzZXIiLCJjb250ZW50IjoiQ2lhbywgdmVyaWZpY2EgdW5pY29kZTogY2FmZSwgY2l0dGEsIGV1cm8sIG1hdGVtYXRpY2EgMisyLiJ9XQ=="

mkdir -p state logs

PYTHONDONTWRITEBYTECODE=1 python3 tools/tokenizer_chat_smoke.py \
  --model-dir models/zai-org/GLM-5.2 \
  --messages-json-b64 "$messages_b64" \
  --assistant-prefix "" \
  --generated-ids "14252" \
  --json \
  > "$out"

python3 - "$out" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
data = json.loads(path.read_text(encoding="utf-8"))
assert data["roundtrip_ok"], "chat template roundtrip failed"
assert data["streaming"]["stable"], "streaming decode is not stable"
assert data["streaming"]["unicode_ok"], "streaming decode contains replacement characters"
assert "<|system|>" in data["prompt_text"], "missing system tag"
assert "<|user|>" in data["prompt_text"], "missing user tag"
assert data["prompt_text"].endswith("<|assistant|>\n"), "missing assistant generation tag"
print("CHAT_STABILITY_SMOKE_OK")
print(f"OUT={path}")
print(f"TOKEN_COUNT={data['token_count']}")
print(f"STREAM_TEXT={data['streaming']['text']!r}")
PY
