#!/usr/bin/env bash
set -euo pipefail

cd /home/deploy/hermes-agent/loop_workspace

mode="${1:---dry-run}"
out="${2:-state/glm52_cleanup_2026-07-04.json}"

if [ "$mode" != "--dry-run" ] && [ "$mode" != "--execute" ]; then
  echo "usage: $0 [--dry-run|--execute] [out.json]" >&2
  exit 2
fi

root="$(pwd -P)"
models_root="$root/models/wohper"
zai_glm_meta="$root/models/zai-org/GLM-5.2"

mkdir -p state logs

before_free_kb="$(df -Pk . | awk 'NR==2 {print $4}')"
before_used_kb="$(df -Pk . | awk 'NR==2 {print $3}')"

mapfile -t candidates < <(
  find "$models_root" -mindepth 1 -maxdepth 1 -type d -name 'GLM-5.2.INT4*' -print 2>/dev/null | sort
)

validated=()
for path in "${candidates[@]}"; do
  real="$(realpath "$path")"
  case "$real" in
    "$models_root"/GLM-5.2.INT4*) validated+=("$real") ;;
    *)
      echo "refusing path outside cleanup allowlist: $real" >&2
      exit 3
      ;;
  esac
done

bytes_planned=0
items_json="$(mktemp)"
: > "$items_json"
for path in "${validated[@]}"; do
  bytes="$(du -sb "$path" | awk '{print $1}')"
  bytes_planned=$((bytes_planned + bytes))
  python3 - "$items_json" "$path" "$bytes" <<'PY'
import json
import sys
from pathlib import Path

items_path = Path(sys.argv[1])
item = {"path": sys.argv[2], "bytes": int(sys.argv[3])}
with items_path.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(item) + "\n")
PY
done

deleted=()
if [ "$mode" = "--execute" ]; then
  for path in "${validated[@]}"; do
    rm -rf -- "$path"
    deleted+=("$path")
  done
fi

after_free_kb="$(df -Pk . | awk 'NR==2 {print $4}')"
after_used_kb="$(df -Pk . | awk 'NR==2 {print $3}')"

python3 - "$out" "$mode" "$before_free_kb" "$after_free_kb" "$before_used_kb" "$after_used_kb" "$bytes_planned" "$items_json" "${deleted[@]}" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
mode = sys.argv[2]
before_free_kb = int(sys.argv[3])
after_free_kb = int(sys.argv[4])
before_used_kb = int(sys.argv[5])
after_used_kb = int(sys.argv[6])
bytes_planned = int(sys.argv[7])
items_path = Path(sys.argv[8])
deleted = sys.argv[9:]
items = []
if items_path.exists():
    for line in items_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            items.append(json.loads(line))
payload = {
    "format": "glm52-cleanup",
    "version": 1,
    "mode": mode,
    "status": "executed" if mode == "--execute" else "dry_run",
    "allowlist": [
        "models/wohper/GLM-5.2.INT4*"
    ],
    "candidate_count": len(items),
    "planned_delete_bytes": bytes_planned,
    "deleted_count": len(deleted),
    "deleted_paths": deleted,
    "candidates": items,
    "disk_before": {
        "free_kb": before_free_kb,
        "used_kb": before_used_kb,
    },
    "disk_after": {
        "free_kb": after_free_kb,
        "used_kb": after_used_kb,
    },
    "freed_kb_observed": after_free_kb - before_free_kb,
    "deepseek_note": (
        "Cleanup can free GLM artifacts but this VPS filesystem is still too "
        "small for the full DeepSeek-V4-Flash checkpoint unless a larger volume "
        "is mounted."
    ),
}
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2, sort_keys=True))
PY

rm -f "$items_json"

if [ "$mode" = "--dry-run" ]; then
  echo "Dry run only. Re-run with --execute to delete allowlisted GLM artifacts." >&2
fi
