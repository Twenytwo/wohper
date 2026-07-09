#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

echo "Bootstrapping isolated Hermes + Wohper workspace at ${ROOT}"

mkdir -p config docs scripts skills state logs worktrees projects vendor .local/bin .local/share

if [ ! -f config/loop.config.json ]; then
  cp config/loop.config.example.json config/loop.config.json
  echo "Created config/loop.config.json from example. Edit agent commands before non-dry-run use."
fi

bash scripts/install_kimchi_isolated.sh
bash scripts/bootstrap_open_design.sh

python3 scripts/loop_runner.py \
  --config config/loop.config.json \
  --objective-file objectives/current.md \
  --dry-run

echo "Bootstrap complete. Review logs/ before spending model tokens."
