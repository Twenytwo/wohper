#!/usr/bin/env python3
"""Build a checksummed catalog for Wohper expert artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build expert file catalog")
    parser.add_argument("--expert-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--hash", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    if not args.expert_root.is_dir():
        raise SystemExit(f"expert root does not exist: {args.expert_root}")
    files = sorted(path for path in args.expert_root.rglob("*") if path.is_file())
    entries = []
    total_bytes = 0
    for path in files:
        size = path.stat().st_size
        total_bytes += size
        entry = {
            "path": path.relative_to(args.expert_root).as_posix(),
            "bytes": size,
        }
        if args.hash:
            entry["sha256"] = sha256_file(path)
        entries.append(entry)
    payload = {
        "format": "wohper-expert-catalog",
        "version": 1,
        "expert_root": str(args.expert_root),
        "file_count": len(entries),
        "total_bytes": total_bytes,
        "hash": args.hash,
        "entries": entries,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"EXPERT_CATALOG_OUT={args.out}")
    print(f"FILE_COUNT={len(entries)}")
    print(f"TOTAL_BYTES={total_bytes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
