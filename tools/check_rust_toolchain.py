#!/usr/bin/env python3
"""Check Rust toolchain availability for Wohper runtime tests."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any


def run_version(binary: str) -> dict[str, Any]:
    path = shutil.which(binary)
    if not path:
        return {"binary": binary, "found": False, "path": None, "version": None}
    result = subprocess.run([path, "--version"], text=True, capture_output=True)
    return {
        "binary": binary,
        "found": result.returncode == 0,
        "path": path,
        "version": (result.stdout or result.stderr).strip(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rust toolchain check")
    parser.add_argument("--crate", type=Path, default=Path("engine/zc_infer_core"))
    parser.add_argument("--run-tests", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("state/rust_toolchain_check_2026-07-05.json"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cargo = run_version("cargo")
    rustc = run_version("rustc")
    blockers = []
    if not args.crate.exists():
        blockers.append("missing_crate_dir")
    if not (args.crate / "Cargo.toml").exists():
        blockers.append("missing_cargo_toml")
    if not cargo["found"] or not rustc["found"]:
        blockers.append("blocked_missing_rust_toolchain")

    test_result = None
    if args.run_tests and not blockers:
        result = subprocess.run(
            [cargo["path"], "test", "-q"],
            cwd=args.crate,
            text=True,
            capture_output=True,
        )
        test_result = {
            "returncode": result.returncode,
            "stdout_tail": result.stdout[-4000:],
            "stderr_tail": result.stderr[-4000:],
        }
        if result.returncode != 0:
            blockers.append("cargo_test_failed")

    payload = {
        "format": "rust-toolchain-check",
        "version": 1,
        "status": "ready" if not blockers else "blocked",
        "blockers": blockers,
        "crate": str(args.crate),
        "cargo": cargo,
        "rustc": rustc,
        "run_tests": bool(args.run_tests),
        "test_result": test_result,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not blockers else 3


if __name__ == "__main__":
    raise SystemExit(main())
