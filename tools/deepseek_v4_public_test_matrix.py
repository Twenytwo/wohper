#!/usr/bin/env python3
"""Safe public DeepSeek-V4 test matrix.

This matrix never downloads weights, never converts payloads, and never deletes
artifacts. Runtime checks run only when the required local artifacts already
exist.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, text=True, encoding="utf-8", errors="replace", capture_output=True)


def load_status(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8-sig")).get("status")
    except Exception:
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DeepSeek-V4 public test matrix")
    parser.add_argument("--model-dir", type=Path, default=Path("models/deepseek-ai/DeepSeek-V4-Flash"))
    parser.add_argument("--core", type=Path, default=Path("models/wohper/DeepSeek-V4-Flash.RAW.L0-SPLIT-GLOBAL-TOP6/dense_core.bin"))
    parser.add_argument("--single-token", type=Path, default=Path("state/deepseek_v4_single_token_l0_math_smoke_scan256_2026-07-05.json"))
    parser.add_argument("--chat", type=Path, default=Path("state/deepseek_v4_bounded_chat_smoke_2026-07-05.json"))
    parser.add_argument("--profile-summary", type=Path)
    parser.add_argument("--max-chat-token-seconds", type=float, default=30.0)
    parser.add_argument("--out", type=Path, default=Path("state/deepseek_v4_public_test_matrix_2026-07-05.json"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    blockers: list[str] = []

    tokenizer_out = args.out.parent / "deepseek_v4_public_matrix_tokenizer_2026-07-05.json"
    if (args.model_dir / "tokenizer.json").exists():
        result = run([sys.executable, "tools/deepseek_v4_tokenizer_smoke.py", "--model-dir", str(args.model_dir), "--out", str(tokenizer_out)])
        status = load_status(tokenizer_out)
        ok = result.returncode == 0 and status == "ready"
        rows.append({"name": "deepseek_tokenizer_smoke", "status": "passed" if ok else "failed", "artifact": str(tokenizer_out)})
        if not ok:
            blockers.append("deepseek_tokenizer_smoke_failed")
    else:
        rows.append({"name": "deepseek_tokenizer_smoke", "status": "skipped_missing_artifacts", "artifact": str(tokenizer_out)})

    guardrail_out = args.out.parent / "deepseek_v4_public_matrix_perf_guardrail_2026-07-05.json"
    if args.core.exists() and args.single_token.exists() and args.chat.exists():
        cmd = [
            sys.executable,
            "tools/deepseek_v4_perf_guardrail.py",
            "--core",
            str(args.core),
            "--single-token",
            str(args.single_token),
            "--chat",
            str(args.chat),
            "--max-chat-token-seconds",
            str(args.max_chat_token_seconds),
            "--out",
            str(guardrail_out),
        ]
        if args.profile_summary:
            cmd.extend(["--profile-summary", str(args.profile_summary)])
        result = run(cmd)
        status = load_status(guardrail_out)
        ok = result.returncode == 0 and status == "ready"
        rows.append({"name": "deepseek_perf_guardrail", "status": "passed" if ok else "failed", "artifact": str(guardrail_out)})
        if not ok:
            blockers.append("deepseek_perf_guardrail_failed")
    else:
        rows.append({"name": "deepseek_perf_guardrail", "status": "skipped_missing_artifacts", "artifact": str(guardrail_out)})

    rust_out = args.out.parent / "rust_toolchain_public_matrix_2026-07-05.json"
    result = run([sys.executable, "tools/check_rust_toolchain.py", "--run-tests", "--out", str(rust_out)])
    rust_payload = json.loads(rust_out.read_text(encoding="utf-8-sig")) if rust_out.exists() else {}
    if rust_payload.get("status") == "ready":
        rust_status = "passed"
    elif "blocked_missing_rust_toolchain" in rust_payload.get("blockers", []):
        rust_status = "expected_blocked_missing_toolchain"
    else:
        rust_status = "failed"
        blockers.append("rust_toolchain_unexpected_failure")
    rows.append({"name": "rust_toolchain", "status": rust_status, "artifact": str(rust_out)})

    payload = {
        "format": "deepseek-v4-public-test-matrix",
        "version": 1,
        "status": "passed" if not blockers else "failed",
        "blockers": blockers,
        "tests": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not blockers else 3


if __name__ == "__main__":
    raise SystemExit(main())
