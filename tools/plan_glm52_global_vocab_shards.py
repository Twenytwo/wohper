#!/usr/bin/env python3
"""Plan GLM-5.2 global vocab row shards without downloading model bytes."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def mib(value: int) -> float:
    return value / (1024 * 1024)


def load_config(metadata_dir: Path) -> dict[str, Any]:
    config_path = metadata_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"missing config: {config_path}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an open-repo-safe plan for GLM-5.2 embed/lm_head row shards."
    )
    parser.add_argument("--metadata-dir", type=Path, default=Path("models/zai-org/GLM-5.2"))
    parser.add_argument("--out-root", type=Path, default=Path("models/wohper"))
    parser.add_argument("--prefix", default="GLM-5.2.INT4.GLOBAL")
    parser.add_argument("--row-start", type=int, default=0)
    parser.add_argument("--row-end", type=int, default=0, help="Exclusive end row. Defaults to config vocab_size.")
    parser.add_argument("--rows-per-shard", type=int, default=32768)
    parser.add_argument("--max-shard-mib", type=float, default=256.0)
    parser.add_argument("--max-total-mib", type=float, default=1024.0)
    parser.add_argument("--emit-merge-command", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.metadata_dir)
    vocab_size = int(config.get("vocab_size") or 0)
    hidden_size = int(config.get("hidden_size") or 0)
    if vocab_size <= 0 or hidden_size <= 0:
        raise ValueError("config must contain positive vocab_size and hidden_size")
    if args.rows_per_shard <= 0:
        raise ValueError("--rows-per-shard must be positive")

    row_start = max(0, args.row_start)
    row_end = int(args.row_end or vocab_size)
    if row_end > vocab_size:
        raise ValueError(f"--row-end {row_end} exceeds config vocab_size {vocab_size}")
    if row_start >= row_end:
        raise ValueError(f"invalid row range {row_start}..{row_end}")

    shards: list[dict[str, Any]] = []
    for start in range(row_start, row_end, args.rows_per_shard):
        end = min(row_end, start + args.rows_per_shard)
        rows = end - start
        packed_bytes_per_tensor = rows * math.ceil(hidden_size / 2)
        source_bf16_bytes_per_tensor = rows * hidden_size * 2
        packed_bytes_total = packed_bytes_per_tensor * 2
        source_bf16_bytes_total = source_bf16_bytes_per_tensor * 2
        out_dir = args.out_root / f"{args.prefix}-ROWS-{start}-{end}"
        command = (
            "powershell -ExecutionPolicy Bypass -File scripts\\convert_glm52_global_micro.ps1 "
            f"-Rows {rows} -StartRow {start} -OutDir \"{out_dir.as_posix()}\""
        )
        shard = {
            "start": start,
            "end": end,
            "rows": rows,
            "out_dir": out_dir.as_posix(),
            "dense_core": (out_dir / "dense_core.bin").as_posix(),
            "packed_int4_bytes_estimate": packed_bytes_total,
            "packed_int4_mib_estimate": round(mib(packed_bytes_total), 2),
            "source_bf16_bytes_to_stream": source_bf16_bytes_total,
            "source_bf16_mib_to_stream": round(mib(source_bf16_bytes_total), 2),
            "convert_command": command,
        }
        if shard["packed_int4_mib_estimate"] > args.max_shard_mib:
            shard["risk"] = "exceeds_max_shard_mib"
        shards.append(shard)

    total_packed = sum(int(shard["packed_int4_bytes_estimate"]) for shard in shards)
    total_source = sum(int(shard["source_bf16_bytes_to_stream"]) for shard in shards)
    risks = [shard for shard in shards if shard.get("risk")]
    if mib(total_packed) > args.max_total_mib:
        risks.append({"risk": "exceeds_max_total_mib", "packed_int4_mib_estimate": round(mib(total_packed), 2)})

    payload: dict[str, Any] = {
        "metadata_dir": args.metadata_dir.as_posix(),
        "vocab_size": vocab_size,
        "hidden_size": hidden_size,
        "row_range": [row_start, row_end],
        "rows_per_shard": args.rows_per_shard,
        "tensor_set": ["model.embed_tokens.weight", "lm_head.weight"],
        "shard_count": len(shards),
        "packed_int4_bytes_estimate": total_packed,
        "packed_int4_mib_estimate": round(mib(total_packed), 2),
        "source_bf16_bytes_to_stream": total_source,
        "source_bf16_mib_to_stream": round(mib(total_source), 2),
        "max_shard_mib": args.max_shard_mib,
        "max_total_mib": args.max_total_mib,
        "risks": risks,
        "shards": shards,
    }
    if args.emit_merge_command:
        cores = " ".join(f"--global-core \"{shard['dense_core']}\"" for shard in shards)
        payload["merge_global_cores_args"] = cores

    print(json.dumps(payload, indent=2))
    return 3 if risks else 0


if __name__ == "__main__":
    raise SystemExit(main())
