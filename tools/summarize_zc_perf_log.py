#!/usr/bin/env python3
"""Summarize Wohper server/benchmark logs without rerunning heavy inference."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


COMPUTE_RE = re.compile(
    r"compute layer=(?P<layer>\d+) compute_slot=(?P<slot>\d+) "
    r"dense_bytes=(?P<dense>\d+) expert_bytes=(?P<expert>\d+) "
    r"experts=(?P<experts>\d+) dequantized_values=(?P<dequant>\d+)"
)
SAMPLING_RE = re.compile(r"sampling source=(?P<source>\S+).*rows=(?P<rows>\d+)")
INDEXER_LONG_RE = re.compile(r"component=indexer .*equivalent_to_full_causal=false")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Wohper perf logs")
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--bench-json", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    lines = args.server_log.read_text(encoding="utf-8", errors="replace").splitlines()
    compute_events = []
    sampling_sources: dict[str, int] = {}
    lm_head_rows = 0
    long_indexer_blocks = 0
    for line in lines:
        if match := COMPUTE_RE.search(line):
            compute_events.append({key: int(value) for key, value in match.groupdict().items()})
        if match := SAMPLING_RE.search(line):
            source = match.group("source")
            sampling_sources[source] = sampling_sources.get(source, 0) + 1
            lm_head_rows = max(lm_head_rows, int(match.group("rows")))
        if INDEXER_LONG_RE.search(line):
            long_indexer_blocks += 1

    bench = None
    if args.bench_json and args.bench_json.exists():
        bench = json.loads(args.bench_json.read_text(encoding="utf-8"))

    prompt_elapsed = []
    if bench:
        for result in bench.get("results", []):
            prompt_elapsed.append(
                {
                    "id": result.get("id"),
                    "input_tokens": result.get("input_tokens"),
                    "elapsed_ms": result.get("elapsed_ms"),
                    "generated_ids": result.get("generated_ids"),
                    "token_sources": result.get("token_sources"),
                }
            )

    payload = {
        "format": "wohper-perf-log-summary",
        "version": 1,
        "server_log": str(args.server_log),
        "bench_json": str(args.bench_json) if args.bench_json else None,
        "compute_events": len(compute_events),
        "unique_layers_seen": sorted({event["layer"] for event in compute_events}),
        "max_compute_slot": max((event["slot"] for event in compute_events), default=None),
        "total_dense_bytes": sum(event["dense"] for event in compute_events),
        "total_expert_bytes": sum(event["expert"] for event in compute_events),
        "total_dequantized_values": sum(event["dequant"] for event in compute_events),
        "max_experts_per_compute": max((event["experts"] for event in compute_events), default=0),
        "sampling_sources": sampling_sources,
        "lm_head_rows": lm_head_rows,
        "long_indexer_blocks": long_indexer_blocks,
        "bench_total_elapsed_ms": bench.get("total_elapsed_ms") if bench else None,
        "prompt_elapsed": prompt_elapsed,
        "notes": [
            "This summarizes observed logs; it does not rerun inference.",
            "Large prefill cost is visible when prompt elapsed time is high relative to max_new_tokens=1.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"PERF_SUMMARY_OUT={args.out}")
    print(f"COMPUTE_EVENTS={payload['compute_events']}")
    print(f"TOTAL_DENSE_BYTES={payload['total_dense_bytes']}")
    print(f"TOTAL_EXPERT_BYTES={payload['total_expert_bytes']}")
    print(f"BENCH_TOTAL_ELAPSED_MS={payload['bench_total_elapsed_ms']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
