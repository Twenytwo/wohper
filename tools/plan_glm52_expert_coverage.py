#!/usr/bin/env python3
"""Plan targeted GLM-5.2 expert materialization from router probe logs."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


PROBE_RE = re.compile(
    r"router_probe\s+layer=(?P<layer>\d+)"
    r"(?:\s+manifest_layer=(?P<manifest_layer>\d+))?"
    r"\s+top_all=\[(?P<experts>[^\]]*)\]"
)


def parse_int_set(value: str | None) -> list[int]:
    if not value:
        return []
    result: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if end < start:
                raise argparse.ArgumentTypeError(f"invalid range {part}")
            result.update(range(start, end + 1))
        else:
            result.add(int(part))
    return sorted(result)


def parse_probe(path: Path, layer_offset: int) -> dict[int, list[list[int]]]:
    by_layer: dict[int, list[list[int]]] = defaultdict(list)
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = PROBE_RE.search(line)
        if not match:
            continue
        layer = int(match.group("layer"))
        if match.group("manifest_layer") is None:
            layer += layer_offset
        experts = [
            int(item.strip())
            for item in match.group("experts").split(",")
            if item.strip()
        ]
        if experts:
            by_layer[layer].append(experts)
    return dict(sorted(by_layer.items()))


def choose_experts(
    rows: list[list[int]],
    *,
    top_per_layer: int,
    probe_rank_limit: int,
    include_experts: list[int],
) -> tuple[list[int], list[tuple[int, int]]]:
    hits: Counter[int] = Counter()
    for row in rows:
        for expert_id in row[:probe_rank_limit]:
            hits[expert_id] += 1

    selected: list[int] = []
    for expert_id in include_experts:
        if expert_id not in selected:
            selected.append(expert_id)

    for expert_id, _count in sorted(hits.items(), key=lambda item: (-item[1], item[0])):
        if len(selected) >= top_per_layer:
            break
        if expert_id not in selected:
            selected.append(expert_id)

    return sorted(selected), sorted(hits.items(), key=lambda item: (-item[1], item[0]))


def build_plan(args: argparse.Namespace) -> dict:
    probe = parse_probe(args.probe_log, args.layer_offset)
    if not probe:
        raise SystemExit(f"no router_probe rows found in {args.probe_log}")

    include_experts = parse_int_set(args.include_experts)
    layers: dict[str, list[int]] = {}
    layer_stats: dict[str, dict] = {}
    unique_experts: set[int] = set()
    total_slots = 0
    top2_total = 0
    top2_included = 0

    for layer_id, rows in probe.items():
        selected, hits = choose_experts(
            rows,
            top_per_layer=args.top_per_layer,
            probe_rank_limit=args.probe_rank_limit,
            include_experts=include_experts,
        )
        layers[str(layer_id)] = selected
        unique_experts.update(selected)
        total_slots += len(selected)
        for row in rows:
            top2 = row[:2]
            top2_total += len(top2)
            top2_included += sum(1 for expert_id in top2 if expert_id in selected)
        layer_stats[str(layer_id)] = {
            "probe_rows": len(rows),
            "selected": selected,
            "top_hits": hits[: args.stats_limit],
        }

    expert_bytes = total_slots * args.expert_block_bytes
    dense_bytes = len(layers) * args.dense_block_bytes_estimate
    estimated_total_bytes = expert_bytes + dense_bytes
    coverage = top2_included / top2_total if top2_total else 0.0

    return {
        "format": "wohper-glm52-expert-coverage-plan",
        "version": 1,
        "source_probe_log": str(args.probe_log),
        "policy": {
            "top_per_layer": args.top_per_layer,
            "probe_rank_limit": args.probe_rank_limit,
            "include_experts": include_experts,
            "layer_offset_for_legacy_logs": args.layer_offset,
        },
        "estimates": {
            "layers": len(layers),
            "expert_slots": total_slots,
            "unique_experts": len(unique_experts),
            "expert_block_bytes": args.expert_block_bytes,
            "dense_block_bytes_estimate": args.dense_block_bytes_estimate,
            "expert_bytes": expert_bytes,
            "dense_bytes_estimate": dense_bytes,
            "total_bytes_estimate": estimated_total_bytes,
            "expert_gib": expert_bytes / 1024**3,
            "dense_gib_estimate": dense_bytes / 1024**3,
            "total_gib_estimate": estimated_total_bytes / 1024**3,
            "top2_covered_by_plan": top2_included,
            "top2_total": top2_total,
            "top2_coverage_ratio": coverage,
        },
        "layers": layers,
        "layer_stats": layer_stats,
    }


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan targeted GLM-5.2 expert coverage from router_probe logs")
    parser.add_argument("--probe-log", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--top-per-layer", type=int, default=8)
    parser.add_argument("--probe-rank-limit", type=int, default=8)
    parser.add_argument(
        "--include-experts",
        default="0-3",
        help="Comma/range list always included per layer, e.g. 0-3 or empty string.",
    )
    parser.add_argument(
        "--layer-offset",
        type=int,
        default=0,
        help="Offset for legacy logs that printed manifest layer instead of physical layer.",
    )
    parser.add_argument("--expert-block-bytes", type=int, default=20_971_520)
    parser.add_argument("--dense-block-bytes-estimate", type=int, default=104_857_600)
    parser.add_argument("--stats-limit", type=int, default=12)
    return parser.parse_args(list(argv))


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv or [])
    if args.top_per_layer <= 0:
        raise SystemExit("--top-per-layer must be positive")
    if args.probe_rank_limit <= 0:
        raise SystemExit("--probe-rank-limit must be positive")
    plan = build_plan(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    estimates = plan["estimates"]
    print(f"layers={estimates['layers']}")
    print(f"expert_slots={estimates['expert_slots']}")
    print(f"unique_experts={estimates['unique_experts']}")
    print(f"top2_coverage={estimates['top2_coverage_ratio']:.3f}")
    print(f"expert_gib={estimates['expert_gib']:.2f}")
    print(f"total_gib_estimate={estimates['total_gib_estimate']:.2f}")
    print(f"out={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
