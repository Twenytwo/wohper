#!/usr/bin/env python3
"""Compare GLM-5.2 numeric trace summaries with explicit tolerances.

This compares compact JSON summaries, not full tensors. It is meant for the
first parity gate: same prompt, same token IDs, layer summaries close enough.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare GLM-5.2 trace summaries")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mean-atol", type=float, default=1.0e-3)
    parser.add_argument("--std-atol", type=float, default=1.0e-3)
    parser.add_argument("--sample-atol", type=float, default=1.0e-2)
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def hidden_states(payload: dict[str, Any]) -> list[dict[str, Any]]:
    numeric = payload.get("numeric_trace") or payload
    states = numeric.get("hidden_states")
    if not isinstance(states, list):
        raise ValueError("trace does not contain numeric_trace.hidden_states")
    return states


def compare_samples(left: list[Any], right: list[Any], atol: float) -> dict[str, Any]:
    count = min(len(left), len(right))
    diffs = [abs(float(left[i]) - float(right[i])) for i in range(count)]
    max_diff = max(diffs, default=0.0)
    return {
        "count": count,
        "max_abs_diff": max_diff,
        "passed": max_diff <= atol and len(left) == len(right),
    }


def main() -> int:
    args = parse_args()
    reference = load(args.reference)
    candidate = load(args.candidate)
    ref_states = hidden_states(reference)
    cand_states = hidden_states(candidate)
    layer_count = min(len(ref_states), len(cand_states))
    layer_results = []
    passed = len(ref_states) == len(cand_states)
    for index in range(layer_count):
        ref = ref_states[index]
        cand = cand_states[index]
        mean_diff = abs(float(ref.get("mean", 0.0)) - float(cand.get("mean", 0.0)))
        std_diff = abs(float(ref.get("std", 0.0)) - float(cand.get("std", 0.0)))
        sample = compare_samples(ref.get("sample", []), cand.get("sample", []), args.sample_atol)
        layer_passed = (
            mean_diff <= args.mean_atol
            and std_diff <= args.std_atol
            and sample["passed"]
            and ref.get("shape") == cand.get("shape")
        )
        passed = passed and layer_passed
        layer_results.append(
            {
                "index": index,
                "shape_ref": ref.get("shape"),
                "shape_candidate": cand.get("shape"),
                "mean_abs_diff": mean_diff,
                "std_abs_diff": std_diff,
                "sample": sample,
                "passed": layer_passed,
            }
        )
    payload = {
        "format": "glm52-trace-summary-comparison",
        "version": 1,
        "status": "passed" if passed else "failed",
        "reference": str(args.reference),
        "candidate": str(args.candidate),
        "reference_layers": len(ref_states),
        "candidate_layers": len(cand_states),
        "tolerances": {
            "mean_atol": args.mean_atol,
            "std_atol": args.std_atol,
            "sample_atol": args.sample_atol,
        },
        "layers": layer_results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"TRACE_COMPARE_STATUS={payload['status']}")
    print(f"REFERENCE_LAYERS={len(ref_states)}")
    print(f"CANDIDATE_LAYERS={len(cand_states)}")
    print(f"OUT={args.out}")
    return 0 if passed else 5


if __name__ == "__main__":
    raise SystemExit(main())
