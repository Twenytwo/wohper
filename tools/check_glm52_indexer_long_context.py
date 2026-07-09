#!/usr/bin/env python3
"""Public guardrail for GLM-DSA indexer long-context readiness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


INDEX_TOPK = 2048
INDEX_HEADS = 32
INDEX_HEAD_DIM = 128
BYTES_F32 = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check GLM-DSA long-context indexer safety")
    parser.add_argument("--context-len", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.context_len < 1:
        raise SystemExit("--context-len must be positive")

    equivalent = args.context_len <= INDEX_TOPK
    # Upper bound for a naive per-layer dense score buffer. This is intentionally
    # conservative and documents why public scripts must avoid silent long runs.
    naive_score_bytes = INDEX_HEADS * args.context_len * BYTES_F32
    index_k_cache_bytes_per_layer = args.context_len * INDEX_HEAD_DIM * BYTES_F32
    payload = {
        "format": "glm52-indexer-long-context-guard",
        "version": 1,
        "context_len": args.context_len,
        "index_topk": INDEX_TOPK,
        "status": "equivalent_full_causal" if equivalent else "blocked_requires_real_indexer",
        "runtime_expected_reason": "not_required_context_le_topk" if equivalent else "bypassed_long_context",
        "runtime_expected_equivalent_to_full_causal": equivalent,
        "naive_score_buffer_bytes_per_layer": naive_score_bytes,
        "index_k_cache_bytes_per_layer": index_k_cache_bytes_per_layer,
        "public_policy": (
            "allow short-context full-causal path"
            if equivalent
            else "fail loudly; do not run long-context DSA with bypass"
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"INDEXER_STATUS={payload['status']}")
    print(f"CONTEXT_LEN={args.context_len}")
    print(f"INDEX_TOPK={INDEX_TOPK}")
    print(f"OUT={args.out}")
    return 0 if equivalent else 4


if __name__ == "__main__":
    raise SystemExit(main())
