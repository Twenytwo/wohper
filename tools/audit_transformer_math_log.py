#!/usr/bin/env python3
"""Audit Wohper transformer math fidelity signals from runtime logs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


COMPONENT_RE = re.compile(
    r"math_fidelity layer=(?P<layer>\d+) component=(?P<component>\S+) "
    r"present=(?P<present>true|false) applied=(?P<applied>true|false)"
)
INDEXER_RE = re.compile(
    r"math_fidelity layer=(?P<layer>\d+) component=indexer present=true "
    r"applied=false reason=(?P<reason>\S+) equivalent_to_full_causal=(?P<equivalent>true|false)"
)
LM_HEAD_RE = re.compile(r"sampling source=(?P<source>\S+).*rows=(?P<rows>\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit transformer math fidelity log signals")
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-lm-head-rows", type=int, default=154880)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    text = args.log.read_text(encoding="utf-8", errors="replace")
    components: dict[str, int] = {}
    failed_components = []
    for match in COMPONENT_RE.finditer(text):
        component = match.group("component")
        components[component] = components.get(component, 0) + 1
        if component == "indexer":
            continue
        if match.group("present") != "true" or match.group("applied") != "true":
            failed_components.append(match.group(0))
    indexer_reasons: dict[str, int] = {}
    indexer_not_equivalent = 0
    for match in INDEXER_RE.finditer(text):
        reason = match.group("reason")
        indexer_reasons[reason] = indexer_reasons.get(reason, 0) + 1
        if match.group("equivalent") != "true":
            indexer_not_equivalent += 1
    lm_head_rows = 0
    sampling_sources: dict[str, int] = {}
    for match in LM_HEAD_RE.finditer(text):
        source = match.group("source")
        sampling_sources[source] = sampling_sources.get(source, 0) + 1
        lm_head_rows = max(lm_head_rows, int(match.group("rows")))
    hidden_placeholder = "sampling source=hidden_state_placeholder" in text
    required_components = ["q_a_layernorm", "kv_a_layernorm", "shared_expert"]
    missing_required = [name for name in required_components if components.get(name, 0) == 0]
    passed = (
        not missing_required
        and not failed_components
        and not hidden_placeholder
        and lm_head_rows >= args.min_lm_head_rows
        and indexer_not_equivalent == 0
    )
    payload = {
        "format": "wohper-transformer-math-log-audit",
        "version": 1,
        "log": str(args.log),
        "status": "passed" if passed else "failed",
        "components": components,
        "missing_required": missing_required,
        "failed_components_sample": failed_components[:16],
        "indexer_reasons": indexer_reasons,
        "indexer_not_equivalent": indexer_not_equivalent,
        "sampling_sources": sampling_sources,
        "lm_head_rows": lm_head_rows,
        "hidden_state_placeholder": hidden_placeholder,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"MATH_AUDIT_STATUS={payload['status']}")
    print(f"LM_HEAD_ROWS={lm_head_rows}")
    print(f"MISSING_REQUIRED={','.join(missing_required)}")
    print(f"OUT={args.out}")
    return 0 if passed else 6


if __name__ == "__main__":
    raise SystemExit(main())
