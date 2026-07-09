#!/usr/bin/env python3
"""E3 fase 0: simula la quantizzazione 2-bit dei pesi expert nel reference.

Monkeypatch di base.decode_fp4_e2m1_array: i valori FP4 decodificati vengono
ri-quantizzati a 2 bit per gruppi di 32 colonne (scale = amax del gruppo,
livelli uniformi simmetrici {-1, -1/3, +1/3, +1} * amax) e dequantizzati.
Poi esegue il multilayer smoke L0-L4 sui casi parity e stampa i top-k da
confrontare coi reference originali.

Uso: py tools/deepseek_v4_2bit_quality_sim.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import deepseek_v4_single_token_l0_math_smoke as base  # noqa: E402

_original_decode = base.decode_fp4_e2m1_array


def quant_dequant_2bit(values: np.ndarray, group: int = 32) -> np.ndarray:
    """Uniform symmetric 2-bit quant-dequant along the last axis."""
    original_shape = values.shape
    flat = values.reshape(-1, original_shape[-1]).astype(np.float32, copy=False)
    cols = flat.shape[1]
    out = np.empty_like(flat)
    for c0 in range(0, cols, group):
        c1 = min(c0 + group, cols)
        block = flat[:, c0:c1]
        amax = np.maximum(np.abs(block).max(axis=1, keepdims=True), 1e-8)
        # livelli: q in {0,1,2,3} -> v = (q - 1.5) / 1.5 * amax
        q = np.clip(np.round(block / amax * 1.5 + 1.5), 0, 3)
        out[:, c0:c1] = (q - 1.5) / 1.5 * amax
    return out.reshape(original_shape)


def patched_decode(raw: np.ndarray) -> np.ndarray:
    values = _original_decode(raw)
    if values.ndim == 2 and values.shape[1] >= 32:
        return quant_dequant_2bit(values)
    return values


def main() -> int:
    base.decode_fp4_e2m1_array = patched_decode
    # il modulo multilayer importa base come attributo: patch anche li'
    import deepseek_v4_multilayer_math_smoke as smoke

    smoke.base.decode_fp4_e2m1_array = patched_decode

    index = (
        "models/wohper/DeepSeek-V4-Flash.RAW.L0-L43-SPLIT-GLOBAL-CATALOGSEED/"
        "dense_core.tensor_index.json"
    )
    cases = [
        ("single_42", "42"),
        ("single_32974", "32974"),
        ("ctx_42_32974", "42,32974"),
        ("ctx_7_1000", "7,1000"),
        ("story_ctx4", "16600,4465,260,1014"),
    ]
    reference_top1 = {
        "single_42": 32974,
        "single_32974": 82270,
        "ctx_42_32974": 107529,
        "ctx_7_1000": 69146,
        "story_ctx4": 57502,
    }
    results = {}
    for name, tokens in cases:
        out_path = f"state/e3_2bit_sim_{name}_2026-07-07.json"
        argv_backup = sys.argv
        sys.argv = [
            "sim",
            "--index", index,
            "--context-token-ids", tokens,
            "--layer-count", "5",
            "--scan-vocab", "129280",
            "--top-k", "3",
            "--compact-output",
            "--out", out_path,
        ]
        try:
            smoke.main()
        except SystemExit:
            pass
        finally:
            sys.argv = argv_backup
        data = json.load(open(out_path))
        topk = [(e["token_id"], round(e["score"], 3)) for e in data.get("bounded_lmhead_topk", [])]
        expected = reference_top1[name]
        verdict = "MATCH" if topk and topk[0][0] == expected else "DIVERGE"
        results[name] = {"top3": topk, "expected_top1": expected, "verdict": verdict}
        print(f"[2bit-sim] {name}: top3={topk} atteso={expected} -> {verdict}", flush=True)

    matches = sum(1 for r in results.values() if r["verdict"] == "MATCH")
    print(f"\n[2bit-sim] VERDETTO FINALE: {matches}/{len(cases)} argmax invariati")
    json.dump(
        {"results": results, "matches": matches, "total": len(cases)},
        open("state/e3_2bit_sim_summary_2026-07-07.json", "w"),
        indent=2,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
