#!/usr/bin/env python3
"""Bounded multi-token chat smoke using the DeepSeek multi-layer math path."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from deepseek_v4_bounded_chat_smoke import (
    load_messages,
    materialize_missing_experts,
    sample_token,
    write_json,
)
from deepseek_v4_tokenizer_smoke import robust_streaming_decode_report
from render_deepseek_v4_prompt import render_messages
from tokenizer_chat_smoke import TokenizerSmoke


DEFAULT_MODEL_DIR = Path("models/deepseek-ai/DeepSeek-V4-Flash")
DEFAULT_INDEX = Path("models/wohper/DeepSeek-V4-Flash.RAW.L0-L2-SPLIT-GLOBAL-CATALOG22/dense_core.tensor_index.json")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_command(
    cmd: list[str],
    timeout_seconds: float | None = None,
    heartbeat_seconds: float = 0.0,
    heartbeat_label: str = "command",
) -> subprocess.CompletedProcess[str]:
    if heartbeat_seconds <= 0:
        return subprocess.run(
            cmd,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=timeout_seconds,
        )
    started = time.perf_counter()
    process = subprocess.Popen(
        cmd,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    while True:
        try:
            stdout, stderr = process.communicate(timeout=heartbeat_seconds)
            return subprocess.CompletedProcess(cmd, process.returncode, stdout, stderr)
        except subprocess.TimeoutExpired:
            elapsed = time.perf_counter() - started
            if timeout_seconds is not None and elapsed >= timeout_seconds:
                process.kill()
                stdout, stderr = process.communicate()
                raise subprocess.TimeoutExpired(cmd, timeout_seconds, output=stdout, stderr=stderr)
            print(
                json.dumps(
                    {
                        "event": "heartbeat",
                        "label": heartbeat_label,
                        "elapsed_seconds": elapsed,
                    },
                    sort_keys=True,
                ),
                file=sys.stderr,
                flush=True,
            )


def run_multilayer_step(
    token_id: int,
    context_token_ids: list[int],
    index: Path,
    layer_count: int,
    scan_vocab: int,
    top_k: int,
    extra_token_ids: list[int],
    lmhead_chunk_rows: int,
    timeout_seconds: float,
    step_out: Path,
    heartbeat_seconds: float,
) -> dict[str, Any]:
    cmd = [
        sys.executable,
        "tools/deepseek_v4_multilayer_math_smoke.py",
        "--index",
        str(index),
        "--token-id",
        str(token_id),
        "--context-token-ids",
        ",".join(str(item) for item in context_token_ids),
        "--layer-count",
        str(layer_count),
        "--scan-vocab",
        str(scan_vocab),
        "--top-k",
        str(top_k),
        "--extra-token-ids",
        ",".join(str(item) for item in sorted(set(extra_token_ids))),
        "--lmhead-chunk-rows",
        str(lmhead_chunk_rows),
        "--skip-lmhead-when-blocked",
        "--compact-output",
        "--out",
        str(step_out),
    ]
    try:
        result = run_command(
            cmd,
            timeout_seconds=timeout_seconds,
            heartbeat_seconds=heartbeat_seconds,
            heartbeat_label=f"multilayer_step:{step_out.name}",
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "status": "blocked",
            "blockers": ["multilayer_step_timeout"],
            "timeout_seconds": timeout_seconds,
            "stdout_tail": (exc.stdout or "")[-4000:] if isinstance(exc.stdout, str) else "",
            "stderr_tail": (exc.stderr or "")[-4000:] if isinstance(exc.stderr, str) else "",
        }
    if result.returncode not in {0, 3}:
        raise RuntimeError(result.stderr or result.stdout)
    return load_json(step_out)


def missing_by_layer(payload: dict[str, Any]) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}
    for layer in payload.get("layers", []):
        missing = [int(v) for v in layer.get("missing_experts", [])]
        if missing:
            out[int(layer["layer_id"])] = missing
    return out


def materialize_for_index(
    index: Path,
    layer_id: int,
    expert_ids: list[int],
    out: Path,
    heartbeat_seconds: float,
) -> dict[str, Any]:
    if not expert_ids:
        return {"status": "skipped", "layer_id": layer_id, "expert_ids": []}
    index_payload = load_json(index)
    core = Path(index_payload["core_file"])
    if not core.exists():
        core = index.parent / index_payload["core_file"]
    cmd = [
        sys.executable,
        "tools/materialize_deepseek_v4_expert_shards.py",
        "--core",
        str(core),
        "--layer-id",
        str(layer_id),
        "--expert-ids",
        ",".join(str(item) for item in sorted(set(expert_ids))),
        "--execute",
        "--out",
        str(out),
    ]
    result = run_command(
        cmd,
        heartbeat_seconds=heartbeat_seconds,
        heartbeat_label=f"materialize_layer:{layer_id}",
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr or result.stdout)
    return load_json(out)


def compact_materialization(payload: dict[str, Any]) -> dict[str, Any]:
    written = payload.get("written", [])
    return {
        "status": payload.get("status"),
        "blockers": payload.get("blockers", []),
        "layer_id": payload.get("layer_id"),
        "requested_expert_ids": payload.get("requested_expert_ids", []),
        "to_write_expert_ids": payload.get("to_write_expert_ids", []),
        "written_count": len(written) if isinstance(written, list) else 0,
        "required_output_bytes": payload.get("required_output_bytes"),
        "free_before_bytes": payload.get("free_before_bytes"),
        "free_after_bytes": payload.get("free_after_bytes"),
        "min_free_after_bytes": payload.get("min_free_after_bytes"),
        "catalog_expert_count": payload.get("catalog_expert_count"),
        "out": payload.get("out"),
    }


def filtered_candidates(candidates: list[dict[str, Any]], suppress: set[int]) -> list[dict[str, Any]]:
    kept = [item for item in candidates if int(item["token_id"]) not in suppress]
    return kept or candidates


def unicode_safe_candidates(
    tokenizer: TokenizerSmoke,
    prefix_ids: list[int],
    candidates: list[dict[str, Any]],
    allow_incomplete: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if allow_incomplete:
        return candidates, {"enabled": False, "dropped_token_ids": []}
    kept = []
    dropped = []
    for item in candidates:
        token_id = int(item["token_id"])
        report = robust_streaming_decode_report(tokenizer, prefix_ids + [token_id])
        blockers = {str(blocker) for blocker in report.get("blockers", [])}
        if blockers or not report.get("unicode_ok", False):
            dropped.append({"token_id": token_id, "blockers": sorted(blockers)})
            continue
        kept.append(item)
    return kept or candidates, {
        "enabled": True,
        "dropped_token_ids": [int(item["token_id"]) for item in dropped],
        "dropped": dropped[:16],
        "fallback_used": not kept and bool(candidates),
    }


def summarize_id_set(values: set[int]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "min": None, "max": None, "sample": []}
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "max": ordered[-1],
        "sample": ordered[:16],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DeepSeek bounded multi-layer chat smoke")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--messages", type=Path, default=Path("config/deepseek_v4_flash.prompt_smoke.json"))
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--layer-count", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=2)
    parser.add_argument("--scan-vocab", type=int, default=8192)
    parser.add_argument("--lmhead-chunk-rows", type=int, default=1024)
    parser.add_argument(
        "--extra-token-texts",
        default=" pronto|pronto|Pronto|Sono|sono|Si| si| ok|OK|.|,|Ciao",
        help="Pipe-separated texts whose token ids are scored in addition to the bounded vocab scan.",
    )
    parser.add_argument("--candidate-top-k", type=int, default=32)
    parser.add_argument("--sample-top-k", type=int, default=8)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.15)
    parser.add_argument(
        "--allow-incomplete-utf8-candidates",
        action="store_true",
        help="Debug mode: allow candidate tokens that leave streaming decode with pending invalid UTF-8 bytes.",
    )
    parser.add_argument("--max-materialize-rounds", type=int, default=4)
    parser.add_argument("--step-timeout-seconds", type=float, default=600.0)
    parser.add_argument("--heartbeat-seconds", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("state/deepseek_v4_bounded_chat_multilayer_smoke_2026-07-05.json"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tokenizer = TokenizerSmoke.load(args.model_dir)
    messages = load_messages(args.messages)
    prompt = render_messages(messages)
    prompt_ids = tokenizer.encode(prompt)
    extra_token_ids = sorted(
        {
            int(token_id)
            for text in args.extra_token_texts.split("|")
            if text
            for token_id in tokenizer.encode(text)
        }
    )
    special_ids = {int(value) for value in tokenizer.specials.values()}
    stop_token_ids = {
        tokenizer.specials["<｜end▁of▁sentence｜>"],
        tokenizer.specials["<|EOT|>"],
    }
    suppress_ids = set(special_ids)
    generated: list[int] = []
    steps = []
    materializations = []
    blockers = []
    stop_reason = "max_new_tokens"
    current_token = prompt_ids[-1]

    import random

    rng = random.Random(args.seed)
    for step in range(args.max_new_tokens):
        step_out = args.out.parent / f"{args.out.stem}.step{step}.json"
        payload: dict[str, Any] | None = None
        for round_id in range(args.max_materialize_rounds + 1):
            payload = run_multilayer_step(
                current_token,
                prompt_ids + generated,
                args.index,
                args.layer_count,
                args.scan_vocab,
                args.candidate_top_k,
                extra_token_ids,
                args.lmhead_chunk_rows,
                args.step_timeout_seconds,
                step_out,
                args.heartbeat_seconds,
            )
            missing = missing_by_layer(payload)
            if payload.get("status") == "ready" or not missing:
                break
            if round_id >= args.max_materialize_rounds:
                break
            for layer_id, expert_ids in missing.items():
                mat_out = args.out.parent / f"{args.out.stem}.materialize_l{layer_id}_step{step}_round{round_id}.json"
                materializations.append(
                    compact_materialization(
                        materialize_for_index(args.index, layer_id, expert_ids, mat_out, args.heartbeat_seconds)
                    )
                )
        assert payload is not None
        if payload.get("status") != "ready":
            blockers.extend(payload.get("blockers", ["multilayer_step_blocked"]))
            stop_reason = "blocked"
            break
        candidates = filtered_candidates(payload["bounded_lmhead_topk"], suppress_ids)
        candidates, unicode_filter = unicode_safe_candidates(
            tokenizer,
            generated,
            candidates,
            args.allow_incomplete_utf8_candidates,
        )
        chosen = sample_token(
            candidates,
            previous=prompt_ids + generated,
            temperature=args.temperature,
            top_k=args.sample_top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            rng=rng,
        )
        next_token = int(chosen["token_id"])
        generated.append(next_token)
        steps.append(
            {
                "step": step,
                "input_token_id": current_token,
                "chosen": chosen,
                "raw_top_token_id": int(payload["bounded_lmhead_topk"][0]["token_id"]),
                "unicode_filter": unicode_filter,
                "layer_count": args.layer_count,
                "elapsed_seconds": payload.get("elapsed_seconds"),
                "bytes_read_upper_bound": payload.get("bytes_read_upper_bound"),
                "semantic_warnings": payload.get("semantic_warnings", []),
            }
        )
        current_token = next_token
        if next_token in stop_token_ids:
            stop_reason = "stop_token"
            break

    streaming = robust_streaming_decode_report(tokenizer, generated)
    if not streaming["stable"]:
        blockers.append("streaming_unstable_delta")
    if not streaming["unicode_ok"]:
        blockers.append("streaming_unicode_replacement")
    blockers.extend(str(item) for item in streaming.get("blockers", []))
    if any(token in suppress_ids for token in generated):
        blockers.append("generated_suppressed_special")
    payload = {
        "format": "deepseek-v4-bounded-chat-multilayer-smoke",
        "version": 1,
        "status": "ready" if not blockers else "blocked",
        "blockers": sorted(set(blockers)),
        "semantic_warnings": [
            "bounded multi-layer smoke; not full-depth yet",
            "attention uses sliding-window plus compressor prefill; ratio-4 long-context indexer top-k is bounded in the validation path",
            "LM-head candidates are bounded by scan_vocab plus explicit extra token ids",
        ],
        "model_dir": str(args.model_dir),
        "index": str(args.index),
        "layer_count": args.layer_count,
        "messages": messages,
        "prompt_token_count": len(prompt_ids),
        "prompt_tail_token_id": prompt_ids[-1],
        "extra_token_ids": extra_token_ids,
        "extra_token_texts": args.extra_token_texts,
        "lmhead_chunk_rows": args.lmhead_chunk_rows,
        "generated_token_ids": generated,
        "generated_text": streaming["text"],
        "streaming": streaming,
        "stop_reason": stop_reason,
        "sampling": {
            "temperature": args.temperature,
            "sample_top_k": args.sample_top_k,
            "candidate_top_k": args.candidate_top_k,
            "scan_vocab": args.scan_vocab,
            "top_p": args.top_p,
            "repetition_penalty": args.repetition_penalty,
            "allow_incomplete_utf8_candidates": args.allow_incomplete_utf8_candidates,
            "max_materialize_rounds": args.max_materialize_rounds,
            "step_timeout_seconds": args.step_timeout_seconds,
            "heartbeat_seconds": args.heartbeat_seconds,
            "seed": args.seed,
            "suppress_special_ids": summarize_id_set(suppress_ids),
        },
        "steps": steps,
        "materializations": materializations,
    }
    write_json(args.out, payload)
    print(json.dumps({k: v for k, v in payload.items() if k not in {"steps", "streaming"}}, ensure_ascii=True, indent=2))
    return 0 if not blockers else 3


if __name__ == "__main__":
    raise SystemExit(main())
