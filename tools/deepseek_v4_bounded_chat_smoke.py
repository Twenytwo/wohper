#!/usr/bin/env python3
"""Bounded multi-token DeepSeek-V4 chat smoke over the L0 math path."""

from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

from deepseek_v4_tokenizer_smoke import robust_streaming_decode_report
from render_deepseek_v4_prompt import render_messages
from tokenizer_chat_smoke import TokenizerSmoke


DEFAULT_MODEL_DIR = Path("models/deepseek-ai/DeepSeek-V4-Flash")
DEFAULT_INDEX = Path("models/wohper/DeepSeek-V4-Flash.RAW.L0-SPLIT-GLOBAL-TOP6/dense_core.tensor_index.json")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def load_messages(path: Path) -> list[dict[str, Any]]:
    payload = load_json(path)
    messages = payload.get("messages") if isinstance(payload, dict) else payload
    if not isinstance(messages, list):
        raise ValueError("messages payload must be a list or an object with messages")
    return messages


def run_command(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, text=True, encoding="utf-8", errors="replace", capture_output=True)


def run_l0_step(
    token_id: int,
    index: Path,
    scan_vocab: int,
    top_k: int,
    step_out: Path,
) -> dict[str, Any]:
    cmd = [
        sys.executable,
        "tools/deepseek_v4_single_token_l0_math_smoke.py",
        "--index",
        str(index),
        "--token-id",
        str(token_id),
        "--scan-vocab",
        str(scan_vocab),
        "--top-k",
        str(top_k),
        "--out",
        str(step_out),
    ]
    result = run_command(cmd)
    if result.returncode not in {0, 3}:
        raise RuntimeError(result.stderr or result.stdout)
    return load_json(step_out)


def materialize_missing_experts(expert_ids: list[int], out: Path) -> dict[str, Any]:
    if not expert_ids:
        return {"status": "skipped", "expert_ids": []}
    cmd = [
        sys.executable,
        "tools/materialize_deepseek_v4_expert_shards.py",
        "--expert-ids",
        ",".join(str(item) for item in sorted(set(expert_ids))),
        "--execute",
        "--out",
        str(out),
    ]
    result = run_command(cmd)
    if result.returncode != 0:
        raise RuntimeError(result.stderr or result.stdout)
    return load_json(out)


def apply_repetition_penalty(candidates: list[dict[str, Any]], previous: list[int], penalty: float) -> list[dict[str, Any]]:
    if penalty == 1.0:
        return candidates
    adjusted = []
    seen = set(previous)
    for item in candidates:
        score = float(item["score"])
        if int(item["token_id"]) in seen:
            score = score / penalty if score > 0 else score * penalty
        adjusted.append({**item, "adjusted_score": score})
    return adjusted


def sample_token(
    candidates: list[dict[str, Any]],
    *,
    previous: list[int],
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    rng: random.Random,
) -> dict[str, Any]:
    adjusted = apply_repetition_penalty(candidates[:top_k], previous, repetition_penalty)
    if not adjusted:
        raise ValueError("empty candidates")
    if temperature <= 0.0 or len(adjusted) == 1:
        winner = max(adjusted, key=lambda item: float(item.get("adjusted_score", item["score"])))
        return {**winner, "sample_probability": 1.0}
    scores = [float(item.get("adjusted_score", item["score"])) for item in adjusted]
    max_score = max(scores)
    weights = [math.exp((score - max_score) / max(temperature, 1.0e-6)) for score in scores]
    total = sum(weights)
    probs = [weight / total for weight in weights]
    ranked = sorted(zip(adjusted, probs), key=lambda item: item[1], reverse=True)
    if top_p < 1.0:
        kept = []
        cumulative = 0.0
        for item, prob in ranked:
            kept.append((item, prob))
            cumulative += prob
            if cumulative >= top_p:
                break
        norm = sum(prob for _, prob in kept)
        ranked = [(item, prob / norm) for item, prob in kept]
    draw = rng.random()
    cumulative = 0.0
    for item, prob in ranked:
        cumulative += prob
        if draw <= cumulative:
            return {**item, "sample_probability": prob}
    item, prob = ranked[-1]
    return {**item, "sample_probability": prob}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DeepSeek bounded multi-token chat smoke")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--messages", type=Path, default=Path("config/deepseek_v4_flash.prompt_smoke.json"))
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--max-new-tokens", type=int, default=2)
    parser.add_argument("--scan-vocab", type=int, default=256)
    parser.add_argument("--candidate-top-k", type=int, default=16)
    parser.add_argument("--sample-top-k", type=int, default=8)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("state/deepseek_v4_bounded_chat_smoke_2026-07-05.json"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tokenizer = TokenizerSmoke.load(args.model_dir)
    messages = load_messages(args.messages)
    prompt = render_messages(messages)
    prompt_ids = tokenizer.encode(prompt)
    stop_token_ids = {
        tokenizer.specials["<｜end▁of▁sentence｜>"],
        tokenizer.specials["<|EOT|>"],
    }
    rng = random.Random(args.seed)
    generated: list[int] = []
    steps = []
    materializations = []
    current_token = prompt_ids[-1]
    blockers = []
    stop_reason = "max_new_tokens"

    for step in range(args.max_new_tokens):
        step_out = args.out.parent / f"{args.out.stem}.step{step}.json"
        payload = run_l0_step(current_token, args.index, args.scan_vocab, args.candidate_top_k, step_out)
        if payload.get("status") != "ready" and payload.get("missing_experts"):
            mat_out = args.out.parent / f"{args.out.stem}.materialize_step{step}.json"
            materializations.append(materialize_missing_experts([int(v) for v in payload["missing_experts"]], mat_out))
            payload = run_l0_step(current_token, args.index, args.scan_vocab, args.candidate_top_k, step_out)
        if payload.get("status") != "ready":
            blockers.extend(payload.get("blockers", ["l0_step_blocked"]))
            stop_reason = "blocked"
            break
        chosen = sample_token(
            payload["bounded_lmhead_topk"],
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
                "routes": payload.get("routes", []),
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
    payload = {
        "format": "deepseek-v4-bounded-chat-smoke",
        "version": 1,
        "status": "ready" if not blockers else "blocked",
        "blockers": sorted(set(blockers)),
        "semantic_warnings": [
            "bounded L0-only smoke; not full-depth quality",
            "attention is the current single-token projection surrogate",
            "LM-head candidates are bounded by scan_vocab",
        ],
        "model_dir": str(args.model_dir),
        "index": str(args.index),
        "messages": messages,
        "prompt_token_count": len(prompt_ids),
        "prompt_tail_token_id": prompt_ids[-1],
        "generated_token_ids": generated,
        "generated_text": streaming["text"],
        "streaming": streaming,
        "stop_reason": stop_reason,
        "sampling": {
            "temperature": args.temperature,
            "sample_top_k": args.sample_top_k,
            "candidate_top_k": args.candidate_top_k,
            "top_p": args.top_p,
            "repetition_penalty": args.repetition_penalty,
            "seed": args.seed,
        },
        "steps": steps,
        "materializations": materializations,
    }
    write_json(args.out, payload)
    print(json.dumps({k: v for k, v in payload.items() if k not in {"steps", "streaming"}}, ensure_ascii=False, indent=2))
    return 0 if not blockers else 3


if __name__ == "__main__":
    raise SystemExit(main())
