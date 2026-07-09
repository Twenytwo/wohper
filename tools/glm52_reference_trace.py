#!/usr/bin/env python3
"""Generate a GLM-5.2 reference trace for a short prompt when full weights exist.

The tokenizer/template portion works with the local metadata-only cache. Numeric
hidden-state traces require the complete Hugging Face checkpoint to be available
under --model-dir.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GLM-5.2 tokenizer + optional HF reference trace")
    parser.add_argument("--model-dir", type=Path, default=Path("models/zai-org/GLM-5.2"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prompt", default="Ciao")
    parser.add_argument("--system", default="Sei GLM-5.2 locale.")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--numeric", action="store_true", help="Attempt to load full model and save hidden-state summaries.")
    parser.add_argument("--local-files-only", action="store_true", default=True)
    return parser.parse_args()


def tensor_summary(tensor: Any) -> dict[str, Any]:
    import torch

    data = tensor.detach().float().cpu()
    flat = data.reshape(-1)
    sample = flat[: min(16, flat.numel())].tolist()
    return {
        "shape": list(data.shape),
        "mean": float(data.mean().item()) if flat.numel() else 0.0,
        "std": float(data.std(unbiased=False).item()) if flat.numel() else 0.0,
        "min": float(data.min().item()) if flat.numel() else 0.0,
        "max": float(data.max().item()) if flat.numel() else 0.0,
        "sample": sample,
        "dtype": str(tensor.dtype),
    }


def main() -> int:
    args = parse_args()
    from transformers import AutoConfig, AutoTokenizer

    config = AutoConfig.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )

    messages = [
        {"role": "system", "content": args.system},
        {"role": "user", "content": args.prompt},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        rendered = f"{args.system}\n\nUser: {args.prompt}\nAssistant:"
    encoded = tokenizer(rendered, return_tensors="pt")
    input_ids = encoded["input_ids"][0].tolist()
    if len(input_ids) > args.max_tokens:
        input_ids = input_ids[: args.max_tokens]
        encoded["input_ids"] = encoded["input_ids"][:, : args.max_tokens]
        if "attention_mask" in encoded:
            encoded["attention_mask"] = encoded["attention_mask"][:, : args.max_tokens]

    payload: dict[str, Any] = {
        "model_dir": str(args.model_dir),
        "model_type": getattr(config, "model_type", None),
        "architectures": getattr(config, "architectures", None),
        "num_hidden_layers": getattr(config, "num_hidden_layers", None),
        "hidden_size": getattr(config, "hidden_size", None),
        "vocab_size": getattr(config, "vocab_size", None),
        "tokenizer_len": len(tokenizer),
        "special_tokens_map": tokenizer.special_tokens_map,
        "messages": messages,
        "rendered_prompt": rendered,
        "input_ids": input_ids,
        "numeric_trace": None,
    }

    if args.numeric:
        try:
            import torch
            from transformers import AutoModelForCausalLM

            model = AutoModelForCausalLM.from_pretrained(
                args.model_dir,
                trust_remote_code=True,
                local_files_only=args.local_files_only,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
            )
            model.eval()
            with torch.no_grad():
                outputs = model(
                    **encoded,
                    output_hidden_states=True,
                    use_cache=False,
                    return_dict=True,
                )
            payload["numeric_trace"] = {
                "status": "ok",
                "hidden_states": [
                    {"index": index, **tensor_summary(hidden)}
                    for index, hidden in enumerate(outputs.hidden_states or [])
                ],
                "logits": tensor_summary(outputs.logits),
            }
        except Exception as exc:  # noqa: BLE001 - trace artifact must capture exact failure.
            payload["numeric_trace"] = {
                "status": "unavailable",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "reason": "full Hugging Face checkpoint weights are not available locally",
            }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"trace_out={args.out}")
    print(f"token_count={len(input_ids)}")
    print(f"numeric_status={(payload['numeric_trace'] or {}).get('status', 'not_requested')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
