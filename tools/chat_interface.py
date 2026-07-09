#!/usr/bin/env python3
"""Interactive Wohper chat bridge.

Backends:

- socket: tokenize locally, send token ids to the Rust Unix socket server, and
  detokenize streamed Token events. This exercises the current Wohper I/O and
  compute path, but language quality is still placeholder until the Rust
  Transformer graph is complete.
- torch: load a local Hugging Face model offline and generate real text with
  Transformers/Torch.
- hybrid: ask the Rust server for a prefetch plan, then use Torch for the real
  Transformer math. This is the Phase 1.1 bridge: publishable now, and ready for
  future Rust events that expose tensor/block compute requests.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


GLM52_EOS_TOKEN_IDS = [154820, 154827, 154829]
DEFAULT_SOCKET = "/tmp/wohper-infer.sock"
DEFAULT_MODEL_DIR = "/mnt/nvme/models/zai-org/GLM-5.2"


def default_model_dir() -> Path:
    env_dir = os.environ.get("ZC_HF_MODEL_DIR")
    if env_dir:
        return Path(env_dir)
    nvme_dir = Path(DEFAULT_MODEL_DIR)
    if nvme_dir.exists():
        return nvme_dir
    return Path(__file__).resolve().parent.parent / "models" / "zai-org" / "GLM-5.2"


@dataclass
class DecodeState:
    token_ids: list[int]
    text: str = ""


def eprint(*args: Any) -> None:
    print(*args, file=sys.stderr)


def load_tokenizer(model_dir: Path, trust_remote_code: bool):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: transformers. Install in a local venv with "
            "`python3 -m pip install transformers sentencepiece protobuf`."
        ) from exc

    if not model_dir.exists():
        raise SystemExit(
            f"Tokenizer/model dir does not exist: {model_dir}\n"
            "Download it first, for example with `hf download zai-org/GLM-5.2 "
            "--local-dir /mnt/nvme/models/zai-org/GLM-5.2`."
        )

    return AutoTokenizer.from_pretrained(
        str(model_dir),
        local_files_only=True,
        trust_remote_code=trust_remote_code,
    )


def load_torch_model(args: argparse.Namespace):
    try:
        import torch
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise SystemExit(
            "Missing dependencies for real Torch generation. Install locally: "
            "`python3 -m pip install torch transformers accelerate safetensors`."
        ) from exc

    kwargs: dict[str, Any] = {
        "local_files_only": True,
        "trust_remote_code": args.trust_remote_code,
        "low_cpu_mem_usage": True,
    }
    if args.torch_dtype != "none":
        kwargs["torch_dtype"] = args.torch_dtype if args.torch_dtype == "auto" else getattr(torch, args.torch_dtype)
    if args.device_map:
        kwargs["device_map"] = args.device_map

    eprint(f"[wohper] loading Torch model from {args.model_dir} (offline)")
    model = AutoModelForCausalLM.from_pretrained(str(args.model_dir), **kwargs)
    model.eval()

    if not args.device_map:
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        eprint(f"[wohper] moving model to {device}")
        model.to(device)

    return model


def parse_csv_ints(value: str | None) -> list[int]:
    if not value:
        return []
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_stop_sequences(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.encode("utf-8").decode("unicode_escape") for item in value.split("|") if item]


def stream_chars(text: str, delay: float = 0.0) -> None:
    for char in text:
        print(char, end="", flush=True)
        if delay > 0:
            time.sleep(delay)


def render_prompt_text(
    tokenizer: Any,
    history: list[dict[str, str]],
    user_text: str,
    use_chat_template: bool,
) -> str:
    messages = [*history, {"role": "user", "content": user_text}]
    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception as exc:
            eprint(f"[wohper] chat template failed, using plain fallback: {exc}")

    lines: list[str] = []
    for message in messages:
        lines.append(f"{message['role']}: {message['content']}")
    lines.append("assistant:")
    return "\n".join(lines)


def encode_prompt(tokenizer: Any, prompt_text: str) -> list[int]:
    encoded = tokenizer(prompt_text, add_special_tokens=True)
    ids = encoded.get("input_ids")
    if not isinstance(ids, list) or not ids:
        raise RuntimeError("tokenizer returned no input_ids")
    return [int(token_id) for token_id in ids]


def build_envelope(
    *,
    request_id: str,
    user_text: str,
    prompt_text: str,
    token_ids: list[int],
    args: argparse.Namespace,
    include_token_ids: bool,
) -> dict[str, Any]:
    envelope: dict[str, Any] = {
        "request_id": request_id,
        "objective": user_text,
        "compact_context": prompt_text[-args.context_chars :],
        "constraints": [
            "offline_tokenizer",
            "read_only_model_weights",
            "low_ram_runtime",
            "no_secret_leakage",
        ],
        "tools_allowed": [
            "wohper_direct_io",
            "wohper_scheduler",
            "zcblk01_parser",
            "hf_torch_runtime",
        ],
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "stop_token_ids": args.eos_token_ids,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "seed": args.seed,
        "route_hint": {
            "warm_layers": parse_csv_ints(args.warm_layers),
            "expert_ids": parse_csv_ints(args.experts),
        },
        "bridge": {
            "mode": args.backend,
            "phase": "1.1",
            "tokenizer_model_dir": str(args.model_dir),
            "torch_compute": args.torch_compute,
            "protocol": "jsonl-unix-socket",
        },
    }
    if include_token_ids:
        envelope["token_ids"] = token_ids
    return envelope


def socket_events(socket_path: str, envelope: dict[str, Any], timeout: float) -> Iterable[dict[str, Any]]:
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.settimeout(timeout)
    client.connect(socket_path)
    client.sendall((json.dumps(envelope) + "\n").encode("utf-8"))

    with client.makefile("r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)
    client.close()


def print_decoded_token(
    tokenizer: Any,
    state: DecodeState,
    token_id: int,
    show_token_ids: bool,
    char_delay: float,
    stop_sequences: list[str],
) -> bool:
    if token_id in GLM52_EOS_TOKEN_IDS:
        return True

    state.token_ids.append(token_id)
    try:
        decoded = tokenizer.decode(state.token_ids, skip_special_tokens=True)
    except Exception as exc:
        decoded = state.text + f"<{token_id}>"
        eprint(f"[wohper] decode warning for token {token_id}: {exc}")
    stopped = False
    for sequence in stop_sequences:
        index = decoded.find(sequence)
        if index >= 0:
            decoded = decoded[:index]
            stopped = True
            break
    delta = decoded[len(state.text) :]
    if delta:
        stream_chars(delta, char_delay)
        state.text = decoded
    elif show_token_ids:
        print(f"<{token_id}>", end="", flush=True)
    return stopped


def run_socket_generation(
    tokenizer: Any,
    prompt_text: str,
    user_text: str,
    token_ids: list[int],
    args: argparse.Namespace,
    model: Any | None = None,
) -> str:
    request_id = f"chat-{uuid.uuid4().hex[:12]}"
    envelope = build_envelope(
        request_id=request_id,
        user_text=user_text,
        prompt_text=prompt_text,
        token_ids=token_ids,
        args=args,
        include_token_ids=True,
    )
    decode_state = DecodeState([])
    finished = False

    for event in socket_events(args.socket, envelope, args.socket_timeout):
        kind = event.get("event") or event.get("status")
        if kind == "Started":
            continue
        if kind == "Token":
            finished = print_decoded_token(
                tokenizer,
                decode_state,
                int(event["token_id"]),
                args.show_token_ids,
                args.char_delay,
                args.stop_sequences,
            )
            if finished:
                break
        elif kind == "Finished":
            finished = True
            break
        elif kind == "HybridComputeRequest":
            if model is None:
                eprint("[wohper] received HybridComputeRequest, but Torch model is not loaded")
                continue
            event_token_ids = [int(token) for token in event.get("token_ids", token_ids)]
            event_prompt = event.get("prompt_text") or tokenizer.decode(
                event_token_ids,
                skip_special_tokens=False,
            )
            eprint("[wohper] handling HybridComputeRequest with local Torch runtime")
            text = run_torch_generation(tokenizer, model, event_prompt, args)
            decode_state.text += text
        elif kind == "planned":
            eprint(f"[wohper] prefetch plan: {json.dumps(event.get('planned_prefetch', {}), ensure_ascii=False)}")
        else:
            eprint(f"[wohper] socket event: {json.dumps(event, ensure_ascii=False)}")

    print()
    if not finished:
        eprint("[wohper] socket stream ended before Finished")
    return decode_state.text


def send_prefetch_hint(
    prompt_text: str,
    user_text: str,
    token_ids: list[int],
    args: argparse.Namespace,
) -> None:
    if not args.socket:
        return
    if not Path(args.socket).exists():
        eprint(f"[wohper] socket not found, skipping Rust prefetch hint: {args.socket}")
        return

    request_id = f"prefetch-{uuid.uuid4().hex[:12]}"
    envelope = build_envelope(
        request_id=request_id,
        user_text=user_text,
        prompt_text=prompt_text,
        token_ids=token_ids,
        args=args,
        include_token_ids=False,
    )
    try:
        for event in socket_events(args.socket, envelope, args.socket_timeout):
            if event.get("status") == "planned":
                summary = event.get("planned_prefetch", {})
                eprint(f"[wohper] Rust prefetch plan accepted: {summary}")
                return
            eprint(f"[wohper] Rust sidecar event: {event}")
    except OSError as exc:
        eprint(f"[wohper] prefetch sidecar unavailable: {exc}")


def run_torch_generation(
    tokenizer: Any,
    model: Any,
    prompt_text: str,
    args: argparse.Namespace,
) -> str:
    import torch

    inputs = tokenizer(prompt_text, return_tensors="pt")
    input_len = int(inputs["input_ids"].shape[-1])

    if hasattr(model, "device"):
        inputs = {key: value.to(model.device) for key, value in inputs.items()}

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id or args.eos_token_ids[0]

    generate_kwargs: dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
        "eos_token_id": args.eos_token_ids,
        "pad_token_id": pad_token_id,
    }
    if args.temperature and args.temperature > 0:
        generate_kwargs.update(
            {
                "do_sample": True,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "repetition_penalty": args.repetition_penalty,
            }
        )
    else:
        generate_kwargs["do_sample"] = False

    with torch.inference_mode():
        output_ids = model.generate(**inputs, **generate_kwargs)[0]

    new_ids = output_ids[input_len:].detach().cpu().tolist()
    text = tokenizer.decode(new_ids, skip_special_tokens=True)
    stream_chars(text, args.char_delay)
    print()
    return text


def interactive_loop(args: argparse.Namespace) -> None:
    tokenizer = load_tokenizer(args.model_dir, args.trust_remote_code)
    use_torch = args.backend == "torch" or (args.backend == "hybrid" and args.torch_compute)
    model = load_torch_model(args) if use_torch else None
    history: list[dict[str, str]] = []

    eprint(f"[wohper] backend={args.backend} socket={args.socket}")
    eprint("[wohper] commands: /exit, /reset")

    while True:
        try:
            user_text = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_text:
            continue
        if user_text in {"/exit", "/quit"}:
            break
        if user_text == "/reset":
            history.clear()
            eprint("[wohper] history cleared")
            continue

        prompt_text = render_prompt_text(tokenizer, history, user_text, args.chat_template)
        token_ids = encode_prompt(tokenizer, prompt_text)

        print("assistant> ", end="", flush=True)
        if args.backend == "socket":
            assistant_text = run_socket_generation(tokenizer, prompt_text, user_text, token_ids, args)
        elif args.backend == "torch":
            assistant_text = run_torch_generation(tokenizer, model, prompt_text, args)
        else:
            if args.hybrid_token_path:
                assistant_text = run_socket_generation(
                    tokenizer,
                    prompt_text,
                    user_text,
                    token_ids,
                    args,
                    model=model,
                )
            else:
                send_prefetch_hint(prompt_text, user_text, token_ids, args)
                if model is not None:
                    assistant_text = run_torch_generation(tokenizer, model, prompt_text, args)
                else:
                    eprint("[wohper] Torch disabled; falling back to Rust socket token stream")
                    assistant_text = run_socket_generation(tokenizer, prompt_text, user_text, token_ids, args)

        history.append({"role": "user", "content": user_text})
        history.append({"role": "assistant", "content": assistant_text})
        if len(history) > args.history_messages:
            history = history[-args.history_messages :]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", default=os.environ.get("ZC_SOCKET", DEFAULT_SOCKET))
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=default_model_dir(),
        help="local Hugging Face model/tokenizer directory",
    )
    parser.add_argument("--backend", choices=["hybrid", "socket", "torch"], default="hybrid")
    parser.add_argument("--torch-compute", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--hybrid-token-path",
        action="store_true",
        help="send token ids to Rust in hybrid mode; use when the server emits HybridComputeRequest events",
    )
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--chat-template", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default=os.environ.get("ZC_TORCH_DEVICE", ""))
    parser.add_argument("--device-map", default=os.environ.get("ZC_TORCH_DEVICE_MAP", ""))
    parser.add_argument("--torch-dtype", default=os.environ.get("ZC_TORCH_DTYPE", "auto"))
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--stop-sequences",
        type=parse_stop_sequences,
        default=[],
        help="literal stop sequences separated by |; supports escapes such as \\n",
    )
    parser.add_argument("--eos-token-ids", type=parse_csv_ints, default=GLM52_EOS_TOKEN_IDS)
    parser.add_argument("--experts", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--warm-layers", default="0,1,2,3")
    parser.add_argument("--context-chars", type=int, default=12000)
    parser.add_argument("--history-messages", type=int, default=12)
    parser.add_argument("--socket-timeout", type=float, default=300.0)
    parser.add_argument("--char-delay", type=float, default=0.0)
    parser.add_argument("--show-token-ids", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        interactive_loop(args)
    except SystemExit:
        raise
    except Exception as exc:
        eprint(f"[wohper] fatal: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
