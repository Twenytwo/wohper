#!/usr/bin/env python3
"""OpenAI-compatible HTTP shim for the Wohper inference server.

Runs INSIDE the zc-chat container next to the Unix-socket engine and exposes
`/v1/chat/completions` (+ `/v1/models`) on port 8114, so any OpenAI-style
client or agent framework can drive the local DeepSeek-V4-Flash.

DeepSeek answers in thinking style ("...reasoning...</think>answer"): the
shim maps the reasoning to `reasoning_content` and the final answer to
`content`, mirroring DeepSeek's official API shape. Passing
`"reasoning": false` in the request pre-fills an empty think block, so the
model answers directly - at ~4-8s/token, skipping a 50-150 token reasoning
preamble is the single biggest latency lever for simple questions.

When the client sends no system message, a default one is injected that
tells the model to always answer in the user's language.

Greedy (temperature 0) keeps the speculative fast path; temperature > 0
falls back to the slower sequential sampler.

Start (inside the container):
  python3 tools/zc_openai_server.py --port 8114 --socket /tmp/wohper-chat.sock
"""
from __future__ import annotations

import argparse
import json
import socket
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TOKENIZER_PATH = REPO / "models" / "deepseek-ai" / "DeepSeek-V4-Flash" / "tokenizer.json"

BOS = 0
EOS = 1
USER = 128803
ASSISTANT = 128804
THINK_END = "</think>"

MODEL_ID = "wohper-deepseek-v4-flash"

DEFAULT_SYSTEM = (
    "You are Wohper, a helpful assistant running entirely on local hardware. "
    "Always respond in the same language the user writes in. "
    "Be concise and direct."
)

ENGINE_LOCK = threading.Lock()  # the engine serves one request at a time
TOKENIZER = None
ENGINE_SOCKET = "/tmp/wohper-chat.sock"


def encode_messages(messages: list[dict], think: bool = True) -> list[int]:
    """Chat template validated against the reference: [BOS] system then
    <|User|>q<|Assistant|>a<|end|> pairs; trailing <|Assistant|> opens the
    reply. With think=False an empty think block ("</think>") is pre-filled
    after the trailing <|Assistant|>, so the model skips the reasoning
    preamble and answers directly. When the client provides no system
    message, DEFAULT_SYSTEM is injected (language-adaptive answers)."""
    if not any(message.get("role") == "system" for message in messages):
        messages = [{"role": "system", "content": DEFAULT_SYSTEM}] + list(messages)
    ids: list[int] = [BOS]
    for index, message in enumerate(messages):
        role = message.get("role", "user")
        content = message.get("content") or ""
        if isinstance(content, list):  # OpenAI content-parts form
            content = "".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )
        if role == "system":
            ids += TOKENIZER.encode(content).ids
        elif role == "user":
            ids += [USER] + TOKENIZER.encode(content).ids
        elif role == "assistant":
            ids += [ASSISTANT] + TOKENIZER.encode(content).ids + [EOS]
        else:  # tool/other roles: treat as user context
            ids += [USER] + TOKENIZER.encode(content).ids
    ids += [ASSISTANT]
    if not think:
        ids += TOKENIZER.encode(THINK_END).ids
    return ids


def engine_stream(token_ids: list[int], max_new: int, temperature: float):
    """Yields generated token ids as they arrive from the engine socket."""
    envelope = {
        "request_id": f"openai-{uuid.uuid4().hex[:12]}",
        "objective": "openai shim",
        "token_ids": token_ids,
        "max_new_tokens": max_new,
        "route_hint": {"expert_ids": []},
        "temperature": temperature,
        "top_k": 1 if temperature <= 0 else 40,
        "top_p": 1.0 if temperature <= 0 else 0.95,
        "repetition_penalty": 1.0,
        "seed": 1,
    }
    payload = json.dumps(envelope, separators=(",", ":")).encode() + b"\n"
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.connect(ENGINE_SOCKET)
        client.sendall(payload)
        client.shutdown(socket.SHUT_WR)
        buffer = b""
        while True:
            chunk = client.recv(65536)
            if not chunk:
                break
            buffer += chunk
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                start = line.find(b'{"event"')
                if start < 0:
                    continue
                try:
                    event = json.loads(line[start:])
                except json.JSONDecodeError:
                    continue
                kind = event.get("event")
                if kind == "Token":
                    yield event.get("token_id")
                elif kind == "Finished":
                    return


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # quiet
        pass

    def _json(self, code: int, body: dict) -> None:
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/v1/models":
            self._json(200, {
                "object": "list",
                "data": [{"id": MODEL_ID, "object": "model", "owned_by": "wohper"}],
            })
        elif self.path in ("/health", "/"):
            self._json(200, {"status": "ok", "model": MODEL_ID})
        else:
            self._json(404, {"error": {"message": "not found"}})

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self._json(404, {"error": {"message": "not found"}})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) or b"{}"
            try:
                request = json.loads(raw)
            except UnicodeDecodeError:
                # Windows curl often ships latin-1 bodies; be forgiving.
                request = json.loads(raw.decode("latin-1"))
            messages = request.get("messages") or []
            if not messages:
                raise ValueError("messages is required")
            max_new = int(request.get("max_tokens") or 64)
            max_new = max(1, min(1024, max_new))
            temperature = float(request.get("temperature") or 0.0)
            stream = bool(request.get("stream"))
            # "reasoning": false (or "think": false) pre-fills an empty
            # think block: the model answers directly.
            think = request.get("reasoning", request.get("think", True))
            think = think not in (False, "false", "off", "none", 0)
            prompt_ids = encode_messages(messages, think=think)
        except Exception as error:  # noqa: BLE001
            self._json(400, {"error": {"message": str(error)}})
            return

        completion_id = f"chatcmpl-{uuid.uuid4().hex[:20]}"
        created = int(time.time())

        with ENGINE_LOCK:
            if stream:
                self._stream_response(
                    completion_id, created, prompt_ids, max_new, temperature, think
                )
            else:
                self._full_response(
                    completion_id, created, prompt_ids, max_new, temperature, think
                )

    # --- non-streaming -----------------------------------------------------

    def _full_response(self, completion_id, created, prompt_ids, max_new, temperature, think=True):
        think_end_id = TOKENIZER.token_to_id(THINK_END)
        generated: list[int] = []
        for token_id in engine_stream(prompt_ids, max_new, temperature):
            if token_id == EOS:
                break
            # With the empty think block pre-filled the model occasionally
            # re-emits </think> and restarts the answer: cut there.
            if not think and token_id == think_end_id:
                break
            generated.append(token_id)
        text = TOKENIZER.decode(generated)
        if THINK_END in text:
            reasoning, answer = text.rsplit(THINK_END, 1)
        else:
            reasoning, answer = None, text
        finish = "stop" if len(generated) < max_new else "length"
        message = {"role": "assistant", "content": answer.strip()}
        if reasoning is not None:
            message["reasoning_content"] = reasoning.strip()
        self._json(200, {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": MODEL_ID,
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": {
                "prompt_tokens": len(prompt_ids),
                "completion_tokens": len(generated),
                "total_tokens": len(prompt_ids) + len(generated),
            },
        })

    # --- streaming (SSE) ----------------------------------------------------

    def _sse(self, obj: dict) -> None:
        self.wfile.write(f"data: {json.dumps(obj)}\n\n".encode())
        self.wfile.flush()

    def _stream_response(self, completion_id, created, prompt_ids, max_new, temperature, think=True):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        def chunk(delta: dict, finish=None):
            return {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": MODEL_ID,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }

        self._sse(chunk({"role": "assistant"}))
        generated: list[int] = []
        printed = ""
        # With the empty think block pre-filled, the generated text never
        # contains "</think>": everything is already the answer.
        thinking = think
        finish = "length"
        think_end_id = TOKENIZER.token_to_id(THINK_END)
        try:
            for token_id in engine_stream(prompt_ids, max_new, temperature):
                if token_id == EOS:
                    finish = "stop"
                    break
                if not think and token_id == think_end_id:
                    finish = "stop"
                    break
                generated.append(token_id)
                text = TOKENIZER.decode(generated)
                if not text.startswith(printed):
                    printed = ""  # byte-level re-decode: resend from scratch
                delta_text = text[len(printed):]
                if not delta_text:
                    continue
                if thinking:
                    boundary = text.find(THINK_END, max(0, len(printed) - len(THINK_END)))
                    if boundary >= 0:
                        pre = text[len(printed):boundary]
                        post = text[boundary + len(THINK_END):]
                        if pre:
                            self._sse(chunk({"reasoning_content": pre}))
                        if post:
                            self._sse(chunk({"content": post}))
                        thinking = False
                    else:
                        self._sse(chunk({"reasoning_content": delta_text}))
                else:
                    self._sse(chunk({"content": delta_text}))
                printed = text
        except (BrokenPipeError, ConnectionResetError):
            return
        self._sse(chunk({}, finish))
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


def main() -> int:
    global TOKENIZER, ENGINE_SOCKET
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8114)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--socket", default="/tmp/wohper-chat.sock")
    args = parser.parse_args()
    ENGINE_SOCKET = args.socket

    from tokenizers import Tokenizer

    TOKENIZER = Tokenizer.from_file(str(TOKENIZER_PATH))
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"wohper openai shim listening on {args.host}:{args.port} -> {ENGINE_SOCKET}", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
