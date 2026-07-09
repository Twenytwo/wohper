#!/usr/bin/env python3
"""E2E gate for the multi-slot session cache (V6).

Interleaves two conversations against the OpenAI shim (A, B, A2, B2): with
the old single-slot cache each turn-2 re-paid the full history prefill;
with multi-slot each conversation must restore its own slot (check
`session_reuse slot=` lines in `docker logs zc-chat`).
"""
import json
import time
import urllib.request

BASE = "http://127.0.0.1:8114/v1/chat/completions"
SYSTEM = "You are a concise assistant. Answer in one short sentence."


def turn(history, question, tag):
    history = history + [{"role": "user", "content": question}]
    payload = {
        "model": "wohper-deepseek-v4-flash",
        "messages": [{"role": "system", "content": SYSTEM}] + history,
        "temperature": 0,
        "max_tokens": 48,
        "stream": False,
    }
    started = time.time()
    request = urllib.request.Request(
        BASE,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=1800) as response:
        body = json.load(response)
    elapsed = time.time() - started
    answer = body["choices"][0]["message"]["content"].strip()
    usage = body.get("usage", {})
    print(
        f"[{tag}] {elapsed:.1f}s prompt_tokens={usage.get('prompt_tokens')} "
        f"completion_tokens={usage.get('completion_tokens')} -> {answer!r}",
        flush=True,
    )
    history.append({"role": "assistant", "content": answer})
    return history


def main():
    convo_a = []
    convo_b = []
    convo_a = turn(convo_a, "What is 2+2?", "A1")
    convo_b = turn(convo_b, "What is the capital of France?", "B1")
    convo_a = turn(convo_a, "And what is that number plus 3?", "A2")
    convo_b = turn(convo_b, "And what country is that city in?", "B2")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
