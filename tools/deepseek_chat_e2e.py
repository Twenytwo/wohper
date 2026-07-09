#!/usr/bin/env python3
"""Chat E2E wrapper: testo -> tokenizer -> zc_infer_server (docker) -> testo.

Uso (da root repo, host Windows con Git Bash/py):
  py tools/deepseek_chat_e2e.py --prompt "The capital of France is" --max-new-tokens 6
  py tools/deepseek_chat_e2e.py --prompt-file prompt.txt --layer-end 5   # solo L0-L4
  py tools/deepseek_chat_e2e.py --decode-log logs/quality_france_full_2026-07-06.log

Il wrapper lancia lo smoke script in docker (stessa configurazione validata),
parsa gli eventi Token JSON dal log e decodifica il testo generato.
Con --dense-cache-mb / --lmhead-cache abilita le cache RAM.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TOKENIZER = REPO / "models" / "deepseek-ai" / "DeepSeek-V4-Flash" / "tokenizer.json"
TOKEN_EVENT = re.compile(r'\{"event":"Token".*?\}')


def load_tokenizer():
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(TOKENIZER))


def extract_tokens(log_text: str) -> list[dict]:
    events = []
    for line in log_text.splitlines():
        start = line.find('{"event":"Token"')
        if start < 0:
            continue
        try:
            events.append(json.loads(line[start:]))
        except json.JSONDecodeError:
            continue
    # dedup per index (il log puo' duplicare le righe stdout/stderr)
    by_index = {}
    for event in events:
        by_index.setdefault(event.get("index"), event)
    return [by_index[key] for key in sorted(by_index)]


def run_generation(token_ids: list[int], args) -> str:
    log_path = REPO / "logs" / f"chat_e2e_{int(time.time())}.log"
    env_flags = [
        "-e", "CARGO_TARGET_DIR=/cargo-target",
        "-e", "ZC_FORCE_BUILD=0",
        "-e", f"ZC_CLIENT_TIMEOUT={args.timeout}s",
        "-e", f"ZC_SERVER_TIMEOUT={args.timeout}s",
        "-e", f"ZC_TOKENS={','.join(map(str, token_ids))}",
        "-e", f"ZC_MAX_NEW_TOKENS={args.max_new_tokens}",
        "-e", "ZC_EXPERTS=",
        "-e", "ZC_SIDECAR_EXPERTS=all",
        "-e", "ZC_ACTIVE_EXPERTS=6",
        "-e", "ZC_IO_BUFFER_COUNT=10",
    ]
    if args.layer_end is not None:
        env_flags += ["-e", f"ZC_LOCAL_LAYER_END={args.layer_end}"]
    if args.dense_cache_mb:
        env_flags += ["-e", f"ZC_DENSE_CACHE_MB={args.dense_cache_mb}"]
    if args.lmhead_cache:
        env_flags += ["-e", "ZC_LMHEAD_CACHE=1"]
    command = [
        "docker", "run", "--rm",
        "--ulimit", "memlock=-1:-1", "--cap-add", "IPC_LOCK",
        "--security-opt", "seccomp=unconfined",
        "-v", f"{REPO}:/workspace",
        "-v", "zc-cargo-target:/cargo-target",
        "-v", "zc-cargo-registry:/usr/local/cargo/registry",
        "-w", "/workspace",
        *env_flags,
        "zc-infer-dev", "bash", "scripts/deepseek_rust_server_smoke.sh",
    ]
    print(f"[chat_e2e] tokens={token_ids} max_new={args.max_new_tokens} log={log_path.name}")
    started = time.time()
    with open(log_path, "w", encoding="utf-8", errors="replace") as log_file:
        result = subprocess.run(command, stdout=log_file, stderr=subprocess.STDOUT)
    elapsed = time.time() - started
    print(f"[chat_e2e] exit={result.returncode} wall={elapsed:.0f}s")
    return log_path.read_text(encoding="utf-8", errors="replace")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt")
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--decode-log", type=Path, help="solo decodifica un log esistente")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--layer-end", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=5400)
    parser.add_argument("--dense-cache-mb", type=int, default=0)
    parser.add_argument("--lmhead-cache", action="store_true")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="preset demo: dense cache completa (6500MB, richiede WSL2 con "
        "memory>=12GB via .wslconfig) + lm_head in RAM + 10 buffer I/O",
    )
    args = parser.parse_args()
    if args.fast:
        args.dense_cache_mb = max(args.dense_cache_mb, 6500)
        args.lmhead_cache = True

    tokenizer = load_tokenizer()

    if args.decode_log:
        log_text = args.decode_log.read_text(encoding="utf-8", errors="replace")
        events = extract_tokens(log_text)
        ids = [event["token_id"] for event in events]
        print(f"[chat_e2e] {len(ids)} token generati: {ids}")
        print(f"[chat_e2e] testo: {tokenizer.decode(ids)!r}")
        return 0

    prompt = args.prompt
    if args.prompt_file:
        prompt = args.prompt_file.read_text(encoding="utf-8")
    if not prompt:
        parser.error("serve --prompt, --prompt-file o --decode-log")

    token_ids = tokenizer.encode(prompt).ids
    print(f"[chat_e2e] prompt {prompt!r} -> {len(token_ids)} token")
    log_text = run_generation(token_ids, args)
    events = extract_tokens(log_text)
    if not events:
        print("[chat_e2e] ERRORE: nessun Token event nel log")
        tail = "\n".join(log_text.splitlines()[-10:])
        print(tail)
        return 1
    generated = [event["token_id"] for event in events]
    logits = [round(event.get("logit", 0.0), 3) for event in events]
    print(f"[chat_e2e] generati {len(generated)} token: {generated}")
    print(f"[chat_e2e] logits: {logits}")
    print(f"[chat_e2e] === OUTPUT ===")
    print(prompt + tokenizer.decode(generated))
    return 0


if __name__ == "__main__":
    sys.exit(main())
