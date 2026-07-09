#!/usr/bin/env python3
"""Quality suite (Q): batch of complex questions against the running
zc-chat server. Each question is an independent conversation (fresh
prefill, no session reuse interference). Results land in a JSON report
for manual grading + repo evidence.

Run on the HOST (needs tokenizers + docker):
  py -X utf8 tools/quality_suite.py --max-reply 24
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TOKENIZER = REPO / "models" / "deepseek-ai" / "DeepSeek-V4-Flash" / "tokenizer.json"

BOS = 0
EOS = 1
USER = 128803
ASSISTANT = 128804

SYSTEM = "You are a helpful assistant. Answer in one short sentence. Be direct and factual."

QUESTIONS = [
    ("arithmetic", "What is 17 * 23?"),
    ("word_math", "If a train travels 120 km in 1.5 hours, what is its average speed in km/h?"),
    ("knowledge", "What is the capital of Australia?"),
    ("code", "Write a Python function that returns the factorial of n."),
    ("italian", "Qual e' la capitale d'Italia?"),
    ("networking", "What is the main difference between TCP and UDP?"),
    ("trick", "A farmer has 17 sheep. All but 9 die. How many sheep are left?"),
    ("history", "In what year did World War II end?"),
    ("translation", "Translate to French: 'The weather is beautiful today.'"),
    ("science", "What is the chemical formula of water?"),
    ("algebra", "If x + 2x = 12, what is x?"),
    ("literature", "Who wrote 'One Hundred Years of Solitude'?"),
    ("explain", "Explain recursion in one simple sentence."),
    ("pattern", "What is the next number in the sequence: 2, 6, 12, 20, 30?"),
    ("prime", "Is 97 a prime number?"),
    ("biology", "Summarize photosynthesis in one sentence."),
]


def run_one(tokenizer, category, question, max_reply, socket, timeout_s):
    ids = [BOS] + tokenizer.encode(SYSTEM).ids
    ids += [USER] + tokenizer.encode(question).ids + [ASSISTANT]
    started = time.time()
    command = [
        "docker", "exec", "zc-chat",
        "python3", "tools/zc_socket_smoke_client.py",
        "--socket", socket,
        "--request-id", f"quality-{category}",
        "--token-ids", ",".join(map(str, ids)),
        "--experts", "",
        "--max-new-tokens", str(max_reply),
        "--temperature", "0", "--top-k", "1", "--top-p", "1",
        "--repetition-penalty", "1", "--seed", "1",
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout_s)
    elapsed = round(time.time() - started, 1)
    generated = []
    seen = set()
    for line in result.stdout.splitlines():
        start = line.find('{"event":"Token"')
        if start < 0:
            continue
        try:
            event = json.loads(line[start:])
        except json.JSONDecodeError:
            continue
        index = event.get("index")
        if index in seen:
            continue
        seen.add(index)
        generated.append((index, event.get("token_id")))
    tokens = [token for _, token in sorted(generated)]
    if EOS in tokens:
        tokens = tokens[: tokens.index(EOS)]
    reply = tokenizer.decode(tokens).strip()
    return {
        "category": category,
        "question": question,
        "reply": reply,
        "tokens_generated": len(tokens),
        "prompt_positions": len(ids),
        "elapsed_s": elapsed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-reply", type=int, default=24)
    parser.add_argument("--socket", default="/tmp/wohper-chat.sock")
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    results = []
    for index, (category, question) in enumerate(QUESTIONS, 1):
        print(f"[{index}/{len(QUESTIONS)}] {category}: {question}", flush=True)
        try:
            entry = run_one(tokenizer, category, question, args.max_reply, args.socket, args.timeout)
        except Exception as error:  # noqa: BLE001 - keep the suite running
            entry = {"category": category, "question": question, "error": str(error)}
        print(f"    -> {entry.get('reply', entry.get('error'))!r} ({entry.get('elapsed_s', '?')}s)", flush=True)
        results.append(entry)

    out_path = args.out or (REPO / "state" / f"quality_suite_{time.strftime('%Y-%m-%d')}.json")
    Path(out_path).write_text(
        json.dumps({"system": SYSTEM, "max_reply": args.max_reply, "results": results}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"saved: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
