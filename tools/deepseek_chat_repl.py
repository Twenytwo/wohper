#!/usr/bin/env python3
"""Wohper chat REPL - DeepSeek-V4-Flash 284B on a 16GB commodity PC.

Streams tokens live from the persistent zc-chat server (Unix socket inside
Docker), reuses the previous turn's KV (history is never re-prefilled) and
pre-warms the system prompt in the background at startup.

Run on the HOST:  py -X utf8 tools/deepseek_chat_repl.py
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TOKENIZER = REPO / "models" / "deepseek-ai" / "DeepSeek-V4-Flash" / "tokenizer.json"

BOS = 0
EOS = 1
USER = 128803
ASSISTANT = 128804
THINK_END_TEXT = "</think>"

SERVER_NAME = "zc-chat"
SERVER_SOCKET = "/tmp/wohper-chat.sock"

DEFAULT_SYSTEM = (
    "You are a helpful assistant. Answer in one short sentence. "
    "Be direct and factual."
)

# --- ANSI helpers ---------------------------------------------------------

USE_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def paint(text: str, code: str) -> str:
    if not USE_COLOR:
        return text
    return f"\x1b[{code}m{text}\x1b[0m"


def dim(text: str) -> str:
    return paint(text, "2")


def cyan(text: str) -> str:
    return paint(text, "36")


def green(text: str) -> str:
    return paint(text, "32")


def yellow(text: str) -> str:
    return paint(text, "33")


def bold(text: str) -> str:
    return paint(text, "1")


# --- server management ----------------------------------------------------


def server_running() -> bool:
    result = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Running}}", SERVER_NAME],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0 and result.stdout.strip() == "true"


def start_server() -> None:
    print(dim("[server] starting the persistent container (~1 min warm-up)..."))
    subprocess.run(["docker", "rm", "-f", SERVER_NAME], capture_output=True)
    subprocess.run(
        [
            "docker", "run", "-d", "--name", SERVER_NAME,
            "--ulimit", "memlock=-1:-1", "--cap-add", "IPC_LOCK",
            "--security-opt", "seccomp=unconfined",
            "-v", f"{REPO}:/workspace",
            "-v", "zc-cargo-target:/cargo-target",
            "-v", "zc-cargo-registry:/usr/local/cargo/registry",
            # V2: model and expert cache on the ext4 volume (the Windows bind
            # mount reads at ~114MB/s, the volume at >1GB/s).
            "-v", "zc-model:/model-fast",
            "-w", "/workspace",
            # OpenAI-compatible API (tools/zc_openai_server.py) on 8114.
            "-p", "127.0.0.1:8114:8114",
            "-e", "CARGO_TARGET_DIR=/cargo-target",
            "-e", "ZC_FORCE_BUILD=0",
            "-e", "ZC_DEEPSEEK_MODEL=/model-fast/dense_core.bin",
            "-e", "ZC_EXPERT_CACHE_DIR=/model-fast/cache-experts",
            "-e", "ZC_EXPERT_CACHE_GB=0",
            "-e", "ZC_MTP_VERIFY=1",
            # Knobs default to the measured 16GB-tier values; the caller
            # (or the user's environment) can override them - wohper_cli
            # sets them from the machine's RAM tier before starting.
            "-e", f"ZC_KV_SLOTS={os.environ.get('ZC_KV_SLOTS', '1024')}",
            "-e", f"ZC_SESSION_SLOTS={os.environ.get('ZC_SESSION_SLOTS', '2')}",
            "-e", f"ZC_DENSE_CACHE_MB={os.environ.get('ZC_DENSE_CACHE_MB', '6100')}",
            "-e", f"ZC_EXPERT_RAM_CACHE_MB={os.environ.get('ZC_EXPERT_RAM_CACHE_MB', '0')}",
            "-e", f"ZC_SOCKET={SERVER_SOCKET}",
            "zc-infer-dev", "bash", "scripts/deepseek_chat_server.sh",
        ],
        check=True,
    )
    for _ in range(120):
        probe = subprocess.run(
            ["docker", "exec", SERVER_NAME, "test", "-S", SERVER_SOCKET],
            capture_output=True,
        )
        if probe.returncode == 0:
            print(dim("[server] ready."))
            break
        time.sleep(1)
    else:
        raise RuntimeError("server socket did not appear; check: docker logs zc-chat")
    # OpenAI shim: bootstrap dependencies and start in background (idempotent).
    subprocess.run(
        ["docker", "exec", SERVER_NAME, "bash", "-c",
         "python3 -c 'import tokenizers' 2>/dev/null || "
         "(apt-get update -qq >/dev/null 2>&1; apt-get install -y -qq python3-pip >/dev/null 2>&1; "
         "python3 -m pip install --break-system-packages -q tokenizers >/dev/null 2>&1)"],
        capture_output=True,
    )
    subprocess.run(
        ["docker", "exec", "-d", SERVER_NAME,
         "python3", "tools/zc_openai_server.py", "--port", "8114",
         "--socket", SERVER_SOCKET],
        capture_output=True,
    )
    print(dim("[api] OpenAI-compatible at http://127.0.0.1:8114/v1"))



def prewarm(system_ids: list[int]) -> None:
    """Prefills the system prompt in the background: the first real turn
    reuses its KV and only pays for the question."""
    if not system_ids:
        return
    warm_ids = ",".join(map(str, [BOS] + system_ids))
    subprocess.Popen(
        [
            "docker", "exec", SERVER_NAME,
            "python3", "tools/zc_socket_smoke_client.py",
            "--socket", SERVER_SOCKET,
            "--request-id", "repl-prewarm",
            "--token-ids", warm_ids,
            "--experts", "",
            "--max-new-tokens", "1",
            "--temperature", "0", "--top-k", "1", "--top-p", "1",
            "--repetition-penalty", "1", "--seed", "1",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    print(dim("[prewarm] system prompt prefilling in background: the first turn reuses it"))


# --- streaming generation --------------------------------------------------


class TurnStats:
    def __init__(self) -> None:
        self.prefill_s = 0.0
        self.decode_s = 0.0
        self.tokens = 0
        self.total_s = 0.0

    @property
    def tokens_per_s(self) -> float:
        return self.tokens / self.decode_s if self.decode_s > 0 else 0.0


def run_generation_streaming(
    tokenizer,
    token_ids: list[int],
    max_new: int,
    temperature: float,
    show_thinking: bool,
    timeout_s: int,
) -> tuple[list[int], TurnStats]:
    """Sends the request and prints tokens AS THEY ARRIVE. Returns the
    generated ids (EOS excluded) and the turn timing stats."""
    command = [
        "docker", "exec", SERVER_NAME,
        "python3", "-u", "tools/zc_socket_smoke_client.py",
        "--socket", SERVER_SOCKET,
        "--request-id", f"chat-{int(time.time())}",
        "--token-ids", ",".join(map(str, token_ids)),
        "--experts", "",
        "--max-new-tokens", str(max_new),
        "--temperature", str(temperature),
        "--top-k", "1" if temperature <= 0 else "40",
        "--top-p", "1" if temperature <= 0 else "0.95",
        "--repetition-penalty", "1",
        "--seed", "1",
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    lines: queue.Queue[str | None] = queue.Queue()

    def reader() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            lines.put(line)
        lines.put(None)

    threading.Thread(target=reader, daemon=True).start()

    stats = TurnStats()
    started = time.time()
    first_token_at: float | None = None
    generated: list[int] = []
    printed_text = ""
    in_thinking = True  # DeepSeek answers thinking-style: text</think>answer
    stopped = False

    def render() -> None:
        """Prints the newly decoded suffix; thinking dim (or dots), answer
        bright green after the `</think>` boundary."""
        nonlocal printed_text, in_thinking
        text = tokenizer.decode(generated)
        if not text.startswith(printed_text):
            # Byte-level BPE re-decoded the tail differently: restart clean.
            printed_text = ""
            sys.stdout.write("\n")
        delta = text[len(printed_text):]
        if not delta:
            return
        if in_thinking:
            search_from = max(0, len(printed_text) - len(THINK_END_TEXT))
            boundary = text.find(THINK_END_TEXT, search_from)
            if boundary >= 0:
                pre = text[len(printed_text):boundary]
                post = text[boundary + len(THINK_END_TEXT):]
                sys.stdout.write(dim(pre) if show_thinking else dim("."))
                sys.stdout.write("\n" + green(bold("dpk> ")) + green(post))
                in_thinking = False
            else:
                sys.stdout.write(dim(delta) if show_thinking else dim("."))
        else:
            sys.stdout.write(green(delta))
        printed_text = text
        sys.stdout.flush()

    spinner = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    spin_i = 0
    while True:
        try:
            line = lines.get(timeout=1.0)
        except queue.Empty:
            if first_token_at is None and time.time() - started < timeout_s:
                spin_i += 1
                elapsed = int(time.time() - started)
                sys.stdout.write(
                    f"\r{dim(f'{spinner[spin_i % len(spinner)]} prefill/wait {elapsed}s ')}"
                )
                sys.stdout.flush()
                continue
            if time.time() - started >= timeout_s:
                process.kill()
                break
            continue
        if line is None:
            break
        start = line.find('{"event"')
        if start < 0:
            continue
        try:
            event = json.loads(line[start:])
        except json.JSONDecodeError:
            continue
        kind = event.get("event")
        if kind == "Token":
            if first_token_at is None:
                first_token_at = time.time()
                stats.prefill_s = round(first_token_at - started, 1)
                sys.stdout.write("\r" + " " * 40 + "\r")  # clear spinner
                sys.stdout.write(dim("pensiero> ") if not show_thinking else dim("pensiero> "))
            token_id = event.get("token_id")
            if token_id == EOS:
                stopped = True
                break
            generated.append(token_id)
            render()
        elif kind == "Finished":
            break

    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        # Il client non deve mai bloccare il REPL: se resta appeso (server
        # ancora in generazione oltre l'EOS o pipe lenta) lo chiudiamo noi;
        # il server e' resiliente alle disconnessioni.
        process.kill()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
    stats.total_s = round(time.time() - started, 1)
    if first_token_at is not None:
        stats.decode_s = round(time.time() - first_token_at, 1)
    stats.tokens = len(generated)
    if in_thinking and generated:
        # Il modello non ha chiuso il thinking (o non l'ha usato): mostra
        # the full answer in clear text anyway.
        sys.stdout.write("\n" + green(bold("dpk> ")) + green(tokenizer.decode(generated).strip()))
    sys.stdout.write("\n")
    sys.stdout.flush()
    return generated, stats


# --- conversation helpers ---------------------------------------------------


def visible_answer(tokenizer, generated: list[int]) -> str:
    text = tokenizer.decode(generated)
    if THINK_END_TEXT in text:
        return text.rsplit(THINK_END_TEXT, 1)[1].strip() or text.strip()
    return text.strip()


HELP = f"""
{bold('Commands:')}
  /help            this list
  /exit            quit
  /reset           clear the conversation (keeps the system prompt)
  /system <text>   new system prompt (+reset +prewarm)
  /reply N         max tokens per answer (current shown in /stats)
  /think on|off    show/hide the model's reasoning
  /temp X          temperature (0 = fast greedy; >0 disables
                   speculative decoding: SLOWER)
  /regen           regenerate the last answer
  /save [name]     save the conversation under logs/
  /stats           timing and speed of the last turn and the session
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-reply", type=int, default=16)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--system", default=DEFAULT_SYSTEM)
    parser.add_argument("--show-thinking", action="store_true")
    args = parser.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(TOKENIZER))

    system_text = args.system
    system_ids = tokenizer.encode(system_text).ids if system_text else []
    history: list[int] = [BOS] + system_ids
    transcript: list[tuple[str, str]] = []  # (role, text) for /save and /regen
    max_reply = args.max_reply
    temperature = 0.0
    show_thinking = bool(args.show_thinking)
    turn = 0
    last_stats: TurnStats | None = None
    session_tokens = 0
    session_decode_s = 0.0

    if not server_running():
        start_server()
    prewarm(system_ids)

    line = "─" * 46
    print(f"{cyan(line)}")
    print(f" {bold('WOHPER')} · DeepSeek-V4-Flash 284B su 16GB RAM")
    print(f" {dim('~3-4s/token at steady state · /help for commands')}")
    print(f" {dim('legacy REPL - the new chat is: py -X utf8 tools/wohper_cli.py')}")
    print(f"{cyan(line)}")

    def do_turn(user_text: str | None) -> None:
        """user_text None = regen (history gia' preparata)."""
        nonlocal history, turn, last_stats, session_tokens, session_decode_s
        if user_text is not None:
            history_local = history + [USER] + tokenizer.encode(user_text).ids + [ASSISTANT]
        else:
            history_local = history
        if user_text is not None and turn > 0:
            new_positions = len(history_local) - len(history)
        elif user_text is not None:
            # first turn: the prewarm covers BOS + system prompt
            new_positions = max(1, len(history_local) - 1 - len(system_ids))
        else:
            new_positions = len(history_local)
        est_s = max(15, new_positions * 6 + max_reply * 5)
        print(dim(f"[~{max(1, (est_s + 30) // 60)} min stimati: "
                  f"{new_positions} posizioni nuove + max {max_reply} token]"))
        generated, stats = run_generation_streaming(
            tokenizer, history_local, max_reply, temperature, show_thinking, args.timeout
        )
        if not generated:
            print(yellow("[no tokens generated - check: docker logs zc-chat]"))
            return
        answer = visible_answer(tokenizer, generated)
        turn += 1
        last_stats = stats
        session_tokens += stats.tokens
        session_decode_s += stats.decode_s
        s_per_token = stats.decode_s / stats.tokens if stats.tokens else 0.0
        print(dim(
            f"[{stats.tokens} token · prefill {stats.prefill_s}s · "
            f"decode {stats.decode_s}s · {s_per_token:.1f} s/token]"
        ))
        history = history_local + generated + [EOS]
        if user_text is not None:
            transcript.append(("tu", user_text))
        transcript.append(("dpk", answer))

    while True:
        try:
            user_text = input(f"\n{cyan(bold('tu> '))}").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nciao!")
            return 0
        if not user_text:
            continue

        if user_text in ("/exit", "/quit"):
            print("ciao!")
            return 0
        if user_text == "/help":
            print(HELP)
            continue
        if user_text == "/reset":
            history = [BOS] + system_ids
            transcript.clear()
            turn = 0
            print(dim("[conversation cleared]"))
            continue
        if user_text.startswith("/system"):
            new_system = user_text[len("/system"):].strip()
            if new_system:
                system_text = new_system
                system_ids = tokenizer.encode(system_text).ids
            history = [BOS] + system_ids
            transcript.clear()
            turn = 0
            prewarm(system_ids)
            print(dim(f"[system prompt: {system_text!r} - conversation cleared]"))
            continue
        if user_text.startswith("/reply"):
            try:
                max_reply = max(1, min(256, int(user_text.split()[1])))
                print(dim(f"[max answer tokens: {max_reply}]"))
            except (IndexError, ValueError):
                print(yellow("uso: /reply N"))
            continue
        if user_text.startswith("/think"):
            arg = user_text.split()[-1].lower()
            show_thinking = arg == "on"
            print(dim(f"[reasoning: {'visible' if show_thinking else 'hidden'}]"))
            continue
        if user_text.startswith("/temp"):
            try:
                temperature = max(0.0, min(2.0, float(user_text.split()[1])))
                note = " (speculative OFF: slower)" if temperature > 0 else ""
                print(dim(f"[temperatura: {temperature}{note}]"))
            except (IndexError, ValueError):
                print(yellow("uso: /temp X"))
            continue
        if user_text == "/regen":
            if turn == 0 or len(transcript) < 2:
                print(yellow("[niente da rigenerare]"))
                continue
            # drop the last answer from the token history and the transcript
            last_question = transcript[-2][1]
            transcript.pop()  # dpk
            transcript.pop()  # tu
            # rebuild the token history from the transcript
            history = [BOS] + system_ids
            for role, text in transcript:
                if role == "tu":
                    history += [USER] + tokenizer.encode(text).ids + [ASSISTANT]
                else:
                    history += tokenizer.encode(text).ids + [EOS]
            print(dim(f"[rigenero: {last_question!r}]"))
            do_turn(last_question)
            continue
        if user_text.startswith("/save"):
            parts = user_text.split(maxsplit=1)
            name = parts[1] if len(parts) > 1 else f"chat_{time.strftime('%Y%m%d_%H%M%S')}"
            out = REPO / "logs" / f"{name}.md"
            body = [f"# Wohper chat - {time.strftime('%Y-%m-%d %H:%M')}",
                    f"_system: {system_text}_", ""]
            for role, text in transcript:
                body.append(f"**{role}**: {text}\n")
            out.write_text("\n".join(body), encoding="utf-8")
            print(dim(f"[saved: {out}]"))
            continue
        if user_text == "/stats":
            if last_stats is None:
                print(dim("[no turns yet]"))
            else:
                avg = session_tokens / session_decode_s if session_decode_s else 0.0
                spt = last_stats.decode_s / last_stats.tokens if last_stats.tokens else 0.0
                savg = session_decode_s / session_tokens if session_tokens else 0.0
                print(dim(
                    f"[last turn: {last_stats.total_s}s total, prefill {last_stats.prefill_s}s, "
                    f"{last_stats.tokens} token a {spt:.1f} s/token · "
                    f"session: {session_tokens} tokens, avg {savg:.1f} s/token · "
                    f"max_reply {max_reply} · temp {temperature}]"
                ))
            continue
        if user_text.startswith("/"):
            print(yellow(f"comando sconosciuto: {user_text} - /help"))
            continue

        do_turn(user_text)


if __name__ == "__main__":
    sys.exit(main())
