#!/usr/bin/env python3
"""Wohper terminal chat - a boxed-composer TUI for the local engine.

A boxed composer at the bottom, streamed answers with a dim thinking
section, slash commands, EN/IT interface language. Talks to the OpenAI
shim on 127.0.0.1:8114 (which serves the language-adaptive default system
prompt and the no-think fast path), and reuses the container/server
bootstrap from deepseek_chat_repl.

Run:  py -X utf8 tools/wohper_cli.py

Keys: Enter send · trailing "\\" continues on a new line · Ctrl+C stops
the current generation (twice at the prompt exits) · /help for commands.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from deepseek_chat_repl import server_running, start_server  # noqa: E402
from wohper_setup import host_ram_gb, tuning_for  # noqa: E402


def apply_autotune() -> str:
    """Sizes the engine knobs from the machine's physical RAM before the
    container starts. setdefault: anything the user already exported wins."""
    ram = host_ram_gb()
    tune = tuning_for(ram)
    os.environ.setdefault("ZC_DENSE_CACHE_MB", str(tune["dense_mb"]))
    os.environ.setdefault("ZC_EXPERT_RAM_CACHE_MB", str(tune["expert_ram_mb"]))
    os.environ.setdefault("ZC_KV_SLOTS", str(tune["kv_slots"]))
    os.environ.setdefault("ZC_SESSION_SLOTS", str(tune["session_slots"]))
    expert = os.environ["ZC_EXPERT_RAM_CACHE_MB"]
    return (
        f"{ram} GB RAM tier: dense cache {os.environ['ZC_DENSE_CACHE_MB']} MB, "
        f"expert cache {'off' if expert == '0' else expert + ' MB'}, "
        f"KV {os.environ['ZC_KV_SLOTS']}, "
        f"{os.environ['ZC_SESSION_SLOTS']} conversation slots"
    )

REPO = Path(__file__).resolve().parent.parent
SETTINGS_PATH = REPO / "state" / "wohper_cli.json"
API = "http://127.0.0.1:8114"
MODEL = "wohper-deepseek-v4-flash"

# --- terminal paint ---------------------------------------------------------

os.system("")  # enable VT sequences on Windows terminals
USE_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def paint(text: str, code: str) -> str:
    return f"\x1b[{code}m{text}\x1b[0m" if USE_COLOR else text


def clay(text: str) -> str:
    return paint(text, "38;2;217;119;87")


def dim(text: str) -> str:
    return paint(text, "2")


def dim_italic(text: str) -> str:
    return paint(text, "2;3")


def bold(text: str) -> str:
    return paint(text, "1")


SPARK = clay("✻")

# --- i18n -------------------------------------------------------------------

STRINGS = {
    "en": {
        "banner_sub": "284B parameters, running entirely on this computer.",
        "direct_line": "You are talking directly to the model itself - nothing leaves this machine.",
        "tips": "Enter to send · end a line with \\ to continue · Ctrl+C to stop · /help for commands",
        "thinking": "Thinking…",
        "working": "Working…",
        "thought": "Thought for {seconds}s",
        "footer": "{seconds}s · {tokens} tokens",
        "footer_thought": "thought {think_seconds}s · {seconds}s total · {tokens} tokens",
        "interrupted": "Interrupted - the engine finishes in the background; the next message waits for it.",
        "server_boot": "Starting the engine (cold start takes a few minutes)…",
        "api_down": "The API is not answering. Is the zc-chat container running?",
        "exit_hint": "Press Ctrl+C again to exit.",
        "bye": "Bye.",
        "new_chat": "Started a new chat.",
        "lang_set": "Interface language: English.",
        "think_on": "Extended thinking: on.",
        "think_off": "Extended thinking: off (direct answers).",
        "max_set": "Max answer tokens: {value}.",
        "system_set": "System prompt replaced.",
        "unknown": "Unknown command. /help lists the commands.",
        "help": """  /new              start a new chat
  /lang en|it       interface language
  /think on|off     extended thinking (off = fast, direct answers)
  /max N            max answer tokens (current: {max_tokens})
  /system TEXT      replace the system prompt for this chat
  /quit             exit""",
        "error": "Engine error: {error}",
    },
    "it": {
        "banner_sub": "284B parametri, gira interamente su questo computer.",
        "direct_line": "Stai parlando direttamente col modello - niente esce da questo computer.",
        "tips": "Invio per inviare · termina una riga con \\ per continuare · Ctrl+C per fermare · /help per i comandi",
        "thinking": "Sto ragionando…",
        "working": "Sto elaborando…",
        "thought": "Ragionamento: {seconds}s",
        "footer": "{seconds}s · {tokens} token",
        "footer_thought": "ragionamento {think_seconds}s · {seconds}s totali · {tokens} token",
        "interrupted": "Interrotto - il motore finisce in background; il prossimo messaggio lo aspetta.",
        "server_boot": "Avvio il motore (a freddo servono alcuni minuti)…",
        "api_down": "L'API non risponde. Il container zc-chat è acceso?",
        "exit_hint": "Premi di nuovo Ctrl+C per uscire.",
        "bye": "Ciao.",
        "new_chat": "Nuova chat.",
        "lang_set": "Lingua interfaccia: italiano.",
        "think_on": "Ragionamento esteso: acceso.",
        "think_off": "Ragionamento esteso: spento (risposte dirette).",
        "max_set": "Token massimi per risposta: {value}.",
        "system_set": "System prompt sostituito.",
        "unknown": "Comando sconosciuto. /help elenca i comandi.",
        "help": """  /new              nuova chat
  /lang en|it       lingua dell'interfaccia
  /think on|off     ragionamento esteso (off = risposte rapide e dirette)
  /max N            token massimi per risposta (attuale: {max_tokens})
  /system TESTO     sostituisce il system prompt per questa chat
  /quit             esci""",
        "error": "Errore del motore: {error}",
    },
}


def load_settings() -> dict:
    try:
        return json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_settings(settings: dict) -> None:
    try:
        SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        SETTINGS_PATH.write_text(json.dumps(settings), encoding="utf-8")
    except OSError:
        pass


settings = load_settings()
lang = settings.get("lang", "en")
think = bool(settings.get("think", False))
max_tokens = int(settings.get("max_tokens", 256))


def t(key: str, **kwargs) -> str:
    text = (STRINGS.get(lang) or STRINGS["en"])[key]
    return text.format(**kwargs) if kwargs else text


# --- composer (framed input box) --------------------------------------------


def terminal_width() -> int:
    if not sys.stdout.isatty():
        return 80
    try:
        return min(os.get_terminal_size().columns - 1, 120)
    except OSError:
        return 80


def rule(left: str, right: str, width: int) -> str:
    return dim(left + "─" * (width - 2) + right)


def read_message() -> str:
    """Boxed prompt: the full frame (all four borders) is drawn first, then
    the cursor moves INTO the middle row for typing - the box always looks
    complete, even while writing. A trailing backslash continues on a plain
    line below the box; Ctrl+C at the prompt asks once, twice exits."""
    width = terminal_width()
    interactive = sys.stdout.isatty() and USE_COLOR
    print(rule("╭", "╮", width))
    if interactive:
        print(dim("│") + " " * (width - 2) + dim("│"))
        print(rule("╰", "╯", width))
        # Up two rows, into the middle of the frame.
        sys.stdout.write("\x1b[2A")
        sys.stdout.flush()
    lines: list[str] = []
    while True:
        first = not lines
        prefix = dim("│ ") + clay("❯ ") if first else dim("… ")
        try:
            line = input(prefix)
        except EOFError:
            raise KeyboardInterrupt from None
        if interactive and first:
            # input() left the cursor on the bottom-border row: step below
            # the frame without overwriting it.
            sys.stdout.write("\n")
            sys.stdout.flush()
        lines.append(line[:-1] if line.endswith("\\") else line)
        if not line.endswith("\\"):
            break
    if not interactive:
        print(rule("╰", "╯", width))
    return "\n".join(lines).strip()


# --- streaming --------------------------------------------------------------


def stream_turn(history: list[dict]) -> tuple[str, bool]:
    """Streams one assistant turn to the terminal. Returns (answer,
    interrupted). Thinking arrives dim/italic, the answer plain."""
    payload = {
        "model": MODEL,
        "messages": history,
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": True,
        "reasoning": think,
    }
    request = urllib.request.Request(
        API + "/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json; charset=utf-8"},
    )
    print()  # breathing room between the composer and the answer
    answer, reasoning_started, answer_started = "", False, False
    tokens = 0
    started = time.time()
    think_seconds = 0
    # Live status with a ticking elapsed counter until the first token
    # lands (that silence is the prompt prefill - the longest wait).
    status_label = t("thinking") if think else t("working")
    status_lock = threading.Lock()
    status_done = threading.Event()

    def draw_status() -> None:
        elapsed = int(time.time() - started)
        print(
            f"\r\x1b[2K{SPARK} " + dim(f"{status_label} {elapsed}{t('seconds')}"),
            end="",
            flush=True,
        )

    def ticker() -> None:
        while not status_done.wait(1.0):
            with status_lock:
                if status_done.is_set():
                    break
                draw_status()

    draw_status()
    threading.Thread(target=ticker, daemon=True).start()

    def clear_status() -> None:
        if not status_done.is_set():
            with status_lock:
                status_done.set()
                print("\r\x1b[2K", end="", flush=True)

    response = None
    try:
        response = urllib.request.urlopen(request, timeout=3600)
        buffer = b""
        while True:
            chunk = response.read(1)
            if not chunk:
                break
            buffer += chunk
            if not buffer.endswith(b"\n\n"):
                continue
            for event in buffer.decode("utf-8", "replace").split("\n\n"):
                line = event.strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    continue
                try:
                    delta = json.loads(data)["choices"][0]["delta"]
                except (ValueError, KeyError, IndexError):
                    continue
                if delta.get("reasoning_content"):
                    clear_status()
                    if not reasoning_started:
                        print(f"{SPARK} " + dim(t("thinking")))
                        reasoning_started = True
                    print(dim_italic(delta["reasoning_content"]), end="", flush=True)
                    tokens += 1
                if delta.get("content"):
                    clear_status()
                    if reasoning_started and not answer_started:
                        print()  # close the thinking block
                    if not answer_started:
                        think_seconds = int(time.time() - started)
                        print(f"{SPARK} ", end="")
                        answer_started = True
                    print(delta["content"], end="", flush=True)
                    answer += delta["content"]
                    tokens += 1
            buffer = b""
    except KeyboardInterrupt:
        clear_status()
        if answer_started or reasoning_started:
            print()
        print(dim(t("interrupted")))
        if response is not None:
            try:
                response.close()
            except OSError:
                pass
        return answer, True
    except (urllib.error.URLError, OSError) as error:
        clear_status()
        print(dim(t("error", error=getattr(error, "reason", error))))
        return answer, True

    clear_status()
    print()
    seconds = round(time.time() - started)
    if reasoning_started:
        footer = t(
            "footer_thought",
            think_seconds=think_seconds,
            seconds=seconds,
            tokens=tokens,
        )
    else:
        footer = t("footer", seconds=seconds, tokens=tokens)
    print(dim("  " + footer))
    return answer, False


# --- commands ----------------------------------------------------------------


def handle_command(command: str, history: list[dict], system_holder: list) -> bool:
    """Returns False when the CLI should exit."""
    global lang, think, max_tokens
    parts = command.split(None, 1)
    name = parts[0].lower()
    argument = parts[1].strip() if len(parts) > 1 else ""
    if name in ("/quit", "/exit", "/q"):
        return False
    if name == "/help":
        print(dim(t("help", max_tokens=max_tokens)))
    elif name == "/new":
        history.clear()
        if system_holder[0]:
            history.append({"role": "system", "content": system_holder[0]})
        print(dim(t("new_chat")))
    elif name == "/lang":
        if argument in STRINGS:
            lang = argument
            settings["lang"] = lang
            save_settings(settings)
        print(dim(t("lang_set")))
    elif name == "/think":
        think = argument != "off"
        settings["think"] = think
        save_settings(settings)
        print(dim(t("think_on" if think else "think_off")))
    elif name == "/max":
        try:
            max_tokens = max(16, min(1024, int(argument)))
            settings["max_tokens"] = max_tokens
            save_settings(settings)
        except ValueError:
            pass
        print(dim(t("max_set", value=max_tokens)))
    elif name == "/system":
        system_holder[0] = argument
        history[:] = [m for m in history if m.get("role") != "system"]
        if argument:
            history.insert(0, {"role": "system", "content": argument})
        print(dim(t("system_set")))
    else:
        print(dim(t("unknown")))
    return True


# --- main --------------------------------------------------------------------


def api_healthy() -> bool:
    try:
        with urllib.request.urlopen(API + "/health", timeout=5) as response:
            return json.load(response).get("status") == "ok"
    except (urllib.error.URLError, OSError, ValueError):
        return False


def main() -> int:
    print()
    print(f"{SPARK} {bold('Wohper')}")
    print(dim("  " + t("banner_sub")))
    print(dim("  " + t("direct_line")))
    print(dim("  " + t("tips")))
    print(dim("  by Ilben · github.com/twenytwo/wohper"))
    print(dim("  " + apply_autotune()))
    print()

    if not api_healthy():
        if not server_running():
            print(dim(t("server_boot")))
            start_server()
        for _ in range(60):
            if api_healthy():
                break
            time.sleep(2)
        else:
            print(dim(t("api_down")))
            return 1

    history: list[dict] = []
    system_holder = [""]  # empty -> the shim injects its default system
    ctrl_c_armed = False
    while True:
        try:
            message = read_message()
        except KeyboardInterrupt:
            if ctrl_c_armed:
                print()
                print(dim(t("bye")))
                return 0
            ctrl_c_armed = True
            print()
            print(dim(t("exit_hint")))
            continue
        ctrl_c_armed = False
        if not message:
            continue
        if message.startswith("/"):
            if not handle_command(message, history, system_holder):
                print(dim(t("bye")))
                return 0
            continue
        history.append({"role": "user", "content": message})
        answer, interrupted = stream_turn(history)
        if answer:
            history.append({"role": "assistant", "content": answer})
        elif interrupted:
            history.pop()  # the aborted question is not part of the context
        print()


if __name__ == "__main__":
    raise SystemExit(main())
