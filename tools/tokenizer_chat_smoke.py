#!/usr/bin/env python3
"""Dependency-free GLM-5.2 tokenizer/chat smoke.

This implements the ByteLevel BPE path described by tokenizer.json for small,
controlled smoke prompts. It is not a replacement for Hugging Face tokenizers,
but it lets the Wohper loop verify local special tokens, BPE ids, and
round-trip decode without installing extra packages.
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path


DEFAULT_MODEL_DIR = Path("models/zai-org/GLM-5.2")
GLM52_SPECIALS = [
    "<|endoftext|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|observation|>",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--user-text", default="privacy")
    parser.add_argument("--system-text", default="")
    parser.add_argument("--assistant-prefix", default="")
    parser.add_argument(
        "--messages-json",
        default="",
        help="optional chat messages JSON list with role/content objects",
    )
    parser.add_argument(
        "--messages-json-b64",
        default="",
        help="base64-encoded UTF-8 JSON list; avoids shell quoting issues",
    )
    parser.add_argument("--ids", default="", help="optional comma/space separated ids to decode")
    parser.add_argument(
        "--generated-ids",
        default="",
        help="optional comma/space separated generated ids for streaming decode smoke",
    )
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def bytes_to_unicode() -> tuple[dict[int, str], dict[str, int]]:
    visible = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    byte_values = visible[:]
    codepoints = visible[:]
    extra = 0
    for byte in range(256):
        if byte not in byte_values:
            byte_values.append(byte)
            codepoints.append(256 + extra)
            extra += 1
    byte_encoder = {byte: chr(codepoint) for byte, codepoint in zip(byte_values, codepoints)}
    byte_decoder = {char: byte for byte, char in byte_encoder.items()}
    return byte_encoder, byte_decoder


TOKEN_SPLIT_RE = re.compile(
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
    r"|[0-9]{1,3}"
    r"| ?\w+"
    r"| ?[^\s\w]+[\r\n]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+",
    re.UNICODE,
)


@dataclass
class TokenizerSmoke:
    vocab: dict[str, int]
    by_id: dict[int, str]
    merge_ranks: dict[tuple[str, str], int]
    specials: dict[str, int]
    byte_encoder: dict[int, str]
    byte_decoder: dict[str, int]
    bpe_cache: dict[str, list[str]]

    @classmethod
    def load(cls, model_dir: Path) -> "TokenizerSmoke":
        tokenizer_json = model_dir / "tokenizer.json"
        if not tokenizer_json.exists():
            raise SystemExit(f"missing tokenizer.json: {tokenizer_json}")
        payload = json.loads(tokenizer_json.read_text(encoding="utf-8"))
        vocab = {str(token): int(token_id) for token, token_id in payload["model"]["vocab"].items()}
        specials: dict[str, int] = {}
        for item in payload.get("added_tokens") or []:
            content = item.get("content")
            token_id = item.get("id")
            if content is None or token_id is None:
                continue
            vocab[str(content)] = int(token_id)
            specials[str(content)] = int(token_id)
        by_id = {token_id: token for token, token_id in vocab.items()}
        merges = payload["model"].get("merges") or []
        merge_ranks: dict[tuple[str, str], int] = {}
        for rank, merge in enumerate(merges):
            if isinstance(merge, str):
                left, right = merge.split()
            else:
                left, right = merge
            merge_ranks[(left, right)] = rank
        byte_encoder, byte_decoder = bytes_to_unicode()
        return cls(vocab, by_id, merge_ranks, specials, byte_encoder, byte_decoder, {})

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        special_pattern = self._special_pattern()
        cursor = 0
        for match in special_pattern.finditer(text):
            if match.start() > cursor:
                ids.extend(self._encode_plain(text[cursor : match.start()]))
            ids.append(self.specials[match.group(0)])
            cursor = match.end()
        if cursor < len(text):
            ids.extend(self._encode_plain(text[cursor:]))
        return ids

    def decode(self, ids: list[int]) -> str:
        out: list[str] = []
        byte_chars: list[str] = []
        for token_id in ids:
            token = self.by_id.get(token_id)
            if token is None:
                raise KeyError(f"unknown token id: {token_id}")
            if token in self.specials:
                if byte_chars:
                    out.append(self._decode_byte_chars(byte_chars))
                    byte_chars.clear()
                out.append(token)
            else:
                byte_chars.extend(token)
        if byte_chars:
            out.append(self._decode_byte_chars(byte_chars))
        return "".join(out)

    def _encode_plain(self, text: str) -> list[int]:
        ids: list[int] = []
        for piece in TOKEN_SPLIT_RE.findall(text):
            byte_piece = "".join(self.byte_encoder[byte] for byte in piece.encode("utf-8"))
            for bpe_token in self._bpe(byte_piece):
                try:
                    ids.append(self.vocab[bpe_token])
                except KeyError as exc:
                    raise KeyError(f"BPE token missing from vocab: {bpe_token!r}") from exc
        return ids

    def _bpe(self, token: str) -> list[str]:
        cached = self.bpe_cache.get(token)
        if cached is not None:
            return cached
        word = tuple(token)
        if len(word) <= 1:
            result = list(word)
            self.bpe_cache[token] = result
            return result

        while True:
            pairs = [(word[index], word[index + 1]) for index in range(len(word) - 1)]
            ranked = [
                (self.merge_ranks[pair], pair)
                for pair in pairs
                if pair in self.merge_ranks
            ]
            if not ranked:
                break
            _, best = min(ranked)
            merged: list[str] = []
            index = 0
            while index < len(word):
                if index < len(word) - 1 and (word[index], word[index + 1]) == best:
                    merged.append(word[index] + word[index + 1])
                    index += 2
                else:
                    merged.append(word[index])
                    index += 1
            word = tuple(merged)
            if len(word) == 1:
                break

        result = list(word)
        self.bpe_cache[token] = result
        return result

    def _decode_byte_chars(self, chars: list[str]) -> str:
        data = bytes(self.byte_decoder[char] for char in chars)
        return data.decode("utf-8", errors="replace")

    def _special_pattern(self) -> re.Pattern[str]:
        ordered = sorted(self.specials, key=len, reverse=True)
        return re.compile("|".join(re.escape(token) for token in ordered))


def parse_ids(value: str) -> list[int]:
    if not value.strip():
        return []
    return [int(item) for item in value.replace(",", " ").split()]


def load_messages(args: argparse.Namespace) -> list[dict[str, str]]:
    messages_json = args.messages_json
    if args.messages_json_b64.strip():
        messages_json = base64.b64decode(args.messages_json_b64).decode("utf-8")
    if messages_json.strip():
        raw = json.loads(messages_json)
        if not isinstance(raw, list):
            raise SystemExit("--messages-json must be a JSON list")
        messages: list[dict[str, str]] = []
        for item in raw:
            if not isinstance(item, dict):
                raise SystemExit("--messages-json items must be objects")
            role = str(item.get("role", "")).strip()
            content = str(item.get("content", ""))
            if not role:
                raise SystemExit("--messages-json item missing role")
            messages.append({"role": role, "content": content})
        return messages

    messages = []
    if args.system_text:
        messages.append({"role": "system", "content": args.system_text})
    messages.append({"role": "user", "content": args.user_text})
    return messages


def build_chat_prompt(system_text: str, user_text: str, assistant_prefix: str) -> str:
    messages = []
    if system_text:
        messages.append({"role": "system", "content": system_text})
    messages.append({"role": "user", "content": user_text})
    return build_chat_prompt_from_messages(messages, assistant_prefix)


def build_chat_prompt_from_messages(
    messages: list[dict[str, str]],
    assistant_prefix: str,
) -> str:
    role_tags = {
        "system": "<|system|>",
        "user": "<|user|>",
        "assistant": "<|assistant|>",
        "observation": "<|observation|>",
        "tool": "<|observation|>",
    }
    parts: list[str] = []
    for message in messages:
        role = message["role"].strip().lower()
        tag = role_tags.get(role)
        if tag is None:
            raise SystemExit(f"unsupported chat role: {message['role']}")
        parts.append(f"{tag}\n{message['content']}\n")
    parts.append(f"<|assistant|>\n{assistant_prefix}")
    return "".join(parts)


def streaming_decode_report(tokenizer: TokenizerSmoke, ids: list[int]) -> dict[str, object]:
    text = ""
    seen: list[int] = []
    deltas: list[dict[str, object]] = []
    for token_id in ids:
        seen.append(token_id)
        decoded = tokenizer.decode(seen)
        delta = decoded[len(text) :]
        deltas.append({"id": token_id, "delta": delta, "text": decoded})
        text = decoded
    return {
        "token_ids": ids,
        "text": text,
        "deltas": deltas,
        "stable": "".join(str(item["delta"]) for item in deltas) == text,
        "unicode_ok": "\ufffd" not in text,
    }


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    tokenizer = TokenizerSmoke.load(args.model_dir)
    messages = load_messages(args)
    prompt = build_chat_prompt_from_messages(messages, args.assistant_prefix)
    token_ids = tokenizer.encode(prompt)
    decoded = tokenizer.decode(token_ids)
    special_ids = {token: tokenizer.specials.get(token) for token in GLM52_SPECIALS}
    decoded_ids = parse_ids(args.ids)
    generated_ids = parse_ids(args.generated_ids)
    report = {
        "model_dir": str(args.model_dir),
        "messages": messages,
        "prompt_text": prompt,
        "token_ids": token_ids,
        "token_count": len(token_ids),
        "decoded_text": decoded,
        "roundtrip_ok": decoded == prompt,
        "special_ids": special_ids,
        "decoded_ids": [
            {
                "id": token_id,
                "piece": tokenizer.by_id.get(token_id),
                "text": tokenizer.decode([token_id]) if token_id in tokenizer.by_id else None,
                "known": token_id in tokenizer.by_id,
            }
            for token_id in decoded_ids
        ],
        "streaming": streaming_decode_report(tokenizer, generated_ids) if generated_ids else None,
    }
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"roundtrip_ok={report['roundtrip_ok']} token_count={report['token_count']}")
        print("token_ids=" + ",".join(str(token_id) for token_id in token_ids))
        print(decoded)
    return 0 if report["roundtrip_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
