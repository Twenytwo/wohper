#!/usr/bin/env python3
"""Build a compact PromptEnvelope for Hermes/Wohper.

No external dependencies. It intentionally favors verified local state over raw
chat history so long-running loops do not flood the model context.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


DEFAULT_CONFIG = Path("config/context.config.json")
WORD_RE = re.compile(r"[A-Za-z0-9_./:-]+")
SPLIT_RE = re.compile(r"[_./:|,;()\[\]{}<>=\"'`]+|(?:\s*->\s*)|(?:\s+-\s+)")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")
VERIFIED_MARKERS = (
    "passed",
    "ok",
    "riuscito",
    "validato",
    "validated",
    "smoke",
    "test",
    "log decisivo",
    "critical log",
    "verifica",
    "completato",
)
RISK_MARKERS = (
    "boundary",
    "blocco",
    "blocked",
    "rischio",
    "manca",
    "missing",
    "fallback",
    "placeholder",
    "non ancora",
    "todo",
    "next",
    "prossimo",
)
ASSUMPTION_MARKERS = ("assumption", "assunzione", "inferenza", "expected", "coerente")


@dataclass
class ContextBlock:
    source: str
    title: str
    text: str
    modified_epoch: float
    relevance: float = 0.0
    freshness: float = 0.0
    authority: float = 0.0
    risk: float = 0.0
    token_cost: int = 0
    score: float = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build compact context PromptEnvelope")
    parser.add_argument("--goal", required=True, help="Current task goal/query")
    parser.add_argument("--mode", default="build", choices=("build", "review", "plan", "execute"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--format", choices=("json", "markdown"), default="json")
    parser.add_argument("--request-id", default=None)
    parser.add_argument("--max-blocks", type=int, default=None)
    parser.add_argument("--max-prompt-tokens", type=int, default=None)
    parser.add_argument("--state-json-limit", type=int, default=12)
    parser.add_argument("--include-low-score", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def words(text: str) -> list[str]:
    """Return searchable tokens while preserving technical identifiers.

    Wohper state uses names such as GLOBAL-L3ATTN-SMOKE and embed_tokens.
    Keeping the full token is useful, but splitting it makes retrieval robust
    when the goal only names one segment of the identifier.
    """
    tokens: list[str] = []
    for raw in WORD_RE.findall(text):
        item = raw.lower().strip("-")
        if not item:
            continue
        tokens.append(item)
        for part in SPLIT_RE.split(item.replace("-", " ")):
            part = part.strip("- ")
            if len(part) >= 2:
                tokens.append(part)
    return tokens


def token_estimate(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))


def split_markdown(path: Path) -> list[ContextBlock]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8", errors="replace")
    modified = path.stat().st_mtime
    blocks: list[ContextBlock] = []
    current_title = path.name
    current: list[str] = []

    def flush() -> None:
        body = "\n".join(current).strip()
        if body:
            blocks.append(
                ContextBlock(
                    source=str(path),
                    title=current_title.strip(),
                    text=body,
                    modified_epoch=modified,
                )
            )

    for line in text.splitlines():
        heading = HEADING_RE.match(line)
        if heading:
            flush()
            current_title = heading.group(2).strip()
            current = [line]
        else:
            current.append(line)
    flush()
    return blocks


def split_json_state(path: Path) -> list[ContextBlock]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    title = str(data.get("loop") or data.get("objective") or path.stem)
    text = json.dumps(compact_json(data), ensure_ascii=False, indent=2)
    return [
        ContextBlock(
            source=str(path),
            title=title,
            text=text,
            modified_epoch=path.stat().st_mtime,
        )
    ]


def compact_json(value: Any, depth: int = 0) -> Any:
    if depth >= 4:
        return "..."
    if isinstance(value, dict):
        preferred = (
            "date",
            "loop",
            "status",
            "objective",
            "validated_path",
            "validation",
            "vps_smoke",
            "critical_logs",
            "boundaries",
            "blocked",
            "next_wall",
            "changed_files",
        )
        out: dict[str, Any] = {}
        for key in preferred:
            if key in value:
                out[key] = compact_json(value[key], depth + 1)
        for key, item in value.items():
            if key not in out and len(out) < 12:
                out[key] = compact_json(item, depth + 1)
        return out
    if isinstance(value, list):
        return [compact_json(item, depth + 1) for item in value[:12]]
    return value


def collect_blocks(config: dict[str, Any], state_json_limit: int) -> list[ContextBlock]:
    sources = config.get("sources", {})
    blocks: list[ContextBlock] = []
    notebook = Path(sources.get("project_notebook") or "PROJECT_NOTEBOOK.md")
    compact = Path(sources.get("compact_memory") or "state/compact_memory.md")
    state_dir = Path(sources.get("state_dir") or "state")

    blocks.extend(split_markdown(notebook))
    blocks.extend(split_markdown(compact))

    if state_dir.exists():
        json_files = sorted(
            (path for path in state_dir.glob("*.json") if not path.name.startswith("context_pack")),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )[:state_json_limit]
        for path in json_files:
            blocks.extend(split_json_state(path))

    obsidian = sources.get("obsidian_vault")
    if obsidian:
        vault = Path(obsidian)
        if vault.exists():
            for path in sorted(vault.rglob("*.md"), key=lambda item: item.stat().st_mtime, reverse=True)[:100]:
                blocks.extend(split_markdown(path))
    return blocks


def score_blocks(blocks: list[ContextBlock], goal: str, config: dict[str, Any]) -> list[ContextBlock]:
    scoring = config.get("scoring", {})
    relevance_weight = float(scoring.get("relevance_weight", 0.45))
    freshness_weight = float(scoring.get("freshness_weight", 0.2))
    authority_weight = float(scoring.get("authority_weight", 0.25))
    token_cost_weight = float(scoring.get("token_cost_weight", 0.1))
    goal_words = set(words(goal))
    goal_terms = extract_goal_terms(goal)
    technical_goal_terms = [term for term in goal_terms if is_technical_identifier(term)]
    primary_technical_term = technical_goal_terms[0] if technical_goal_terms else None
    now = time.time()

    for block in blocks:
        title_lower = block.title.lower()
        text_lower = f"{block.title}\n{block.text}".lower()
        block_words = set(words(text_lower))
        overlap = len(goal_words & block_words)
        exact_hits = sum(1 for term in goal_terms if term in text_lower)
        identifier_hits = sum(1 for term in goal_terms if len(term) >= 8 and term in text_lower)
        title_hits = sum(1 for term in goal_terms if term in title_lower)
        technical_title_hits = sum(
            1 for term in goal_terms if is_technical_identifier(term) and term in title_lower
        )
        technical_exact_hits = sum(1 for term in technical_goal_terms if term in text_lower)
        primary_hit = bool(primary_technical_term and primary_technical_term in text_lower)
        primary_title_hit = bool(primary_technical_term and primary_technical_term in title_lower)
        block.relevance = min(
            1.0,
            (overlap / max(1, len(goal_words)))
            + min(0.35, exact_hits * 0.06)
            + min(0.25, identifier_hits * 0.08)
            + min(0.3, title_hits * 0.08)
            + min(0.35, technical_title_hits * 0.18)
            + (0.22 if primary_title_hit else 0.0),
        )
        if primary_technical_term and not primary_hit:
            block.relevance *= 0.45
        elif technical_goal_terms and technical_exact_hits == 0:
            block.relevance *= 0.6
        age_days = max(0.0, (now - block.modified_epoch) / 86400.0)
        block.freshness = 1.0 / (1.0 + age_days)
        block.authority = marker_score(text_lower, VERIFIED_MARKERS)
        block.risk = marker_score(text_lower, RISK_MARKERS)
        block.token_cost = token_estimate(block.text)
        cost_penalty = min(1.0, block.token_cost / 1000.0)
        block.score = (
            relevance_weight * block.relevance
            + freshness_weight * block.freshness
            + authority_weight * block.authority
            + 0.08 * block.risk
            - token_cost_weight * cost_penalty
        )
    blocks.sort(key=lambda item: item.score, reverse=True)
    return blocks


def extract_goal_terms(goal: str) -> list[str]:
    terms: list[str] = []
    for token in WORD_RE.findall(goal.lower()):
        token = token.strip("-:.,;")
        if len(token) >= 4:
            terms.append(token)
    return list(dict.fromkeys(terms))


def is_technical_identifier(term: str) -> bool:
    return (
        len(term) >= 5
        and any(char.isdigit() for char in term)
        and any(char in term for char in "-_./:")
    )


def marker_score(text: str, markers: tuple[str, ...]) -> float:
    hits = sum(1 for marker in markers if marker in text)
    return min(1.0, hits / 3.0)


def select_blocks(
    blocks: list[ContextBlock],
    config: dict[str, Any],
    max_blocks: int | None,
    max_prompt_tokens: int | None,
    include_low_score: bool,
) -> list[ContextBlock]:
    cc = config.get("context_control", {})
    max_blocks = max_blocks or int(cc.get("max_retrieved_blocks", 8))
    max_prompt_tokens = max_prompt_tokens or int(cc.get("max_prompt_tokens", 1200))
    minimum = 0.0 if include_low_score else float(cc.get("minimum_block_score", 0.62))

    selected: list[ContextBlock] = []
    spent = 0
    for block in blocks:
        if block.score < minimum:
            continue
        if len(selected) >= max_blocks:
            break
        if spent + block.token_cost > max_prompt_tokens and selected:
            continue
        selected.append(block)
        spent += block.token_cost
        if spent >= max_prompt_tokens:
            break
    if not selected and blocks:
        spent = 0
        for block in blocks:
            if len(selected) >= max_blocks:
                break
            if spent + block.token_cost > max_prompt_tokens and selected:
                continue
            selected.append(block)
            spent += block.token_cost
            if spent >= max_prompt_tokens:
                break
    return selected


def build_envelope(goal: str, mode: str, config: dict[str, Any], selected: list[ContextBlock], request_id: str | None) -> dict[str, Any]:
    cc = config.get("context_control", {})
    envelope = {
        "request_id": request_id or f"context-{int(time.time())}",
        "goal": goal,
        "mode": mode,
        "budgets": {
            "max_prompt_tokens": int(cc.get("max_prompt_tokens", 1200)),
            "max_retrieved_blocks": int(cc.get("max_retrieved_blocks", 8)),
            "max_kv_tokens": int(cc.get("max_kv_tokens", 2048)),
            "summarize_after_tokens": int(cc.get("summarize_after_tokens", 1536)),
        },
        "context": {
            "verified": [],
            "assumptions": [],
            "open_risks": [],
            "retrieved_refs": [],
        },
        "next_action": infer_next_action(goal, selected),
    }
    for block in selected:
        item = block_to_item(block)
        envelope["context"]["retrieved_refs"].append(item)
        lower = block.text.lower()
        if marker_score(lower, VERIFIED_MARKERS) >= 0.34:
            envelope["context"]["verified"].append(summarize_block(block))
        if marker_score(lower, RISK_MARKERS) >= 0.34:
            envelope["context"]["open_risks"].append(summarize_block(block))
        if marker_score(lower, ASSUMPTION_MARKERS) >= 0.34:
            envelope["context"]["assumptions"].append(summarize_block(block))

    return envelope


def block_to_item(block: ContextBlock) -> dict[str, Any]:
    return {
        "source": block.source,
        "title": block.title,
        "score": round(block.score, 4),
        "token_cost": block.token_cost,
        "text": trim_text(block.text, 900),
    }


def summarize_block(block: ContextBlock) -> str:
    first = " ".join(line.strip() for line in block.text.splitlines() if line.strip())
    return f"{block.title}: {trim_text(first, 260)}"


def trim_text(text: str, limit: int) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def infer_next_action(goal: str, selected: list[ContextBlock]) -> str:
    ordered = blocks_ordered_for_next_action(goal, selected)
    for block in ordered:
        lines = block.text.splitlines()
        for index, line in enumerate(lines):
            lower = line.lower()
            if "next wall" in lower or "next_wall" in lower or "prossimo muro" in lower:
                cleaned = line.strip("- `: ")
                if len(cleaned) > 18 and not cleaned.lower().endswith(("next wall", "prossimo muro")):
                    return trim_text(cleaned, 240)
                for follow in lines[index + 1 : index + 8]:
                    follow = follow.strip("- `: ")
                    if len(follow) > 18:
                        return trim_text(follow, 240)
    for block in ordered:
        lines = block.text.splitlines()
        for index, line in enumerate(lines):
            lower = line.lower()
            if "next" in lower or "prossimo" in lower:
                cleaned = line.strip("- `: ")
                if len(cleaned) > 18:
                    return trim_text(cleaned, 240)
                for follow in lines[index + 1 : index + 8]:
                    follow = follow.strip("- `: ")
                    if len(follow) > 18:
                        return trim_text(follow, 240)
    return f"Proceed with goal using only verified retrieved context: {goal}"


def blocks_ordered_for_next_action(goal: str, selected: list[ContextBlock]) -> list[ContextBlock]:
    terms = extract_goal_terms(goal)

    def title_match_score(block: ContextBlock) -> tuple[int, int, float]:
        title = block.title.lower()
        exact = sum(1 for term in terms if term in title)
        technical = sum(1 for term in terms if is_technical_identifier(term) and term in title)
        return (technical, exact, block.score)

    return sorted(selected, key=title_match_score, reverse=True)


def render_markdown(envelope: dict[str, Any]) -> str:
    lines = [
        f"# PromptEnvelope {envelope['request_id']}",
        "",
        f"Goal: {envelope['goal']}",
        f"Mode: {envelope['mode']}",
        "",
        "## Verified",
    ]
    lines.extend(f"- {item}" for item in envelope["context"]["verified"][:8])
    lines.append("")
    lines.append("## Open Risks")
    lines.extend(f"- {item}" for item in envelope["context"]["open_risks"][:8])
    lines.append("")
    lines.append("## Retrieved Refs")
    for item in envelope["context"]["retrieved_refs"]:
        lines.append(f"- {item['title']} ({item['source']}, score={item['score']}, tokens={item['token_cost']})")
    lines.append("")
    lines.append(f"Next action: {envelope['next_action']}")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    blocks = collect_blocks(config, args.state_json_limit)
    scored = score_blocks(blocks, args.goal, config)
    selected = select_blocks(
        scored,
        config,
        args.max_blocks,
        args.max_prompt_tokens,
        args.include_low_score,
    )
    envelope = build_envelope(args.goal, args.mode, config, selected, args.request_id)

    if args.format == "json":
        payload = json.dumps(envelope, ensure_ascii=False, indent=2)
    else:
        payload = render_markdown(envelope)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload + ("\n" if not payload.endswith("\n") else ""), encoding="utf-8")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
