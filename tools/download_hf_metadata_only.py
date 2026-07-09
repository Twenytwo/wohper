#!/usr/bin/env python3
"""Download Hugging Face metadata files while refusing weight payloads."""

from __future__ import annotations

import argparse
import fnmatch
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_INCLUDE = [
    "README.md",
    "LICENSE*",
    "config.json",
    "generation_config.json",
    "tokenizer*",
    "special_tokens_map.json",
    "model.safetensors.index.json",
    "encoding/**",
    "inference/**",
]

HARD_EXCLUDE = [
    "*.safetensors",
    "*.bin",
    "*.pt",
    "*.pth",
    "*.gguf",
    "*.onnx",
    "*.tar",
    "*.zip",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Safe metadata-only Hugging Face downloader")
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--max-file-bytes", type=int, default=32 * 1024 * 1024)
    parser.add_argument("--max-total-bytes", type=int, default=256 * 1024 * 1024)
    return parser.parse_args()


def hf_api_url(repo_id: str, revision: str) -> str:
    quoted_repo = urllib.parse.quote(repo_id, safe="/")
    quoted_rev = urllib.parse.quote(revision, safe="")
    return f"https://huggingface.co/api/models/{quoted_repo}/tree/{quoted_rev}?recursive=1"


def hf_raw_url(repo_id: str, revision: str, path: str) -> str:
    return "https://huggingface.co/{}/resolve/{}/{}".format(
        repo_id,
        urllib.parse.quote(revision, safe=""),
        urllib.parse.quote(path, safe="/"),
    )


def fetch_json(url: str) -> Any:
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def download_bytes(url: str, max_bytes: int) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as response:
        content_length = response.headers.get("Content-Length")
        if content_length and int(content_length) > max_bytes:
            raise ValueError(f"remote file too large: {content_length} > {max_bytes}")
        data = response.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError(f"download exceeded max bytes: {len(data)} > {max_bytes}")
    return data


def matches_any(path: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(path, pattern) for pattern in patterns)


def main() -> int:
    args = parse_args()
    tree_url = hf_api_url(args.repo_id, args.revision)
    entries = fetch_json(tree_url)
    if not isinstance(entries, list):
        raise SystemExit("unexpected Hugging Face tree response")

    selected = []
    skipped = []
    for entry in entries:
        path = str(entry.get("path") or entry.get("rfilename") or "")
        entry_type = str(entry.get("type") or "")
        size = int(entry.get("size") or 0)
        if not path or entry_type == "directory":
            continue
        if matches_any(path, HARD_EXCLUDE):
            skipped.append({"path": path, "reason": "hard_excluded_weight_or_archive", "size": size})
            continue
        if not matches_any(path, DEFAULT_INCLUDE):
            skipped.append({"path": path, "reason": "not_metadata_allowlisted", "size": size})
            continue
        if size > args.max_file_bytes:
            skipped.append({"path": path, "reason": "too_large_for_metadata", "size": size})
            continue
        selected.append({"path": path, "size": size})

    total_planned = sum(item["size"] for item in selected)
    if total_planned > args.max_total_bytes:
        raise SystemExit(
            f"metadata selection too large: {total_planned} > {args.max_total_bytes}"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    downloaded = []
    errors = []
    total_downloaded = 0
    for item in selected:
        path = item["path"]
        try:
            data = download_bytes(hf_raw_url(args.repo_id, args.revision, path), args.max_file_bytes)
        except (urllib.error.URLError, ValueError, TimeoutError) as exc:
            errors.append({"path": path, "error": str(exc)})
            continue
        total_downloaded += len(data)
        if total_downloaded > args.max_total_bytes:
            errors.append({"path": path, "error": "total download limit exceeded"})
            break
        out_path = args.out_dir / path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(data)
        downloaded.append({"path": path, "bytes": len(data)})

    payload = {
        "format": "hf-metadata-only-download",
        "version": 1,
        "repo_id": args.repo_id,
        "revision": args.revision,
        "out_dir": str(args.out_dir),
        "status": "ready" if downloaded and not errors else "blocked_errors",
        "selected_count": len(selected),
        "downloaded_count": len(downloaded),
        "downloaded_bytes": total_downloaded,
        "max_file_bytes": args.max_file_bytes,
        "max_total_bytes": args.max_total_bytes,
        "downloaded": downloaded,
        "errors": errors,
        "skipped_count": len(skipped),
        "skipped_sample": skipped[:32],
        "hard_exclude": HARD_EXCLUDE,
        "include": DEFAULT_INCLUDE,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "ready" else 3


if __name__ == "__main__":
    raise SystemExit(main())
