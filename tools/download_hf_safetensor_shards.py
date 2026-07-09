#!/usr/bin/env python3
"""Download declared Hugging Face safetensors shards with resume and preflight."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def raw_url(repo_id: str, revision: str, path: str) -> str:
    return "https://huggingface.co/{}/resolve/{}/{}".format(
        repo_id,
        urllib.parse.quote(revision, safe=""),
        urllib.parse.quote(path, safe="/"),
    )


def declared_shards(index: dict[str, Any]) -> list[str]:
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise SystemExit("model.safetensors.index.json has no weight_map")
    return sorted({str(value) for value in weight_map.values()})


def write_report(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def remote_size(url: str, timeout_sec: int) -> int | None:
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=timeout_sec) as response:
        length = response.headers.get("Content-Length")
        return int(length) if length else None


def download_one(
    url: str,
    dest: Path,
    expected_size: int | None,
    chunk_bytes: int,
    timeout_sec: int,
    report_path: Path,
    payload: dict[str, Any],
    progress_interval_sec: int,
) -> dict[str, Any]:
    part = dest.with_suffix(dest.suffix + ".part")
    start = part.stat().st_size if part.exists() else 0
    if dest.exists() and expected_size is not None and dest.stat().st_size == expected_size:
        return {"path": str(dest), "status": "already_complete", "bytes": dest.stat().st_size}
    headers = {}
    if start:
        headers["Range"] = f"bytes={start}-"
    req = urllib.request.Request(url, headers=headers)
    downloaded = start
    started = time.time()
    last_report = started
    payload["active_shard"] = {
        "name": dest.name,
        "path": str(dest),
        "part_path": str(part),
        "bytes": downloaded,
        "expected_size": expected_size,
        "status": "opening",
    }
    write_report(report_path, payload)
    with urllib.request.urlopen(req, timeout=timeout_sec) as response:
        mode = "ab" if start else "wb"
        with part.open(mode) as handle:
            payload["active_shard"]["status"] = "downloading"
            write_report(report_path, payload)
            while True:
                chunk = response.read(chunk_bytes)
                if not chunk:
                    break
                handle.write(chunk)
                downloaded += len(chunk)
                now = time.time()
                if now - last_report >= progress_interval_sec:
                    payload["active_shard"].update(
                        {
                            "bytes": downloaded,
                            "elapsed_sec": round(now - started, 3),
                            "status": "downloading",
                        }
                    )
                    write_report(report_path, payload)
                    last_report = now
    if expected_size is not None and downloaded != expected_size:
        raise RuntimeError(f"{dest.name}: downloaded {downloaded}, expected {expected_size}")
    part.replace(dest)
    payload.pop("active_shard", None)
    return {
        "path": str(dest),
        "status": "downloaded",
        "bytes": dest.stat().st_size,
        "elapsed_sec": round(time.time() - started, 3),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Safe HF safetensors shard downloader")
    parser.add_argument("--repo-id", default="deepseek-ai/DeepSeek-V4-Flash")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--model-dir", type=Path, default=Path("models/deepseek-ai/DeepSeek-V4-Flash"))
    parser.add_argument("--out", type=Path, default=Path("state/deepseek_v4_flash_shard_download_2026-07-04.json"))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--min-free-after-gb", type=float, default=250.0)
    parser.add_argument("--chunk-mb", type=int, default=1)
    parser.add_argument("--timeout-sec", type=int, default=90)
    parser.add_argument("--progress-interval-sec", type=int, default=15)
    parser.add_argument("--limit", type=int, default=0, help="download only first N missing shards; 0 means all")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    index_path = args.model_dir / "model.safetensors.index.json"
    if not index_path.exists():
        raise SystemExit(f"missing {index_path}; run metadata-only sync first")
    index = load_json(index_path)
    shards = declared_shards(index)
    declared_total = int((index.get("metadata") or {}).get("total_size", 0) or 0)
    present = []
    missing = []
    for shard in shards:
        path = args.model_dir / shard
        if path.exists() and path.stat().st_size > 0:
            present.append(shard)
        else:
            missing.append(shard)
    selected = missing[: args.limit] if args.limit and args.limit > 0 else missing
    disk = shutil.disk_usage(args.model_dir if args.model_dir.exists() else Path("."))
    min_free_after = int(args.min_free_after_gb * 1024**3)
    present_bytes = sum((args.model_dir / shard).stat().st_size for shard in present)
    remaining_estimate = max(0, declared_total - present_bytes)
    feasible = disk.free - remaining_estimate >= min_free_after
    payload: dict[str, Any] = {
        "format": "hf-safetensors-shard-download",
        "version": 1,
        "repo_id": args.repo_id,
        "revision": args.revision,
        "model_dir": str(args.model_dir),
        "mode": "execute" if args.execute else "dry_run",
        "declared_shards": len(shards),
        "present_shards": len(present),
        "missing_shards": len(missing),
        "selected_count": len(selected),
        "declared_total_size_bytes": declared_total,
        "present_bytes": present_bytes,
        "remaining_estimate_bytes": remaining_estimate,
        "filesystem_free_bytes": disk.free,
        "filesystem_total_bytes": disk.total,
        "min_free_after_bytes": min_free_after,
        "download_feasible": feasible,
        "selected_shards": selected,
        "downloads": [],
        "errors": [],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if not args.execute:
        payload["status"] = "dry_run_ready" if feasible else "blocked_low_disk"
        write_report(args.out, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if feasible else 3
    if not feasible:
        payload["status"] = "blocked_low_disk"
        write_report(args.out, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 3
    args.model_dir.mkdir(parents=True, exist_ok=True)
    for shard in selected:
        url = raw_url(args.repo_id, args.revision, shard)
        try:
            payload["active_shard"] = {"name": shard, "status": "probing_remote_size"}
            write_report(args.out, payload)
            size = remote_size(url, args.timeout_sec)
            result = download_one(
                url,
                args.model_dir / shard,
                size,
                args.chunk_mb * 1024 * 1024,
                args.timeout_sec,
                args.out,
                payload,
                args.progress_interval_sec,
            )
            result["remote_size"] = size
            payload["downloads"].append(result)
        except Exception as exc:  # noqa: BLE001 - report exact resumable failure.
            payload.pop("active_shard", None)
            payload["errors"].append({"shard": shard, "error": str(exc)})
            break
        finally:
            write_report(args.out, payload)
    payload["status"] = "ready" if not payload["errors"] and len(payload["downloads"]) == len(selected) else "partial_or_blocked"
    final_present = [shard for shard in shards if (args.model_dir / shard).exists()]
    payload["present_shards_after"] = len(final_present)
    payload["missing_shards_after"] = len(shards) - len(final_present)
    write_report(args.out, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "ready" else 3


if __name__ == "__main__":
    raise SystemExit(main())
