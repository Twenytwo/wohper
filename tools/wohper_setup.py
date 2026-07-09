#!/usr/bin/env python3
"""Wohper setup: detect the machine, recommend tuning, fetch the model.

One command for a fresh clone:

  py -X utf8 tools/wohper_setup.py                # detect + report + advice
  py -X utf8 tools/wohper_setup.py --fetch-model  # + download DeepSeek-V4-Flash
                                                  #   shards from HuggingFace

Detection covers RAM, CPU cores, free disk, GPU (nvidia-smi) and Docker.
The chat launcher (tools/wohper_cli.py) re-runs the same RAM detection at
every start and sizes the engine knobs automatically - this script exists
to see the picture before committing to the ~160 GB one-time download, and
to run that download in a resumable, disk-guarded way.

The download uses tools/download_hf_safetensor_shards.py (resumable,
refuses to fill the disk). The conversion afterwards is one command and is
printed at the end. Low-disk machines can download and convert in batches:
both tools accept subsets (--limit / --layer-range); see the README.
"""
from __future__ import annotations

import argparse
import ctypes
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
HF_REPO_ID = "deepseek-ai/DeepSeek-V4-Flash"
MODEL_DIR = REPO / "models" / "deepseek-ai" / "DeepSeek-V4-Flash"

DOWNLOAD_GB = 160        # official safetensors (natively quantized), one-time:
                         # 46 shards, 159.6 GB measured from the HF index
CONVERTED_GB = 174       # dense core + experts pack after conversion


def host_ram_gb() -> int:
    """Physical RAM in GiB, no external dependencies."""
    if os.name == "nt":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]
        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(MemoryStatus)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
        return round(status.ullTotalPhys / (1024 ** 3))
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return round(pages * page_size / (1024 ** 3))
    except (ValueError, OSError):
        return 0


def tuning_for(ram_gb: int) -> dict:
    """Engine knob tier by physical RAM. The 16 GB tier is the measured
    baseline; larger tiers follow the measured bottlenecks (page cache
    first, then the expert RAM cache that thrashes below ~24 GB and pays
    off above it)."""
    if ram_gb >= 64:
        return dict(wsl_gb=max(24, ram_gb - 12), dense_mb=6500,
                    expert_ram_mb=32000, kv_slots=2048, session_slots=4)
    if ram_gb >= 32:
        return dict(wsl_gb=26, dense_mb=6500,
                    expert_ram_mb=8000, kv_slots=2048, session_slots=3)
    if ram_gb >= 24:
        return dict(wsl_gb=18, dense_mb=6500,
                    expert_ram_mb=0, kv_slots=1024, session_slots=2)
    return dict(wsl_gb=12, dense_mb=6100,
                expert_ram_mb=0, kv_slots=1024, session_slots=2)


def probe(command: list[str]) -> str | None:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=20)
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def report() -> dict:
    ram = host_ram_gb()
    cores = os.cpu_count() or 0
    free_gb = round(shutil.disk_usage(REPO).free / (1024 ** 3))
    gpu = probe(["nvidia-smi", "--query-gpu=name,memory.total",
                 "--format=csv,noheader"])
    docker = probe(["docker", "--version"])
    tune = tuning_for(ram)

    print("Wohper setup - machine report")
    print(f"  RAM:    {ram} GB")
    print(f"  CPU:    {cores} logical cores")
    print(f"  Disk:   {free_gb} GB free on the repo drive")
    print(f"  GPU:    {gpu or 'none detected (CPU path only)'}")
    print(f"  Docker: {docker or 'NOT FOUND - install Docker Desktop first'}")
    print()
    print(f"Auto-tuning for the {ram} GB tier (applied automatically at every")
    print("chat start; ZC_* environment variables you set yourself always win):")
    print(f"  dense RAM cache      {tune['dense_mb']} MB")
    expert = tune["expert_ram_mb"]
    print(f"  expert RAM cache     {'off (thrashes below ~24 GB, measured)' if expert == 0 else f'{expert} MB'}")
    print(f"  KV slots             {tune['kv_slots']}")
    print(f"  conversation slots   {tune['session_slots']}")
    print()
    if os.name == "nt":
        print(f"WSL2 memory: give the VM ~{tune['wsl_gb']} GB. Put this in")
        print(f"C:\\Users\\<you>\\.wslconfig (a file), then `wsl --shutdown`")
        print("and restart Docker Desktop:")
        print()
        print("    [wsl2]")
        print(f"    memory={tune['wsl_gb']}GB")
        print("    swap=4GB")
        print()
    print("One-time model download: "
          f"~{DOWNLOAD_GB} GB from huggingface.co/{HF_REPO_ID},")
    print(f"~{CONVERTED_GB} GB on disk after conversion. Best case needs "
          f"~{DOWNLOAD_GB + CONVERTED_GB + 40} GB free during the process;")
    print("low-disk machines can go in batches (see the README).")
    return {"ram_gb": ram, "free_gb": free_gb, "tune": tune}


def fetch_metadata() -> int:
    """Fetches the small files first (tokenizer, config, the safetensors
    index that lists every shard): the shard downloader and the chat
    tooling both need them, and a fresh clone has none (models/ is not in
    git). A few MB, safe to re-run."""
    index = MODEL_DIR / "model.safetensors.index.json"
    tokenizer = MODEL_DIR / "tokenizer.json"
    if index.exists() and tokenizer.exists():
        print("Metadata already present (index + tokenizer).")
        return 0
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, "-X", "utf8",
        str(REPO / "tools" / "download_hf_metadata_only.py"),
        "--repo-id", HF_REPO_ID,
        "--out-dir", str(MODEL_DIR),
        "--report", str(MODEL_DIR / "metadata_download_report.json"),
    ]
    print("Downloading model metadata (index, tokenizer, config - a few MB):")
    print("  " + " ".join(command))
    result = subprocess.run(command)
    if result.returncode != 0:
        return result.returncode
    if not index.exists():
        print("Metadata download finished but the safetensors index is "
              "missing - check the report JSON.")
        return 1
    return 0


def fetch_model(info: dict, limit: int | None) -> int:
    free_gb = info["free_gb"]
    if limit is None and free_gb < DOWNLOAD_GB + 40:
        print()
        print(f"Refusing the full download: only {free_gb} GB free, "
              f"~{DOWNLOAD_GB + 40} GB needed.")
        print("Either free space, or download in batches with "
              "--fetch-limit N (shards are resumable),")
        print("converting and deleting each batch before the next - "
              "see the README section 'Bringing it to life'.")
        return 1
    status = fetch_metadata()
    if status != 0:
        return status
    command = [
        sys.executable, "-X", "utf8",
        str(REPO / "tools" / "download_hf_safetensor_shards.py"),
        "--repo-id", HF_REPO_ID,
        "--model-dir", str(MODEL_DIR),
        "--min-free-after-gb", "30",
        "--execute",
    ]
    if limit:
        command += ["--limit", str(limit)]
    print()
    print("Downloading (resumable - re-run this command after any interruption):")
    print("  " + " ".join(command))
    result = subprocess.run(command)
    if result.returncode != 0:
        return result.returncode
    print()
    print("Download step finished. Next commands (see the README for details):")
    print(f"  py -X utf8 tools/stream_convert_deepseek_v4.py --model-dir {MODEL_DIR} "
          "--out models/wohper/DeepSeek-V4-Flash.RAW --execute")
    print(f"  py -X utf8 tools/build_deepseek_v4_tensor_index.py --model models/wohper/DeepSeek-V4-Flash.RAW")
    print("  (copy onto the ext4 Docker volume, then) "
          "py -X utf8 tools/build_experts_pack.py --model-dir /model-fast --resume")
    print("  py -X utf8 tools/wohper_cli.py")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fetch-model", action="store_true",
                        help="download the DeepSeek-V4-Flash shards from HuggingFace")
    parser.add_argument("--fetch-limit", type=int, default=None,
                        help="download only N shards (batch mode for low-disk machines)")
    args = parser.parse_args()
    info = report()
    if args.fetch_model:
        return fetch_model(info, args.fetch_limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
