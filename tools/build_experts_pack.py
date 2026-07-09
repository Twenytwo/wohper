#!/usr/bin/env python3
"""T1b: consolidate the 11k per-expert .zcblk shard files into a single
2MB-aligned pack file + JSON index, deleting source shards chunk by chunk
(bounded extra disk usage). Meant to run INSIDE the container against the
fast volume:

  docker exec zc-chat python3 tools/build_experts_pack.py --base /model-fast
"""
import argparse
import hashlib
import json
import os
import random
import sys


def md5_bytes(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="/model-fast")
    parser.add_argument("--chunk-gib", type=float, default=12.0)
    parser.add_argument("--keep-shards", action="store_true", help="do not delete source shards")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume an interrupted build: deleted shards form a verified "
        "prefix, so truncate the pack at the first still-existing shard "
        "and continue from there",
    )
    args = parser.parse_args()

    base = args.base
    shards = json.load(open(os.path.join(base, "dense_core.shards.json")))
    experts = shards["experts"]
    pack_path = os.path.join(base, "dense_core.experts_pack.bin")
    index_path = os.path.join(base, "dense_core.experts_pack.index.json")
    if os.path.exists(index_path):
        print("index already exists - refusing to overwrite", file=sys.stderr)
        return 2

    chunk_limit = int(args.chunk_gib * 2**30)
    entries = []
    offset = 0
    pending = []  # (source_path, offset, disk_bytes)
    chunk_bytes = 0
    verified = 0
    resume_skip = 0

    if args.resume and os.path.exists(pack_path):
        # Shards are deleted only AFTER their chunk verified, in file order:
        # the deleted ones are a verified prefix of the pack. Rebuild their
        # entries, truncate the unverified tail, and append from there.
        for expert in experts:
            source_path = os.path.join(base, expert["path"])
            if os.path.exists(source_path):
                break
            entries.append(
                {
                    "layer_id": expert["layer_id"],
                    "expert_id": expert["expert_id"],
                    "offset": offset,
                    "disk_bytes": expert["disk_bytes"],
                    "payload_bytes": expert["payload_bytes"],
                }
            )
            offset += expert["disk_bytes"]
            resume_skip += 1
        current = os.path.getsize(pack_path)
        if current < offset:
            print(
                f"pack smaller than verified prefix ({current} < {offset}) - aborting",
                file=sys.stderr,
            )
            return 3
        with open(pack_path, "r+b") as pack_fix:
            pack_fix.truncate(offset)
        print(
            f"resume: {resume_skip} experts already packed+verified "
            f"({offset / 2**30:.1f} GiB), truncated unverified tail",
            flush=True,
        )

    def flush_chunk(pack):
        nonlocal pending, chunk_bytes, verified
        if not pending:
            return
        pack.flush()
        os.fsync(pack.fileno())
        # Verify one random shard of the chunk against the pack bytes
        # BEFORE deleting anything.
        sample_path, sample_offset, sample_bytes = random.choice(pending)
        with open(sample_path, "rb") as source:
            source_md5 = md5_bytes(source.read())
        with open(pack_path, "rb") as pack_read:
            pack_read.seek(sample_offset)
            pack_md5 = md5_bytes(pack_read.read(sample_bytes))
        if source_md5 != pack_md5:
            raise RuntimeError(f"verification failed for {sample_path} at {sample_offset}")
        verified += 1
        if not args.keep_shards:
            for path, _, _ in pending:
                os.remove(path)
        print(f"progress {offset / 2**30:.1f} GiB packed, chunk verified ({verified})", flush=True)
        pending = []
        chunk_bytes = 0

    open_mode = "ab" if (args.resume and resume_skip > 0) else "wb"
    with open(pack_path, open_mode) as pack:
        for expert in experts[resume_skip:]:
            source_path = os.path.join(base, expert["path"])
            with open(source_path, "rb") as source:
                data = source.read()
            if len(data) != expert["disk_bytes"]:
                raise RuntimeError(
                    f"size mismatch {source_path}: {len(data)} != {expert['disk_bytes']}"
                )
            pack.write(data)
            entries.append(
                {
                    "layer_id": expert["layer_id"],
                    "expert_id": expert["expert_id"],
                    "offset": offset,
                    "disk_bytes": expert["disk_bytes"],
                    "payload_bytes": expert["payload_bytes"],
                }
            )
            pending.append((source_path, offset, expert["disk_bytes"]))
            offset += len(data)
            chunk_bytes += len(data)
            if chunk_bytes >= chunk_limit:
                flush_chunk(pack)
        flush_chunk(pack)

    index = {
        "format": "wohper-experts-pack",
        "version": 1,
        "pack_file": "dense_core.experts_pack.bin",
        "entries": entries,
    }
    with open(index_path, "w") as handle:
        json.dump(index, handle)
    print(f"DONE pack_bytes={offset} experts={len(entries)} verified_chunks={verified}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
