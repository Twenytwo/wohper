#!/usr/bin/env python3
"""Stream-convert GLM-5.2 safetensors from Hugging Face into Wohper shards.

This tool is for machines that cannot store the full BF16 source checkpoint.
It uses Hugging Face HTTP range requests to read only the tensor bytes needed
for each output block, quantizes immediately, and writes only Wohper output:

  dense_core.bin on the master
  experts/layerN_expertM.zcblk locally or via HTTP PUT to a worker relay

It deliberately does not call `hf download` for the full model.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import struct
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import convert_safetensors as zc


DEFAULT_REPO_ID = "zai-org/GLM-5.2"
DEFAULT_REVISION = "main"
DEFAULT_METADATA_FILES = [
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
]

SAFETENSORS_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "U16": 2,
    "I16": 2,
    "F16": 2,
    "BF16": 2,
    "U32": 4,
    "I32": 4,
    "F32": 4,
    "U64": 8,
    "I64": 8,
    "F64": 8,
}

DTYPE_TO_ZC_NAME = {
    "BOOL": "bool",
    "U8": "uint8",
    "I8": "int8",
    "U16": "uint16",
    "I16": "int16",
    "F16": "float16",
    "BF16": "bfloat16",
    "U32": "uint32",
    "I32": "int32",
    "F32": "float32",
    "U64": "uint64",
    "I64": "int64",
    "F64": "float64",
}


def retry_delay(attempt: int, base_sleep: float) -> float:
    return min(30.0, base_sleep * (2 ** max(0, attempt - 1)))


def should_retry_http_error(exc: urllib.error.HTTPError) -> bool:
    return exc.code in {408, 409, 425, 429, 500, 502, 503, 504}


def urlopen_with_retries(
    request: urllib.request.Request,
    *,
    timeout: int,
    retries: int,
    retry_base_sleep: float,
    context: str,
) -> bytes:
    last_error: BaseException | None = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            last_error = exc
            if not should_retry_http_error(exc) or attempt >= retries:
                raise
        except urllib.error.URLError as exc:
            last_error = exc
            if attempt >= retries:
                raise
        except (TimeoutError, OSError) as exc:
            last_error = exc
            if attempt >= retries:
                raise
        delay = retry_delay(attempt + 1, retry_base_sleep)
        print(
            f"[retry] {context}: attempt {attempt + 1}/{retries + 1} failed: "
            f"{last_error}; sleeping {delay:.1f}s",
            flush=True,
        )
        time.sleep(delay)
    raise RuntimeError(f"{context} failed after retries: {last_error}")


@dataclass(frozen=True)
class TensorMeta:
    name: str
    filename: str
    dtype: str
    shape: tuple[int, ...]
    data_start: int
    data_end: int
    absolute_start: int
    absolute_end: int

    @property
    def byte_len(self) -> int:
        return self.data_end - self.data_start

    @property
    def element_size(self) -> int:
        return SAFETENSORS_DTYPE_BYTES[self.dtype]

    @property
    def element_count(self) -> int:
        total = 1
        for dim in self.shape:
            total *= int(dim)
        return total

    @property
    def runtime_bytes(self) -> int:
        return self.element_count * self.element_size

    def row_slice(self, row_start: int, row_limit: int) -> "TensorMeta":
        if len(self.shape) < 2:
            return self
        total_rows = int(self.shape[0])
        start = max(0, int(row_start))
        if start >= total_rows:
            raise ValueError(f"row start {start} outside tensor {self.name} rows={total_rows}")
        rows = total_rows - start if row_limit <= 0 else min(int(row_limit), total_rows - start)
        if start == 0 and rows >= total_rows:
            return self
        row_elements = 1
        for dim in self.shape[1:]:
            row_elements *= int(dim)
        byte_start = start * row_elements * self.element_size
        byte_len = rows * row_elements * self.element_size
        end = start + rows
        return TensorMeta(
            name=f"{self.name}.rows_{start}_{end}",
            filename=self.filename,
            dtype=self.dtype,
            shape=(rows, *self.shape[1:]),
            data_start=self.data_start + byte_start,
            data_end=self.data_start + byte_start + byte_len,
            absolute_start=self.absolute_start + byte_start,
            absolute_end=self.absolute_start + byte_start + byte_len,
        )


class HfRangeClient:
    def __init__(
        self,
        repo_id: str,
        revision: str,
        token: str | None,
        timeout: int = 60,
        retries: int = 4,
        retry_base_sleep: float = 1.0,
    ):
        self.repo_id = repo_id
        self.revision = revision
        self.token = token or os.environ.get("HF_TOKEN")
        self.timeout = timeout
        self.retries = max(0, retries)
        self.retry_base_sleep = max(0.0, retry_base_sleep)

    def file_url(self, path: str) -> str:
        quoted_repo = urllib.parse.quote(self.repo_id, safe="/")
        quoted_revision = urllib.parse.quote(self.revision, safe="")
        quoted_path = urllib.parse.quote(path, safe="/")
        return f"https://huggingface.co/{quoted_repo}/resolve/{quoted_revision}/{quoted_path}"

    def request(self, path: str, *, byte_range: tuple[int, int] | None = None) -> bytes:
        headers = {"User-Agent": "wohper-stream-converter/0.1"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        if byte_range:
            start, end = byte_range
            headers["Range"] = f"bytes={start}-{end}"
        req = urllib.request.Request(self.file_url(path), headers=headers)
        return urlopen_with_retries(
            req,
            timeout=self.timeout,
            retries=self.retries,
            retry_base_sleep=self.retry_base_sleep,
            context=f"HF range {path} {byte_range or 'full'}",
        )

    def download_file(self, path: str, out_path: Path, *, optional: bool = False) -> bool:
        try:
            data = self.request(path)
        except urllib.error.HTTPError as exc:
            if optional and exc.code == 404:
                return False
            raise
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(data)
        return True


class SafetensorsRemoteIndex:
    def __init__(self, client: HfRangeClient):
        self.client = client
        self._headers: dict[str, dict[str, Any]] = {}
        self._header_lens: dict[str, int] = {}

    def shard_header(self, filename: str) -> dict[str, Any]:
        if filename in self._headers:
            return self._headers[filename]
        raw_len = self.client.request(filename, byte_range=(0, 7))
        (header_len,) = struct.unpack("<Q", raw_len)
        header_raw = self.client.request(filename, byte_range=(8, 8 + header_len - 1))
        header = json.loads(header_raw.decode("utf-8"))
        self._headers[filename] = header
        self._header_lens[filename] = header_len
        return header

    def tensor_meta(self, ref: zc.TensorRef) -> TensorMeta:
        header = self.shard_header(ref.filename)
        if ref.name not in header:
            raise KeyError(f"tensor {ref.name} not found in {ref.filename}")
        entry = header[ref.name]
        dtype = str(entry["dtype"]).upper()
        if dtype not in SAFETENSORS_DTYPE_BYTES:
            raise ValueError(f"unsupported dtype {dtype} for tensor {ref.name}")
        data_start, data_end = (int(v) for v in entry["data_offsets"])
        header_len = self._header_lens[ref.filename]
        absolute_start = 8 + header_len + data_start
        absolute_end = 8 + header_len + data_end
        return TensorMeta(
            name=ref.name,
            filename=ref.filename,
            dtype=dtype,
            shape=tuple(int(dim) for dim in entry["shape"]),
            data_start=data_start,
            data_end=data_end,
            absolute_start=absolute_start,
            absolute_end=absolute_end,
        )

    def iter_tensor_f32(
        self,
        meta: TensorMeta,
        *,
        chunk_bytes: int,
    ) -> Iterator[Any]:
        import numpy as np

        item_size = meta.element_size
        chunk_bytes = max(item_size, (chunk_bytes // item_size) * item_size)
        cursor = meta.absolute_start
        while cursor < meta.absolute_end:
            end = min(meta.absolute_end, cursor + chunk_bytes) - 1
            raw = self.client.request(meta.filename, byte_range=(cursor, end))
            cursor = end + 1
            yield decode_safetensors_chunk(raw, meta.dtype, np)


def decode_safetensors_chunk(raw: bytes, dtype: str, np):
    if dtype == "BF16":
        values = np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16
        return values.view(np.float32)
    if dtype == "F16":
        return np.frombuffer(raw, dtype="<f2").astype(np.float32)
    if dtype == "F32":
        return np.frombuffer(raw, dtype="<f4").astype(np.float32)
    if dtype == "F64":
        return np.frombuffer(raw, dtype="<f8").astype(np.float32)
    if dtype in {"I8", "U8", "I16", "U16", "I32", "U32", "I64", "U64", "BOOL"}:
        dtype_map = {
            "BOOL": "?",
            "U8": "u1",
            "I8": "i1",
            "U16": "<u2",
            "I16": "<i2",
            "U32": "<u4",
            "I32": "<i4",
            "U64": "<u8",
            "I64": "<i8",
        }
        return np.frombuffer(raw, dtype=dtype_map[dtype]).astype(np.float32)
    raise ValueError(f"unsupported dtype: {dtype}")


def quantize_remote_tensor(
    remote: SafetensorsRemoteIndex,
    meta: TensorMeta,
    out,
    quant_format: int,
    chunk_bytes: int,
) -> tuple[int, float, float]:
    import numpy as np

    max_abs = 0.0
    for values in remote.iter_tensor_f32(meta, chunk_bytes=chunk_bytes):
        if values.size:
            local = float(np.max(np.abs(values)))
            if math.isfinite(local):
                max_abs = max(max_abs, local)
    if max_abs == 0.0 or not math.isfinite(max_abs):
        max_abs = 1.0

    written = 0
    if quant_format == zc.QUANT_INT8:
        scale = max_abs / 127.0
        for values in remote.iter_tensor_f32(meta, chunk_bytes=chunk_bytes):
            q = np.clip(np.rint(values / scale), -127, 127).astype(np.int8, copy=False)
            data = q.tobytes(order="C")
            out.write(data)
            written += len(data)
        return written, float(scale), 0.0

    if quant_format == zc.QUANT_INT4:
        scale = max_abs / 7.0
        carry: int | None = None
        for values in remote.iter_tensor_f32(meta, chunk_bytes=chunk_bytes):
            q = np.clip(np.rint(values / scale), -8, 7).astype(np.int16, copy=False)
            q = (q + 8).astype(np.uint8, copy=False)
            if carry is not None:
                first = int(q[0]) if q.size else 0
                out.write(bytes([carry | (first << 4)]))
                written += 1
                q = q[1:]
                carry = None
            if q.size >= 2:
                even_count = (q.size // 2) * 2
                low = q[:even_count:2]
                high = q[1:even_count:2] << 4
                packed = (low | high).astype(np.uint8, copy=False)
                data = packed.tobytes(order="C")
                out.write(data)
                written += len(data)
                q = q[even_count:]
            if q.size == 1:
                carry = int(q[0])
        if carry is not None:
            out.write(bytes([carry]))
            written += 1
        return written, float(scale), 8.0

    raise ValueError(f"unsupported quant format: {quant_format}")


def write_stream_block_payload(
    refs: list[zc.TensorRef],
    remote: SafetensorsRemoteIndex,
    block_tmp: Path,
    quant_format: int,
    chunk_bytes: int,
    *,
    max_tensors: int | None = None,
    global_row_limit: int = 0,
    global_row_start: int = 0,
) -> tuple[int, int, int]:
    if max_tensors is not None:
        refs = refs[:max_tensors]
    records: list[zc.BlockTensorRecord] = []
    names_blob = bytearray()
    tensor_items: list[tuple[zc.TensorRef, TensorMeta]] = []
    for ref in refs:
        meta = remote.tensor_meta(ref)
        if global_row_limit > 0 and zc.tensor_role_code(ref.name) in {
            zc.TENSOR_ROLE_EMBED,
            zc.TENSOR_ROLE_LM_HEAD,
        }:
            meta = meta.row_slice(global_row_start, global_row_limit)
        tensor_items.append((ref, meta))

    with block_tmp.open("wb") as out:
        out.write(b"\0" * zc.BLOCK_HEADER_STRUCT.size)
        record_table_offset = out.tell()
        out.write(b"\0" * (len(tensor_items) * zc.TENSOR_RECORD_STRUCT.size))

        name_offsets: list[int] = []
        for _, meta in tensor_items:
            name_offsets.append(len(names_blob))
            names_blob += zc.pack_tensor_name(meta.name)
        names_offset = out.tell()
        out.write(names_blob)

        for ref, meta in tensor_items:
            print(
                f"  tensor {meta.name} dtype={meta.dtype} shape={list(meta.shape)} "
                f"source_bytes={meta.byte_len:,}",
                flush=True,
            )
            data_offset = out.tell()
            data_bytes, scale, zero_point = quantize_remote_tensor(
                remote,
                meta,
                out,
                quant_format,
                chunk_bytes,
            )
            records.append(
                zc.BlockTensorRecord(
                    name=meta.name,
                    dtype_original=DTYPE_TO_ZC_NAME.get(meta.dtype, meta.dtype.lower()),
                    shape=meta.shape,
                    quant_format=quant_format,
                    data_offset=data_offset,
                    data_bytes=data_bytes,
                    runtime_bytes=meta.runtime_bytes,
                    scale=scale,
                    zero_point=zero_point,
                    tensor_role=zc.tensor_role_code(ref.name),
                )
            )

        out.seek(0)
        out.write(
            zc.BLOCK_HEADER_STRUCT.pack(
                zc.BLOCK_MAGIC,
                1,
                len(records),
                quant_format,
                0,
                record_table_offset,
                names_offset,
            )
        )

        out.seek(record_table_offset)
        for record, name_offset in zip(records, name_offsets):
            shape_offset = zc.append_shape_blob(out, record.shape)
            out.write(
                zc.TENSOR_RECORD_STRUCT.pack(
                    min(zc.DTYPE_CODES.get(record.dtype_original, 0), 65535),
                    record.quant_format,
                    len(record.shape),
                    record.tensor_role,
                    name_offset,
                    shape_offset,
                    record.data_offset,
                    record.data_bytes,
                    record.scale,
                    record.zero_point,
                )
            )
        out.seek(0, os.SEEK_END)
        payload_size = out.tell()

    return payload_size, sum(record.runtime_bytes for record in records), zc.checksum64_file(block_tmp)


def put_file(
    url: str,
    path: Path,
    token: str | None = None,
    *,
    retries: int = 4,
    retry_base_sleep: float = 1.0,
) -> None:
    headers = {
        "Content-Type": "application/octet-stream",
        "Content-Length": str(path.stat().st_size),
        "User-Agent": "wohper-stream-converter/0.1",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    last_error: BaseException | None = None
    for attempt in range(retries + 1):
        try:
            with path.open("rb") as handle:
                req = urllib.request.Request(url, data=handle, headers=headers, method="PUT")
                with urllib.request.urlopen(req, timeout=600) as response:
                    if response.status not in (200, 201, 204):
                        raise RuntimeError(f"PUT {url} returned HTTP {response.status}")
                    return
        except urllib.error.HTTPError as exc:
            last_error = exc
            if not should_retry_http_error(exc) or attempt >= retries:
                raise
        except (urllib.error.URLError, RuntimeError) as exc:
            last_error = exc
            if attempt >= retries:
                raise
        delay = retry_delay(attempt + 1, retry_base_sleep)
        print(
            f"[retry] PUT {url}: attempt {attempt + 1}/{retries + 1} failed: "
            f"{last_error}; sleeping {delay:.1f}s",
            flush=True,
        )
        time.sleep(delay)
    raise RuntimeError(f"PUT {url} failed after retries: {last_error}")


def http_json(url: str, timeout: int = 30, retries: int = 4, retry_base_sleep: float = 1.0) -> dict[str, Any]:
    req = urllib.request.Request(url, headers={"User-Agent": "wohper-stream-converter/0.1"})
    raw = urlopen_with_retries(
        req,
        timeout=timeout,
        retries=retries,
        retry_base_sleep=retry_base_sleep,
        context=f"GET {url}",
    )
    return json.loads(raw.decode("utf-8"))


def http_head_exists(
    url: str,
    *,
    expected_bytes: int | None = None,
    timeout: int = 30,
) -> bool:
    req = urllib.request.Request(url, headers={"User-Agent": "wohper-stream-converter/0.1"}, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            if response.status not in (200, 204):
                return False
            if expected_bytes is None:
                return True
            length = response.headers.get("Content-Length")
            return length is not None and int(length) == expected_bytes
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return False
        raise


def load_ledger(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"version": 1, "events": [], "completed": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def save_ledger(path: Path | None, ledger: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(ledger, indent=2), encoding="utf-8")
    tmp.replace(path)


def ledger_event(path: Path | None, ledger: dict[str, Any], kind: str, payload: dict[str, Any]) -> None:
    event = {"ts": time.time(), "kind": kind, **payload}
    ledger.setdefault("events", []).append(event)
    key = payload.get("key")
    if key:
        ledger.setdefault("completed", {})[str(key)] = event
    save_ledger(path, ledger)


def ledger_completed_payload(ledger: dict[str, Any], key: str) -> dict[str, Any] | None:
    event = ledger.get("completed", {}).get(key)
    if not isinstance(event, dict):
        return None
    return event


def required_expert_fields(payload: dict[str, Any] | None) -> bool:
    if not payload:
        return False
    return all(
        field in payload
        for field in ("disk_bytes", "payload_bytes", "dequant_bytes", "checksum")
    )


def local_free_bytes(path: Path) -> int:
    path.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(path).free


def preflight_space(args: argparse.Namespace) -> None:
    if args.min_free_gb_master:
        target_dir = args.out.parent
        free = local_free_bytes(target_dir)
        required = int(args.min_free_gb_master * 1024**3)
        print(f"master_free_bytes={free:,}", flush=True)
        if free < required:
            raise SystemExit(
                f"master free space too low: {free / 1024**3:.1f} GiB < "
                f"{args.min_free_gb_master:.1f} GiB required"
            )

    if args.worker_endpoint and args.min_free_gb_worker:
        stats_url = f"{args.worker_endpoint.rstrip('/')}/stats"
        stats = http_json(stats_url, retries=args.http_retries, retry_base_sleep=args.retry_base_sleep)
        free = int(stats.get("disk_free_bytes", 0))
        required = int(args.min_free_gb_worker * 1024**3)
        print(f"worker_free_bytes={free:,}", flush=True)
        if free < required:
            raise SystemExit(
                f"worker free space too low: {free / 1024**3:.1f} GiB < "
                f"{args.min_free_gb_worker:.1f} GiB required"
            )


def patch_remote_fetch_in_shard_index(shard_index: Path, endpoint: str | None) -> None:
    if not endpoint:
        return
    data = json.loads(shard_index.read_text(encoding="utf-8"))
    data["remote_fetch"] = {
        "enabled": True,
        "endpoint_template": endpoint.rstrip("/"),
        "path_template": "experts/layer{layer_id}_expert{expert_id}.zcblk",
    }
    tmp = shard_index.with_suffix(shard_index.suffix + ".part")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(shard_index)


def download_metadata(args: argparse.Namespace, client: HfRangeClient) -> None:
    args.metadata_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[str] = []
    for name in DEFAULT_METADATA_FILES:
        ok = client.download_file(name, args.metadata_dir / name, optional=True)
        if ok:
            downloaded.append(name)
    print("metadata_dir:", args.metadata_dir)
    print("downloaded:", ", ".join(downloaded))
    if "model.safetensors.index.json" not in downloaded:
        raise SystemExit("model.safetensors.index.json was not found; cannot plan conversion")


def load_weight_map_from_metadata(metadata_dir: Path) -> tuple[dict[str, str], dict[str, Any]]:
    index_path = metadata_dir / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"missing {index_path}; run --metadata-only first")
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    return payload["weight_map"], payload.get("metadata", {})


def build_plan_from_metadata(metadata_dir: Path) -> tuple[dict[str, str], dict[str, Any], dict[str, Any], zc.ConversionPlan]:
    weight_map, metadata = load_weight_map_from_metadata(metadata_dir)
    config_path = metadata_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    plan = zc.build_conversion_plan(weight_map)
    return weight_map, metadata, config, plan


def infer_glm_args(args: argparse.Namespace, config: dict[str, Any], plan: zc.ConversionPlan) -> tuple[list[int], int, int]:
    num_layers, experts_per_layer = zc.infer_arch_config(plan, args.num_layers, args.experts_per_layer)
    if not args.num_layers:
        args.num_layers = zc.first_int(config, "num_hidden_layers", "n_layer") or num_layers
    if not args.hidden_size:
        args.hidden_size = zc.first_int(config, "hidden_size", "n_embd") or 0
    if not args.heads:
        args.heads = zc.first_int(config, "num_attention_heads", "n_head") or 0
    if not args.kv_heads:
        args.kv_heads = zc.first_int(config, "num_key_value_heads", "multi_query_group_num") or args.heads
    if args.experts_per_layer is None:
        args.experts_per_layer = (
            zc.first_int(config, "n_routed_experts", "num_experts", "moe_num_experts")
            or experts_per_layer
        )
    selected_layers = zc.layers_for_range(args.num_layers, args.layer_range)
    return selected_layers, args.num_layers, args.experts_per_layer


def load_expert_plan(path: Path | None, selected_layers: list[int], max_expert_id: int) -> dict[int, list[int]]:
    if path is None:
        return {layer_id: list(range(max_expert_id)) for layer_id in selected_layers}
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_layers = payload.get("layers", payload)
    if not isinstance(raw_layers, dict):
        raise ValueError(f"{path} must contain a layers object or a layer-id mapping")

    plan: dict[int, list[int]] = {}
    selected = set(selected_layers)
    for key, value in raw_layers.items():
        layer_id = int(key)
        if layer_id not in selected:
            continue
        if not isinstance(value, list):
            raise ValueError(f"expert plan layer {layer_id} must be a list")
        expert_ids = sorted({int(item) for item in value})
        if not expert_ids:
            raise ValueError(f"expert plan layer {layer_id} is empty")
        for expert_id in expert_ids:
            if expert_id < 0 or expert_id >= max_expert_id:
                raise ValueError(
                    f"expert {expert_id} for layer {layer_id} outside architecture range 0..{max_expert_id - 1}"
                )
        plan[layer_id] = expert_ids

    missing = sorted(selected - set(plan))
    if missing:
        raise ValueError(f"expert plan missing selected layers: {missing[:12]}")
    return plan


def print_plan(args: argparse.Namespace) -> None:
    _, metadata, config, plan = build_plan_from_metadata(args.metadata_dir)
    selected_layers, num_layers_total, experts_per_layer = infer_glm_args(args, config, plan)
    expert_plan = load_expert_plan(args.expert_plan_json, selected_layers, experts_per_layer)
    zc.print_plan_summary(plan, num_layers_total, experts_per_layer, config)
    print(f"selected_layers={len(selected_layers)}")
    if selected_layers:
        print(f"selected_layer_range={selected_layers[0]},{selected_layers[-1] + 1}")
    selected_expert_count = sum(len(expert_ids) for expert_ids in expert_plan.values())
    max_selected_experts = max((len(expert_ids) for expert_ids in expert_plan.values()), default=0)
    print(f"selected_experts={selected_expert_count}")
    print(f"max_selected_experts_per_layer={max_selected_experts}")
    if args.expert_plan_json:
        print(f"expert_plan_json={args.expert_plan_json}")
    total_size = int(metadata.get("total_size", 0) or 0)
    if total_size:
        estimated_int4 = int(total_size * 0.28)
        estimated_int8 = int(total_size * 0.53)
        layer_fraction = len(selected_layers) / max(1, num_layers_total)
        selected_int4 = int(estimated_int4 * layer_fraction)
        selected_int8 = int(estimated_int8 * layer_fraction)
        print(f"source_total_size: {total_size:,} bytes ({total_size / (1024 ** 4):.3f} TiB)")
        print(f"rough_int4_output_estimate: {estimated_int4:,} bytes ({estimated_int4 / (1024 ** 3):.1f} GiB)")
        print(f"rough_int8_output_estimate: {estimated_int8:,} bytes ({estimated_int8 / (1024 ** 3):.1f} GiB)")
        print(
            f"rough_selected_int4_estimate: {selected_int4:,} bytes "
            f"({selected_int4 / (1024 ** 3):.1f} GiB)"
        )
        print(
            f"rough_selected_int8_estimate: {selected_int8:,} bytes "
            f"({selected_int8 / (1024 ** 3):.1f} GiB)"
        )
    if args.worker_endpoint:
        print(f"worker_endpoint: {args.worker_endpoint.rstrip('/')}")
        print(f"worker_policy: {args.worker_policy}")
    print("dry_run: no model bytes were downloaded or written")


def convert_stream(args: argparse.Namespace, client: HfRangeClient) -> None:
    preflight_space(args)
    _, metadata, config, plan = build_plan_from_metadata(args.metadata_dir)
    selected_layers, num_layers_total, experts_per_layer = infer_glm_args(args, config, plan)
    if not selected_layers:
        raise ValueError("no selected layers")
    expert_plan = load_expert_plan(args.expert_plan_json, selected_layers, experts_per_layer)
    selected_expert_total = sum(len(expert_ids) for expert_ids in expert_plan.values())
    max_selected_experts = max((len(expert_ids) for expert_ids in expert_plan.values()), default=0)
    if args.experts_only:
        convert_experts_only(
            args,
            metadata,
            plan,
            selected_layers,
            num_layers_total,
            experts_per_layer,
            expert_plan,
            selected_expert_total,
            max_selected_experts,
            client,
        )
        return

    quant_format = zc.QUANT_INT8 if args.quant == "int8" else zc.QUANT_INT4
    core_path = args.out
    experts_dir = args.experts_dir or (core_path.parent / "experts")
    shard_index = zc.sharded_manifest_path(core_path, args.shard_index_out)
    manifest_size = (
        zc.MANIFEST_HEADER_SIZE
        + len(selected_layers) * zc.LAYER_DESC_SIZE
        + selected_expert_total * zc.EXPERT_DESC_SIZE
    )

    core_path.parent.mkdir(parents=True, exist_ok=True)
    experts_dir.mkdir(parents=True, exist_ok=True)
    remote = SafetensorsRemoteIndex(client)
    tensor_regex = re.compile(args.tensor_regex) if args.tensor_regex else None
    layer_plans: list[zc.LayerPlan] = []
    expert_plans: list[zc.ExpertPlan] = []
    expert_shards: list[zc.ExpertShardPlan] = []
    ledger = load_ledger(args.resume_ledger)
    if args.skip_existing and core_path.exists() and shard_index.exists():
        finished = ledger_completed_payload(ledger, "conversion_finished")
        suffix = "ledger=finished" if finished else "ledger=missing-finished"
        print(
            f"skip_existing: conversion output already present ({suffix}); "
            f"core={core_path} shard_index={shard_index}",
            flush=True,
        )
        return

    ledger_event(
        args.resume_ledger,
        ledger,
        "conversion_started",
        {
            "key": "conversion_started",
            "repo_id": args.repo_id,
            "revision": args.revision,
            "quant": args.quant,
            "layer_range": list(args.layer_range) if args.layer_range else None,
            "worker_endpoint": args.worker_endpoint,
            "worker_policy": args.worker_policy,
            "global_row_start": args.global_row_start,
            "global_row_limit": args.global_row_limit,
            "expert_plan_json": str(args.expert_plan_json) if args.expert_plan_json else None,
            "selected_experts": selected_expert_total,
            "max_selected_experts_per_layer": max_selected_experts,
        },
    )

    with tempfile.TemporaryDirectory(prefix="zc_stream_convert_") as tmp_dir_text:
        tmp_dir = Path(tmp_dir_text)
        with core_path.open("wb") as out:
            manifest_offset = zc.reserve_header_and_manifest(out, manifest_size)

            for layer_id in selected_layers:
                print(f"[layer {layer_id}] dense", flush=True)
                dense_refs = list(plan.dense_by_layer.get(layer_id, []))
                if layer_id == 0 and args.pack_global_into_layer0:
                    dense_refs = list(plan.global_tensors) + dense_refs
                if tensor_regex:
                    dense_refs = [ref for ref in dense_refs if tensor_regex.search(ref.name)]
                if not dense_refs:
                    if args.allow_empty_dense:
                        dense_refs = []
                    else:
                        raise ValueError(f"layer {layer_id} has no dense tensors")

                dense_tmp = tmp_dir / f"dense_layer_{layer_id}.bin"
                _, dense_dequant, dense_checksum = write_stream_block_payload(
                    dense_refs,
                    remote,
                    dense_tmp,
                    quant_format,
                    args.chunk_mb * 1024 * 1024,
                    max_tensors=args.max_tensors_per_block,
                    global_row_limit=args.global_row_limit,
                    global_row_start=args.global_row_start,
                )
                dense_offset, dense_disk, dense_payload = zc.copy_aligned_block(out, dense_tmp)
                dense_tmp.unlink(missing_ok=True)
                ledger_event(
                    args.resume_ledger,
                    ledger,
                    "dense_converted",
                    {
                        "key": f"dense:{layer_id}",
                        "layer_id": layer_id,
                        "disk_bytes": dense_disk,
                        "payload_bytes": dense_payload,
                        "dequant_bytes": dense_dequant,
                        "checksum": dense_checksum,
                    },
                )

                first_expert_index = len(expert_plans)
                layer_expert_ids = expert_plan[layer_id]
                for expert_id in layer_expert_ids:
                    expert_name = f"layer{layer_id}_expert{expert_id}.zcblk"
                    expert_path = experts_dir / expert_name
                    put_url = f"{args.worker_endpoint.rstrip('/')}/experts/{expert_name}" if args.worker_endpoint else None
                    refs = plan.experts_by_layer.get(layer_id, {}).get(expert_id, [])
                    if tensor_regex:
                        refs = [ref for ref in refs if tensor_regex.search(ref.name)]
                    if not refs and not args.allow_missing_experts:
                        raise ValueError(f"missing tensors for layer {layer_id} expert {expert_id}")

                    expert_key = f"expert:{layer_id}:{expert_id}"
                    upload_key = f"expert:{layer_id}:{expert_id}:uploaded"
                    completed = ledger_completed_payload(ledger, expert_key)
                    uploaded = ledger_completed_payload(ledger, upload_key)
                    skip_source: str | None = None
                    remote_present = False
                    if args.skip_existing and required_expert_fields(completed):
                        expected_bytes = int(completed["disk_bytes"])
                        if expert_path.exists() and expert_path.stat().st_size == expected_bytes:
                            skip_source = "local"
                        elif (
                            put_url
                            and should_send_to_worker(args, layer_id, expert_id)
                            and uploaded
                        ):
                            try:
                                remote_present = http_head_exists(put_url, expected_bytes=expected_bytes)
                            except urllib.error.URLError as exc:
                                print(f"[layer {layer_id}] remote HEAD failed for expert {expert_id}: {exc}", flush=True)
                            if remote_present:
                                skip_source = "worker"

                    if skip_source:
                        print(f"[layer {layer_id}] expert {expert_id} skip_existing={skip_source}", flush=True)
                        expert_disk = int(completed["disk_bytes"])
                        expert_payload = int(completed["payload_bytes"])
                        expert_dequant = int(completed["dequant_bytes"])
                        expert_checksum = int(completed["checksum"])
                        ledger_event(
                            args.resume_ledger,
                            ledger,
                            "expert_skipped_existing",
                            {
                                "key": f"expert:{layer_id}:{expert_id}:skipped",
                                "layer_id": layer_id,
                                "expert_id": expert_id,
                                "source": skip_source,
                                "disk_bytes": expert_disk,
                            },
                        )
                    else:
                        print(f"[layer {layer_id}] expert {expert_id}", flush=True)
                        expert_tmp = tmp_dir / f"expert_layer_{layer_id}_{expert_id}.bin"
                        _, expert_dequant, expert_checksum = write_stream_block_payload(
                            refs,
                            remote,
                            expert_tmp,
                            quant_format,
                            args.chunk_mb * 1024 * 1024,
                            max_tensors=args.max_tensors_per_block,
                            global_row_limit=0,
                        )
                        expert_disk, expert_payload = zc.write_aligned_file(expert_path, expert_tmp)
                        expert_tmp.unlink(missing_ok=True)

                    if (
                        args.worker_endpoint
                        and should_send_to_worker(args, layer_id, expert_id)
                        and skip_source != "worker"
                    ):
                        if not expert_path.exists():
                            raise RuntimeError(
                                f"cannot upload missing local expert {expert_path}; "
                                "rerun without --skip-existing or restore the local shard"
                            )
                        print(f"[layer {layer_id}] uploading expert {expert_id} -> {put_url}", flush=True)
                        put_file(
                            put_url,
                            expert_path,
                            retries=args.http_retries,
                            retry_base_sleep=args.retry_base_sleep,
                        )
                        ledger_event(
                            args.resume_ledger,
                            ledger,
                            "expert_uploaded",
                            {
                                "key": f"expert:{layer_id}:{expert_id}:uploaded",
                                "layer_id": layer_id,
                                "expert_id": expert_id,
                                "url": put_url,
                                "disk_bytes": expert_disk,
                                "payload_bytes": expert_payload,
                                "checksum": expert_checksum,
                            },
                        )
                        if args.delete_after_worker_upload:
                            expert_path.unlink()
                    ledger_event(
                        args.resume_ledger,
                        ledger,
                        "expert_converted",
                        {
                            "key": f"expert:{layer_id}:{expert_id}",
                            "layer_id": layer_id,
                            "expert_id": expert_id,
                            "path": str(expert_path),
                            "disk_bytes": expert_disk,
                            "payload_bytes": expert_payload,
                            "dequant_bytes": expert_dequant,
                            "checksum": expert_checksum,
                        },
                    )

                    try:
                        expert_rel = expert_path.relative_to(core_path.parent)
                    except ValueError:
                        expert_rel = expert_path
                    expert_plans.append(
                        zc.ExpertPlan(
                            layer_id=layer_id,
                            expert_id=expert_id,
                            disk_offset=0,
                            disk_bytes=expert_disk,
                            payload_bytes=expert_payload,
                            dequant_bytes=expert_dequant,
                            quant_format=quant_format,
                            route_rank_hint=expert_id,
                            checksum=expert_checksum,
                        )
                    )
                    expert_shards.append(
                        zc.ExpertShardPlan(
                            layer_id=layer_id,
                            expert_id=expert_id,
                            path=str(expert_rel).replace("\\", "/"),
                            disk_bytes=expert_disk,
                            payload_bytes=expert_payload,
                            dequant_bytes=expert_dequant,
                            quant_format=quant_format,
                            checksum=expert_checksum,
                        )
                    )

                layer_plans.append(
                    zc.LayerPlan(
                        layer_id=layer_id,
                        dense_offset=dense_offset,
                        dense_disk_bytes=dense_disk,
                        dense_payload_bytes=dense_payload,
                        dense_dequant_bytes=dense_dequant,
                        first_expert_index=first_expert_index,
                        num_experts=len(layer_expert_ids),
                        checksum=dense_checksum,
                    )
                )

            file_size = out.tell()
            manifest_payload = zc.pack_manifest(layer_plans, expert_plans, quant_format)
            manifest_checksum = zc.checksum64_bytes(manifest_payload)
            header = zc.ENGINE_HEADER_STRUCT.pack(
                zc.MODEL_MAGIC,
                zc.FORMAT_VERSION,
                0,
                file_size,
                manifest_offset,
                len(manifest_payload),
                0,
                0,
                0,
                0,
                args.model_family,
                2,
                len(selected_layers),
                args.hidden_size,
                args.heads,
                args.kv_heads,
                experts_per_layer,
                args.active_experts,
                zc.ALIGN_2MB,
                quant_format,
                manifest_checksum,
                0,
            )
            out.seek(0)
            out.write(header)
            out.seek(manifest_offset)
            out.write(manifest_payload)

    zc.validate_output(core_path, layer_plans, [], manifest_offset)
    zc.write_debug_index(args.index_out or core_path.with_suffix(".index.json"), plan, layer_plans, expert_plans)
    zc.write_sharded_index(
        shard_index,
        core_path=core_path,
        experts_dir=experts_dir,
        selected_layers=selected_layers,
        num_layers_total=num_layers_total,
        experts_per_layer=max_selected_experts if args.expert_plan_json else experts_per_layer,
        layer_plans=layer_plans,
        expert_shards=expert_shards,
        quant_format=quant_format,
        metadata=metadata,
    )
    patch_remote_fetch_in_shard_index(shard_index, args.worker_endpoint)
    zc.print_sharded_summary(core_path, experts_dir, shard_index, layer_plans, expert_shards, metadata)
    ledger_event(
        args.resume_ledger,
        ledger,
        "conversion_finished",
        {
            "key": "conversion_finished",
            "core_path": str(core_path),
            "shard_index": str(shard_index),
            "layers": len(layer_plans),
            "experts": len(expert_shards),
        },
    )


def convert_experts_only(
    args: argparse.Namespace,
    metadata: dict[str, Any],
    plan: zc.ConversionPlan,
    selected_layers: list[int],
    num_layers_total: int,
    experts_per_layer: int,
    expert_plan: dict[int, list[int]],
    selected_expert_total: int,
    max_selected_experts: int,
    client: HfRangeClient,
) -> None:
    quant_format = zc.QUANT_INT8 if args.quant == "int8" else zc.QUANT_INT4
    core_path = args.out
    experts_dir = args.experts_dir or (core_path.parent / "experts")
    shard_index = zc.sharded_manifest_path(core_path, args.shard_index_out)
    experts_dir.mkdir(parents=True, exist_ok=True)
    shard_index.parent.mkdir(parents=True, exist_ok=True)
    remote = SafetensorsRemoteIndex(client)
    tensor_regex = re.compile(args.tensor_regex) if args.tensor_regex else None
    expert_shards: list[zc.ExpertShardPlan] = []
    ledger = load_ledger(args.resume_ledger)

    if args.skip_existing and shard_index.exists():
        finished = ledger_completed_payload(ledger, "conversion_finished")
        if finished:
            print(f"skip_existing: expert-only shard index already present; shard_index={shard_index}", flush=True)
            return

    ledger_event(
        args.resume_ledger,
        ledger,
        "conversion_started",
        {
            "key": "conversion_started",
            "mode": "experts_only",
            "repo_id": args.repo_id,
            "revision": args.revision,
            "quant": args.quant,
            "layer_range": list(args.layer_range) if args.layer_range else None,
            "worker_endpoint": args.worker_endpoint,
            "worker_policy": args.worker_policy,
            "expert_plan_json": str(args.expert_plan_json) if args.expert_plan_json else None,
            "selected_experts": selected_expert_total,
            "max_selected_experts_per_layer": max_selected_experts,
        },
    )

    with tempfile.TemporaryDirectory(prefix="zc_stream_convert_experts_") as tmp_dir_text:
        tmp_dir = Path(tmp_dir_text)
        for layer_id in selected_layers:
            for expert_id in expert_plan[layer_id]:
                expert_name = f"layer{layer_id}_expert{expert_id}.zcblk"
                expert_path = experts_dir / expert_name
                put_url = f"{args.worker_endpoint.rstrip('/')}/experts/{expert_name}" if args.worker_endpoint else None
                refs = plan.experts_by_layer.get(layer_id, {}).get(expert_id, [])
                if tensor_regex:
                    refs = [ref for ref in refs if tensor_regex.search(ref.name)]
                if not refs and not args.allow_missing_experts:
                    raise ValueError(f"missing tensors for layer {layer_id} expert {expert_id}")

                expert_key = f"expert:{layer_id}:{expert_id}"
                upload_key = f"expert:{layer_id}:{expert_id}:uploaded"
                completed = ledger_completed_payload(ledger, expert_key)
                uploaded = ledger_completed_payload(ledger, upload_key)
                skip_source: str | None = None
                remote_present = False
                if args.skip_existing and required_expert_fields(completed):
                    expected_bytes = int(completed["disk_bytes"])
                    if expert_path.exists() and expert_path.stat().st_size == expected_bytes:
                        skip_source = "local"
                    elif put_url and should_send_to_worker(args, layer_id, expert_id) and uploaded:
                        try:
                            remote_present = http_head_exists(put_url, expected_bytes=expected_bytes)
                        except urllib.error.URLError as exc:
                            print(f"[layer {layer_id}] remote HEAD failed for expert {expert_id}: {exc}", flush=True)
                        if remote_present:
                            skip_source = "worker"

                if skip_source:
                    print(f"[layer {layer_id}] expert {expert_id} skip_existing={skip_source}", flush=True)
                    expert_disk = int(completed["disk_bytes"])
                    expert_payload = int(completed["payload_bytes"])
                    expert_dequant = int(completed["dequant_bytes"])
                    expert_checksum = int(completed["checksum"])
                else:
                    print(f"[layer {layer_id}] expert {expert_id}", flush=True)
                    expert_tmp = tmp_dir / f"expert_layer_{layer_id}_{expert_id}.bin"
                    _, expert_dequant, expert_checksum = write_stream_block_payload(
                        refs,
                        remote,
                        expert_tmp,
                        quant_format,
                        args.chunk_mb * 1024 * 1024,
                        max_tensors=args.max_tensors_per_block,
                        global_row_limit=0,
                    )
                    expert_disk, expert_payload = zc.write_aligned_file(expert_path, expert_tmp)
                    expert_tmp.unlink(missing_ok=True)

                if (
                    args.worker_endpoint
                    and should_send_to_worker(args, layer_id, expert_id)
                    and skip_source != "worker"
                ):
                    if not expert_path.exists():
                        raise RuntimeError(f"cannot upload missing local expert {expert_path}")
                    print(f"[layer {layer_id}] uploading expert {expert_id} -> {put_url}", flush=True)
                    put_file(
                        put_url,
                        expert_path,
                        retries=args.http_retries,
                        retry_base_sleep=args.retry_base_sleep,
                    )
                    ledger_event(
                        args.resume_ledger,
                        ledger,
                        "expert_uploaded",
                        {
                            "key": f"expert:{layer_id}:{expert_id}:uploaded",
                            "layer_id": layer_id,
                            "expert_id": expert_id,
                            "url": put_url,
                            "disk_bytes": expert_disk,
                            "payload_bytes": expert_payload,
                            "checksum": expert_checksum,
                        },
                    )
                    if args.delete_after_worker_upload:
                        expert_path.unlink()

                ledger_event(
                    args.resume_ledger,
                    ledger,
                    "expert_converted",
                    {
                        "key": f"expert:{layer_id}:{expert_id}",
                        "layer_id": layer_id,
                        "expert_id": expert_id,
                        "path": str(expert_path),
                        "disk_bytes": expert_disk,
                        "payload_bytes": expert_payload,
                        "dequant_bytes": expert_dequant,
                        "checksum": expert_checksum,
                    },
                )
                try:
                    expert_rel = expert_path.relative_to(core_path.parent)
                except ValueError:
                    expert_rel = expert_path
                expert_shards.append(
                    zc.ExpertShardPlan(
                        layer_id=layer_id,
                        expert_id=expert_id,
                        path=str(expert_rel).replace("\\", "/"),
                        disk_bytes=expert_disk,
                        payload_bytes=expert_payload,
                        dequant_bytes=expert_dequant,
                        quant_format=quant_format,
                        checksum=expert_checksum,
                    )
                )

    zc.write_sharded_index(
        shard_index,
        core_path=core_path,
        experts_dir=experts_dir,
        selected_layers=selected_layers,
        num_layers_total=num_layers_total,
        experts_per_layer=max_selected_experts if args.expert_plan_json else experts_per_layer,
        layer_plans=[],
        expert_shards=expert_shards,
        quant_format=quant_format,
        metadata=metadata,
    )
    patch_remote_fetch_in_shard_index(shard_index, args.worker_endpoint)
    expert_bytes = sum(shard.disk_bytes for shard in expert_shards)
    print(f"expert_only: true")
    print(f"experts_dir: {experts_dir}")
    print(f"expert_files: {len(expert_shards)}")
    print(f"expert_disk_total: {expert_bytes:,} bytes ({expert_bytes / (1024 ** 3):.3f} GiB)")
    print(f"shard_index: {shard_index}")
    ledger_event(
        args.resume_ledger,
        ledger,
        "conversion_finished",
        {
            "key": "conversion_finished",
            "mode": "experts_only",
            "shard_index": str(shard_index),
            "layers": len(selected_layers),
            "experts": len(expert_shards),
        },
    )


def should_send_to_worker(args: argparse.Namespace, layer_id: int, expert_id: int) -> bool:
    if args.worker_policy == "none":
        return False
    if args.worker_policy == "all":
        return True
    if args.worker_policy == "odd-experts":
        return expert_id % 2 == 1
    if args.worker_policy == "upper-half-experts":
        return expert_id >= max(1, args.experts_per_layer // 2)
    if args.worker_policy == "odd-layers":
        return layer_id % 2 == 1
    return False


def parse_layer_range(value: str | None) -> tuple[int, int] | None:
    return zc.parse_layer_range(value)


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream-convert GLM-5.2 from HF without full source download")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--metadata-dir", type=Path, default=Path("models/zai-org/GLM-5.2"))
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Alias for --plan-only with selected-space estimates.")
    parser.add_argument("--out", type=Path, default=Path("models/wohper/GLM-5.2.INT4/dense_core.bin"))
    parser.add_argument("--experts-dir", type=Path, default=None)
    parser.add_argument("--shard-index-out", type=Path, default=None)
    parser.add_argument("--index-out", type=Path, default=None)
    parser.add_argument("--quant", choices=("int4", "int8"), default="int4")
    parser.add_argument("--layer-range", type=parse_layer_range, default=None, metavar="START,END")
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--experts-per-layer", type=int, default=None)
    parser.add_argument(
        "--expert-plan-json",
        type=Path,
        default=None,
        help="Optional mapping of physical layer ids to exact expert ids to materialize.",
    )
    parser.add_argument("--active-experts", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=0)
    parser.add_argument("--heads", type=int, default=0)
    parser.add_argument("--kv-heads", type=int, default=0)
    parser.add_argument("--model-family", type=int, default=1)
    parser.add_argument("--pack-global-into-layer0", action="store_true")
    parser.add_argument("--allow-missing-experts", action="store_true")
    parser.add_argument("--allow-empty-dense", action="store_true")
    parser.add_argument("--experts-only", action="store_true", help="Only materialize selected expert shards and shard index.")
    parser.add_argument("--tensor-regex", default=None, help="Optional regex to select tensors for smoke tests.")
    parser.add_argument("--max-tensors-per-block", type=int, default=None)
    parser.add_argument(
        "--global-row-limit",
        type=int,
        default=0,
        help="For smoke tests, keep only the first N rows of global embed/lm_head tensors.",
    )
    parser.add_argument(
        "--global-row-start",
        type=int,
        default=0,
        help="First vocab row to keep when --global-row-limit slices global embed/lm_head tensors.",
    )
    parser.add_argument("--chunk-mb", type=int, default=64)
    parser.add_argument("--worker-endpoint", default=None, help="Relay endpoint, e.g. http://127.0.0.1:9101")
    parser.add_argument(
        "--worker-policy",
        choices=("none", "all", "odd-experts", "upper-half-experts", "odd-layers"),
        default="none",
    )
    parser.add_argument("--delete-after-worker-upload", action="store_true")
    parser.add_argument("--resume-ledger", type=Path, default=None)
    parser.add_argument("--skip-existing", action="store_true", help="Reuse completed expert shards from the resume ledger.")
    parser.add_argument("--min-free-gb-master", type=float, default=0.0)
    parser.add_argument("--min-free-gb-worker", type=float, default=0.0)
    parser.add_argument("--http-retries", type=int, default=4)
    parser.add_argument("--retry-base-sleep", type=float, default=1.0)
    return parser.parse_args(list(argv))


def main(argv: Iterable[str] = sys.argv[1:]) -> int:
    args = parse_args(argv)
    client = HfRangeClient(
        args.repo_id,
        args.revision,
        args.hf_token,
        retries=args.http_retries,
        retry_base_sleep=args.retry_base_sleep,
    )
    if args.metadata_only:
        download_metadata(args, client)
        return 0
    if args.plan_only or args.dry_run:
        print_plan(args)
        return 0
    convert_stream(args, client)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
