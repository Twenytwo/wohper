# DeepSeek-V4-Flash on a 16 GB RAM / 1 TB SSD PC - demo guide

Status 2026-07-07: the full model (284B total / 13B active, 44 layers,
11,008 expert shards, ~130 GB on disk at the time of this demo) runs
end to end on this machine (16 GB physical) and generates correct text:

| Prompt | Output |
|--------|--------|
| "Once upon a time" | ", there was a little girl" |
| "The capital of France is" | " Paris. The capital of France" |
| "2+2=" | "4" |
| "The capital of Italy is Rome. The capital of Germany is" | " Berlin." |

Parity with the official numpy reference: 4/4 argmax match on the
parity sweep plus the ctx4 gate (token 57502).

Note: this document describes the original single-shot demo path. The
current recommended entry points are the persistent server plus
`tools/wohper_cli.py` (terminal) or `http://127.0.0.1:8114/` (web) -
see the main README. The environment variables below still apply.

## Prerequisites (one-time)

1. Docker Desktop with WSL2.
2. `C:\Users\<user>\.wslconfig` (a FILE, not a folder):

       [wsl2]
       memory=12GB
       swap=4GB

   then `wsl --shutdown` and restart Docker Desktop. With 12 GB the
   container holds all 44 dense blocks (6.5 GB) plus the lm_head (1 GB)
   in RAM: the only NVMe reads per token are the selected experts.
3. The `zc-infer-dev` image built; the `zc-cargo-target` and
   `zc-cargo-registry` volumes.
4. `py -m pip install tokenizers` on the host.

## Usage

From the repo root (Git Bash or PowerShell):

    py tools/deepseek_chat_e2e.py --fast --prompt "Once upon a time" --max-new-tokens 8

`--fast` = full dense cache + lm_head in RAM + 10 I/O buffers.
Other flags: `--layer-end 5` (L0-L4 only, for ~20-30 s quick tests),
`--decode-log <file>` (re-decode an existing log).

## Key runtime variables

| Env | Demo value | Notes |
|-----|-----------|-------|
| ZC_DENSE_CACHE_MB | 6500 | all 44 dense layers in RAM |
| ZC_LMHEAD_CACHE | 1 | bf16 lm_head (1.06 GB) in RAM |
| ZC_IO_BUFFER_COUNT | 10 | 1 dense + 6 experts + prefetch + margin |
| ZC_SIDECAR_EXPERTS | all | merge of the 11,008 expert shards |
| ZC_ACTIVE_EXPERTS | 6 | top-6 (model configuration) |
| ZC_COMPUTE_THREADS | (auto) | default: cores - 2 |

## Memory budget (16 GB host / 12 GB WSL2)

Dense cache 6.5 GB + lm_head 1.06 GB + runtime/scratch/KV ~1.5 GB ≈
9 GB inside the container; Windows keeps the remaining 4 GB. No swap
under normal conditions.

## Known limits at the time of this demo (since superseded - see README)

- Speed: ~40-70 s/position full model at the time; the levers that
  followed (experts pack, batched prefill, speculative decoding,
  session KV reuse) brought a warm turn to ~67 s total.
- Contexts beyond ~2048 tokens: the top-k indexer (nb>512) and the DSA
  compressor rotate_fp4 path are not implemented; retrieval is
  field-validated up to 532 positions.
- The server handles one request at a time.
