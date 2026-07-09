# Wohper Compute Kernel Plan

## Scope

`core/compute.rs` is the first compute boundary after the I/O subsystem.
It consumes compressed bytes already loaded by `io_uring` fixed buffers and
must avoid heap allocation in the hot path.

## Current Skeleton

- `dequantize_block_simd` dispatches INT4/INT8 decode through AVX-512, AVX2 or scalar fallback.
- INT4 symmetric and affine layouts decode low nibble then high nibble.
- Decode output is written into caller-provided tile scratch.
- `compute_layer` accepts a dense block pointer, active expert block pointers and mutable hidden states.
- `GemmKernel` now includes a fused INT4 affine GEMV path for token inference.

## Fused Micro-Kernel

Implemented:

- `FusedInt4Gemm`
- `fused_gemv_i4_affine`
- AVX-512 dispatch boundary
- AVX2 + FMA implementation
- scalar reference fallback

Layout:

- output-major weight rows;
- two INT4 weights per byte;
- low nibble first;
- scales per output row and quant group;
- zero-points per output row and quant group.

The fused path does not write dequantized weights to RAM. It decodes nibbles,
applies affine scale/zero-point and accumulates the dot product directly in
vector registers.

## ZCBLK01 Runtime Parser

Implemented:

- `QuantBlockHeaderDisk` and `QuantTensorRecordDisk` disk structs in `model_format.rs`.
- `parse_quant_block(buffer: &[u8]) -> QuantBlockLayout`.
- zero-copy tensor record access via `QuantBlockLayout::tensor(index)`.
- zero-copy name, data and shape slicing.
- `gemv_i4_affine_tensorwise` bridge for the converter's current scalar
  per-tensor quant metadata.

Current converter compatibility:

- magic: `ZCBLK01\0`;
- header size: 40 bytes;
- tensor record size: 52 bytes;
- record fields include `scale:f32` and `zero_point:f32`;
- grouped scale/zero-point arrays are not emitted yet.

The compute layer now parses the ready block payload, extracts the first tensor
layout, infers `n x k` from shape and hidden size, and calls the fused GEMV path
without allocating or materializing dequantized weights.

## GEMM Recommendation

Start with a two-step path:

1. Use a small `GemmKernel` adapter around `matrixmultiply` only for baseline correctness and profiling.
2. Replace the hottest shapes with custom packed INT4/BF16/FP32 micro-kernels after block layout and router behavior are stable.

Reason:

- `matrixmultiply` is light and useful as a reference.
- GLM-5.2 MoE expert shapes will benefit from fused dequant + GEMM, where a generic SGEMM forces an avoidable full dequant tile.
- The final fast path should decode INT4 nibbles directly into vector registers or a tiny L1/L2 scratch tile, then feed FMA/AVX-512 accumulators.

## Next Steps

- Parse `ZCBLK01` mini headers from converted blocks to get per-tensor quant scales.
- Add a compute smoke benchmark that consumes ready I/O buffers before release.
- Add synthetic compute delay to verify overlap with N-buffer prefetch.
- Add `compute_bench.rs` for fused GEMV throughput and RSS.
- Connect `ReadyBlock` consumption to `FusedInt4Gemm`.
- Specialize packed expert projection kernels for GLM-5.2 shapes.
- Extend converter/runtime to grouped quant arrays for production INT4.

## DeepSeek FP4 Expert Hot Path

Status on 2026-07-05:

- L0-L43 technical smoke is ready.
- Warm L0-L43 one-token chat smoke takes about 644 seconds in the Python
  validation path.
- Prompt-profile L0-L43 takes about 650 seconds for 43 layers and 7 prompt
  tokens.
- `ffn_expert_forward_seconds` is about 465.835 seconds, or 71.65% of the
  measured runtime.
- LM-head bounded scan is not the current bottleneck.

Required kernel contract:

- Input hidden width: 4096.
- Expert active count: top6 routed experts.
- Expert projection shapes:
  - `w1`: 2048 x 4096 packed FP4 E2M1 plus UE8M0 scales;
  - `w3`: 2048 x 4096 packed FP4 E2M1 plus UE8M0 scales;
  - `w2`: 4096 x 2048 packed FP4 E2M1 plus UE8M0 scales.
- Activation:
  - quant/dequant activation simulation must match the Python validation path
    until reference parity is available;
  - hidden uses SwiGLU: `silu(w1(x)) * w3(x)`;
  - output is `w2(hidden)`.
- Memory rule:
  - do not materialize a full expert as FP32 persistently;
  - tile decode into caller-owned scratch;
  - scratch must be bounded and reported by the benchmark;
  - never allocate per token inside the hot loop.
- Safety rule:
  - kernel must have scalar reference fallback;
  - benchmark must fail if output contains NaN/Inf;
  - benchmark must report bytes read, scratch bytes, elapsed time and tokens/sec.

Release threshold:

- The public DeepSeek chat path must not pass the performance guardrail while
  warm single-token latency is above `--max-chat-token-seconds`.
- Current L0-L43 guardrail is expected to block with
  `chat_token_latency_budget_exceeded` until this kernel lands.

Target-machine verification:

```text
cargo test --manifest-path engine/zc_infer_core/Cargo.toml deepseek
cargo run --manifest-path engine/zc_infer_core/Cargo.toml --bin deepseek_fp4_expert_bench -- --iterations 1
```

The synthetic benchmark intentionally does not read real model weights. It uses
DeepSeek expert shapes and deterministic packed FP4 payloads to validate runtime
cost, scratch size and finite output before wiring the kernel to real `ZCBLK01`
expert tensors.

## DeepSeek ZCBLK FP4 Expert Bridge

Status on 2026-07-06:

- `deepseek_v4_fp4_expert_from_quant_block` maps one expert `ZCBLK01` block to
  the Rust FP4 expert kernel contract without copying payloads.
- The bridge requires the six tensor suffixes:
  - `.w1.weight`;
  - `.w1.scale`;
  - `.w3.weight`;
  - `.w3.scale`;
  - `.w2.weight`;
  - `.w2.scale`.
- Weight tensors must use `deepseek_fp4_e2m1_packed`.
- Scale tensors must use `deepseek_ue8m0_scale`.
- Shape and payload sizes are validated before compute:
  - weight rank 2;
  - scale rank 2;
  - matching rows;
  - compatible `w1/w3/w2` projection sizes.
- A synthetic `ZCBLK01` unit test validates the bridge plus scalar forward with
  caller-owned scratch.

Local verification available in this environment:

```text
PowerShell source-balance check: passed
Docker dev image: built
cargo test deepseek_fp4 in Docker: passed, 2 tests
cargo test deepseek in Docker: passed, 14 tests
```

Target-machine verification remains useful outside Docker:

```text
cargo test --manifest-path engine/zc_infer_core/Cargo.toml deepseek_fp4
cargo test --manifest-path engine/zc_infer_core/Cargo.toml deepseek
```

## DeepSeek Runtime Expert Dispatch

Status on 2026-07-06:

- `compute_expert_block` now checks each expert `ZCBLK01` for DeepSeek
  FP4/UE8M0 tensor formats before the legacy GLM INT4 branch.
- DeepSeek FP4 expert blocks are routed to:
  - `deepseek_v4_fp4_expert_from_quant_block`;
  - `deepseek_v4_fp4_expert_forward_scalar`.
- Input hidden state is copied into caller-owned scratch before output is
  written back into `hidden_states`, avoiding mutable aliasing and keeping the
  scratch bounded.
- `ComputeConfig::prefill_scratch_f32` now reserves enough scratch for the
  expert hot path instead of only dense/attention paths.
- Synthetic test coverage now includes:
  - bridge-only `ZCBLK01` read;
  - `compute_expert_block` automatic DeepSeek FP4 dispatch.

This is still a scalar reference path. It is needed for correctness and
plumbing. The first Docker release benchmark reports about `0.169020` seconds
per full expert forward at DeepSeek shape `4096 -> 2048 -> 4096`, reading about
12.58MB packed weights plus 1.05MB scales and using 16KB caller scratch.

The next release-performance wall is no longer compilation. It is wiring a safe
Rust server/runtime smoke to real converted DeepSeek blocks, then deciding
whether the scalar path is enough for a first demo or whether the FP4 matvec
needs immediate tiled SIMD.
