# GPU offload plan - RTX 2070 Super Max-Q 8GB (2026-07-08)

## Feasibility (verified)

- GPU: NVIDIA GeForce RTX 2070 Super with Max-Q Design, 8 GB VRAM,
  driver 592.27, CUDA 13.1 (Windows host).
- WSL2 + Docker passthrough: WORKS - `docker run --gpus all
  zc-infer-dev nvidia-smi` sees the GPU inside the container (driver
  libraries mounted from WSL, /usr/lib/wsl).
- Turing (sm_75): no native FP8, but the model uses FP8-E4M3 with
  per-group UE8M0 scales → LUT dequantization in a CUDA kernel is
  trivial (256-entry constant memory) and bit-compatible with the CPU
  LUT.

## Why the GPU (numbers from the ZC_PROF=1 profiling, 2026-07-08)

Steady-state decode pass (npos=3, chain K=2):
- total ~16.7 s, phase1 attention 9.1 s (55%) - inside phase1: o_proj
  4.8 s, q_proj 2.3 s (78% of phase1), window softmax ~0.3 s,
  compressor ~0.7 s.
- phase1 is MEMORY-BOUND on the attention weights: per position per
  layer q_b 33.5 MB + wo_a 33.5 MB + wo_b 58.7 MB (+ wq_a 7 MB, wkv
  ~4 MB) ≈ 133 MB → ~5.5 GB streamed per position full-model from RAM
  (measured effective bandwidth ~2.5 GB/s with the AVX2 gather-LUT
  kernel).

VRAM budget: the FP8 attention weights of all 44 layers ≈ 5.5-5.9 GB
(+ UE8M0 scales, negligible) → they FIT in 8 GB with headroom for
activations. GPU bandwidth (448 GB/s) makes that segment ~50-100x
faster than the CPU path; the per-position transfer is negligible
(hidden state 7168×4 B up, 7168×4 B down per layer).

## Proposed design

- The `cudarc` crate (runtime API + NVRTC - no nvcc toolchain in the
  container; only libcuda from the WSL passthrough is needed).
- One-time load: at startup, per layer, upload the FP8 tensors
  wq_a/wq_b/wkv/wo_a/wo_b plus scales (already contiguous in the dense
  block).
- Kernel: FP8-LUT GEMV (one warp per row, dequant in registers, f32
  FMA). Guaranteeing the exact CPU f32 reduction order is not
  practical → the gate is NUMERIC parity (tolerance) plus the argmax
  gate on the 15/16 quality suite, not bit-identity. Alternative for
  the initial gate: a kernel with serial per-128-group reduction
  (same order as the CPU LUT) - slower but near-bit-exact.
- Full fallback: env ZC_GPU=0 → CPU path unchanged.
- Phase 2 (optional): lm_head (1 GB FP8) in VRAM → faster sampling;
  watch the budget (5.9 + 1 = 6.9 GB, ~1 GB left).

## Risks

- WDDM (a Windows GPU shared with the display): higher submit latency
  than native Linux; mitigated by batching per layer (5 kernels/layer).
- 8 GB shared with the desktop: ~0.5 GB already in use; watch for OOM.
- Parity: different f32 reduction order → gate = full quality suite +
  numeric parity sweep on the fewshot (max |Δlogit| < 1e-3).

## Work order

1. Spike: cudarc hello + FP8 GEMV on a real tensor, CPU comparison.
2. wo_b alone (the single largest weight, 58.7 MB × 44).
3. All attention projections per layer.
4. Same-build A/B bench on the fewshot + quality suite.
5. (Optional) lm_head.

## Spike result (2026-07-08, step 1 DONE)

`gpu_gemv_spike` (build with `--features gpu`), synthetic data at the
real wo_b shape (7168×8192), engine-exact FP8/UE8M0 LUT semantics:

- GPU 0.83 ms vs scalar CPU 54.9 ms → 66x on the kernel; 70.5 GMAC/s
  (~10x the production AVX2-gather path in aggregate).
- max relative error 8.8e-4 < 1e-3 → PASS (f32 reduction order
  differs, as expected).
- The kernel is still memory-suboptimal (70 GB/s effective out of a
  theoretical 448 - byte loads not optimally coalesced) → headroom
  with uchar4/half2 loads.

Runtime recipe (traps solved):
- cudarc 0.12: the cuda-12060 feature is mandatory (driver 13.1 is
  backwards compatible); WITHOUT "dynamic-linking" it dlopens libcuda
  at runtime, so the build container needs no GPU.
- libnvrtc does NOT come with the WSL driver: install the
  nvidia-cuda-nvrtc-cu12 pip wheel (on a fresh container: apt-get
  install python3-pip first), then
  `LD_LIBRARY_PATH=$(dirname $(find /usr -name "libnvrtc.so*" | head -1))`.
- Run with `docker run --gpus all ... /cargo-target/release/gpu_gemv_spike`.

T2 (CPU batched phase1, same day) already covers the PREFILL side; the
GPU pays off mostly in pure DECODE (small npos) where CPU batching
cannot amortize. Expected ~2x on a decode pass once integrated.
