# DeepSeek-V4-Flash Architecture Parity Notes

Date: 2026-07-06. Source of truth: `models/deepseek-ai/DeepSeek-V4-Flash/inference/model.py`
(official reference implementation) + `config.json`. This document maps the real
architecture against the current Wohper Rust runtime and defines the parity
work queue.

## Key config facts

```text
hidden_size=4096  num_hidden_layers=43  vocab=129280
num_attention_heads=64  num_key_value_heads=1 (single latent KV)
head_dim=512  qk_rope_head_dim=64 (nope=448)
q_lora_rank=1024  o_lora_rank=1024  o_groups=8
n_routed_experts=256  num_experts_per_tok=6  n_shared_experts=1
moe_intermediate_size=2048  swiglu_limit=10.0  expert_dtype=fp4
scoring_func=sqrtsoftplus  topk_method=noaux_tc  routed_scaling_factor=1.5
norm_topk_prob=True  num_hash_layers=3
hc_mult=4  hc_sinkhorn_iters=20  hc_eps=1e-6
sliding_window=128  index_topk=512  compress_rope_theta=160000
num_nextn_predict_layers=1 (MTP)
```

## 1. Router / Gate (model.py `Gate`)

Reference math:

1. `scores = x @ gate.weight.T` (float32)
2. `scores = sqrt(softplus(scores))`  (scoring_func=sqrtsoftplus)
3. selection scores: `scores + e_score_correction_bias` - bias affects ONLY
   top-k selection, not routing weights
4. **hash layers** (`layer_id < 3`): `indices = tid2eid[token_id]` - static
   per-token-id lookup table `[129280, 6]`, no dynamic selection at all
5. `weights = original_scores[indices]`, normalized to sum 1
   (norm_topk_prob), then `weights *= 1.5` (routed_scaling_factor)

Wohper status (P1 DONE 2026-07-06, validated on real slice):

- [x] gate GEMV over 256 experts (fixed 2026-07-06: `router_gate_tensor`
  selects the gate by cols==hidden and always skips `tid2eid`)
- [x] sqrtsoftplus scoring (`RouterMath::DeepSeekV4SqrtSoftplus`)
- [x] e_score_correction_bias for selection only (`router_bias_values`)
- [x] tid2eid static routing for layers 0-2 (`tid2eid_expert_ids`, int32/int64)
- [x] sum-normalized weights x route_scale 1.5 (`finalize_route_weights`;
  legacy softmax skipped in generation for this family)

Slice caveat: hash layers can route to experts not materialized in the local
catalogseed slice (observed: token 42 -> layer 1/2 route to zero available
experts). Fidelity on hash layers needs targeted expert materialization for
the quality prompt token set (`tools/materialize_deepseek_v4_expert_shards.py`).

## 2. Attention (model.py `Attention`) - MLA + sliding window + DSA

Reference flow (per token):

1. `qr = q_norm(wq_a(x))` - q_lora 1024, weighted RMS norm
2. `q = wq_b(qr)` -> 64 heads x 512
3. **per-head parameter-free RMS**: `q *= rsqrt(mean(q^2) + eps)` per head
4. RoPE on last 64 dims of each q head
5. `kv = kv_norm(wkv(x))` - SINGLE 512-dim latent KV per token (MQA over
   latent). kv_norm covers the full 512 dims (448 nope + 64 rope)
6. RoPE on last 64 dims of kv; FP8-simulate quant on first 448 dims (QAT)
7. Sparse attention: sliding window 128 + (if compress_ratio) compressed
   history via `Compressor` (ratio 4) selected by `Indexer` top-512;
   `attn_sink` [64] adds a per-head sink logit; softmax scale = 512^-0.5
8. Attention output per head = weighted sum of latent kv (512-dim);
   **inverse RoPE applied to output rope dims** (`apply_rotary_emb(..., True)`)
9. Output projection is grouped low-rank: reshape o to 8 groups x (8 heads x
   512 = 4096); per group `wo_a` einsum -> o_lora 1024; concat (8192);
   `wo_b` [4096, 8192] -> hidden

Wohper status (P2 core DONE 2026-07-06 - `compute_deepseek_v4_mla_attention`
in compute.rs, selected via `AttentionKind::DeepSeekV4Mla`):

- [x] wq_a -> q_norm -> wq_b chain (roles QProj by cols)
- [x] weighted q_a_layernorm / kv norm application (marker match)
- [x] kv_norm applied to the full 512 dims
- [x] per-head parameter-free q RMS after wq_b
- [x] single latent KV (512) used as both key and value, no kv_b expansion
- [x] attn_sink added to the softmax denominator (fp32/bf16 read)
- [x] inverse RoPE on attention output rope dims
- [x] grouped wo_a/wo_b output projection (per-group FP8/BF16 row-range GEMV
  `gemv_fp8_e4m3_ue8m0_rows` / `gemv_bf16_rows`) - o path now actually runs
- [ ] sliding window 128 + compressor/indexer (bounded contexts attend over
  all cached positions, equivalent for context < 128; long-context needs DSA)
- [ ] RoPE variant parity: reference uses complex-pair rotary with YaRN
  corrections (`precompute_freqs_cis` with rope_factor/beta) and
  compress_rope_theta 160000 for compressed paths; Wohper uses plain
  interleaved RoPE theta 10000 - fine for short contexts, needs YaRN for long

## 3. Expert / MoE (model.py `Expert`, `MoE`)

Reference:

- SwiGLU: `silu(clamp(w1(x), max=10)) * clamp(w3(x), -10, 10)`; w1=gate, w3=up
- routing weight applied to the intermediate activation before w2
- shared expert always added (n_shared_experts=1)

Wohper status:

- [x] FP4 expert w1/w3/w2 forward with silu
- [x] shared expert FP8 path
- [x] swiglu_limit clamp (10.0) in FP4 expert scalar forward and FP8 shared
  expert path (P4 DONE 2026-07-06, `DEEPSEEK_V4_SWIGLU_LIMIT`)
- [x] per-expert weight application (equivalent post-scaling)

## 4. Hyper-Connections residual (model.py `Block`) - mHC

The residual stream is hc_mult=4 copies of the hidden state (4 x 4096).
Per block, twice (attn and ffn):

1. `hc_pre`: flatten 4x4096 -> 16384; `mixes = hc_fn @ x * rsqrt(mean(x^2))`
   (hc_fn [24, 16384]); split into pre[4], post[4], comb[4x4] via
   `hc_split_sinkhorn(mixes, hc_scale[3], hc_base[24], 4, 20 iters, eps)`;
   `y = sum(pre_i * x_i)` -> single 4096 stream
2. attn_norm/ffn_norm -> module -> output x
3. `hc_post`: `y_i = post_i * x + sum_j comb_ij * residual_j`

LM head consumes the 4-copy stream through `hc_head` (sigmoid gating, no
sinkhorn), then final RMS norm, then lm_head in fp32.

Wohper status (P3 core DONE 2026-07-06 - `compute_layer_deepseek_mhc`):

- [x] 4-copy hidden stream in the generation loop (hc_mult from config,
  embedding replicated across copies)
- [x] `hc_pre_into` / `hc_post_into` with scalar `hc_split_sinkhorn_scalar`
  (softmax rows + eps, then alternating col/row normalization, 20 iters)
- [x] hc tensors read zero-copy from the dense block (`hc_attn_*`, `hc_ffn_*`,
  fp32)
- [x] `hc_head_pool` at the LM boundary (sigmoid gating over copies, tensors
  `hc_head_fn/scale/base` read via row index from global aux) + final
  weighted RMS `norm.weight` before the LM head
- [ ] router input approximation: the gate consumes the FIRST residual copy
  pre-layer (prefetch requirement), while the reference gate consumes the
  post-attention working stream. Hash layers (0-2) are exact. Fixing this
  needs deferred expert I/O after the attention sub-block.
- [ ] MTP blocks ignored (fine: optional speculative head)

## 5. Embedding / LM head

- embed scaled? (check `ParallelEmbedding`) - plain lookup
- lm_head fp32 GEMV on final normalized single stream after hc_head
- MTP block (`mtp.*` tensors, num_nextn_predict_layers=1) - optional
  speculative next-token head; NOT required for base parity

## Parity work queue (status 2026-07-06)

1. **P1 router parity** - DONE (validated on real slice).
2. **P2 attention corrections** - DONE (core; window/DSA and YaRN pending).
3. **P3 mHC residual** - DONE (core; router-input approximation documented).
4. **P4 swiglu_limit clamp** - DONE.
5. **P5 DSA compressor/indexer** - TODO for contexts beyond the sliding
   window; short-context smokes are exact without it.
6. **P6 MTP** - optional, synergy with DeepSpec speculative plan.

Remaining fidelity gaps after P1-P4: router pre-layer input approximation,
sliding-window/DSA selection for long contexts, YaRN rope corrections,
FP8-simulated activation quant on kv nope dims (QAT match), and numeric
verification against the reference (reference-parity contract fixtures).
Next hard milestone: a reference logit-parity test on a bounded slice
(same tokens through inference/model.py on CPU vs zc_infer_server).
