# Transformer Math Fidelity Audit

Date: 2026-07-03

Scope: Wohper GLM-5.2 runtime after `GLOBAL-32K-L3-L15-E0E3-AE2-SMOKE` and `GLOBAL-64K-L3-L15-E0E3-AE1` validation.

## Initial Verdict

The L3-L15 runtime path is operational, but it is not yet a faithful GLM-MoE-DSA transformer block.

The strongest current evidence is the L3-L15 AE2 VPS smoke with the patched dense math profiler:

```text
math_fidelity_gap layer=0 component=shared_expert present=true applied=false
...
math_fidelity_gap layer=11 component=shared_expert present=true applied=false
token_id=0
```

This proves the converter is packing shared-expert tensors into real sparse layer dense blocks, while the runtime detects but does not yet apply them.

## Follow-up Patch Verdict

The first fidelity wall is now implemented and VPS-validated:

```text
math_fidelity layer=0 component=shared_expert present=true applied=true
...
math_fidelity layer=11 component=shared_expert present=true applied=true
token_id=24336 -> _det
```

This does not make the model chat-ready yet, but it removes the previous `token_id=0` collapse for the `GLOBAL-32K-L3-L15-E0E3-AE2-SMOKE` path.

## Current Runtime Order

The patched `compute_layer` order is:

```text
input_layernorm if present -> attention probe -> attention residual
post_attention_layernorm if present -> routed expert MoE + shared expert -> MLP residual
```

Known implemented pieces:

- GLM-DSA q/kv/o projection probe;
- RoPE over q and shared kv rope;
- causal KV cache;
- router logits with softmax route weights;
- routed expert gate/up/down with SiLU gate;
- shared expert gate/up/down from dense block;
- explicit attention and MLP residual adds;
- weighted `input_layernorm` and `post_attention_layernorm` when packed;
- LM-head streaming/chunked argmax.

Known missing or approximate pieces:

- `q_a_layernorm` and `kv_a_layernorm` are still approximated with unweighted RMSNorm inside the GLM-DSA probe;
- GLM block order still needs reference matching against upstream implementation;
- output quality is not chat-valid yet; token `_det` is only a non-collapse smoke signal.

## Patch Added

Added `DenseMathProfile` and `dense_math_profile` in `engine/zc_infer_core/src/compute.rs`.

The profiler detects:

- attention tensors;
- router tensors;
- shared expert tensors;
- norm tensor count.

When a dense block contains shared experts, runtime logs:

```text
math_fidelity_gap layer=N component=shared_expert present=true applied=false
```

Unit coverage added:

```text
compute::tests::dense_math_profile_detects_shared_expert_gap_inputs
compute::tests::shared_expert_dense_branch_computes_gate_up_down
```

## Validation

VPS Rust tests:

```text
cargo test --release --manifest-path engine/zc_infer_core/Cargo.toml
23 passed
```

VPS L3-L15 AE2 smoke:

```text
bash scripts/vps_merged_smoke.sh GLM-5.2.INT4.GLOBAL-32K-L3-L15-E0E3-AE2-SMOKE 3,4,5,6,7,8,9,10,11,12,13,14 2 256 3
CLIENT_EXIT=0
token_id=24336 -> _det
shared_expert applied=true for compute slots 0..11
scratch_logits=53824
```

## Next Engineering Wall

Build a reference trace for a tiny prompt and compare:

1. `q_a_layernorm` / `kv_a_layernorm` weighted behavior;
2. exact GLM-DSA indexer behavior;
3. hidden-state scale after each layer;
4. router logits before softmax.

After that, move to the chat-template/multi-token task.
