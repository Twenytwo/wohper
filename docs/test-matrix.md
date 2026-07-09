# Wohper Public Test Matrix

Date: 2026-07-04

Run on the VPS workspace:

```text
bash scripts/vps_public_test_matrix.sh
```

## Safe Default Tests

- open-repo preflight;
- expert storage preflight;
- reference readiness, expected blocked without 282 checkpoint shards;
- GLM-DSA long-context guard, expected blocked at 2049 tokens;
- chat template/Unicode/streaming smoke;
- performance log summary from existing top4 artifacts.

## Heavy Tests

Run only intentionally:

- `scripts/vps_quality_bench.sh` on top4;
- `scripts/vps_sampling_top4_smoke.sh`;
- full conversion or TOP8 expert materialization.

Heavy tests must keep explicit buffer, prompt, timeout, cache, and disk guardrails.

## DeepSeek-V4 Local Matrix

Date: 2026-07-05

Safe local command:

```text
python tools/deepseek_v4_public_test_matrix.py
```

This command does not download weights, does not convert payloads, and does not
delete artifacts. It runs runtime checks only when the local artifacts already
exist.

Current local result:

- `deepseek_tokenizer_smoke`: passed;
- `deepseek_perf_guardrail`: failed on L0-L43, expected until the FP4 expert
  hot path is compiled/optimized;
- `rust_toolchain`: expected blocked on this machine because `cargo`/`rustc`
  are not in PATH.

Current L0-L43 command:

```text
python tools/deepseek_v4_public_test_matrix.py --model-dir models/deepseek-ai/DeepSeek-V4-Flash --core models/wohper/DeepSeek-V4-Flash.RAW.L0-L43-SPLIT-GLOBAL-CATALOGSEED/dense_core.bin --single-token state/deepseek_v4_multilayer_profile_l0_l43_prompt_micro_2026-07-05.json --chat state/deepseek_v4_bounded_chat_l0_l43_warm_scan1024_1tok_2026-07-05.json --profile-summary state/deepseek_v4_l0_l43_prompt_profile_summary_2026-07-05.json
```

L0-L43 performance blockers:

- `single_token_read_budget_exceeded`;
- `chat_token_latency_budget_exceeded`.

Post FP4-dispatch local result on 2026-07-06:

- command used the bundled Codex Python runtime, not a global Python install;
- `deepseek_tokenizer_smoke`: passed;
- `deepseek_perf_guardrail`: failed, still expected;
- `rust_toolchain`: expected blocked on host because `cargo`/`rustc` are absent;
- Docker dev Rust verification: passed separately with `zc-infer-dev`;
- artifact:
  `state/deepseek_v4_public_test_matrix_l0_l43_after_fp4_dispatch_2026-07-06.json`.

Post-dispatch L0-L43 command:

```text
python tools/deepseek_v4_public_test_matrix.py --core models/wohper/DeepSeek-V4-Flash.RAW.L0-L43-SPLIT-GLOBAL-CATALOGSEED/dense_core.bin --single-token state/deepseek_v4_single_token_l0_math_smoke_scan256_2026-07-05.json --chat state/deepseek_v4_bounded_chat_l0_l43_fp4lut_indexer_guard_scan1024_1tok_2026-07-05.json --profile-summary state/deepseek_v4_l0_l43_prompt_profile_summary_2026-07-05.json --max-chat-token-seconds 30 --out state/deepseek_v4_public_test_matrix_l0_l43_after_fp4_dispatch_2026-07-06.json
```

Docker Rust verification:

```text
docker build -f engine/zc_infer_core/Dockerfile.dev -t zc-infer-dev .
docker run --rm -v "${PWD}:/workspace" -w /workspace zc-infer-dev cargo test --manifest-path engine/zc_infer_core/Cargo.toml deepseek
```

Current Docker Rust result:

- `cargo test deepseek_fp4`: passed, 2 tests;
- `cargo test deepseek`: passed, 14 tests;
- `deepseek_fp4_expert_bench --iterations 10`: passed,
  `0.169020` seconds per scalar expert forward.

DeepSeek heavy tests:

- `tools/deepseek_v4_single_token_l0_math_smoke.py`;
- `tools/deepseek_v4_bounded_chat_smoke.py`;
- `tools/materialize_deepseek_v4_expert_shards.py --execute`.

Run heavy tests only on a machine with the DeepSeek safetensors already present,
enough SSD headroom, and an explicit free-space floor.
