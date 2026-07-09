# Wohper Model Quality Prompt Set

Date: 2026-07-04

Quality checks must stay small, repeatable and cheap enough to run during the
engineering loop.

Prompt file:

```text
config/quality_prompts.small.json
```

Current prompt IDs:

- `it_greeting`;
- `it_math_small`;
- `it_instruction`;
- `unicode_smoke`.

Default public smoke:

```text
bash scripts/vps_quality_bench.sh GLM-5.2.INT4.GLOBAL-FULL-L3-L78-TOP4-AE2-SMOKE top4_smoke2 2 256 3 1 900 2
```

Current result:

```text
it_greeting -> " bounds"
it_math_small -> " bounds"
```

Interpretation:

- the runtime reaches real LM-head logits;
- the model is not qualitatively ready;
- do not scale to TOP8 only because TOP4 output is poor;
- fix parity, indexer and performance first, then rerun this same prompt set.

Expansion rule:

Add prompts only when they test a distinct behavior. Keep the default limit at
two prompts until prefill performance improves.
