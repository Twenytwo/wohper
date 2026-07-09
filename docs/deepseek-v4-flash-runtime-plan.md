# DeepSeek-V4-Flash Runtime Plan

DeepSeek-V4-Flash is the **primary project target** (decision 2026-07-06); the
GLM-5.2 700B-class target is paused as a later milestone. The end goal is
running this model on a commodity PC with 16 GB RAM and 1 TB SSD. It keeps the
same Wohper thesis: read-only NVMe weights, bounded RAM, explicit I/O, and no
accidental swap-style execution.

## Current Status (2026-07-06 checkpoint)

Phase progress against this plan:

| Phase | Status |
| --- | --- |
| 1 - Metadata and inventory | **Done** - tokenizer/template contract passes, L0-L43 slice converted with `global_aux` split. |
| 2 - Converter dry-run | **Done** - role mapping validated, streaming conversion produced real L0-L43 slice. |
| 3 - Runtime adapter | **Mostly done in Rust** - FP8 dense/BF16 GEMV, FP4 expert dispatch, shared expert FP8, DeepSeek norm mapping, row-wise embed/LM-head. Missing: real top6 router (no-hint) and DeepSeek-specific attention (`wq_a/wq_b/wkv/wo_a/wo_b` still on GLM-derived compatibility path). |
| 4 - Smoke and quality | **Single-token smoke passed**; quality and multi-token not ready. |
| 5 - DeepSpec speculation | Not started (by design). |

Validated evidence:

```text
cargo_test=55 passed
mini_l0_l4_smoke=passed token=60574 logit=53.55931
full_l0_l43_smoke=passed token=10589 logit=360.1532
pre-norm-fix full logit=1.9098921e28 (numeric blow-up resolved)
```

Exact stop point: router no-hint validation blocked by Docker escalation
usage limit, saved in
`state/deepseek_v4_router_no_hint_validation_pending_2026-07-06.json`.

Next steps in order:

1. Mini L0-L4 smoke with `ZC_EXPERTS=` (empty) and `ZC_ROUTER_PROBE_TOPK=8`;
   expect `router layer=0 source=router`.
2. Build manifest/slice with at least 6 materialized experts per layer
   (current compact manifest exposes 2) to enable real top6 routing.
3. Full L0-L43 no-hint smoke with real top6 router.
4. Model-family-specific DeepSeek attention replacing the GLM compatibility
   path.
5. Optimize scalar FP8 GEMV (compiled/SIMD hot path).
6. Quality prompt set without repeated-token collapse, then public gates.

## Contract Facts

The public model card reports:

- 284B total parameters;
- 13B activated parameters;
- 1M token context;
- FP4 + FP8 mixed precision, with MoE expert parameters in FP4 and most other
  parameters in FP8;
- MIT license;
- Hugging Face repository size around 160 GB;
- 46 safetensors shards in the published repository view.

The local contract is:

```text
config/deepseek_v4_flash.contract.json
```

The checker is:

```bash
python3 tools/check_deepseek_v4_flash_readiness.py --contract-only
```

## Non-Negotiable Safety Rules

- Do not download weights by default.
- Do not run DeepSpec training or target-cache preparation by default.
- Do not claim 16 GB quick-start readiness until the adapter has a bounded
  single-token and multi-token smoke.
- Require a free-space plan before any local download.
- Keep DeepSeek adapter code separate from GLM code paths.

## Implementation Phases

### Phase 1 - Metadata And Inventory

Deliverables:

- local metadata directory;
- tokenizer/encoding folder present;
- safetensors index or deterministic shard inventory;
- contract checker status `ready_for_converter_dry_run`.

Inventory dry-run:

```bash
python3 tools/plan_deepseek_v4_flash_inventory.py \
  --model-dir models/deepseek-ai/DeepSeek-V4-Flash
```

This is expected to block until `model.safetensors.index.json` and local shards
are available. Unknown tensor roles are treated as blockers, not warnings.

Converter dry-run:

```bash
bash scripts/vps_deepseek_v4_flash_converter_dry_run.sh
```

The dry-run consumes only metadata and index reports. It validates role mapping,
reports FP4/FP8/aux tensor groups, and emits the disk plan. It does not read
safetensors payloads and cannot produce runtime weights.

Metadata-only sync:

```bash
bash scripts/vps_deepseek_v4_flash_metadata_sync.sh
```

The sync script hard-excludes `*.safetensors`, `*.bin`, `*.pt`, `*.pth`,
`*.gguf`, archives and any file outside a metadata allowlist. It has per-file
and total byte caps. Its purpose is to fetch configuration and inventory inputs,
not weights.

Tokenizer/template contract:

```bash
python3 tools/check_deepseek_v4_flash_tokenizer_contract.py --contract-only
```

DeepSeek-V4-Flash must not fall back to a generic ChatML or Jinja renderer. The
adapter needs the local `encoding` metadata and an explicit model-specific
prompt renderer before quality tests mean anything.

Risk:

- wrong tokenizer or prompt encoding invalidates every quality result.

### Phase 2 - Converter Dry-Run

Deliverables:

- tensor-name inventory;
- role mapping report;
- skipped/unknown tensors explicitly listed;
- estimated dense/expert/global disk plan.

Risk:

- silently skipped tensors produce plausible-looking but wrong logits.

### Phase 3 - Runtime Adapter

Deliverables:

- FP8 dense unpack path;
- FP4 expert unpack path;
- MoE router and expert dispatch;
- DeepSeek attention path;
- mHC residual path;
- embed and LM-head flow.

Risk:

- adapting GLM assumptions to DeepSeek will hide numerical bugs. The adapter
  must be model-family explicit.

### Phase 4 - Smoke And Quality

Deliverables:

- single-token logit smoke;
- bounded multi-token smoke;
- quality prompt set without repeated-token collapse;
- disk/cache cleanup proof.

Risk:

- fast generation is meaningless before semantic smoke passes.

### Phase 5 - DeepSpec-Inspired Speculation

DeepSpec enters only after the base target path is correct.

Deliverables:

- draft proposal protocol;
- target verification protocol;
- acceptance length and verify-rate metrics;
- optional external draft model.

Risk:

- DeepSpec's training/cache workflow is not compatible with the commodity quick
  start and must remain advanced-only.
