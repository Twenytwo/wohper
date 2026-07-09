# GLM-5.2 Reference Parity Contract

Date: 2026-07-04

Wohper must not claim GLM-5.2 correctness from output text alone. Reference
parity requires one of these inputs:

- full local HF checkpoint with all 282 safetensors shards; or
- an official external numeric trace for the exact rendered prompt and token IDs.

## Required Trace Fields

```json
{
  "input_ids": [154827, 198],
  "numeric_trace": {
    "status": "ok",
    "hidden_states": [
      {
        "index": 0,
        "shape": [1, 2, 6144],
        "mean": 0.0,
        "std": 1.0,
        "sample": [0.0, 0.1]
      }
    ],
    "logits": {
      "shape": [1, 2, 154880],
      "mean": 0.0,
      "std": 1.0,
      "sample": [0.0, 0.1]
    }
  }
}
```

## Comparison Tool

```text
python3 tools/compare_glm52_trace_summaries.py \
  --reference state/reference_trace.json \
  --candidate state/wohper_trace.json \
  --out state/glm52_trace_compare.json
```

The comparison is a gate, not a benchmark. Any failed layer becomes the next
debug target.
