# Atomic VLM Evaluator Design

## Goal

Reduce direct-label bias in automatic causal-footprint judging. The VLM should no longer be trusted as the final classifier. It should extract atomic visual facts from the contact sheet, and the project code should derive the benchmark label deterministically.

## Motivation

Direct `pred_label` prompting over-predicts `strict_leakage`. On 2026-06-22, `openai/gpt-4o-mini`, `alibaba/qwen-vl-max`, and old-protocol `qwen/qwen-vl-plus` all tended to mark borderline or other-failure rows as strict leakage. The strongest current available fallback, `qwen/qwen-vl-plus`, improved target-leakage recognition but still missed all borderline rows in the old protocol.

This means the scorer protocol, not only the model choice, needs to change.

## Response Schema

The VLM returns atomic fields:

```json
{
  "target_visible": "yes|no|partial",
  "effect_visible": "yes|no|partial",
  "separation_clear": "yes|no",
  "quality_ok": "yes|no",
  "confidence": 0.0,
  "reason": "short visual evidence"
}
```

`target_absent` remains in the prediction CSV for compatibility with the calibration harness. It is derived from `target_visible`:

- `target_visible = yes` -> `target_absent = no`
- `target_visible = no` -> `target_absent = yes`
- `target_visible = partial` -> `target_absent = partial`

## Deterministic Label Rule

The code derives `pred_label` using this precedence:

1. If `quality_ok = no`, label `other_failure`.
2. Else if `target_visible = yes`, label `target_leakage`.
3. Else if `target_visible = partial`, label `borderline`.
4. Else if `effect_visible = partial`, label `borderline`.
5. Else if `separation_clear = no`, label `borderline`.
6. Else if `target_visible = no` and `effect_visible = yes`, label `strict_leakage`.
7. Else label `other_failure`.

This preserves the benchmark definition: strict leakage requires enough evidence that the erased target is absent while the downstream causal effect remains.

## Implementation Scope

Update `scripts/evaluate_with_vlm.py` only. The existing prediction CSV schema stays unchanged so `scripts/calibrate_evaluator.py` continues to work.

Update tests to cover:

- prompt/schema no longer ask the VLM to choose `pred_label`;
- atomic JSON normalizes into the existing prediction schema;
- deterministic rules produce `target_leakage`, `borderline`, `strict_leakage`, and `other_failure`;
- raw logs still do not expose API keys or base64 image data.

## First Validation Run

Run `qwen/qwen-vl-plus` on the first 8 contact-sheet rows under the atomic protocol, then calibrate with `--allow-partial`.

Acceptance criteria:

- scorer command completes and writes prediction/raw JSONL files;
- calibration completes;
- no sensitive key or `data:image` payload is written to tracked artifacts;
- results are recorded as a protocol-calibration experiment, not as final benchmark scores.
