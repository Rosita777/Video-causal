# GPT-4o VLM Scorer Design

## Goal

Connect the contact-sheet evaluator inputs to a real, mainstream VLM judge using `openai/gpt-4o` through an OpenAI-compatible endpoint. The scorer should produce the prediction CSV consumed by `scripts/calibrate_evaluator.py`.

## Model Choice

`openai/gpt-4o` is the primary judge because it is broadly recognized and easy to justify in a paper. `alibaba/qwen-vl-max` remains a secondary cross-check candidate, but the first real run should use GPT-4o.

## API Contract

The scorer reads `experiments/eval_calibration/vlm_inputs.csv`, base64-encodes each contact sheet, sends the sheet plus the deterministic prompt to `/chat/completions`, and expects JSON with:

```json
{
  "target_absent": "yes|no|partial",
  "effect_visible": "yes|no|partial",
  "quality_ok": "yes|no",
  "pred_label": "strict_leakage|borderline|target_leakage|other_failure",
  "confidence": 0.0,
  "reason": "short explanation"
}
```

The script writes:

```text
experiments/eval_calibration/gpt4o_sample8_predictions.csv
experiments/eval_calibration/gpt4o_sample8_raw.jsonl
```

## Secret Handling

API credentials are read at runtime from environment variables or a local config file containing `key:` and `url:` lines. Keys are never printed, never committed, and never written to prediction artifacts.

## Safety

The default script behavior remains dry-run. Real API calls require `--run-api`. This prevents accidental cost or network calls during tests and future refactors.
