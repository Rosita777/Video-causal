# GPT-4o VLM Scorer Design

## Goal

Connect the contact-sheet evaluator inputs to a real, mainstream VLM judge using `openai/gpt-4o` through an OpenAI-compatible endpoint. The scorer should produce the prediction CSV consumed by `scripts/calibrate_evaluator.py`.

## Model Choice

`openai/gpt-4o` is the preferred primary judge because it is broadly recognized and easy to justify in a paper. The current retained non-GPT fallback artifact uses `anthropic/claude-sonnet-4-6` with the reference-aware atomic protocol. Qwen trials are documented as exploratory failures or high-recall screeners, but their raw artifacts are not retained.

Current endpoint status on 2026-06-22: the provided `https://api.360.cn/v1` default group lists `openai/gpt-4o`, but real calls return `no available channel`. The implementation supports GPT-4o-compatible calls, while tracked fallback artifacts show that `openai/gpt-4o-mini` over-predicts `strict_leakage`. Later Qwen and Claude trials show complementary judge biases; neither replaces human labels.

## API Contract

The scorer reads `experiments/eval_calibration/vlm_inputs.csv`, base64-encodes each contact sheet, optionally includes a clean-reference contact sheet, sends the images plus the deterministic prompt to `/chat/completions`, and expects atomic JSON with:

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

The script derives the existing prediction CSV fields, including `target_absent` and `pred_label`, from those atomic fields.

The script writes:

```text
experiments/eval_calibration/gpt4o_sample8_predictions.csv
experiments/eval_calibration/gpt4o_sample8_raw.jsonl
```

Fallback smoke artifacts:

```text
experiments/eval_calibration/gpt4o_mini_sample8_predictions.csv
experiments/eval_calibration/gpt4o_mini_sample8_raw.jsonl
experiments/eval_calibration/gpt4o_mini_sample8/
experiments/eval_calibration/claude_sonnet_4_6_reference_atomic_full_predictions.csv
experiments/eval_calibration/claude_sonnet_4_6_reference_atomic_full_raw.jsonl
experiments/eval_calibration/claude_sonnet_4_6_reference_atomic_full/
```

## Secret Handling

API credentials are read at runtime from environment variables or a local config file containing `key:` and `url:` lines. Keys are never printed, never committed, and never written to prediction artifacts.

## Safety

The default script behavior remains dry-run. Real API calls require `--run-api`. This prevents accidental cost or network calls during tests and future refactors.
