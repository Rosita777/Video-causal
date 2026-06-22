# VLM Contact-Sheet Evaluator Input Design

## Goal

Prepare the first real-evaluator interface for causal-footprint scoring by converting each generated video in the calibration gold set into a compact 5-frame contact sheet and a model-ready input CSV. Add a dry-run VLM prompt builder that emits deterministic request payloads before any paid or external API call is wired in.

## Scope

This stage builds the input and dry-run layers only. It does not call a third-party model, upload videos, or require network access. The output should be enough to inspect prompts, image paths, and expected prediction schema before connecting an API.

## Data Flow

1. Read `experiments/eval_calibration/causal_footprint_v0_gold_outputs.csv`.
2. For each row with an existing `video_path`, sample five evenly spaced frames from the video.
3. Save a single horizontal contact sheet per video under `experiments/eval_calibration/frame_sheets/`.
4. Write `experiments/eval_calibration/vlm_inputs.csv`, one row per video, containing identity fields, target/effect metadata, video path, sheet path, and human label for calibration.
5. Read `vlm_inputs.csv` in `scripts/evaluate_with_vlm.py`.
6. In `--dry-run` mode, write JSONL payloads containing the image path, prompt text, expected JSON schema, and output identity fields.

## Contact Sheet Policy

Each sheet contains five frames corresponding approximately to `t=0.00`, `t=0.25`, `t=0.50`, `t=0.75`, and `t=1.00`. The sheet title and prompt text are not burned into the image; metadata remains in CSV/JSONL so the same image can be reused with different prompt variants.

## VLM Prediction Contract

The dry-run prompt asks the model to answer with exactly:

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

The future non-dry-run scorer must convert model responses into the existing prediction CSV schema consumed by `scripts/calibrate_evaluator.py`.

## Failure Handling

Missing or unreadable videos are not silently dropped. The input builder writes rows with `sheet_exists=false`, empty `sheet_path`, and a `sheet_error` message. This keeps the denominator auditable and makes missing artifacts visible before model calls.

## Testing

Tests create tiny synthetic videos and gold CSVs. They verify frame-sheet generation, missing-video handling, VLM input CSV fields, dry-run payload content, and strict validation for malformed input rows.
