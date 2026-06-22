# Evaluator Calibration Harness Design

## Goal

Build a lightweight calibration harness for automatic causal-footprint evaluators. The harness converts the current human-reviewed benchmark outputs into a gold CSV, accepts prediction CSVs from any future video scorer, and reports agreement metrics against the human labels.

## Scope

This stage does not run a VLM or decode videos. It standardizes the interface and metrics so GPT-style, open-source VLM, rule-based, or future custom evaluators can be compared later without changing the benchmark.

## Data Flow

1. `benchmarks/causal_footprint_v0/items.jsonl` remains the source of truth for human labels.
2. `scripts/export_calibration_gold.py` flattens every `baseline_outputs` row into `experiments/eval_calibration/causal_footprint_v0_gold_outputs.csv`.
3. Any automatic evaluator writes a prediction CSV with the same output identity fields plus predicted labels.
4. `scripts/calibrate_evaluator.py` joins gold and predictions by `(item_id, baseline)` and computes classification metrics.

## Gold Label Policy

The gold file exposes both raw human fields and derived labels:

- `human_label=strict_leakage` when `usable_for_claim == yes`.
- `human_label=borderline` when `usable_for_claim == borderline`.
- `human_label=target_leakage` when `failure_mode == target_leakage`.
- `human_label=other_failure` for all remaining `usable_for_claim == no` rows.

The primary headline task is strict causal-footprint leakage. A secondary relaxed task treats `strict_leakage` and `borderline` as positive.

## Prediction Schema

Prediction CSVs must contain:

```text
item_id,baseline,video_path,target_absent,effect_visible,quality_ok,pred_label,confidence,reason
```

Allowed `pred_label` values are:

```text
strict_leakage,borderline,target_leakage,other_failure
```

This schema intentionally separates observable sub-judgments from the final predicted label. Future scorers can populate the booleans directly from prompts, VLM answers, or rule-based detectors.

## Metrics

`scripts/calibrate_evaluator.py` writes:

- `calibration_metrics_summary.md`: headline metrics and confusion matrix.
- `calibration_metrics_by_label.csv`: precision, recall, and F1 for each label.
- `calibration_confusion_matrix.csv`: full gold-vs-predicted counts.

The summary reports strict leakage as the primary binary task, relaxed leakage as the secondary binary task, and macro-F1 over the four-class label set.

## Testing

Tests use tiny synthetic `items.jsonl` and prediction CSV files. They verify that gold export preserves identity fields, derives labels correctly, rejects bad prediction schemas, and computes strict/relaxed leakage metrics correctly.
