# Evaluator Calibration Harness Implementation Plan

**Goal:** Build a reusable interface for calibrating automatic video evaluators against the current human-labeled causal-footprint benchmark outputs.

**Architecture:** Add one exporter that flattens `items.jsonl` into a gold CSV. Add one calibration script that joins gold and prediction CSVs by `(item_id, baseline)` and writes label metrics, binary leakage metrics, and a confusion matrix.

**Tech Stack:** Python standard library only (`argparse`, `csv`, `json`, `collections`, `pathlib`) plus pytest subprocess tests.

---

### Task 1: Gold CSV Export

**Files:**
- Create: `scripts/export_calibration_gold.py`
- Create: `tests/test_export_calibration_gold.py`
- Create by running script: `experiments/eval_calibration/causal_footprint_v0_gold_outputs.csv`

- [ ] Write a failing subprocess test that creates a tiny `items.jsonl` and expects derived `human_label` values.
- [ ] Run the test and verify it fails because `scripts/export_calibration_gold.py` does not exist.
- [ ] Implement `scripts/export_calibration_gold.py` with `--items` and `--output`.
- [ ] Run the focused test and verify it passes.
- [ ] Export the real gold CSV from `benchmarks/causal_footprint_v0/items.jsonl`.

### Task 2: Calibration Metrics

**Files:**
- Create: `scripts/calibrate_evaluator.py`
- Create: `tests/test_calibrate_evaluator.py`
- Create by running script: `experiments/eval_calibration/example_predictions.csv`
- Create by running script: `experiments/eval_calibration/calibration_metrics_by_label.csv`
- Create by running script: `experiments/eval_calibration/calibration_confusion_matrix.csv`
- Create by running script: `experiments/eval_calibration/calibration_metrics_summary.md`

- [ ] Write a failing subprocess test with a tiny gold CSV and prediction CSV.
- [ ] Run the test and verify it fails because `scripts/calibrate_evaluator.py` does not exist.
- [ ] Implement schema validation, joining, confusion counts, per-label metrics, strict binary metrics, and relaxed binary metrics.
- [ ] Run the focused test and verify it passes.
- [ ] Create an example prediction file from the current human labels and run calibration as a smoke test.

### Task 3: Documentation and Verification

**Files:**
- Modify: `README.md`
- Modify: `benchmarks/causal_footprint_v0/README.md`
- Modify: `docs/experiment_log.md`

- [ ] Document the calibration interface and generated artifacts.
- [ ] Run `PYTHONNOUSERSITE=1 /home/deepseek_VG/.conda/envs/vcecf/bin/python -m pytest tests -q`.
- [ ] Run `git diff --check`.
- [ ] Commit and push.
