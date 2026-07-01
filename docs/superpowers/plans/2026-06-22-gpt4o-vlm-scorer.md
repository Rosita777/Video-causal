# GPT-4o VLM Scorer Implementation Plan

**Goal:** Add a real GPT-4o scorer path that consumes contact-sheet inputs and writes prediction CSVs for calibration.

**Architecture:** Extend `scripts/evaluate_with_vlm.py` with an explicit `--run-api` mode while preserving existing `--dry-run`. Add helper functions for config loading, image data URLs, OpenAI-compatible request construction, fenced-JSON parsing, response validation, prediction CSV writing, and raw response logging.

**Tech Stack:** Python standard library only (`argparse`, `base64`, `csv`, `json`, `os`, `urllib`) plus pytest subprocess/unit tests.

---

### Task 1: API Helpers and Tests

**Files:**
- Modify: `scripts/evaluate_with_vlm.py`
- Modify: `tests/test_evaluate_with_vlm.py`

- [ ] Add tests for parsing plain JSON and fenced JSON VLM responses.
- [ ] Add tests that `--run-api` writes prediction CSV rows when a fake transport returns valid JSON.
- [ ] Implement response parsing, validation, and prediction CSV writing.

### Task 2: Real GPT-4o Sample Run

**Files:**
- Create by running script: `experiments/eval_calibration/gpt4o_sample8_predictions.csv`
- Create by running script: `experiments/eval_calibration/gpt4o_sample8_raw.jsonl`
- Create by running script: `experiments/eval_calibration/gpt4o_sample8/calibration_metrics_summary.md`

- [ ] Run `openai/gpt-4o` on the first 8 VLM inputs through `https://api.360.cn/v1`.
- [ ] Calibrate the sample predictions against the current gold CSV.
- [ ] Inspect whether target leakage and strict leakage are separated well enough for a full 56-row run.

### Task 3: Documentation and Verification

**Files:**
- Modify: `README.md`
- Modify: `benchmarks/causal_footprint_v0/README.md`
- Modify: `docs/experiment_log.md`

- [ ] Document the GPT-4o sample command and artifacts.
- [ ] Run the full lightweight test suite.
- [ ] Commit and push.
