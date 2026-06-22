# GPT-4o VLM Scorer Implementation Plan

**Goal:** Add a real GPT-4o scorer path that consumes contact-sheet inputs and writes prediction CSVs for calibration.

**Architecture:** Extend `scripts/evaluate_with_vlm.py` with an explicit `--run-api` mode while preserving existing `--dry-run`. Add helper functions for config loading, image data URLs, OpenAI-compatible request construction, fenced-JSON parsing, response validation, prediction CSV writing, and raw response logging.

**Tech Stack:** Python standard library only (`argparse`, `base64`, `csv`, `json`, `os`, `urllib`) plus pytest subprocess/unit tests.

---

### Task 1: API Helpers and Tests

**Files:**
- Modify: `scripts/evaluate_with_vlm.py`
- Modify: `tests/test_evaluate_with_vlm.py`

- [x] Add tests for parsing plain JSON and fenced JSON VLM responses.
- [x] Add tests that `--run-api` writes prediction CSV rows when a fake transport returns valid JSON.
- [x] Implement response parsing, validation, and prediction CSV writing.

### Task 2: Real GPT-4o Sample Run

**Files:**
- Create when channel is available: `experiments/eval_calibration/gpt4o_sample8_predictions.csv`
- Create when channel is available: `experiments/eval_calibration/gpt4o_sample8_raw.jsonl`
- Create when channel is available: `experiments/eval_calibration/gpt4o_sample8/calibration_metrics_summary.md`
- Create fallback smoke: `experiments/eval_calibration/gpt4o_mini_sample8_predictions.csv`
- Create fallback smoke: `experiments/eval_calibration/gpt4o_mini_sample8_raw.jsonl`
- Create fallback smoke: `experiments/eval_calibration/gpt4o_mini_sample8/calibration_metrics_summary.md`

- [x] Attempt `openai/gpt-4o` on the first VLM input through `https://api.360.cn/v1`; blocked by `no available channel`.
- [x] Run fallback `openai/gpt-4o-mini` on the first 8 VLM inputs.
- [x] Calibrate fallback sample predictions against the current gold CSV with `--allow-partial`.
- [x] Inspect result: `gpt-4o-mini` over-predicts `strict_leakage` and is not suitable as the main judge.

### Task 3: Documentation and Verification

**Files:**
- Modify: `README.md`
- Modify: `benchmarks/causal_footprint_v0/README.md`
- Modify: `docs/experiment_log.md`

- [x] Document the GPT-4o channel block, fallback sample command, and artifacts.
- [x] Run the full lightweight test suite.
- [x] Commit and push.

### Task 4: Qwen Fallback Scorer Trial

**Files:**
- Modify: `README.md`
- Modify: `benchmarks/causal_footprint_v0/README.md`
- Modify: `docs/experiment_log.md`

- [x] Run `alibaba/qwen-vl-max` on the first 8 VLM inputs.
- [x] Calibrate `alibaba/qwen-vl-max`; result over-predicts `strict_leakage` for all 8 rows.
- [x] Run `qwen/qwen-vl-plus` on the first 8 VLM inputs.
- [x] Calibrate the 8-row `qwen/qwen-vl-plus` sample; result distinguishes target-leakage rows better than previous fallbacks.
- [x] Run `qwen/qwen-vl-plus` on all 56 VLM inputs.
- [x] Calibrate the full run; result supports using `qwen/qwen-vl-plus` as a high-recall leakage screener, not as the final four-class judge.
- [x] Document fallback model status. Qwen raw artifacts were summarized but not retained after the atomic/reference-aware protocol replaced the direct-label fallback.
