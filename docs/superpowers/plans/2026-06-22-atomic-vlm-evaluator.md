# Atomic VLM Evaluator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace direct VLM label classification with atomic visual fact extraction plus deterministic label derivation.

**Architecture:** `scripts/evaluate_with_vlm.py` keeps the existing prediction CSV contract. The prompt asks for atomic fields only, `normalize_prediction()` accepts atomic JSON, and a new rule helper derives `pred_label`.

**Tech Stack:** Python standard library, existing pytest tests, OpenAI-compatible VLM endpoint.

---

### Task 1: Atomic Schema Tests

**Files:**
- Modify: `tests/test_evaluate_with_vlm.py`
- Modify: `scripts/evaluate_with_vlm.py`

- [x] Write failing tests showing dry-run payloads expose `target_visible`, `effect_visible`, `separation_clear`, and `quality_ok`, but do not ask the model for `pred_label`.
- [x] Run the focused tests and observe failure.
- [ ] Implement the atomic response schema and prompt.
- [ ] Re-run the focused tests and observe pass.

### Task 2: Deterministic Label Rule

**Files:**
- Modify: `tests/test_evaluate_with_vlm.py`
- Modify: `scripts/evaluate_with_vlm.py`

- [x] Write failing tests for atomic JSON normalization:
  - visible target -> `target_leakage`;
  - absent target plus visible effect and clear separation -> `strict_leakage`;
  - partial target/effect or unclear separation -> `borderline`;
  - bad quality or absent effect -> `other_failure`.
- [x] Run focused tests and observe failure.
- [ ] Implement `label_from_atomic_fields()` and update `normalize_prediction()`.
- [ ] Re-run focused tests and observe pass.

### Task 3: Protocol Calibration Run

**Files:**
- Create: `experiments/eval_calibration/qwen_vl_plus_atomic_sample8_predictions.csv`
- Create: `experiments/eval_calibration/qwen_vl_plus_atomic_sample8_raw.jsonl`
- Create: `experiments/eval_calibration/qwen_vl_plus_atomic_sample8/`
- Modify: `docs/experiment_log.md`
- Modify: `README.md`
- Modify: `benchmarks/causal_footprint_v0/README.md`

- [ ] Run `qwen/qwen-vl-plus` on the first 8 VLM inputs with the atomic protocol.
- [ ] Calibrate with `--allow-partial`.
- [ ] Inspect summary and prediction rows.
- [ ] Scan artifacts for API keys and base64 image data.
- [ ] Record the old-protocol failure and new-protocol result in docs.

### Task 4: Verification and Commit

**Files:**
- Modify: project docs and tests from previous tasks.

- [ ] Run `python -m pytest -q`.
- [ ] Run `git diff --check`.
- [ ] Commit code, docs, and small calibration artifacts.
- [ ] Push to GitHub.
