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
- [x] Implement the atomic response schema and prompt.
- [x] Re-run the focused tests and observe pass.

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
- [x] Implement `label_from_atomic_fields()` and update `normalize_prediction()`.
- [x] Re-run focused tests and observe pass.

### Task 3: Protocol Calibration Run

**Files:**
- Modify: `scripts/export_calibration_gold.py`
- Modify: `scripts/build_vlm_eval_inputs.py`
- Create: `experiments/eval_calibration/claude_sonnet_4_6_reference_atomic_full_predictions.csv`
- Create: `experiments/eval_calibration/claude_sonnet_4_6_reference_atomic_full_raw.jsonl`
- Create: `experiments/eval_calibration/claude_sonnet_4_6_reference_atomic_full/`
- Modify: `docs/experiment_log.md`
- Modify: `README.md`
- Modify: `benchmarks/causal_footprint_v0/README.md`

- [x] Add clean-reference video metadata to calibration gold export.
- [x] Generate clean-reference contact sheets when reference videos are available.
- [x] Run `qwen/qwen-vl-plus` on the first 8 VLM inputs with the atomic protocol.
- [x] Calibrate with `--allow-partial`.
- [x] Inspect summary and prediction rows.
- [x] Scan artifacts for API keys and base64 image data.
- [x] Run `qwen/qwen-vl-plus` on the first 8 reference-backed VLM inputs with the atomic protocol.
- [x] Calibrate the reference-aware sample with `--allow-partial`.
- [x] Run `anthropic/claude-sonnet-4-6` on the first 8 reference-backed VLM inputs with the atomic protocol.
- [x] Calibrate the Claude reference-aware sample with `--allow-partial`.
- [x] Run `anthropic/claude-sonnet-4-6` on all 36 reference-backed VLM inputs with the atomic protocol.
- [x] Calibrate the Claude reference-aware full run.
- [x] Record the old-protocol failure and new-protocol result in docs; keep only the Claude full run as the retained non-GPT fallback artifact.

### Task 4: Verification and Commit

**Files:**
- Modify: project docs and tests from previous tasks.

- [x] Run focused tests during implementation.
- [x] Run `git diff --check`.
- [x] Run `python -m pytest -q`.
- [x] Commit code, docs, and small calibration artifacts.
- [ ] Push to GitHub.
