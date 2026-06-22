# VLM Contact-Sheet Evaluator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the dry-run input layer for a third-party VLM scorer by converting calibration videos into 5-frame contact sheets and model-ready request payloads.

**Architecture:** Add `scripts/build_vlm_eval_inputs.py` to read the calibration gold CSV, extract five evenly spaced frames per existing video, save contact sheets, and write `vlm_inputs.csv`. Add `scripts/evaluate_with_vlm.py` to read those inputs and emit dry-run JSONL request payloads that can later be sent to a real model adapter.

**Tech Stack:** Python standard library, PyAV for video decoding, Pillow for image sheet writing, pytest subprocess tests.

---

### Task 1: VLM Input Builder

**Files:**
- Create: `scripts/build_vlm_eval_inputs.py`
- Create: `tests/test_build_vlm_eval_inputs.py`
- Create by running script: `experiments/eval_calibration/vlm_inputs.csv`
- Create by running script: `experiments/eval_calibration/frame_sheets/*.jpg`

- [x] Write a failing test with a tiny synthetic mp4 and gold CSV; expect one generated sheet and one `vlm_inputs.csv` row.
- [x] Write a failing test for a missing video; expect `sheet_exists=false` and a non-empty `sheet_error`.
- [x] Implement `scripts/build_vlm_eval_inputs.py` with `--gold`, `--sheet-dir`, `--output`, `--frames-per-video`, and image size options.
- [x] Run the focused tests and verify they pass.
- [x] Generate real VLM inputs from the current 56-row gold CSV.

### Task 2: Dry-Run VLM Payload Builder

**Files:**
- Create: `scripts/evaluate_with_vlm.py`
- Create: `tests/test_evaluate_with_vlm.py`
- Create by running script: `experiments/eval_calibration/vlm_payloads_dryrun.jsonl`

- [x] Write a failing test with a tiny `vlm_inputs.csv`; expect one JSONL payload containing target/effect metadata and the required output schema.
- [x] Write a failing test that rows with `sheet_exists=false` are skipped by default and can be included with `--include-missing`.
- [x] Implement dry-run mode with `--inputs`, `--output-jsonl`, `--limit`, and `--include-missing`.
- [x] Run the focused tests and verify they pass.
- [x] Generate dry-run payloads for the current VLM inputs.

### Task 3: Documentation and Verification

**Files:**
- Modify: `README.md`
- Modify: `benchmarks/causal_footprint_v0/README.md`
- Modify: `docs/experiment_log.md`

- [x] Document the contact-sheet input flow and dry-run payload command.
- [x] Run `PYTHONNOUSERSITE=1 /home/deepseek_VG/.conda/envs/vcecf/bin/python -m pytest tests -q`.
- [x] Run `git diff --check`.
- [x] Commit and push.
