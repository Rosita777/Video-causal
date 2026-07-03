# Method C Counterfactual Grid Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a runnable C0 counterfactual prompt-grid audit for target/footprint separability.

**Architecture:** Add one builder/runner for four counterfactual variants and one review builder that adapts its manifest into the existing fable VLM evaluator format. Reuse existing ZeroScope loading and frame-strip/VLM evaluation utilities.

**Tech Stack:** Python, pytest, ZeroScope/Diffusers, existing fable evaluator CSV schema.

---

### Task 1: Counterfactual Grid Runner

**Files:**
- Create: `scripts/adapters/run_c0_counterfactual_grid.py`
- Test: `tests/test_run_c0_counterfactual_grid.py`

- [ ] Write tests for four variants, expected verifier states, shared seed, dry-run manifest, and real-mode generator dispatch.
- [ ] Run the tests and confirm they fail because the script does not exist.
- [ ] Implement variant prompt construction, manifest writing, argument validation, dry run, and real generation dispatch.
- [ ] Run the focused tests and confirm they pass.

### Task 2: Review CSV Builder

**Files:**
- Create: `scripts/build_c0_counterfactual_review.py`
- Test: `tests/test_build_c0_counterfactual_review.py`

- [ ] Write tests that convert a generated C0 manifest into clean-reference plus variant review rows.
- [ ] Run the tests and confirm they fail because the script does not exist.
- [ ] Implement frame-strip creation through existing helpers and write `review.csv`.
- [ ] Run the focused tests and confirm they pass.

### Task 3: Smoke Run And Log

**Files:**
- Modify: `docs/experiment_log.md`

- [ ] Run a dry run on three MVP-0 items.
- [ ] Run a compact real generation if a GPU is available.
- [ ] Build review rows and run fable if strips are available.
- [ ] Append the C0 result and limitations to the experiment log.
