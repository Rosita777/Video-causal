# C0 Base-Validity Screening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an original-only C0 screening mode and export valid original items for full-grid follow-up.

**Architecture:** Extend the existing C0 grid runner with a `--variant-set` selector instead of creating a separate runner. Extend the scorer with one additional CSV output, `c0_valid_originals.csv`, derived from item scores.

**Tech Stack:** Python standard library, pytest, existing C0 runner/scorer scripts.

---

### Task 1: Runner Original-Only Mode

**Files:**
- Modify: `scripts/adapters/run_c0_counterfactual_grid.py`
- Modify: `tests/test_run_c0_counterfactual_grid.py`

- [ ] Add a failing test that `--variant-set original` creates one manifest item with `variant_grid=["original"]`.
- [ ] Run the focused runner test and confirm it fails.
- [ ] Implement `--variant-set {all,original}` and pass selected variants into `build_items`.
- [ ] Run the focused runner tests and confirm they pass.

### Task 2: Valid Original Export

**Files:**
- Modify: `scripts/score_c0_counterfactual_grid.py`
- Modify: `tests/test_score_c0_counterfactual_grid.py`

- [ ] Add a failing CLI test that expects `c0_valid_originals.csv` containing only rows with `original_valid=true`.
- [ ] Run the focused scorer test and confirm it fails.
- [ ] Implement the valid-original export.
- [ ] Run the focused scorer tests and confirm they pass.

### Task 3: Dry-Run Screening Manifest And Log

**Files:**
- Modify: `docs/experiment_log.md`

- [ ] Run an original-only dry run over a larger candidate slice.
- [ ] Record the exact next commands for real generation, review building, fable evaluation, and scoring.
- [ ] Run focused C0 regression tests.
