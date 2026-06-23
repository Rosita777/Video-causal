# Benchmark Evaluation V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the manifest, annotation review, and metric scripts for the causal-footprint benchmark evaluation layer.

**Architecture:** Three focused scripts operate on CSV files. `build_evaluation_manifest.py` merges gold rows, contact-sheet inputs, and optional VLM predictions. `build_annotation_review.py` renders a static review page and queue CSV. `compute_evaluation_metrics.py` turns the manifest into paper-facing tables.

**Tech Stack:** Python standard library, CSV/HTML output, pytest subprocess tests.

---

### Task 1: Evaluation Manifest Builder

**Files:**
- Create: `scripts/build_evaluation_manifest.py`
- Create: `tests/test_build_evaluation_manifest.py`

- [ ] Write tests that create tiny gold, VLM input, and prediction CSV files.
- [ ] Verify the tests fail because the script does not exist.
- [ ] Implement CSV loading, key joins, derived fields, and optional prediction columns.
- [ ] Run the targeted test and full pytest suite.

### Task 2: Annotation Review Builder

**Files:**
- Create: `scripts/build_annotation_review.py`
- Create: `tests/test_build_annotation_review.py`

- [ ] Write tests that build a manifest with one reference-backed row and one output-only row.
- [ ] Verify the tests fail because the script does not exist.
- [ ] Implement static HTML rendering and annotation queue CSV writing.
- [ ] Run the targeted test and full pytest suite.

### Task 3: Manifest Metrics

**Files:**
- Create: `scripts/compute_evaluation_metrics.py`
- Create: `tests/test_compute_evaluation_metrics.py`

- [ ] Write tests for baseline/mechanism counts and VLM disagreement metrics.
- [ ] Verify the tests fail because the script does not exist.
- [ ] Implement grouped metrics and summary Markdown.
- [ ] Run the targeted test and full pytest suite.

### Task 4: Generate Current Artifacts

**Files:**
- Create/update: `experiments/evaluation/causal_footprint_v1_manifest.csv`
- Create/update: `experiments/evaluation/annotation_queue.csv`
- Create/update: `experiments/evaluation/review.html`
- Create/update: `experiments/evaluation/metrics_by_baseline.csv`
- Create/update: `experiments/evaluation/metrics_by_mechanism.csv`
- Create/update: `experiments/evaluation/metrics_summary.md`

- [ ] Run the manifest builder on current gold rows and Claude predictions.
- [ ] Run the annotation review builder.
- [ ] Run the metrics builder.
- [ ] Check that generated files contain no API keys or embedded image payloads.

### Task 5: Documentation

**Files:**
- Modify: `README.md`
- Modify: `benchmarks/causal_footprint_v0/README.md`
- Modify: `docs/experiment_log.md`

- [ ] Document the v1 evaluation commands.
- [ ] Document the label semantics and current artifact locations.
- [ ] Run `git diff --check` and full tests.
