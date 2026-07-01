# Benchmark Items and Metrics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the current valid5 and round4-valid9 evidence into a formal benchmark artifact plus reproducible metric tables.

**Architecture:** Add one exporter that merges tracked clean-source manifests/labels and baseline annotation CSVs into `benchmarks/causal_footprint_v0/items.jsonl`. Add one metrics script that reads `items.jsonl` and writes aggregate CSV/Markdown tables under `experiments/metrics/`.

**Tech Stack:** Python standard library only (`argparse`, `csv`, `json`, `collections`, `pathlib`) plus pytest subprocess tests.

---

### Task 1: Benchmark Item Exporter

**Files:**
- Create: `scripts/build_benchmark_items.py`
- Create: `tests/test_build_benchmark_items.py`
- Create by running script: `benchmarks/causal_footprint_v0/items.jsonl`

- [ ] Write a failing test that builds a tiny clean manifest plus a tiny baseline summary and expects one JSONL item with clean metadata and baseline outputs.
- [ ] Implement `scripts/build_benchmark_items.py` with explicit inputs for `--clean-manifest`, `--summary-csv`, `--source-name`, and `--output`.
- [ ] Run the exporter for valid5 and round4-valid9 and verify pair ids are unique in the combined output.

### Task 2: Metric Computation

**Files:**
- Create: `scripts/compute_benchmark_metrics.py`
- Create: `tests/test_compute_benchmark_metrics.py`
- Create by running script: `experiments/metrics/causal_footprint_v0_metrics_by_baseline.csv`
- Create by running script: `experiments/metrics/causal_footprint_v0_metrics_by_mechanism.csv`
- Create by running script: `experiments/metrics/causal_footprint_v0_metrics_summary.md`

- [ ] Write a failing test where target leakage, clean causal-footprint leakage, borderline, and unusable rows are counted separately.
- [ ] Implement metrics from `items.jsonl`: total outputs, target leakage count/rate, causal-footprint leakage count/rate, borderline count/rate, no/unusable count/rate, and `cfp_at_target_erased`.
- [ ] Generate baseline and mechanism tables for the current v0 items.

### Task 3: Documentation and Verification

**Files:**
- Modify: `README.md`
- Modify: `benchmarks/causal_footprint_v0/README.md`
- Modify: `docs/experiment_log.md`

- [ ] Document `items.jsonl` as the current benchmark-v0 source of truth.
- [ ] Document the generated metric table paths and headline conservative counts.
- [ ] Run `pytest tests -q`, `git diff --check`, commit, and push.
