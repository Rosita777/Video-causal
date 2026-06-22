# Benchmark Items and Metrics Implementation Plan

**Goal:** Convert the current valid5 and round4-valid9 evidence into a formal benchmark artifact plus reproducible metric tables.

**Architecture:** Add one exporter that merges tracked clean-source manifests/labels and baseline annotation CSVs into `benchmarks/causal_footprint_v0/items.jsonl`. Add one metrics script that reads `items.jsonl` and writes aggregate CSV/Markdown tables under `experiments/metrics/`.

**Tech Stack:** Python standard library only (`argparse`, `csv`, `json`, `collections`, `pathlib`) plus pytest subprocess tests.

---

### Task 1: Benchmark Item Exporter

**Files:**
- Create: `scripts/build_benchmark_items.py`
- Create: `tests/test_build_benchmark_items.py`
- Create by running script: `benchmarks/causal_footprint_v0/items.jsonl`

- [x] Write a failing test that builds a tiny clean manifest plus a tiny baseline summary and expects one JSONL item with clean metadata and baseline outputs.
- [x] Implement `scripts/build_benchmark_items.py` with repeatable `--source name,manifest,summary` inputs and `--output`.
- [x] Run the exporter for valid5 and round4-valid9 and verify pair ids are unique in the combined output.

### Task 2: Metric Computation

**Files:**
- Create: `scripts/compute_benchmark_metrics.py`
- Create: `tests/test_compute_benchmark_metrics.py`
- Create by running script: `experiments/metrics/causal_footprint_v0_metrics_by_baseline.csv`
- Create by running script: `experiments/metrics/causal_footprint_v0_metrics_by_mechanism.csv`
- Create by running script: `experiments/metrics/causal_footprint_v0_metrics_summary.md`

- [x] Write a failing test where target leakage, clean causal-footprint leakage, borderline, and unusable rows are counted separately.
- [x] Implement metrics from `items.jsonl`: total outputs, target leakage count/rate, causal-footprint leakage count/rate, borderline count/rate, no/unusable count/rate, and strict leakage conditional on target-erased outputs.
- [x] Generate baseline and mechanism tables for the current v0 items.

### Task 3: Documentation and Verification

**Files:**
- Modify: `README.md`
- Modify: `benchmarks/causal_footprint_v0/README.md`
- Modify: `docs/experiment_log.md`

- [x] Document `items.jsonl` as the current benchmark-v0 source of truth.
- [x] Document the generated metric table paths and headline conservative counts.
- [x] Run `pytest tests -q`, `git diff --check`, commit, and push.
