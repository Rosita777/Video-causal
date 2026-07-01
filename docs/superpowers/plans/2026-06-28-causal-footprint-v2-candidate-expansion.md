# Causal Footprint V2 Candidate Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a larger causal-footprint v2 candidate pool so CogVideoX-2B clean-source screening can yield enough valid rows for the main benchmark.

**Architecture:** Add a v2 builder that imports all v1 candidates and appends a targeted expansion pool generated from explicit mechanism templates. The output keeps the v1 file contract: `candidate_items.jsonl`, `candidate_pairs.tsv`, `controls_specs.jsonl`, candidate/control prompt files, and export manifests.

**Tech Stack:** Python standard library, existing benchmark TSV/JSONL schema, existing CogVideoX prompt and review scripts.

---

### Task 1: Add V2 Builder Tests

**Files:**
- Create: `tests/test_build_benchmark_v2_candidates.py`
- Later create: `scripts/build_benchmark_v2_candidates.py`

- [ ] **Step 1: Write failing tests**

Create tests that run the v2 builder on a tiny fixture with one v1 row, request a small expansion, and verify that the builder exports candidate JSONL/TSV, prompt files, controls, manifests, and v2-specific source metadata.

- [ ] **Step 2: Run the test and confirm it fails**

Run:

```bash
PYTHONNOUSERSITE=1 pytest tests/test_build_benchmark_v2_candidates.py -q
```

Expected: fail because `scripts/build_benchmark_v2_candidates.py` does not exist.

### Task 2: Implement V2 Candidate Builder

**Files:**
- Create: `scripts/build_benchmark_v2_candidates.py`
- Test: `tests/test_build_benchmark_v2_candidates.py`

- [ ] **Step 1: Implement the builder**

The builder should:

- read v1 `candidate_items.jsonl`;
- append generated expansion rows with unique `pair_id`s;
- target 360 rows by default;
- bias expansion toward weak v1 clean-source buckets: `fracture_damage`, `surface_trace`, `elastic_deformation`, then `fluid_impact`;
- write v2 outputs under `benchmarks/causal_footprint_v2`;
- write prompts to `prompts/causal_footprint_v2_candidates.txt` and `prompts/causal_footprint_v2_controls.txt`;
- preserve 3 controls per candidate.

- [ ] **Step 2: Run tests**

Run:

```bash
PYTHONNOUSERSITE=1 pytest tests/test_build_benchmark_v2_candidates.py tests/test_build_benchmark_v1_candidates.py -q
```

Expected: pass.

### Task 3: Generate V2 Benchmark Files

**Files:**
- Create: `benchmarks/causal_footprint_v2/*`
- Create: `prompts/causal_footprint_v2_candidates.txt`
- Create: `prompts/causal_footprint_v2_controls.txt`
- Modify: `README.md`
- Modify: `docs/experiment_log.md`

- [ ] **Step 1: Run the builder**

Run:

```bash
PYTHONNOUSERSITE=1 python scripts/build_benchmark_v2_candidates.py
```

Expected: 360 candidate rows and 1080 control rows.

- [ ] **Step 2: Validate output counts**

Run:

```bash
wc -l benchmarks/causal_footprint_v2/candidate_items.jsonl benchmarks/causal_footprint_v2/controls_specs.jsonl
```

Expected: 360 and 1080.

### Task 4: Start CogVideoX-2B Clean-Source Screening

**Files:**
- Use: `prompts/causal_footprint_v2_candidates.txt`
- Create: `outputs/causal_footprint_v2_clean_cogvideox2b_bf16_step20_*`

- [ ] **Step 1: Check GPU availability**

Run:

```bash
nvidia-smi
```

- [ ] **Step 2: Launch parallel generation**

Use existing generation scripts and distribute prompts across available GPUs. Prefer clean-source generation first; controls and baselines wait until clean-source candidates are screened.

### Task 5: Build Review and Chunked VLM Screening

**Files:**
- Create: clean-source review CSV/HTML under `outputs/analysis_contact_sheets`
- Create: chunked VLM predictions under `experiments/evaluation`

- [ ] **Step 1: Build clean review artifacts**

Use `scripts/build_clean_source_review.py` after generation manifests are available.

- [ ] **Step 2: Run chunked GPT-5.4 prelabel**

Use `scripts/evaluate_clean_chunks_with_vlm.py` with five-frame-range chunk coverage, sharded by candidate index.

- [ ] **Step 3: Human review queue**

Use the chunked aggregate predictions to prioritize human review. Official benchmark rows still require human adjudication.
