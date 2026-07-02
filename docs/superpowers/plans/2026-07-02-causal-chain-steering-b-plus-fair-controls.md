# Causal Chain Steering B+ Fair Controls Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the orthogonal semantic control as strong and fair as Phase B full-chain steering by paraphrase-averaging and footprint-norm-matching it.

**Architecture:** Extend the Phase B manifest builder so each item carries a three-pair `orthogonal_semantic` control. Extend the runner so orthogonal semantic predictions are averaged like any other multi-pair link, then scaled to the averaged footprint direction norm before steering is applied. Keep existing single-pair manifests compatible.

**Tech Stack:** Python standard library, pytest, existing ZeroScope runner and VLM evaluation scripts.

---

### Task 1: Builder Emits Orthogonal Multi-Pair Control

**Files:**
- Modify: `tests/test_build_mvp0_phase_b_paraphrase_probe.py`
- Modify: `scripts/build_mvp0_phase_b_paraphrase_probe.py`

- [ ] **Step 1: Write failing test**

Add assertions that the Phase B/B+ builder writes `minimal_pairs["orthogonal_semantic"]`
as a three-item list and records `phase_b_control_method`.

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python -m pytest tests/test_build_mvp0_phase_b_paraphrase_probe.py::test_phase_b_builder_expands_each_link_to_three_pairs -q
```

Expected: fail because the manifest currently has no `orthogonal_semantic`
pairs.

- [ ] **Step 3: Implement minimal builder change**

Add an `orthogonal_pairs()` helper with three unrelated semantic minimal pairs,
include it in `expand_item()`, and add manifest metadata:
`phase_b_control_method = "paraphrase_averaged_norm_matched_orthogonal"`.

- [ ] **Step 4: Run test to verify it passes**

Run the same pytest command. Expected: pass.

### Task 2: Runner Norm-Matches Orthogonal Control

**Files:**
- Modify: `tests/test_run_mvp0_zeroscope_probe.py`
- Modify: `scripts/adapters/run_mvp0_zeroscope_probe.py`

- [ ] **Step 1: Write failing tests**

Add tests that:

1. `steering_contract(..., "orthogonal_semantic")` preserves manifest-provided
   multi-pair controls instead of overwriting them with the old single pair.
2. `synthesize_orthogonal_control_prediction()` scales the averaged orthogonal
   prediction to the averaged footprint-reference norm.

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python -m pytest tests/test_run_mvp0_zeroscope_probe.py::test_steering_contract_preserves_manifest_orthogonal_pairs tests/test_run_mvp0_zeroscope_probe.py::test_synthesize_orthogonal_control_prediction_matches_reference_norm -q
```

Expected: fail because the contract overwrites orthogonal pairs and no norm
matching helper exists yet.

- [ ] **Step 3: Implement minimal runner change**

Add `control_reference = "footprint"` for orthogonal semantic controls, preserve
manifest-provided orthogonal pairs, encode `__orthogonal_reference__`, and scale
the averaged orthogonal direction to the footprint direction norm before
`apply_steering_residual()`.

- [ ] **Step 4: Run tests to verify they pass**

Run the same pytest command. Expected: pass.

### Task 3: Generate and Validate B+ Manifest

**Files:**
- Output: `experiments/method_probe/zeroscope_mvp0_causal_chain_probe_20260702/phase_b_plus_fair_controls_probe_manifest.json`
- Output: `experiments/method_probe/zeroscope_mvp0_causal_chain_probe_20260702/phase_b_plus_dry_run/generation_manifest.json`

- [ ] **Step 1: Run full focused test suite**

Run:

```bash
python -m pytest tests/test_run_mvp0_zeroscope_probe.py tests/test_build_mvp0_causal_chain_probe.py tests/test_run_mvp0_zeroscope_sweep.py tests/test_build_mvp0_phase_b_paraphrase_probe.py -q
```

Expected: all pass.

- [ ] **Step 2: Build B+ manifest**

Run:

```bash
python scripts/build_mvp0_phase_b_paraphrase_probe.py \
  --probe-manifest experiments/method_probe/zeroscope_mvp0_causal_chain_probe_20260702/probe_manifest.json \
  --output experiments/method_probe/zeroscope_mvp0_causal_chain_probe_20260702/phase_b_plus_fair_controls_probe_manifest.json
```

Expected: output manifest exists and first three items have three cause,
mechanism, footprint, and orthogonal semantic pairs.

- [ ] **Step 3: Dry run B+ cell**

Run:

```bash
PYTHONNOUSERSITE=1 /home/deepseek_VG/.conda/envs/vcecf/bin/python scripts/adapters/run_mvp0_zeroscope_probe.py \
  --probe-manifest experiments/method_probe/zeroscope_mvp0_causal_chain_probe_20260702/phase_b_plus_fair_controls_probe_manifest.json \
  --output-dir experiments/method_probe/zeroscope_mvp0_causal_chain_probe_20260702/phase_b_plus_dry_run \
  --model models/zeroscope_v2_576w \
  --seed 15000 \
  --condition target_negative \
  --condition target_footprint_negative \
  --condition full_chain_steering \
  --condition random_direction \
  --condition orthogonal_semantic \
  --limit-items 3 \
  --dry-run \
  --strict-prompt-length
```

Expected: dry-run manifest exists; orthogonal semantic rows carry three encoded
prompt pairs and `control_reference = "footprint"`.

### Task 4: Run B+ Cell and Evaluation

**Files:**
- Output: `experiments/method_probe/zeroscope_mvp0_causal_chain_probe_20260702/phase_b_plus_real_alpha_0p25_window_3_6/`
- Output: `experiments/evaluation/mvp0_zeroscope_phase_b_plus_alpha_0p25_window_3_6_fable_20260702/`
- Modify: `docs/experiment_log.md`

- [ ] **Step 1: Run real B+ generation**

Use the emptiest GPU and run the same Phase B generation command with
`--probe-manifest phase_b_plus_fair_controls_probe_manifest.json` and
`--output-dir phase_b_plus_real_alpha_0p25_window_3_6`.

- [ ] **Step 2: Build evaluation package and run fable**

Use the existing VLM evaluation scripts to produce `review.csv`, frame strips,
`vlm_predictions.csv`, and raw responses. Keep the API key in environment
variables only.

- [ ] **Step 3: Compute low-level proxy**

Write `low_level_proxy.csv` and `low_level_proxy_summary.csv` with the same
schema used for Phase A and B.

- [ ] **Step 4: Log decision**

Append B+ results to `docs/experiment_log.md`, including fable summary,
low-level proxy summary, and whether the fair-control success gate passed.
