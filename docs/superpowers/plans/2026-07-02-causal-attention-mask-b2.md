# Causal Attention Mask B2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a small cross-attention masking intervention runner for Method B2.

**Architecture:** Extend `scripts/adapters/run_zeroscope_attention_probe.py` rather than creating a parallel runner. The existing tokenizer span resolution, attention processor, denoising loop, and trace writers are reused; B2 adds intervention specs, condition row expansion, selected-token attention reweighting, and control conditions.

**Tech Stack:** Python, pytest, torch, Diffusers `TextToVideoSDPipeline`, existing ZeroScope adapter helpers.

---

### Task 1: Attention Reweight Primitive

**Files:**
- Modify: `scripts/adapters/run_zeroscope_attention_probe.py`
- Modify: `tests/test_run_zeroscope_attention_probe.py`

- [x] **Step 1: Write failing test**

Add a test for `reweight_attention_columns()` with a small torch tensor. It
should suppress selected key-token columns and renormalize rows to sum to one.

- [x] **Step 2: Run test and confirm RED**

Run:

```bash
PYTHONNOUSERSITE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/deepseek_VG/.conda/envs/vcecf/bin/python \
  -m pytest tests/test_run_zeroscope_attention_probe.py::test_reweight_attention_columns_suppresses_and_renormalizes -q
```

Expected: fail because the helper does not exist.

- [x] **Step 3: Implement helper**

Implement `reweight_attention_columns(attention_probs, selected_indices,
scale)`.

- [x] **Step 4: Run test and confirm GREEN**

Run the same pytest command. Expected: pass.

### Task 2: Intervention-Aware Processor

**Files:**
- Modify: `scripts/adapters/run_zeroscope_attention_probe.py`
- Modify: `tests/test_run_zeroscope_attention_probe.py`

- [x] **Step 1: Write failing test**

Add a fake-attention processor test showing that `RecordingAttnProcessor`
applies masking only to the text-conditioned CFG half when configured with
selected token indices.

- [x] **Step 2: Run test and confirm RED**

Run the targeted test. Expected: fail because the processor ignores
intervention configuration.

- [x] **Step 3: Implement processor intervention fields**

Add `intervention_indices` and `intervention_scale` to the processor. Apply
reweighting after softmax and before `bmm(value)`.

- [x] **Step 4: Run test and confirm GREEN**

Run the targeted test and the existing processor tests. Expected: pass.

### Task 3: B2 Condition Row Expansion

**Files:**
- Modify: `scripts/adapters/run_zeroscope_attention_probe.py`
- Modify: `tests/test_run_zeroscope_attention_probe.py`

- [x] **Step 1: Write failing dry-run test**

Add a dry-run test with `--condition baseline --condition chain_mask
--condition random_token_mask` and assert three manifest rows are written with
the expected intervention token counts.

- [x] **Step 2: Run test and confirm RED**

Run the dry-run test. Expected: fail because `--condition` does not exist.

- [x] **Step 3: Implement conditions**

Add condition choices and row expansion. Add deterministic random-token
selection excluding special and chain tokens.

- [x] **Step 4: Run test and confirm GREEN**

Run the dry-run test and full attention probe tests. Expected: pass.

### Task 4: Real B2 Smoke

**Files:**
- Modify: `docs/experiment_log.md`
- Output: `experiments/method_probe/zeroscope_attention_mask_b2_20260702_smoke/`

- [x] **Step 1: Run dry-run**

Run a three-item dry-run with all B2 conditions. Expected: manifest writes
without model loading.

- [x] **Step 2: Run one-item smoke**

Run `baseline`, `chain_mask`, and `random_token_mask` on one item with compact
settings. Expected: MP4 files and attention summaries are written.

- [x] **Step 3: Run three-item B2 matrix if smoke succeeds**

Run three items over the B2 conditions with compact settings. Expected:
generated videos plus trace files.

- [x] **Step 4: Log result**

Append generated paths, row counts, and attention-mass sanity checks to
`docs/experiment_log.md`.
