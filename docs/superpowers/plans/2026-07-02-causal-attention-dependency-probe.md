# Causal Attention Dependency Probe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a minimal ZeroScope cross-attention dependency probe that records target-token attention summaries without changing generated videos.

**Architecture:** Add a small probe script under `scripts/adapters/` that loads ZeroScope through the existing adapter utilities, resolves target/footprint token spans with the CLIP tokenizer, wraps UNet cross-attention processors with a summary recorder, runs a tiny generation job, and writes JSONL/CSV reports. Keep the probe independent from the residual-steering runner so failed attention experiments do not destabilize Phase A/B artifacts.

**Tech Stack:** Python, pytest, Diffusers attention processors, existing ZeroScope adapter utilities, CSV/JSONL artifacts.

---

### Task 1: Token Span Utilities

**Files:**
- Create: `scripts/adapters/run_zeroscope_attention_probe.py`
- Create: `tests/test_run_zeroscope_attention_probe.py`

- [x] **Step 1: Write failing tests**

Add tests for `normalize_token_text()`, `find_token_indices()`, and
`comparison_token_indices()` using fake tokenizer tokens. The tests should
prove that target words can be found even when CLIP tokens contain word-boundary
markers.

- [x] **Step 2: Run tests to verify they fail**

Run:

```bash
python -m pytest tests/test_run_zeroscope_attention_probe.py -q
```

Expected: fail because the script does not exist.

- [x] **Step 3: Implement utilities**

Implement token normalization, span lookup, and deterministic comparison-token
selection. Keep functions pure and independent of torch.

- [x] **Step 4: Run tests to verify they pass**

Run the same pytest command. Expected: pass.

### Task 2: Attention Summary Recorder Interface

**Files:**
- Modify: `scripts/adapters/run_zeroscope_attention_probe.py`
- Modify: `tests/test_run_zeroscope_attention_probe.py`

- [x] **Step 1: Write failing tests**

Add tests for an `AttentionSummaryRecorder` that accepts an attention probability
matrix and records mean target, footprint, comparison, and all-token attention
mass for a named module/step.

- [x] **Step 2: Run tests to verify they fail**

Run the recorder tests. Expected: fail because the recorder does not exist.

- [x] **Step 3: Implement recorder**

Implement a lightweight recorder that stores per-call dictionaries and can
write JSONL/CSV summaries. Do not store full attention maps.

- [x] **Step 4: Run tests to verify they pass**

Run the recorder tests. Expected: pass.

### Task 3: Diffusers Attention Hook

**Files:**
- Modify: `scripts/adapters/run_zeroscope_attention_probe.py`
- Modify: `tests/test_run_zeroscope_attention_probe.py`

- [x] **Step 1: Inspect local Diffusers attention processor signatures**

Use the installed `vcecf` environment to inspect `diffusers.models.attention_processor`.
Record which processor class ZeroScope uses.

- [x] **Step 2: Implement wrapper conservatively**

Create a wrapper processor that delegates to the original processor when
`encoder_hidden_states is None` and records only cross-attention calls. Preserve
output shape and dtype.

- [x] **Step 3: Add shape-level tests**

Use small fake tensors or monkeypatches to prove the wrapper records a
cross-attention call and leaves self-attention unrecorded.

### Task 4: Probe Runner and Smoke Run

**Files:**
- Modify: `scripts/adapters/run_zeroscope_attention_probe.py`
- Output: `experiments/method_probe/zeroscope_mvp0_causal_chain_probe_20260702/attention_probe_smoke/`
- Modify: `docs/experiment_log.md`

- [x] **Step 1: Implement CLI**

Support:

```text
--probe-manifest
--output-dir
--model
--seed
--limit-items
--steps
--num-frames
--height
--width
--device
--enable-model-cpu-offload
--vae-slicing
--dry-run
```

- [x] **Step 2: Run dry-run**

Dry-run should write token spans and planned output paths without loading the
model.

- [x] **Step 3: Run one-item smoke**

Run one item with low steps/resolution on the emptiest GPU. Expected artifacts:

```text
attention_trace.jsonl
attention_summary.csv
generation_manifest.json
videos/*.mp4  # optional; skipped for fast diagnostic trace runs
```

- [x] **Step 4: Log result**

Append whether attention calls were captured, how many modules/steps produced
cross-attention summaries, and whether target/footprint mass differs from
comparison tokens enough to justify an intervention pass.
