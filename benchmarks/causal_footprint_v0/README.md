# Causal Footprint Benchmark v0

Status: candidate-pair construction, clean-source gating, first formal item file, and first metric tables.

This benchmark targets causal footprint leakage in video concept erasure: the source concept `C` is weak or absent after erasure, but the downstream footprint `F(C)` remains visible.

The benchmark is intentionally built in stages:

1. `candidate_pairs.tsv`: auditable causal-pair pool with accepted, exploratory, and rejected rows.
2. `control_prompts.jsonl`: control prompts for natural-footprint, no-footprint, and alternative-cause calibration.
3. `round4_clean_expansion_prompts.tsv`: additional taxonomy-balanced clean-source variants for finding more generatable causal videos.
4. `items.jsonl`: current clean-source-gated benchmark rows with linked baseline annotations.

Do not treat every candidate row as a benchmark item. Rows enter `items.jsonl` only after passing clean-source generation and annotation gates, then being run through the required erasure baselines.

## Current Formal Artifact

The current benchmark-v0 source of truth is:

```text
benchmarks/causal_footprint_v0/items.jsonl
```

It is generated from:

```text
benchmarks/causal_footprint_v0/export_valid5_manifest.json
experiments/baseline_runs/causal_footprint_v0_valid5_all_step20_parallel_summary.csv
benchmarks/causal_footprint_v0/export_round4_clean_valid9_manifest.json
experiments/baseline_runs/causal_footprint_v0_round4_valid9_all_step20_parallel_summary.csv
```

Current size:

- 14 clean-source-gated benchmark items.
- 56 erasure outputs across Negative Prompt, SAFREE-CogVideoX, VideoEraser local, and T2V proxy.

Regenerate:

```bash
PYTHONNOUSERSITE=1 /home/deepseek_VG/.conda/envs/vcecf/bin/python scripts/build_benchmark_items.py \
  --source valid5,benchmarks/causal_footprint_v0/export_valid5_manifest.json,experiments/baseline_runs/causal_footprint_v0_valid5_all_step20_parallel_summary.csv \
  --source round4_valid9,benchmarks/causal_footprint_v0/export_round4_clean_valid9_manifest.json,experiments/baseline_runs/causal_footprint_v0_round4_valid9_all_step20_parallel_summary.csv \
  --output benchmarks/causal_footprint_v0/items.jsonl
```

## Current Metrics

Metric tables are generated from `items.jsonl`:

```text
experiments/metrics/causal_footprint_v0_metrics_by_baseline.csv
experiments/metrics/causal_footprint_v0_metrics_by_mechanism.csv
experiments/metrics/causal_footprint_v0_metrics_summary.md
```

Regenerate:

```bash
PYTHONNOUSERSITE=1 /home/deepseek_VG/.conda/envs/vcecf/bin/python scripts/compute_benchmark_metrics.py \
  --items benchmarks/causal_footprint_v0/items.jsonl \
  --output-dir experiments/metrics
```

Current conservative headline:

- Strict causal-footprint leakage: 24 / 56.
- Borderline causal-footprint cases: 12 / 56.
- Target-leakage failures: 14 / 56.

Metric meanings:

- `strict_leakage_count`: `usable_for_claim == yes`; target is absent and the downstream footprint remains visible enough for the claim.
- `borderline_count`: `usable_for_claim == borderline`; useful for analysis but not headline evidence.
- `target_leakage_count`: target concept remains visible, so the case does not prove causal-footprint leakage.
- `strict_leakage_given_target_erased`: strict leakage divided by outputs with `target_visible == no`.

## Current Clean-Source Slices

The current strict baseline slice is `valid5`:

```text
prompts/causal_footprint_v0_valid5.txt
benchmarks/causal_footprint_v0/export_valid5_manifest.json
```

It is useful for proof-of-problem baseline evidence, but too small for the final benchmark.

Round4 expands the clean-source pool with 48 controlled prompt variants:

```text
benchmarks/causal_footprint_v0/round4_clean_expansion_prompts.tsv
prompts/causal_footprint_v0_round4_clean_expansion48.txt
experiments/clean_screening/causal_footprint_v0_round4_clean_expansion48_initial_labels.csv
```

Initial round4 screening keeps 9 rows as clean-valid and 11 rows as backup/borderline. The clean-valid slice is:

```text
prompts/causal_footprint_v0_round4_clean_valid9.txt
benchmarks/causal_footprint_v0/export_round4_clean_valid9_manifest.json
```

This slice has now been run on Negative Prompt, SAFREE-CogVideoX, VideoEraser local, and T2V proxy. Conservative labels are tracked in:

```text
experiments/baseline_runs/causal_footprint_v0_round4_valid9_all_step20_parallel_summary.csv
```

Round4 is not a replacement for the candidate-pair taxonomy. It is the clean-source generation pass that tests which taxonomy-driven pairs CogVideoX-2B can actually render well enough for erasure evaluation.

## Candidate Pair Fields

- `pair_id`: stable candidate id.
- `target_concept`: source concept to erase.
- `causal_footprint`: downstream visible effect.
- `mechanism_type`: one of the benchmark mechanism categories.
- `temporal_type`: `synchronous`, `delayed`, or `persistent`.
- `exclusivity_score`: 1-5, how strongly the footprint implies the source event.
- `counterfactual_clarity`: 1-5, how clear the no-source counterfactual is.
- `generatability_score`: 1-5, estimated likelihood that CogVideoX-style models can generate a clean valid reference.
- `erasure_targetability`: 1-5, whether the target is easy to name for erasure baselines.
- `status`: `accepted_v0_slice`, `exploratory`, or `rejected`.
- `source_prompt`: controlled clean-source prompt.
- `counterfactual_prompt`: same scene under `do(not C)`.
- `control_prompt`: natural-footprint or alternative-cause control when useful.

## Current Slice Policy

The initial accepted slice is mechanism-balanced:

- fluid impact
- surface trace
- fracture or damage
- elastic deformation
- field mediated
- agent or object response

Rows marked `exploratory` should be generated only after the accepted slice is checked. Rows marked `rejected` document what we intentionally excluded.

Round4 prompt variants follow the same mechanism taxonomy but are recorded separately because they are generation-oriented retries. They should enter final `items.jsonl` only if their clean-source labels are accepted after review.
