# Causal Footprint Benchmark v0

Status: candidate-pair construction stage.

This benchmark targets causal footprint leakage in video concept erasure: the source concept `C` is weak or absent after erasure, but the downstream footprint `F(C)` remains visible.

The benchmark is intentionally built in stages:

1. `candidate_pairs.tsv`: auditable causal-pair pool with accepted, exploratory, and rejected rows.
2. `control_prompts.jsonl`: control prompts for natural-footprint, no-footprint, and alternative-cause calibration.
3. `items.jsonl`: final clean-source-gated benchmark rows, created only after candidate screening.

Do not treat every candidate row as a benchmark item. The first runnable slice should use rows marked `accepted_v0_slice`, then pass clean-source generation and annotation gates before entering baseline evaluation.

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
