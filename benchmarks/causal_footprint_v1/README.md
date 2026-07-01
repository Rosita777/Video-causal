# Causal Footprint Benchmark v1 Candidate Pool

Status: generated candidate pool, not yet clean-source-gated or baseline-evaluated.

This directory expands the causal-footprint benchmark from the 47-item v0 formal set to a 150-item v1 candidate pool. The goal is to support a paper-scale benchmark before running the expensive clean-reference, control, erasure-baseline, and human-adjudication stages.

## Current Size

- 150 candidate causal items.
- 450 generated control prompts: 150 items x `no_cause`, `effect_only`, and `alternative_cause`.
- Source composition:
  - 47 items imported from the clean-source-gated v0 benchmark.
  - 97 additional unique items imported from the round6 taxonomy pool.
  - 6 supplemental legacy candidate-pair rows used to reach 150 while improving mechanism balance.
- Mechanism distribution:
  - `field_mediated`: 28
  - `fluid_impact`: 25
  - `elastic_deformation`: 25
  - `fracture_damage`: 24
  - `surface_trace`: 24
  - `particle_dispersion`: 24

## Artifacts

```text
benchmarks/causal_footprint_v1/candidate_items.jsonl
benchmarks/causal_footprint_v1/candidate_pairs.tsv
benchmarks/causal_footprint_v1/controls_specs.jsonl
benchmarks/causal_footprint_v1/export_candidate_manifest.json
benchmarks/causal_footprint_v1/export_controls_manifest.json
prompts/causal_footprint_v1_candidates.txt
prompts/causal_footprint_v1_controls.txt
```

`candidate_items.jsonl` is the easiest file to inspect programmatically. `candidate_pairs.tsv` keeps compatibility with the older candidate-pair format. The prompt files use the project-standard format:

```text
<prompt> | <target> | <effect>
```

## Regenerate

```bash
PYTHONNOUSERSITE=1 python scripts/build_benchmark_v1_candidates.py
```

The generator reads:

```text
benchmarks/causal_footprint_v0/items.jsonl
benchmarks/causal_footprint_v0/round6_taxonomy_expansion_prompts.tsv
benchmarks/causal_footprint_v0/candidate_pairs.tsv
```

and writes the v1 artifacts listed above.

## Next Gates

The v1 pool is intentionally broader than the current formal benchmark. It must pass these gates before it becomes a paper-facing formal benchmark:

1. Generate clean-reference videos for `prompts/causal_footprint_v1_candidates.txt`.
2. Human-label clean references for target visibility, footprint visibility, and causal-chain clarity.
3. Generate controls from `prompts/causal_footprint_v1_controls.txt` on each backbone.
4. Human-label controls and build the model-specific valid item subset.
5. Run all erasure baselines only on the valid subset.
6. Use VLM prelabels for triage, then human-adjudicate final target/footprint/leakage labels.

Do not report v1 candidate counts as leakage metrics. Until the gates above are complete, benchmark-v0 remains the formal evaluated artifact.
