# Causal Footprint Benchmark V2

V2 is a larger clean-source candidate pool for CogVideoX-2B screening.
It keeps all v1 candidate rows and appends targeted expansions after the v1
chunked GPT-5.4 audit showed that 150 rows yielded only 72 clean-source candidates.

The expansion is intentionally biased toward weak v1 buckets:
`fracture_damage`, `surface_trace`, and `elastic_deformation`.

Files:

- `candidate_items.jsonl`: machine-readable candidate rows.
- `candidate_pairs.tsv`: tabular candidate metadata.
- `controls_specs.jsonl`: three controls per candidate.
- `export_candidate_manifest.json`: prompt export manifest.
- `export_controls_manifest.json`: control export manifest.

Mechanism counts:

- `elastic_deformation`: 59
- `field_mediated`: 34
- `fluid_impact`: 38
- `fracture_damage`: 72
- `particle_dispersion`: 30
- `surface_trace`: 71

V2 rows are not final benchmark rows. They must pass clean-source chunked VLM
triage and human adjudication before controls and erasure baselines are run.
