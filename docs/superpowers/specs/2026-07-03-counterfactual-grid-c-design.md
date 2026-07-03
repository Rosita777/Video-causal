# Method C Counterfactual Grid Design

## Goal

Method C turns the failed B2 attention-mask result into a stricter
counterfactual controllability audit. It does not claim repair. It asks whether
the text-to-video model can separately realize target presence and causal
footprint presence under paired seeds.

## Fable Review Constraints

`claude-fable-5` flagged four attacks that the implementation must make visible:

- VLM circularity: verifier labels are evidence, not ground truth.
- Seed control weakness: same seed does not guarantee non-footprint regions stay fixed.
- Negative-control weakness: an undisturbed prompt may simply reflect model priors.
- Search leakage: verifier-guided prompt search can become prompt hacking.

The first implementation therefore records expected target/footprint states,
variant roles, and review artifacts, but reports C0 as an audit. Any later C1
search must keep rejected candidates and human-review hooks.

## C0 Variant Grid

For each probe item, build four paired prompt variants with the same seed:

- `original`: target present and footprint present.
- `remove_target`: target absent and footprint absent. Use the existing
  counterfactual prompt when available.
- `footprint_only`: target absent and footprint present. Use the existing
  control prompt when available.
- `target_only`: target present and footprint absent. Use a conservative
  generated prompt that keeps the target visible but asks for no contact,
  impact, disturbance, or named footprint.

Each row records `expected_target_visible`, `expected_footprint_visible`,
`variant_role`, `seed`, `prompt`, `target_concept`, `causal_footprint`, and
`video_path`.

## Runner And Review

The C0 runner reads a probe manifest and writes a `generation_manifest.json`.
In dry run, it only writes prompts and metadata. In real mode, it generates
ZeroScope videos using the same deterministic denoising path already used by
the attention probe. A companion review builder converts the generated manifest
into the existing VLM review CSV shape so `evaluate_v2_baseline_with_vlm.py`
can label target and footprint separately.

## Success Boundary

C0 succeeds only if the infrastructure works: the four-variant grid is
deterministic, generated videos are reviewable, and fable labels can be
summarized by variant. A positive scientific claim requires later evidence
that `remove_target` and `footprint_only` can be judged reliably and that
variant differences are not dominated by background drift.
