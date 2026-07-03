# C0 Counterfactual Grid Scorer Design

## Goal

Turn Method C0 VLM predictions into clear variant-level and item-level scores so
we can screen valid causal items before scaling the experiment.

## Scope

The scorer consumes a `vlm_predictions.csv` produced from a C0 review run. It
does not generate videos, call VLMs, or alter prompts. It only checks whether
each of the four C0 variants matches its expected target/footprint state:

- `original`: target visible, footprint visible
- `remove_target`: target absent, footprint absent
- `footprint_only`: target absent, footprint visible
- `target_only`: target visible, footprint absent

## Outputs

The scorer writes three files to an output directory:

- `c0_variant_scores.csv`: one row per predicted video with expected state,
  observed state, match flags, and `variant_pass`.
- `c0_item_scores.csv`: one row per item with per-variant pass flags,
  `original_valid`, `counterfactual_pass`, `c0_grid_pass`, and a compact failure
  mode.
- `c0_summary.json`: aggregate counts for total items, valid originals, full
  grid pass, and per-variant pass rates.

## Scoring Rules

Boolean VLM labels are normalized from common yes/no strings. A variant passes
when both target visibility and footprint visibility match the expected state.
By default, rows with `video_quality=no` fail even if the semantic labels match.

An item is `original_valid` only if the `original` variant passes. This is the
base-validity gate for larger experiments. An item is `counterfactual_pass` if
`remove_target`, `footprint_only`, and `target_only` all pass. The stricter
`c0_grid_pass` requires both `original_valid` and `counterfactual_pass`.

## Failure Modes

The item-level failure mode is deliberately simple:

- `missing_variants` if any of the four C0 variants are absent.
- `invalid_original` if the original cell fails.
- `failed:<variants>` if the original is valid but one or more counterfactual
  cells fail.
- `pass` if the full grid passes.

## Tests

Tests cover boolean normalization, variant scoring, item aggregation, CLI output,
and the current pilot's expected high-level counts. The script follows existing
project style: standalone Python under `scripts/`, importable by tests via
`importlib.util`, and subprocess-tested as a CLI.
