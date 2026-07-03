# C0 Base-Validity Screening Design

## Goal

Add a cheap screening stage before full C0 counterfactual grids. The screen
generates and evaluates only the `original` cell, then keeps items whose base
video visibly contains both the target and the causal footprint.

## Motivation

The first C0 pilot showed that two of three items fail before any intervention:
their `original` videos do not contain both required causal states. Running the
full four-cell grid on such items wastes GPU and makes the method look worse for
the wrong reason. We need a base-validity gate.

## Runner Change

`scripts/adapters/run_c0_counterfactual_grid.py` gains `--variant-set`:

- `all` keeps the current four-cell behavior and remains the default.
- `original` generates only the original target-plus-footprint cell.

The manifest still records `variant_grid`, so downstream tools can tell whether
a run is a screen or a full grid.

## Scorer Change

`scripts/score_c0_counterfactual_grid.py` keeps its existing outputs and adds
`c0_valid_originals.csv`, containing only item rows with `original_valid=true`.
This is the handoff table for selecting items into the full C0 grid.

## Workflow

1. Run C0 runner with `--variant-set original` over a larger candidate slice.
2. Build review rows and evaluate with fable.
3. Run the scorer.
4. Use `c0_valid_originals.csv` to choose the full-grid candidates.

This keeps base-model controllability separate from counterfactual separability.
