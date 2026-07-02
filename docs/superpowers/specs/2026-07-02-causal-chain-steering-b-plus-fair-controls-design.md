# Causal Chain Steering B+ Fair Controls Design

## Goal

Phase B showed a partial signal: paraphrase-averaged full-chain steering beat
the random control, but the orthogonal semantic control was weaker because it
used only one unrelated semantic pair. B+ makes the controls fair before
scaling the experiment.

## Scope

B+ keeps the Phase B method unchanged for `full_chain_steering`: cause,
mechanism, and footprint links each use three paraphrased minimal pairs whose
UNet residual directions are averaged per link. The only experimental-method
change is control construction.

## Design

The Phase B manifest builder will also add an `orthogonal_semantic` minimal-pair
list with three unrelated semantic pairs. These pairs are intentionally
non-causal and unrelated to the target scene, but they are still ordinary video
semantics. This lets the runner encode and average the orthogonal control in
the same way it averages full-chain links.

The runner will norm-match the averaged orthogonal semantic direction to the
averaged footprint direction, mirroring the random control's footprint
reference. The control remains semantically unrelated, but no longer has a
weaker single-pair construction or arbitrary residual magnitude.

## Success Gate

Run the same first B+ cell as Phase B:

```text
alpha = 0.25
timestep_window = 3:6
limit_items = 3
conditions = target_negative, target_footprint_negative, full_chain_steering,
             random_direction, orthogonal_semantic
```

B+ is promising only if `full_chain_steering` has more strict causal-footprint
leakage than both `random_direction` and the paraphrase-averaged,
norm-matched `orthogonal_semantic` control, without increasing target leakage
or showing a large low-level proxy imbalance.

## Non-Goals

B+ does not expand the sample size, change alpha/window, change the evaluator,
or introduce learned counterfactual prompt generation. Those come only after
the control comparison is defensible.
