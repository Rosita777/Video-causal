# Waterdrop seeds 8600-8629 review (2026-07-30)

## Summary

- Factual candidates generated: 30
- Factual videos passing first screening: 12
- Aligned targets produced: 12
- Final aligned pairs accepted: 11
- Current cumulative accepted pairs: 45

The generation settings match the previous two batches. GPU 2 generated seeds 8600-8614 and GPU 3 generated seeds 8615-8629.

## Screening result

The reliable samples are still concentrated in water surfaces and rigid receivers. The model continues to fail on flexible receivers such as leaves, petals, fabric, plastic film, grass, and paper strips: it generates water motion but does not reliably generate the receiver's bend or rebound.

All 12 constructed targets passed the numerical clean-prefix stability check. Their first-frame-to-reference mean absolute errors range from `0.65` to `1.77 / 255`.

`wdt022` (coffee grounds, seed 8622) was rejected after final visual review. The receiver resembles seeds rather than coffee grounds, and the generated footprint becomes an implausible black liquid pool. Its diagnostics are retained to document the rejection.

## Result

The 11 accepted counterfactual targets remove the droplet and its downstream splash, ripple, or wet footprint while preserving a stable receiver/background.

Detailed factual screening is in `data/waterdrop_scene_probe30_seed8600_review.csv`. Pair-level metrics and final decisions are in `data/waterdrop_seed8600_pair_results.csv`. Contact-sheet diagnostics are stored under `results/waterdrop_seed8600_aligned_pair_diagnostics/`.

## Dataset progress

- First static batch: 6 pairs
- Seeds 8400-8429 batch: 16 pairs
- Seeds 8500-8529 batch: 12 pairs
- Seeds 8600-8629 batch: 11 pairs
- Total: 45 pairs
- Remaining before first training run: 5 pairs
