# Waterdrop seeds 8500-8529 review (2026-07-30)

## Summary

- Factual candidates generated: 30
- Factual videos passing causal-chain screening: 13
- Aligned targets produced: 13
- Final aligned pairs accepted: 12
- Current cumulative accepted pairs: 34

The generation settings match the previous batch. GPU 2 generated seeds 8500-8514 and GPU 3 generated seeds 8515-8529.

## Important rejection after pair construction

`wds019` (soil, seed 8519) looked acceptable in the contact sheet, but its supposedly clean prefix was not stable. Its first-frame-to-reference mean absolute error was `6.42 / 255`, while the accepted samples in this batch range from about `0.63` to `1.99 / 255`.

The soil pair is therefore rejected from aligned training supervision. This shows why visual review and a simple numerical stability check should both be used.

## Result

The 12 accepted pairs contain no visible droplet or causal-footprint residue, preserve receiver geometry, and have essentially static encoded targets.

Detailed factual screening is in `data/waterdrop_scene_probe30_seed8500_review.csv`. Pair-level metrics and decisions are in `data/waterdrop_seed8500_pair_results.csv`. Diagnostic contact sheets, including the rejected soil example, are stored under `results/waterdrop_seed8500_aligned_pair_diagnostics/`.

## Dataset progress

- First static batch: 6 pairs
- Seeds 8400-8429 batch: 16 pairs
- Seeds 8500-8529 batch: 12 pairs
- Total: 34 pairs
- Target before first training run: about 50 pairs
