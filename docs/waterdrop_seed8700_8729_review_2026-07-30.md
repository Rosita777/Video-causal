# Waterdrop seeds 8700-8729 review (2026-07-30)

## Summary

- Factual candidates generated: 30
- Factual videos passing first screening: 11
- Aligned targets produced: 11
- Final aligned pairs accepted: 10
- Current cumulative accepted pairs: 55

GPU 2 generated seeds 8700-8714 and GPU 3 generated seeds 8715-8729. This batch was screened with the same fixed-camera and clean-prefix criteria as the preceding batches.

## Screening result

The strongest new samples are the pond, cup, bucket, puddle, tile, tray, and glass-surface scenes. Cardboard, sponge, and pale wood are retained as lower-level absorption/footprint examples. Flexible receivers remain unreliable: the leaf, petal, fabric, film, grass, and paper-strip candidates were rejected because their causal bend or rebound was not clear.

The soil candidate `wdu019` looked plausible in the contact sheet but failed numerical stability: first-frame-to-reference MAE was `4.38 / 255`, versus `0.60-1.39 / 255` for the accepted pairs. It is rejected and retained only in diagnostics.

## Result

The 10 accepted targets remove the droplet and its downstream splash, ripple, or wet footprint while preserving a stable receiver/background. Together with the previous batches, the pilot aligned dataset now contains 55 pairs.

Detailed factual screening is in `data/waterdrop_scene_probe30_seed8700_review.csv`. Pair-level metrics and final decisions are in `data/waterdrop_seed8700_pair_results.csv`. Contact-sheet diagnostics are stored under `results/waterdrop_seed8700_aligned_pair_diagnostics/`.

## Dataset progress

- First static batch: 6 pairs
- Seeds 8400-8429 batch: 16 pairs
- Seeds 8500-8529 batch: 12 pairs
- Seeds 8600-8629 batch: 11 pairs
- Seeds 8700-8729 batch: 10 pairs
- Total: 55 pairs
