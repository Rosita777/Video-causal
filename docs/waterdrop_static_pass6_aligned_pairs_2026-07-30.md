# Waterdrop static pass-6 aligned pairs (2026-07-30)

## What was tested

We applied the fixed-camera construction to the six factual scenes whose receivers are static. The pond scene `wd000` was intentionally excluded because its background water surface moves even before the drop.

For each included video, we selected only the clean prefix before the droplet entered the frame, computed a temporal median reference image, and repeated that image for all 49 target frames.

## Result

| Check | Result |
|---|---:|
| Aligned pairs produced | 6 / 6 |
| Droplet and causal footprint absent | 6 / 6 |
| Receiver geometry preserved | 6 / 6 |
| Counterfactual temporally stable | 6 / 6 |

The maximum encoded frame-to-frame variation in the six target videos is below `0.007 / 255`; this is compression noise, not visible motion. The first factual frame versus the clean reference is also very close (`0.45` to `1.10 / 255` mean absolute error).

## Included scenes

- `wd001`: glass bowl filled with water
- `wd002`: ceramic cup filled with water
- `wd007`: stainless-steel tray
- `wd008`: glass tabletop
- `wd010`: ceramic plate
- `wd016`: unfinished pale wood

The tray uses only frames 0 and 1 because the droplet enters at frame 2. This was caught during visual review and corrected before accepting the pair.

## What this establishes

The first data-construction rule is workable:

> Start with fixed-camera, static-receiver scenes where a clean prefix is visible. Use the clean prefix to construct the aligned counterfactual target.

This gives trustworthy supervision for a first adapter experiment. It does **not** cover moving backgrounds or events that begin at frame 0. The pond scene remains out of scope for this first training set.

The six pairs are still only one video per receiver. They are a validated construction prototype, not yet a sufficient training set. The next data step is to generate more factual prompt variants under the same scope and apply this construction automatically.

Detailed metrics are in `data/waterdrop_static_pass6_pair_results.csv`; contact sheets are in `results/waterdrop_static_pass6_aligned_pairs/`.
