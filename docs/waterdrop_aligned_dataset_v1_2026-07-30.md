# Waterdrop aligned dataset v1 (2026-07-30)

## What it contains

`data/waterdrop_aligned_pairs_v1.csv` contains 55 reviewed factual/counterfactual pairs.

For every pair:

- the factual video contains a falling water droplet and a visible downstream effect;
- the counterfactual target repeats a clean pre-contact reference frame;
- the target therefore removes both the droplet and its splash, ripple, crater, or wet footprint;
- factual and counterfactual videos have the same frame count, resolution, and camera composition.

The manifest records the factual video, target video, clean reference image, receiver, seed, clean-prefix boundary, and background-stability score.

## How to rebuild the manifest

```bash
python3 scripts/build_waterdrop_aligned_manifest.py
```

The builder reads the final decision from every batch's pair-result CSV, includes only `pass` rows, and verifies that every referenced artifact exists.

## Important limitations

This is an engineering pilot dataset, not yet a benchmark split.

- The 55 pairs are seed variations over a much smaller set of receiver types.
- Reliable samples are concentrated in water surfaces, rigid surfaces, and simple wet footprints.
- Flexible receivers are mostly absent because Wan did not reliably generate bending or rebound.
- The static counterfactual target is appropriate only for fixed-camera scenes with a stable clean prefix.
- No train/validation/test assignment has been made yet.

The next step is to define a receiver-level split before adapter training. Videos from the same prompt family must not be placed on both sides of a generalization test.
