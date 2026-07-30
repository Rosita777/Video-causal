# Waterdrop repeat seeds 8400-8429 review (2026-07-30)

## Purpose

Generate a second video for each of the existing 30 waterdrop prompts, using new seeds, and test the fixed-camera aligned-pair pipeline at a larger scale.

## Generation

- Model: Wan2.1-T2V-1.3B-Diffusers
- Seeds: 8400-8429
- Candidates: 30
- Resolution: 832 x 480
- Frames: 49 at 8 fps
- Steps: 25
- GPUs: 2 and 3, 15 videos each

## Factual screening

| Result | Count |
|---|---:|
| Pass | 16 |
| Fail | 14 |

The 16 passing videos show a stable receiver, a usable clean prefix, a visible waterdrop, contact, and a causal footprint that begins after contact.

The main failure pattern is the flexible-receiver group. Wan often generated water motion but did not make the leaf, petal, fabric, film, grass, or paper bend as requested. Other failures included a pre-existing soil crater, an unrecognizable sponge, and incorrect granular-surface effects.

## Aligned counterfactual construction

All 16 passing factual videos produced clean aligned targets.

| Check | Result |
|---|---:|
| Counterfactual pairs built | 16 / 16 |
| Visible droplet residue | 0 / 16 |
| Visible causal-footprint residue | 0 / 16 |
| Receiver geometry preserved | 16 / 16 |

Reference-prefix lengths were selected per video rather than using one fixed number. They range from 4 frames for the cutting board to 28 frames for the towel. This avoids contaminating the clean reference with an early droplet.

## Dataset progress

- Earlier validated pairs: 6
- New validated pairs: 16
- Current total: 22
- Initial target before training: about 50

Detailed factual decisions are in `data/waterdrop_scene_probe30_repeat_review.csv`. Pair metrics are in `data/waterdrop_repeat_pass16_pair_results.csv`. Contact sheets and clean references are under `results/waterdrop_repeat_pass16_aligned_pairs/`.
