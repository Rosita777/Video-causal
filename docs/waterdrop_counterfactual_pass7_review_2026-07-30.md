# Waterdrop counterfactual pass-7 review (2026-07-30)

## Purpose

Test whether Wan can create a clean counterfactual for each of the seven factual scenes that passed the earlier capability screen. Each counterfactual uses the factual video's original seed.

The intended pair is:

- factual: a droplet appears and causes its causal footprint;
- counterfactual: the same receiver stays clean and still, with no droplet or footprint.

## Setup

- Model: `Wan2.1-T2V-1.3B-Diffusers`
- Resolution: 832 x 480
- Frames: 49 at 8 fps
- Steps: 25
- Guidance scale: 5.0
- Seeds: 8300, 8301, 8302, 8307, 8308, 8310, 8316
- Prompt file: `prompts/waterdrop_counterfactual_pass7.txt`
- Output videos: `outputs/waterdrop_counterfactual_pass7_wan_matched_seeds/` (not tracked by Git)

## Result

| Check | Result |
|---|---:|
| No droplet or causal footprint | 7 / 7 |
| Temporally stable clean scene | 7 / 7 |
| Receiver is clearly recognizable | 6 / 7 |
| Strictly aligned with the factual composition | 0 / 7 |

The important result is that reusing the same seed does **not** preserve the scene when the prompt changes. It sometimes keeps a broad visual style, but object shape, camera angle, framing, color, and background can all change.

Therefore these videos can be used as standalone clean-scene references, but they should **not** be treated as pixel-aligned targets for supervised erasure training.

## Per-scene decision

| Scene | Receiver | Clean counterfactual | Strict pair | Decision |
|---|---|---:|---:|---|
| wd000 | calm shallow pond | pass | fail | clean reference only |
| wd001 | glass bowl with water | pass | fail | clean reference only |
| wd002 | ceramic cup with water | pass | fail | clean reference only |
| wd007 | stainless-steel tray | pass | fail | clean reference only |
| wd008 | glass tabletop | uncertain | fail | reject |
| wd010 | ceramic plate | pass | fail | clean reference only |
| wd016 | unfinished wood | pass | fail | clean reference only |

## Implication for the data pipeline

Independent text-to-video generation is not enough to make supervised before/after pairs, even with matched seeds. The next data experiment should preserve the factual video's geometry directly, for example by video editing/inpainting or by rendering both versions from one controlled synthetic scene.

Do not start adapter training with these seven pairs as aligned supervision. First solve the pair-construction problem on one simple receiver.

Detailed labels are in `data/waterdrop_counterfactual_pass7_review.csv`. Contact sheets under `results/waterdrop_counterfactual_pass7_pairs/` show factual frames on top and counterfactual frames on the bottom.
