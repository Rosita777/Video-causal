# WD010 aligned counterfactual pair (2026-07-30)

## Question

Can we build a genuinely aligned factual/counterfactual training pair for one simple waterdrop scene?

The selected scene is `wd010`: a droplet falls onto a plain white ceramic plate and produces a small splash and ripple.

## Construction

The camera and plate are static, and the event has not started during the early clean frames. We use frames 0 through 15 to estimate one clean plate image by taking their temporal median. That clean image is repeated for all 49 target frames.

- Factual input: the original seed-8310 Wan video
- Counterfactual target: the same plate stays clean and still
- Resolution: 832 x 480
- Length: 49 frames at 8 fps
- Tool: `scripts/build_static_counterfactual_pair.py`
- Reproduction: `scripts/run_waterdrop_wd010_aligned_pair.sh`

## Result

| Check | Result |
|---|---|
| Same plate position, size, and camera view | pass |
| Droplet removed | pass |
| Splash and ripple removed | pass |
| Counterfactual temporal stability | pass |
| First factual frame versus clean reference MAE | 0.769 / 255 |
| Maximum encoded counterfactual temporal MAE | 0.0013 / 255 |

This is the first valid aligned pair in the new pipeline. The contact sheet in `results/waterdrop_wd010_aligned_pair/` shows factual frames first and counterfactual frames second.

## Scope and limitation

This construction only works when:

1. the camera is fixed;
2. the receiver is static;
3. clean pre-event frames are visible;
4. the causal event starts later in the video.

It does not solve general video erasure. A repeated clean frame would incorrectly remove legitimate background motion in scenes such as a moving pond surface.

## Data-design implication

For the first training experiment, deliberately restrict the dataset to fixed-camera, static-receiver scenes with a clean prefix. This makes the supervision trustworthy and keeps the engineering problem small.

Do not mix moving-background scenes into the first adapter training run. They need a stronger counterfactual construction method later.
