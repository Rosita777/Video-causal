# Waterdrop dual-trajectory LoRA: held-out eval20

Date: 2026-08-02

## Question

Does the dual-trajectory redirect objective improve causal erasure beyond the
plain paired LoRA on prompts and receivers that were not used for training?

## Setup

- Model: Wan T2V with the 100-step waterdrop LoRA checkpoints.
- Methods: plain paired LoRA and dual-trajectory LoRA.
- Test set: 20 frozen, semantically screened base videos.
- Seed: one fixed seed per case (the same seed for both methods).
- Balanced groups: explicit/implicit causal wording crossed with liquid/hard
  receiving surfaces, 5 cases per group.
- Main proxy: reduction of post-change video motion relative to the frozen base.
  Higher is better.
- Preservation proxy: early-frame MAE relative to the frozen base. Lower is
  better.

## Results

| Method | All 20 causal suppression | Early-frame MAE |
| --- | ---: | ---: |
| Plain paired LoRA | 74.90% | 0.1338 |
| Dual trajectory | **85.44%** | 0.1447 |

The dual-trajectory objective improves the main proxy by **10.54 percentage
points**. It improves 19 of 20 cases, with a median improvement of 5.12 points.
The only decrease is -0.42 points on a case where the plain method is already at
96.80% suppression.

| Held-out group | Plain | Dual trajectory | Improvement |
| --- | ---: | ---: | ---: |
| Explicit causal, liquid surface | 54.64% | **80.81%** | +26.17 |
| Explicit causal, hard surface | 81.72% | **88.35%** | +6.63 |
| Implicit causal, liquid surface | 74.95% | **81.59%** | +6.64 |
| Implicit causal, hard surface | 88.30% | **90.99%** | +2.70 |

## Visual check

The sampled contact sheets agree with the automatic proxy. Compared with the
plain LoRA, the dual-trajectory model consistently weakens the falling drop,
splash crown, and later ripple. The largest gain is on the held-out teacup case,
where the plain model leaves a strong splash and the dual-trajectory model leaves
only a faint remnant.

The current limitation is preservation. Early-frame MAE rises from 0.1338 to
0.1447, and the generated scene can shift in color or appearance even before the
causal event. The next experiment should therefore tune the redirect weight to
retain most of the suppression gain while reducing background change.

## Interpretation

This is stronger evidence than the earlier three-case pilot: the gain appears in
all four held-out groups and in 19/20 individual cases. The result supports the
dual-trajectory training direction. It is not yet a final semantic evaluation,
because the motion proxy cannot by itself distinguish correct erasure from every
kind of reduced motion. Final reporting should combine this proxy with human or
video-language-model judgments.

Raw metrics and all 20 contact sheets are stored under:

- `experiments/pilot_week1/waterdrop_dual_traj_eval20_results/`
- `experiments/pilot_week1/waterdrop_dual_traj_eval20_sheets/`
