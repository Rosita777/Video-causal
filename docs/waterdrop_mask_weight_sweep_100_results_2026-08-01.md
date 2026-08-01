# Waterdrop mask-weight sweep at 100 steps

## Question

Does increasing the residual-mask weight improve removal of the waterdrop and its causal footprint while preserving the rest of the video?

## Setup

- Backbone: Wan2.1-T2V-1.3B-Diffusers
- Data: the same 14 erase rows for every run
- LoRA: rank 16, alpha 16, learning rate 1e-4, 100 steps
- Background distillation weight: fixed at 1.0
- Residual-mask weights: 0, 1, 2, and 4
- Evaluation: the same five prompts and seeds for every adapter
- Causal cases: one trained scene and two held-out scenes
- Controls: one unrelated footprint and one already-clean scene

## Automatic probes

`early_base_mae` compares frames 0-16 with the same-seed frozen-base video. Lower is better for preservation. `post_change_suppression` measures how much less the generated video changes from its opening state than the frozen-base video. It is only an activity proxy and cannot establish successful causal erasure.

| Method | Mean early base MAE (5 cases) | Mean post-change suppression (3 causal cases) | Human full-erasure success |
| --- | ---: | ---: | ---: |
| Plain LoRA | 0.15533 | 62.20% | 0/3 |
| Mask 0 + background 1 | 0.15240 | 63.79% | 0/3 |
| Mask 1 + background 1 | 0.15757 | 62.53% | 0/3 |
| Mask 2 + background 1 | 0.15124 | 65.78% | 0/3 |
| Mask 4 + background 1 | 0.14707 | 65.39% | 0/3 |

Mask 2 has the largest automatic activity reduction, while mask 4 has the lowest early-frame base MAE. The differences are small and are not supported by the visual result.

## Visual result

Across all four mask weights:

- The trained tile scene still contains the dark impact object and persistent ring.
- The held-out hard surface still contains the splash and ripple.
- The held-out liquid surface still contains the waterdrop, splash, and expanding ripple.
- The unrelated chalk circle is preserved.
- The clean control remains clean.

Therefore, all four mask settings fail the strict goal of removing both the object and its causal footprint. The comparison sheets are stored in `experiments/pilot_week1/waterdrop_mask_weight_sweep_100_sheets/`.

## Why the sweep is insensitive

The current residual mask only reweights the ordinary counterfactual SFT error:

`remove_loss = mean(element_loss * (1 + mask_weight * mask))`

Only seven of the 14 rows use a residual mask, and their mean mask coverage is about 0.11-0.13. Increasing the weight changes the relative loss magnitude, but it does not introduce a new constraint that explicitly separates the factual causal video from the desired counterfactual video.

The final 20-step losses increase normally with the loss weight: 0.06806, 0.07239, 0.07675, and 0.08551 for weights 0, 1, 2, and 4. This confirms that the mask is active in training, but the visual behavior is almost unchanged.

## Decision

Do not spend more GPU time tuning this mask weight. The sweep provides a useful negative ablation: background distillation helps preservation incrementally, but residual-weighted SFT does not provide a meaningful removal gain.

The next training objective should use each factual/counterfactual pair directly: keep the counterfactual denoising target, preserve the outside-mask prediction, and add an explicit paired separation term inside the causal region. That objective must first beat plain SFT on the same three-scene probe before scaling the dataset or training duration.
