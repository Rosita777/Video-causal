# Waterdrop paired-separation weight-4 LoRA at 100 steps

## Question

Was the weight-1 paired-separation result weak only because the new pair term was too small?

## Controlled setup

This run changes only the pair weight from 1.0 to 4.0. Data, backbone, LoRA settings, learning rate, 100 training steps, pair margin 0.05, background weight 1.0, prompts, seeds, and generation settings are unchanged.

The final 20-step means were:

- Total loss: 0.09099
- Counterfactual loss: 0.06757
- Background loss: 0.00215
- Raw pair loss: 0.00532, multiplied by weight 4 in the total objective

## Aggregate result

| Method | Mean early base MAE (5 cases) | Mean post-change suppression (3 causal cases) | Human full-erasure success |
| --- | ---: | ---: | ---: |
| Plain LoRA | 0.15533 | 62.20% | 0/3 |
| Paired separation, weight 1 | 0.15022 | 66.85% | 0/3 |
| Paired separation, weight 4 | 0.14172 | 64.39% | 0/3 |

Weight 4 produces the lowest early-frame base MAE, but it does not improve the causal activity proxy over weight 1. The automatic metric should not be interpreted alone because it mixes event activity with other temporal changes.

## Per-case result

| Case | Early base MAE | Post-change suppression | Human result |
| --- | ---: | ---: | --- |
| Trained explicit scene | 0.18128 | 70.84% | Dark impact object and expanding ring remain |
| Held-out hard surface | 0.07161 | 63.84% | Splash and ripple become visibly smaller, but remain present |
| Held-out liquid surface | 0.16254 | 58.50% | Waterdrop, splash, and expanding ripple remain |
| Unrelated footprint | 0.22243 | 65.02% | Chalk circle preserved |
| Clean control | 0.07073 | -12.30% | Clean scene remains visually clean, with slightly more temporal drift than the frozen base |

The held-out hard-surface scene is the clearest local qualitative improvement observed so far. It is not repeated on the trained or held-out liquid scenes, so it does not establish general causal-footprint erasure.

## Decision

Increasing pair strength alone does not solve the problem. Do not continue a scalar sweep to weight 8 or above.

The likely limitation is where the constraint is applied. Both weight-1 and weight-4 models are evaluated only on noised counterfactual latents during training. The loss compares counterfactual and factual targets at that clean trajectory point, but inference may follow a factual waterdrop trajectory from pure noise.

The next objective should train on both trajectories with shared noise and timestep:

1. Counterfactual trajectory: retain ordinary counterfactual denoising.
2. Factual trajectory: feed a noised factual latent and explicitly redirect its causal-region prediction toward the counterfactual velocity.
3. Outside the causal mask: keep frozen-base distillation.

This dual-trajectory intervention tests whether the model can be pulled off an already-forming causal chain, rather than only learning around the clean target trajectory.
