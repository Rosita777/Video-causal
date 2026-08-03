# Collision adapter: category-agnostic preservation pilot

## Question

Can a collision-erasure adapter remove the red ball and its collision footprint
without suppressing an unseen causal mechanism that was never used for
preservation training?

## Training data

- 31 collision counterfactual pairs: erase the ball and collision footprint.
- 5 screened target-only counterfactual pairs: strengthen red-ball removal without a
  collision event.
- 32 generic non-target videos: match the frozen base model's prediction.
- No waterdrop examples are included in training.

Generic videos do not need semantic screening. They provide latent and prompt
support points where the adapter is trained to preserve the frozen model's
behavior.

## Training objective

For erase rows, keep the existing dual-trajectory causal erasure objective with
background weight 4. For preserve rows, evaluate the same noisy latent and text
condition with adapters disabled and enabled, then minimize their full-latent
prediction MSE.

## Evaluation

1. Collision validation7: object removal, footprint suppression, and scene
   preservation.
2. Waterdrop specificity8: compare original and adapted motion and visually
   verify that droplets, splashes, and ripples remain.
3. Compare against the collision-only and waterdrop-enumerated preservation
   adapters under the same prompts and seeds.

The pilot succeeds if collision erasure remains materially better than the
base model while waterdrop suppression is substantially lower than the
collision-only adapter. The waterdrop set is held out from training and model
selection.

## First pilot result

The same adapter was evaluated at three checkpoints. The collision metric is
post-event motion suppression; the waterdrop metric is computed only on the
four held-out waterdrop cases in the specificity set.

| checkpoint | collision suppression | collision early MAE | waterdrop suppression |
| --- | ---: | ---: | ---: |
| 50 | 82.89% | 0.2736 | 22.72% |
| 100 | 85.82% | 0.2118 | 60.71% |
| 200 | 85.28% | 0.1673 | 73.09% |

This is a useful trade-off curve, not a final model. Longer training improves
the target erasure and target-scene stability but gradually recovers the old
generic-motion-freezing failure. The next run should increase the frozen-teacher
preservation weight and use an explicit balanced sampler, then select a single
checkpoint using collision validation only. Waterdrop remains a held-out report
metric.
