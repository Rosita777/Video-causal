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
