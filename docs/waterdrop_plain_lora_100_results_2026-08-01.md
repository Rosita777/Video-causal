# Waterdrop plain LoRA 100-step result

## Setup

- Backbone: Wan2.1-T2V-1.3B-Diffusers
- Adapter: LoRA rank 16, alpha 16
- Learning rate: 1e-4
- Training: 100 steps, batch size 1
- Data: 14 erase rows from `waterdrop_train_pilot40_sft_v0.csv`
- Target: aligned clean counterfactual videos
- Objective: ordinary flow-matching SFT without mask or preservation losses
- Seed: 20260801

The training pipeline completed successfully. The final 20-step mean loss was 0.066413. Checkpoints were saved at steps 25, 50, 75, and 100. The final LoRA file is about 23.7 MB and is intentionally not tracked by Git.

## Quick evaluation

Five videos were generated with the step-100 LoRA using the same prompts and seeds as their frozen-base comparisons.

1. On a trained ceramic-tile causal prompt, the large persistent water ring became a smaller delayed dark water mark. Removal was incomplete.
2. On a held-out white stove surface, the splash and wet footprint became much smaller and later. This is positive cross-surface transfer, but not full removal.
3. On a held-out pond, the droplet and ripples were also smaller and later. This is an initial out-of-domain signal, but the causal chain remained visible.
4. An unrelated dry chalk circle remained visible. The adapter did not simply erase all circular structures.
5. A clean scene remained clean.

All cases showed noticeable composition or viewpoint drift relative to the frozen base result, including the two preservation controls.

## Decision

Plain counterfactual LoRA is a valid baseline: it learns the intended suppression direction and shows limited transfer. It is not the final method because removal is incomplete and background preservation is weak.

The next ablation should add the causal residual mask and outside-mask background distillation. This experiment provides the baseline that those components must beat.
