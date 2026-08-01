# Training method v0: aligned causal-residual LoRA

This is a method specification for the first Wan waterdrop adapter experiment. It is provisional until the pilot ablations are run.

## Goal

Train one waterdrop adapter that suppresses both the waterdrop and its downstream footprint while preserving the receiver, unrelated effect-like structures, and clean scenes.

## Baseline

The baseline is ordinary LoRA diffusion SFT. A causal or target-only prompt is paired with a clean counterfactual target video. The adapter is active at inference while the original prompt remains unchanged.

## Aligned counterfactual target

For each fixed-camera causal video, detect the clean prefix before the falling droplet enters. Take the temporal median of those clean frames and repeat it for the full video length. This produces a target with the same camera, receiver, texture, and lighting as the factual video, but without the target or footprint.

Independently generated clean videos are retained as controls, but they are not used as the primary explicit-causal target because their composition may differ.

## Proposed full objective

The factual and aligned counterfactual latents define an automatic causal residual:

```text
M = smooth_normalize(abs(z_factual - z_counterfactual))
```

The mask follows the falling target and its downstream splash, wet patch, ring, or trail through space and time. It requires no manual per-frame segmentation.

The training objective has three parts:

```text
L = L_remove + lambda_bg * L_background + lambda_keep * L_keep
```

- `L_remove`: counterfactual diffusion loss, weighted strongly inside `M`.
- `L_background`: outside `M`, keep the adapter prediction close to the frozen base model.
- `L_keep`: on unrelated-footprint and clean prompts, keep the adapter prediction close to the frozen base model.

Target-only examples use counterfactual SFT without the residual mask in v0 because independently generated target-only videos are not pixel-aligned with the causal clean prefix.

## Why each component is necessary

- Alignment prevents the adapter from learning accidental camera or background differences.
- The residual mask focuses learning on the whole observed causal chain, not only the object token.
- Background distillation limits global scene damage.
- Unrelated and clean distillation prevents blanket removal of rings, marks, or motion.

## Required ablations

| Variant | Counterfactual target | Residual mask | Background distillation | Control distillation |
|---|---:|---:|---:|---:|
| A. Plain LoRA SFT | yes | no | no | no |
| B. Masked counterfactual | yes | yes | no | no |
| C. Mask + background | yes | yes | yes | no |
| D. Full method | yes | yes | yes | yes |

An additional data ablation should compare aligned targets against independently generated clean targets.

## Pilot decision rule

Proceed to paper-scale data only if the full method improves explicit and implicit causal-footprint removal over plain LoRA while keeping unrelated-footprint and clean preservation near the frozen base model. If the mask adds no benefit, do not retain it merely as a novelty claim.
