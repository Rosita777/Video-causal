# Waterdrop mask-only LoRA 100-step result

## Setup

- Backbone: Wan2.1-T2V-1.3B-Diffusers
- Adapter: LoRA rank 16, alpha 16
- Learning rate: 1e-4
- Training: 100 steps, batch size 1
- Data: the same 14 erase rows used by the plain and mask + background runs
- Explicit causal rows: residual-weighted counterfactual loss
- Target-only rows: ordinary counterfactual SFT
- Residual mask weight: 4.0
- Background distillation weight: 0.0
- Seed: 20260801

The final 20-step mean loss was 0.083915. Checkpoints were saved at steps 25, 50, 75, and 100.

## Quick evaluation

The same five prompts and seeds used by the other two variants were generated with the step-100 adapter.

- Waterdrop and footprint suppression remained partial on the trained hard surface, held-out hard surface, and held-out pond.
- The unrelated chalk circle remained visible.
- The clean control remained clean.
- Qualitatively, mask-only and mask + background had very similar removal strength and composition.

## Three-way preservation comparison

Normalized RGB MAE on pre-event frames 0-16 was computed against the same-seed frozen-base videos. Lower is better.

The mean MAE over five cases was 0.155332 for plain LoRA, 0.158890 for mask-only, and 0.147069 for mask + background. Relative to mask-only, background distillation reduced early-frame MAE in all five cases by 3.03% to 9.54%, with a mean reduction of 7.44%.

This changes the interpretation of the previous full-method pilot. Although mask + background was only mixed when compared directly with plain LoRA, the controlled mask-only ablation shows that the background term consistently recovers drift introduced by residual-weighted training.

## Decision

The background-distillation component has positive incremental evidence and should remain in the method for the next experiment.

The residual mask does not yet have positive evidence for stronger causal-chain removal on this five-case visual probe. The next experiment should test mask weights, including zero, on a larger fixed evaluation slice. The mask should be retained only if it improves removal without undoing the preservation gain.
