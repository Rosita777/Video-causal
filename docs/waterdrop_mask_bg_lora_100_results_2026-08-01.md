# Waterdrop mask + background LoRA 100-step result

## Setup

- Backbone: Wan2.1-T2V-1.3B-Diffusers
- Adapter: LoRA rank 16, alpha 16
- Learning rate: 1e-4
- Training: 100 steps, batch size 1
- Data: the same 14 erase rows used by the plain LoRA baseline
- Explicit causal rows: residual-weighted counterfactual loss plus outside-mask base distillation
- Target-only rows: ordinary counterfactual SFT
- Residual mask weight: 4.0
- Background distillation weight: 1.0
- Seed: 20260801

Seven explicit rows had latent residual masks. Their mean soft-mask weights ranged from 0.1084 to 0.1265. The other seven target-only rows did not use masks. The final 20-step mean total loss was 0.085510. Checkpoints were saved at steps 25, 50, 75, and 100.

## Quick evaluation

The same five prompts and seeds used for the plain LoRA quick evaluation were generated with the step-100 adapter.

- Partial waterdrop and footprint suppression was retained on the trained hard surface, held-out hard surface, and held-out pond.
- The unrelated chalk circle remained visible.
- The clean control remained clean.
- Background composition still visibly drifted from the frozen-base videos.

For a simple background-drift probe, early frames 0-16 were compared against the same-seed frozen-base videos with normalized RGB MAE. This interval precedes the waterdrop event in the evaluation prompts. Relative to plain LoRA, mask + background reduced early-frame MAE by 8.34% on the trained scene and 13.06% on the unrelated control, was unchanged on the two held-out causal scenes, and increased MAE by 12.85% on the clean control.

## Decision

The current background objective has a mixed result and does not yet support a preservation-improvement claim. It should not be presented as successful based on this five-case pilot.

The next controlled experiment is the masked-counterfactual variant with background weight zero. This separates the effect of residual weighting from background distillation. A later background-weight sweep is justified only if the mask-only comparison is positive.
