# Waterdrop paired-separation LoRA at 100 steps

## Method

The paired-separation objective uses each aligned factual/counterfactual pair directly. It keeps ordinary counterfactual flow-matching SFT, preserves the frozen-base prediction outside the causal residual, and adds a hinge loss inside the residual:

```text
L_pair = M * relu(margin + error_to_counterfactual - error_to_factual)
L = L_counterfactual + L_background + L_pair
```

The added term requires the adapter prediction to be closer to the clean counterfactual target than to the original factual causal target. This differs from residual-weighted SFT, which only changes the magnitude of the existing positive loss.

## Setup

- Backbone: Wan2.1-T2V-1.3B-Diffusers
- Data: the same 14 erase rows as the earlier ablations
- LoRA: rank 16, alpha 16, learning rate 1e-4
- Training: 100 steps, seed 20260801
- Pair weight: 1.0
- Pair margin: 0.05
- Background weight: 1.0
- Evaluation: the same five prompts and seeds

The final 20-step means were 0.07421 total loss, 0.06657 counterfactual loss, 0.00156 background loss, and 0.00607 paired-separation loss. The pair term remained active through the end of training.

## Results

| Method | Mean early base MAE (5 cases) | Mean post-change suppression (3 causal cases) | Human full-erasure success |
| --- | ---: | ---: | ---: |
| Plain LoRA | 0.15533 | 62.20% | 0/3 |
| Best residual-weighted SFT (mask 2) | 0.15124 | 65.78% | 0/3 |
| Paired separation | 0.15022 | 66.85% | 0/3 |

Per-case paired-separation results:

| Case | Early base MAE | Post-change suppression | Human result |
| --- | ---: | ---: | --- |
| Trained explicit scene | 0.19651 | 70.69% | Impact object and ring remain |
| Held-out hard surface | 0.07543 | 71.06% | Early splash is slightly weaker, but splash and ripple remain |
| Held-out liquid surface | 0.17020 | 58.79% | Waterdrop, splash, and expanding ripple remain |
| Unrelated footprint | 0.23606 | 61.32% | Chalk circle preserved |
| Clean control | 0.07290 | -1.36% | Clean scene preserved |

The automatic causal activity proxy improves by 4.65 percentage points over plain LoRA and 1.07 points over mask-2 SFT. Visual inspection supports only a small weakening on the held-out hard surface. It does not support successful full erasure.

## Decision

Paired separation is a more useful direction than residual loss weighting because it adds a genuinely different constraint and produces a small consistent signal without breaking the controls. However, the current setting still fails the strict goal on all three causal scenes.

Do not claim successful causal erasure from this pilot. The next controlled experiment should increase only the pair contribution while keeping data, margin, background weight, LoRA settings, prompts, and seeds fixed. This tests whether the weak result is caused by insufficient pair strength before changing the objective again.
