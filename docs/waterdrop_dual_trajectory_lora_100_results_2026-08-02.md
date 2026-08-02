# Waterdrop dual-trajectory LoRA at 100 steps

## Method

Earlier objectives trained only on noised clean counterfactual latents. Dual-trajectory training additionally feeds a noised factual latent that already contains the waterdrop and footprint. The adapter predicts a clean endpoint from that factual trajectory, and the endpoint is pulled toward the aligned counterfactual inside the causal residual mask.

```text
z0_pred_from_factual = zt_factual - sigma * velocity_adapter(zt_factual)
L_redirect = masked_mse(z0_pred_from_factual, z_counterfactual)
```

The full explicit-causal objective keeps:

- Ordinary counterfactual flow-matching SFT
- Weight-1 paired separation on the counterfactual trajectory
- Frozen-base distillation outside the residual on both trajectories
- Weight-0.05 factual-trajectory redirect loss

Target-only examples retain ordinary counterfactual SFT.

## Setup

- Backbone: Wan2.1-T2V-1.3B-Diffusers
- Data: the same 14 erase rows as all previous ablations
- LoRA: rank 16, alpha 16, learning rate 1e-4
- Training: 100 steps, seed 20260801
- Background weight: 1.0
- Pair weight: 1.0, margin 0.05
- Redirect weight: 0.05
- Evaluation: the same five prompts and seeds

The raw redirect loss was about 1.07 in the one-step smoke test, so weight 0.05 was chosen to place its weighted contribution near the ordinary SFT loss rather than letting it dominate training.

Final 20-step means:

- Total loss: 0.08921
- Counterfactual loss: 0.06665
- Background loss: 0.00141
- Pair loss: 0.00603
- Raw redirect loss: 0.30240

## Aggregate result

| Method | Mean early base MAE (5 cases) | Mean post-change suppression (3 causal cases) |
| --- | ---: | ---: |
| Plain LoRA | 0.15533 | 62.20% |
| Best residual-weighted SFT (mask 2) | 0.15124 | 65.78% |
| Paired separation, weight 1 | 0.15022 | 66.85% |
| Paired separation, weight 4 | 0.14172 | 64.39% |
| Dual trajectory | 0.15154 | **77.56%** |

Dual trajectory improves the causal activity proxy by 15.36 percentage points over plain LoRA and 10.71 points over paired separation, while its early-frame base MAE remains similar to the other methods.

## Per-case result

| Case | Plain suppression | Dual suppression | Change | Human result |
| --- | ---: | ---: | ---: | --- |
| Trained explicit scene | 68.46% | 81.10% | +12.64 | Waterdrop remains visible, but the dark wet ring is much weaker |
| Held-out hard surface | 61.94% | 72.16% | +10.22 | Splash is nearly absent and the remaining ring is faint |
| Held-out liquid surface | 56.19% | 79.42% | +23.23 | Waterdrop, splash, and ripple are visibly smaller and weaker |
| Unrelated footprint | 63.72% | 62.17% | -1.55 | Chalk circle remains intact |
| Clean control | 6.60% | 11.05% | +4.45 | Scene remains visually clean |

The automatic improvement is consistent across all three causal cases. Visual inspection also confirms weaker footprints in all three, rather than only a metric change. The unrelated footprint and clean controls remain semantically correct.

## Decision

Dual-trajectory redirect is the first pilot objective that gives a clear, consistent causal-footprint improvement over plain LoRA and the previous masked or paired objectives. Full object erasure is not required to retain this result: the proper claim at this stage is stronger suppression with preserved controls.

This should become the provisional main method. The next step is broader held-out evaluation using more existing test prompts. After confirming the effect is not specific to the three quick-probe scenes, tune redirect weight around 0.05 and expand training data.
