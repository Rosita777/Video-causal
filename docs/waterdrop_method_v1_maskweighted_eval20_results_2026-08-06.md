# Waterdrop Method v1 Mask-Weighted Eval20

Date: 2026-08-06

## Change

The dual-trajectory counterfactual flow-matching loss is now weighted by the factual/counterfactual causal residual mask. This prevents the small causal region from being diluted by the full-frame average.

Training used 14 erase and 14 preserve rows for 150 steps. Evaluation used the same 20 held-out prompts and seed 9100 as the previous balanced run. Inference LoRA scale was 0.75.

## Results

| Run | Samples | Mean footprint suppression | Mean early-frame MAE |
| --- | ---: | ---: | ---: |
| Previous balanced dual trajectory | 20 | 32.68% | 0.0305 |
| Mask-weighted dual trajectory | 20 | 81.41% | 0.0791 |

Breakdown:

| Prompt condition | Receiver family | Suppression |
| --- | --- | ---: |
| Explicit causal | Liquid surface | 70.06% |
| Explicit causal | Hard surface | 86.59% |
| Implicit causal | Liquid surface | 79.53% |
| Implicit causal | Hard surface | 89.47% |

## Human check

Contact sheets are in `experiments/waterdrop_method_v1_maskweighted_eval20_sheets/`. Representative hard-surface and liquid-surface cases show that the droplet and downstream ripples are mostly removed. The weakest liquid cases retain faint residual structure. Receiver geometry is preserved, but mild brightness and color changes remain.

## Decision

Mask-weighted dual trajectory is the current waterdrop operating point. It provides a large erasure improvement but increases early-frame drift. Future tuning should target preservation without reducing the causal-mask supervision strength. This result is strong enough to proceed to the next mechanism-specific adapter while keeping preservation as a required ablation and evaluation axis.
