# Waterdrop redirect and inference-scale sweep

Date: 2026-08-02

## Why this sweep

The dual-trajectory LoRA at redirect weight `0.05` improved causal suppression,
but changed the scene before the causal event more than the plain baseline. We
tested both the training loss weight and the inference LoRA scale.

## Quick5 redirect-weight sweep

The fixed five-video probe contains three removal cases and two controls.

| Setting | Removal suppression | Removal early MAE | Control early MAE |
| --- | ---: | ---: | ---: |
| Plain | 62.20% | 0.1504 | 0.1627 |
| Redirect `0.025` | 66.49% | 0.1505 | 0.1630 |
| Redirect `0.05` | **77.56%** | 0.1496 | **0.1545** |
| Redirect `0.10` | 80.64% | 0.1516 | 0.1634 |

The lower weight loses too much erasure. The higher weight gains only 3.08
points over `0.05` while worsening preservation, so `0.05` remains the training
default.

## Held-out eval20 inference-scale test

The same `0.05` checkpoint was evaluated at full scale `1.0` and conservative
scale `0.75` on the frozen balanced 20-case set.

| Setting | Causal suppression | Early-frame MAE |
| --- | ---: | ---: |
| Plain | 74.90% | 0.1338 |
| Dual trajectory, scale `1.0` | **85.44%** | 0.1447 |
| Dual trajectory, scale `0.75` | **81.69%** | **0.1242** |

Scale `0.75` is 3.75 points below maximum suppression, but still improves over
plain by 6.79 points. Its early-frame MAE is 14% lower than scale `1.0`, and
visual sheets show less global color/background drift. The gain is present in
all four groups:

| Group | Plain | Scale `0.75` |
| --- | ---: | ---: |
| Explicit causal, liquid surface | 54.64% | 70.18% |
| Explicit causal, hard surface | 81.72% | 86.24% |
| Implicit causal, liquid surface | 74.95% | 80.07% |
| Implicit causal, hard surface | 88.30% | 90.29% |

## Decision

Keep the redirect weight at `0.05`. Use inference scale `0.75` as the default
when preservation matters, and report scale `1.0` as the maximum-erasure setting.
This is a practical operating-point choice, not a new training method.

Raw quick5 metrics, full eval20 metrics, per-case scale comparisons, and contact
sheets are stored under `experiments/pilot_week1/`.
