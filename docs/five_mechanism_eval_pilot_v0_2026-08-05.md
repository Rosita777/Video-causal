# Five-mechanism erasure pilot v0

## Purpose

This pool is an internal feasibility screen before the paper benchmark is frozen. It tests whether the current adapter method can beat erasure baselines on five simple, visually auditable causal mechanisms.

## Fixed mechanisms

| Mechanism | Erased target | Causal footprint |
| --- | --- | --- |
| Waterdrop impact | one large clear water droplet | splash, ripple, or wet spot |
| Ball collision | one small red rubber ball | receiver falls after contact |
| Brittle fracture | one small black steel ball | cracks and fragments |
| Particle impact | one small blue rubber ball | scattered particles and depression |
| Surface trace | one small yellow toy car | persistent wheel tracks |

Each mechanism contains 30 evaluation candidates with one generation per prompt. Wan and CogVideoX receive the same prompt list.

## Freeze protocol

1. Generate all candidates independently on Wan and CogVideoX.
2. Screen the full video for target visibility, footprint visibility, temporal order, and basic video quality.
3. Freeze 10 valid cases per mechanism per backbone.
4. Also record the prompt intersection that is valid on both backbones.
5. Never use any candidate marked as a frozen evaluation case for adapter training.

The candidate CSV is `data/five_mechanism_eval_candidates_v0.csv`. The pipe-delimited generation input is `prompts/five_mechanism_eval_candidates_v0.prompts`.

Before full generation, `prompts/five_mechanism_eval_smoke10_v0.prompts` runs two cases from every mechanism on each backbone.

## Pilot metrics

The first comparison uses four human-readable binary judgments:

- target absent;
- causal footprint absent;
- receiver preserved;
- full success: all three conditions pass.

Motion suppression and frame MAE remain diagnostic proxies only. They are not final semantic success metrics.
