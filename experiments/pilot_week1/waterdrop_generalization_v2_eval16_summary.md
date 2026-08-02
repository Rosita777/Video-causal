# Waterdrop generalization v2 eval16

The 31-pair receiver-diverse training split was trained for 100 steps with the dual-trajectory objective. Evaluation uses 16 receiver-held-out causal prompts with their original fixed seeds.

## Automatic metrics

| LoRA scale | Mean post-change suppression | Mean early/base MAE |
| --- | ---: | ---: |
| 0.75 | 88.58% | 0.15053253 |
| 1.00 | 85.58% | 0.17507653 |

Scale 0.75 is preferable: it suppresses slightly more causal change while producing less early-scene drift.

## Visual finding

Across the 16 contact sheets, the adapter strongly suppresses the falling droplet and its splash, ripple, wet mark, crater, or spreading wet patch. However, the receiver appearance often changes substantially. Some cases preserve the broad receiver category but alter viewpoint, geometry, color, and texture; other cases become an overly flat or background-like scene.

Therefore this run demonstrates strong mechanism-level erasure generalization, but not satisfactory scene preservation. The next training revision should add explicit preservation/control rows rather than increasing erasure-only data again.
