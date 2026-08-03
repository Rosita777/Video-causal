# Collision causal-subspace feasibility

## Setup

- Wan transformer block 15 at diffusion sigma 0.5.
- 64 highest-residual spatiotemporal tokens per video pair.
- Subspace training collisions exclude four receiver families: paper cups,
  short tins, wide dominoes, and wood pegs.
- Generic motion is split into 24 training and 8 held-out videos.
- All 31 waterdrop pairs are report-only and are not used to learn the basis.
- The object anchor is learned from five screened target-only red-ball pairs.

## Results

| representation | collision vs generic AUC | collision vs waterdrop AUC |
| --- | ---: | ---: |
| positive-only PCA | 0.929 | 0.410 |
| discriminative motion subspace | 1.000 | 0.673 |
| object-anchored discriminative subspace | 1.000 | 0.774 |

The receiver-held-out results show that generic-motion rejection is necessary
and that a target-object anchor adds useful mechanism specificity. However, a
static product of object and motion scores does not fully separate collision
from another causal mechanism. This is a feasibility signal, not a completed
method result.

## Next Method Step

Replace the static anchor product with temporal causal propagation. Activation
starts at target-object tokens and propagates through selected Wan self-attention
links only toward later tokens. The propagated causal cone, rather than raw
motion similarity, should define where the erasure adapter may act. The next
feasibility check is whether propagated activation separates held-out collision
receivers from waterdrop better than the 0.774 anchored baseline.
