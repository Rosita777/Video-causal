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

## Factual Anchor And Temporal Propagation

The follow-up replaces the pair-difference anchor with a deployable linear
detector trained only on factual Wan block-15 features from four target-only
red-ball videos and 24 generic videos. Each frame retains 16 candidate tokens.
Activation propagates forward using adjacent-frame hidden-feature affinity and
spatial proximity.

| metric | direct anchor | propagated cone |
| --- | ---: | ---: |
| collision vs generic AUC | 1.000 | 1.000 |
| collision vs waterdrop AUC | 0.995 | 0.995 |
| collision late-footprint coverage | 68.9% | 85.0% |
| waterdrop late false coverage | 6.8% | 11.6% |

Temporal propagation substantially expands coverage of the target collision
footprint, while false waterdrop coverage remains much lower than collision
coverage. Its increase from 6.8% to 11.6% shows that transition constraints
still need improvement.

This experiment may benefit from the repeated `red ball` prompt phrase through
Wan's text-conditioned hidden states. Before treating it as method evidence,
the detector must be tested on other-colored-ball collisions, red-ball negation
prompts, and prompts mentioning a red ball when the generated video does not
contain one.

## Strict Control Audit

The strict rerun removes low-motion generic clips from evaluation. The 32
generic clips are ranked by measured motion: ranks 1--16 train the detector,
ranks 17--24 are held out, and ranks 25--32 are ignored. Manual review retains
six clear other-colored-ball collisions and two red-ball-negation generations
where no red ball is visible.

| control | direct AUC | propagated AUC | propagated coverage |
| --- | ---: | ---: | ---: |
| held-out generic motion | 1.000 | 1.000 | 0.1% |
| held-out waterdrop | 0.986 | 0.986 | 12.6% |
| other-colored-ball collision | 0.909 | 0.909 | 46.2% |
| red-ball prompt, no visible red ball | 1.000 | 1.000 | 28.8% |

The target collision reaches 82.0% propagated coverage. The negation control
argues against a purely text-driven detector, and waterdrop remains well below
the target. However, the 46.2% coverage on other-colored-ball collisions is too
high: the current cone partly follows the collision mechanism without requiring
enough target-object identity. This is positive feasibility evidence, but not
yet sufficient specificity for adapter training.

## Immediate Next Step

Factor the cone into two explicit gates:

1. a target-object gate, calibrated with target-only positives and visually
   similar non-target objects;
2. a forward causal-propagation gate, calibrated with factual/counterfactual
   collision pairs.

The adapter may act only on their intersection. The next ablation should compare
motion-only, object-only, the current soft product, and the gated intersection.
The acceptance target is to retain roughly 80% target collision coverage while
reducing other-colored-ball collision coverage substantially below 46.2%.

## Dual-Gate Ablation

The follow-up separates the representation into an object detector and a
mechanism detector. The object detector uses four target-only clips, 16 generic
motion clips, and three other-colored-ball collisions as hard negatives. The
remaining three other-colored-ball collisions are held out. The mechanism
detector uses 24 collision clips and is evaluated on seven clips from held-out
receiver families. Gate parameters are selected using only training collisions
and generic motion, targeting 80% training coverage; none of the reported
control groups participates in this selection.

| method | target collision | generic motion | waterdrop | other-colored ball | no visible red ball |
| --- | ---: | ---: | ---: | ---: | ---: |
| motion only | 100.0% | 0.0% | 99.9% | 100.0% | 0.0% |
| object only | 50.8% | 0.3% | 2.4% | 37.4% | 5.6% |
| soft product | 49.0% | 0.0% | 2.3% | 36.8% | 0.0% |
| calibrated gated cone | 63.1% | 0.0% | 9.4% | 17.8% | 0.0% |

Motion alone cannot distinguish causal mechanisms. Object anchoring supplies
most of the specificity, while the calibrated temporal cone recovers additional
target-footprint coverage and reduces held-out other-ball coverage from 36.8%
to 17.8%. Relative to the earlier ungated propagated cone, target coverage drops
from 82.0% to 63.1%, but other-ball coverage drops from 46.2% to 17.8%.

This is a useful precision/coverage trade-off, not a final method result. The
hard-negative split contains only three training and three test videos. The next
experiment should apply the gate as a soft spatial-temporal weight during LoRA
training and measure actual object erasure, footprint erasure, and preservation
on generated videos.
