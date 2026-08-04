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

## LoRA Integration

The dual gate is now exported as one spatiotemporal latent mask per erase scene.
Thirty-one collision scenes and five target-only scenes produce 36 non-empty
gates. After spatial dilation, the gates cover 3.45% of the patch-token grid on
average. They are generated artifacts and are not committed, but are fully
reproducible with `run_collision_dual_gate_ablation.sh`.

The new `causal_gate` training objective changes the existing dual-trajectory
loss as follows:

- counterfactual flow matching, paired separation, and factual redirection are
  optimized only inside `residual_mask * causal_gate`;
- outside that effective mask, both trajectories match the frozen Wan teacher;
- generic preservation rows continue to distill the frozen teacher over the
  complete latent.

A 10-step smoke test completed without memory or numerical errors. The formal
balanced run completed 100 steps and saved checkpoints every 25 steps at
`outputs/adapters/collision_causal_gate_100`. A checkpoint-25 single-scene probe
reduced the target collision post-event motion by 57.8%; this is only an early
motion signal and does not replace semantic object/footprint evaluation.

### Checkpoint-100 Video Results

| adapter | target motion suppression | static-control suppression | target early MAE | control early MAE |
| --- | ---: | ---: | ---: | ---: |
| general preserve, checkpoint 100 | 85.82% | 35.93% | 0.2118 | 0.1612 |
| balanced preserve, checkpoint 50 | 83.02% | 17.28% | 0.1998 | 0.1550 |
| causal gate, checkpoint 100 | 85.31% | 31.16% | 0.2232 | 0.1756 |

The gated adapter retains strong target suppression and modestly improves the
static-control motion metric relative to the earlier checkpoint-100 adapter.
It does not beat the balanced checkpoint-50 preservation result, and its
base-adapter frame divergence is higher. Therefore the current experiment
validates engineering integration, but does not yet demonstrate a superior
end-to-end adapter. The next tuning step should reduce global LoRA strength or
increase preservation pressure while retaining the gate, then select the
checkpoint on the target-versus-preservation Pareto frontier.

### Checkpoint And Scale Probe

A matched 2-target/2-control probe was used for inexpensive model selection.
These four videos are only a tuning split and do not replace the full 7/8
checkpoint-100 evaluation.

| checkpoint | LoRA scale | target suppression | control suppression | target early MAE | control early MAE |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 25 | 1.00 | 69.57% | 9.11% | 0.0844 | 0.0849 |
| 50 | 1.00 | 60.25% | 6.32% | 0.2588 | 0.0991 |
| 25 | 0.75 | 53.40% | 9.43% | 0.0614 | 0.0605 |

Checkpoint 25 at scale 1.0 is the preferred primary operating point: it has the
strongest target suppression in this probe while retaining low control
suppression. Scale 0.75 is a conservative alternative with lower base-adapter
frame divergence but substantially weaker target suppression. Checkpoint 50 is
dominated on the target and frame-divergence metrics, so checkpoint 75 and scale
0.5 were not generated. The next full semantic evaluation should use checkpoint
25 at scale 1.0 and explicitly inspect target-object removal and causal-footprint
removal rather than relying only on motion suppression.

### Checkpoint-25 Full Semantic Audit

The selected checkpoint was generated on all seven target collision prompts and
eight specificity prompts. Automatic metrics improved substantially over
checkpoint 100:

| checkpoint | target suppression | control suppression | target early MAE | control early MAE |
| ---: | ---: | ---: | ---: | ---: |
| 25 | 53.11% | 7.09% | 0.0781 | 0.0588 |
| 100 | 85.31% | 31.16% | 0.2232 | 0.1756 |

Manual contact-sheet review is less positive than the motion metric. None of the
seven target videos cleanly removes both the red ball and its collision
footprint. Five fail and two are partial: the ball is often only reduced in size
or contrast, while the receiver still falls. Among eight controls, four pass,
two are partial, and two fail. One static-can scene loses an object, and one
waterdrop scene has its drop and ripple substantially suppressed.

This establishes that checkpoint selection reduces broad side effects, but the
current adapter still performs causal attenuation rather than complete causal
erasure. The next method change must improve the counterfactual target signal or
apply the causal gate to adapter activations at inference; further scalar loss
or LoRA-scale tuning alone is unlikely to solve the remaining semantic failure.
