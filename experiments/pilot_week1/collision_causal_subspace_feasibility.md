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

## Activation-Gated LoRA Feasibility

Loss masking does not prevent a globally active LoRA adapter from changing
tokens outside the causal region. The new activation controller therefore
modifies each LoRA projection output using

`base_output + gate * (adapter_output - base_output)`.

Erase rows use their exported spatiotemporal causal gate. Preservation rows
leave the adapter globally active so frozen-teacher distillation can still
constrain its behavior on ordinary prompts. Teacher forward passes bypass the
activation hook. Video-token projections use the full spatiotemporal gate;
text K/V projections use a per-sample nonempty-gate switch because text tokens
do not have video-grid coordinates. This makes an empty gate disable all LoRA
paths instead of leaving a global text-side residual active.

A two-step Wan smoke test completed on one erase row and one preservation row.
The controller found 240 PEFT projection modules, backward propagation completed
without numerical errors, and checkpoint 2 was saved at
`outputs/adapters/collision_activation_gate_smoke/checkpoint-000002`. A unit test
also verifies that a zero gate blocks adapter gradients while preserving the
frozen base path.

This validates the training mechanism only. The present implementation loads
precomputed gates for training scenes; it cannot yet gate a novel prompt during
generation. The next experiment should first compare a short activation-gated
run with the loss-gated checkpoint on fixed target and control prompts. If that
improves the semantic trade-off, the remaining method requirement is an online
gate predictor for inference.

### Checkpoint-25 Inference Probe

A 25-step activation-gated adapter was trained with the same balanced 68-row
manifest and loss weights as the earlier pilot. For an inexpensive inference
upper-bound probe, approximate gates were built from the first-pass base videos:
persistent red regions from the opening frames are removed, newly appearing red
regions seed the gate, and temporally reachable local motion expands the causal
region. The two target gates cover 10.6% and 13.1% of the video-token grid.

The first implementation left text K/V LoRA projections active when the video
gate was empty. This caused a false preservation change and was corrected by
using a per-sample nonempty-gate switch for text tokens. After the fix, a fresh
base control and an empty-gate adapter control have identical MP4 checksums and
zero frame MAE. This verifies exact no-op behavior when no target region is
detected.

The semantic target result is still negative. On the first target, the red ball
remains faintly visible and roughly one cup still falls, although the original
three-cup collision chain is shortened. On the second target, multiple red dots
remain and the receiver still collapses with visible melting artifacts. Neither
sample removes both the object and its causal footprint. Mean motion suppression
over the two samples is 35.41%, but this again reflects attenuation rather than
successful semantic erasure.

Activation gating therefore solves the spatial preservation problem when its
gate is empty, but it does not supply the missing counterfactual content inside
the gate. Expanding this checkpoint to the full 7+8 evaluation is not justified.
The next method experiment should improve the counterfactual target itself,
for example by distilling a clean static reconstruction inside the gate, before
investing in an online gate predictor.

### Counterfactual-SFT Ablation

Three factual/counterfactual training pairs were visually audited before the
next run. Their static targets are clean and well aligned: the red ball and
collision outcome are absent while the original receivers remain upright.
This rules out an obvious target-video construction failure in the sampled
pairs.

The `counterfactual_sft` objective removes both paired separation and factual
redirection. Erase rows optimize counterfactual flow matching only inside the
causal gate and frozen-teacher matching outside it. Preservation rows retain
full-video teacher distillation. A 50-step balanced run saved checkpoints 25
and 50; erase updates report zero pair and redirect losses as intended.

| checkpoint | target motion suppression | target early MAE | semantic success |
| ---: | ---: | ---: | ---: |
| 25 | 12.90% | 0.0137 | 0/2 |
| 50 | -32.76% | 0.0575 | 0/2 |

At checkpoint 25, both red balls remain and the receiver still falls. At
checkpoint 50, the balls still remain while deformation and extra motion become
stronger, especially in the blue-cup scene. Removing the dominant redirect loss
therefore does not recover semantic erasure, and simply training this objective
longer is counterproductive at the current learning rate.

The next diagnostic should deliberately overfit a very small set of aligned
pairs and evaluate those same training prompts. Failure there would show that
the gated LoRA and flow objective cannot express the intervention; success there
would instead isolate the problem to data coverage or generalization. This test
should precede further full-manifest tuning.

### Four-Scene Overfit Diagnostic

Four visually aligned collision pairs were isolated in
`data/collision_overfit4.csv`. The run uses erase rows only, no preservation
rows, and repeats these four pairs for 200 steps with the activation-gated
`counterfactual_sft` objective. Checkpoints were saved at steps 50, 100, 150,
and 200.

The loss does not show stable memorization. Its final 20-step mean is 0.1119,
with recurring spikes despite each sample being revisited many times. More
importantly, the first training prompt was regenerated with its original seed
and training gate at checkpoints 50 and 200. Checkpoint 50 only fades the red
ball and still allows the boxes to collapse. Checkpoint 200 retains the ball at
multiple times, still collapses the boxes, and degrades the receiver layout.
Neither output approaches the clean static training target.

This failure occurs on a seen prompt, seen seed, and seen gate. It therefore
rules out insufficient dataset diversity and held-out generalization as the
primary explanation. Under the current formulation, activation-gated rank-16
LoRA plus counterfactual flow matching cannot reliably express the requested
intervention. Further data expansion, longer training, or an online gate
predictor should be paused. The next method change must alter how the
counterfactual signal enters the denoising process rather than retune this loss.

### All-Ones Gate Diagnosis

To separate gate coverage from adapter expressivity, the same four-scene
overfit run was repeated with all activation gates set to one. The training
loss curve and its spikes were nearly unchanged. At checkpoint 50, the global
adapter changed the scene more strongly and kept the boxes somewhat more
stable, but the red ball still appeared. At checkpoint 100, the ball still
appeared and the box layout degraded further.

The all-ones control therefore does not recover object erasure. Sparse gates
are useful for limiting side effects, but insufficient gate coverage is not the
main failure. The current adapter learns a broad visual perturbation or motion
attenuation; it does not block the prompt-conditioned object formation. The
next method must inject an explicit target-conditioned suppression signal into
the generation path, rather than only changing the spatial support of a normal
LoRA residual.

### Target-Token Attention Suppression Diagnosis

A training-free diagnostic directly reduced cross-attention from selected video
queries to the exact `red rubber ball` and `ball` prompt tokens. The first run
used the oracle causal gate (10.6% of the latent video grid) with attention-logit
strength 20. The ball remained as red fragments, the receiver still fell, and
the cup geometry changed. This is not semantic erasure.

A deliberately extreme control used an all-ones spatiotemporal gate and strength
100, effectively blocking the selected target tokens for every video query in
all 30 Wan cross-attention blocks. Red ball fragments still appeared, while the
three cups changed shape and layout substantially. Thus the failure cannot be
explained by an inaccurate or too-small causal gate.

| attention intervention | gate | object removed | footprint removed | receivers preserved |
| --- | --- | ---: | ---: | ---: |
| target-token suppression, strength 20 | oracle causal region | no | no | no |
| target-token suppression, strength 100 | all video tokens | no | no | no |

This ablation shows that target information is distributed through the full
prompt representation and the iterative denoising state; zeroing a few token
links at inference produces artifacts instead of the desired counterfactual.
The implementation is retained as a reproducible diagnostic baseline, but
further strength tuning is not justified.

The next training formulation should use a target-conditioned intervention
branch rather than a fixed negative attention bias. It will receive the target
phrase, learn a residual from factual-prompt denoising toward the aligned clean
counterfactual target, and retain frozen-teacher preservation outside the causal
gate. A zero target condition must disable the branch exactly. The first test
remains the four seen training pairs: it must remove the object and keep the
receivers upright before any larger training or generalization experiment.

### Target-Conditioned LoRA Overfit

The activation controller was extended with a target-token gate. Cross-attention
K/V LoRA residuals are active only on exact target-phrase tokens, while all
video-side LoRA residuals require both a nonempty target condition and the
spatiotemporal causal gate. An empty target condition makes every adapter path
an exact base-model no-op. During classifier-free guidance, the controller also
detects whether the selected target-token positions contain actual embeddings,
so the adapter is disabled for the empty unconditional prompt.

The first four-scene overfit retained counterfactual flow matching and trained
for 100 steps. A seen training prompt, seed, and gate were evaluated at three
checkpoints.

| checkpoint | target object | collision footprint | receiver preservation |
| ---: | --- | --- | --- |
| 25 | red ball remains | boxes still fall | fail |
| 50 | ball is smaller and disappears earlier | boxes still fall | fail |
| 100, before CFG fix | ball is faint but still visible | boxes still fall | fail |
| 100, correct conditional CFG | ball is absent early but reappears near frame 32 | collision is delayed; boxes still fall | fail |

The target-conditioned architecture improves specificity and progressively
attenuates the target appearance, but counterfactual flow training still does
not stop the downstream collision. Its final 20-step mean loss is 0.0681 and it
does not memorize the aligned target even on a seen example. The likely mismatch
is trajectory-level: training denoises states around the clean counterfactual,
whereas factual prompting drives inference toward the collision trajectory.
The next ablation therefore adds explicit factual-to-counterfactual endpoint
redirection while retaining the same target and spatial gates.

### Target-Conditioned Trajectory Redirection

A separate `target_conditioned_redirect` objective adds a factual-trajectory
forward pass. Its predicted endpoint is pushed toward the aligned clean
counterfactual inside the causal gate, with weight 4. This is kept separate from
`target_conditioned_sft` so the flow-only ablation remains reproducible.

The redirect loss on the audited seen scene decreases from about 0.91 at the
start to 0.28 at step 50, and the final 20-step total-loss mean is 3.2605.
However, manual video review remains negative. At checkpoint 25 the ball is
smaller but the boxes still fall. At checkpoint 50 the ball is absent in early
frames, reappears around frame 32, and the boxes fall later in the clip.

For a fair control, the flow-only checkpoint 100 was regenerated after fixing
conditional-versus-unconditional CFG gating. Its video is nearly identical to
redirect checkpoint 50: the ball and collision are delayed rather than erased.
Therefore the visible improvement comes mainly from correctly disabling the
adapter on the unconditional prompt; endpoint redirection has no clear semantic
benefit in this probe.

The remaining failure is a temporal-relocation shortcut. Evaluation must treat
delaying the target event as failure, and the next loss must penalize target and
footprint presence over every post-intervention frame. More training steps or a
larger redirect weight are not justified until that temporal constraint is
implemented.

### Persistent-Time Gate Ablation

To close the temporal-relocation loophole, a persistent gate was added. After
the first causal frame, it keeps the spatial union of the entire factual causal
chain active through the end of the clip. The same expansion is applied to the
training loss gate and the inference LoRA gate. The original gate covers 3.6%
of the latent grid on average; the persistent version covers 13.8% while still
leaving the first two static frames untouched.

The persistent redirect run has lower training loss than the dynamic-gate run:
its final 20-step mean is 1.66, and redirect loss reaches 0.098 on the audited
scene. The videos do not support a success claim:

| checkpoint | target object | receiver/background preservation |
| ---: | --- | --- |
| 25 | red ball reappears in the middle and late frames | boxes still change and collapse |
| 50 | only a tiny red remnant remains | the initial box arrangement is already corrupted |

The persistent gate therefore trades temporal suppression for broad visual
damage. It does not solve object-plus-footprint erasure and should remain an
ablation, not the main method. The next method change must supervise target
presence and receiver identity directly, while using the causal gate only to
localize that supervision. Gate expansion and longer training are now ruled out.

### Component-Balanced Supervision

The next prototype separates the counterfactual residual into two independently
normalized terms. An `object_gate` covers the target object trajectory, while a
`receiver_gate` covers the receiver changes after excluding the target. The
factual denoising trajectory is redirected to the clean counterfactual endpoint
inside each gate:

`L = L_flow + 4 L_background + 4 L_object + 2 L_receiver`.

This addresses a concrete failure of the previous loss: the small red ball was
averaged together with the much larger moving receiver, so a low aggregate loss
could coexist with visible target remnants. Component normalization prevents
receiver area from hiding target failure. For this collision feasibility test,
the red target gate is extracted by a simple color rule and the receiver gate by
factual-versus-counterfactual video difference. The color rule is explicitly a
prototype annotation tool, not a proposed general method; a general experiment
must replace its output with category-agnostic object masks without changing the
loss.

The first component run confirms the intended separation but also exposes an
activation mismatch. At `ck25`, the red ball is absent from roughly frame 16
onward, while the boxes still fall. At `ck50`, the ball is mostly suppressed,
but the initial box arrangement is already distorted and the later boxes still
collapse. The component losses alone are therefore insufficient: the adapter
was still activated by the older causal gate, whose coverage was concentrated
on the target. The follow-up uses the union of `object_gate` and
`receiver_gate` as the activation gate, keeping the loss weights unchanged.

The union-gate follow-up validates the mismatch diagnosis:

| checkpoint | target object | causal footprint | receiver preservation |
| ---: | --- | --- | --- |
| 25 | large ball removed; one tiny late red remnant | boxes remain upright through frame 48 | good on the audited seen scene |
| 50 | removed | suppressed | catastrophic washout; boxes become faint and nearly disappear |

`ck25` is the first audited checkpoint that approximately satisfies object
removal and footprint removal together without changing the receiver layout.
It is not yet a final success because of the small late target remnant and the
single seen-scene scope. `ck50` demonstrates overtraining rather than further
improvement. Subsequent experiments should use early stopping and strengthen
receiver identity preservation; simply increasing steps is ruled out.

### Four-Receiver Seen-Scene Audit

The best `union ck25` checkpoint was then applied to all four training scenes,
using the original seed for each receiver. The result is mixed rather than a
general success:

| receiver | target removal | receiver / footprint | judgment |
| --- | --- | --- | --- |
| cardboard boxes | mostly removed; tiny late red remnant | boxes stay upright | good |
| cork blocks | mostly removed; tiny late red specks | blocks stay upright | good |
| white dominoes | visible red remnants at middle frames | dominoes stay upright | partial |
| stone-like blocks | target and motion suppressed | global washout / low contrast | failure |

This is the first useful generalization diagnosis. The loss and activation
design can work across receivers, but the automatically built component gates
are not equally reliable: small or low-contrast target masks produce either
residual target pixels or broad visual washout. The next data step is therefore
gate-quality auditing and rejection before training, not more adapter capacity.

### Unseen-Receiver Inference Audit

The same `union ck25` Adapter was applied to seven unseen receivers. First, an
all-ones spatial gate removed much of the ball and collision motion, but every
receiver became visibly washed out. This is not acceptable preservation.

We then built an inference-only gate from the original generated video: a
strict red-object detector plus local frame-difference motion, without using a
counterfactual video. Gate coverage was reduced to 4.5%--9.3%. The result was
better but still mixed:

| unseen receiver group | result with automatic local gate |
| --- | --- |
| blue cups | ball reduced, receiver geometry changes |
| yellow cups | repeated red remnants and blur |
| silver tins | red ball persists and tins move |
| blue cans | mostly preserved, small target remnants |
| green pegs | receiver preserved, early red-ball remnant |
| yellow pins | receiver mostly preserved, late red remnant |
| white pawns | mostly preserved, small late remnant |

This separates two effects: the Adapter can transfer the deletion behavior, but
automatic gate quality controls whether that behavior is local or destructive.
The next method step is therefore not a larger Adapter. It is a conservative
gate-confidence rule: reject or weaken gates with broad color detections or
unstable temporal tracks, and measure the tradeoff explicitly.
