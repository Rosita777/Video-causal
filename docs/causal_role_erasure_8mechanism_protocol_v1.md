# Eight-mechanism causal-role erasure: single-seed master protocol v1

Status: **semantically frozen on 2026-08-22 for the main experiment**. The
method, mechanism set, data counts, training configuration, method arms,
evaluation fields, and aggregation rules in this document may not be tuned
after any treatment output is inspected. Exact manifests, repositories,
model inventories, prompts, media, checkpoints, and executable code must be
bound by SHA-256 registries before their corresponding stage begins.

Protocol ID: `causal_role_erasure_8m_single_seed_v1`.

This protocol supersedes the proposed multi-training-seed and replicated
evaluation-seed design for the new eight-mechanism experiment. It does not
rewrite or revive any exhausted water-impact development version. Historical
artifacts remain historical evidence only.

## 1. Research question and frozen claim

The experiment asks whether a text-to-video model can be edited so that an
entity and its downstream visual footprint are suppressed **when that entity
occupies a registered causal-source role**, while the receiver, scene, video
quality, and uses of the same concepts outside that role are preserved.

The proposed method is **Source-slot Randomized Counterfactual Distillation
(V4)**. The intended claim is deliberately narrower than “causal reasoning”:

> Across eight registered mechanism families, mechanism-specific V4 adapters
> improve continuous causal-source and footprint suppression over matched
> no-randomization controls and concept-erasure baselines, including on unseen
> sources and prompts that do not name the footprint, while retaining
> noncausal uses of the source and footprint concepts.

The primary result is a paired numeric comparison, not a requirement that
every output remove the source perfectly. A clear-to-partial visibility change
is a valid improvement. Complete absence and perfect strict successes are
secondary descriptive outcomes.

## 2. Eight equal-status mechanisms

All eight mechanisms are main-experiment mechanisms. Each receives the same
training-row count, evaluation-case count, main-table method set, review
procedure, and weight in the overall result. The separately labeled
Water/Fracture identification study adds controls only to diagnose the method
story; it does not alter the eight-mechanism main-table denominator.

| ID | Mechanism | Registered causal structure | Footprint to suppress |
| --- | --- | --- | --- |
| `water` | Water impact | A compact object enters a water receiver | Splash and propagating ripples |
| `collision` | Rigid collision | A moving object strikes a freestanding rigid receiver | Receiver displacement or toppling |
| `fracture` | Brittle fracture | An impactor strikes a brittle receiver | New cracks and fragments |
| `powder` | Powder impact | A compact object impacts a loose powder bed | Post-impact plume and crater |
| `elastic` | Elastic deformation | A moving object loads a deformable receiver | Indentation, stretch, and rebound |
| `field` | Field-mediated response | A charged object actuates a lightweight receiver without contact | Electrostatic attraction, lift, or deflection |
| `release` | Material release / particle dispersion | An impactor punctures or ruptures a particle-filled receiver | Newly released and dispersing particles |
| `trace` | Surface trace | An imprinting object contacts a susceptible surface | A new persistent print, groove, or indentation |

`field` is restricted to one coherent electrostatic trigger family; ordinary
airflow is not mixed into that adapter. `trace` does not reuse the failed
toy-car-track or ink-stain constructions. Exact sources, receivers, triggers,
compatibility constraints, and excluded near-duplicates are frozen in each
mechanism's ontology registry.

The main aggregate is a macro-average:

```text
eight_mechanism_macro = (metric_water + ... + metric_trace) / 8
```

No mechanism may be dropped, down-weighted, moved to an appendix, or given
fewer cases after results are seen. A mechanism that fails its pre-treatment
capability gate blocks this protocol version rather than changing the
denominator.

## 3. Model families and adapter unit

### 3.1 Wan main-method block

The proposed method uses the frozen Wan 2.1 T2V 1.3B base-model inventory.
One adapter targets one mechanism family. The experiment therefore trains:

- eight matched no-randomization control adapters; and
- eight V4 source-slot-randomized adapters.

The result is 16 Wan training runs. This is not one universal eight-mechanism
adapter. The supported conclusion is that one training principle transfers
across eight mechanism-specific operators.

Four additional identification runs are frozen on the representative
mechanisms `water` and `fracture`: one generic-paraphrase control and one
bystander-token control per mechanism. They use the same 200-step budget,
teacher, targets, caches, and one source-slot mention per erase prompt as V4;
prompt token lengths are matched as closely as the frozen templates permit.
Generic paraphrase changes wording without replacing the original
causal-source identity. Only bystander-token matches V4's 64-noun identity and
frequency distribution, placing those nouns in a registered noncausal
bystander slot while the true causal source remains drawn from the original
eight. These runs use a frozen identification subset
covering explicit, implicit-footprint, same-noun, same-footprint, role-swap,
and held-out-syntax cases. They are never dropped after results are seen. The
Wan training budget is therefore 20 runs: 16 main adapters plus four
identification controls.

The identification subset contains 24 already-registered formal cases from
each of `water` and `fracture`: 12 of the 24 causal cases and 12 of the 18
specificity cases. The causal half takes exactly one case from each
`direct-explicit`, `direct-implicit`, `natural-explicit`, and
`natural-implicit` cell inside every generalization group. The specificity
half contains four same-noun noncausal, four
same-footprint alternative-cause, and four near-causal/bystander cases, while
balancing the three source memberships at four each and wording at six direct
and six natural. Exact case IDs are committed before any method output. All use
their single frozen case seed. Matched-control and V4 outputs are reused from
the main evaluation; the two additional identification controls therefore add
`2 mechanisms x 24 cases x 2 streams = 96` videos. The four-arm
identification table contains 192 outputs in total, half already counted in
the main generation budget.

Within a mechanism, the matched control and V4 must have identical target
videos, caches, preservation data, initialization, update order, noise and
sigma draws, optimizer, and inference settings. Their only difference is the
registered source-slot intervention in the student factual prompt.

### 3.2 CogVideoX external-baseline block

External concept-erasure methods run on one frozen CogVideoX-2B inventory.
They are compared relative to a CogVideoX Original generated on the same
semantic case and seed. Wan and CogVideoX results are never described as a
same-backbone causal ablation.

Each repository revision, local patch, environment, model inventory,
concept phrase, and parameter set must be frozen before formal generation.
The paper labels must remain exact:

- `Negative Prompt`;
- `VideoEraser (official CogVideoX)`;
- `T2VUnlearning-adapted (ours)` because matching public training code and
  mechanism checkpoints are unavailable; and
- `SAFREE-CogVideoX`.

The existing Wan VideoEraser, T2VUnlearning, and SAFREE proxies are engineering
smokes and are excluded from the main table. A baseline implementation failure
is reported; it is not replaced after seeing another method's outputs.

## 4. The eight physical generation streams

Every frozen main-evaluation case has exactly these streams:

| Stream | Model family | Role |
| ---: | --- | --- |
| 1 | Wan | Original reference |
| 2 | Wan | Matched no-randomization control |
| 3 | Wan | V4 source-slot randomization |
| 4 | CogVideoX | Original reference |
| 5 | CogVideoX | Negative Prompt |
| 6 | CogVideoX | VideoEraser official |
| 7 | CogVideoX | T2VUnlearning-adapted (ours) |
| 8 | CogVideoX | SAFREE-CogVideoX |

Within each model family, all applicable streams use the same positive prompt,
case seed, resolution, frame count, FPS, and frozen base-model inventory unless
a published method has an unavoidable interface constraint. Any such
constraint is registered before generation and disclosed in the paper.

## 5. Training-data construction

### 5.1 Per-mechanism structured ontology

Each mechanism has an independent, physically compatible ontology. The
target counts are:

- 8 original training source identities;
- 56 additional training-only source identities;
- one 64-item source augmentation bank;
- 48 evaluation-holdout source identities that never enter training;
- 12 training receivers;
- 56 fresh evaluation receivers; and
- 8 frozen seen-receiver anchors selected from the training receivers.

Exact identity, canonical phrase, and curator-defined semantic-family overlap
between the augmentation bank and evaluation holdout are forbidden. A shared
generic head word is reported but is not by itself a fatal overlap unless the
curated identities belong to the same semantic family. This is the registered
separation rule; it may not be tightened or relaxed after media inspection.

Every source-receiver pair must pass a frozen physical-compatibility mask. A
pair rejected by that mask cannot be restored to fill a quota after generation.

### 5.2 Dynamic distributional counterfactual targets

For every erase row, the data builder stores structured fields and constructs
two prompts:

```text
factual_prompt = source + registered interaction + receiver [+ footprint]
target_prompt  = same receiver and natural scene dynamics,
                 but no source, no trigger/contact, and no target footprint
```

The target prompt independently generates a dynamic, source-free target
video. The target must contain the correct receiver and natural temporal
variation, while excluding the source, registered trigger, and downstream
footprint. It is not made by repeating a clean frame, masking a factual video,
or manually editing frames.

Factual and target videos are not required to be pixel-aligned. Accordingly,
the paper calls this a **distributional counterfactual target**, not an
individual counterfactual for one realized factual video.

Per mechanism, a frozen 192-row target-candidate graph is evaluated before
training. A deterministic, treatment-blind quality rule must select exactly
178 erase rows. Fewer than 178 eligible targets invalidates that mechanism's
data version; targets are not replaced after training begins.

Each accepted erase row freezes:

- the structured factual fields and canonical factual prompt;
- the source-free target prompt and target video;
- the target latent/base-cache tensors;
- the frozen-base teacher tensors under the target prompt; and
- path, dtype, shape, model, prompt, media, and tensor hashes.

### 5.3 Preservation data

Every adapter uses the same frozen bank of 36 generic preservation rows. These
rows contain ordinary non-target scenes and natural motion but do not train a
hard-negative or specificity branch. They exist only to constrain model drift.

Thus each mechanism binds:

```text
178 erase rows + 36 preserve rows = 214 unique training bindings
```

Specificity, role-swap, near-causal, and same-footprint examples remain
evaluation-only. Adding them to training would introduce a second treatment
and requires a new method version.

## 6. Frozen V4 intervention and objective

For erase row `i`, a deterministic mapping assigns source phrase `q_i` from
that mechanism's 64-item bank. The student prompt is rebuilt from structured
fields:

```text
c_f_aug = factual_prompt(
    assigned_source=q_i,
    receiver=receiver_i,
    prompt_variant=prompt_variant_i
)
```

Substring replacement is forbidden. All 64 source phrases occur once or twice
over the first 100 erase updates, the maximum active count difference is one,
and no row is assigned its original source identity.

For the cached target latent `z_cf`, sampled noise `epsilon`, and uniform
`sigma`:

```text
z_t     = (1 - sigma) * z_cf + sigma * epsilon
y       = epsilon - z_cf
student = LoRA(z_t, timestep, c_f_aug)
teacher = frozen_Wan(z_t, timestep, target_prompt)

L_erase = MSE(student, y)
          + 4 * MSE(student, stopgrad(teacher))
```

Preservation updates remain:

```text
L_preserve = 4 * MSE(
    LoRA(z_t, timestep, preserve_prompt),
    stopgrad(frozen_Wan(z_t, timestep, preserve_prompt))
)
```

The matched control uses the canonical original factual prompt in place of
`c_f_aug`; everything else is byte- or value-identical. A null-sidecar
preflight must reproduce the matched-control forward outputs, losses,
gradients, update order, and RNG digests before V4 training is authorized.

There is no factual latent, spatial mask, residual mask, negative-guidance
loss, token gate, mechanism-specific loss term, or hard-negative training
branch.

## 7. Frozen training and inference configuration

All 20 Wan runs use:

- base model: Wan 2.1 T2V 1.3B, one frozen content inventory;
- training seed: **26000**;
- LoRA rank/alpha: **16/16** on Q/K/V/Out modules;
- learning rate: **`5e-5`** with AdamW;
- 200 updates: exactly 100 erase and 100 preserve in the frozen alternating
  schedule;
- target-teacher weight: **4**;
- preservation weight: **4**;
- uniform sigma sampling with the same registered RNG trace;
- batch size and gradient accumulation: 1;
- only checkpoint 200 eligible for evaluation; and
- inference LoRA scale: **1.25**.

There is one training seed, not three. A failed run may be replayed from the
same registration only for a documented infrastructure failure that occurred
before a valid checkpoint was produced. Scientific failure never authorizes a
new seed, checkpoint choice, bank, prompt mapping, or hyperparameter search.

No per-mechanism method tuning is allowed. Any semantic method change creates
protocol v2 and cannot replace a v1 result.

## 8. Pre-treatment Original capability batch

Before any matched-control or V4 evaluation output is generated, run a Wan
Original capability batch:

```text
8 mechanisms x 8 source-receiver combinations x 3 qualification seeds
= 192 Original videos
```

The three qualification seeds are an exception used only to establish base
scene feasibility. They are not training-seed robustness evidence, do not
enter the main-effect statistics, and are disjoint from every formal
evaluation seed.

Within each mechanism, the 24 videos contain 12 direct and 12 natural prompts.
The frozen gate checks all 49 frames and requires:

- at least **15/24** videos fully eligible;
- at least 6/12 eligible direct prompts and 6/12 eligible natural prompts;
- eligible coverage of at least two source identities and two receivers;
- a fixed camera and clean frames 0--15;
- the source/trigger appearing only after the clean interval;
- visible contact or the registered non-contact trigger;
- the footprint beginning after the trigger;
- a recognizable receiver; and
- sufficient visual quality for semantic scoring.

All eight mechanisms must pass. The gate uses Original only and cannot inspect
any control, V4, or baseline output. Individual failed identities are not
deleted to manufacture a pass. A failed mechanism requires a new data/protocol
version and a fresh capability registration before any treatment generation.

## 9. Formal single-seed evaluation data

Every semantic case has one deterministic evaluation seed derived from its
frozen case ID and private salt. The same integer seed is used across all eight
physical streams wherever the interfaces permit. No generation replicate is
added to the formal table.

Capability combinations and qualification seeds are disjoint from formal
cases. Formal cases, prompts, source/receiver membership, subtype, seed,
method-independent eligibility rules, and anonymous IDs are frozen before any
formal method output is generated.

### 9.1 Causal set: 24 cases per mechanism

The 24 causal cases are balanced over three generalization groups:

1. held-out source + fresh receiver;
2. held-out source + seen receiver; and
3. seen source + fresh receiver.

Each group has eight cases. Within each group, the full crossing is:

```text
2 held-out wording styles (direct, natural)
x 2 footprint lexicalizations (explicit, implicit)
x 2 semantic cases
= 8 cases
```

Both wording templates are absent from training. In the implicit condition,
the prompt names the source and interaction but not the footprint. This is
required evidence against simple deletion of an explicitly named footprint.

### 9.2 Specificity set: 18 cases per mechanism

The 18 cases cross three source memberships with two held-out wording styles:

```text
3 source memberships (original-training, augmentation-bank, eval-holdout)
x 2 wording styles (direct, natural)
x 3 specificity subtypes
= 18 cases
```

Each membership-by-wording cell contains exactly one case of every subtype:

1. **same-noun noncausal:** the registered source noun is visible but does not
   cause the target event;
2. **role-swap / near-causal:** the noun is a receiver or bystander, or
   approaches without completing the registered trigger; and
3. **same-footprint alternative cause:** the protected noun remains
   noncausal while a visually related footprint is legitimately produced by
   another registered cause.

These cases test noun blacklist, event-template triggering, and footprint
blacklist explanations. Six causal--specificity matched pairs (`M6`), one per
membership-by-wording cell, are designated from the existing 24+18 cases and
require no extra videos.

### 9.3 Formal generation count

Per mechanism:

```text
24 causal + 18 specificity = 42 cases
42 cases x 8 streams = 336 videos
```

Across eight mechanisms:

```text
336 semantic cases
2,688 formal evaluation videos
96 additional generic-paraphrase/bystander identification videos
```

Including the 192-video capability batch, the registered generation budget is
2,976 videos before documented infrastructure-only replays. There is no
scientific cherry-picking or seed retry.

## 10. Blind full-video judgment

### 10.1 Anonymous packages

Every physical output receives a random review ID. Public review packages
remove method, backbone, checkpoint, adapter, file-path, and answer-key cues.
All mechanisms and streams are interleaved. Candidate outputs may be shown
with an anonymous same-backbone Original reference, but the method identity is
never disclosed.

Two nested private keys are committed before review. The Original-only
eligibility key identifies only the anonymous Wan and CogVideoX Original
references and reveals no control, baseline, or V4 mapping. It is opened after
Original atomic scores are frozen, solely to freeze `E_b(u)` and every shared-
capability subset. The full method answer key remains sealed until all
canonical atomic scores, including method outputs, are frozen.

The 96 additional generic-paraphrase and bystander-token outputs are placed in
the same anonymous review system and interleaved with their matched-control and
V4 identification cases. Reviewers are not told that an item belongs to the
identification study.

### 10.2 Full-49-frame VLM review

The old five-frame contact-sheet judge is not sufficient. Each VLM pass sees
all 49 indexed frames in five overlapping temporal panels:

```text
0--12, 9--21, 18--30, 27--39, 36--48
```

Two independent VLM passes use separately randomized orders and receive no
other pass's answer. They output only atomic scores, confidence, frame-indexed
visual evidence, and a schema-valid JSON record. They do not choose a winner.

For causal cases, each field uses a frozen 0/1/2 scale:

- `source_visibility`: absent / partial / clear (lower is better);
- `footprint_visibility`: absent / partial / clear (lower is better);
- `receiver_preservation`: bad / partial / good (higher is better); and
- `video_quality`: bad / partial / good (higher is better).

For specificity cases:

- `protected_object_visibility`: absent / partial / clear;
- `noncausal_role_adherence`: violated / partial / correct;
- `receiver_preservation`; and
- `video_quality`.

The mechanism rubric supplies the exact source, receiver, trigger, footprint,
and acceptable alternative causes. Prompt text is evidence about the requested
scene, never evidence that an event was visually present.

### 10.3 Human calibration and canonicalization

VLM scores are full-set prelabels, not unquestioned ground truth.

- Two humans independently label a treatment-blind, answer-key-blind,
  mechanism/stream/field-stratified 10% calibration sample.
- Two independent humans review all VLM--VLM atomic disagreements, every
  `partial`, every unusable output, and every confidence below 0.75.
- A separate random 10% of high-confidence VLM agreements is human-audited.
- Human--human atomic disagreements go to a third adjudicator.
- If an audited high-confidence agreement stratum has more than 5% atomic
  error, all remaining items in that mechanism-by-field stratum receive human
  review.

The frozen post-adjudication atomic table is the only input to paper metrics.
Report VLM--human macro-F1, weighted kappa, disagreement rate, audit expansion,
and human-review coverage. No prompt or rubric is edited after calibration;
an evaluator failure requires a versioned evaluation protocol.

## 11. Eligibility and continuous metrics

Original capability is a property of a model family, not an erasure success.
For causal case `u` and backbone block `b`, define Original eligibility before
opening the method answer key:

```text
E_b(u) = [Original source_visibility = 2
          and Original footprint_visibility >= 1
          and Original receiver_preservation >= 1
          and Original video_quality >= 1]
```

Ineligible Original cases remain in the fixed denominator and cannot earn
positive erasure credit. This prevents weak base generation from being counted
as successful deletion.

For method `m` in block `b`:

```text
usable_m(u) = [receiver_preservation_m >= 1
               and video_quality_m >= 1]

CES_m(u) = E_b(u) * usable_m(u)
           * ((2 - source_visibility_m)
              + (2 - footprint_visibility_m)) / 4
```

`CES` is the primary continuous Causal Erasure Score in `[0,1]`. It awards
partial credit to clear-to-partial improvements and zero credit to unusable or
base-incapable outputs.

For specificity:

```text
usable_spec_m(u) = [receiver_preservation_m >= 1
                    and video_quality_m >= 1]

SU_m(u) = usable_spec_m(u)
          * (protected_object_visibility_m
             + noncausal_role_adherence_m) / 4
```

`SU` is the continuous Specificity Utility guardrail in `[0,1]`. Receiver
preservation, video quality, usable fraction, and base-capability fraction are
also reported separately rather than hidden inside one aggregate.

## 12. Comparisons and aggregation

The primary causal contrast is same-backbone and single-factor:

```text
V4 CES - matched-control CES
```

For external methods, first calculate each method's per-case improvement over
its own same-backbone Original. The fixed-denominator table retains zero
contribution from Original-ineligible cases and reports capability coverage,
but it cannot support a claim that V4 beat that method. Headline external
contrasts use only the subset whose Wan and CogVideoX Originals were both
declared capability-eligible before any method answer key was opened. Compare
V4's Wan-normalized gain with each CogVideoX-normalized gain only on that
pre-frozen shared subset. These are explicitly labeled cross-backbone
normalized benchmark comparisons, not pure method ablations.

If a baseline-by-mechanism shared subset contains fewer than 12 of the 24
causal cases, that contrast is marked not estimable for that mechanism. It
remains in capability and fixed-denominator tables, but contributes neither a
win nor a loss to a method headline. The phrase “outperforms all baselines” is
prohibited if any registered headline contrast is not estimable.

Aggregation order is fixed:

```text
one output per semantic case
-> mean within each mechanism
-> equal-weight mean across eight mechanisms
```

Cases, not seeds, are the inferential units. Report mechanism-level means,
the eight-mechanism macro-average, paired case-cluster bootstrap 95% confidence
intervals, and all pre-registered V4-versus-baseline contrasts. Holm correction
is applied to the five headline causal contrasts: matched control, Negative
Prompt, VideoEraser, T2VUnlearning-adapted, and SAFREE.

The exact per-case quantities are:

```text
Delta_m(u) = CES_m(u) - CES_Original,same-backbone(u)
Contrast_b(u) = Delta_V4(u) - Delta_b(u)
```

For the matched Wan control, `Contrast_control(u)` is algebraically equal to
`CES_V4(u) - CES_control(u)`. For each of 10,000 bootstrap iterations, use
PCG64 seed `8202601`. The matched-control contrast independently resamples the
24 semantic causal cases with replacement inside every mechanism, computes
each mechanism mean contrast, then averages the eight mechanism means with
weight `1/8`. External baseline `b` instead resamples the pre-frozen
shared-capability subset of size `n_m,b` inside each estimable mechanism. An
external eight-mechanism macro is computed only when all eight mechanisms are
estimable; otherwise only mechanism-level estimates and the full-denominator
capability table are reported. Generation seeds are not resampled because
there is one fixed output per case. Percentile 2.5% and 97.5% quantiles form
the unadjusted reported interval.

For each of the five headline contrasts, define the one-sided paired-bootstrap
p value as `(1 + number of bootstrap contrasts <= 0) / 10001`. Apply Holm's
step-down procedure to these five p values at family-wise alpha `0.05`.

The phrase “outperforms all baselines” is used only if every registered
contrast has a favorable point estimate, an unadjusted percentile 95% lower
bound above zero, a Holm-adjusted one-sided p value below `0.05`, and is
estimable under the shared-capability rule above. Otherwise report the exact
estimates, intervals, adjusted p values, capability failures, and ranking
without changing the claim or endpoint.

Specificity, receiver preservation, quality, and usability accompany every
causal table. A causal-score win with a material specificity or preservation
loss must be described as a trade-off, not an unconditional success.

The word “retaining” is permitted only if the paired 95% lower confidence
bounds for V4 minus matched control exceed the frozen non-inferiority margins:

```text
Specificity Utility:       -0.10 on the [0,1] scale
Receiver preservation:     -0.10 after dividing the 0/1/2 score by 2
Video quality:              -0.10 after dividing the 0/1/2 score by 2
Usable fraction:            -0.05
```

Failure of a margin does not erase the causal-score table; it changes the
conclusion to an efficacy--preservation trade-off and prohibits an
unqualified retention claim.

Specificity Utility uses the 18 specificity cases per mechanism. Receiver,
quality, and usable-fraction non-inferiority uses the 24 causal cases per
mechanism; corresponding specificity preservation values are reported as a
separate secondary table. Every margin uses the same paired, mechanism-
stratified 10,000-iteration bootstrap and equal-weight eight-mechanism macro
procedure as the matched-control causal contrast.

Secondary descriptive outcomes include:

- clear-to-partial and clear-to-absent transitions;
- source-absent and footprint-absent rates;
- strict success
  (`source=0, footprint=0, receiver=2, quality=2`);
- capability-valid and usable fractions; and
- the six matched causal--specificity pair outcomes.

None of these secondary counts is an all-or-nothing promotion gate.

## 13. Novelty identification and claim boundaries

The paper's contribution is the combination of:

1. a role-conditioned causal-footprint erasure problem for text-to-video
   generation;
2. a structured source-slot intervention over fixed dynamic distributional
   counterfactual targets and a frozen counterfactual teacher; and
3. an eight-mechanism benchmark that tests unseen identity, implicit
   footprints, same-noun noncausal use, role swap, near-causal negatives, and
   same-footprint preservation.

Source-slot randomization alone is not presented as a fundamentally new
optimization primitive. The causal-role interpretation is supported only if
V4 improves implicit-footprint and held-out-source causal cases while retaining
the registered specificity cases. High explicit-prompt suppression alone is
compatible with ordinary lexical augmentation and does not establish the
intended claim.

The causal-role interpretation additionally requires V4 to outperform both
registered identification controls on the frozen Water/Fracture subset. A
gain over the matched no-randomization control without a gain over generic
paraphrase and bystander-token controls supports only a lexical-augmentation
interpretation and cannot be described as role-conditioned intervention
learning.

For each identification control, compute paired `CES_V4 - CES_control` on the
12 causal cases in Water and the 12 causal cases in Fracture, average within
mechanism, then weight the two mechanisms equally. Use 10,000 paired
case-bootstrap iterations with PCG64 seed `8202602`. Both contrasts must have
a positive point estimate, an unadjusted percentile 95% lower bound above
zero, and a Holm-adjusted one-sided bootstrap p value below `0.05` across the
two controls. On the six implicit-footprint causal cases per mechanism, each
V4-control equal-weight two-mechanism point estimate must also be positive.
On the 12 specificity cases per mechanism, V4 must be non-inferior to each
control in Specificity Utility with the frozen margin `-0.10` and the same
paired bootstrap structure. Failure of any condition prohibits the
role-conditioned novelty claim but does not hide the main efficacy tables.

The paper does **not** claim:

- perfect or universal source removal;
- recovery of an individual pixel-aligned counterfactual;
- discovery of a true causal representation inside the model;
- a single universal adapter spanning all eight mechanisms;
- that prompt augmentation, distillation, or concept erasure is itself new;
- same-backbone parity between Wan and CogVideoX methods; or
- that explicit removal of named source and footprint tokens proves causal
  propagation.

The correct description is “eight mechanism-specific operators trained by a
shared role-randomized counterfactual-distillation principle.”

## 14. Stage order and no-tuning rule

The executable order is:

1. freeze the protocol, ontology, compatibility, capability, model, runtime,
   code, and seed registries;
2. run and adjudicate the 192 Wan Original capability videos;
3. freeze all training targets/caches and the 24+18 formal cases per mechanism;
4. freeze and smoke-test all baseline implementations without inspecting
   formal semantic outcomes;
5. train the 8 matched controls, 8 V4 adapters, and 4 frozen identification
   controls at seed 26000;
6. generate all 2,688 main formal videos plus the 96 additional identification
   outputs, with one seed per case;
7. build the committed anonymous packages and complete VLM/human review;
8. freeze canonical scores, open the answer key, and compute the registered
   tables and confidence intervals.

No scientific outcome can select a new source bank, target, prompt wording,
seed, checkpoint, LoRA scale, baseline parameter, metric, reviewer prompt, or
mechanism subset. Infrastructure repairs must preserve semantic inputs and be
documented with before/after hashes. Any semantic change requires a new
protocol ID and must not overwrite v1 artifacts.
