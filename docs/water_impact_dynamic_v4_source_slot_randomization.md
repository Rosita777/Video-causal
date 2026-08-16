# Water-impact dynamic v4: source-slot randomized counterfactual distillation

Status: method design selected and independently reviewed on 2026-08-16. This
document is a design specification, not yet an executable preregistration. No
v4 training or generation is authorized until the exact source ontology,
prompt mapping, new-development manifests, gate registry, implementation
hashes, and frozen input hashes have been created and independently reviewed.

Eval12 and fresh-dev24 are exhausted. Sealed-final36 remains unopened and must
not be generated, decoded, inspected, or scored during v4 development.

## 1. Decision

The next single-factor experiment is **Source-slot Randomized Counterfactual
Distillation**. V4 keeps the complete v3b training objective and trajectory
fixed. Its only treatment is to replace the causal source-object phrase in
each erase-row factual prompt with a deterministically assigned phrase from a
frozen source bank.

This tests one falsifiable hypothesis:

> V3b learned strong suppression for repeated event language and a small set
> of training nouns, but did not learn that an arbitrary object occupying the
> causal-source role should be absent from the counterfactual video.

V4 succeeds only if it improves deletion for source nouns excluded from the
augmentation bank while preserving the same nouns in noncausal hard-negative
prompts. Improvement on bank nouns alone is insufficient evidence of
source-role generalization.

## 2. Evidence motivating the treatment

The frozen v3b/v3c results motivate a lexical-generalization hypothesis and
weaken the simpler teacher-dose hypothesis.

| Fresh-dev24 subset | V3b target suppression | V3c | Change |
|---|---:|---:|---:|
| Source seen, 8 cases | 6/16 | 8/16 | +2 |
| Source unseen, 16 cases | 5/32 | 4/32 | -1 |
| Direct, 12 cases | 2/24 | 4/24 | +2 |
| Natural, 12 cases | 9/24 | 8/24 | -1 |

V3c left target visibility unchanged in 18/24 cases. Its target changes were
four improvements, two regressions, and 18 ties; its footprint changes were
two improvements, two regressions, and 20 ties. Three salient unseen source
families—gray stone, strawberry, and pine cone—remained fully visible in all
eight evaluated cases under both methods. Walnut was unchanged in all three
cases.

The earlier eval12 comparison points in the same direction: v3b improved
target suppression in 3/4 seen-source cases but only 1/8 cases containing an
unseen source. In contrast, footprint suppression improved by exactly three
points in each of the three generalization groups. Training contains only
eight source identities, while the splash/ripple language is repeated across
all erase rows.

V3c's realized sigma weighting showed no useful dose-response pattern. It
also produced no joint `target=0, footprint=0` case; every absent target kept a
visible footprint, and every absent-target output had only quality 1. These
facts do not justify another sigma/window adjustment as the next registered
intervention.

The analysis is exploratory because source salience, prompt variant, and
generalization group are partly confounded. It is used to choose a new
falsifiable hypothesis, not as a statistical claim or a source-specific tuning
signal.

## 3. The only training intervention

For erase row `i` at erase ordinal `j`, let `q_j` be the source phrase assigned
by the frozen bank mapping. Rebuild the factual prompt from structured fields:

```text
c_f_aug = factual_prompt(q_j, receiver_i, prompt_variant_i)
```

Do not use substring replacement. The canonical prompt builder capitalizes
the direct template and must first reproduce every original factual prompt
byte for byte from `source_object`, `receiver`, and `prompt_variant`.

All v3b mathematics remain unchanged. With the cached source-free target
latent `z_cf`, the same sampled noise and sigma, the augmented factual prompt,
and the original source-free target prompt:

```text
z_t     = (1 - sigma) * z_cf + sigma * epsilon
y       = epsilon - z_cf
student = LoRA(z_t, timestep, c_f_aug)
teacher = frozen_base(z_t, timestep, target_generation_prompt)

L_erase = MSE(student, y)
          + 4 * MSE(student, stopgrad(teacher))
```

Preserve rows remain exactly:

```text
L_preserve = 4 * MSE(
    LoRA(z_t, timestep, preserve_prompt),
    stopgrad(frozen_base(z_t, timestep, preserve_prompt)),
)
```

The target latent, target prompt, teacher weight, uniform sigma distribution,
LoRA architecture, initialization, optimizer, learning rate, 100/100 role
schedule, 200-step budget, and inference scale are unchanged. There is no
anti-guidance term, spatial mask, factual latent, token gate, sigma schedule,
new target video, or hard-negative training branch.

This is not ordinary paraphrase augmentation. The intervention changes the
identity occupying the causal-source slot while the counterfactual world—the
same receiver with no source or impact footprint—stays fixed. The held-out
source and same-noun hard-negative gates are intended to distinguish
source-role generalization from lexical suppression; passing would support,
not prove, that interpretation.

## 4. Frozen source ontology, privacy, and assignment

Before implementation, an evaluator who is isolated from method development
must curate 80 new singular source phrases. Each phrase describes one compact,
visually recognizable, physically plausible object that can enter water. The
ontology should cover shape, color, material, texture, manufactured/natural,
and food/non-food variation without extreme size or implausible physics.

The private registry must reject overlap with historical water-impact sources,
the complete receiver ontology, and event/mechanism vocabulary such as water,
splash, ripple, impact, or collision. Enforce normalized phrase, normalized
head lemma, and a curator-reviewed source-versus-receiver semantic-equivalence
matrix; repository IDs alone are not a disjointness test. The eight original
training phrases are the only intentional historical-source exception.

Using a frozen private salt and SHA-256 ranking:

- 56 new phrases join the eight original training phrases to form the
  64-item augmentation bank visible to the implementer;
- 24 new phrases form a private lexical holdout and never enter training;
- before training, the evaluator publishes only the bank registry, a
  commitment hash for the ordered holdout registry, its count, and aggregate
  strata statistics;
- the implementer receives neither the full 80-item new-source registry nor
  the split salt and therefore cannot derive the holdout;
- the holdout candidate manifests, screening media, and labels stay in an
  evaluator-only location until the checkpoint, generation code, and
  generation protocol have been frozen;
- after registered evaluation is complete, an independent audit opens the
  private registry and recomputes the commitment, split, disjointness, and
  deterministic selection;
- no phrase moves between sets after any cache, model, or video is inspected.

Reconstruct the exact v3b balanced sample schedule without consuming the
noise/sigma RNG. Assign the source bank by erase ordinal using a frozen
permutation and deterministic collision swaps. The mapping must guarantee:

- all 64 phrases occur once or twice in the first 100 erase updates;
- the maximum active count difference is one;
- no row receives its original source phrase;
- direct/natural and receiver counts are reported;
- the first-100 active mapping and full 178-row mapping each have a canonical
  SHA-256 digest;
- the sample-order and noise/sigma RNG hashes remain exactly equal to v3b.

Bank size, ontology membership, assignment salt, and collision policy are not
tunable hyperparameters. A failed sanity check does not authorize another
bank on the same development set.

## 5. Prompt sidecar and implementation boundary

The frozen base cache and v3b target-prompt teacher cache remain read-only.
V4 adds one sidecar entry for each of the 178 erase rows. Each entry contains:

- manifest index and `scene_id`;
- original and assigned source IDs and phrases;
- canonical augmented factual prompt;
- bf16 prompt embedding with shape `(1, 226, 4096)`;
- tensor-content SHA-256.

The sidecar manifest binds the training manifest, model revision and content
inventory, tokenizer/text encoder identity, source-bank registry, active and
full mappings, prompt builder, dtype, shape, and ordered byte inventory. The
raw embedding payload is approximately 314 MiB. Encoding is a one-time T5
preparation step; training still uses two transformer forwards per update and
has essentially the same compute cost as v3b.

The preparer must also re-encode every unmodified factual prompt and compare
each tensor, shape, dtype, and byte hash with the corresponding frozen v3b
base-cache embedding. A null-sidecar integration test must substitute those
original embeddings and demonstrate equality of v3b/v4 forward outputs,
losses, LoRA gradients, sample order, and RNG digests. This rules out tokenizer
or text-encoder drift masquerading as the source-slot treatment.

V3b and v3c files are immutable because completed registrations bind their
hashes. V4 therefore uses dedicated files and paths, proposed as:

```text
scripts/build_water_impact_dynamic_v4_source_bank.py
scripts/prepare_water_impact_dynamic_v4_prompt_cache.py
scripts/train_wan_waterdrop_lora_v4.py
scripts/run_water_impact_dynamic_sft_v4_source_slot.sh
docs/water_impact_dynamic_v4_source_slot_randomization.md
outputs/water_impact_dynamic_v4/source_slot_prompt_cache_v1
outputs/water_impact_dynamic_v4/adapter_source_slot_randomized_v1
```

The dedicated trainer must reject cache rebuilding, cache-only modes, output
reuse, non-frozen paths, unexpected files, and any configuration differing
from the registered v3b control except the augmented factual-prompt sidecar.

## 6. Training invariants and fail-closed sanity

The v4 run must retain the following v3b values:

- Wan 2.1 T2V 1.3B model and full model-content inventory;
- 178 erase rows plus 36 preserve rows;
- frozen base-cache and target-teacher-cache byte inventories;
- seed 26000, 200 steps, exactly 100 erase and 100 preserve updates;
- learning rate `5e-5`, LoRA rank/alpha `16/16`;
- preservation weight 4 and target-teacher weight 4;
- identical initial LoRA, sample-order, and initial/final noise RNG hashes;
- checkpoint 200 as the only evaluation checkpoint;
- inference scale 1.25 and the frozen generation configuration.

Before any model loading, validate the exact bank, mappings, canonical prompt
reconstruction, 178 finite stored embeddings, shapes, dtypes, registered token
lengths `<=226`, no truncation, source-slot-only changes, and all frozen
content hashes. Model validation uses the same ordered path-plus-file-bytes
inventory algorithm frozen by v3c stage 2, including the actual transformer
weights; a revision marker alone is insufficient.

Then, in an isolated preflight process before optimizer creation or any
training backward, load the frozen text encoder and transformer to perform the
fresh re-encoding and null-sidecar forward/loss/gradient equivalence checks
from Section 5. Atomically freeze that preflight artifact and its hash in the
run registration. The formal trainer must validate it before starting and
must never perform preparation writes itself.

During the first 16 actual erase updates, retain the v3b teacher-scale check:

```text
g_i = 4 * sqrt(L_teacher_i / L_flow_i)
```

All observations must be finite with positive flow loss. The arithmetic mean
must lie in `[0.20, 0.50]`, and the individual maximum must be `<=1.0`. The
sanity artifact must be written atomically before the 16th erase backward pass;
no checkpoint may exist before it passes. Failure ends v4 without changing
the bank, loss weight, schedule, seed, or checkpoint.

At step 200, the trainer rechecks the exact 100/100 role counts, active source
mapping, sample-order digest, noise RNG digest, frozen caches, registration,
and model inventory before writing the eligible checkpoint.

## 7. Mechanistic audit that cannot select the model

Only after the causal and specificity blind scores, answer-key opening, and
machine gate are immutable may a training-latent audit compare v3b and v4
across bank and held-out source embeddings. Freeze the audited latent rows,
noun sampling weights, noise/timestep seeds, tensor reduction, and distance
normalization in advance. Report the across-noun prediction variance

```text
V = E_q || f(z_t, c_f(q)) - E_q'[f(z_t, c_f(q'))] ||^2
```

and distance to the frozen counterfactual teacher on identical cached latents,
noise, and timesteps. The expected signature is a reduction for held-out as
well as bank nouns. This diagnostic is explanatory only: it cannot choose a
checkpoint, change the bank, control whether generation proceeds, or alter the
registered visual decision.

## 8. New causal development set: `v4_dev72_v1`

No independent existing water-impact development row remains. Changing seeds
on eval12 or fresh-dev24 would not make those semantic cases fresh.

All group labels are relative to the v4 training inputs, not merely to an old
repository ID:

- `holdout_source_seen_receiver`: a private 24-item holdout source and a
  receiver present in v3b training;
- `seen_source_new_receiver`: one of the original eight training sources and
  a genuinely new receiver;
- `holdout_source_new_receiver`: a private holdout source and a genuinely new
  receiver.

The isolated evaluator creates a 48-case candidate pool with eight candidates
in each `group x prompt_variant` cell. New identities must be disjoint from
history by normalized phrase, head lemma, and curator-reviewed semantic
equivalence. Candidate manifests and media remain private.

Dataset freezing has two stages. At Stage 0, before any Original generation,
the evaluator publishes commitments to the private 48-case candidate
manifest, canonical templates, field normalization, screening seed,
generation configuration, selector salt, ranking formula, constrained-subset
algorithm, evaluation-seed salt, and seed-derivation formula. At Stage 1,
after screening, it publishes commitments to both raw reviews, the blank
dispute set, adjudication, eligibility table, selector output, selected
24-case manifest, and fully derived 72-unit U manifest. Private contents stay
hidden, but their bytes cannot change. Both stages finish before v4 training.

Qualification and selection proceed as follows:

1. Use one frozen screening seed that is never an evaluation seed.
2. Generate Original only and give two screening reviewers every full 49-frame
   video, not only a seven-frame composite. Screening reviewers cannot be the
   final treatment reviewers.
3. Independently score source visibility, footprint visibility, receiver,
   quality, and `causal_link`, where `2` means visible source-water contact
   clearly precedes the water response, `1` is ambiguous, and `0` is absent or
   temporally incompatible. A third screening reviewer adjudicates every
   atomic disagreement.
4. A candidate qualifies only with source `2`, footprint `>=1`, receiver
   `>=1`, quality `>=1`, and `causal_link=2`.
5. Enumerate subsets containing exactly four qualified cases per cell. A
   feasible subset must use 16 distinct holdout head lemmas across the two
   holdout groups, use each original training source exactly once in the
   seen-source group, use eight different historical receivers in the
   holdout-source/seen-receiver group, and use 16 distinct new receivers
   across the two new-receiver groups. New receivers are also disjoint from
   historical receivers, so all 24 selected receiver identities are unique.
6. Rank each candidate by a frozen salted SHA-256 of its canonical record.
   Choose the feasible subset whose ordered tuple of ranks is
   lexicographically smallest. There is no post-hoc reserve queue.
7. If no feasible subset exists, `v4_dev72_v1` is invalid before training. Do
   not replace prompts, sources, receivers, or seeds within this version.

The selected 24 cases contain eight per group and four direct plus four
natural prompts per group. Derive three evaluation seeds per case from
`pair_id x replicate x frozen_salt`, separate from screening and all prior
seeds. This yields the fixed unit set `U` of 72 paired evaluation units. Treat
the 24 semantic cases, not 72 videos, as the independent clusters in
uncertainty reporting.

Original, frozen v3b, and v4 use the same 72 prompts, seeds, model inventory,
25 steps, CFG 5, 49 frames, 480x832, 8 fps, and bf16. V3b and v4 both use LoRA
scale 1.25. Only checkpoint 200 is eligible.

Original screening conditions the experiment on cases the base model can
render as valid causal events. Results therefore apply to this qualified
base-model regime, not to arbitrary object prompts.

## 9. Same-noun noncausal specificity set

Source-slot randomization could degenerate into a lexical blacklist. A
separate hard-negative set and its selection rule must therefore also be
commitment-hashed before training.

Use three source memberships: `original_source` for the original eight
training nouns, `new_bank_source` for the 56 new nouns exposed during v4
training, and `holdout_source` for private nouns never exposed during
training. After causal selection, create one specificity candidate from every
selected causal case in the original and holdout memberships, keeping its
exact source phrase and receiver. Add 12 candidates using 12 distinct
new-bank nouns, six per prompt variant, paired by frozen hash with 12 distinct
receivers from the causal selection. All candidates use one of two frozen
noncausal templates. The direct and natural templates state that the object
stays on a dry support beside the receiver, never contacts the water, and does
not cause a splash or ripple. This yields a 36-case candidate pool.

Specificity also uses two-stage freezing. Before any specificity Original is
generated, Stage 0 publishes commitments to the private candidate manifest,
new-bank selection and receiver assignment, exact templates, normalization,
screening seed, generation configuration, selector salt and algorithm,
evaluation-seed salt, and seed-derivation formula. After screening, Stage 1
commits the two raw reviews, dispute set, adjudication, eligibility, selected
18-case subset, and fully derived 36-unit W manifest. Both stages finish
before v4 training.

Generate Original only with a screening seed distinct from every causal and
specificity evaluation seed. The independent screening panel inspects all 49
frames and scores protected-object visibility, receiver, quality, and
`noncausal_role_adherence`, where `2` means the object remains separate from
water and the water stays free of an impact response, `1` is ambiguous, and
`0` contains contact/entry or a clear induced response. Eligibility requires
protected object `2`, receiver `>=1`, quality `>=1`, and adherence `2`.

Select exactly three eligible cases from each
`original/new-bank/holdout x direct/natural` cell with the same salted global
SHA-ranking rule. The final 18 cases use 18 unique nouns. The six original and
six holdout noun-receiver pairs exactly match selected causal cases; the six
holdout cases cover both holdout causal groups in both variants. If no
feasible subset exists, the specificity data version is invalid before
training; there is no prompt replacement or reserve after inspection.

Derive two evaluation seeds per case from
`specificity_case_id x replicate x private_salt`. All 36 seeds are unique and
disjoint from U, both screening sets, training, and every historical
evaluation seed. This produces the fixed unit set `W` of 36 units. Original,
frozen v3b, and v4 use the same prompt and seed per unit, the same model
inventory, 25 steps, CFG 5, 49 frames, 480x832, 8 fps, and bf16; v3b and v4
use checkpoint 200 and LoRA scale 1.25. Validation requires exact file counts,
successful full-video decode, and no cross-arm path, inode, or content reuse.

Hard negatives are evaluation-only; adding them to the v4 preserve branch
would change a second training factor and is forbidden in this ablation.

## 10. Blind review

Build separate public/private packages for the causal and specificity sets.
Each unit contains independently copied full-length Reference Original,
Candidate A, and Candidate B videos plus a matched composite. Original has its
own blank scoring row and receives the same full-49-frame review as both
candidates.

Use deterministic hash-blocked A/B assignment, never one global mapping. In
the causal set, each `group x variant x replicate` block has two v4-in-A and
two v4-in-B units, while each semantic case places v4 in A for either one or
two of its three replicates. In specificity, each case swaps v4 between A and
B across its two seeds; every membership-by-variant cell is therefore
balanced.

Public packages contain only anonymous media, composites, and blank review
rows. They display the source or protected-object phrase and receiver
description needed for scoring, but expose no method, pair ID, group, variant,
replicate, seed, source path, checkpoint, answer key, or provenance hash. The
reviewer environment mounts only the public package; private keys, scorers,
the repository, prior scores, and sibling packages are not readable.

Two final reviewers independently inspect all 49 frames of every Reference,
A, and B video. In the causal package they score target visibility, footprint
visibility, receiver, and quality; Reference rows also score `causal_link`.
In specificity they score protected-object visibility, receiver, quality, and
`noncausal_role_adherence` for all three arms. Every atomic disagreement goes
to a third blinded reviewer. Exact agreement is canonical; otherwise use the
three-reviewer majority, with an exact `0/1/2` split resolved to median 1.

Freeze and hash the two raw reviews, blank dispute manifest, adjudication, and
canonical anonymous table before opening the answer key. Only after that
opening may the method-labeled tables and gate be generated and hash-bound.

The causal rubric remains target visibility, footprint visibility, receiver
preservation, and video quality on the frozen 0/1/2 scale. The specificity
rubric replaces target deletion with protected-object visibility, where 2 is
clear preservation and 0 is absence.

## 11. All-or-nothing causal gate

For method `m` and causal unit `u`, define:

```text
usable_m(u) = receiver_m >= 1 and quality_m >= 1
ST_m(u)     = usable_m ? 2 - target_visibility_m : 0
SF_m(u)     = usable_m ? 2 - footprint_visibility_m : 0
strict_m(u) = target_m=0, footprint_m=0, receiver_m=2, quality_m=2
```

The registered denominator is the integer `|U|=72`. Let:

```text
E = {u in U: Original target=2, footprint>=1, receiver>=1,
              quality>=1, causal_link=2}
C = {u in E: usable_v3b(u)}
K = {semantic case k: all three replicates of k are in C}
C_hold = C intersect {holdout_source_seen_receiver,
                      holdout_source_new_receiver}
Delta_T(S) = sum over u in S of (ST_v4(u) - ST_v3b(u))
Delta_F(S) = sum over u in S of (SF_v4(u) - SF_v3b(u))
G_T(k) = sum over the three units u of complete case k of
         (ST_v4(u) - ST_v3b(u))
```

All target/footprint deltas, paired improvements, absent counts, and strict
counts below are computed on `C` or the explicitly named intersection.
Unusable v4 output contributes zero suppression and cannot count as an
improvement, absence, or strict success. Receiver, quality, and v4 usability
totals are computed on all 72 units in `U`.

The causal gate passes only if every condition holds:

1. all training, generation, media, review, assignment, and scorer provenance
   validates;
2. `|E| >= 66` of 72 and `|C| >= 64` of 72;
3. `|K| >= 20` of 24, with at least three complete cases in every one of the
   six `group x variant` cells;
4. `Delta_T(C) >= +18`;
5. every replicate slice has `Delta_T(C_replicate) >= +3`;
6. every group has `Delta_T(C_group) >= +3`, and direct and natural variants
   each have `Delta_T(C_variant) >= +3`;
7. none of the six group-by-variant cells has negative target delta;
8. at least 18 units in C have usable paired target improvement;
9. at least 10 of 24 semantic cases in K have `G_T(k) > 0`;
10. `Delta_T(C_hold) >= +12`, with at least 12 usable paired target
    improvements in `C_hold`;
11. at least seven of the nominal 16 holdout semantic cases are in K and have
    positive three-replicate target gain;
12. there are at least six clear-to-absent (`v3b target=2 -> v4 target=0`)
    units overall;
13. at least four clear-to-absent units in `C_hold` come from four distinct
    holdout cases and cover both holdout groups and both variants;
14. at least two holdout cases in K achieve clear-to-absent on at least two of
    their three replicates;
15. usable absent-target count on C is at least v3b plus six;
16. v4 has at least 68 usable outputs in U;
17. v4 receiver points on U are at least
    `max(114, v3b_receiver_points_on_U - 6)`;
18. v4 quality points on U are at least
    `max(96, v3b_quality_points_on_U - 6)`;
19. `Delta_F(C) >= 0`, `Delta_F(C_hold) >= 0`, and every group's footprint
    delta is nonnegative; each of the six group-by-variant cells has footprint
    delta at least `-1`;
20. v4 has at least six strict successes on C, including at least four in
    `C_hold`; strict successes cover all three groups, both variants, and at
    least four semantic cases;
21. at least four strict successes are paired gains for which v3b was not
    strict on the same unit; at least three paired strict gains come from
    three distinct cases in `C_hold`.

Report a cluster bootstrap interval for the mean of `G_T(k)` over complete
cases `k in K` as a secondary uncertainty analysis. It is explicitly a
complete-case estimand and cannot override the deterministic gate.

## 12. All-or-nothing specificity and role-selectivity gate

The registered denominator is the integer `|W|=36`. Let:

```text
H = {u in W: Original protected_object=2, receiver>=1, quality>=1,
              noncausal_role_adherence=2}
D = {u in H: usable_v3b(u)}
K_D = {semantic case k: both replicates of k are in D}
PV_m(S) = sum over u in S of (usable_m ? protected_visibility_m : 0)
NR_m(S) = sum over u in S of (usable_m ? noncausal_role_adherence_m : 0)
A_m(S)  = count u in S with usable_m and protected_visibility_m=0
```

Form partitions by intersecting H or D with `original_source`,
`new_bank_source`, `holdout_source`, `direct`, `natural`, and the six
membership-by-variant cells. For any partition P, the absolute preservation
floor is `ceil(1.5 * |P|)`. Unusable v4 outputs receive zero PV and NR points.

The specificity gate passes only if every condition holds:

1. `|H| >= 33` of 36 and `|D| >= 32` of 36;
2. each source-membership stratum has at least 11 of 12 units in H and at
   least 10 of 12 in D; each of the six cells has at least five of six units
   in D;
3. `|K_D| >= 15` of 18 semantic cases;
4. v3b satisfies `PV_v3b(D) >= ceil(1.5*|D|)` and v4 satisfies
   `PV_v4(D) >= max(PV_v3b(D)-3, ceil(1.5*|D|))`;
5. the same v3b absolute floor holds independently for each D membership and
   variant partition. V4 remains above that floor and no more than two PV
   points below v3b in each such partition; in each cell it is no more than
   one point below v3b and remains above the absolute floor;
6. conditions 4 and 5 also hold after replacing PV with NR, so object
   preservation cannot hide contact or a newly induced water response;
7. on the entire Original-eligible H, v4 satisfies both
   `PV_v4(H) >= ceil(1.5*|H|)` and `NR_v4(H) >= ceil(1.5*|H|)`; the same
   absolute floors hold in every H membership, variant, and cell partition;
8. `A_v3b(D) <= 3` and `A_v4(D) <= min(A_v3b(D)+1, 3)`. Also `A_v4(H) <= 3`,
   and each membership stratum contains at most one usable absent protected
   object under either method on D and under v4 on H;
9. v4 has at least 33 usable outputs in W;
10. v4 receiver points on W are at least
    `max(57, v3b_receiver_points_on_W - 3)`;
11. v4 quality points on W are at least
    `max(48, v3b_quality_points_on_W - 3)`.

The private manifest additionally binds a one-to-one mapping M from the six
holdout specificity cases to their exact causal semantic cases, with identical
source phrase and receiver. A mapping pair is complete when its causal case is
in K and its specificity case is in `K_D`; at least five of six pairs must be
complete. For a complete pair p, define:

```text
role_selective(p) =
    G_T(causal_case(p)) > 0
    and all three v4 causal outputs are usable
    and PV_v4(two specificity units) >= PV_v3b(those units)
    and NR_v4(two specificity units) >= NR_v3b(those units)
    and both v4 specificity outputs are usable
    and neither v4 specificity output has protected_visibility=0
    and both v4 specificity outputs have noncausal_role_adherence=2
```

At least three of six mapped pairs must be role-selective, covering both
holdout causal groups and both prompt variants. At least two of those
role-selective causal cases must contain a clear-to-absent replicate. This is
the direct same-noun evidence: the same held-out source improves when it is
causal and remains present when it is noncausal.

V4 is promoted only if the causal, specificity, and matched role-selectivity
conditions all pass. Otherwise the experiment has not demonstrated selective
causal-source-role deletion over lexical suppression.

## 13. Decision boundary

Before an eligible checkpoint exists, distinguish three terminal outcomes:

- `preflight_dataset_invalid`: either Stage-0/Stage-1 selector cannot produce
  a valid frozen dataset; do not edit candidates within that data version;
- `registered_scale_sanity_termination`: the normal v4 run reaches the frozen
  16-erase scale check but fails it. This is neither a provenance error nor a
  visual negative result; it ends this v4 method without changing the bank,
  weight, or sanity bounds and without generation;
- `invalid_training_run`: after the formal trainer launches, any crash,
  nonfinite value, registration mismatch, RNG/order mismatch, cache/model
  drift, or other termination before the eligible step-200 checkpoint that is
  not the registered scale-sanity failure. It is not a visual negative result,
  but it ends this registered run; do not resume or restart it under the same
  method/data version.

After an eligible checkpoint exists, record exactly one of:

- `invalid_run`: causal condition 1 or any training, generation, media,
  assignment, review, scorer, or commitment provenance check fails;
- `inconclusive_invalid_evaluation`: causal validity conditions 2–3,
  specificity validity conditions 1–3, matched-pair completeness, or any v3b
  baseline clause in specificity conditions 4–6 and 8 fails;
- `valid_negative_ablation`: all provenance and evaluation-validity checks
  pass, but any remaining v4 causal efficacy, specificity, or matched
  role-selectivity clause fails.

None of these outcomes promotes v4 or opens sealed-final36. Once the formal
trainer launches, the method/data version is exhausted by any terminal
outcome: do not resume, rerun, replace cases, change bank membership, source
assignment, teacher weight, sigma, layer set, checkpoint, inference scale,
wording, or seeds. A static failure caught before formal training begins is a
repairable preflight failure only when registered scientific inputs remain
byte-identical.

If both gates pass, freeze the exact code, model, checkpoint, bank, mapping,
and hashes. Only then may a separate preregistered multi-seed paper main
experiment open sealed-final36.

## 14. Alternatives deliberately not combined with v4

The second-ranked hypothesis is an ESD-style paired conditional-contrast
extrapolation teacher. With guidance parameter `eta=1`:

```text
b_f  = frozen_base(z_t, factual_prompt)
b_cf = frozen_base(z_t, counterfactual_prompt)
a    = b_cf - eta * (b_f - b_cf) = 2 * b_cf - b_f
L    = L_flow + lambda_ag * MSE(student, stopgrad(a))
lambda_ag = 2
```

Under zero-LoRA initialization `student=b_f`, stop-gradient targets, and the
same mean-MSE reduction, `lambda_ag=2` makes this auxiliary output gradient
equal to the v3b weight-4 teacher gradient at initialization. The separate
factor 2 inside `a` comes from `eta=1`. This changes the auxiliary fixed point
while matching its initial scale; effectiveness remains an empirical
hypothesis. Unlike ESD and T2VUnlearning, `b_cf` is a conditional
counterfactual branch rather than an unconditional branch; the resemblance is
algebraic. It still trains on only eight source nouns, mixes
source/event/wording in one direction, adds a frozen factual forward, and does
not address the current unseen-noun evidence. Test it only in a later,
separate experiment if v4 supports lexical generalization yet still stops at
partial deletion.

Also excluded from v4 are spatial masks and dual trajectories, because the
current independently generated counterfactual targets are not pixel-aligned
and the cache has no factual latent; CFG-branch training, cross-attention/span
gates, hard-negative preserve training, and another backbone each introduce a
separate factor.

## 15. Relation to prior work

The design is informed by two primary concept-erasure results:

- [Erasing Concepts from Diffusion Models](https://arxiv.org/abs/2303.07345)
  introduced frozen-model negative guidance as an erasure teacher.
- [T2VUnlearning](https://arxiv.org/abs/2505.17550) combines negatively guided
  velocity prediction with prompt augmentation, localization, and preservation
  for text-to-video concept erasure. Its HunyuanVideo nudity ablation reports
  that its LLM-based contextual prompt refinement improves robustness to those
  refined prompts; that is not direct evidence for held-out nouns or causal
  roles.

V4 is not an implementation claim for either paper and does not reproduce the
T2VUnlearning augmentation algorithm. Those results motivate widening the
conditioning support. V4 instead tests structured source-identity substitution
under paired causal counterfactual supervision: the receiver and source-free
target stay fixed while the causal source identity is systematically
randomized.
