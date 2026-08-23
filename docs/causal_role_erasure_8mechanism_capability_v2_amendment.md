# Eight-mechanism capability qualification: protocol v2 amendment

Status: **post-hoc after the bound v1 capability failure, but before any
treatment generation or training authorization**. Date: 2026-08-23.

Protocol ID: `causal_role_erasure_8mechanism_capability_v2`.

This document creates a new capability-qualification version. It does not
modify, reinterpret, or rescue
[`causal_role_erasure_8mechanism_capability_v1`](causal_role_erasure_8mechanism_protocol_v1.md).
All v1 prompts, scores, thresholds, artifacts, and its final fail status remain
bound as v1 evidence.

## 1. Bound v1 result and reason for amendment

The aggregate-only v1 adjudication failed its frozen all-eight-mechanism gate.
In particular, the electrostatic `field_mediated_response` block had **0/24
eligible rows**. Because v1 required every mechanism to pass, the v1 protocol
is permanently failed and cannot authorize treatment generation.

The amendment is motivated by a mismatch between the intended scientific
target and the v1 qualification operator. The intended target accepts a
genuine **partial causal effect**: a clearly generated source together with
partial but usable trigger or footprint evidence is sufficient evidence that
the Original can instantiate the mechanism. Perfect source timing, perfect
scene cleanliness, complete multi-part effects, and artifact-free rendering
are not the primary target. This is consistent with the registered continuous
CES endpoint, under which clear-to-partial change is meaningful and strict
success is secondary.

V1 instead made a row eligible only when every ordinal atomic field equaled
`2`. It therefore conjoined mechanism capability with exact frames 0--15
choreography, perfect camera locking, complete effects, and perfect quality.
The electrostatic subtype added an invisible charged state, tiny lightweight
receivers, and repeated no-contact constraints. Its 0/24 result is valid for
that frozen v1 operationalization, but it does not isolate a general absence
of field-mediated capability.

## 2. Scope and unchanged scientific safeguards

V2 remains an Original-only, pre-method capability qualification. It does not
compare methods and does not select formal evaluation cases. A pass does not
by itself authorize training or treatment generation; those actions require a
separate explicit authorization after the canonical v2 review and aggregate
gate are complete.

The following safeguards remain mandatory:

- all eight mechanisms have equal status and weight `1/8`;
- every mechanism has 24 formal Original rows;
- direct and natural wording each contribute 12 rows per mechanism;
- all 49 frames are reviewed independently under the frozen atomic rubric;
- two independent reviews and blinded adjudication produce one canonical
  atomic table before aggregate scoring;
- every registered row remains in the fixed denominator; and
- all eight mechanisms must pass the same row and mechanism rules.

## 3. Fresh all-eight formal batch

The v2 formal batch is fresh for **all eight mechanisms**, not only for the
failed v1 mechanism. No v1 qualification output, prompt-instance, seed, row,
or judgment may be reused as a v2 formal observation.

Each mechanism contains two fresh source identities and two fresh receiver
identities. Their physical `2 x 2` crossing is instantiated once in direct
wording and once in natural wording, giving eight semantic cases. Three fixed
repetitions per case give 24 rows per mechanism and 192 rows overall.

Formal seeds use the fresh namespace

```text
940000 + 1000*mechanism_index + 10*combination_index + repetition_index
```

with base seed `940000`. The canonical manifest is the sole authority for
formal case IDs, generation IDs, prompts, and seeds.

## 4. Uniform two-sentence prompt contract

The same prompt contract applies to every mechanism:

1. sentence one states one simple receiver initial state; and
2. sentence two states one source action and one visible result.

Every prompt has exactly two sentences and at most 55 English words. Exact
two-second timing, a mandatory offscreen-source clause, human actors, and
negative-prompt lists are excluded. Frames 0--15 remain available for public
diagnosis, but exact clean-prefix compliance is not a v2 eligibility input.

This simplification is uniform. Prompt length, timing, negative clauses, or
eligibility thresholds may not be tuned separately by mechanism.

## 5. Pre-freeze development selection

Development smoke was completed before the formal v2 manifest and review
freeze became authoritative. These smoke results selected ontology; they are
not formal qualification evidence and do not change the bound v1 result.

The magnetic non-contact development prompts produced **0/4** usable smoke
examples. The magnetic candidate is therefore not the final v2 subtype and no
v2 claim may be made about magnetic or electrostatic generalization. A separate
airflow development smoke produced **2/2** usable examples. Before formal
freeze, `field_mediated_response` was consequently fixed to the subtype
`airflow_non_contact_motion`.

The final field ontology uses desk-fan sources crossed with visually clear
pinwheel and ribbon receivers. The registered trigger is fan-driven airflow
across a visible gap without source--receiver contact. The registered footprint
is pinwheel rotation or ribbon flutter while that gap remains. The supported
v2 claim is limited to airflow-mediated non-contact response.

The elastic development smoke also replaced the blue foam receiver candidate
with a small square blue trampoline. This change occurred before formal freeze;
the final elastic receiver set uses the retained black trampoline and the blue
square trampoline.

These are the final development selections for v2. Once the regenerated formal
manifest and review artifacts are frozen, failure does not authorize another
field subtype, elastic receiver, prompt, row, or threshold substitution.

## 6. Uniform v2 row gate

All eight atomic fields must be completed, but `clean_prefix` is diagnostic
only. A row is eligible exactly when:

```text
decodable == 1
source_after16 == 2
trigger_visible >= 1
footprint_after_trigger >= 1
trigger_visible + footprint_after_trigger >= 3
receiver_recognizable >= 1
fixed_camera >= 1
quality >= 1
```

The coupled trigger-plus-footprint requirement excludes rows in which both
pieces of causal evidence are merely ambiguous while accepting a genuine
partial effect. A footprint clearly present before the trigger remains a
failure. `clean_prefix` is still scored and its distribution is reported, but
its value cannot add or remove eligibility.

This gate aligns the shared fields with formal Original eligibility:
`source_visibility = 2`, `footprint_visibility >= 1`,
`receiver_preservation >= 1`, and `video_quality >= 1`. Formal `E_b`, formal
CES, and formal-case eligibility remain unchanged and are recomputed from the
same-backbone Original for each formal case; capability rows never substitute
for formal-case Originals.

## 7. Uniform mechanism gate

Every mechanism must satisfy all of the following:

- at least **12/24** eligible rows;
- at least **6/12** eligible direct rows;
- at least **6/12** eligible natural rows;
- at least two distinct eligible source IDs; and
- at least two distinct eligible receiver IDs.

All eight mechanisms must pass. No mechanism may be dropped, down-weighted,
moved out of the main denominator, or assigned a mechanism-specific threshold.

## 8. Development smoke separation

The prompt sets under
[`prompts/capability_v2_smoke16/`](../prompts/capability_v2_smoke16/) and
[`prompts/capability_v2_smoke_patch4/`](../prompts/capability_v2_smoke_patch4/)
contain the magnetic and elastic development probes. The final airflow probe is
recorded under
[`prompts/capability_v2_smoke_airflow2/`](../prompts/capability_v2_smoke_airflow2/).
The aggregate development outcomes were magnetic `0/4` and airflow `2/2`;
these outcomes informed the pre-treatment ontology selection but are not
qualification evidence.

The hash-bound
[development-smoke decision record](../data/causal_role_erasure_8mechanism_capability_v2_development_smoke_decision.json)
records these counts, the elastic repair, and the disjoint development seed
namespaces. Formal seeds use base `940000`; smoke used the separate `950000`,
`964000`, `965000`, and `975000` namespaces.

No smoke prompt-instance, seed, generation ID, media output, review judgment,
or pass/fail result may enter, replace, or supplement a formal row. Formal
outputs must be freshly generated from the frozen canonical manifest after
the review artifacts are bound. Semantic object names may recur in the frozen
ontology; the evidentiary rows, seeds, output bindings, and judgments remain
disjoint.

## 9. Anti-cherry-picking and authorization boundary

After the v2 artifacts are frozen:

- no row, identity, physical pair, wording style, or mechanism may be deleted;
- no failed row may be replaced;
- no seed may be added, retried for scientific failure, or selected by output;
- no prompt or threshold may be tuned by mechanism;
- no development-smoke output may be promoted into the formal batch;
- no mechanism subtype may be swapped after formal output inspection; and
- no treatment output may be inspected before Original capability scores and
  the aggregate gate are canonicalized.

Infrastructure-only replay requires a documented failure before a valid
output exists and must reproduce the same registered binding and seed.

Training, formal evaluation selection, and treatment generation remain
**unauthorized**. The hash-bound Original capability batch may be launched only
through the frozen v2 runner after separate operator authorization; this
amendment does not itself start generation or authorize any treatment.

## 10. Bound v2 artifacts

The earlier magnetic candidate artifacts are superseded development artifacts.
Their paths and hashes are not final protocol bindings and must not be used for
formal generation or review. The airflow/elastic revision regenerated all
eight mechanisms together at the following canonical paths:

- [CSV manifest](../data/causal_role_erasure_8mechanism_capability_v2_manifest.csv)
- [canonical manifest](../data/causal_role_erasure_8mechanism_capability_v2_manifest.canonical.json)
- [formal prompt file](../prompts/causal_role_erasure_8mechanism_capability_v2.prompts)
- [manifest summary and hashes](../data/causal_role_erasure_8mechanism_capability_v2_summary.json)
- [development-smoke decision record](../data/causal_role_erasure_8mechanism_capability_v2_development_smoke_decision.json)
- [blank review template](../data/causal_role_erasure_8mechanism_capability_v2_review_template.csv)
- [review rubric and gate](../data/causal_role_erasure_8mechanism_capability_v2_review_rubric.json)
- [blank review freeze](../data/causal_role_erasure_8mechanism_capability_v2_review_freeze.json)

Final frozen SHA-256 commitments:

- CSV manifest: `e92d68e23687ec72642e3620b4254e0e1e22e0d1483f84718e6dd4b1dd4ea5c3`;
- canonical manifest: `8e8cc14a6d327193cb7f270d2386490dc00ee783d9e7d0f76a697837f218dbee`;
- formal prompt file: `472eb889d200d9d26c5863f545281468632c0185b8f148f830dfb1589467e1d4`;
- summary: `f52dcc9dbef02e62fd4e3c82d0e9726c1d49fd19e47bec61a5a55a7676a0894c`;
- development-smoke decision: `60a3377c74ea01c777b865efd3f8e5d4c3d34a33089e4825770fe5a7e5dd8010`;
- blank review template: `4f8f3853a25931145d86e663d70ded54da0d162ce2d17174e5c9df7957b2820b`;
- review rubric: `052232152cbf97a070f2479cbea2289955923d144fc48cc1420238080b2547ef`; and
- review freeze: `3c735ac020c63c62b15bab324258cd3d7a6a5bc457dd2da32793ea5bb29f92f0`.

No hash in a superseded magnetic artifact is a v2 formal commitment. The
commitments above were added only after the regenerated manifest, prompts,
summary, review template, rubric, and freeze agreed on the airflow subtype and
blue square trampoline receiver.

The regenerated artifacts bind subtype `airflow_non_contact_motion`, desk-fan
sources, pinwheel/ribbon receivers, and the black-round/blue-square trampoline
elastic receiver scope.

If this document and a generated artifact disagree, generation and review
must stop. The discrepancy requires a new explicit version; it may not be
resolved by silently editing v1 or by selecting the more favorable rule.
