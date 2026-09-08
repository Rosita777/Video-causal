# Writing plan before prose

Status: accepted writing contract. Progress updated 2026-09-08: Section 3,
Method 4.1--4.3, and Appendices A--C have draft prose. The next evidence-ready
batch is Appendix D/H followed by Section 5.1.

This plan separates three things:

1. the order in which a reader encounters the paper;
2. the order in which we should write and verify it;
3. the evidence gate that licenses every claim.

## 1. What to learn from CleanVideo

CleanVideo is effective because its design appears necessary rather than
arbitrary. Its recurring logic is:

> observation -> structural limitation -> matching design -> decisive result
> -> mechanism evidence -> preservation and efficiency

Transferable techniques:

- The abstract is a complete argument, not a component inventory.
- Introduction figures have distinct jobs: one establishes the premise and
  another previews the proposed mechanism or capability.
- A method section starts with a short roadmap.
- Every method component begins from a concrete failure mode.
- Every equation is followed immediately by symbol definitions, plain-language
  intuition, and the behavior it is intended to change.
- Dense setup and result sections use short bold run-in headings.
- A result paragraph starts with the question or conclusion, reports the
  decisive estimate, discusses difficult cases, and ends at the claim
  boundary.
- The appendix is ordered by reviewer questions: why the method is sensible,
  how it is reproduced, whether evidence is complete, and how data are built.

Practices not to copy:

- Do not invent a proposition or theorem to make the method appear deeper.
- Do not use significant, confirms, indispensable, negligible loss, or
  complete erasure without the exact evidence required by those words.
- Do not let qualitative captions claim preservation when the subject, action,
  receiver, or scene visibly changes.
- Do not show only successful cases or omit the selection rule.
- Do not add hyperparameter, activation, or efficiency sections only because
  the reference paper has them.

## 2. The paper spine

The entire manuscript should support one narrow argument:

1. Object visibility is an incomplete endpoint for erasing physical events
   from generated video because a source-dependent footprint can remain.
2. The same noun must be preserved when it occupies a receiver or bystander
   role, so unconditional lexical suppression is also insufficient.
3. Causal-role erasure therefore targets the registered source role and its
   dependent footprint while preserving receiver, context, quality, and
   noncausal use.
4. Source-slot Randomized Counterfactual Distillation varies source identity
   while holding the independently generated no-source target and training
   trajectory fixed.
5. The Matched Control, two-mechanism identification study,
   implicit-footprint and held-out-source subsets, specificity cases, and
   failure analysis determine which interpretation survives.

The stable vocabulary is:

> source - footprint - specificity

Use causal-role erasure for the task, distributional source-free target for
the supervision target, and role-conditioned behavior for the strongest
behavioral interpretation. Do not rotate through looser synonyms.

## 3. Writing order and readiness gates

The writing order should not follow the final section order.

| Stage | Material | Current readiness | Gate before writing |
| --- | --- | --- | --- |
| 0 | This plan and claim matrix | Accepted | Structure and boundaries frozen for drafting |
| 1 | Section 3 problem formulation | Drafted | Initial prose complete; final compression remains |
| 2 | Section 4 method and Figure 2 specification | Method drafted | Figure 2 asset remains gated on a source manifest and hash |
| 3 | Appendices A--C | Drafted | Initial prose compiled against frozen artifacts |
| 4 | Appendix D protocol and Appendix H provenance | Next | Freeze a deviation-aware provenance amendment; leave final human agreement and final-table hashes open |
| 5 | Section 5.1 experimental setup | Queued after D | Compress only verified appendix facts |
| 6 | Final quantitative Appendices E--F | Not ready | Human calibration, audit, adjudication, and canonical metrics complete |
| 7 | Sections 5.2--5.5 | Not ready | Generated human-canonical tables and registered claim gates available |
| 8 | Appendix G and Section 5.6 | Not ready | Metadata-only selection manifest frozen and disclosed as post-unblinding descriptive |
| 9 | Related Work | Needs source ledger | Closest-work claims and citations verified |
| 10 | Introduction | Partly ready | Problem and method stable; result paragraph waits for canonical evidence |
| 11 | Scope, Conclusion, and ICLR statements | Partly ready | Final evidence boundary and author-reviewed AI-use history known |
| 12 | Abstract | Write last | Every sentence is supported by the completed paper |

Results prose must be distilled from generated Appendix E/F tables. We must
not write a desired result story first and then search the tables for support.

## 4. Paragraph-level plan

### Abstract - write last

Use seven functional sentences:

1. Identify the missed video failure: a source can weaken while its footprint
   remains.
2. Explain why this is role-dependent rather than ordinary object erasure.
3. Define the valid target: reduce source and dependent footprint while
   retaining receiver and noncausal uses.
4. State the one method idea: vary the source-slot identity, hold the no-source
   target fixed, and distill with preservation.
5. State the study: seven-mechanism/eight-stream main experiment plus a
   Water/Fracture identification study.
6. Insert only the final human-canonical headline result.
7. End at behavioral evidence; do not claim internal causal representation.

No preliminary VLM number enters the abstract.

### 1. Introduction

Do not add numbered subsections. Use five paragraphs and three contribution
bullets.

#### Paragraph 1 - Is object disappearance event disappearance?

- Start from the conventional visibility endpoint.
- Introduce residual splash, fracture, deformation, release, or trace.
- Use Figure 1 as a premise figure.
- End with the gap: visual absence of the named object is not sufficient
  evidence that the event has been erased.
- Figure 1 is a schematic or pre-treatment task illustration, not a selected
  method success.

#### Paragraph 2 - Why not erase the noun everywhere?

- Show that the same entity may be source, receiver, or bystander.
- Explain under-erasure and over-erasure as two consequences of the wrong
  intervention unit.
- Derive role, rather than noun identity alone, as the unit of interest.
- Do not describe baselines as literal string deletion unless their methods
  actually do that.

#### Paragraph 3 - What makes the task technically difficult?

- Define the role-conditioned source-free world.
- Introduce implicit footprints, unseen source identities, and non-pixel-
  aligned targets.
- Use this paragraph to make the method requirement inevitable.

#### Paragraph 4 - What is the minimal method explanation?

- Introduce structured event fields.
- Change only the identity in the causal-source slot.
- Hold the dynamic no-source target fixed.
- State counterfactual distillation and preservation as the training response.
- Refer to Figure 2; avoid listing every implementation component.

#### Paragraph 5 - What evidence can falsify the interpretation?

- Preview the seven-mechanism main study and same-backbone Matched Control.
- Introduce the two-mechanism Generic Paraphrase and Bystander Token controls.
- Add implicit-footprint, held-out-source, specificity, and failure evidence.
- The final result sentence waits for human-canonical analysis.

Contributions:

1. task and evaluation protocol;
2. method and single-factor Matched Control;
3. seven-mechanism main evidence plus the two-mechanism identification,
   generalization, preservation, and failure study.

### 2. Related Work

Each subsection should be one compact argument rather than an author list:

> area objective -> method families -> closest setting -> exact remaining gap
> -> our position

#### 2.1 Concept Erasure in Image and Video Diffusion

Organize by suppression unit and finish with the absence of event-role and
dependent-footprint modeling.

#### 2.2 Video Object Removal and Controllable Editing

Distinguish input-video, mask, trajectory, or reference assumptions from
open-ended text-to-video generation of a distributional no-source world.

#### 2.3 Counterfactual Supervision and Role-Based Generalization

Position structured role intervention and paired supervision. State that
roles are provided and behavior is tested; no unsupervised causal discovery is
claimed.

Priority language such as first, only, or no prior work is forbidden until the
source ledger verifies its precise scope.

### 3. Causal-Role Erasure

This is a contribution section, not a generic diffusion preliminary.

#### 3.1 Source, Receiver, and Causal Footprint

Paragraph 1: explain why source visibility alone misses the event consequence.

Paragraph 2: define a registered event with mechanism, source, trigger,
receiver, footprint, and context. Explain that a footprint is source-dependent
evidence appearing after the trigger, not arbitrary co-occurring content.

Paragraph 3: instantiate Water and Fracture in the main text, list all seven
mechanisms once, and send the complete ontology to Appendix A. End by
distinguishing entity identity from event role.

#### 3.2 The Role-Conditioned Source-Free World

Paragraph 1: explain the logical failure of unconditional noun suppression.

Paragraph 2: define the structured source-free target distribution: source,
trigger, and source-dependent footprint are removed; receiver, context, and
natural dynamics remain.

Paragraph 3: state the causal-prompt and noncausal-prompt behavioral contracts.

Paragraph 4: state the scope immediately. The target is distributional, not
frame-aligned; the role is provided by data, not discovered.

#### 3.3 Efficacy, Specificity, and Validity

Paragraph 1: define the three desiderata.

Paragraph 2: introduce Original eligibility, usability, CES, and SU in one
compact block.

Paragraph 3: explain partial credit. Clear-to-partial is a valid improvement;
complete absence and strict success are secondary outcomes.

Full rubric, fixed denominator, aggregation, bootstrap, human audit, and claim
gates belong in Appendix D.

### 4. Source-slot Randomized Counterfactual Distillation

Open with three roadmap sentences only:

1. construct dynamic source-free supervision;
2. vary identity only in the causal-source slot;
3. train with counterfactual distillation and preservation.

Do not preview results in the roadmap.

#### 4.1 Structured Dynamic Counterfactual Pairs

Use three bold run-in paragraphs:

- Structured factual bindings: define prompts from event fields rather than
  post-hoc string replacement.
- Dynamic source-free targets: explain independent generation and the absence
  of pixel alignment.
- Screening and preservation: state 192 candidates to 178 frozen targets per
  mechanism and 36 shared preservation bindings; move templates, rejection
  reasons, examples, and hashes to Appendix B.

This subsection explains supervision. It does not establish that the method
outperforms Matched because both arms share these targets.

#### 4.2 Randomizing the Causal-Source Slot

This is the method-novelty center and receives the most space.

Paragraph 1: fixed training identities permit noun or event-template
shortcuts.

Paragraph 2: define the deterministic balanced assignment from the frozen
64-item source bank, with assigned identity different from the original.

Paragraph 3: place Matched and SRCD training pairs side by side. Immediately
list what changes and what remains byte- or value-identical.

Paragraph 4: give the hypothesis, not a theorem. Identity variation makes the
identity shortcut unreliable and places pressure on the adapter to associate
suppression with the registered source slot and dependent footprint.

Paragraph 5: preview its falsification interfaces: Matched, held-out,
implicit, Generic Paraphrase, Bystander Token, and specificity. State that the
last two controls cover only Water and Fracture.

#### 4.3 Counterfactual Distillation and Preservation

Use three bold run-in paragraphs:

- Counterfactual erasure update: noised target latent, flow target, student,
  frozen source-free teacher, and stop-gradient.
- Preservation update: frozen-model matching on generic non-target prompts.
- Schedule and scope: alternating 100 erasure and 100 preservation updates,
  registered weights, LoRA, checkpoint, and one adapter per mechanism.

After each equation: define symbols, explain the mechanism in plain language,
and identify the later measurement.

Do not write a summed overall loss if the implementation uses alternating
updates. Do not claim that every loss component is necessary because no
registered component-removal ablation exists. The Matched comparison isolates
source-slot replacement, not every training ingredient.

### 5. Experiments

Use the evidence order:

> overall efficacy -> role identification -> surface-form generalization ->
> preservation and failures -> qualitative evidence

#### 5.1 Experimental Setup

Keep this near half a page with bold run-in headings:

- Seven mechanisms and cases.
- Models and baselines in separate Wan and CogVideoX blocks.
- Full-video evaluation, CES/SU, human-canonical authority, and statistics.

No result or preliminary number belongs in this subsection.

#### 5.2 Main Results across Seven Mechanisms

Paragraph 1: Table 1 observed eight-by-seven ranking.

Paragraph 2: Table 2 same-backbone SRCD-minus-Matched estimate, interval, and
multiplicity status.

Paragraph 3: mechanism heterogeneity and difficult mechanisms.

Paragraph 4: preservation guardrails and external shared-capability
estimability boundary.

Template:

> question or licensed conclusion -> decisive estimate and interval ->
> heterogeneity -> preservation -> boundary

#### 5.3 Role or Word?

Paragraph 1: explain what the two controls remove as alternative
explanations.

Paragraph 2: Generic Paraphrase causal result, positive implicit-footprint
point-estimate condition, and specificity result.

Paragraph 3: Bystander Token causal result, positive implicit-footprint
point-estimate condition, and specificity result.

Paragraph 4: joint interpretation and the Water/Fracture-only scope.

If either registered guardrail fails, write partial support or consistent
with; do not claim that the full role-conditioned criterion passed.

#### 5.4 Beyond Explicit Footprints and Seen Sources

Paragraph 1: implicit-footprint contrast.

Paragraph 2: held-out-source contrast.

Paragraph 3: subset overlap, mechanism heterogeneity, and the exact behavioral
interpretation. Do not call the two subsets independent replications.

#### 5.5 Preservation and Failure Analysis

Paragraph 1: separate gated source and footprint contributions.

Paragraph 2: contrast graded suppression with complete absence and strict
success.

Paragraph 3: report weak and negative mechanisms.

Paragraph 4: report receiver, quality, usability, Original capability, and
cross-backbone limits.

#### 5.6 Qualitative Comparisons

Paragraph 1: disclose the metadata-only deterministic rule and that it is
frozen after aggregate unblinding, so the display is descriptive. The rule
selects one explicit causal, one implicit causal, and one specificity case; it
does not select outcomes.

Paragraph 2: describe the actual visible outcomes using only observable facts;
do not guarantee that any selected case is a success or failure.

The selection manifest must be frozen before Figure 3 is generated. After the
figure is viewed, no case replacement is allowed.

### 6. Scope and Limitations

Use four compact paragraphs:

1. seven mechanism-specific adapters, not one universal adapter;
2. distributional targets and provided roles, not individual or discovered
   causality;
3. single training seed, capability limits, cross-backbone estimability, and
   mechanism scope;
4. the full-key protocol deviation and remaining answer-key-blind human
   review.

### 7. Conclusion

Use three moves:

1. recover the overlooked task;
2. recover the source-slot intervention;
3. state the final supported behavioral conclusion.

Do not add numbers, new applications, or a stronger causal claim.

### Required ICLR 2027 statements

Place these unnumbered sections after the conclusion and before references:

- AI Use Statement: mandatory, author-reviewed, and matched to the actual use
  of generative AI across methodology, code, analysis, interpretation,
  literature work, structure, and writing.
- Ethics Statement: recommended here because selective video-model editing has
  both safety and misuse implications.
- Reproducibility Statement: a short navigation paragraph pointing to the
  relevant main sections, appendices, and anonymous artifacts.

These statements do not count toward the ICLR 2027 main-text page limit.
References follow them, and appendices follow the references.

## 5. Figure and table choreography

| ID | Function | Placement | Readiness gate |
| --- | --- | --- | --- |
| Figure 1 | Premise: object visibility is not event absence | Introduction P1 | Schematic/pre-treatment only; no method-success selection |
| Figure 2 | Mechanism: what changes and what is fixed | Section 4 opening | Method and Matched contracts verified |
| Figure 3 | Explicit causal, implicit causal, and specificity cases | Section 5.6 | Post-unblinding metadata-only category selection manifest frozen |
| Table 1 | Eight methods by seven mechanisms plus macro CES | Section 5.2 | Final human-canonical table |
| Table 2 | Wan single-factor contrast and preservation guardrails | Section 5.2 | Final CI, Holm, NI, and claim gates |
| Table 3 | Two-mechanism identification controls | Section 5.3 | Final causal, implicit-point, and specificity results |
| Table 4 | Implicit-footprint and held-out-source evidence | Section 5.4 | Final subset contrasts and counts |

Figure 2 should have four panels:

1. factual event anatomy;
2. independently generated no-source target;
3. Matched versus randomized source slot with all fixed fields visible;
4. student, frozen teacher, and preservation paths.

All temporal grids use fixed frame indices and method order, visibly separate
Wan and CogVideoX, and annotate source, receiver, and footprint.

## 6. Appendix writing sequence

Appendix writing follows evidence readiness, not alphabetical order:

1. A Task Scope and Ontology.
2. B Counterfactual Data Construction.
3. C Training, Baselines, and Generation Integrity.
4. D Evaluation, protocol deviation, human calibration, metrics, and claim
   gates.
5. H Reproducibility facts already frozen.
6. E Complete Results after human canonicalization.
7. F Role Controls and Generalization after canonical metrics.
8. G Qualitative and Failure Cases after the selection manifest is frozen.

Appendix E extends main Tables 1--2 with uncertainty, decomposition,
capability, complete contrasts, non-estimable reasons, and secondary outcomes;
it does not repeat the same compact tables verbatim.

The A/B atomic-mean aggregation analysis is explicitly post-unblinding,
non-canonical evaluator sensitivity. It is not a final primary endpoint or a
claim gate. Final evaluator evidence is VLM-human agreement, coverage,
adjudication, and human-canonical scores.

## 7. Claim-to-evidence matrix

| Intended statement | Required evidence | Current final-writing status |
| --- | --- | --- |
| Object-only erasure misses causal footprints | Task definition, related work, Figure 1 | Ready after citation verification |
| Source-slot replacement improves causal erasure | Human-canonical SRCD versus Matched paired contrast | Wait for human canonical |
| One principle applies across seven mechanisms | Table 1 macro plus all mechanism rows and heterogeneity | Wait for human canonical |
| Behavior extends beyond explicit footprint words | Final implicit-footprint contrast | Wait for human canonical |
| Behavior extends beyond training identities | Final held-out-source contrast | Wait for human canonical |
| Evidence disfavors noun-frequency explanation | Bystander causal result, positive implicit point estimate, and specificity gate | Wait for human canonical |
| Evidence disfavors ordinary paraphrasing | Generic causal result, positive implicit point estimate, and specificity gate | Wait for human canonical |
| Preservation remains within tolerance | All registered NI confidence bounds versus margins | Wait for human canonical |
| Method beats all external baselines | Every shared-capability contrast estimable and passes Holm gate | Currently unsupported; do not plan as headline |
| Model learned true causal representation | No behavioral result is sufficient | Prohibited |
| Source and footprint are completely removed | Complete-absence and strict-success evidence | Not the task requirement; prohibited as general claim |

The paper must disclose that the full method key was opened by a read-only
audit before human canonicalization and before the exploratory A/B-mean rule
was materialized. Final human labels can remain answer-key blind, but the
global pre-unblinding state cannot be restored.

The current v1 canonicalizer and metric builder encode provenance states that
assume the answer key and full method key have not been opened. They must not
produce final artifacts unchanged. A versioned amendment must preserve every
scoring, expansion, estimability, and claim rule while distinguishing
process-local answer-key-blind human review from the already-opened global key.
This amendment must be additive: all evaluation-registry-bound v1 files remain
byte-identical, while new versioned files or wrappers bind the old registry and
the amendment digest. The final human-canonical table exporter is implemented
and frozen before canonical metrics are opened.

## 8. Paragraph style contract

- The first sentence states the question or conclusion.
- A result paragraph contains an estimate and interval, not only adjectives.
- One paragraph performs one argumentative job.
- Use stable terms instead of decorative synonym changes.
- Use transitions only to expose logic: however for a limitation, to address
  for a design response, in contrast for a controlled comparison, and
  importantly only for a consequence that changes the claim.
- Equations appear only when later prose uses every symbol.
- Every equation is followed by a plain-language interpretation.
- Captions state what the reader should verify and the scope of that
  observation.
- Cross-backbone raw values are labeled descriptive.
- Non-estimable is shown as NE, never as a null result or p-value of one.
- Non-inferiority tables report estimate, interval, margin, and pass/fail;
  superiority p-values relative to zero are not labeled NI p-values.
- Avoid defensive prose. State the scope directly and continue.

Preferred claim verbs:

- define, isolate, measure, retain, reduce;
- yields a positive observed difference;
- is consistent with role-conditioned behavior;
- disfavors a specific lexical explanation;
- remains within a prespecified margin.

Prohibited without stronger evidence:

- proves, discovers, guarantees, universally;
- significantly outperforms all baselines;
- completely erases;
- every component is indispensable;
- learns causality.

## 9. Section exit checklist

A section is ready for author review only if:

1. its opening sentence states one clear job;
2. every factual statement points to a frozen artifact or verified citation;
3. every method design maps to an experiment or is labeled implementation;
4. every result claim maps to a generated human-canonical table;
5. failure and boundary sentences are adjacent to the claim they limit;
6. figures and captions do not exceed visible evidence;
7. terminology matches preamble/macros.tex;
8. there is no preliminary result imported into a final table;
9. there is no unverified priority claim;
10. removing the subsection would leave one identifiable reviewer question
    unanswered.

## 10. First writing batch after plan approval

Status: complete. Section 3, Method 4.1--4.3, and Appendices A--C now have
evidence-grounded draft prose. Figure 2 remains a separately gated asset.

Do not begin with the abstract or introduction. The first batch should be:

1. Section 3.1--3.3;
2. Section 4.1--4.2 and the Figure 2 specification;
3. Appendix A;
4. Appendix B.

Review that batch for logical necessity and terminology before writing the
loss equations, experiment setup, or any result prose.
