# Project Handoff

Last updated: 2026-09-09

This is the authoritative handoff document. When another document conflicts
with this one, treat the other document as historical until explicitly updated.

## 1. Current Formal Experiment

The active protocol is `causal_role_erasure_7m_single_seed_v2`.  It evaluates
one source-slot training principle across seven equal-status mechanism-specific
operators: water impact, rigid collision, brittle fracture, powder impact,
elastic deformation, material release, and surface trace.  Airflow was
excluded by a frozen Original-capability amendment before formal training or
treatment generation and was not replaced.

The eight main streams are:

1. Wan Original;
2. Matched Control;
3. Source-slot Randomized Counterfactual Distillation (SRCD), recorded as
   `V4` in frozen artifacts;
4. CogVideoX Original;
5. Negative Prompt;
6. VideoEraser (official CogVideoX);
7. T2VUnlearning-adapted (ours); and
8. SAFREE-CogVideoX.

Here “(ours)” denotes the internal adaptation required for that baseline; it
does not refer to the proposed SRCD method.

The causal comparison that isolates the proposed intervention is SRCD versus
Matched on Wan.  The CogVideoX streams form a separately normalized benchmark
block.  Raw Wan--CogVideoX values are not a pure method ablation.

## 2. Completed Data, Training, and Generation

Each mechanism has 192 generated target candidates, exactly 178 frozen
source-free targets, a 64-item source bank containing 8 original training
sources and 56 augmentation-only sources, and 36 generic preservation bindings
shared across mechanism-specific adapters.  The selected target inventory is
therefore 7 x 178 = 1,246.

All 18 registered Wan method/control runs completed with an eligible step-200
checkpoint and passed finite-parameter validation:

- 7 Matched Control adapters;
- 7 SRCD adapters; and
- 4 Water/Fracture identification adapters: two Generic Paraphrase and two
  Bystander Token.

All use seed 26000, rank/alpha 16/16, learning rate `5e-5`, 200 strictly
alternating updates (100 erasure and 100 preservation), teacher and
preservation weights of 4, and inference scale 1.25.  Seven separate
T2VUnlearning-adapted CogVideoX operators also completed at their precommitted
step-100 checkpoints.

Formal generation is complete:

| Block | Jobs | Expected | Validated |
| --- | ---: | ---: | ---: |
| Wan Original | 7 | 294 | 294 |
| Trained Wan | 18 | 684 | 684 |
| CogVideoX core four | 28 | 1,176 | 1,176 |
| SAFREE-CogVideoX | 7 | 294 | 294 |
| **Total** | **60** | **2,448** | **2,448** |

The total contains 2,352 main-study videos (294 semantic cases x 8 streams)
and 96 Water/Fracture identification additions.  The anonymous package receipt
has status `frozen_after_exact_2448_media_and_blinding_validation` and SHA-256
`63933393ea04883bbec1f3d383f5ef49d4b933adf66040755bb622ad0170837c`.

## 3. Evaluation State

Two blinded, independently ordered passes of the same `gpt-5.6-luna` judge
are complete.  Each pass covers all 2,448 videos and 9,792 video-field atoms;
together they contain 19,584 scored values, satisfy the registered schema, and
record zero scientific fallback values.  The formal launch receipt has status
`completed_two_independent_schema_valid_passes` and SHA-256
`833254ac9b38a609f233b0d09d9507d316326ab46a6956a0d07b4c1fdf0d3ff1`.

Human labeling has not started.  The frozen initial audit contains
3,633 atomic judgments over 2,009 anonymous videos.  Reviewer 1, reviewer 2,
and adjudicator fields are all blank.  Two answer-key-blind reviewers must
score the same queue independently; a third reviewer adjudicates every human
disagreement.  If a mechanism-by-field stratum has greater than 5% error among
the audited high-confidence VLM agreements, the frozen workflow expands that
stratum before canonical scores are frozen.

No `canonical_scores`, canonical manifest, Original-eligibility freeze, or
formal metric manifest exists yet.  Current numerical exports are labeled
`POST_UNBLINDING_EXPLORATORY_PRELIMINARY`, `NOT_CANONICAL`, and
`WRITING_PREVIEW_ONLY_NOT_FOR_FINAL_SCIENTIFIC_CLAIMS`; they cannot be copied
into final paper tables or result claims.

A separate read-only audit opened the complete method key before human
canonicalization and before the exploratory A/B-mean rule was materialized.
The final human reviewers can remain answer-key blind, but the global
pre-unblinding state cannot be restored.  The evaluation section must disclose
this protocol deviation.

This deviation also makes the current v1 provenance labels
`canonical_anonymous_scores_frozen_before_answer_key_opening` and
`original_eligibility_and_shared_subsets_frozen_before_full_key_opening`
globally false.  Do not run the v1 `freeze`, `freeze-eligibility`, or formal
metrics stages unchanged.  Before canonical freeze, create and freeze a
versioned amendment that preserves the scoring, audit-expansion, and
estimability rules while recording that the reviewers remain process-locally
answer-key blind after a separate process opened the global key.
Keep every v1 evaluation-registry-bound implementation byte-identical.  The
amendment must be additive: new versioned files or wrappers bind the old
registry and the new amendment digest rather than rewriting frozen v1 code.

## 4. Exact Next Actions

No additional method tuning, training seed, checkpoint selection, generation
replicate, case replacement, or GPU experiment is authorized on the formal
data.

1. Freeze the deviation-aware canonicalization and metric-provenance amendment
   described above; do not alter any score, audit, or claim threshold and do
   not edit any v1 registry-bound implementation.
2. Implement and freeze the human-canonical table exporter before opening
   canonical metrics, so presentation cannot be tailored to observed values.
3. Freeze sanitized 0/1/2 human-review instructions and two isolated copies of
   `../causal7m_formal_review_launch_v5/human_audit_stage0/public/`.  Never
   copy its parent directory or any `private/` directory to a reviewer.
4. Have reviewer 1 and reviewer 2 independently complete all 3,633 atomic
   judgments without seeing the other's labels.
5. Merge only the reviewer score/note columns into one schema-preserving
   completed CSV; all binding and context field values must remain exactly
   unchanged.
6. Adjudicate every disagreement with a third reviewer.
7. Run the amended `expand-audit` stage.  If it emits an expansion, repeat
   independent review and adjudication for the added atoms.
8. Run the amended `freeze` stage to produce anonymous canonical scores.
9. Run the amended `freeze-eligibility` stage to freeze
   Original capability and shared subsets.
10. Run the amended formal metric builder and the pre-metric-frozen final
    table exporter.  The existing exporter remains preview-only.

The evidence-independent evaluation protocol, reproducibility appendix, and
experimental setup now have draft prose.  Citation-ledger construction and
Related Work may proceed in parallel with human review.  Appendix E/F tables,
Sections 5.2--5.5, the abstract result sentence, and the introduction's result
paragraph wait for the final human-canonical artifacts.

## 5. Artifact Locations

The persistent A100 project is:

`/data/xiaohuang_workspace/ljc/Video-causal-v4`

Formal outputs are under:

`outputs/causal_role_erasure_7mechanism_main_v2`

Important A100 records include:

- `training_inputs_v2/training_input_registry.json`;
- `training/*/run_receipt.json` (18 Wan receipts);
- `baseline_registry_v2_final/baseline_registry.json`;
- `formal_eval_v2/eval_aggregate.json`;
- `formal_wan_original_v2/wan_original_aggregate.json`;
- `formal_baselines_core4_v2/aggregate.json`; and
- `formal_baselines_safree_v2/aggregate.json`.

Frozen local workspaces outside this checkout are:

- `../causal7m_formal_media_snapshot_v1` (2,154 non-Wan-Original videos);
- `../causal7m_wan_original_snapshot_v1` (294 Wan Original videos); and
- `../causal7m_formal_review_launch_v5` (review package, VLM passes, audit
  queue, and exploratory previews).

Do not clean, move, overwrite, or regenerate these roots.  Private key files,
answer mappings, and blind secrets must not enter Git or be shown to human
reviewers.

## 6. Paper and Claim Boundary

The paper uses the official ICLR 2027 template. Section 3, Method 4.1--4.3,
Section 5.1, and Appendices A--D and H have draft prose. Appendices D and H
retain explicit completion gates. The final result tables and result prose
remain blocked on human canonicalization.

Permitted before final scoring:

- task, ontology, data-construction, method, and evaluation definitions;
- registered settings and exact execution counts;
- checkpoint and generation-integrity statements bound to frozen receipts.

Not permitted before final scoring:

- a final CES/SU value or ranking;
- a significance, non-inferiority, or win claim;
- a claim of superiority over all external baselines.

The paper never claims complete erasure, one universal seven-mechanism
adapter, or discovery of an internal causal representation.  The method's
strongest possible conclusion is behavioral and must follow the registered
Matched, identification, implicit-footprint, held-out-source, specificity, and
preservation gates.

## Historical Development References

The earlier water-impact, v3b/v3c, and failed v4-development gates remain
available as dated forensic records rather than duplicated current-state text:

- `water_impact_dynamic_v3b_eval12_results_2026-08-16.md`;
- `water_impact_dynamic_v3c_fresh_dev24_results_2026-08-16.md`;
- `water_impact_dynamic_v4_source_slot_randomization.md`;
- `../results/water_impact_dynamic_v4_causal_screening_termination_v2.md`;
- `../results/water_impact_dynamic_v4_causal_screening_termination_v3.md`; and
- `experiment_log.md`.

The complete 2026-08-16 handoff snapshot is also recoverable from Git history.
None of these historical next-step instructions supersedes Sections 1--6 above.
