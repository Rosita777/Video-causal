# Project Handoff

Last updated: 2026-09-10

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
and adjudicator fields are all blank.  Two process-locally answer-key-blind
reviewers must score the same queue independently; a third reviewer adjudicates every human
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
The final human reviewers can remain process-locally answer-key blind, but the
global pre-unblinding state cannot be restored.  The evaluation section must disclose
this protocol deviation.

This deviation also makes the v1 provenance labels
`canonical_anonymous_scores_frozen_before_answer_key_opening` and
`original_eligibility_and_shared_subsets_frozen_before_full_key_opening`
globally false.  Do not run the v1 `freeze`, `freeze-eligibility`, or formal
metrics stages unchanged.  The additive v2 amendment, wrappers, release
environment, reviewer-process layer, final exporter, command ledger, and
prefix DAG are now frozen by implementation commit `b0fffa4fc8d9` and
out-of-tree pre-metric manifest `b3892f5d6b59...`.  All five v1
evaluation-registry-bound implementations remain byte-identical.
The bound release-environment receipt records 64 passed tests with zero
failures or skips; the independently rerun prefix DAG verifier reports 40
materialized nodes and leaves all nine human/canonical/final nodes pending.

## 4. Exact Next Actions

No additional method tuning, training seed, checkpoint selection, generation
replicate, case replacement, or GPU experiment is authorized on the formal
data.

1. Preserve implementation commit `b0fffa4fc8d9` and
   `../causal7m_pre_metric_freeze_v2/`; do not edit a bound component or use a
   direct v1 final entry point.
2. Use the frozen delivery builder to prepare two receipt-bound isolated
   reviewer copies of
   `../causal7m_formal_review_launch_v5/human_audit_stage0/public/`.  Never
   copy its parent directory or any `private/` directory to a reviewer.
3. Have reviewer 1 and reviewer 2 independently complete all 3,633 atomic
   judgments without seeing the other's labels.
4. Merge only the reviewer score/note columns into one schema-preserving
   completed CSV; all binding and context field values must remain exactly
   unchanged.
5. Adjudicate every disagreement with a third reviewer and freeze the initial
   review-process receipt.
6. Run the registered `expand-audit` stage.  Prepare reviewer-specific final
   projections, independently score any added rows, adjudicate, and freeze the
   final process receipt; this round remains required when no expansion fires.
7. Run the v2 `freeze` stage to produce anonymous canonical scores.
8. Run the v2 `freeze-eligibility` stage to freeze
   Original capability and shared subsets.
9. Run the v2 formal metric wrapper and the pre-metric-frozen final
   table exporter.  The existing exporter remains preview-only.

The evidence-independent abstract, Introduction, Related Work, Scope and
Limitations, Conclusion, evaluation protocol, reproducibility appendix, and
experimental setup now have draft prose.
`paper/CITATION_LEDGER.md` records the verified sources and licensed claim
boundaries. Appendix E/F tables, Sections 5.2--5.4, and the quantitative
headline sentences in the abstract and Introduction wait for the final
human-canonical artifacts.

For drafting only, the outline build loads
`paper/tables/preliminary/results_writing_preview_v2_preliminary.tex`, a
post-unblinding VLM-only narrative bound to preview manifest
`fc4c9409f7f4...`. It is non-canonical, absent from prose-mode PDFs, and must
be replaced rather than promoted when human-canonical tables exist.

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
  queue, and exploratory previews);
- `../causal7m_environment_receipt_v1` (64-pass release-environment receipt);
  and
- `../causal7m_pre_metric_freeze_v2` (clean implementation, command ledger,
  18-run inventory, and verified prefix DAG).

Do not clean, move, overwrite, or regenerate these roots.  Private key files,
answer mappings, and blind secrets must not enter Git or be shown to human
reviewers.

## 6. Paper and Claim Boundary

The paper uses the official ICLR 2027 template. The evidence-independent
abstract, Introduction, Related Work, Sections 3 and 6--7, Method 4.1--4.3,
Section 5.1, and Appendices A--D and H have draft prose. Related Work is
source-checked in `paper/CITATION_LEDGER.md`.
Appendices D and H retain explicit completion gates. The final result tables,
result prose, and abstract/Introduction headline sentences remain blocked on
human canonicalization.

The outline-only VLM preview for Sections 5.2--5.4 is a drafting aid, not
result prose licensed for submission. Its positive and negative branches must
be reselected from the generated human-canonical tables rather than promoted
verbatim.

The combined task-and-method schematic is integrated as Figure 1 in the
Introduction, with a Method back-reference. Its vector PDF, readable
publication-size PPTX, caption, and source hashes are recorded in
paper/figures/main/fig_task_method_overview.manifest.json. The separately
planned qualitative comparison is now main Figure 2 and still requires its
metadata-only selection commitment. The schematic contains no result media.

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
