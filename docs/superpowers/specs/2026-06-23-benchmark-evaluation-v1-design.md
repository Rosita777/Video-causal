# Benchmark Evaluation V1 Design

## Goal

Build a reproducible benchmark evaluation layer for causal-footprint video concept erasure. The layer turns the current human-labeled erasure outputs into a stable manifest, a reviewable annotation page, and paper-facing metric tables.

## Scope

This v1 does not generate new videos and does not change any baseline implementation. It uses existing gold rows, contact sheets, and optional VLM predictions to make evaluation easier to audit.

## Inputs

- `experiments/eval_calibration/causal_footprint_v0_gold_outputs.csv`: human labels and atomic visual judgments.
- `experiments/eval_calibration/vlm_inputs.csv`: output contact sheets and optional clean-reference contact sheets.
- Optional VLM prediction CSVs such as `experiments/eval_calibration/claude_sonnet_4_6_reference_atomic_full_predictions.csv`.

## Output Manifest

The v1 manifest is a CSV with one row per baseline output. It keeps stable identifiers, source prompt information, paths, human atomic labels, final human label, and optional VLM labels.

Required columns:

- `sample_id`: stable output id, usually `item_id::baseline`.
- `item_id`
- `mechanism_id`
- `mechanism_type`
- `source_name`
- `target_concept`
- `causal_effect`
- `clean_prompt`
- `erasure_target`
- `baseline`
- `seed`
- `reference_video_path`
- `output_video_path`
- `reference_sheet_path`
- `contact_sheet_path`
- `expected_target_absent`
- `expected_effect_visible`
- `human_target_visible`
- `human_effect_visible`
- `human_separation_clear`
- `human_video_quality`
- `human_label`
- `human_failure_mode`
- `human_notes`
- optional `<model>_label`, `<model>_confidence`, `<model>_reason`, `<model>_disagrees`

## Label Rules

The benchmark keeps four final labels:

- `strict_leakage`: the erased target is absent, the expected downstream effect remains visible, and the effect is visually separable from any remaining source.
- `borderline`: the target/effect separation is partial, the target may have residual cues, or the effect is weak but relevant.
- `target_leakage`: the erased target remains visible enough that the example is not evidence for causeless effects.
- `other_failure`: the effect is not visible, video quality is poor, or the scene has collapsed.

For existing rows, these final labels are read from the gold CSV. The manifest also records the human atomic fields so future labels can be audited.

## Annotation Review

The review page is static HTML. It shows each output contact sheet, optional clean-reference sheet, target/effect metadata, human label, and optional VLM disagreement status. It is meant for human review and disagreement mining, not for browser-side editing.

The accompanying annotation queue CSV contains the same rows plus blank review columns:

- `review_label`
- `review_target_visible`
- `review_effect_visible`
- `review_separation_clear`
- `review_notes`

## Metrics

The metrics script reads the v1 manifest and reports counts/rates by baseline and mechanism:

- total outputs
- strict leakage
- borderline
- relaxed leakage (`strict_leakage` + `borderline`)
- target leakage
- other failure
- strict leakage rate
- relaxed leakage rate
- target leakage rate

If VLM label columns exist, the script also reports agreement and disagreement rates against the human label. This makes Claude/Qwen useful for triage without treating either model as ground truth.

## Artifact Policy

Tracked artifacts should be lightweight CSV/Markdown/HTML files. Large videos remain untracked. Contact-sheet images may remain untracked if they are generated from videos.

## Current V1 Choice

Use the existing 56 human-labeled outputs as the first evaluation manifest. The 36 round4 rows with clean-reference sheets support reference-aware review; the 20 valid5 rows remain useful for metrics but have no clean-reference sheet.
