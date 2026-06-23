# Causal Footprint Benchmark v0

Status: candidate-pair construction, clean-source gating, first formal item file, first metric tables, and round5 taxonomy-balanced expansion.

This benchmark targets causal footprint leakage in video concept erasure: the source concept `C` is weak or absent after erasure, but the downstream footprint `F(C)` remains visible.

The benchmark is intentionally built in stages:

1. `candidate_pairs.tsv`: auditable causal-pair pool with accepted, exploratory, and rejected rows.
2. `control_prompts.jsonl`: control prompts for natural-footprint, no-footprint, and alternative-cause calibration.
3. `round4_clean_expansion_prompts.tsv`: first taxonomy-driven clean-source expansion pass.
4. `round5_taxonomy_expansion_prompts.tsv`: broader taxonomy-balanced candidate pool for the next clean-source pass.
5. `items.jsonl`: current clean-source-gated benchmark rows with linked baseline annotations.

Do not treat every candidate row as a benchmark item. Rows enter `items.jsonl` only after passing clean-source generation and annotation gates, then being run through the required erasure baselines.

## Current Formal Artifact

The current benchmark-v0 source of truth is:

```text
benchmarks/causal_footprint_v0/items.jsonl
```

It is generated from:

```text
benchmarks/causal_footprint_v0/export_valid5_manifest.json
experiments/baseline_runs/causal_footprint_v0_valid5_all_step20_parallel_summary.csv
benchmarks/causal_footprint_v0/export_round4_clean_valid9_manifest.json
experiments/baseline_runs/causal_footprint_v0_round4_valid9_all_step20_parallel_summary.csv
```

Current size:

- 14 clean-source-gated benchmark items.
- 56 erasure outputs across Negative Prompt, SAFREE-CogVideoX, VideoEraser local, and T2V proxy.

Regenerate:

```bash
PYTHONNOUSERSITE=1 /home/deepseek_VG/.conda/envs/vcecf/bin/python scripts/build_benchmark_items.py \
  --source valid5,benchmarks/causal_footprint_v0/export_valid5_manifest.json,experiments/baseline_runs/causal_footprint_v0_valid5_all_step20_parallel_summary.csv \
  --source round4_valid9,benchmarks/causal_footprint_v0/export_round4_clean_valid9_manifest.json,experiments/baseline_runs/causal_footprint_v0_round4_valid9_all_step20_parallel_summary.csv \
  --output benchmarks/causal_footprint_v0/items.jsonl
```

## Current Metrics

Metric tables are generated from `items.jsonl`:

```text
experiments/metrics/causal_footprint_v0_metrics_by_baseline.csv
experiments/metrics/causal_footprint_v0_metrics_by_mechanism.csv
experiments/metrics/causal_footprint_v0_metrics_summary.md
```

Regenerate:

```bash
PYTHONNOUSERSITE=1 /home/deepseek_VG/.conda/envs/vcecf/bin/python scripts/compute_benchmark_metrics.py \
  --items benchmarks/causal_footprint_v0/items.jsonl \
  --output-dir experiments/metrics
```

Current conservative headline:

- Strict causal-footprint leakage: 24 / 56.
- Borderline causal-footprint cases: 12 / 56.
- Target-leakage failures: 14 / 56.

Metric meanings:

- `strict_leakage_count`: `usable_for_claim == yes`; target is absent and the downstream footprint remains visible enough for the claim.
- `borderline_count`: `usable_for_claim == borderline`; useful for analysis but not headline evidence.
- `target_leakage_count`: target concept remains visible, so the case does not prove causal-footprint leakage.
- `strict_leakage_given_target_erased`: strict leakage divided by outputs with `target_visible == no`.

## Evaluation V1 Manifest

The current evaluation layer is the preferred entry point for paper-facing tables and human review:

```text
experiments/evaluation/causal_footprint_v1_manifest.csv
experiments/evaluation/review.html
experiments/evaluation/annotation_queue.csv
experiments/evaluation/metrics_by_baseline.csv
experiments/evaluation/metrics_by_mechanism.csv
experiments/evaluation/model_agreement.csv
experiments/evaluation/metrics_summary.md
```

Regenerate:

```bash
PYTHONNOUSERSITE=1 /home/deepseek_VG/.conda/envs/vcecf/bin/python scripts/build_evaluation_manifest.py \
  --gold experiments/eval_calibration/causal_footprint_v0_gold_outputs.csv \
  --vlm-inputs experiments/eval_calibration/vlm_inputs.csv \
  --prediction claude=experiments/eval_calibration/claude_sonnet_4_6_reference_atomic_full_predictions.csv \
  --output experiments/evaluation/causal_footprint_v1_manifest.csv

PYTHONNOUSERSITE=1 /home/deepseek_VG/.conda/envs/vcecf/bin/python scripts/build_annotation_review.py \
  --manifest experiments/evaluation/causal_footprint_v1_manifest.csv \
  --output-dir experiments/evaluation \
  --project-root /home/deepseek_VG/JUNCHI/Video-causal

PYTHONNOUSERSITE=1 /home/deepseek_VG/.conda/envs/vcecf/bin/python scripts/compute_evaluation_metrics.py \
  --manifest experiments/evaluation/causal_footprint_v1_manifest.csv \
  --output-dir experiments/evaluation
```

Current v1 result:

- Strict leakage: 24 / 56.
- Borderline: 12 / 56.
- Relaxed leakage: 36 / 56.
- Target leakage: 14 / 56.
- Claude exact agreement with human labels: 12 / 36 on the reference-backed subset.

The review page is static. Use `annotation_queue.csv` for manual revision columns; keep final benchmark labels in the source gold data after review.

## Evaluator Calibration Interface

Automatic video scorers should be calibrated against the current human labels before their numbers are used in the paper. The gold file is:

```text
experiments/eval_calibration/causal_footprint_v0_gold_outputs.csv
```

Regenerate it from `items.jsonl`:

```bash
PYTHONNOUSERSITE=1 /home/deepseek_VG/.conda/envs/vcecf/bin/python scripts/export_calibration_gold.py \
  --items benchmarks/causal_footprint_v0/items.jsonl \
  --output experiments/eval_calibration/causal_footprint_v0_gold_outputs.csv
```

Prediction CSVs from any evaluator must contain:

```text
item_id,baseline,video_path,target_absent,effect_visible,quality_ok,pred_label,confidence,reason
```

Allowed `pred_label` values:

```text
strict_leakage
borderline
target_leakage
other_failure
```

Run calibration:

```bash
PYTHONNOUSERSITE=1 /home/deepseek_VG/.conda/envs/vcecf/bin/python scripts/calibrate_evaluator.py \
  --gold experiments/eval_calibration/causal_footprint_v0_gold_outputs.csv \
  --predictions experiments/eval_calibration/example_predictions.csv \
  --output-dir experiments/eval_calibration
```

Current calibration artifacts:

```text
experiments/eval_calibration/example_predictions.csv
experiments/eval_calibration/calibration_metrics_by_label.csv
experiments/eval_calibration/calibration_confusion_matrix.csv
experiments/eval_calibration/calibration_metrics_summary.md
```

`example_predictions.csv` is an oracle copy of the human labels for schema testing only. It is not an automatic evaluator result.

## Contact-Sheet VLM Inputs

The first real scorer interface uses 5-frame contact sheets rather than raw videos. This keeps model calls cheaper and mirrors the current human review workflow.

Generate VLM inputs:

```bash
PYTHONNOUSERSITE=1 /home/deepseek_VG/.conda/envs/vcecf/bin/python scripts/build_vlm_eval_inputs.py \
  --gold experiments/eval_calibration/causal_footprint_v0_gold_outputs.csv \
  --sheet-dir experiments/eval_calibration/frame_sheets \
  --output experiments/eval_calibration/vlm_inputs.csv
```

Current local generation result:

```text
vlm input rows: 56
contact sheets generated locally: 56
missing videos: 0
```

The contact-sheet JPEGs under `experiments/eval_calibration/frame_sheets/` are generated media and are intentionally ignored by git. Regenerate them from local videos when needed.

Generate dry-run request payloads:

```bash
PYTHONNOUSERSITE=1 /home/deepseek_VG/.conda/envs/vcecf/bin/python scripts/evaluate_with_vlm.py \
  --inputs experiments/eval_calibration/vlm_inputs.csv \
  --output-jsonl experiments/eval_calibration/vlm_payloads_dryrun.jsonl \
  --dry-run
```

`vlm_payloads_dryrun.jsonl` contains the exact prompt, image path, target concept, expected effect, and required JSON response schema for each scorer call. It does not include `human_label`, so gold labels are not leaked into model prompts.

When `clean_reference.video_path` exists in `items.jsonl`, `vlm_inputs.csv` also contains a clean-reference contact sheet. Current reference coverage is 36 / 56 rows, all from `round4_valid9`; the older `valid5` rows do not have clean-reference videos. Pass `--require-reference` to evaluate only rows with the two-image clean-reference / erased-output protocol.

## GPT-4o Scorer Status

The preferred primary VLM judge is `openai/gpt-4o`, because it is a mainstream model that is easier to justify in a paper. On 2026-06-22, the provided `https://api.360.cn/v1` default group listed the model but returned:

```text
gpt-4o: no available channel
```

The implemented API path is ready, but the full GPT-4o sample is blocked until that channel is available.

Fallback smoke:

```text
model: openai/gpt-4o-mini
sample rows: first 8 VLM inputs
matched predictions: 8
strict leakage binary F1: 0.4000
relaxed leakage binary F1: 0.7692
macro F1: 0.1000
```

`gpt-4o-mini` predicted `strict_leakage` for all 8 sample rows, including target-leakage and borderline human labels. It should not be used as the main judge.

Current reference-aware atomic trial results:

```text
model: qwen/qwen-vl-plus
protocol: atomic visual fields with clean-reference image
sample rows: all 36 rows with reference sheets
strict leakage binary F1: 0.6087
relaxed leakage binary F1: 0.8364
macro F1: 0.3060
strict leakage precision: 0.4516
strict leakage recall: 0.9333
status: high-recall but over-strict; misses all borderline rows
artifact policy: summarized in docs, raw outputs not retained
```

```text
model: anthropic/claude-sonnet-4-6
protocol: atomic visual fields with clean-reference image
sample rows: all 36 rows with reference sheets
strict leakage binary F1: 0.4000
relaxed leakage binary F1: 0.7600
macro F1: 0.3438
strict leakage precision: 0.8000
strict leakage recall: 0.2667
```

This creates a useful diagnostic contrast: `qwen/qwen-vl-plus` is high-recall but over-strict, while `anthropic/claude-sonnet-4-6` is conservative and has low strict-leakage recall. Neither is ready as the final automatic judge.

Artifacts:

```text
experiments/eval_calibration/gpt4o_mini_sample8_predictions.csv
experiments/eval_calibration/gpt4o_mini_sample8_raw.jsonl
experiments/eval_calibration/gpt4o_mini_sample8/
experiments/eval_calibration/claude_sonnet_4_6_reference_atomic_full_predictions.csv
experiments/eval_calibration/claude_sonnet_4_6_reference_atomic_full_raw.jsonl
experiments/eval_calibration/claude_sonnet_4_6_reference_atomic_full/
```

Run the current retained reference-aware atomic scorer:

```bash
PYTHONNOUSERSITE=1 /home/deepseek_VG/.conda/envs/vcecf/bin/python scripts/evaluate_with_vlm.py \
  --inputs experiments/eval_calibration/vlm_inputs.csv \
  --output-predictions experiments/eval_calibration/claude_sonnet_4_6_reference_atomic_full_predictions.csv \
  --raw-output-jsonl experiments/eval_calibration/claude_sonnet_4_6_reference_atomic_full_raw.jsonl \
  --run-api \
  --model anthropic/claude-sonnet-4-6 \
  --api-config-file /path/to/local/token.txt \
  --require-reference \
  --temperature 0 \
  --max-tokens 1000 \
  --timeout 180

PYTHONNOUSERSITE=1 /home/deepseek_VG/.conda/envs/vcecf/bin/python scripts/calibrate_evaluator.py \
  --gold experiments/eval_calibration/causal_footprint_v0_gold_outputs.csv \
  --predictions experiments/eval_calibration/claude_sonnet_4_6_reference_atomic_full_predictions.csv \
  --output-dir experiments/eval_calibration/claude_sonnet_4_6_reference_atomic_full \
  --allow-partial
```

## Current Clean-Source Slices

The current strict baseline slice is `valid5`:

```text
prompts/causal_footprint_v0_valid5.txt
benchmarks/causal_footprint_v0/export_valid5_manifest.json
```

It is useful for proof-of-problem baseline evidence, but too small for the final benchmark.

Round4 expands the clean-source pool with 48 controlled prompt variants:

```text
benchmarks/causal_footprint_v0/round4_clean_expansion_prompts.tsv
prompts/causal_footprint_v0_round4_clean_expansion48.txt
experiments/clean_screening/causal_footprint_v0_round4_clean_expansion48_initial_labels.csv
```

Initial round4 screening keeps 9 rows as clean-valid and 11 rows as backup/borderline. The clean-valid slice is:

```text
prompts/causal_footprint_v0_round4_clean_valid9.txt
benchmarks/causal_footprint_v0/export_round4_clean_valid9_manifest.json
```

This slice has now been run on Negative Prompt, SAFREE-CogVideoX, VideoEraser local, and T2V proxy. Conservative labels are tracked in:

```text
experiments/baseline_runs/causal_footprint_v0_round4_valid9_all_step20_parallel_summary.csv
```

Round4 is not a replacement for the candidate-pair taxonomy. It is the clean-source generation pass that tests which taxonomy-driven pairs CogVideoX-2B can actually render well enough for erasure evaluation.

## Round5 Taxonomy Expansion

Round5 is the active clean-source candidate pool. It expands beyond the current water/ball-heavy valid slice and removes semantic-response prompts such as remote controls and buttons.

```text
benchmarks/causal_footprint_v0/round5_taxonomy_expansion_prompts.tsv
prompts/causal_footprint_v0_round5_taxonomy_expansion60.txt
```

Current composition:

```text
fluid_impact: 10
surface_trace: 10
fracture_damage: 10
elastic_deformation: 10
field_mediated: 10
particle_dispersion: 10
```

Dry-run command used to verify prompt parsing and manifest creation:

```bash
PYTHONNOUSERSITE=1 /home/deepseek_VG/.conda/envs/vcecf/bin/python scripts/generate_cogvideox_clean.py \
  --prompts prompts/causal_footprint_v0_round5_taxonomy_expansion60.txt \
  --output-dir outputs/causal_footprint_v0_round5_taxonomy_expansion60_dryrun \
  --model zai-org/CogVideoX-2b \
  --seed 5200 \
  --steps 20 \
  --guidance-scale 6.0 \
  --num-frames 49 \
  --fps 8 \
  --dtype bf16 \
  --dry-run
```

Real CogVideoX-2B clean-source generation is complete for all 60 prompts.

```text
outputs/causal_footprint_v0_round5_taxonomy_expansion60_bf16_step20_parallel/clean/generation_manifest.json
outputs/analysis_contact_sheets/causal_footprint_v0_round5_taxonomy_expansion60_step20/clean_gallery.html
outputs/analysis_contact_sheets/causal_footprint_v0_round5_taxonomy_expansion60_step20/clean_source_screening.csv
```

Generation settings:

```text
model: models/CogVideoX-2b
seed: 5200
steps: 20
guidance_scale: 6.0
frames: 49
resolution: 720x480
dtype: bf16
```

Run note: the first 8-GPU pass produced 51 / 60 videos and OOMed on 9 jobs assigned to GPU5 while unrelated `dyme` jobs occupied most GPU memory. Retrying only the 9 failed prompt indices on GPUs `0,1,2,3,4,6,7` succeeded, yielding 60 / 60 videos. Generated media and gallery outputs stay outside git.

Initial clean-source screening is complete and tracked in:

```text
experiments/clean_screening/causal_footprint_v0_round5_taxonomy_expansion60_initial_labels.csv
```

Preliminary screening result:

```text
yes: 10
borderline: 11
no: 39
```

Strict `yes` examples currently concentrate in field-mediated and particle-dispersion mechanisms, with only one fluid-impact and one surface-trace source. Many rejected samples are pure-color/blank generations or static footprint-only scenes where the target/action is missing.

Current next step: review the 10 `yes` and 11 `borderline` rows, then decide whether the next erasure-baseline run should use strict `yes` only or exploratory `yes + borderline`.

## Candidate Pair Fields

- `pair_id`: stable candidate id.
- `target_concept`: source concept to erase.
- `causal_footprint`: downstream visible effect.
- `mechanism_type`: one of the benchmark mechanism categories.
- `temporal_type`: `synchronous`, `delayed`, or `persistent`.
- `exclusivity_score`: 1-5, how strongly the footprint implies the source event.
- `counterfactual_clarity`: 1-5, how clear the no-source counterfactual is.
- `generatability_score`: 1-5, estimated likelihood that CogVideoX-style models can generate a clean valid reference.
- `erasure_targetability`: 1-5, whether the target is easy to name for erasure baselines.
- `status`: `accepted_v0_slice`, `exploratory`, or `rejected`.
- `source_prompt`: controlled clean-source prompt.
- `counterfactual_prompt`: same scene under `do(not C)`.
- `control_prompt`: natural-footprint or alternative-cause control when useful.

## Current Slice Policy

The initial accepted slice is mechanism-balanced:

- fluid impact
- surface trace
- fracture or damage
- elastic deformation
- field mediated
- agent or object response

Rows marked `exploratory` should be generated only after the accepted slice is checked. Rows marked `rejected` document what we intentionally excluded.

Round4 prompt variants follow the same mechanism taxonomy but are recorded separately because they are generation-oriented retries. They should enter final `items.jsonl` only if their clean-source labels are accepted after review.
