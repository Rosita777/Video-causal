# Video Concept Erasure: Causal Footprint Audit

**Research question:** Current video concept erasure methods can remove the visible target concept while leaving its causal footprint: downstream video effects that should require the erased concept as their cause.

**Canonical example:** after erasing "ball" from "a red ball rolls and knocks over wooden blocks", the ball disappears but the blocks still fall.

## Current Direction

The project is now benchmark-first. Before proposing a new erasure method, we are defining **causal footprint leakage** as a distinct failure mode from ordinary target-visible leakage and building a structured benchmark for it.

Current benchmark design spec:

```text
docs/superpowers/specs/2026-06-20-causal-footprint-benchmark-v0-design.md
```

The current spec covers three pieces: causal-pair data construction, hybrid MLLM/human evaluation, and the first implementation targets for `benchmarks/causal_footprint_v0/`.

Current benchmark candidate pool:

```text
benchmarks/causal_footprint_v0/candidate_pairs.tsv
benchmarks/causal_footprint_v0/control_prompts.jsonl
benchmarks/causal_footprint_v0/round4_clean_expansion_prompts.tsv
benchmarks/causal_footprint_v0/round5_taxonomy_expansion_prompts.tsv
prompts/causal_footprint_v0_accepted24.txt
prompts/causal_footprint_v0_valid5.txt
prompts/causal_footprint_v0_round4_clean_expansion48.txt
prompts/causal_footprint_v0_round4_clean_valid9.txt
prompts/causal_footprint_v0_round5_taxonomy_expansion60.txt
benchmarks/causal_footprint_v0/items.jsonl
```

## Recovery Status

This project is now tracked on GitHub and the active stable working copy is:

```text
/home/deepseek_VG/JUNCHI/Video-causal
```

The project has had two filesystem-loss events. The current working tree was restored on 2026-07-01 from the GitHub snapshot plus Codex session logs. The recovered code surface includes the v2 benchmark/evaluation pipeline, ZeroScope interfaces, and Wan interfaces; generated videos, contact sheets, model weights, external baseline checkouts, and interrupted run outputs were not recoverable and must be regenerated.

Detailed status:

```text
docs/recovery_status.md
docs/recovery/codex_sessions_recovery_manifest_20260701.json
```

Current lightweight verification:

```bash
PYTHONNOUSERSITE=1 /home/deepseek_VG/.conda/envs/vcecf/bin/python -m pytest -q
```

Expected result:

```text
79 passed
```

Do not treat this repository as containing the full experiment artifacts. Large generated videos, model weights, adapters, and external baseline checkouts remain intentionally outside git; code, prompts, manifests, metrics, and summaries should be backed up frequently.

## Current Evidence

ZeroScope v2 control-free slice:

- Clean-source gate: 96 / 304 candidate prompts accepted with target visible and causal footprint visible.
- Four-baseline closure is complete: 384 / 384 erasure outputs across `negative_prompt`, `videoeraser`, `t2vunlearning`, and `safree_zeroscope`.
- VLM evaluation uses paired 5-frame contact sheets: clean reference plus erased output. Atomic fields are target visibility, footprint visibility, footprint match, separation clarity, and video quality; final labels are rule-derived from those fields.
- Final merged labels: `experiments/evaluation/zeroscope_v2_clean_valid96_baselines_gpt54_sharded32_20260701/vlm_predictions_merged_retry1.csv`.
- Metrics: `experiments/metrics/zeroscope_v2_clean_valid96_baselines_gpt54_20260701/`.
- Headline result: strict causal-footprint leakage is 46 / 384 overall. By baseline: Negative Prompt 11 / 96, VideoEraser local 21 / 96, T2V proxy 11 / 96, SAFREE-ZeroScope 3 / 96.

Authoritative recovered matrix:

```text
experiments/pilot_week1/cross_round_summary/rounds_1_3_master_matrix.csv
```

Coverage tracker:

```text
experiments/pilot_week1/cross_round_summary/rounds_1_3_required_baseline_coverage.csv
```

Current recovered counts:

- 59 annotated rows.
- 13 clean-source-valid cases.
- 65 required coverage slots.
- 6 missing slots, all round2 car-barrier `t2vunlearning` / `safree_cogvideox` summary rows.

Strict causal-footprint positives:

- Negative Prompt: `pitcher_seed63`, `ice_cube_seed66`, `ice_cube_seed67`.
- VideoEraser: `pitcher_seed63`.
- T2VUnlearning: none so far.
- SAFREE-CogVideoX: none so far.

Fresh v0 baseline evidence on the current CogVideoX-2B slice:

- `prompts/causal_footprint_v0_valid5.txt` has been run on all four required baselines.
- Manual valid5 baseline labels are tracked in `experiments/baseline_runs/causal_footprint_v0_valid5_all_step20_parallel_summary.csv`.
- Annotated local gallery: `outputs/analysis_contact_sheets/causal_footprint_v0_valid5_baseline_step20/baseline_gallery_annotated.html`.
- Strong usable causal-footprint leakage examples: 9 / 20 outputs.

Fresh clean-source expansion:

- `benchmarks/causal_footprint_v0/round4_clean_expansion_prompts.tsv` contains 48 taxonomy-driven clean-source variants, 8 per mechanism type.
- `experiments/clean_screening/causal_footprint_v0_round4_clean_expansion48_initial_labels.csv` records initial visual labels: 9 `yes`, 11 `borderline`, 28 `no`.
- `prompts/causal_footprint_v0_round4_clean_valid9.txt` has been run on all four required baselines.
- Manual round4-valid9 labels are tracked in `experiments/baseline_runs/causal_footprint_v0_round4_valid9_all_step20_parallel_summary.csv`.
- Annotated local gallery: `outputs/analysis_contact_sheets/causal_footprint_v0_round4_valid9_baseline_step20/baseline_gallery_annotated.html`.
- Conservative round4-valid9 erasure-output counts after re-review: 15 `yes`, 9 `borderline`, 12 `no`.

Round5 taxonomy-balanced expansion pool:

- `benchmarks/causal_footprint_v0/round5_taxonomy_expansion_prompts.tsv` contains 60 physical causal-footprint candidates.
- `prompts/causal_footprint_v0_round5_taxonomy_expansion60.txt` is the direct CogVideoX clean-source prompt file.
- The pool has 6 mechanism types with 10 candidates each: `fluid_impact`, `surface_trace`, `fracture_damage`, `elastic_deformation`, `field_mediated`, and `particle_dispersion`.
- Round5 deliberately avoids the earlier button/remote semantic-response prompts and reduces dependence on water-drop and ball-net examples.
- Round5 CogVideoX-2B clean-source generation is complete: 60 / 60 videos at 49 frames, 720x480, 20 steps, seed 5200.
- Review artifacts are under `outputs/analysis_contact_sheets/causal_footprint_v0_round5_taxonomy_expansion60_step20/clean_gallery.html` and `clean_source_screening.csv`.
- Initial round5 clean-source labels are tracked in `experiments/clean_screening/causal_footprint_v0_round5_taxonomy_expansion60_initial_labels.csv`: 10 `yes`, 11 `borderline`, 39 `no`.
- Exported round5 slices: `prompts/causal_footprint_v0_round5_clean_yes10.txt` for the main run and `prompts/causal_footprint_v0_round5_clean_yes_borderline21.txt` for exploratory backup.
- Round5 strict `yes10` has been run on all four erasure baselines: 40 / 40 erasure videos plus 10 clean references. Review artifacts are under `outputs/analysis_contact_sheets/causal_footprint_v0_round5_yes10_baseline_step20/baseline_gallery.html`, and the pending-label summary is `experiments/baseline_runs/causal_footprint_v0_round5_yes10_all_step20_parallel_summary.csv`.
- Round5 `borderline11` has also been run as an exploratory backup: 44 / 44 erasure videos plus 11 clean references. Review artifacts are under `outputs/analysis_contact_sheets/causal_footprint_v0_round5_borderline11_baseline_step20/baseline_gallery.html`, and the pending-label summary is `experiments/baseline_runs/causal_footprint_v0_round5_borderline11_all_step20_parallel_summary.csv`.

Formal benchmark-v0 artifact:

- `benchmarks/causal_footprint_v0/items.jsonl` merges the current valid5 and round4-valid9 clean-source-gated rows.
- Current size: 14 benchmark items and 56 erasure baseline outputs.
- Metrics are generated from `items.jsonl` under `experiments/metrics/`.
- Conservative current counts: 24 / 56 strict causal-footprint leakage, 12 / 56 borderline causal-footprint cases, 14 / 56 target-leakage failures.
- By baseline strict leakage: Negative Prompt 5 / 14, SAFREE-CogVideoX 5 / 14, T2V proxy 6 / 14, VideoEraser local 8 / 14.
- Evaluator calibration gold, VLM contact-sheet inputs, and schema-smoke files are under `experiments/eval_calibration/`.

Regenerate the formal artifact and tables:

```bash
PYTHONNOUSERSITE=1 /home/deepseek_VG/.conda/envs/vcecf/bin/python scripts/build_benchmark_items.py \
  --source valid5,benchmarks/causal_footprint_v0/export_valid5_manifest.json,experiments/baseline_runs/causal_footprint_v0_valid5_all_step20_parallel_summary.csv \
  --source round4_valid9,benchmarks/causal_footprint_v0/export_round4_clean_valid9_manifest.json,experiments/baseline_runs/causal_footprint_v0_round4_valid9_all_step20_parallel_summary.csv \
  --output benchmarks/causal_footprint_v0/items.jsonl

PYTHONNOUSERSITE=1 /home/deepseek_VG/.conda/envs/vcecf/bin/python scripts/compute_benchmark_metrics.py \
  --items benchmarks/causal_footprint_v0/items.jsonl \
  --output-dir experiments/metrics
```

Regenerate the evaluator calibration gold file and run the oracle-format smoke test:

```bash
PYTHONNOUSERSITE=1 /home/deepseek_VG/.conda/envs/vcecf/bin/python scripts/export_calibration_gold.py \
  --items benchmarks/causal_footprint_v0/items.jsonl \
  --output experiments/eval_calibration/causal_footprint_v0_gold_outputs.csv

PYTHONNOUSERSITE=1 /home/deepseek_VG/.conda/envs/vcecf/bin/python scripts/calibrate_evaluator.py \
  --gold experiments/eval_calibration/causal_footprint_v0_gold_outputs.csv \
  --predictions experiments/eval_calibration/example_predictions.csv \
  --output-dir experiments/eval_calibration
```

Build 5-frame contact-sheet inputs and dry-run VLM request payloads:

```bash
PYTHONNOUSERSITE=1 /home/deepseek_VG/.conda/envs/vcecf/bin/python scripts/build_vlm_eval_inputs.py \
  --gold experiments/eval_calibration/causal_footprint_v0_gold_outputs.csv \
  --sheet-dir experiments/eval_calibration/frame_sheets \
  --output experiments/eval_calibration/vlm_inputs.csv

PYTHONNOUSERSITE=1 /home/deepseek_VG/.conda/envs/vcecf/bin/python scripts/evaluate_with_vlm.py \
  --inputs experiments/eval_calibration/vlm_inputs.csv \
  --output-jsonl experiments/eval_calibration/vlm_payloads_dryrun.jsonl \
  --dry-run
```

`vlm_inputs.csv` now includes clean-reference contact-sheet fields when a `clean_reference.video_path` is available. Current coverage is 36 / 56 rows with reference sheets, all from `round4_valid9`; the older `valid5` rows do not have clean-reference video paths. Use `--require-reference` for scorer runs that should only use the two-image reference/output protocol.

Current real VLM scorer status:

- `openai/gpt-4o` is the preferred primary judge, but the current `https://api.360.cn/v1` default group returned `no available channel` for this model on 2026-06-22.
- `openai/gpt-4o-mini` was run as a fallback smoke on the first 8 rows, but it over-predicted `strict_leakage` for all 8 rows and is not reliable enough as the main judge.
- `qwen/qwen-vl-plus` with the reference-aware atomic protocol was tested as a high-recall fallback. On all 36 reference-backed rows it gets strict leakage F1 0.6087, relaxed leakage F1 0.8364, and strict leakage recall 0.9333, but it still collapses many borderline and other-failure rows into strict leakage. Its raw artifacts were summarized in the experiment log and not retained.
- `anthropic/claude-sonnet-4-6` with the reference-aware atomic protocol has the opposite bias. On all 36 reference-backed rows it predicts all four labels and reaches macro F1 0.3438, but strict leakage recall is only 0.2667. It is useful as a conservative four-class cross-check, not yet as the main judge.
- Tracked scorer artifacts are under `experiments/eval_calibration/gpt4o_mini_sample8*` and `claude_sonnet_4_6_reference_atomic_full*`.

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

## Benchmark Evaluation V1

The current paper-facing evaluation layer is under `experiments/evaluation/`. It converts human gold rows, contact sheets, and optional VLM predictions into a single manifest, a static annotation review page, and metric tables.

Regenerate the v1 manifest with Claude disagreement columns:

```bash
PYTHONNOUSERSITE=1 /home/deepseek_VG/.conda/envs/vcecf/bin/python scripts/build_evaluation_manifest.py \
  --gold experiments/eval_calibration/causal_footprint_v0_gold_outputs.csv \
  --vlm-inputs experiments/eval_calibration/vlm_inputs.csv \
  --prediction claude=experiments/eval_calibration/claude_sonnet_4_6_reference_atomic_full_predictions.csv \
  --output experiments/evaluation/causal_footprint_v1_manifest.csv
```

Build the human review page and queue:

```bash
PYTHONNOUSERSITE=1 /home/deepseek_VG/.conda/envs/vcecf/bin/python scripts/build_annotation_review.py \
  --manifest experiments/evaluation/causal_footprint_v1_manifest.csv \
  --output-dir experiments/evaluation \
  --project-root /home/deepseek_VG/JUNCHI/Video-causal
```

Compute benchmark metrics from the manifest:

```bash
PYTHONNOUSERSITE=1 /home/deepseek_VG/.conda/envs/vcecf/bin/python scripts/compute_evaluation_metrics.py \
  --manifest experiments/evaluation/causal_footprint_v1_manifest.csv \
  --output-dir experiments/evaluation
```

Current v1 headline:

- Total outputs: 56.
- Strict leakage: 24 / 56.
- Borderline: 12 / 56.
- Relaxed leakage: 36 / 56.
- Target leakage: 14 / 56.
- Claude agreement with human labels on the 36 reference-backed rows: 12 / 36.

## Baseline Policy

The required comparison rows are:

- Negative Prompt: prompt-only inference control.
- SAFREE-CogVideoX: training-free / inference-time erasure control.
- VideoEraser: dedicated video erasure baseline.
- T2VUnlearning: finetuning/unlearning baseline.

Weak, collapsed, residual-cause, or target-visible outputs are method outcomes, not reasons to omit a baseline.

## Quick Checks

```bash
cd /home/deepseek_VG/JUNCHI/Video-causal
python -m pytest tests -q
```

Expected lightweight result:

```text
53 passed
```

## CogVideoX Clean Generation

Plan a clean-source CogVideoX-2B generation run without downloading models:

```bash
python scripts/generate_cogvideox_clean.py \
  --prompts prompts/cogvideox_causal_screening.txt \
  --output-dir outputs/cogvideox_clean_smoke \
  --model zai-org/CogVideoX-2b \
  --limit 2 \
  --seed 42 \
  --dry-run
```

Run real generation after installing `torch` and `diffusers` and making the model available locally or via Hugging Face:

```bash
python scripts/generate_cogvideox_clean.py \
  --prompts prompts/cogvideox_causal_screening.txt \
  --output-dir outputs/cogvideox_clean_v0 \
  --model models/CogVideoX-2b \
  --limit 4 \
  --seed 42 \
  --enable-model-cpu-offload \
  --vae-tiling
```

Outputs under `outputs/` and generated `videos/` are ignored by git.

Current local smoke assets, also ignored by git:

```text
models/CogVideoX-2b
outputs/cogvideox_clean_tech_smoke/
outputs/cogvideox_clean_v0_smoke/
outputs/cogvideox_clean_screening_round1_seed200/
```

Observed smoke result on 2026-06-19:

- `ice cube` / cola seed 101: visually usable from contact-sheet screening.
- `ball` / wooden blocks seed 100: not clean-valid; the blocks/effect are absent.

Current clean-source screening result:

```text
experiments/clean_screening/cogvideox_clean_screening_round1_seed200_summary.csv
```

Round1 seed200-205 produced two clean-valid candidates:

- `ice_cube_seed200`
- `stone_seed204`

Current baseline result:

```text
experiments/baseline_runs/negative_prompt_round1_seed200_summary.csv
```

Negative Prompt produced two strict causal-footprint candidates on the current clean-valid sources:

- `ice_cube_seed200`
- `stone_seed204`

Current real suite generation:

```text
experiments/baseline_runs/baseline_suite_round1_seed200_real_gpu_fp32_summary.csv
```

This run generated 12 ignored `.mp4` files under `outputs/baseline_suite_round1_seed200_real_gpu_fp32/`: 6 Negative Prompt videos and 6 SAFREE-CogVideoX videos. Manual review is pending.

## Baseline Suite Interface

For future experiments, use the suite interface first so all required baselines are planned together:

```bash
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
  /home/deepseek_VG/.conda/envs/vcecf/bin/python scripts/run_baseline_suite.py \
  --prompts prompts/cogvideox_clean_screening_round1.txt \
  --output-root outputs/baseline_suite_round1_seed200 \
  --model models/CogVideoX-2b \
  --seed 200 \
  --steps 20 \
  --guidance-scale 6.0 \
  --num-frames 49 \
  --fps 8 \
  --enable-model-cpu-offload \
  --vae-tiling \
  --parallel \
  --dry-run
```

Current local suite status: all four required baselines are runnable through the suite interface. Negative Prompt and SAFREE-CogVideoX use their existing runners; VideoEraser-local and T2VUnlearning-local use paper-faithful CogVideoX reimplementation/proxy paths when complete official training code or checkpoints are unavailable. Use `--parallel` for real suite runs so all ready baselines launch together.


T2VUnlearning uses `scripts/adapters/run_t2vunlearning_cogvideox.py`. The default `--mode local` path (`receler_cogvideox_proxy_v0`) mirrors the public inference contract: if no eraser checkpoint is available, it runs CogVideoX with concept-suppressed prompt embeddings plus target-concept negative guidance and records the Receler-style eraser rank/config in the manifest. Small, 1-step full-size, and 10-step full-size bf16 real GPU smokes succeeded.

VideoEraser uses `scripts/adapters/run_videoeraser_cogvideox.py`. The default `--mode local` path is a CogVideoX-oriented, training-free reimplementation (`spea_arng_cogvideox_v0`): it constructs an erased prompt by replacing the target concept, uses the target concept as negative guidance, and applies a prompt-embedding displacement away from the original concept-bearing prompt. Small, 1-step full-size, and 10-step full-size bf16 real GPU smokes succeeded; use `--dtype bf16 --enable-model-cpu-offload --vae-tiling` for current full-size runs.

SAFREE-CogVideoX uses `scripts/adapters/run_safree_cogvideox.py`. The wrapper calls the official SAFREE CogVideoX pipeline under `baselines/external/SAFREE/cogvideox/cogvideox_pipeline.py` and injects each prompt row's `target_concept` into SAFREE's concept dictionary as a single concept entry. The external SAFREE checkout is intentionally ignored by git.

For the current SAFREE-CogVideoX adapter, use `--dtype fp32` in real suite runs. The official SAFREE CogVideoX path produced dtype mismatches with `fp16` under the current `torch` / `diffusers` environment. GPU generation also requires running outside the managed filesystem sandbox; inside the sandbox, PyTorch could not see `/dev/nvidia*` even though `nvidia-smi` worked.

## Project Structure

```text
video_concept_erasure_causal_footprint/
├── README.md
├── environment.yml
├── docs/
│   ├── baseline_setup.md
│   ├── current_open_questions.md
│   ├── experiment_log.md
│   ├── recovery_status.md
│   ├── research_notes.md
│   └── superpowers/specs/2026-06-20-causal-footprint-benchmark-v0-design.md
├── benchmarks/causal_footprint_v0/
│   ├── README.md
│   ├── candidate_pairs.tsv
│   ├── control_prompts.jsonl
│   ├── export_accepted24_manifest.json
│   ├── export_valid5_manifest.json
│   ├── export_round4_clean_valid9_manifest.json
│   ├── items.jsonl
│   ├── round4_clean_expansion_prompts.tsv
│   └── round5_taxonomy_expansion_prompts.tsv
├── experiments/pilot_week1/
│   ├── causal_audit_round1/round1_summary.csv
│   ├── causal_audit_round2_car_barrier/round2_summary.csv
│   ├── causal_audit_round3_liquid_surface/round3_summary.csv
│   └── cross_round_summary/
├── experiments/clean_screening/
│   ├── cogvideox_clean_screening_round1_seed200_summary.csv
│   ├── causal_footprint_v0_clean_accepted24_initial_labels.csv
│   └── causal_footprint_v0_round4_clean_expansion48_initial_labels.csv
├── experiments/baseline_runs/
│   ├── negative_prompt_round1_seed200_summary.csv
│   ├── causal_footprint_v0_valid5_all_step20_parallel_summary.csv
│   └── causal_footprint_v0_round4_valid9_all_step20_parallel_summary.csv
├── experiments/metrics/
│   ├── causal_footprint_v0_metrics_by_baseline.csv
│   ├── causal_footprint_v0_metrics_by_mechanism.csv
│   └── causal_footprint_v0_metrics_summary.md
├── experiments/eval_calibration/
│   ├── causal_footprint_v0_gold_outputs.csv
│   ├── example_predictions.csv
│   ├── calibration_metrics_by_label.csv
│   ├── calibration_confusion_matrix.csv
│   ├── calibration_metrics_summary.md
│   ├── vlm_inputs.csv
│   ├── vlm_payloads_dryrun.jsonl
│   ├── gpt4o_mini_sample8_predictions.csv
│   ├── gpt4o_mini_sample8_raw.jsonl
│   ├── gpt4o_mini_sample8/
│   ├── claude_sonnet_4_6_reference_atomic_full_predictions.csv
│   ├── claude_sonnet_4_6_reference_atomic_full_raw.jsonl
│   └── claude_sonnet_4_6_reference_atomic_full/
├── experiments/evaluation/
│   ├── causal_footprint_v1_manifest.csv
│   ├── annotation_queue.csv
│   ├── review.html
│   ├── metrics_by_baseline.csv
│   ├── metrics_by_mechanism.csv
│   ├── model_agreement.csv
│   └── metrics_summary.md
├── prompts/
│   ├── causal_footprint_v0_accepted24.txt
│   ├── causal_footprint_v0_valid5.txt
│   ├── causal_footprint_v0_round4_clean_expansion48.txt
│   ├── causal_footprint_v0_round4_clean_valid9.txt
│   ├── causal_footprint_v0_round5_taxonomy_expansion60.txt
│   ├── causal_pilot.txt
│   ├── cogvideox_causal_screening.txt
│   ├── cogvideox_clean_screening_round1.txt
│   └── cogvideox_clean_smoke.txt
├── scripts/
│   ├── build_baseline_comparison.py
│   ├── build_annotation_review.py
│   ├── build_benchmark_items.py
│   ├── build_clean_source_review.py
│   ├── build_evaluation_manifest.py
│   ├── build_vlm_eval_inputs.py
│   ├── calibrate_evaluator.py
│   ├── check_baselines.py
│   ├── compute_benchmark_metrics.py
│   ├── compute_evaluation_metrics.py
│   ├── evaluate_with_vlm.py
│   ├── export_calibration_gold.py
│   ├── generate_cogvideox_clean.py
│   ├── adapters/run_safree_cogvideox.py
│   ├── adapters/run_videoeraser_cogvideox.py
│   ├── adapters/run_t2vunlearning_cogvideox.py
│   ├── run_baseline_suite.py
│   ├── run_parallel_baseline_jobs.py
│   ├── export_benchmark_prompts.py
│   └── run_pilot.py
└── tests/
```

## Next Actions

1. Run all four baselines on `prompts/causal_footprint_v0_round5_clean_yes10.txt`.
2. Build the round5 yes10 baseline gallery and annotate causal-footprint leakage.
3. Use the `yes + borderline` slice only as exploratory backup if the strict slice is too small.
4. Use Claude/Qwen disagreement as a triage signal, not as ground truth.
5. Add no-source and alternative-cause controls to separate real causal footprints from generic visual priors.
