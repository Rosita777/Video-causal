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
prompts/causal_footprint_v0_accepted24.txt
prompts/causal_footprint_v0_valid5.txt
prompts/causal_footprint_v0_round4_clean_expansion48.txt
prompts/causal_footprint_v0_round4_clean_valid9.txt
```

## Recovery Status

This project is now tracked on GitHub and the active stable working copy is:

```text
/home/deepseek_VG/JUNCHI/Video-causal
```

It was recovered on 2026-06-19 from an intermediate copy under:

```text
/home/deepseek_VG/deepseek/video_concept_erasure_causal_footprint
```

The real path is:

```text
/dev/shm/deepseek/video_concept_erasure_causal_footprint
```

Source directory lost from the active filesystem:

```text
/home/deepseek_VG/JUNCHI/video_concept_erasure_causal_footprint
```

Recovered from Codex logs and prior summaries:

- Core docs, prompts, lightweight scripts, and lightweight tests.
- Manual annotation summaries for rounds 1-3.
- Cross-round evidence matrices under `experiments/pilot_week1/cross_round_summary/`.

Not recovered here:

- Generated `.mp4` videos and contact sheets.
- Local model weights under `models/`.
- External baseline checkouts under `baselines/external/`.
- T2VUnlearning zip/source unless recovered separately from backup.

Do not treat this repository as containing the full experiment artifacts. It is enough to continue the research state and regenerate missing artifacts. Large generated videos, model weights, adapters, and external baseline checkouts remain intentionally outside git.

## Current Evidence

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
32 passed
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
│   └── round4_clean_expansion_prompts.tsv
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
├── prompts/
│   ├── causal_footprint_v0_accepted24.txt
│   ├── causal_footprint_v0_valid5.txt
│   ├── causal_footprint_v0_round4_clean_expansion48.txt
│   ├── causal_footprint_v0_round4_clean_valid9.txt
│   ├── causal_pilot.txt
│   ├── cogvideox_causal_screening.txt
│   ├── cogvideox_clean_screening_round1.txt
│   └── cogvideox_clean_smoke.txt
├── scripts/
│   ├── build_baseline_comparison.py
│   ├── build_clean_source_review.py
│   ├── check_baselines.py
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

1. Jointly review the round4-valid9 annotated baseline gallery at `outputs/analysis_contact_sheets/causal_footprint_v0_round4_valid9_baseline_step20/baseline_gallery_annotated.html`.
2. Revise any disputed round4-valid9 labels in `experiments/baseline_runs/causal_footprint_v0_round4_valid9_all_step20_parallel_summary.csv`.
3. Merge agreed valid5 and round4 clean-valid rows into `items.jsonl`, then add control prompts for metric calibration.
4. Compute `CFP@TPS<=1` and separate failure modes: target leakage, target-erased causal footprint, and collapsed/low-quality video.
