# Baseline Setup and Reproduction

Updated: 2026-06-20

This is the current stable project copy after recovery. It records what should be present and what must be regenerated before running heavy baselines again.

Active working path:

```text
/home/deepseek_VG/JUNCHI/Video-causal
```

## Current Runtime

The conda environment `vcecf` may still exist outside the lost project directory:

```text
/home/deepseek_VG/.conda/envs/vcecf
```

Use it for lightweight checks when available:

```bash
PYTHONNOUSERSITE=1 /home/deepseek_VG/.conda/envs/vcecf/bin/python -m pytest -q
```

If unavailable, the recovered lightweight tests also pass with the system Python used by this session.

Current CogVideoX smoke runtime as of 2026-06-19:

```text
python 3.10.20
torch 2.6.0+cu124
diffusers 0.34.0
transformers 4.51.3
accelerate 1.6.0
safetensors 0.8.0
huggingface_hub 0.36.2
tokenizers 0.21.4
```

Notes:
- `transformers 4.51.3` requires `tokenizers>=0.21,<0.22`; `tokenizers` was downgraded from `0.22.2` to `0.21.4`.
- PyTorch only saw CUDA reliably when running with `CUDA_VISIBLE_DEVICES=0`.
- Direct `huggingface.co` access timed out, but `HF_ENDPOINT=https://hf-mirror.com` worked for downloading `zai-org/CogVideoX-2b`.

## Recovered Locally

- `scripts/run_pilot.py`: dry-run manifest driver.
- `scripts/build_baseline_comparison.py`: contact-sheet/annotation CSV helper.
- `scripts/build_clean_source_review.py`: clean-source screening CSV helper.
- `scripts/check_baselines.py`: source/package readiness checker.
- `scripts/generate_cogvideox_clean.py`: CogVideoX-2B generation runner for clean and Negative Prompt videos with a dependency-free dry-run mode.
- `scripts/adapters/run_safree_cogvideox.py`: SAFREE-CogVideoX wrapper with dependency-free dry-run mode and official-pipeline real-run mode.
- `scripts/run_baseline_suite.py`: suite-level baseline interface that plans/runs all required baselines for the same prompt/seed set.
- Rounds 1-3 summary CSVs and cross-round evidence tables.

## Not Recovered

The following were present before the workspace loss but are not present in this recovery copy:

```text
models/zeroscope_v2_576w
baselines/external/VideoEraser
baselines/external/T2VUnlearning
T2VUnlearning-main.zip
round1/round2/round3 generated videos
contact sheets
T2VUnlearning local adapters
```

Current local ignored assets:

```text
models/CogVideoX-2b
baselines/external/SAFREE/cogvideox/cogvideox_pipeline.py
```

## Required Baselines

The final matrix must include:

| Baseline | Current recovered state | Action |
| --- | --- | --- |
| Negative Prompt | Summary rows recovered | Regenerate videos if visual artifacts are needed |
| VideoEraser | Summary rows recovered | Reclone repo and regenerate videos if no snapshot exists |
| T2VUnlearning | Summary rows recovered for rounds 1 and 3 | Recover source/adapters or retrain; fill round2 gaps |
| SAFREE-CogVideoX | Summary rows recovered for rounds 1 and 3; adapter restored locally | Run on current clean-valid cases; fill round2 gaps |
| CLEAR | No public code in recovered state | Cite as related work until code appears |

## Unified Baseline Suite Interface

Use `scripts/run_baseline_suite.py` as the entry point for a given clean-valid prompt/seed set. The suite always plans the same required baselines, so experiments do not silently omit difficult or slow methods:

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
  --dtype fp32 \
  --enable-model-cpu-offload \
  --vae-tiling \
  --parallel \
  --dry-run
```

Current dry-run statuses:

| Baseline | Suite status | Meaning |
| --- | --- | --- |
| Negative Prompt | `ready` | Runs through `scripts/generate_cogvideox_clean.py --baseline negative_prompt` |
| SAFREE-CogVideoX | `ready` locally; `blocked_missing_external` on fresh clones | Runs through `scripts/adapters/run_safree_cogvideox.py` when the official SAFREE CogVideoX pipeline exists under `baselines/external/SAFREE` |
| VideoEraser | `blocked_missing_adapter` | Need external VideoEraser repo plus CogVideoX adapter wrapper |
| T2VUnlearning | `blocked_missing_adapter` | Need external T2VUnlearning repo plus training/adaptation config |

When adapters are restored or implemented, the suite status should change from blocked to ready without changing the high-level experiment command. For real runs, pass `--parallel` so all ready baselines start together; slower methods such as T2VUnlearning can finish later without forcing the entire interface to be serial.

## SAFREE-CogVideoX Adapter

The local wrapper expects the official SAFREE CogVideoX pipeline at:

```text
baselines/external/SAFREE/cogvideox/cogvideox_pipeline.py
```

The file was fetched locally from the official SAFREE repository:

```text
https://github.com/jaehong31/SAFREE
```

The external checkout is ignored by git. On a fresh clone, download or clone the SAFREE repository into `baselines/external/SAFREE` before running real SAFREE jobs.

Dry-run the adapter:

```bash
PYTHONNOUSERSITE=1 \
  /home/deepseek_VG/.conda/envs/vcecf/bin/python scripts/adapters/run_safree_cogvideox.py \
  --prompts prompts/cogvideox_clean_screening_round1.txt \
  --output-dir outputs/safree_cogvideox_round1_seed200_dryrun \
  --model models/CogVideoX-2b \
  --seed 200 \
  --steps 20 \
  --guidance-scale 6.0 \
  --num-frames 49 \
  --fps 8 \
  --enable-model-cpu-offload \
  --vae-tiling \
  --dry-run
```

For real runs, remove `--dry-run`. The wrapper injects each row's `target_concept` into SAFREE's `CONCEPT_DICT` as `[target_concept]`, then passes that key as the `concept` argument to the official pipeline.

The 2026-06-20 real suite run used `--dtype fp32`. Under the current `torch 2.6.0+cu124` / `diffusers 0.34.0` environment, SAFREE-CogVideoX failed with a `Float` vs `Half` time-embedding dtype mismatch when run as `fp16`. Also run real generation outside the managed sandbox, because sandboxed PyTorch could not see CUDA devices even when `nvidia-smi` worked.

## CogVideoX-2B Clean Source Runner

Use the runner in dry-run mode first. This validates prompts, output naming, generation parameters, and manifest structure without importing `torch` or `diffusers`:

```bash
python scripts/generate_cogvideox_clean.py \
  --prompts prompts/cogvideox_causal_screening.txt \
  --output-dir outputs/cogvideox_clean_smoke \
  --model zai-org/CogVideoX-2b \
  --limit 2 \
  --seed 42 \
  --dry-run
```

After heavy dependencies and weights are available, remove `--dry-run` and point `--model` to the local model directory or a Hugging Face model ID:

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

The manifest is written to `generation_manifest.json`. Videos are written under the selected output directory's `videos/` subdirectory. These generated artifacts are intentionally ignored by git.

For Negative Prompt, add `--baseline negative_prompt`. The runner keeps the original positive prompt and passes each row's `target_concept` as the pipeline `negative_prompt`.

The first local real-generation smoke used:

```bash
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
  /home/deepseek_VG/.conda/envs/vcecf/bin/python scripts/generate_cogvideox_clean.py \
  --prompts prompts/cogvideox_clean_smoke.txt \
  --output-dir outputs/cogvideox_clean_v0_smoke \
  --model models/CogVideoX-2b \
  --limit 2 \
  --seed 100 \
  --steps 20 \
  --guidance-scale 6.0 \
  --num-frames 49 \
  --fps 8 \
  --enable-model-cpu-offload \
  --vae-tiling
```

Initial contact-sheet screening:
- `ice cube` / cola seed 101: visually usable clean source.
- `ball` / wooden blocks seed 100: not clean-valid; the blocks/effect are absent.

## Immediate Reproduction Order After Recovery

1. Expand clean-source CogVideoX screening with more seeds/templates until each target has clean-valid source videos.
2. Preserve `models/CogVideoX-2b` and generated videos locally, outside git.
3. Reclone/import external VideoEraser and T2VUnlearning into `baselines/external/` outside git.
4. Review `outputs/baseline_suite_round1_seed200_real_gpu_fp32/review/` for clean / Negative Prompt / SAFREE-CogVideoX comparisons on the clean-valid cases.
5. Fill the six round2 car-barrier `T2VUnlearning` / `SAFREE-CogVideoX` missing rows.
6. Rebuild review contact sheets only after videos exist again.
