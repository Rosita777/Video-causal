# Baseline Setup and Reproduction

Updated: 2026-06-19

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

## Recovered Locally

- `scripts/run_pilot.py`: dry-run manifest driver.
- `scripts/build_baseline_comparison.py`: contact-sheet/annotation CSV helper.
- `scripts/build_clean_source_review.py`: clean-source screening CSV helper.
- `scripts/check_baselines.py`: source/package readiness checker.
- `scripts/generate_cogvideox_clean.py`: clean-source CogVideoX-2B generation runner with a dependency-free dry-run mode.
- Rounds 1-3 summary CSVs and cross-round evidence tables.

## Not Recovered

The following were present before the workspace loss but are not present in this recovery copy:

```text
models/CogVideoX-2b
models/zeroscope_v2_576w
baselines/external/VideoEraser
baselines/external/T2VUnlearning
T2VUnlearning-main.zip
round1/round2/round3 generated videos
contact sheets
T2VUnlearning local adapters
```

## Required Baselines

The final matrix must include:

| Baseline | Current recovered state | Action |
| --- | --- | --- |
| Negative Prompt | Summary rows recovered | Regenerate videos if visual artifacts are needed |
| VideoEraser | Summary rows recovered | Reclone repo and regenerate videos if no snapshot exists |
| T2VUnlearning | Summary rows recovered for rounds 1 and 3 | Recover source/adapters or retrain; fill round2 gaps |
| SAFREE-CogVideoX | Summary rows recovered for rounds 1 and 3 | Recreate disclosed adaptation; fill round2 gaps |
| CLEAR | No public code in recovered state | Cite as related work until code appears |

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

## Immediate Reproduction Order After Recovery

1. Recreate `models/CogVideoX-2b` or configure access to `zai-org/CogVideoX-2b` outside git.
2. Run the CogVideoX clean-source runner and screen for clean-valid causal chains.
3. Reclone/import external baselines into `baselines/external/` outside git.
4. Fill the six round2 car-barrier `T2VUnlearning` / `SAFREE-CogVideoX` missing rows.
5. Rebuild review contact sheets only after videos exist again.
