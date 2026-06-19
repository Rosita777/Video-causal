# Baseline Setup and Reproduction

Updated: 2026-06-19

This is a recovered copy. It records what should be present and what must be regenerated before running heavy baselines again.

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

## Immediate Reproduction Order After Recovery

1. Ask for filesystem snapshot restoration of `/home/deepseek_VG/JUNCHI` before regenerating large artifacts.
2. If no snapshot exists, recreate `models/CogVideoX-2b` and clone/import external baselines into this recovered project.
3. Re-run lightweight tests.
4. Fill the six round2 car-barrier `T2VUnlearning` / `SAFREE-CogVideoX` missing rows.
5. Rebuild review contact sheets only after videos exist again.
