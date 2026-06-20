# Recovery Status

Updated: 2026-06-20

## What Happened

The original workspace path `/home/deepseek_VG/JUNCHI` disappeared from the active filesystem view. Both Codex and the user's IDE terminal could no longer see it. Bash history did not show an explicit `rm -rf JUNCHI` or `mv JUNCHI` command. Trash did not contain the project. The top-level `/home/deepseek_VG` directory had an mtime near `2026-06-19 04:54`, so an external cleanup, move, mount change, or workspace reset remains the most plausible cause.

## Recovery Target

The project was restored under:

```text
/home/deepseek_VG/deepseek/video_concept_erasure_causal_footprint
```

`/home/deepseek_VG/deepseek` is a symlink to `/dev/shm/deepseek`.

The active stable working copy is now:

```text
/home/deepseek_VG/JUNCHI/Video-causal
```

The stable copy keeps the GitHub remote `https://github.com/Rosita777/Video-causal.git`. Generated videos, model weights, adapter checkpoints, and external baseline checkouts remain outside git; lightweight suite adapters are tracked in git.

## Recovered

- Top-level project docs and prompt files available in Codex logs.
- Lightweight scripts:
  - `scripts/run_pilot.py`
  - `scripts/build_baseline_comparison.py`
  - `scripts/build_clean_source_review.py`
  - `scripts/check_baselines.py`
  - `scripts/generate_cogvideox_clean.py`
  - `scripts/adapters/run_safree_cogvideox.py`
  - `scripts/adapters/run_videoeraser_cogvideox.py`
  - `scripts/adapters/run_t2vunlearning_cogvideox.py`
  - `scripts/run_baseline_suite.py`
- Lightweight tests:
  - `tests/test_run_pilot.py`
  - `tests/test_build_baseline_comparison.py`
  - `tests/test_recovered_evidence.py`
  - `tests/test_check_baselines.py`
  - `tests/test_run_baseline_suite.py`
  - `tests/test_run_safree_cogvideox.py`
  - `tests/test_run_external_adapters.py`
- Recovered manual summary CSVs:
  - `experiments/pilot_week1/causal_audit_round1/round1_summary.csv`
  - `experiments/pilot_week1/causal_audit_round2_car_barrier/round2_summary.csv`
  - `experiments/pilot_week1/causal_audit_round3_liquid_surface/round3_summary.csv`
- Recovered cross-round CSVs:
  - `rounds_1_3_master_matrix.csv`
  - `rounds_1_3_required_baseline_coverage.csv`
  - counts by baseline/template/round.

## Not Recovered

- Generated videos.
- Contact sheets.
- Model weights.
- External baseline repositories.
- Adapter checkpoints.
- Any file content that existed only on disk and was never visible in logs.

## Verification

```bash
cd /home/deepseek_VG/JUNCHI/Video-causal
python -m pytest tests -q
```

Current lightweight result:

```text
24 passed
```

## Current Scientific State

The recovered evidence supports the same narrow claim as before the loss:

- Negative Prompt repeatedly produced strict causal-footprint positives across pitcher-water and ice-cube-cola.
- VideoEraser produced one strict positive on pitcher-water, with no recovered strict positive on round3 bottle/ice.
- SAFREE-CogVideoX is now runnable through the restored wrapper, but has no strict positives in the recovered summaries.
- T2VUnlearning remains a required baseline with a suite adapter; it currently has no strict positives in the recovered summaries and still needs external source/config for new real runs.
- Round2 car-barrier still lacks T2VUnlearning and SAFREE-CogVideoX summary rows.
