# Recovery Status

Updated: 2026-07-01

## 2026-07-01 Deletion And Recovery

`/home/deepseek_VG/JUNCHI/Video-causal` disappeared again during a cleanup run outside this project. The clearest local evidence is:

```text
/home/deepseek_VG/phaseA_fluxmem_sota_gate/audits/cleanup_and_datadir_probe_20260701_021335.log
```

That log lists `/home/deepseek_VG/JUNCHI/Video-causal` as a cleanup candidate and records a delete phase for it at `2026-07-01 02:13:35 +08:00` on `worker18` under user `deepseek_VG`. We should treat this as an external cleanup collision rather than a normal project operation.

The repository was restored in two layers:

- GitHub snapshot from `Rosita777/Video-causal`, pushed on 2026-06-23. This restored the early project structure, v0 docs, prompts, scripts, and lightweight historical results.
- Codex session-log extraction from `/home/deepseek_VG/.codex/sessions`, which recovered 110 later files including v2 benchmark/evaluation scripts, ZeroScope adapters, Wan adapters, and their tests.

The extracted recovery manifest is tracked at:

```text
docs/recovery/codex_sessions_recovery_manifest_20260701.json
```

Current verification after restoring and reconciling the recovered files:

```bash
cd /home/deepseek_VG/JUNCHI/Video-causal
PYTHONNOUSERSITE=1 /home/deepseek_VG/.conda/envs/vcecf/bin/python -m pytest -q
```

Result:

```text
79 passed
```

Recovered current-code surface:

- CogVideoX clean generation and baseline suite scripts.
- v2 candidate construction, clean-source chunk VLM evaluation, v2 baseline VLM evaluation, metric computation, paper asset builders.
- ZeroScope clean/baseline generation interfaces and parallel launchers.
- Wan clean/baseline generation interfaces, parallel launcher, and wait-then-evaluate helper.

Post-recovery robustness updates now covered by tests:

- Clean-source review accepts both JSON and JSONL metadata manifests.
- Chunked VLM clean-source evaluation accepts single-item enum lists returned by VLMs.
- VLM API calls support retries and resumable shard runs, so interrupted or slow clean-gate jobs can continue without repeating completed chunk requests.

Not recovered from the 2026-07-01 deletion:

- Generated `.mp4` videos, contact sheets, and partial Wan outputs.
- Model weights and external baseline checkouts.
- Live experiment process state. The Wan baseline run that was in progress must be restarted.
- The latest Wan/ZeroScope result artifacts that existed only in `outputs/` or `experiments/` and were not present in GitHub/session patches.

Important pre-deletion scientific results preserved in conversation notes, but not fully backed by local artifact files after deletion:

- CogVideoX-2B: 100 clean-valid prompts, 400 baseline outputs, strict causal-footprint leakage `87/400 = 21.75%`.
- ZeroScope: 163 clean-valid prompts, 652 baseline outputs, strict causal-footprint leakage `116/652 = 17.79%`.
- CogVideoX-5B: diagnostic only; 59 clean-valid prompts, excluded from main plan because the clean-valid yield was too low and the model was slow/large.
- Wan2.1-T2V-1.3B: 335 clean-valid prompts before deletion; baseline generation was in progress and needs to be rerun.

Next operational step is to re-download or verify required model weights, then rerun Wan baseline generation/evaluation from the restored scripts. Large outputs should remain out of git; code, prompts, manifests, metrics, and summaries should be pushed regularly.

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
