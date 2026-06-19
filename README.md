# Video Concept Erasure: Causal Footprint Audit

**Research question:** Current video concept erasure methods can remove the visible target concept while leaving its causal footprint: downstream video effects that should require the erased concept as their cause.

**Canonical example:** after erasing "ball" from "a red ball rolls and knocks over wooden blocks", the ball disappears but the blocks still fall.

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
10 passed
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
│   └── research_notes.md
├── experiments/pilot_week1/
│   ├── causal_audit_round1/round1_summary.csv
│   ├── causal_audit_round2_car_barrier/round2_summary.csv
│   ├── causal_audit_round3_liquid_surface/round3_summary.csv
│   └── cross_round_summary/
├── prompts/
│   └── causal_pilot.txt
├── scripts/
│   ├── build_baseline_comparison.py
│   ├── build_clean_source_review.py
│   ├── check_baselines.py
│   ├── generate_cogvideox_clean.py
│   └── run_pilot.py
└── tests/
```

## Next Actions

1. Install or restore CogVideoX-2B runtime dependencies and weights outside git.
2. Run clean-source CogVideoX generation and screen for high-quality causal chains.
3. Rebuild baseline runners for Negative Prompt, SAFREE-CogVideoX, VideoEraser, and T2VUnlearning.
4. Fill the six round2 car-barrier `T2VUnlearning` / `SAFREE-CogVideoX` coverage gaps.
5. Continue the causal-footprint audit from the recovered cross-round matrix.
