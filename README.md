# Video Causal Erasure

This repository studies a narrow failure of text-to-video concept erasure:
removing an object should also remove the downstream visual effect it caused,
while leaving the receiver, camera, and unrelated scene content usable.

## Current Status

The active experiment is **water impact** on **Wan 2.1 T2V 1.3B**:

`object enters water -> splash and expanding ripples`

The retained control is a training-based, preservation-balanced dynamic SFT
LoRA. It is trained from 178 accepted counterfactual targets plus 36 generic
preservation videos. The reported checkpoint is step 200, inferred at LoRA
scale 1.25. This is a working research prototype, not a finished method or a
universal adapter.

The first controlled eval has 12 held-out prompts covering unseen source
objects, unseen receivers, and both unseen. All methods use the same prompts,
seeds, Wan backbone, resolution, frame count, and inference steps.

The latest development result is the frozen v3b-versus-v3c comparison on a
fresh 24-case split, documented in
[`docs/water_impact_dynamic_v3c_fresh_dev24_results_2026-08-16.md`](docs/water_impact_dynamic_v3c_fresh_dev24_results_2026-08-16.md).
V3c preserved quality but failed the preregistered complete-deletion gate, so
it was not promoted and the sealed 36-case final set remains unopened. The
paper main experiment has not started.

The important metrics are valid target and footprint suppression, computed
only on outputs whose receiver and video quality remain usable. Apparent
suppression from a collapsed video is not counted as success.

## Start Here

Read [`docs/PROJECT_HANDOFF.md`](docs/PROJECT_HANDOFF.md) first. It is the
single handoff document and explains the current data, training, evaluation,
baselines, known failures, and exact commands.

## Current Commands

Commands assume the project is on the A100 machine at
`/data/xiaohuang_workspace/ljc/Video-causal`, with Wan weights available under
`models/Wan2.1-T2V-1.3B-Diffusers` and the environment at
`models/.wan-runtime/bin/python`.

Build the dynamic pair manifests:

```bash
models/.wan-runtime/bin/python scripts/build_water_impact_dynamic_pairs_v1.py
```

Train the current adapter:

```bash
bash scripts/run_water_impact_dynamic_sft_preserve_v2.sh
```

Run the current 12-sample method evaluation:

```bash
bash scripts/run_water_impact_dynamic_eval12.sh
```

Run the matched Wan baselines one at a time to avoid GPU memory contention:

```bash
bash scripts/run_water_impact_dynamic_eval12_baselines.sh negative_prompt
bash scripts/run_water_impact_dynamic_eval12_baselines.sh videoeraser
bash scripts/run_water_impact_dynamic_eval12_baselines.sh t2vunlearning
```

## Repository Map

- `scripts/`: generation, training, evaluation, and baseline entry points.
- `data/water_impact_dynamic_v1/`: current water-impact manifests.
- `prompts/water_impact_dynamic_v1/`: current train/eval prompt files.
- `docs/PROJECT_HANDOFF.md`: current project handoff.
- `docs/*water_impact_dynamic*`: current experiment documentation.
- `docs/experiment_log.md` and older `waterdrop_*` documents: historical
  evidence; they are not the current training or evaluation protocol.
- `outputs/`, model weights, videos, and checkpoints: local/remote artifacts,
  intentionally not versioned.

## Important Caveats

The current counterfactual targets are separately generated videos, so they are
not pixel-aligned with factual videos. The current adapter weakens causal
footprints more reliably than it removes the source object. The Wan baseline
proxies for T2VUnlearning and VideoEraser can collapse the scene; therefore
their apparent erasure must always be reported with receiver preservation and
video quality.

Keep code, manifests, scores, summaries, and documentation in GitHub. Do not
commit model weights, generated videos, or adapter checkpoints.
