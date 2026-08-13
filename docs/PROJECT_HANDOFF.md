# Project Handoff

Last updated: 2026-08-13

This is the authoritative handoff document. When another document conflicts
with this one, treat the other document as historical until explicitly updated.

## 1. Research Question

Given a text prompt describing a causal video event, erase the source object
and the downstream visual footprint caused by it. For the current prototype:

`object enters water -> splash -> expanding ripples`

The receiver, camera, lighting, and unrelated motion should remain usable. The
project is not currently claiming a universal adapter across all mechanisms.

## 2. Current Method

The active method is a Wan LoRA trained with dynamic counterfactual SFT and a
generic preservation branch:

1. Generate factual videos containing the object and water-impact event.
2. Generate target videos describing the same receiver without the object or
   impact footprint.
3. Screen target videos and keep 178 accepted rows.
4. Add 36 generic preservation videos with no water-impact event.
5. Alternate erase and preservation samples during LoRA SFT.

| Field | Value |
|---|---|
| Backbone | Wan 2.1 T2V 1.3B Diffusers |
| Erase rows | 178 |
| Preservation rows | 36 |
| LoRA rank / alpha | 16 / 16 |
| Learning rate | `5e-5` |
| Steps | 200 |
| Checkpoint | `checkpoint-000200` |
| Inference scale | `1.25` |
| Frames / resolution | 49 / 480x832 |
| FPS / diffusion steps | 8 / 25 |

The exact launcher is `scripts/run_water_impact_dynamic_sft_preserve_v2.sh`.
The training manifest is `data/water_impact_dynamic_v1/train_dynamic_sft_preserve_v2.csv`.

## 3. Data and Splits

The pair-construction protocol is documented in
`docs/water_impact_dynamic_counterfactual_v1.md`.

Current training data:

- 8 source objects and 12 receivers in the accepted training set;
- 192 generated target videos, 14 rejected, 178 retained;
- direct and natural wording variants;
- generic preservation rows from `data/protocol_v1/wan_train_manifest.csv`.

Current held-out eval12:

- 4 unseen-source cases;
- 4 unseen-receiver cases;
- 4 cases with both source and receiver unseen;
- one fixed seed per row, recorded in `data/water_impact_dynamic_v1/eval12.csv`.

Do not silently regenerate or replace a row. If a row changes, create a new
manifest version and record why.

## 4. Evaluation

The main comparison contains Original, Negative Prompt, the local Wan
T2VUnlearning proxy, the local Wan VideoEraser proxy, and the current adapter.
All use the same prompts, seeds, and Wan generation settings.

The review sheet samples seven frames per video. Atomic fields are:

- target visibility: `0 absent, 1 partial/weaker, 2 clear`;
- footprint visibility: `0 absent, 1 partial/weaker, 2 clear`;
- receiver preservation: `0 bad, 1 partial, 2 good`;
- video quality: `0 bad, 1 partial, 2 good`.

Keep two kinds of suppression separate:

- **Apparent suppression**: target/footprint is not visible, even if the scene
  collapsed.
- **Valid suppression**: computed only for outputs with receiver preservation
  and quality at least 1. A broken black/noise output is not success.

The preliminary summary is in
`experiments/water_impact_dynamic_eval12/manual_summary_v1.csv`; interpretation
is in `docs/water_impact_dynamic_eval12_results_2026-08-13.md`.

| Method | Valid footprint suppression | Valid outputs |
|---|---:|---:|
| Original | 0.0% | 12/12 |
| Negative Prompt | 8.3% | 12/12 |
| T2VUnlearning | N/A | 0/12 |
| VideoEraser | N/A | 0/12 |
| Ours v2 | 36.4% | 11/12 |

These are first manual scores on 12 samples, not final paper statistics. A
larger evaluation should use the same rubric with a second reviewer or a
calibrated VLM plus spot-checks.

## 5. Exact Reproduction Path

On the A100 machine:

```bash
cd /data/xiaohuang_workspace/ljc/Video-causal
PYTHON=models/.wan-runtime/bin/python

# Check the current manifest and create/refresh target prompt files.
$PYTHON scripts/build_water_impact_dynamic_pairs_v1.py

# Train v2. Existing latent caches make reruns cheaper.
bash scripts/run_water_impact_dynamic_sft_preserve_v2.sh

# Generate the current adapter and matched Original.
bash scripts/run_water_impact_dynamic_eval12.sh

# Generate baselines one at a time. Do not put two Wan pipelines on one GPU.
bash scripts/run_water_impact_dynamic_eval12_baselines.sh negative_prompt
bash scripts/run_water_impact_dynamic_eval12_baselines.sh videoeraser
bash scripts/run_water_impact_dynamic_eval12_baselines.sh t2vunlearning

# Build the blank review table after all videos exist.
$PYTHON scripts/build_water_impact_dynamic_eval12_review.py \
  --output experiments/water_impact_dynamic_eval12/review.csv
```

Weights and generated videos are not in Git. Check them before starting. The
baseline runners are local Wan proxy implementations, not claims of official
released baseline training code.

## 6. Known Problems

1. Source-object removal is weak compared with footprint suppression.
2. Separately generated counterfactual targets are not pixel-aligned with the
   factual videos, contributing to receiver drift.
3. T2VUnlearning and VideoEraser proxy outputs collapse under this Wan setup;
   report them honestly as unusable, not successful erasure.
4. Never run two full Wan pipelines on one 80GB GPU. The adapter baseline
   helpers now encode prompts under inference mode and move embeddings to CPU,
   but one pipeline per GPU remains the safe default.
5. CogVideoX and Hunyuan probes are historical capability checks, not part of
   the current main experiment.

## 7. Next Research Task

Improve source-object deletion without losing current preservation behavior.
The conservative next experiment is a v3 loss ablation using the same data and
eval12, followed by a larger held-out set. Do not expand to five mechanisms or
add another backbone until the water-impact pipeline has a stable,
independently reviewed score table.

## 8. Git Policy

Commit and push source code, manifests, metrics, score tables, and documents
frequently. Do not commit model weights, generated videos, caches, or LoRA
checkpoints. Before handoff, run `git status` and mention intentional
untracked local artifacts.
