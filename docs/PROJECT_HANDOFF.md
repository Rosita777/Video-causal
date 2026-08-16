# Project Handoff

Last updated: 2026-08-16

This is the authoritative handoff document. When another document conflicts
with this one, treat the other document as historical until explicitly updated.

## 1. Research Question

Given a text prompt describing a causal video event, erase the source object
and the downstream visual footprint caused by it. For the current prototype:

`object enters water -> splash -> expanding ripples`

The receiver, camera, lighting, and unrelated motion should remain usable. The
project is not currently claiming a universal adapter across all mechanisms.

## 2. Current Operating Point

The retained operating point is the seeded-balanced Wan LoRA trained with
dynamic counterfactual SFT and a generic preservation branch. V3b is an
archived negative development ablation and has **not** replaced this control.

The data construction is:

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

The original v2 launcher is `scripts/run_water_impact_dynamic_sft_preserve_v2.sh`.
The frozen seeded-balanced control used for the v3 comparisons is
`outputs/water_impact_dynamic_v1/adapter_dynamic_sft_v3_balanced_seeded/checkpoint-000200`.
Both use
`data/water_impact_dynamic_v1/train_dynamic_sft_preserve_v2.csv`; the seeded
control fixes the 200-step sample order at exactly 100 erase and 100 preserve
updates.

## 3. Data and Splits

The pair-construction protocol is documented in
`docs/water_impact_dynamic_counterfactual_v1.md`.

Current training data:

- 8 source objects and 12 receivers in the accepted training set;
- 192 generated target videos, 14 rejected, 178 retained;
- direct and natural wording variants;
- generic preservation rows from `data/protocol_v1/wan_train_manifest.csv`.

The original eval12 (now an exhausted development set) contains:

- 4 unseen-source cases;
- 4 unseen-receiver cases;
- 4 cases with both source and receiver unseen;
- one fixed seed per row, recorded in `data/water_impact_dynamic_v1/eval12.csv`.

Eval12 has now been inspected repeatedly while designing v3a and v3b. Treat it
as an exhausted development set, not as a paper-final test. The remaining 60
rows in `data/water_impact_dynamic_v1/test_pairs.csv` have not been used for
the v3b decision. Before producing any v3c videos, deterministically freeze a
stratified fresh-dev24 split (8 per generalization group, with 4 direct and 4
natural prompts) and a sealed-final36 split. Do not inspect the final36 while
developing v3c.

Do not silently regenerate or replace a row. If a row changes, create a new
manifest version and record why.

## 4. Evaluation

The preliminary five-way comparison contains Original, Negative Prompt, the
local Wan T2VUnlearning proxy, the local Wan VideoEraser proxy, and the current
adapter. All use the same prompts, seeds, and Wan generation settings.

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

The completed v3b development comparison used two independent blinded
reviewers plus blinded adjudication of every disagreement. The treatment adds
a source-free target-prompt frozen teacher to the erase loss with weight 4;
training used the same initialization, data, sample order, and 200-step budget
as the seeded-balanced control.

| Method | Usable | Receiver /24 | Quality /24 | Target suppression /24 | Footprint suppression /24 | Strict |
|---|---:|---:|---:|---:|---:|---:|
| Seeded-balanced control | 11/12 | 21 | 17 | 3 | 3 | 0 |
| V3b target-prompt teacher | 12/12 | 22 | 18 | 7 | 12 | 0 |

V3b passed the usability, preservation, paired target-gain, and footprint
floors, but it produced no registered clear-to-absent target improvement and
no strict success. Therefore `mechanism_positive=false` and
`promote_v3b_operating_point=false`. It is promising mechanistic evidence, not
a promoted method. The full interpretation and frozen hashes are in
`docs/water_impact_dynamic_v3b_eval12_results_2026-08-16.md`.

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

The actively maintained clean checkout for the v3 work is
`/data/xiaohuang_workspace/ljc/Video-causal-v3`. The original checkout at
`/data/xiaohuang_workspace/ljc/Video-causal` is intentionally preserved and
must not be reset or cleaned: it contains a large mixed historical working
tree. In the clean checkout, the completed v3b protocol can be verified with:

```bash
cd /data/xiaohuang_workspace/ljc/Video-causal-v3
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=scripts \
  models/.wan-runtime/bin/python -m unittest -v \
  tests.test_water_impact_dynamic_v3b_eval
bash scripts/run_water_impact_dynamic_v3b_eval12.sh preflight
```

The v3b training checkpoint is
`outputs/water_impact_dynamic_v3b/adapter_target_prompt_teacher_scale4_v1/checkpoint-000200`;
the 12 generated treatment videos are under
`outputs/water_impact_dynamic_v3b/eval12_target_prompt_teacher_scale4_v1_ckpt200_scale1p25`.
Do not rerun or overwrite these frozen paths.

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
6. Eval12 is exhausted for method selection. Reusing it to select another
   teacher weight, noise schedule, or checkpoint would be adaptive tuning.
7. V3b suppresses the causal footprint strongly, but source objects usually
   move only from clear to partial rather than clear to absent. This is the
   immediate failure mode to target.

## 7. Next Research Task

The single recommended next ablation is v3c: keep v3b's data, initialization,
sample order, teacher weight, preservation branch, and 200-step budget fixed,
but allocate the teacher term toward the high-noise part of the diffusion
trajectory:

```text
L_erase = L_flow + 4 * (2 * sigma) * L_target_prompt_teacher
```

Because `E[2*sigma]=1`, this keeps the mean teacher budget at 4; it is a
schedule ablation, not another weight sweep. Train from the same initialization
rather than continuing from v3b. Before any eligible checkpoint, the first 16
erase updates must have finite
`g_i = 8*sigma_i*sqrt(L_teacher/L_flow)`, arithmetic mean in `[0.20, 0.50]`,
and maximum at most `1.0`. A frozen-RNG arithmetic replay predicts mean
`0.2868` and maximum `0.8582`; this is only a pre-run safety check.

Do not evaluate v3c on eval12. Freeze the fresh-dev24/sealed-final36 split
described above before generation, compare only frozen v3b versus v3c on
fresh-dev24 using two blinded reviewers plus adjudication, and preregister the
gate before viewing outputs. The proposed all-or-nothing gate is:

- at least 20 usable v3b controls;
- v3c target-suppression points at least v3b plus 6;
- at least 6 usable paired target improvements, including at least two
  clear-to-absent (`2 -> 0`) cases across at least two generalization groups;
- at least two more absent-target cases than v3b;
- v3c usable at least 22/24;
- receiver at least `max(38, v3b_receiver - 2)` and quality at least
  `max(32, v3b_quality - 2)`;
- no loss in footprint suppression on the v3b-usable set;
- at least two strict `(0, 0, 2, 2)` successes.

Only after v3c passes every condition should final36 be unsealed for a
multi-seed paper main experiment. If it fails, record the negative result and
do not sweep teacher weight, sigma window, or checkpoint.

Do not expand to five mechanisms or add another backbone until this
water-impact method has passed the fresh-development gate.

## 8. Git Policy

Commit and push source code, manifests, metrics, score tables, and documents
frequently. Do not commit model weights, generated videos, caches, or LoRA
checkpoints. Before handoff, run `git status` and mention intentional
untracked local artifacts.
