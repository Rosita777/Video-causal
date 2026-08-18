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
dynamic counterfactual SFT and a generic preservation branch. V3b and v3c are
archived negative development ablations and have **not** replaced this
control. No method has yet passed the gate required to start the paper main
experiment.

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
rows in `data/water_impact_dynamic_v1/test_pairs.csv` were deterministically
partitioned before v3c generation into a stratified fresh-dev24 split (8 per
generalization group, with 4 direct and 4 natural prompts) and sealed-final36.
The split registry SHA-256 is
`4f31a291e8ffca07da4bf057e9a86df72f656c03aab65bc06d4c3c155b72962a`.
V3c failed its fresh-dev24 promotion gate, so final36 remains sealed and has
not been generated, inspected, or scored.

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

V3c kept the v3b teacher weight and mean teacher budget fixed while assigning
the erase-row teacher term weight `4 * (2 * sigma)`. It was trained from the
same initialization and evaluated only on the frozen fresh-dev24 split. Two
independent blinded reviewers inspected all videos in full, and a third
blinded reviewer adjudicated all 26 atomic disagreements.

| Method | Usable | Receiver /48 | Quality /48 | Target suppression /48 | Footprint suppression /48 | Absent target | Strict |
|---|---:|---:|---:|---:|---:|---:|---:|
| V3b | 24/24 | 47 | 32 | 11 | 17 | 4 | 0 |
| V3c sigma-weighted teacher | 24/24 | 47 | 32 | 12 | 18 | 3 | 0 |

V3c preserved usability, receiver, and quality, but gained only one target
point and one footprint point. It produced zero clear-to-absent improvements,
zero strict successes, and one fewer usable absent-target output. Six
registered checks failed, so
`promote_v3c_and_unseal_final36=false`. The full result and frozen hashes are
in `docs/water_impact_dynamic_v3c_fresh_dev24_results_2026-08-16.md`.

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

The completed v3c development run can be verified in the clean checkout with:

```bash
cd /data/xiaohuang_workspace/ljc/Video-causal-v3
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=scripts \
  models/.wan-runtime/bin/python -m unittest -v \
  tests.test_train_wan_waterdrop_lora_v3c \
  tests.test_water_impact_dynamic_v3c_split \
  tests.test_water_impact_dynamic_v3c_eval
bash scripts/run_water_impact_dynamic_v3c_fresh_dev24.sh preflight
bash scripts/run_water_impact_dynamic_v3c_fresh_dev24.sh stage2-preflight
```

The frozen v3c checkpoint is
`outputs/water_impact_dynamic_v3c/adapter_target_prompt_teacher_sigma2_scale4_v1/checkpoint-000200`.
The fresh-dev24 generations and review media remain server artifacts; the
small manifests, raw blind scores, canonical scores, and gate are committed.
Do not invoke any generation or review action again on these frozen paths.

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
8. V3c's high-noise teacher redistribution preserves scene quality but does
   not materially improve complete source deletion over v3b. Fresh-dev24 is
   now also exhausted for method selection.

## 7. Next Research Task

V3c failed the frozen development gate. The immediate task is therefore **not**
to start the paper main experiment and not to tune v3c. Keep sealed-final36
closed.

The selected v4 hypothesis is **Source-slot Randomized Counterfactual
Distillation**, specified in
`docs/water_impact_dynamic_v4_source_slot_randomization.md`. It keeps the v3b
loss, target latent, target-prompt teacher, preservation branch, teacher
weight, initialization, 100/100 schedule, optimizer, 200-step budget, and
generation settings fixed. The only training treatment is deterministic
replacement of the erase-row factual prompt's causal-source phrase from a
frozen 64-item bank. This directly tests whether the adapter can learn the
causal-source role rather than memorize the eight training nouns.

V4 has an independently approved fail-closed implementation and a formally
frozen `v4_dev72_v2` Stage-0 public registry. The first v1 registry and the
first pre-freeze v2 draft were rejected by physical/source audits; their
aggregate failures and public hashes remain recorded in the final v2 registry.
The final curated new ontology passes 80/80 strict physical checks, with 56
public bank sources and 24 committed private holdout sources. The eight
historical training sources are the only explicit legacy exception and were
still subject to full-video Original screening. The deterministic 178-row
public mapping is frozen at
`data/water_impact_dynamic_v4/source_mapping_v2.json`.

The implementation/public artifacts are committed, the remote runtime and
full-model byte inventories validate, and the causal Stage-0 opening is frozen
at `data/water_impact_dynamic_v4/causal_stage0_commitment_v2.json` (SHA-256
`29696ad8031bb164fe1c6819c8c382d7e4e828835f750f0d245e4877d4167b38`).
The authorized 48-case Original-only screen completed and its review freeze
succeeded, but only 24 candidates were eligible. Two of the six registered
cells had one eligible candidate against the exact quota of four. The selector
therefore failed closed before producing selected24 or U72. The formal outcome
is `preflight_dataset_invalid`, recorded in
`results/water_impact_dynamic_v4_causal_screening_termination_v2.md`.

A fresh follow-on dataset, `v4_dev72_v3`, was subsequently preregistered and
prepared without opening sealed-final36. Static preparation and private
candidate construction completed (576 candidates and 1,728 unique evaluation
seeds), but the mandatory isolated v2/v3 identity audit found two fresh-source
normalized-head intersections. Fresh-source intersections are required to be
zero and are not covered by the historical exceptions. The construct auditor
also stopped on a non-exact cell-quota representation; the forbidden-seed
audit passed.

V3 stopped before the public holdout commitment, cost calibration, pending
Stage-0 freeze, authorizing wrapper, GPU generation, or review. Because the
pending boundary was never crossed, no machine-readable invalid-outcome JSON
was emitted. The scientific version is nevertheless exhausted and must not be
recurated or repaired. The aggregate-only record is
`results/water_impact_dynamic_v4_causal_screening_termination_v3.md`.

There is no causal Stage-1 commitment, specificity dataset, prompt sidecar,
training authorization, v4 training run, checkpoint, treatment generation, or
v4 evaluation. `v4_dev72_v2` must not be retried or repaired by replacing
candidates, prompts, sources, receivers, or seeds. Sealed-final36 remains
unopened.

The registered `v4_dev72_v2` intervention may not now be run. Do not combine
it with anti-guidance, masks, span gates, hard-negative preserve training,
another teacher weight, or another sigma schedule. Both the fresh causal gate
and the same-noun specificity gate must pass completely before any paper-final
use. An invalid or negative v4 result does not authorize tuning on the
inspected set.

Only a method that passes a genuinely fresh development gate may be frozen and
evaluated on sealed-final36 in a multi-seed paper main experiment. Do not
expand to five mechanisms or add another backbone before the water-impact
method reaches that point.

## 8. Git Policy

Commit and push source code, manifests, metrics, score tables, and documents
frequently. Do not commit model weights, generated videos, caches, or LoRA
checkpoints. Before handoff, run `git status` and mention intentional
untracked local artifacts.
