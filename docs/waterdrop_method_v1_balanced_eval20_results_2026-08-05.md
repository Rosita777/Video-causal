# Waterdrop Method v1 Balanced Eval20

Date: 2026-08-05

## Setup

- Model: Wan2.1-T2V-1.3B-Diffusers.
- Training manifest: `data/waterdrop_train_pilot40_sft_v0.csv` (28 rows: 14 erase and 14 preserve).
- Adapter: dual-trajectory objective, checkpoint `checkpoint-000100`, LoRA scale 0.75.
- Evaluation: 20 fixed prompts from `data/waterdrop_dual_traj_eval20.csv`, seed 9100 for every prompt.
- Frozen-base videos were generated previously with the identical prompt and seed list.

## Automatic proxy metrics

| Method | Samples | Mean footprint suppression | Mean early-frame MAE |
| --- | ---: | ---: | ---: |
| Frozen base | 20 | 0.00% | 0.0000 |
| Method v1 balanced dual trajectory | 20 | 32.68% | 0.0305 |

Breakdown by condition:

- Explicit causal: liquid 3.44%, hard 33.52%.
- Implicit causal: liquid 40.41%, hard 53.36%.

The suppression value is only a temporal pixel-change proxy. It is not a semantic proof that the object and its causal footprint were erased.

## Human check

The contact sheets in `experiments/waterdrop_method_v1_balanced_eval20_sheets/` show that the adapter still produces visible droplets and ripples in representative liquid and hard-surface cases. The adapter changes the trajectory, but does not yet remove the footprint cleanly enough for the main claim.

## Decision

This run is a valid diagnostic checkpoint, not a final result. The earlier erase-only run is discarded because it did not include preserve rows and visibly damaged the receiver. Before comparing all five mechanisms, we should improve the waterdrop training recipe and rerun this same eval20.
