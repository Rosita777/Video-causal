# Water-Drop 30-Scene Manual Review (2026-07-30)

## Scope

Thirty Wan2.1-T2V-1.3B factual videos were reviewed from `outputs/waterdrop_scene_probe30_wan_seed8300_8329/`. Each video was inspected as a 12-frame temporal contact sheet. Labels are strict: an ambiguous sample is not counted as a pass.

The structured per-video labels are in `data/waterdrop_scene_probe30_review.csv`.

## Result

| Family | Pass | Uncertain | Fail |
| --- | ---: | ---: | ---: |
| Liquid surfaces | 3 | 2 | 1 |
| Hard non-absorbent surfaces | 3 | 0 | 3 |
| Absorbent surfaces | 1 | 1 | 4 |
| Granular surfaces | 0 | 0 | 6 |
| Flexible surfaces | 0 | 0 | 6 |
| **Total** | **7** | **3** | **20** |

## Passed Receivers

1. Calm shallow pond (`wd000`)
2. Glass bowl filled with water (`wd001`)
3. Ceramic cup filled with water (`wd002`)
4. Stainless-steel tray (`wd007`)
5. Glass tabletop (`wd008`)
6. Ceramic plate (`wd010`)
7. Unfinished wood (`wd016`)

The metal bucket, shallow puddle, and cotton cloth are uncertain and must not be used as accepted samples without a second review or a regenerated prompt.

## Main Failure Modes

- The footprint already exists before the droplet arrives.
- The receiver is missing, changes identity, or is not visually recognizable.
- The model produces a generic liquid puddle instead of absorption, granular displacement, or deformation.
- The droplet misses the intended receiver or repeatedly reappears.
- Flexible receivers do not bend or rebound.

## Decision

Generate matched counterfactual candidates only for the seven passed receivers. Do not expand prompt paraphrases yet. First verify that each passed factual scene also has a usable counterfactual target with a reasonably compatible receiver and composition.
