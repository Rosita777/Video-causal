# Protocol v1 CogVideoX manual review

Status: complete single-reviewer pass over all 80 Original outputs and all 90 baseline outputs whose Original passed the strict capability gate. A second reviewer is still required before treating these numbers as paper-final annotations.

## Original capability gate

A row passes only when the contact sheet shows:

1. the intended source object;
2. the expected downstream footprint;
3. source/contact/footprint evidence in a valid temporal order;
4. the intended receiver and a judgeable video.

| Mechanism | Pass | Total |
| --- | ---: | ---: |
| Water impact | 14 | 20 |
| Rigid collision | 0 | 20 |
| Brittle fracture | 11 | 20 |
| Powder impact | 5 | 20 |
| **All** | **30** | **80** |

Rigid collision is not usable for conditional erasure evaluation on this CogVideoX batch: source objects often appear, but blocks, pins, cans, or cups do not visibly tip or move. These rows remain part of the unconditional capability report and are not counted as successful erasure.

The exact labels and failure notes are stored in `experiments/protocol_v1/cogvideox_original_gate_manual_2026-08-10.csv`.

## Conditional baseline review

The three baselines were reviewed on the same 30 Original-gate-positive prompts. Strict success requires source absence, footprint absence, receiver preservation, and acceptable quality.

| Baseline | Source absent | Footprint absent | Receiver preserved | Strict success |
| --- | ---: | ---: | ---: | ---: |
| Negative Prompt | 2 / 30 | 1 / 30 | 30 / 30 | 0 / 30 |
| T2VUnlearning-adapted, checkpoint 100 | 0 / 30 | 0 / 30 | 30 / 30 | 0 / 30 |
| VideoEraser official | 0 / 30 | 5 / 30 | 28 / 30 | 0 / 30 |

Mechanism-level atomic counts:

| Baseline | Mechanism | Gate-positive count | Source absent | Footprint absent |
| --- | --- | ---: | ---: | ---: |
| Negative Prompt | Water impact | 14 | 2 | 0 |
| Negative Prompt | Brittle fracture | 11 | 0 | 0 |
| Negative Prompt | Powder impact | 5 | 0 | 1 |
| T2VUnlearning-adapted | Water impact | 14 | 0 | 0 |
| T2VUnlearning-adapted | Brittle fracture | 11 | 0 | 0 |
| T2VUnlearning-adapted | Powder impact | 5 | 0 | 0 |
| VideoEraser official | Water impact | 14 | 0 | 4 |
| VideoEraser official | Brittle fracture | 11 | 0 | 0 |
| VideoEraser official | Powder impact | 5 | 0 | 1 |

The exact per-output labels are stored in `experiments/protocol_v1/cogvideox_baseline_manual_gate_positive_2026-08-10.csv`.

## Interpretation

The zero strict-success result does not mean every baseline output is identical to Original. It means their partial changes do not satisfy the full task definition.

- Negative Prompt occasionally removes the visible source while leaving the splash or ripples.
- VideoEraser occasionally suppresses a footprint while leaving the source object visible; two such water outputs also fail receiver preservation.
- The non-collapsed T2VUnlearning-adapted checkpoint usually retains both the source and footprint. The step-500 checkpoint is excluded from semantic success because it collapses toward near-black output.

This decomposition is the central evaluation requirement: source removal alone and footprint suppression alone cannot be reported as complete causal erasure.

## Limitations and next decision

- The labels are from one review pass over seven-frame contact sheets and need independent verification.
- Contact sheets can miss short events between sampled frames; all apparent successes and ambiguous cases should be checked in the full videos.
- CogVideoX rigid collision should either be replaced by a mechanism that the backbone reliably generates, or reported only as a base-capability failure. It should not be silently filtered after method outputs are observed.
- These numbers compare baselines only. The proposed method still needs a CogVideoX implementation before a same-backbone main comparison is possible.
