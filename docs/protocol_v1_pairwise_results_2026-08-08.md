# Protocol v1 Pairwise Results

This is the first automatic comparison between the frozen Protocol v1 Original and Ours Wan outputs. Both methods use the same 80 prompts, seeds, 25 denoising steps, and 49-frame layout.

## Main result

The pairwise temporal proxy measures how much late motion is suppressed by Ours relative to Original:

| Group | Samples | Footprint suppression |
| --- | ---: | ---: |
| All mechanisms | 80 | 82.91% |
| Water impact | 20 | 89.94% |
| Rigid collision | 20 | 91.35% |
| Brittle fracture | 20 | 85.24% |
| Powder impact | 20 | 65.12% |

## Negative Prompt baseline

The same paired proxy gives Negative Prompt -3.02% overall suppression, compared with 82.91% for Ours. By mechanism:

| Mechanism | Negative Prompt | Ours |
| --- | ---: | ---: |
| Water impact | 25.72% | 89.94% |
| Rigid collision | 4.82% | 91.35% |
| Brittle fracture | -27.04% | 85.24% |
| Powder impact | -15.57% | 65.12% |

Negative values mean the baseline produces more late temporal change than Original rather than suppressing it.

For the fully unseen source and receiver split, suppression is 86.28%. The seen/seen split is 76.75%; the other three split groups are 84.10% to 86.28%.

## Interpretation

The first three mechanisms show strong causal-footprint suppression and transfer to unseen source/receiver combinations. Powder impact is weaker and should be inspected manually before deciding whether to keep it as a main mechanism or report it as a harder case.

These are automatic temporal proxies, not the final erasure score. The next required step is paired contact-sheet review using four labels: source-object removal, footprint removal, receiver preservation, and unrelated-mechanism preservation.

## Initial visual audit

Spot checks show that water impact usually removes both the source and its footprint. Rigid collision is mostly successful but can leave a small source remnant. Brittle fracture often preserves intact glass, although some Original videos do not generate a valid fracture and must be reported as base-model failures. Powder impact frequently suppresses the powder response while leaving the falling object visible, so its 65.12% temporal score overstates complete erasure quality.

The 82.91% aggregate must therefore be reported only as a temporal suppression proxy. Final claims require separate human labels for source removal, footprint removal, receiver preservation, and Original validity.

Raw rows are in `outputs/protocol_v1/pairwise_metrics/pairwise_metrics.csv`; grouped values are in `outputs/protocol_v1/pairwise_metrics/summary.csv`.
