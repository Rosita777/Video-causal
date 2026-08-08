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

For the fully unseen source and receiver split, suppression is 86.28%. The seen/seen split is 76.75%; the other three split groups are 84.10% to 86.28%.

## Interpretation

The first three mechanisms show strong causal-footprint suppression and transfer to unseen source/receiver combinations. Powder impact is weaker and should be inspected manually before deciding whether to keep it as a main mechanism or report it as a harder case.

These are automatic temporal proxies, not the final erasure score. The next required step is paired contact-sheet review using four labels: source-object removal, footprint removal, receiver preservation, and unrelated-mechanism preservation.

Raw rows are in `outputs/protocol_v1/pairwise_metrics/pairwise_metrics.csv`; grouped values are in `outputs/protocol_v1/pairwise_metrics/summary.csv`.
