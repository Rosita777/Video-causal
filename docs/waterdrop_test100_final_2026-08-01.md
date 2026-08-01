# Frozen waterdrop test100 (2026-08-01)

## Final composition

The frozen manifest is `data/waterdrop_test100_final.csv`.

| Condition | Samples |
|---|---:|
| Explicit causal chain | 20 |
| Implicit causal chain | 20 |
| Target only, no footprint | 20 |
| Unrelated footprint preservation | 20 |
| Clean-scene preservation | 20 |
| Total | 100 |

Every selected base video was manually reviewed from a 12-frame contact sheet. The manifest contains the generated-video path and contact-sheet path on A100_sc.

## Replacements

- Two failed unrelated-footprint samples were replaced by `wdcontrolfix000` and `wdcontrolfix004`.
- The unstable clean puddle was replaced by `wdcleanfix001`, a dry empty stone-step control.
- The suspended-droplet target-only condition was replaced completely by the final 20 resting-bead controls.

## Important interpretation

This dataset is balanced by condition but is not a strict five-way paired dataset. The target-only controls use independently screened hard surfaces because Wan could not reliably produce a suspended droplet on the original receivers. Evaluation must therefore report each condition separately; it must not claim that all five conditions share the same receiver for every sample.

## Freeze rule

Do not use these prompts, receivers, or generated videos for adapter training. Training data must be constructed separately and checked against the `scene_id`, `receiver_id`, and prompt fields in the frozen manifest.
