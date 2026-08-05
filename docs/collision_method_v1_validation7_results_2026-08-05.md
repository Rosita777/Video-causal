# Collision method v1 validation7

The new 100-step dual-trajectory checkpoint was evaluated on the frozen seven-case collision validation set at inference LoRA scale 0.75.

| Metric | Mean |
| --- | ---: |
| Post-change motion suppression proxy | 76.48% |
| Early base-adapter MAE | 0.12523732 |

The suppression value is an automatic motion proxy, not a semantic proof that the red ball and every downstream toppled object were erased. The seven base/adapter contact sheets remain the required human review artifact.
