# Evaluator Calibration Summary

- Matched predictions: 36
- Strict leakage binary F1: 0.4000
- Relaxed leakage binary F1: 0.7600
- Macro F1: 0.3438

## Label Metrics

| Label | Support | TP | FP | FN | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| strict_leakage | 15 | 4 | 1 | 11 | 0.8000 | 0.2667 | 0.4000 |
| borderline | 9 | 4 | 17 | 5 | 0.1905 | 0.4444 | 0.2667 |
| target_leakage | 9 | 3 | 4 | 6 | 0.4286 | 0.3333 | 0.3750 |
| other_failure | 3 | 1 | 2 | 2 | 0.3333 | 0.3333 | 0.3333 |

## Confusion Matrix

| Gold | Predicted | Count |
| --- | --- | ---: |
| strict_leakage | strict_leakage | 4 |
| strict_leakage | borderline | 10 |
| strict_leakage | other_failure | 1 |
| borderline | strict_leakage | 1 |
| borderline | borderline | 4 |
| borderline | target_leakage | 3 |
| borderline | other_failure | 1 |
| target_leakage | borderline | 6 |
| target_leakage | target_leakage | 3 |
| other_failure | borderline | 1 |
| other_failure | target_leakage | 1 |
| other_failure | other_failure | 1 |
