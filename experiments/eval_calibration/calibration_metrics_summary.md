# Evaluator Calibration Summary

- Matched predictions: 56
- Strict leakage binary F1: 1.0000
- Relaxed leakage binary F1: 1.0000
- Macro F1: 1.0000

## Label Metrics

| Label | Support | TP | FP | FN | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| strict_leakage | 24 | 24 | 0 | 0 | 1.0000 | 1.0000 | 1.0000 |
| borderline | 12 | 12 | 0 | 0 | 1.0000 | 1.0000 | 1.0000 |
| target_leakage | 14 | 14 | 0 | 0 | 1.0000 | 1.0000 | 1.0000 |
| other_failure | 6 | 6 | 0 | 0 | 1.0000 | 1.0000 | 1.0000 |

## Confusion Matrix

| Gold | Predicted | Count |
| --- | --- | ---: |
| strict_leakage | strict_leakage | 24 |
| borderline | borderline | 12 |
| target_leakage | target_leakage | 14 |
| other_failure | other_failure | 6 |
