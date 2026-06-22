# Evaluator Calibration Summary

- Matched predictions: 8
- Strict leakage binary F1: 0.4000
- Relaxed leakage binary F1: 0.7692
- Macro F1: 0.1000

## Label Metrics

| Label | Support | TP | FP | FN | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| strict_leakage | 2 | 2 | 6 | 0 | 0.2500 | 1.0000 | 0.4000 |
| borderline | 3 | 0 | 0 | 3 | 0.0000 | 0.0000 | 0.0000 |
| target_leakage | 2 | 0 | 0 | 2 | 0.0000 | 0.0000 | 0.0000 |
| other_failure | 1 | 0 | 0 | 1 | 0.0000 | 0.0000 | 0.0000 |

## Confusion Matrix

| Gold | Predicted | Count |
| --- | --- | ---: |
| strict_leakage | strict_leakage | 2 |
| borderline | strict_leakage | 3 |
| target_leakage | strict_leakage | 2 |
| other_failure | strict_leakage | 1 |
