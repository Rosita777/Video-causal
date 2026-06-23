# Causal Footprint Evaluation Metrics

- Total outputs: 56
- Strict leakage: 24/56 (0.4286)
- Borderline: 12/56 (0.2143)
- Relaxed leakage: 36/56 (0.6429)
- Target leakage: 14/56 (0.2500)

## By Baseline

| Baseline | Outputs | Strict | Borderline | Relaxed | Target leakage | Other failure | Relaxed rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ALL | 56 | 24 | 12 | 36 | 14 | 6 | 0.6429 |
| negative_prompt | 14 | 5 | 2 | 7 | 6 | 1 | 0.5000 |
| safree_cogvideox | 14 | 5 | 2 | 7 | 5 | 2 | 0.5000 |
| t2vunlearning | 14 | 6 | 3 | 9 | 3 | 2 | 0.6429 |
| videoeraser | 14 | 8 | 5 | 13 | 0 | 1 | 0.9286 |

## Model Agreement

| Model | Compared | Agree | Disagree | Agreement rate |
| --- | ---: | ---: | ---: | ---: |
| claude | 36 | 12 | 24 | 0.3333 |
