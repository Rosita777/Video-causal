# V2 Baseline Metrics

These metrics use human-verified v2 target/footprint labels.

- Total outputs: 384
- Target erased: 316/384 (0.8229)
- Footprint retained given target erased: 124/316 (0.3924)
- Strict causal-footprint leakage: 46/384 (0.1198)
- Strict leakage given target erased: 0.1456
- Erased clean: 149/384 (0.3880)
- Target leakage: 23/384 (0.0599)
- Borderline: 143/384 (0.3724)
- Other failure: 23/384 (0.0599)

## By Baseline

| Baseline | Outputs | Target erased | Strict | Erased clean | Target leakage | Borderline | Other failure | Strict rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ALL` | 384 | 316 | 46 | 149 | 23 | 143 | 23 | 0.1198 |
| `negative_prompt` | 96 | 67 | 11 | 25 | 8 | 52 | 0 | 0.1146 |
| `safree_zeroscope` | 96 | 81 | 3 | 57 | 0 | 23 | 13 | 0.0312 |
| `t2vunlearning` | 96 | 81 | 11 | 32 | 10 | 38 | 5 | 0.1146 |
| `videoeraser` | 96 | 87 | 21 | 35 | 5 | 30 | 5 | 0.2188 |
