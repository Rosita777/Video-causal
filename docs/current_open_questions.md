# Current Open Questions

Updated: 2026-06-21

## Recovery

1. Can a filesystem snapshot restore `/home/deepseek_VG/JUNCHI` from before 2026-06-19 04:54?
2. If not, which non-git artifacts should be regenerated first: VideoEraser external checkout, T2VUnlearning external source/config, or generated videos?

## Experiments

1. Which round4 `borderline` clean-source rows should be promoted or rejected after joint review of the annotated gallery?
2. Should `prompts/causal_footprint_v0_round4_clean_valid9.txt` be run as-is on all four baselines, or should we first add replacement seeds for surface-trace and agent-object response?
3. How many clean-valid rows per mechanism type are enough for v0 before freezing the benchmark slice?
4. Should control prompts be generated now for annotation calibration, or after clean-valid benchmark rows are finalized?
5. Should the six missing round2 car-barrier `T2VUnlearning` / `SAFREE-CogVideoX` rows be regenerated for historical completeness, or left as recovered-pilot evidence only?

## Writing

1. How should the paper name the failure: `causal footprint leakage`, `counterfactual erasure failure`, or both?
2. Should the main table use MLLM first-pass scores, human-adjudicated scores, or both side by side?
3. Should CLEAR remain related work only unless public code or a faithful implementation path appears?
