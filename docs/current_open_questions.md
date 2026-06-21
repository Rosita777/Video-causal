# Current Open Questions

Updated: 2026-06-21

## Recovery

1. Can a filesystem snapshot restore `/home/deepseek_VG/JUNCHI` from before 2026-06-19 04:54?
2. If not, which non-git artifacts should be regenerated first: VideoEraser external checkout, T2VUnlearning external source/config, or generated videos?

## Experiments

1. Which candidate `C -> F(C)` pairs should enter the first candidate pool, and which should be rejected before generation?
2. Which control prompts should be paired with the main causal pairs to distinguish true footprints from natural background patterns?
3. Which 16-24 high-exclusivity pairs should form the first compute-light runnable benchmark slice?
4. Which round3 clean-pass prompts should become canonical figure examples after checking full videos, not only middle frames?
5. Should the six missing round2 car-barrier `T2VUnlearning` / `SAFREE-CogVideoX` rows be regenerated for historical completeness, or left as recovered-pilot evidence only?

## Writing

1. How should the paper name the failure: `causal footprint leakage`, `counterfactual erasure failure`, or both?
2. Should the main table use MLLM first-pass scores, human-adjudicated scores, or both side by side?
3. Should CLEAR remain related work only unless public code or a faithful implementation path appears?
