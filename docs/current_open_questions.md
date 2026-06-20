# Current Open Questions

Updated: 2026-06-20

## Recovery

1. Can a filesystem snapshot restore `/home/deepseek_VG/JUNCHI` from before 2026-06-19 04:54?
2. If not, which non-git artifacts should be regenerated first: VideoEraser external checkout, T2VUnlearning external source/config, or generated videos?

## Experiments

1. Which 30-50 prompts should enter causal-footprint benchmark v0, and which 16 should be the first compute-light runnable slice?
2. Which round3 clean-pass prompts should become canonical figure examples after checking full videos, not only middle frames?
3. Should the six missing round2 car-barrier `T2VUnlearning` / `SAFREE-CogVideoX` rows be regenerated for historical completeness, or left as recovered-pilot evidence only?
4. Should the next clean-source expansion retarget `pipette` to `ink droplet`, where the visible cause is cleaner?

## Writing

1. How should the paper name the failure: `causal footprint leakage`, `counterfactual erasure failure`, or both?
2. What threshold should define strict leakage in the main text: `target_presence_score <= 1` and `footprint_presence_score >= 2`, or a more conservative rule?
3. Should CLEAR remain related work only unless public code or a faithful implementation path appears?
