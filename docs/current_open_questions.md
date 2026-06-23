# Current Open Questions

Updated: 2026-06-23

## Recovery

1. GitHub push is still blocked intermittently by network/TLS connectivity. Local commits should continue to be made; push when the network path to GitHub is stable.
2. Generated videos and model weights remain local ignored artifacts and should not be pushed.

## Experiments

1. Which round5 `yes10` baseline outputs are strict causal-footprint leakage after full-video review?
2. Which round5 `borderline11` outputs are worth promoting to figure-candidate or backup evidence after adjudication?
3. How should round4-valid9 and round5-yes10 be combined in the first benchmark table without overstating sample size?
4. Should control prompts be generated now for annotation calibration, or after round5 baseline labels are finalized?
5. Should the six missing round2 car-barrier `T2VUnlearning` / `SAFREE-CogVideoX` rows be regenerated for historical completeness, or left as recovered-pilot evidence only?

## Writing

1. How should the paper name the failure: `causal footprint leakage`, `counterfactual erasure failure`, or both?
2. Should the main table use MLLM first-pass scores, human-adjudicated scores, or both side by side?
3. Should CLEAR remain related work only unless public code or a faithful implementation path appears?
