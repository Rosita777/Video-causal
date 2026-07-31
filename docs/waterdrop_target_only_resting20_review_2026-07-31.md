# Resting-bead target-only review (2026-07-31)

## Purpose

The original target-only control asked Wan to leave a falling droplet suspended in air. That was unnecessarily difficult and did not represent the intended control. The replacement prompt asks for one water bead already resting on a hard surface from the first frame.

## Strict acceptance rule

A video passes only when:

1. one water bead is visible from the first frame;
2. it stays in the same place and remains approximately unchanged;
3. there is no falling water, impact, splash, ripple, trail, or spreading wet patch;
4. the receiver and camera remain stable.

## Result

- Generated: 20
- Semantic pass: 10
- Semantic fail: 10
- Pass rate: 50%

Passing scene IDs:

`wdresting000`, `wdresting001`, `wdresting002`, `wdresting005`, `wdresting006`, `wdresting007`, `wdresting010`, `wdresting011`, `wdresting016`, `wdresting017`.

Most failures ignored “already present in the first frame” and generated a delayed bead-formation or impact-like sequence. The automatic temporal screen is not a reliable judge for this static condition because it was designed to detect event onset. The manual labels in `data/waterdrop_target_only_resting20_semantic_review.csv` should be used.

## Decision

Keep the 10 passing clips as valid target-only controls. Do not count the other 10 as valid test samples. If more target-only clips are needed, generate extra candidates and select by the same strict rule rather than weakening the rule.
