# Waterdrop Adapter Cross-Object Water-Impact Test

Date: 2026-08-06

## Question

Does the waterdrop-trained mask-weighted dual-trajectory adapter affect unseen objects that fall into water?

Four fixed-seed comparisons were generated with the frozen Wan base and the waterdrop adapter: two apples and two stones, each falling into water. All four frozen-base videos produced a recognizable source object and water-impact chain.

## Human result

- Red apple into pond: apple, splash, and most ripples were removed.
- Green apple into glass tank: splash was removed, but the apple remained and the receiver geometry was rewritten.
- Gray stone into pond: stone, splash, and most ripples were removed.
- Dark stone into basin: stone, splash, and most ripples were removed.

Thus, three of four cases show strong combined cross-object erasure, while all four show clear footprint suppression. One case has a major preservation failure. These are four qualitative probes, not a statistically sufficient generalization benchmark.

## Interpretation

The current adapter is not strictly specific to the lexical object `water droplet`. It appears to learn a broader water-impact causal mechanism and can transfer from droplets to unseen apples and stones. This is promising mechanism-level generalization if the claimed adapter scope is `object impacts water`, but it is over-broad behavior if the claimed scope is only `waterdrop removal`.

The method and evaluation should therefore separate two claims:

1. Within-mechanism generalization across source objects is desirable.
2. Preservation on unrelated mechanisms and scenes is still required to measure collateral effects.

Contact sheets are in `experiments/waterdrop_adapter_cross_object_water4_sheets/`.
