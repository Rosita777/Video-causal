# Waterdrop five-condition test100 generation

Date: 2026-07-31

## Completion

- Generated videos: 100 / 100
- Contact sheets: 100 / 100
- Held-out receivers: 20 (10 liquid surface, 10 hard surface)
- Conditions per receiver: explicit causal, implicit causal, target only, unrelated footprint, and clean control
- Fixed seed: 9100

## First-pass temporal screen

| Condition | Candidate | Short-prefix review | No-clean-prefix | No detectable event |
|---|---:|---:|---:|---:|
| Explicit causal | 16 | 0 | 4 | 0 |
| Implicit causal | 17 | 0 | 3 | 0 |
| Target only | 14 | 3 | 3 | 0 |
| Unrelated footprint | 11 | 5 | 4 | 0 |
| Clean control | 2 | 5 | 6 | 7 |
| Total | 60 | 13 | 20 | 7 |

## Interpretation constraint

This screen was inherited from aligned-pair construction. It detects when a video changes relative to its first frame; it does not verify the semantics required by the five-condition evaluation.

In particular:

- a clean control with no detectable event can be correct rather than rejected;
- a target-only video must be checked for a visible droplet that never contacts the receiver;
- an unrelated-footprint video must be checked for the footprint and the absence of a falling droplet;
- explicit and implicit causal videos must be checked for the complete contact-to-footprint chain.

Therefore the temporal labels above are diagnostics, not final acceptance labels. The 100 contact sheets require condition-aware semantic review before the frozen evaluation set is finalized.
