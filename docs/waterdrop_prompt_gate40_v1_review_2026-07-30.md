# Waterdrop prompt gate40 v1 review (2026-07-30)

## Setup

- 40 prompts sampled from the 800-prompt bank
- 10 prompts per family
- One generation per prompt
- The same fixed seed (`9000`) for all prompts
- Wan 2.1 T2V 1.3B, 49 frames, 8 fps, 25 steps

## Result

| Family | Pass | Borderline | Fail | Potentially usable |
| --- | ---: | ---: | ---: | ---: |
| Liquid surface | 6 | 2 | 2 | 8/10 |
| Hard surface | 3 | 1 | 6 | 4/10 |
| Absorbent surface | 0 | 1 | 9 | 1/10 |
| Granular surface | 0 | 0 | 10 | 0/10 |
| Total | 9 | 4 | 27 | 13/40 |

`Potentially usable` includes borderline samples. Borderline does not mean accepted training supervision; it means that a narrower prompt or final pair-level review might recover the sample.

## Main findings

Liquid surfaces are the only consistently viable family. Wan usually generates a visible falling droplet followed by a splash or ripple, although it often ignores the requested two-second clean prefix.

Hard surfaces are viable only for simple center impacts and local beads or wet spots. Edge impacts, inclined sliding, and hydrophobic recoil are unreliable.

Absorption is not reliably modeled. Wan often renders the intended wet patch as a glossy black liquid pool, creates the mark before contact, or fails to preserve the receiver identity.

Granular displacement is not reliably modeled. Common failures are pre-existing craters, droplets resting on top without changing the particles, and powder surfaces behaving like liquid.

## Decision

Do not launch all 800 prompts as currently written. First revise the large bank around the two supported regimes:

1. liquid surface splash/ripple;
2. simple hard-surface impact with a local bead or wet spot.

Absorbent and granular families should be removed from the first large training run unless a stronger factual generator is introduced.

Detailed decisions are stored in `data/waterdrop_prompt_gate40_v1_review.csv`. Contact sheets are stored in `results/waterdrop_prompt_gate40_v1_contact_sheets/`.

## Parallel generation note

One Wan process used about 21 GB on an 80 GB A100. Three processes per GPU are likely to fit in memory, so the next large generation can start with six total processes across GPU 2 and GPU 3. Since one process already reports near-100% GPU utilization, throughput should be measured rather than assuming a three-times speedup.
