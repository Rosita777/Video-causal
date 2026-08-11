# Dynamic water-impact target smoke test (2026-08-11)

## Result

The dynamic counterfactual design is feasible, but generated targets must be
screened before training.

The first target prompt described natural `surface undulations` and also named
forbidden circular waves in a negated sentence. On a glass-bowl sample, Wan
rendered strong concentric rings from the first frame. This target was rejected.

The revised prompt:

- uses only positive target-state language;
- keeps the water smooth and level;
- expresses temporal change through moving reflections and highlights; and
- passes event terms through the generator's negative-prompt channel.

The same glass-bowl seed then produced a clean surface with no rings. It had 49
frames, mean adjacent-frame absolute difference `0.214`, and first-to-last
difference `1.373`, so it was dynamic rather than a repeated frame.

## Receiver coverage

One direct target was generated for each of the 12 training receivers. Ten had
a recognizable receiver and no impact event. Two seeds produced unusable scene
composition: the pond became a small dark object on an empty background, and
the glass bowl was poorly framed. These are seed-level generation failures,
not valid training targets.

The full target run therefore follows this policy:

1. Generate one target per training row with its recorded seed.
2. Screen receiver presence, temporal variation, source absence, and impact
   footprint absence.
3. Regenerate each failed row with a new recorded seed.
4. Build the SFT manifest only from accepted target videos.

The smoke-test videos remain server-side. The compact receiver overview is in
`experiments/water_impact_dynamic_v1_smoke/receiver_smoke12_overview.jpg`.
