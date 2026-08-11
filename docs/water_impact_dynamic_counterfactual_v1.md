# Water-impact dynamic counterfactual protocol v1

## Purpose

This dataset trains one adapter for the mechanism **object enters water**. Given
a factual water-impact prompt, the desired video removes both the incoming
object and its impact footprint. The water, receiver, reflections, and other
natural scene motion remain visible.

The counterfactual target is a separately generated dynamic video. It is not a
clean frame repeated through time, and the protocol makes no assumption that a
fixed number of opening frames is static.

## Scale and split

| Split | Construction | Pairs |
| --- | --- | ---: |
| Train | 8 seen sources x 12 seen receivers x 2 phrasings | 192 |
| Test: unseen source | 6 unseen sources x 4 seen receivers | 24 |
| Test: unseen receiver | 3 seen sources x 8 unseen receivers | 24 |
| Test: both unseen | balanced subset of 6 x 8 combinations | 24 |

The unseen source and receiver vocabularies are disjoint from training. Test
prompts are never used to train the adapter.

## Pair semantics

Each row contains two prompts:

- `training_prompt`: the factual condition presented to the adapter, including
  the source, contact, splash, and expanding ripples.
- `target_generation_prompt`: a target-only prompt used to create the desired
  dynamic counterfactual video. It keeps mild water movement and shifting
  reflections, but contains no incoming source or impact-generated footprint.

Training uses `training_prompt` with the video made from
`target_generation_prompt`. Pixel alignment between a factual video and its
counterfactual is not required by the first SFT baseline. Seeds are unique per
row, so repeated receiver descriptions still produce distinct target videos.

## Quality gate

Before full generation, inspect a small target-video sample. Accept a target
only when all of the following hold:

1. The named receiver is clearly present.
2. The video has natural temporal motion and is not a repeated static frame.
3. No object falls into or contacts the water.
4. No localized splash, cavity, or newly expanding circular impact wave appears.
5. The camera and receiver remain stable.

The target videos are training data and therefore require a stricter gate than
ordinary benchmark inputs. Failed targets should be regenerated with a new
seed, not silently retained.

## Rebuild

```bash
python3 scripts/build_water_impact_dynamic_pairs_v1.py
```

The command writes structured CSV manifests under
`data/water_impact_dynamic_v1/` and generator-ready prompt lists under
`prompts/water_impact_dynamic_v1/`.
