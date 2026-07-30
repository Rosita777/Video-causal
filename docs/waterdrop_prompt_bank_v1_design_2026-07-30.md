# Waterdrop prompt bank v1 design (2026-07-30)

## Scope

This prompt bank covers local causal-footprint erasure for one falling water droplet. The receiver remains globally stationary. Large receiver deformation, bending, rebound, swinging, tearing, and breaking are out of scope.

## Composition

The bank contains 800 candidate scenes built from 200 receiver setups:

| Family | Receiver setups | Physical variants per receiver | Prompts |
| --- | ---: | ---: | ---: |
| Liquid surface | 50 | 4 | 200 |
| Hard non-absorbent surface | 50 | 5 | 250 |
| Shape-stable absorbent surface | 50 | 4 | 200 |
| Granular or powder surface | 50 | 3 | 150 |
| Total | 200 | - | 800 |

Every prompt is generated once with one seed. The variants change impact location, surface condition, or causal footprint; they are not seed repeats.

## Prompt contract

Every scene requires:

1. A fixed camera and unchanged background.
2. A clean, motionless two-second prefix.
3. Exactly one visibly falling water droplet.
4. A visible contact event.
5. A local footprint that begins only after contact.

Liquid scenes use a separate clean-prefix sentence so that the existing water surface is not contradicted.

## Files

- Structured metadata: `data/waterdrop_prompt_bank_v1.csv`
- Wan-compatible prompt list: `prompts/waterdrop_prompt_bank_v1.txt`
- Deterministic builder: `scripts/build_waterdrop_prompt_bank_v1.py`

Rebuild with:

```bash
python3 scripts/build_waterdrop_prompt_bank_v1.py
```

## Before generation

The 800 prompts are a candidate bank, not automatically accepted training data. Before launching all videos, sample each family for a small capability gate. Any receiver or footprint pattern that repeatedly fails should be removed from the large run.
