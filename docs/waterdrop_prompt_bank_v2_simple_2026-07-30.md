# Waterdrop prompt bank v2 simple (2026-07-30)

## Scope change

Gate40 showed that Wan reliably supports liquid-surface splash/ripple and sometimes supports simple hard-surface impact. Absorption and granular displacement are not reliable enough for large-scale generation.

V2 therefore removes:

- all absorbent-surface prompts;
- all granular and powder prompts;
- hard-surface hydrophobic recoil;
- hard-surface edge gathering;
- hard-surface inclined sliding;
- liquid impacts near a container edge.

## V2 composition

| Family | Receivers | Variants per receiver | Prompts |
| --- | ---: | ---: | ---: |
| Liquid surface | 50 | 3 simple impacts | 150 |
| Hard surface | 50 | 2 simple impacts | 100 |
| Total | 100 | - | 250 |

Every prompt is generated once with one fixed seed. V1 and its gate results remain in the repository as evidence for the scope decision.

## Files

- `data/waterdrop_prompt_bank_v2_simple.csv`
- `prompts/waterdrop_prompt_bank_v2_simple.txt`
- `scripts/build_waterdrop_prompt_bank_v2_simple.py`
