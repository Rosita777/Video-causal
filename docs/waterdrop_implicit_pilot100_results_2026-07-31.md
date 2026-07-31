# Waterdrop implicit-outcome pilot 100

Date: 2026-07-31

## Setup

- Backbone: Wan2.1-T2V-1.3B-Diffusers
- Samples: 100 prompts, one fixed seed (9000) per prompt
- Families: 50 liquid-surface scenes and 50 hard-surface scenes
- The generation prompts describe the droplet entering and contacting the receiver, but do not mention splash, ripple, impact cavity, beads, or wet marks.
- The same receiver and variant-0 scenes from the explicit prompt bank are used for a matched comparison.

## Automatic screen

| Family | Candidate | Short-prefix review | No-clean-prefix reject | Total |
|---|---:|---:|---:|---:|
| Liquid surface | 25 | 2 | 23 | 50 |
| Hard surface | 28 | 2 | 20 | 50 |
| Total | 53 | 4 | 43 | 100 |

## Matched explicit-versus-implicit comparison

For the same 100 receivers and impact locations:

| Prompt type | Candidate | Short-prefix review | No-clean-prefix reject |
|---|---:|---:|---:|
| Explicit causal outcome | 60 | 7 | 33 |
| Implicit causal outcome | 53 | 4 | 43 |

- 51 scenes are candidates under both prompt types.
- 9 explicit candidates become non-candidates with the implicit prompt.
- 2 explicit non-candidates become candidates with the implicit prompt.
- Omitting the outcome words reduces the candidate rate from 60% to 53% on this matched set.

## Interpretation

Wan can often produce a temporally detectable event without being told the causal footprint words. The seven-point candidate-rate drop is meaningful but not large enough to make the implicit condition unusable. The implicit set should therefore be retained as a separate generalization/evaluation condition rather than merged blindly with the explicit set.

The automatic screen checks video validity and the presence of a clean prefix followed by detectable change. It is not, by itself, proof that every generated change is the intended splash, ripple, bead, or wet mark.
