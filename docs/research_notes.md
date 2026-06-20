# Research Notes: Video Concept Erasure and Causal Footprint

Updated: 2026-06-20

## Core Hypothesis

Current text-to-video concept erasure methods may remove the visible target concept but leave downstream effects that require the erased concept as their cause. We call this residual event evidence a **causal footprint**.

## Current Framing

The project is now a benchmark-first effort. The immediate goal is to define and evaluate **causal footprint leakage** before designing a new erasure method.

Current benchmark design spec:

```text
docs/superpowers/specs/2026-06-20-causal-footprint-benchmark-v0-design.md
```

Working definition:

- `C`: source concept or event participant to erase.
- `E(C)`: direct visual evidence of the source concept.
- `F(C)`: causal footprint left by the source concept.

The key failure is `E(C)` low while `F(C)` remains high. This separates causal-footprint leakage from ordinary target-visible erasure failure.

## Clean-Source Gate

A prompt/seed can enter erasure evaluation only if the clean video has:

- visible target cause;
- visible downstream effect;
- clear cause-before-effect temporal order;
- plausible dependence of the effect on the target cause;
- sufficient quality for human judgment.

This gate prevents generic T2V failure from being misread as erasure failure.

Current CogVideoX-2B clean-source status:

- `ice_cube_seed101` from the initial two-prompt smoke is visually usable.
- `ice_cube_seed200` and `stone_seed204` from round1 seed200-205 are clean-valid candidates.
- `ball_seed100`, `bottle_seed201`, `pitcher_seed202`, `pipette_seed203`, and `sugar_cube_seed205` are not clean-valid under the current screening notes.

The most common clean-source failure is not conceptual erasure failure; it is base-model generation failure: the target cause, downstream effect, or both are absent in the clean source.

## New CogVideoX-2B Baseline Evidence

Negative Prompt round1 on the two current clean-valid sources produced two strict causal-footprint candidates:

- `ice_cube_seed200`: the ice cube is not clearly visible, but cola surface turbulence and bubbles remain.
- `stone_seed204`: the stone/impact object is absent, but circular ripples still appear and expand.

This is consistent with the core hypothesis: prompt-level target suppression can remove or weaken the visible cause while preserving downstream event evidence.

## Recovered Evidence

The recovered cross-round matrix has 59 annotated rows across 13 clean-source-valid cases.

Strict positives:

- Negative Prompt on `pitcher_seed63`, `ice_cube_seed66`, `ice_cube_seed67`.
- VideoEraser on `pitcher_seed63`.

No strict positives are recovered for T2VUnlearning or SAFREE-CogVideoX.

Round2 car-barrier has clean/Negative Prompt/VideoEraser rows but still lacks T2VUnlearning and SAFREE-CogVideoX summary rows.

## Related Work Positioning

- **VideoEraser**: strong official video erasure baseline; current recovered strict positive only on pitcher-water.
- **T2VUnlearning**: finetune/unlearning baseline; locally reproduced before workspace loss, but source/adapters must be recovered or rerun.
- **SAFREE**: CLEAR-aligned training-free baseline; the project used a disclosed CogVideoX adaptation before workspace loss.
- **CLEAR / Concept-Layer Alignment**: closest ICML 2026 T2V concept-erasure work; code was unavailable in the recovered project state.

## Claim Strength

Current safe wording: causal-footprint failures are observable in multiple CogVideoX-2B prompt families and repeat across several baseline interfaces. The next scientific step is not to claim a final method-level ranking, but to build a structured benchmark with clean-source gates, target-presence annotations, and causal-footprint annotations.

The strongest planned metric is conditional footprint persistence:

```text
CFP@TPS<=1
```

This means causal-footprint persistence measured only when the target concept is already weak or absent. It directly addresses the reviewer objection that examples are merely incomplete erasure.
