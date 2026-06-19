# Research Notes: Video Concept Erasure and Causal Footprint

Updated: 2026-06-19

## Core Hypothesis

Current text-to-video concept erasure methods may remove the visible target concept but leave downstream effects that require the erased concept as their cause. We call this residual event evidence a **causal footprint**.

## Current Framing

The project is currently an empirical audit and evaluation-protocol effort. A new method should wait until the failure mode is demonstrated cleanly across valid clean-source cases.

## Clean-Source Gate

A prompt/seed can enter erasure evaluation only if the clean video has:

- visible target cause;
- visible downstream effect;
- clear cause-before-effect temporal order;
- plausible dependence of the effect on the target cause;
- sufficient quality for human judgment.

This gate prevents generic T2V failure from being misread as erasure failure.

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

Current safe wording: causal-footprint failures are observable and repeat for Negative Prompt across valid clean-source cases; VideoEraser shows one strict causal-footprint failure but needs more breadth before a strong method-level claim.
