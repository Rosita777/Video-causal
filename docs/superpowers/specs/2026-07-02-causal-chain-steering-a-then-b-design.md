# Causal Chain Steering A-Then-B Design

## Goal

Make the ZeroScope causal-chain steering probe harder to attack before scaling:
first test whether the previous negative result was caused by overly strong
steering parameters, then test whether paraphrase-averaged causal directions
separate better from random and unrelated semantic controls.

## Context

The current MVP-0 pilot produced a useful diagnostic failure. Fable judged
`full_chain_steering`, `random_direction`, and `orthogonal_semantic` with the
same outcome distribution on the 3-item control pilot. Low-level motion proxies
did not show simple global freezing, but semantic performance still failed to
separate from controls. Therefore the next work should not scale the current
configuration as a positive result.

## Phase A: Conservative Parameter Sweep

Phase A asks whether `alpha=0.5` and window `2:8` were too aggressive. It keeps
the same three pilot items and the same five core conditions:

```text
target_negative
target_footprint_negative
full_chain_steering
random_direction
orthogonal_semantic
```

It sweeps a small grid:

```text
alpha: 0.15, 0.25, 0.35
timestep_window: 2:5, 3:6, 4:7
```

Each grid cell should generate 15 videos: 3 items x 5 conditions. This is 135
videos total if every grid cell is run. The implementation should support
creating a dry-run manifest first, then running selected cells on GPU. The
first real execution can run one or two cells before the full grid if GPU time
or visual quality looks poor.

## Phase A Success Gate

Phase A only supports further scaling if at least one grid cell satisfies all
of the following on the 3-item pilot:

```text
1. full_chain_steering has fewer strict_causal_footprint_leakage labels than
   both random_direction and orthogonal_semantic.
2. full_chain_steering does not increase target_leakage relative to
   target_footprint_negative.
3. video_quality remains yes for all or nearly all full_chain_steering outputs.
4. low-level motion proxies do not indicate trivial global freezing or collapse.
```

If no grid cell passes this gate, Phase A should be reported as a negative
result and Phase B becomes the next method-improvement step.

## Phase B: Paraphrase-Averaged Causal Directions

Phase B asks whether one brittle minimal pair per causal link is too noisy. It
extends the steering contract so each link may have multiple positive/negative
paraphrase pairs. During each denoising step, the runner computes each pair's
residual direction and averages those directions before applying steering.

The first Phase B version should use 3 paraphrase pairs per link:

```text
cause:      3 pairs
mechanism:  3 pairs
footprint:  3 pairs
```

Random controls should be norm-matched to the averaged footprint direction, not
to a single original pair. Orthogonal semantic control should remain present.
The same Phase A success gate applies.

## Phase B Scope Boundary

Phase B should not add a scene-preservation regularizer yet. That would make it
harder to tell whether improvement comes from causal direction stability or
from an additional prompt-engineering constraint. Scene preservation can become
Phase C only if Phase B still fails to separate from controls.

## Evaluation Flow

Every real batch should produce:

```text
generation_manifest.json
videos/*.mp4
contact_sheet.jpg
review.csv
frame_strips/*.jpg
vlm_predictions.csv
low_level_proxy.csv
low_level_proxy_summary.csv
```

Fable VLM scoring remains the main semantic gate. Low-level proxies remain
guardrails for collapse, freezing, excessive blur, or scene destruction. Manual
inspection is useful for triage but should not be the main claim.

## Implementation Shape

Phase A can be implemented with a small orchestration script that calls the
existing runner repeatedly with different `--alpha` and `--timestep-window`
values. The core runner does not need behavior changes for Phase A unless
manifest metadata needs clearer grid identifiers.

Phase B requires runner changes:

```text
1. Accept minimal-pair values that are either one pair or a list of pairs.
2. Encode all paraphrase pairs for each active steering link.
3. Average residual directions per link at each denoising step.
4. Use the averaged footprint direction as the random-control norm reference.
5. Preserve current single-pair behavior for existing manifests.
```

Tests should be written before runner changes. Phase A tests should verify grid
manifest planning without invoking the diffusion model. Phase B tests should
verify backward compatibility, averaged residual arithmetic, and random-control
norm matching against the averaged footprint direction.

## Non-Claims

This design does not claim causal repair yet. The claim under test is narrower:
whether causal-chain steering can reduce causal-footprint leakage more than
norm-matched random steering and unrelated semantic steering while preserving
video quality. A failure to beat those controls is a negative method result,
not an implementation failure.

## Review Note

This project directory is not currently a git repository, so the design file
cannot be committed from this workspace. The file itself is the review artifact.
