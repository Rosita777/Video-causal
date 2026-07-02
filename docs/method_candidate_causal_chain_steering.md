# Method Hypothesis and Stress-Test Plan: Minimal-Pair Causal Chain Steering

Updated: 2026-07-02

This note defines a **method hypothesis**, not a settled method. The current
hypothesis is that causal-footprint leakage can be reduced by estimating and
steering away from decomposed cause-mechanism-footprint directions in the
diffusion trajectory. The immediate goal is not to ship a full method, but to
stress-test whether this hypothesis survives the strongest reviewer objections:
prompt confounding, planner dependence, over-erasure, weak controls, and
backbone-specific artifacts.

The design should be treated as provisional until a small mechanism probe shows
that it beats strong prompt-only controls without damaging video quality.

## Core Intuition

A user request such as "remove the stone" from a video prompt is not only a
request to suppress stone pixels. In a physically plausible counterfactual
world, the event caused by the stone should also disappear:

```text
stone -> water impact -> splashes and ripples
```

The method should therefore act before or during generation, not as a
post-generation repair. It should steer the denoising trajectory away from the
causal chain that produces the target and its footprint.

## Reviewer Risk in the Naive Version

A naive counterfactual steering method would compare one factual prompt against
one counterfactual prompt:

```text
P_factual:        a stone falls into a lake and creates splashes and ripples
P_counterfactual: a calm lake surface with no stone, no impact, no splashes
```

The difference between their noise predictions is not guaranteed to isolate
the causal footprint. It can also include background, camera, lighting, style,
water texture, and motion differences. A reviewer can fairly attack this as a
confounded prompt-difference direction.

The design below avoids relying on one monolithic counterfactual prompt.

## Working Hypothesis

Minimal-Pair Causal Chain Steering decomposes the requested erasure into a
small causal chain and estimates denoising directions from controlled prompt
pairs. Each pair is designed to change only one causal sub-concept while
holding the surrounding scene context as fixed as possible.

The working hypothesis is:

```text
If a downstream footprint is encoded along cause/mechanism/effect directions in
the denoising trajectory, then steering away from a decomposed chain direction
should reduce footprint leakage more cleanly than prompt-only erasure.
```

The null hypothesis is equally important:

```text
If minimal-pair steering matches footprint-aware prompting or damages video
quality, then this is not yet a defensible main method.
```

```text
input prompt + target concept + benchmark causal metadata
        |
        v
causal chain decomposition
        |
        v
minimal-pair prompt construction
        |
        v
per-link denoising direction estimation
        |
        v
composite causal-chain steering during denoising
        |
        v
target-erased, footprint-erased video
```

## Inputs

For the mechanism probe, use metadata already present in the benchmark rather
than an external planner:

- original prompt;
- target concept;
- causal mechanism label;
- expected footprint descriptor;
- optional scene/context descriptor.

This avoids API dependence and makes the causal decomposition reproducible.
This choice weakens generality claims, so the method must not claim automatic
causal graph discovery. An LLM planner can later be tested as an ablation, not
as the core mechanism.

## Causal Chain Decomposition

Each benchmark item is mapped to a chain with three slots:

```text
cause      C: the removed target or event participant
mechanism  M: the contact, force, emission, or interaction that transmits the cause
effect     F: the visible downstream footprint
```

Examples:

| Mechanism family | Cause | Mechanism | Footprint |
|---|---|---|---|
| fluid impact | stone | impact with water | splashes and circular ripples |
| fracture damage | hammer | impact with glass/tile | cracks and broken fragments |
| surface trace | shoe | pressure/contact on mud | footprint trace |
| elastic deformation | ball | collision with net | net deformation |
| field mediated | magnet | magnetic field near filings | filings alignment |
| particle dispersion | object | collision or blast | dust or particles spreading |

The decomposition does not claim to discover a full causal graph. It supplies a
structured intervention target for a known benchmark item.

## Minimal-Pair Prompt Construction

Instead of a single factual/counterfactual pair, construct several local pairs.
For each causal link, keep scene context fixed and flip only one concept:

```text
Cause pair:
  A_C: {scene context} with {cause}
  B_C: {scene context} without {cause}

Mechanism pair:
  A_M: {scene context} with {mechanism}
  B_M: {scene context} without {mechanism}

Footprint pair:
  A_F: {scene context} with {footprint}
  B_F: {scene context} without {footprint}
```

For the stone-water example:

```text
Cause:
  a lake scene with a stone above the water
  a lake scene with no stone above the water

Mechanism:
  a lake surface with a visible water impact
  a lake surface with no impact or disturbance

Footprint:
  a lake surface with splashes and circular ripples
  a lake surface with no splashes and no circular ripples
```

These pairs are still prompts, so they are imperfect. Their purpose is not to
solve the task directly, but to estimate cleaner denoising directions than a
single broad counterfactual rewrite.

Prompt wording remains a major confound. The stress test must therefore include
prompt-only and monolithic-counterfactual baselines. If those baselines match
the steering result, the method is prompt engineering rather than trajectory
intervention.

## Denoising Direction Estimation

At denoising step `t` and latent state `z_t`, estimate one direction for each
causal link:

```text
d_C(t) = eps(z_t, t, A_C) - eps(z_t, t, B_C)
d_M(t) = eps(z_t, t, A_M) - eps(z_t, t, B_M)
d_F(t) = eps(z_t, t, A_F) - eps(z_t, t, B_F)
```

The composite causal-chain direction is:

```text
d_chain(t) = w_C d_C(t) + w_M d_M(t) + w_F d_F(t)
```

The MVP should start with simple weights such as `(1, 1, 1)` and then sweep
per-link weights to test which link dominates footprint leakage.

## Steering Rule Under Test

Generate from the original or target-erased prompt while subtracting the
estimated chain direction:

```text
eps_steered(t) = eps_base(t) - alpha(t) * normalize(d_chain(t))
```

The base condition can be:

- the original prompt with target-erasure guidance;
- a target-only negative-prompt baseline condition;
- a counterfactual-neutral prompt condition.

The first probe should keep the base condition simple and compare all three as
ablations if runtime permits. The steering rule itself is under test; it should
not be described as validated before prompt-only and random-direction controls
are run.

## Timestep and Temporal Structure Probe

Causal footprints may be laid down before final appearance details. The MVP
should therefore test timestep windows:

- early-only steering;
- mid-only steering;
- late-only steering;
- full-range steering;
- scheduled steering, strong early and weaker late.

For video, a later version may also apply link-specific temporal masks:

```text
cause: early frames
mechanism: middle frames
footprint: later frames
```

The first implementation can omit explicit temporal masks and use timestep
windows only. Temporal localization should be treated as a second-stage
extension unless the first results are too noisy.

## Localization Requirement

Attention-localized steering would reduce global scene corruption by applying
directions only where causal tokens attend. This is attractive but
implementation-heavy across ZeroScope, Wan, and CogVideoX.

For a **mechanism probe**, global steering is acceptable only as a diagnostic.
For a **paper method**, some localization story is likely required. If global
steering reduces footprint leakage by erasing broad scene content, the method is
not successful. The first implementation should therefore log quality and
background-preservation failures aggressively, and the next method version
should include attention, token, frame, or latent-region localization.

## Required Controls

The method is only defensible if it beats strong controls:

1. **Target-only negative prompt**: existing erasure floor.
2. **Target-plus-footprint negative prompt**: tests whether enumeration of
   footprints is enough.
3. **Monolithic counterfactual prompt**: tests whether a single `do(not C)`
   prompt is enough.
4. **Prompt-only minimal-pair rewrite**: uses generated counterfactual or
   footprint-free prompts without denoising steering.
5. **Random-direction steering**: same norm as `d_chain(t)`, but random in
   noise-prediction space.
6. **Single-link steering**: cause-only, mechanism-only, footprint-only.

The strongest claim requires:

```text
minimal-pair steering > target+footprint negative prompt
minimal-pair steering > monolithic counterfactual prompt
minimal-pair steering > random-direction steering
```

## MVP-0: Mechanism Probe, Not Final Method

Start on ZeroScope because the v2 branch is closed and the evaluator path is
already stable.

Recommended slice:

- 24-36 clean-valid ZeroScope prompts;
- prioritize `fluid_impact`, `fracture_damage`, `elastic_deformation`, and
  `particle_dispersion`;
- include a few cases where existing baselines produce strict leakage.

Conditions:

```text
1. target-only negative prompt
2. target+footprint negative prompt
3. monolithic counterfactual prompt
4. random-direction steering
5. cause-only steering
6. mechanism-only steering
7. footprint-only steering
8. full minimal-pair chain steering
9. timestep-window variants for the best chain setting
```

Metrics:

- target erased;
- strict causal-footprint leakage;
- footprint retained given target erased;
- erased clean;
- target leakage;
- borderline / quality failure;
- per-mechanism breakdown.

Mechanism-probe success criterion:

```text
full minimal-pair chain steering reduces strict leakage by at least 30%
relative to target+footprint negative prompt, while preserving target-erased
rate and avoiding a large increase in quality failure.
```

Strong success:

```text
full chain steering reduces strict leakage by at least 50% and shows a clear
timestep-window effect, especially early/mid steering outperforming late-only.
```

Pivot criterion:

```text
if full chain steering matches prompt-only or target+footprint negative prompt,
then the method is not adding enough beyond prompt engineering. Pivot to either
attention-localized steering or a benchmark-first paper with prompt ablations.
```

MVP-0 should be explicitly framed as a **go/no-go probe**:

```text
Go:
  chain steering beats prompt-only controls and shows meaningful timestep
  sensitivity without quality collapse.

No-go:
  prompt-only controls match chain steering, random directions behave similarly,
  or quality collapse explains the leakage reduction.
```

## Reviewer Attack Points and Planned Defenses

Attack: "This is just a better prompt."

Defense: include prompt-only, target+footprint, and monolithic counterfactual
baselines. The method must win through denoising steering, not text alone.

Attack: "Your counterfactual direction is confounded."

Defense: use decomposed minimal pairs, single-link ablations, and
random-direction controls. Avoid claiming exact causal discovery.

Attack: "The causal planner is doing the work."

Defense: use deterministic benchmark metadata and templates for the MVP. Treat
LLM planning as an ablation or future generalization.

Attack: "You erase too much."

Defense: report erased-clean, quality failures, target leakage, and
background-preservation examples. If quality drops sharply, the method is not
successful even if footprint leakage falls.

Attack: "The method is not general."

Defense: first show a controlled ZeroScope MVP. Port to Wan only after the
direction works and after Wan baselines establish the same failure mode.

Attack: "The benchmark metadata is doing the reasoning."

Defense: do not claim automatic reasoning in the MVP. The claim is that, given
a structured causal chain, denoising trajectory intervention can reduce
footprint leakage beyond prompt-only controls. General causal-chain extraction
is a separate module and should be tested later.

Attack: "Minimal pairs are still prompt hacks."

Defense: treat minimal pairs as controlled probes rather than the final
erasure interface. The decisive comparison is whether denoising steering using
the pairs outperforms directly using the same words as prompts or negative
prompts.

## Relationship to Existing Work

This candidate borrows the objective from counterfactual video generation and
object/effect removal, but changes the interface:

- unlike VOID, ROSE, and EffectErase, it does not assume an input video mask or
  paired counterfactual video;
- unlike standard concept erasure, it targets the cause, mechanism, and
  footprint chain;
- unlike prompt-only counterfactual steering, it intervenes in the denoising
  trajectory;
- unlike a full causal world model, it does not claim to infer a complete scene
  graph.

The safest paper claim is:

```text
We show that causal-footprint leakage can be reduced by steering video
diffusion away from decomposed cause-mechanism-effect directions estimated from
minimal prompt pairs.
```

## Immediate Next Steps

1. Do targeted literature checks for concept directions, ESD-style erasure,
   CFG/negative prompt variants, timestep intervention, and attention-localized
   editing.
2. Map a small benchmark v2 slice to `(scene, cause, mechanism, footprint)`
   slots.
3. Create template-generated minimal pairs for 6-12 ZeroScope prompts first,
   not the full 24-36 slice.
4. Implement prompt-only and negative-prompt ablations before denoising
   steering.
5. Prototype denoising-direction extraction and full-chain steering on one
   ZeroScope prompt before launching a slice.
6. Decide whether localization is required before scaling beyond the probe.

## MVP-0 Probe Status

As of 2026-07-02, the first validation artifact is a dry-run mechanism probe,
not a real steering result. It selects 12 clean-valid ZeroScope v2 cases from
the already evaluated leakage set, balances them across mechanism families, and
expands each case into prompt-only controls plus cause, mechanism, footprint,
full-chain, and random-direction steering contracts.

Artifact root:

```text
experiments/method_probe/zeroscope_mvp0_causal_chain_probe_20260702/
```

Current dry-run checks:

- 12 probe cases, balanced across `fluid_impact`, `fracture_damage`,
  `elastic_deformation`, `particle_dispersion`, `surface_trace`, and
  `field_mediated`.
- 96 planned generation rows: 12 cases times 8 conditions.
- Minimal-pair prompts are sanitized so footprint-only controls and explicit
  "no target" counterfactual clauses do not become contradictory contexts.
- Real ZeroScope steering remains disabled until the denoising-loop insertion
  path is inspected and written down.

This means the next scientific gate is still open: run one or a few real
ZeroScope probes and check whether minimal-pair directions reduce footprint
leakage beyond prompt-only and random-direction controls.

### Real Runner Implementation Decision

Chosen path: copy a small ZeroScope `TextToVideoSDPipeline` denoising loop into
the MVP-0 runner rather than using the public callback hook.

Reason: in the installed generation environment (`dyme`, `diffusers==0.34.0`),
`TextToVideoSDPipeline.__call__` exposes the old `callback(step, timestep,
latents)` API. That callback is invoked after `scheduler.step(...)`, so it can
observe updated latents but cannot cleanly modify the guided `noise_pred`
before the scheduler consumes it. The steering probe needs exactly that
pre-scheduler boundary: compute positive/negative minimal-pair noise residuals,
derive a direction, subtract it from the main guided residual, and then call
`scheduler.step(...)`. Subclassing would still require replacing the whole
`__call__` body, so a focused copied loop is the most auditable MVP path. This
is intentionally a probe implementation, not a reusable production pipeline.
