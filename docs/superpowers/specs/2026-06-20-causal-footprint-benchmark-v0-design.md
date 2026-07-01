# Causal Footprint Benchmark v0 Design

Date: 2026-06-20

Status: design spec for review before implementation.

## Purpose

The project now has qualitative evidence that video concept erasure methods can weaken or remove the visible source concept while preserving downstream effects that imply the source event occurred. The benchmark should turn this observation into a systematic evaluation protocol before we design a new erasure method.

The benchmark is not a generic unsafe-concept erasure benchmark. It is a causal video benchmark for testing whether an erasure method produces the counterfactual video as if the erased concept never happened.

## Core Definitions

For each source video prompt, define:

- `C`: the erased source concept or event participant, such as `raindrop`, `pebble`, `baseball`, or `shoe`.
- `E(C)`: direct visual evidence of the source concept, such as the object body, impact object, or visible actor.
- `F(C)`: causal footprints caused by the source concept, such as ripples, splashes, cracks, footprints, net deformation, or material motion.

The central failure mode is:

```text
target presence is low, but causal footprint presence is high
```

This is different from ordinary incomplete erasure. We only count strict causal-footprint leakage when the target concept is weak or absent enough that reviewers cannot dismiss the case as direct target leakage.

## Research Claims Enabled by the Benchmark

The benchmark should support these claims, in order of strength:

1. Existing methods can remove direct visual evidence of a source concept while preserving causal consequences of that source.
2. This failure is measurable across multiple causal mechanisms, not only a single cherry-picked prompt.
3. Stronger prompt suppression or larger erasure strength does not automatically solve the problem, because the erased target and its downstream effects are different variables.
4. Video concept erasure should be evaluated as counterfactual causal editing, not only concept-word suppression.

The benchmark should avoid overclaiming that every baseline always fails. Weak or collapsed generations remain method outcomes, but strict causal-footprint conclusions require clean-source validity and target-erased evidence.

## Approach Options

### Option A: Small Curated Qualitative Set

Use 8 to 12 high-confidence prompts, run all baselines, and manually select strong examples for the paper.

Pros:
- Fast.
- Easy to visually inspect.
- Good for figures and early writing.

Cons:
- Easy for reviewers to call cherry-picking.
- Hard to separate base-model generation failures from erasure failures.

### Option B: Structured Benchmark v0

Create a JSONL benchmark with 30 to 50 prompts across causal mechanism categories, explicit metadata, clean-source gates, and baseline evaluation rules.

Pros:
- Scientific enough for a main contribution.
- Still feasible on the current 8-GPU node.
- Supports both quantitative tables and qualitative figures.

Cons:
- Requires careful clean-source filtering and annotation.
- Metrics need a small amount of manual validation.

### Option C: Full Human-Study Benchmark

Create a larger benchmark with hundreds of prompts, multiple annotators, agreement statistics, and human inference tests.

Pros:
- Strongest scientific evidence.
- Best defense against metric skepticism.

Cons:
- Too heavy for the immediate phase.
- Slows method development before the problem framing is stable.

## Chosen Design

Use Option B now, with a path to Option C later. Benchmark v0 should contain a structured prompt set, clean-source validation, baseline outputs, contact sheets, and annotation tables. The initial goal is 30 to 50 prompt rows, not hundreds.

The v0 benchmark should produce:

- a machine-readable prompt dataset;
- a clean-source validity table;
- a baseline output manifest for every method and prompt;
- contact sheets for human inspection;
- an annotation table separating target presence from causal-footprint persistence;
- aggregate metrics that condition on successful target erasure.

## Data Taxonomy

Each prompt belongs to one causal mechanism:

- `fluid_impact`: object or droplet causes splash, turbulence, or ripples.
- `surface_trace`: source contact leaves a persistent mark or indentation.
- `fracture_damage`: impact causes cracks, dents, or breakage.
- `elastic_deformation`: source contact bends, stretches, or displaces a material.
- `field_mediated`: source induces spatial reorganization without direct contact.
- `agent_or_object_response`: source event triggers another object or agent reaction.

Each prompt also has one temporal type:

- `synchronous`: source and effect are visible together.
- `delayed`: source appears first, effect follows.
- `persistent`: source may disappear, but the footprint remains.

The first benchmark version should prioritize high-exclusivity footprints where the effect strongly implies the source event. Examples include:

- raindrop -> crown splash and expanding ripples;
- pebble or stone -> outward water rings;
- baseball -> branching glass cracks;
- shoe -> footprint in wet sand;
- soccer ball -> net deformation;
- magnet -> iron filings forming field lines.

Lower-exclusivity prompts can remain in the exploratory pool but should not drive the main claim.

## Dataset Schema

Benchmark rows should be stored as JSONL. Each row should use these fields:

```json
{
  "id": "fluid_impact_raindrop_puddle_001",
  "prompt": "A realistic close-up video of a single raindrop hitting a shallow puddle, and a crown splash rises with ripples expanding outward.",
  "target_concept": "raindrop",
  "direct_visual_cues": ["visible falling droplet", "impact point"],
  "causal_footprints": [
    {
      "name": "crown splash",
      "description": "water rises upward around the impact point",
      "temporal_type": "synchronous"
    },
    {
      "name": "expanding ripples",
      "description": "circular water rings move outward after impact",
      "temporal_type": "delayed"
    }
  ],
  "mechanism_type": "fluid_impact",
  "temporal_type": "delayed",
  "exclusivity_score": 5,
  "counterfactual_prompt": "A realistic close-up video of a still shallow puddle with no raindrop impact, no crown splash, and no expanding ripples.",
  "erasure_prompt": "Remove the raindrop from the video.",
  "counterfactual_erasure_prompt": "Generate the video as if the raindrop never hit the puddle.",
  "notes": "High-confidence causal footprint candidate."
}
```

Required field constraints:

- `id` must be unique and stable.
- `prompt`, `target_concept`, `mechanism_type`, `temporal_type`, and `exclusivity_score` are required.
- `causal_footprints` must contain at least one footprint.
- `exclusivity_score` uses a 1 to 5 scale. The main benchmark subset should use rows with score 4 or 5.
- `counterfactual_prompt` must describe the scene under `do(not C)`, not just say "without C".

## Clean-Source Gate

A row can enter baseline evaluation only if its clean CogVideoX-2B sample passes these checks:

1. The target concept is visible or clearly implied before the effect.
2. The causal footprint is visible.
3. The temporal order supports cause before effect.
4. The scene quality is sufficient for human judgment.
5. The footprint is not already explainable as an unrelated background pattern.

Rows that fail the clean-source gate should remain in an exploratory log, but should not count against erasure methods.

## Baseline Evaluation Protocol

Run every clean-valid row through the same baseline set:

- Clean reference.
- Negative Prompt.
- SAFREE-CogVideoX.
- VideoEraser local CogVideoX reimplementation.
- T2VUnlearning local proxy/reimplementation.

T2VUnlearning should be reported as a faithful local proxy unless official training code and checkpoints become available. This is acceptable because the paper can still evaluate the behavior of the available method family without claiming official checkpoint parity.

For each prompt and baseline, save:

- generated video path;
- generation manifest;
- seed, steps, dtype, size, frame count, and model path;
- contact sheet with 5 evenly spaced frames;
- annotation row with target and footprint judgments.

## Annotation Schema

Each generated video should have one annotation row:

```text
sample_id
benchmark_id
baseline
seed
clean_valid
target_presence_score
footprint_presence_score
scene_fidelity_score
quality_score
strict_causal_footprint_leak
target_visible_failure
quality_failure
notes
```

Score meanings:

- `target_presence_score`: 0 to 3, where 0 means absent and 3 means clearly visible.
- `footprint_presence_score`: 0 to 3, where 0 means absent and 3 means clearly visible.
- `scene_fidelity_score`: 0 to 3, where 3 means non-target scene content is preserved.
- `quality_score`: 0 to 3, where 3 means usable video quality.

Strict causal-footprint leakage is true when:

```text
clean_valid == true
target_presence_score <= 1
footprint_presence_score >= 2
quality_score >= 2
```

Target-visible failure is true when:

```text
target_presence_score >= 2
```

Quality failure is true when:

```text
quality_score < 2
```

## Metrics

The primary metric is conditional footprint persistence:

```text
CFP@TPS<=1 = mean(footprint_presence_score >= 2 | clean_valid and target_presence_score <= 1 and quality_score >= 2)
```

Report together:

- `TPS`: target presence score distribution.
- `CFP`: causal footprint persistence distribution.
- `CFP@TPS<=1`: causal footprint leakage after target erasure.
- `strict_leak_rate`: fraction of clean-valid usable outputs with strict leakage.
- `target_visible_rate`: fraction where the target remains too visible.
- `quality_failure_rate`: fraction unusable because generation collapses or becomes incoherent.

For paper figures, select examples from rows satisfying strict leakage, not from arbitrary visual failures.

## Reviewer Objection Handling

Objection: "This is just incomplete erasure."

Response: The strict leakage subset requires low target presence but high footprint presence. We evaluate the downstream causal evidence after direct target evidence is weak or absent.

Objection: "The method could just erase the footprint too."

Response: The benchmark tests whether existing erasure methods understand counterfactual causal consequences. Users cannot be expected to enumerate every downstream effect, and a correct method should reason over the source event's causal graph.

Objection: "The footprints are subjective."

Response: Use high-exclusivity rows first, record explicit footprint descriptions, and add human agreement once the v0 protocol stabilizes.

Objection: "The problem is model-specific."

Response: v0 starts on CogVideoX-2B for controlled reproduction. The schema and metrics are model-agnostic, so later versions can add another T2V family.

## Implementation Scope for v0

The first implementation phase should add:

1. `benchmarks/causal_footprint_v0/items.jsonl`
2. `benchmarks/causal_footprint_v0/annotation_template.tsv`
3. `benchmarks/causal_footprint_v0/README.md`
4. a validator that checks JSONL schema constraints and uniqueness;
5. a prompt-export script that converts JSONL rows into the existing pipe-delimited prompt format used by the generation runners;
6. docs linking the benchmark to the existing baseline scheduler and contact-sheet workflow.

The first implementation phase should not add automatic vision scoring. Manual annotation is more reliable for v0, and the existing contact-sheet tooling is already enough for screening.

## Success Criteria

Benchmark v0 is successful when:

- at least 30 structured prompt rows exist;
- at least 20 rows have exclusivity score 4 or 5;
- JSONL validation passes;
- prompt export produces files compatible with existing generation scripts;
- a small dry-run confirms all four baseline interfaces can consume the exported prompts;
- the documentation tells a new researcher how to run clean generation, filter clean-valid rows, run all baselines, build contact sheets, and fill annotations.

## Out of Scope for v0

- Full human study with multiple annotators.
- Automatic detector or VLM scoring.
- New erasure method design.
- Another base T2V model family.
- Large generated video artifacts in git.

## Open Decisions Before Implementation

The default v0 choice is to keep the benchmark in English because the generation prompts, code, and paper draft will likely be English. Conversation and lab notes can remain Chinese when useful.

The default v0 size is 30 to 50 rows. If compute becomes tight, the first runnable slice should be 16 high-exclusivity rows across at least four mechanism types.
