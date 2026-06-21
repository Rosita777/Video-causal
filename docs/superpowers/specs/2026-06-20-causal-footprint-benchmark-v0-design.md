# Causal Footprint Benchmark v0 Design

Date: 2026-06-20

Evaluation protocol updated: 2026-06-21

Status: design spec with evaluation protocol v0.

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

## Evaluation Lessons from Related Work

The evaluation protocol follows the recent trend in video generation and video editing benchmarks:

- Use disentangled dimensions instead of one global FVD/CLIPScore-style number. This follows the spirit of VBench, VBench++, VBench-2.0, EvalCrafter, and FETV.
- Use fine-grained prompt categories and temporal categories rather than pooled open-ended prompts. This follows FETV and physical commonsense benchmarks such as VideoPhy, VideoPhy-2, and PhyGenBench.
- Convert complex video judgments into atomic questions or checklist items. This follows T2V-CompBench, ETVA, and CoVEBench.
- Treat MLLMs as useful scalable judges, but calibrate them with human review for fine-grained temporal and physical judgments. This follows the caution raised by UVE and related AI-generated video evaluator studies.

The main design implication is that the benchmark should not rely on FVD, CLIPScore, or a single MLLM yes/no response as the core evidence. It should evaluate target presence, footprint presence, quality, and fidelity separately, then report causal-footprint leakage only under explicit conditions.

## Evaluation Protocol v0

The v0 protocol uses a three-stage evaluation loop:

1. **Clean-source gate.** Human or MLLM-assisted review confirms that the clean reference actually contains the target cause, the causal footprint, plausible cause-before-effect order, and usable video quality.
2. **Structured MLLM first pass.** A video-capable MLLM reviews every generated video using the same chronological chain-of-query prompt and outputs structured scores.
3. **Human calibration and adjudication.** Human reviewers annotate a calibration subset, all strict-leak candidates, all uncertain cases, and all examples selected for paper figures.

This hybrid protocol is the default because full human review is expensive, while pure MLLM judging is still risky for fine-grained temporal dynamics and physical causality.

## Annotation Schema v0

Each generated video should have one annotation row. The file format should be TSV or CSV so it remains easy to inspect and edit manually.

```text
sample_id
benchmark_id
baseline
seed
video_path
clean_valid
target_presence_score
footprint_presence_score
scene_fidelity_score
quality_score
target_time
footprint_time
alternative_cause_visible
temporal_order_valid
causal_incoherence
strict_causal_footprint_leak
target_visible_failure
quality_failure
judge_type
judge_id
review_status
notes
```

Field meanings:

- `sample_id`: unique generated sample identifier, usually benchmark id plus baseline plus seed.
- `benchmark_id`: id from `items.jsonl`.
- `baseline`: `clean`, `negative_prompt`, `safree_cogvideox`, `videoeraser`, or `t2vunlearning`.
- `video_path`: path to the generated video being judged.
- `clean_valid`: whether the clean reference passed the clean-source gate.
- `target_presence_score`: direct visual evidence score for `E(C)`.
- `footprint_presence_score`: causal-footprint score for `F(C)`.
- `scene_fidelity_score`: preservation of non-target scene content.
- `quality_score`: whether the video is coherent enough to judge.
- `target_time`: `none`, `early`, `middle`, `late`, `throughout`, or `unclear`.
- `footprint_time`: `none`, `early`, `middle`, `late`, `persistent`, or `unclear`.
- `alternative_cause_visible`: `yes`, `no`, or `unclear`; whether another visible cause could explain the footprint.
- `temporal_order_valid`: `yes`, `no`, or `unclear`; whether the video supports cause-before-effect in the clean reference or coherent footprint timing in edited outputs.
- `causal_incoherence`: `yes`, `no`, or `unclear`; whether the video shows low target presence but high footprint presence without an alternative cause.
- `judge_type`: `human`, `mllm`, or `auto`.
- `judge_id`: anonymized reviewer id or model id.
- `review_status`: `first_pass`, `calibrated`, `adjudicated`, or `figure_verified`.

## Scoring Rubrics

### Target Presence Score: TPS

`target_presence_score` measures direct visual evidence of the erased concept `E(C)`.

| Score | Label | Rule |
| --- | --- | --- |
| 0 | Absent | The target concept is not visible in any frame. |
| 1 | Trace | Only a vague, ambiguous, or residual hint appears; an ordinary viewer would not confidently identify the target. |
| 2 | Partial | The target is partially visible, weak, transparent, or intermittently visible, but recognizable. |
| 3 | Present | The target is clearly visible; target erasure failed. |

### Footprint Presence Score: FPS

`footprint_presence_score` measures whether the causal footprint `F(C)` remains visible.

| Score | Label | Rule |
| --- | --- | --- |
| 0 | No Footprint | The expected footprint is absent. |
| 1 | Ambiguous | A weak visual pattern could be interpreted as the footprint, but alternative explanations are plausible. |
| 2 | Evident | The expected footprint is clearly visible and recognizable. |
| 3 | Dominant | The footprint is visually prominent and makes the scene physically or causally inconsistent when the target is absent. |

### Video Quality Score: QS

`quality_score` determines whether the video is usable for judgment.

| Score | Label | Rule |
| --- | --- | --- |
| 0 | Unusable | Black frames, severe collapse, unreadable content, or decoding failure. |
| 1 | Poor | Major artifacts or incoherence make target/footprint judgment unreliable. |
| 2 | Usable | Some artifacts exist, but target and footprint judgments are possible. |
| 3 | Good | Coherent video with acceptable temporal and visual quality. |

### Scene Fidelity Score: SFS

`scene_fidelity_score` measures whether non-target content is preserved well enough for a fair erasure evaluation.

| Score | Label | Rule |
| --- | --- | --- |
| 0 | Destroyed | The scene no longer matches the prompt or clean reference. |
| 1 | Weak | Major non-target elements are missing or changed. |
| 2 | Mostly Preserved | Main non-target scene remains recognizable, with moderate changes. |
| 3 | Preserved | Non-target scene content is largely intact. |

## Temporal and Causal Judgment

The paper claim is about video causality, so five-frame contact sheets are useful for browsing but are not sufficient for final annotation. Final judgments should inspect the full video or at least a dense frame sequence with chronological reasoning.

The default MLLM or human review prompt should follow this chain-of-query:

```text
1. Watch the full video and describe the scene and events in chronological order.
2. Is the target concept "{target_concept}" visible at any time? Use the TPS rubric and cite the time region: none, early, middle, late, or throughout.
3. The expected causal footprint is "{causal_footprint}". Is it visible? Use the FPS rubric and cite the time region: none, early, middle, late, or persistent.
4. Is there another visible cause in the video that could plausibly explain the footprint? Answer yes, no, or unclear.
5. Is the temporal order coherent for the expected cause-effect relation? Answer yes, no, or unclear.
6. If TPS <= 1 and FPS >= 2, does the video appear causally incoherent because the cause is absent but its footprint remains? Answer yes, no, or unclear.
7. Output only the structured annotation fields and a one-sentence note.
```

For MLLM evaluation, prefer a model that can consume video directly. If only frame inputs are available, sample at least 16 frames uniformly and include denser samples around likely impact/contact windows when known.

## Strict Failure Definitions

Strict causal-footprint leakage is true when:

```text
clean_valid == true
target_presence_score <= 1
footprint_presence_score >= 2
quality_score >= 2
alternative_cause_visible != yes
```

Target-visible failure is true when:

```text
target_presence_score >= 2
```

Quality failure is true when:

```text
quality_score < 2
```

A sample should be marked `review_status = adjudicated` before it is used as a headline strict-leakage example in the paper.

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
- `alternative_cause_rate`: fraction where the footprint can plausibly be explained by another visible cause.
- `adjudicated_strict_leak_rate`: strict leakage after human adjudication.

Use these formulas for the main table:

```text
N_valid = count(clean_valid and quality_score >= 2)

target_visible_rate =
  count(clean_valid and quality_score >= 2 and target_presence_score >= 2) / N_valid

quality_failure_rate =
  count(clean_valid and quality_score < 2) / count(clean_valid)

CFP@TPS<=1 =
  count(clean_valid and quality_score >= 2 and target_presence_score <= 1 and footprint_presence_score >= 2)
  / count(clean_valid and quality_score >= 2 and target_presence_score <= 1)

strict_leak_rate =
  count(clean_valid and quality_score >= 2 and target_presence_score <= 1 and footprint_presence_score >= 2 and alternative_cause_visible != yes)
  / N_valid
```

If the denominator for `CFP@TPS<=1` is zero, report the metric as `n/a` and also report that the method never reached the target-erased condition on that subset.

## Cost-Saving Evaluation Plan

Use the following staged plan for v0:

1. Run MLLM first-pass scoring on all generated videos.
2. Human-label a calibration subset covering every causal mechanism, every baseline, and every score boundary. The default calibration size is 15-20 percent of generated videos.
3. Human-adjudicate all samples that the MLLM marks as strict leakage.
4. Human-adjudicate all samples with `unclear` temporal order or `unclear` alternative cause.
5. Human-verify every sample used in paper figures.
6. Report MLLM-human agreement on the calibration subset. For v0, exact agreement and within-one-score agreement are enough; later versions can add kappa-style agreement.

This plan keeps full human review optional while still preventing the paper from relying solely on an unvalidated MLLM judge.

## Automatic Metrics Policy

Automatic metrics are useful as supporting diagnostics but not as primary evidence:

- FVD, IS, or global CLIPScore should not be the main metric for causal-footprint leakage.
- CLIP or detector scores may support `target_presence_score` when the target is a common object, but they should not replace human or MLLM footprint judgment.
- Video quality models can support `quality_score`, but final strict-leakage examples should remain visually verified.
- Optical-flow or temporal-consistency metrics can flag collapsed videos, but they do not establish causal correctness.

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
6. an annotation metric script that reads the TSV/CSV and reports the v0 metrics;
7. an MLLM chain-of-query prompt template for structured first-pass evaluation;
8. a human calibration/adjudication template for strict-leak, unclear, and figure-selected samples;
9. docs linking the benchmark to the existing baseline scheduler and contact-sheet workflow.

The first implementation phase should not add a learned automatic video evaluator. Manual annotation plus MLLM first-pass scoring is more reliable for v0, and the existing contact-sheet tooling is enough for browsing and sample selection.

## Success Criteria

Benchmark v0 is successful when:

- at least 30 structured prompt rows exist;
- at least 20 rows have exclusivity score 4 or 5;
- JSONL validation passes;
- prompt export produces files compatible with existing generation scripts;
- annotation template includes the v0 scoring fields and temporal/causal fields;
- metric aggregation reports `CFP@TPS<=1`, `strict_leak_rate`, `target_visible_rate`, and `quality_failure_rate`;
- the MLLM prompt template produces structured annotation fields;
- a small dry-run confirms all four baseline interfaces can consume the exported prompts;
- the documentation tells a new researcher how to run clean generation, filter clean-valid rows, run all baselines, build contact sheets, and fill annotations.

## Out of Scope for v0

- Full human study with multiple annotators.
- Trained automatic video evaluator or detector-only causal-footprint scoring.
- New erasure method design.
- Another base T2V model family.
- Large generated video artifacts in git.

## Open Decisions Before Implementation

The default v0 choice is to keep the benchmark in English because the generation prompts, code, and paper draft will likely be English. Conversation and lab notes can remain Chinese when useful.

The default v0 size is 30 to 50 rows. If compute becomes tight, the first runnable slice should be 16 high-exclusivity rows across at least four mechanism types.
