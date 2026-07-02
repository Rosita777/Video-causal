# Research Notes: Video Concept Erasure and Causal Footprint

Updated: 2026-07-02

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

## Evaluation Protocol v0

The current evaluation plan is hybrid:

1. Use a clean-source gate before any erasure method is judged.
2. Use a video-capable MLLM for structured first-pass scoring over all generated videos.
3. Human-calibrate a 15-20 percent subset covering all causal mechanisms and baselines.
4. Human-adjudicate all strict-leak candidates, unclear temporal/causal cases, and figure-selected examples.

The required annotation dimensions are:

- `target_presence_score`: direct evidence of `E(C)`, scored 0-3.
- `footprint_presence_score`: causal footprint `F(C)`, scored 0-3.
- `quality_score`: whether the video is usable for judgment, scored 0-3.
- `scene_fidelity_score`: whether non-target scene content is preserved, scored 0-3.
- `target_time`, `footprint_time`, `alternative_cause_visible`, and `temporal_order_valid`.

## Data Construction Protocol v0

The benchmark should be described as taxonomy-driven causal pair construction, not hand-written prompt collection. A valid `C -> F(C)` pair must satisfy:

- an explicit causal mechanism from `C` to `F(C)`;
- counterfactual dependence, meaning `F(C)` should disappear or become much less likely under `do(not C)`;
- temporal asymmetry, meaning the footprint cannot precede the source event;
- visual observability, meaning the footprint is visible in video frames.

Candidate pairs should also record:

- `exclusivity_score`: whether the footprint strongly implies the source event;
- `counterfactual_clarity`: whether the no-source counterfactual is visually clear;
- `generatability_score`: whether current T2V models can generate a clean valid reference;
- `erasure_targetability`: whether the source concept can be named and passed to erasure baselines.

The main benchmark should prioritize high-exclusivity, high-counterfactual-clarity pairs and maintain mechanism balance so the benchmark is not dominated by water/ripple examples.

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

## Solution-Side Literature Snapshot

The solution literature splits into three layers. The first layer is classic concept erasure: suppress a target concept in text-to-image or text-to-video generation. The second layer is video object/effect removal: remove an object from an existing video and also erase its shadows, reflections, deformations, or interactions. The third layer is counterfactual video generation: explicitly reason about what the video should have looked like under an intervention.

### 1. Concept-erasure methods suppress targets, not causal consequences

- **VideoEraser** (`VideoEraser: Concept Erasure in Text-to-Video Diffusion Models`, EMNLP 2025 / arXiv 2508.15314): a training-free T2V concept-erasure framework with Selective Prompt Embedding Adjustment and Adversarial-Resilient Noise Guidance. It is a strong baseline for target suppression, but its task definition is still "prevent the concept from appearing"; it does not explicitly model downstream footprints such as ripples, cracks, or deformations.
- **SAFREE** (`Training-Free and Adaptive Guard for Safe Text-to-Image and Video Generation`, ICLR 2025 / arXiv 2410.12761): training-free filtering in text embedding and visual latent spaces. It is useful as a general inference-time erasure guard, but its unsafe-concept subspace is not causal-chain aware.
- **T2VUnlearning** (`A Concept Erasing Method for Text-to-Video Diffusion Models`, arXiv 2505.17550): T2V unlearning with negative guidance, localization, and preservation regularization. It is relevant for training-based target suppression, but the public repo mainly exposes placeholders/checkpoints rather than a fully reproducible training pipeline.
- **CLEAR** (`Where Concept Erasure Should Occur: Concept-Layer Alignment in Text-to-Video Diffusion Models`, ICML 2026 / arXiv 2605.25941): identifies layers where concept and non-target signals are more separable, then erases at those depths. This is important for us because causal footprint erasure may require suppressing a *compound causal set* rather than a single noun.

Implication for our paper: these methods are not "wrong" because they leave footprints; rather, their objective does not ask them to enforce counterfactual causal consistency. Our benchmark exposes that missing objective.

### 2. Object/effect removal directly studies "object plus consequence"

- **ROSE** (`Remove Objects with Side Effects in Videos`, NeurIPS 2025 / arXiv 2508.18633): explicitly models object-induced side effects such as shadows, reflections, light, translucency, and mirrors, using synthetic paired data and side-effect masks. This is close in spirit, but mostly focuses on appearance-side effects rather than broader physical causal chains like collisions or traces.
- **EffectErase** (`Joint Video Object Removal and Insertion for High-Quality Effect Erasing`, CVPR 2026 / arXiv 2603.19224): introduces VOR, a 60K paired-video dataset where target-present videos have object effects and target-absent videos remove both object and effect. The method uses reciprocal insertion/removal learning and task-aware region guidance.
- **GenEraser** (`Generalizable Video Object Removal via Balanced Text-Mask Guidance and Decoupled Locator-Preserver`, arXiv 2605.30045): explicitly addresses target objects plus physical effects such as smoke, reflections, light, and ripples. Its key idea is combining text guidance and mask guidance, plus a locator/preserver split to balance semantic generalization and pixel preservation.
- **VOID** (`Video Object and Interaction Deletion`, ECCV 2026 / arXiv 2604.02296): the closest work to our causal-footprint framing. It removes an object from an existing video and also changes downstream physical interactions, e.g. objects falling, collisions, or altered trajectories. It uses counterfactual paired data from Kubric/HUMOTO and a VLM-based reasoning pipeline to identify causally affected regions, encoded as a quadmask for a video diffusion model.

Implication for our method: the strongest design signal is not "stronger negative prompt"; it is **affected-region / affected-concept discovery followed by counterfactual generation**. However, those object-removal papers usually assume an input video plus object masks. Our setting is different: text-to-video concept erasure, where the model generates the video and the erasure method must suppress both target and footprint from the generative process.

### 3. Counterfactual video generation gives the right objective language

- **Causally Steered Diffusion for Automated Video Counterfactual Generation** (arXiv 2506.14404): treats video editing as counterfactual generation. It uses an assumed causal graph and VLM feedback to optimize prompts for causal effectiveness, minimality, video quality, and temporal consistency.
- **Counterfactual World Models via Digital Twin-conditioned Video Diffusion** (arXiv 2511.17481): argues that raw pixel/latent video models are entangled, so targeted interventions are difficult. It constructs structured "digital twin" scene representations, lets an LLM reason about intervention propagation, and conditions video diffusion on the modified representation.

Implication for our paper: our problem should be written as `do(not C)` erasure, not just `remove token C`. The correct success criterion is: direct evidence of `C` is absent, downstream footprint `F(C)` is absent or replaced by a plausible no-cause alternative, and unrelated scene content remains preserved.

## Candidate Method Directions

### Direction A: Causal-set erasure as a stronger training-free baseline

Use benchmark metadata or an LLM/VLM parser to expand the erased concept from a single target `C` to a causal set:

```text
S(C) = {target concept C, expected footprint F(C), common visual synonyms of F(C)}
```

Then apply prompt-embedding and latent-space suppression to both `C` and `F(C)`. This can be implemented quickly on top of the current negative-prompt, SAFREE-style, and VideoEraser-style adapters.

Pros:
- easy to implement on CogVideoX/ZeroScope/Wan;
- directly tests whether footprint-aware erasure reduces leakage;
- strong ablation story: target-only suppression vs target-plus-footprint suppression.

Cons:
- may over-suppress legitimate background patterns, e.g. natural ripples or pre-existing cracks;
- still lacks explicit spatial/temporal localization;
- reviewers may see it as a better baseline rather than a full method unless paired with causal reasoning.

### Direction B: VLM-planned causal footprint suppression

Before generation or during post-hoc editing, ask a VLM/LLM to predict what scene regions/effects should change under `do(not C)`. The output is a structured causal plan:

```json
{
  "target": "stone",
  "footprint": "circular ripples",
  "affected_regions": ["impact center", "expanding ripple rings"],
  "counterfactual_replacement": "calm water surface with only background motion"
}
```

The plan can drive either:
- semantic negative guidance (`no stone, no impact ripples`);
- attention/latent suppression for footprint tokens;
- mask-based video inpainting if a generated clean video is available.

Pros:
- aligns with VOID's VLM-based affected-region reasoning, but adapts it to T2V concept erasure;
- produces interpretable method artifacts useful for paper figures;
- can support mechanism-specific behavior without hand-coded templates.

Cons:
- requires robust VLM planning and may add API/model dependence;
- mask generation for abstract footprints is hard;
- needs careful evaluation to show gains are not just prompt engineering.

### Direction C: Counterfactual adapter training from synthetic pairs

Train a lightweight adapter/LoRA on paired examples:

```text
source: target present + footprint present
counterfactual: target absent + footprint absent / plausible background
```

The adapter learns not only object removal but causal-consequence removal. Existing object/effect-removal work suggests paired data is powerful, but our pairs must cover benchmark mechanisms beyond shadows/reflections: fluid impact, fracture, traces, elastic deformation, field-mediated motion, and particle dispersion.

Pros:
- closest to a principled solution;
- can borrow ideas from EffectErase, ROSE, GenEraser, and VOID;
- paper contribution can be stronger if the adapter improves over training-free baselines.

Cons:
- data generation/training cost is high;
- synthetic-to-real and T2V-backbone transfer are risks;
- needs more engineering than the current benchmark timeline.

### Direction D: Post-generation counterfactual repair

Generate with a normal erasure baseline, detect cases where `target_visible=no` and `footprint_visible=yes`, then repair the footprint using video inpainting / object-effect removal. This treats causal-footprint removal as a second-stage correction.

Pros:
- cleanly uses our evaluator as a trigger;
- compatible with existing VOID/ROSE/EffectErase-style tools;
- useful as a practical system.

Cons:
- less elegant for concept erasure because it relies on post-hoc editing;
- needs masks or region proposals for footprints;
- may be viewed as a pipeline rather than a model-level erasure method.

## Current Recommendation

The most practical next method path is a two-step ladder:

1. **Footprint-aware training-free erasure** as a fast first method/strong ablation. Extend our current adapters so the erased concept set includes both the target and the expected footprint. Evaluate whether leakage drops without excessive target leakage or quality collapse.
2. **VLM-planned counterfactual erasure** as the main method candidate. Use a VLM/LLM to infer affected causal footprint descriptors and plausible counterfactual replacements, then feed those into guidance or inpainting. This is more novel and maps naturally to our benchmark claim.

The paper can then position itself clearly:

- Existing T2V erasure methods optimize target absence.
- Object/effect removal shows that object consequences matter, but assumes an input video/mask and often focuses on object-level editing.
- We define and evaluate causal-footprint leakage for T2V concept erasure, then propose erasure under `do(not C)`: suppress both the target and the causally dependent footprint while preserving unrelated content.

## Opus-4.6 Method Discussion Notes, 2026-07-02

We asked Opus-4.6 to critique the solution space after the ZeroScope closure and Wan clean-source gate. Its strongest recommendations were:

1. Do not present simple target-plus-footprint negative prompting as the main method. It is a useful ablation, but reviewers are likely to call it prompt engineering.
2. A stronger method should be mechanistically grounded: existing erasure suppresses target appearance, while causal-footprint leakage persists because the model's denoising trajectory still contains the learned causal consequence of the prompt.
3. Cross-attention intervention is attractive but too expensive for the first implementation across CogVideoX, ZeroScope, and Wan because their attention layouts differ substantially.
4. The most practical MVP is **Counterfactual Latent Steering (CLS)**: construct a counterfactual prompt under `do(not C)`, run both the factual and counterfactual conditioning through the denoising step, estimate a causal direction in noise-prediction space, and steer the prediction away from that direction.

The method sketch is:

```text
epsilon_full = noise_pred(z_t, prompt_with_target_and_footprint)
epsilon_cf   = noise_pred(z_t, counterfactual_prompt_under_do_not_C)
d_t          = epsilon_full - epsilon_cf
epsilon_out  = epsilon_full - lambda(t) * normalized(d_t)
```

The counterfactual prompt should not merely delete the target word. It should describe the minimally changed no-cause world:

```text
stone hits water -> calm water surface with no impact point or expanding rings
ball hits net    -> still net hanging naturally with no inward deformation
shoe steps sand  -> smooth wet sand with no fresh footprint
```

This direction is more defensible than causal-set negative prompting because the intervention happens in the diffusion denoising prediction rather than only in text. It also gives a clean ablation ladder:

- target-only baseline;
- target-plus-footprint negative prompt;
- counterfactual prompt only;
- CLS with fixed strength;
- CLS with timestep schedule;
- CLS with VLM-planned counterfactual prompt.

First implementation target should be one backbone, preferably ZeroScope or Wan, because the v2 evaluation slice is already closed or close to closed and generation cost is manageable. If CLS shows a reduction in strict leakage without large quality collapse, port the same wrapper to the other backbone through a shared `CounterfactualEraser` interface.

## Broader Method Search Stance, 2026-07-02

Do not freeze the method yet. The benchmark/results branch can keep running in the background, but the main intellectual work now shifts to the solution side. The method must be allowed to differ across CogVideoX, ZeroScope, and Wan if their pipeline internals make a unified implementation unnatural. The paper can still present one conceptual framework, with model-specific instantiations.

Current method search should track four technical families:

### 1. Reason-before-edit video methods

Recent video editing work is moving from "direct instruction -> edited video" to "see/reason/edit" or "temporal reasoning before editing".

- **VideoCoF** (`Unified Video Editing with Temporal Reasoner`, CVPR 2026 Highlight / arXiv 2512.07469): predicts reasoning tokens or edit-region latents before generating target video tokens. It is valuable because our causal footprint is exactly an edit-region / affected-consequence discovery problem.
- **ChronoEdit** (`Towards Temporal Reasoning for Image Editing and World Simulation`, arXiv 2510.04290): reframes editing as video generation and uses temporal reasoning tokens to constrain physically plausible transformations. Its framing is useful for our paper language: causal-footprint erasure is not just visual deletion, it is world-consistent counterfactual editing.
- **IF-Edit** (`Are Image-to-Video Models Good Zero-Shot Image Editors?`, arXiv 2511.19435): uses chain-of-thought prompt enhancement, temporal latent dropout, and self-consistent post-refinement. Its engineering lesson is that video diffusion priors can help reasoning-centric edits, but lightweight inference-time modules may be enough for a first version.

Possible adaptation:

```text
source prompt + target concept
  -> causal reasoning tokens / affected footprint description
  -> model-specific erasure/editing step
```

### 2. Latent / denoising-level steering

This includes CLS-style methods, optimal latent trajectory methods, and diffusion self-correction. The key value is that the intervention happens below the prompt string, so it is easier to defend as more than prompt engineering.

- **Causally Steered Diffusion for Video Counterfactuals** (arXiv 2506.14404): prompt-level causal steering with a VLM and an assumed causal graph; useful as a black-box baseline and objective reference.
- **Self-correcting LLM-controlled Diffusion Models** (CVPR 2024 / arXiv 2311.16090): iteratively detects misalignment and applies latent-space operations. This supports a closed-loop variant where our evaluator detects footprint leakage and triggers a repair.
- **Reflect-DiT / reflection-style inference-time scaling**: VLM critique or reflection is fed into subsequent generation attempts. For us, a reflection loop could use "target absent but footprint still present" as the critique.
- **Optimal latent trajectory / prompt embedding control methods**: useful for understanding how to steer without training, but likely insufficient alone for video causal chains.

Possible adaptation:

```text
factual denoising trajectory
counterfactual denoising trajectory
  -> estimate causal/footprint direction
  -> steer latent/noise prediction with timestep schedule
```

### 3. Object/effect removal and affected-region reasoning

These works often assume an input video and masks, but they are the closest analogues to "remove cause plus consequence".

- **VOID**: VLM-based causal-region reasoning and counterfactual paired data for deleting an object and its physical interactions.
- **ROSE / EffectErase / GenEraser**: object removal with side effects, physical effects, or balanced text-mask guidance.

Possible adaptation:

```text
generated clean video or erased video
  -> detect target/footprint regions
  -> repair only affected footprint regions
```

This may become a second-stage system rather than the main T2V erasure method, but it is useful for upper-bound experiments.

### 4. Planner / critic / reflection methods

ReasonEdit, Inline Critic, and similar editing systems point toward a "planner + editor + critic" architecture. This is relevant because our benchmark already has an evaluator that can diagnose:

```text
target_visible = no
footprint_visible = yes
footprint_match = yes
```

That diagnosis can be turned into method feedback:

```text
The target is removed, but the downstream footprint remains.
Revise the counterfactual plan so the footprint is replaced by [background state].
```

This family is especially attractive if direct latent hooks become too brittle across backbones.

## Open Method Hypotheses

Keep these as hypotheses until we test small prototypes:

1. **H1: Footprint-aware prompting alone will reduce strict leakage but increase over-erasure.** It is important as a strong baseline, not as the final method.
2. **H2: Counterfactual latent/denoising steering will reduce leakage more cleanly than negative prompting, because it compares factual and no-cause trajectories.**
3. **H3: Reasoning-token or planner-based methods are the most novel, but may require training or model internals that are hard to reproduce across all three backbones.**
4. **H4: A critic/reflection loop may be easier to unify across models than attention hooks, because it treats each generator as a black box.**
5. **H5: Different model-specific instantiations are acceptable if they share the same causal objective and evaluation protocol.**

## Method-Side Survey Scope, Opus-4.6 Discussion, 2026-07-02

We asked Opus-4.6 to help delimit the solution-side survey before committing to a method. The important correction from the second round is that our method survey should not be centered only on weight-level concept erasure or model editing. The solution problem is closer to **counterfactual visual editing**:

```text
Given a generated video containing cause C and footprint F(C),
produce a counterfactual video under do(not C),
where direct evidence of C and the causally dependent footprint F(C) disappear or become a plausible no-cause background,
while unrelated scene content and temporal coherence are preserved.
```

### Survey Tier 1: Must Read Deeply

1. **Object/effect removal and video inpainting with side effects**
   - Examples: VOID, ROSE, EffectErase, GenEraser, ProPainter-style video inpainting.
   - Why: these works directly study removal of objects plus effects, which is the closest analogue to target-plus-footprint erasure.
   - Key questions:
     - Does the method explicitly handle downstream effects or only fill object masks?
     - Does it require an input video, object mask, side-effect mask, paired data, or optical flow?
     - Can it handle non-rigid or abstract footprints such as ripples, cracks, traces, smoke, deformation, or particle spread?
     - Is the method training-free, fine-tuned, or trained from paired counterfactual data?

2. **Counterfactual visual/video generation**
   - Examples: causally steered diffusion, digital-twin-conditioned counterfactual video generation, counterfactual image editing.
   - Why: gives the right objective language for `do(not C)` rather than token suppression.
   - Key questions:
     - How is the counterfactual state represented: text prompt, structured scene graph, mask, latent trajectory, or learned world state?
     - Does the method reason about causal propagation, or only local object removal?
     - How does it measure minimality and preservation?

3. **Reason-before-edit / planner-critic video editing**
   - Examples: VideoCoF, ChronoEdit, IF-Edit, ReasonEdit / reflection-style editing.
   - Why: our task requires discovering what should change after removing the cause. A planner can output affected footprints, replacement states, and edit constraints before generation.
   - Key questions:
     - Does the method use reasoning tokens, chain-of-thought plans, VLM critiques, or iterative reflection?
     - Is the planner grounded in visual evidence or only text?
     - Can the planner identify causal consequences rather than just target regions?
     - Can it be used as a black-box wrapper across CogVideoX, ZeroScope, and Wan?

4. **Instruction-guided video editing and temporal consistency mechanisms**
   - Examples: TokenFlow, FateZero, Video-P2P, InsV2V, AnyV2V, flow-guided propagation.
   - Why: even if they do not solve causal erasure, they provide mechanisms for preserving non-edited content over time.
   - Key questions:
     - What mechanism propagates edits across frames: attention, flow, latent warp, keyframe inversion, or feature matching?
     - Can that propagation be inverted or redirected to remove footprints?
     - Does the method require DDIM inversion or access to the original video?

### Survey Tier 2: Mechanistic Background / Selective Reading

1. **Text/image/video concept erasure**
   - Examples: ESD, UCE, MACE, Forget-Me-Not, SAFREE, VideoEraser, T2VUnlearning, CLEAR.
   - Role: establishes the current target-erasure objective and why it misses causal footprints.
   - We should read enough to explain why existing objectives are target-presence based, but not let this dominate the solution survey.

2. **Diffusion causal tracing, attribution, and activation/attention editing**
   - Examples: diffusion lens, activation patching adaptations, Prompt-to-Prompt, MasaCtrl, attention control.
   - Role: useful if we implement attention/latent steering or need mechanistic diagnostics.
   - Main question: can these tools localize footprint-causing signals across denoising timesteps and frames?

3. **Localized model editing / LoRA surgery / task arithmetic**
   - Role: background if we later train an adapter, but not the primary solution unless we move to weight-level editing.

### Survey Tier 3: Background Only

- General machine unlearning not specific to visual generation.
- Post-hoc safety classifiers and NSFW filters.
- Generic video generation metrics not tied to editing/causal consistency.
- NLP-only counterfactual text generation.

### Required Survey Tables

The method survey should produce at least two tables.

**Table 1: Related Work Capability Matrix**

Columns:

```text
method
year_or_venue
modality: image / video / 3D / T2V
task_type: concept erasure / object removal / effect removal / counterfactual edit / instruction edit
input_requirement: text only / input video / mask / flow / paired data / VLM plan
target_granularity: object / effect / attribute / region / causal chain
causal_propagation: explicit / implicit / none
temporal_mechanism: attention / flow / latent warp / recurrent / none
training_requirement: training-free / fine-tune / full training
backbone_dependence: black-box / scheduler-level / attention-hook / model-specific / weights
handles_footprints_like_ours: yes / partial / no
open_source_status
main_limitation_for_our_task
```

**Table 2: Method Route Decision Matrix**

Columns:

```text
route
core idea
novelty risk
engineering risk
cross-backbone portability
expected reviewer objection
first prototype model
minimal experiment size
success criterion
```

Candidate routes:

```text
causal-set negative prompt
counterfactual latent/denoising steering
VLM-planned guidance
planner-critic iterative refinement
post-generation object/effect repair
adapter training from synthetic counterfactual pairs
```

### Boundary Decision

The survey should be framed as method scouting for **counterfactual causal erasure in generated videos**, not as a general concept-erasure survey. Concept erasure remains essential related work and baseline context, but the solution method should borrow heavily from object/effect removal, counterfactual visual editing, and reasoning-augmented video editing.

## Immediate Method Experiments After Baseline Closure

1. Implement a `causal_set_negative_prompt` baseline:
   - negative prompt contains target plus expected footprint phrase;
   - example: `stone, pebble, circular ripples, splash, impact rings`.
2. Implement a `causal_set_embedding_suppression` adapter:
   - construct concept embeddings for target and footprint separately;
   - suppress both in prompt/latent guidance;
   - compare against target-only SAFREE / VideoEraser proxies.
3. Add one VLM planner dry run:
   - input: source prompt, target, expected footprint;
   - output: footprint synonyms, affected-region textual plan, counterfactual replacement phrase;
   - no video generation at first, only inspect plan quality on 20 cases.
4. If the training-free ladder is promising, prototype a post-generation repair branch using an open object/effect removal model, likely ROSE or VOID, only on a small subset where masks/affected regions are easy.

## Solution-Side Reference Links

- VideoEraser: https://arxiv.org/abs/2508.15314
- SAFREE: https://arxiv.org/abs/2410.12761
- T2VUnlearning: https://arxiv.org/abs/2505.17550
- CLEAR / Concept-Layer Alignment: https://arxiv.org/abs/2605.25941
- ROSE: https://arxiv.org/abs/2508.18633
- EffectErase: https://arxiv.org/abs/2603.19224
- GenEraser: https://arxiv.org/abs/2605.30045
- VOID: https://arxiv.org/abs/2604.02296
- Causally Steered Diffusion: https://arxiv.org/abs/2506.14404
