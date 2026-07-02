# Method-Side Survey for Causal-Footprint Erasure

Updated: 2026-07-02

This document tracks solution-side related work for **counterfactual causal erasure in generated videos**. It is intentionally stricter than a generic related-work list. A paper should help us decide how to remove a target concept *and* its downstream footprint under `do(not C)`.

## Source Hygiene

We should not treat every arXiv/project-page claim as equally reliable. Use the following credibility levels:

- **A: strong source**: peer-reviewed venue page or official proceedings, ideally with official code/project page.
- **B: credible preprint**: arXiv plus project page or official repository, but no confirmed peer-reviewed venue yet.
- **C: weak but potentially useful**: arXiv only, no code/project, or unclear acceptance status.
- **D: unverified/noisy**: third-party summaries, social posts, or papers that cannot be located on arXiv/venue/project pages. Do not cite as evidence until upgraded.

When reading papers, record:

- whether the method actually handles **downstream effects**, not just target masks;
- whether it requires an **input video**, **object/effect masks**, **paired counterfactual data**, or **model internals**;
- whether it can transfer across CogVideoX, ZeroScope, and Wan, or is tied to one backbone;
- whether its evaluation measures **causal propagation / counterfactual consistency**, not only visual quality.

## Capability Matrix, First Pass

| Method | Source status | Task type | Input requirement | Causal propagation | Training requirement | Backbone dependence | Relevance to us | Notes / risk |
|---|---:|---|---|---|---|---|---|---|
| VOID: Video Object and Interaction Deletion | A/B: arXiv, official project, Netflix GitHub; project claims ECCV 2026 | Video object + interaction deletion | Input video, object/affected-region reasoning, counterfactual paired data | Explicit: physical interactions and downstream dynamics | Trained video diffusion model | Model-specific editing framework | Very high | Closest analogue to causal footprint; different domain because it edits an existing video rather than T2V concept generation. |
| ROSE: Remove Objects with Side Effects in Videos | B: arXiv, project, GitHub | Video object removal with side effects | Input video, object/effect localization, synthetic paired data | Explicit for side effects: shadows, reflections, light, translucency, mirror | Trained DiT/inpainting model | Inpainting/editing backbone | High | Strong for appearance side effects; less direct for collision, ripple, trace, or fracture chains. |
| EffectErase | A/B: arXiv, CVPR 2026 project/GitHub/openaccess | Video object removal and insertion for effect erasing | Input video, object masks, VOR paired data | Explicit for object effects: deformation, shadows, reflections | Trained on paired VOR data | Model-specific | High | Strong dataset/method reference; likely not directly usable as T2V erasure because it assumes masks/input video. |
| GenEraser | B/C: arXiv and project found; venue not yet confirmed | Generalizable video object/effect removal | Input video, mask + text guidance | Explicit/partial for effects such as smoke, reflections, light, ripples | Trained framework | Model-specific | High but verify carefully | Useful because it combines text and mask guidance; newer 2026 preprint, check code/reproducibility before relying on claims. |
| Causally Steered Diffusion for Video Counterfactuals | B: arXiv + GitHub | Counterfactual video editing | Existing video editing system as black box, causal graph, VLM feedback | Explicit counterfactual objective | Training-free / black-box prompt optimization | Black-box compatible | Very high | Useful for objective and planner/critic framing; may be prompt-level rather than strong latent intervention. |
| Counterfactual World Models via Digital Twin-conditioned Video Diffusion | C: arXiv only found | Counterfactual video/world simulation | Structured digital-twin scene representation + LLM reasoning | Explicit counterfactual propagation | Framework-level; likely requires scene extraction and video diffusion conditioning | Not directly plug-and-play | Medium-high | Good conceptual framing; engineering burden high for our current project. |
| VideoCoF | A/B: arXiv, project, GitHub; project claims CVPR 2026 Highlight | Reason-before-edit video editing | Input video + instruction; learns reasoning tokens/edit-region latents | Implicit/partial: reasons about edit regions and temporal alignment | Trained with video pairs | Model-specific | High | Key for "see -> reason -> edit"; could inspire causal-footprint planner or region-latent reasoning. |
| ChronoEdit | A/B: arXiv, NVIDIA project/GitHub; project claims ICLR 2026 | Temporal reasoning for image editing/world simulation | Input image/edit instruction, video-generation-as-reasoning | Implicit physical/temporal consistency | Model framework; released variants claimed | Model-specific | Medium-high | Strong conceptual support for temporal reasoning tokens; not directly video concept erasure. |
| IF-Edit | A/B: arXiv, CVPR 2026 openaccess found | Zero-shot image editing using I2V priors | Input image + instruction | Implicit physical/temporal reasoning | Tuning-free | Depends on I2V model | Medium | Good for lightweight reasoning/editing recipe; not a video erasure method. |
| RVEDiT / Reasoning to Align | C: arXiv only found | Reasoning Video Editing DiT | Input video + instruction, MLLM-distilled editing tokens | Implicit reasoning/localization | Trained DiT | Model-specific | Medium | Interesting but new; use cautiously until code/venue/reproducibility improves. |
| ProPainter | A: ICCV 2023, official code | Video inpainting | Input video + masks | None or implicit through propagation | Trained inpainting model | Editing model | Medium | Useful for temporal inpainting and preservation; not causal-aware. |
| TokenFlow | A: ICLR 2024, official code/project | Text-driven video editing consistency | Input video + target prompt | None; preserves feature correspondences | Training-free | Uses T2I diffusion features | Medium | Useful temporal consistency mechanism; can preserve too much if footprint should change. |
| FateZero | A: ICCV 2023, official code | Zero-shot text-based video editing | Input video + prompts, inversion/attention maps | None explicit | Training-free | Attention/inversion pipeline | Medium | Useful attention-fusion baseline; not effect-removal specific. |
| Video-P2P | A: CVPR 2024, project/GitHub | Video editing with cross-attention control | Input video + source/target prompts, inversion | None explicit | Optimization/inversion based | Attention-control pipeline | Medium | Useful if we later attempt attention-level intervention. |
| VideoEraser | A: EMNLP 2025 ACL page + arXiv | T2V concept erasure | Text prompt + target concept; no input video | No explicit footprint handling | Training-free | Plug-in to T2V backbones | Baseline / Tier 2 | Strong target-erasure baseline, but objective is target suppression. |
| SAFREE | A: ICLR 2025 + project | Safe T2I/T2V concept filtering | Text prompt / embedding and latent guard | No explicit footprint handling | Training-free | Broad diffusion compatibility | Baseline / Tier 2 | Good training-free concept guard; may over/under-filter footprints. |
| T2VUnlearning | B/C: arXiv/OpenReview + repo, but repo TODO indicates missing training/inference release | T2V concept erasure via unlearning | Model fine-tuning / checkpoints | No explicit footprint handling | Training-based | T2V-specific | Baseline / Tier 2 | Important baseline, but reproducibility is weak; treat our proxy carefully. |
| CLEAR / Concept-Layer Alignment | C/B: arXiv; claimed ICML 2026 via author/social/project-like sources, verify when proceedings are live | T2V concept erasure layer selection | Model internals and separability objective | No explicit footprint handling | Optimization / method-specific | Transformer-depth dependent | Tier 2 | Useful for "where to intervene"; not a causal-footprint solution by itself. |
| Concept attribution / activation erasure in diffusion | C/B mixed: arXiv/OpenReview/project depending on paper | Diffusion interpretability / concept localization | Model internals, activations, components | No direct video footprint, but useful diagnostics | Usually analysis or training-free edit | Model-internal | Tier 2 | Useful to justify latent/attention steering; not the central solution family. |

## Work That Needs Extra Verification

These are not rejected, but should not anchor claims until verified from primary sources:

- Any 2026 paper with only a project page and no arXiv/venue/proceedings entry.
- Papers discovered only from awesome-lists, AI-summary sites, ResearchGate, Twitter/X, or news articles.
- Repositories whose README promises code/model/data but most files are still TODO.
- Works whose title suggests "reasoning" but whose method is only a trained black-box editor without inspectable planning or causal propagation.

## Survey Questions Per Paper

For each Tier 1/2 paper, fill these fields after reading:

```text
1. What is the editing/erasure target?
2. Does the method remove downstream effects, or only the target object/concept?
3. What supervision is required: masks, flow, paired videos, synthetic engine, VLM labels, human labels?
4. Is the method generation-time, post-generation editing, or model-weight editing?
5. Is it training-free, fine-tuned, or fully trained?
6. Does it support video temporal consistency, and how?
7. Does it use reasoning/planning/critic feedback?
8. What would break if applied to our benchmark?
9. Could it be adapted to CogVideoX, ZeroScope, and Wan?
10. What would a reviewer expect us to compare against or cite?
```

## Method Route Decision Matrix, Draft

| Route | Core idea | Novelty risk | Engineering risk | Cross-backbone portability | Reviewer objection | First prototype |
|---|---|---:|---:|---:|---|---|
| Causal-set negative prompt | Add target and footprint terms to negative prompt | High | Low | High | "Just prompt engineering" | All backbones, small ablation |
| Counterfactual latent/denoising steering | Compare factual and counterfactual denoising predictions and steer away from causal direction | Medium | Medium | Medium | "Is this more than CFG variant?" | ZeroScope or Wan |
| VLM-planned guidance | VLM outputs affected footprint and no-cause replacement; guidance uses that plan | Medium | Medium | High if black-box | "Planner quality / API dependence" | Planner dry run on 50 cases |
| Planner-critic iterative refinement | Generate, evaluate target/footprint, revise plan/guidance, retry | Medium | Medium-high | High | "Expensive pipeline, not model method" | ZeroScope strict leakage subset |
| Post-generation effect repair | Use object/effect removal/inpainting to repair footprint after target erasure | Medium-high | High | Medium | "Two-stage patch, needs masks" | Small figure subset |
| Synthetic paired adapter | Train LoRA/adapter on factual/counterfactual pairs | Low if strong results, but data-heavy | Very high | Low-medium | "Data synthetic / scalability" | Not first |

## Current Reading Priority

1. VOID, ROSE, EffectErase, GenEraser: determine what "object plus effect" methods actually do and what assumptions they require.
2. Causally Steered Diffusion, Counterfactual World Models: extract objective language and evaluation criteria.
3. VideoCoF, ChronoEdit, IF-Edit, RVEDiT: understand reasoning/planner mechanisms and whether they can become black-box wrappers.
4. TokenFlow, FateZero, Video-P2P, ProPainter: borrow temporal consistency and localization mechanics.
5. VideoEraser, SAFREE, T2VUnlearning, CLEAR: position as target-erasure baselines and intervention-depth references.

## Verified Source Log, Batch 1

### Object / Effect Removal

**VOID: Video Object and Interaction Deletion**

- Primary sources checked: arXiv `2604.02296`, project page `void-model.github.io`, Netflix GitHub `netflix/void-model`.
- Credibility: **A/B**. arXiv and official repo are consistent; project/repo claim ECCV 2026, but final proceedings should still be checked when available.
- What it actually does: removes objects from existing videos and changes downstream physical interactions. It trains from counterfactual paired data generated with Kubric/HUMOTO. At inference, a VLM identifies affected regions and a video diffusion model generates physically plausible counterfactual outcomes.
- Relevance: **very high conceptual relevance**. It is the closest existing work to "remove cause plus causal consequence".
- Limitation for us: assumes input video/object removal setting with affected-region guidance. Our task is T2V concept erasure during generation, not post-hoc deletion from a given video.

**ROSE: Remove Objects with Side Effects in Videos**

- Primary sources checked: arXiv `2508.18633`, project page `rose2025-inpaint.github.io`, GitHub `Kunbyte-AI/ROSE`.
- Credibility: **B**. Strong arXiv/project/repo package; no confirmed top-tier proceedings found in this pass.
- What it actually does: video inpainting for object removal with side effects. It explicitly covers five side-effect categories: shadows, reflections, light, translucency, and mirror effects. It uses synthetic paired data from a rendering engine and predicts side-effect affected areas from paired differences.
- Relevance: **high**, especially for the idea of side-effect masks and paired factual/counterfactual supervision.
- Limitation for us: mostly appearance-side effects. Our benchmark includes broader physical footprints such as ripples, cracks, traces, elastic deformation, and particle dispersion.

**EffectErase**

- Primary sources checked: arXiv `2603.19224`, CVPR 2026 openaccess PDF, project page `henghuiding.com/EffectErase`, GitHub `FudanCVL/EffectErase`.
- Credibility: **A**. CVPR 2026 page/project/GitHub are consistent.
- What it actually does: introduces VOR, a 60K paired-video dataset with object-present/effect-present and object-absent/effect-absent counterpart videos plus object masks. The method learns removal and insertion reciprocally, with task-aware region guidance and insertion-removal consistency.
- Relevance: **high**. This is a strong dataset/method reference for paired counterfactual object-effect learning.
- Limitation for us: still assumes video object removal/insertion with masks and paired training data, not black-box T2V concept erasure.

**GenEraser**

- Primary sources checked: arXiv `2605.30045`, project page `cyqii.github.io/GenEraser.github.io/`.
- Credibility: **B/C**. arXiv and project exist, but venue/code/reproducibility need follow-up. Treat claims cautiously.
- What it claims: generalizable video object/effect removal with balanced text-mask guidance, a multi-conditional MoE, learnable deep CFG fusion, and a locator/preserver split. It explicitly mentions effects such as smoke, reflections, light, and ripples.
- Relevance: **high if verified**, because it combines textual guidance and masks for weakly correlated effects.
- Limitation for us: still input-video/mask oriented; latest preprint, so do not rely on it as central evidence until code/benchmarks are checked.

### Counterfactual Video Generation

**Causally Steered Diffusion for Automated Video Counterfactual Generation**

- Primary sources checked: arXiv `2506.14404`, GitHub `nysp78/counterfactual-video-generation`.
- Credibility: **B**. arXiv and code exist; no main-conference venue found in this pass.
- What it actually does: black-box framework that uses a VLM, an assumed causal graph, and prompt optimization to steer an underlying video editing system toward counterfactual outcomes. It evaluates causal effectiveness and minimality, not only visual quality.
- Relevance: **very high for objective framing**. It directly treats counterfactual video generation and causal relations.
- Limitation for us: appears prompt-level/black-box rather than a model-internal erasure method, so it may be a method inspiration or baseline rather than the final mechanism.

**Counterfactual World Models via Digital Twin-conditioned Video Diffusion**

- Primary sources checked: arXiv `2511.17481`.
- Credibility: **C/B**. arXiv exists; no code/venue found in this pass.
- What it claims: builds structured digital twins of scenes, uses LLMs to reason about intervention propagation over time, and conditions a video diffusion model on modified representations.
- Relevance: **medium-high conceptually**. It strongly supports separating reasoning about counterfactual dynamics from pixel synthesis.
- Limitation for us: requires explicit scene representation extraction and modified world-state conditioning, which is much heavier than our current T2V erasure setup.

### Reason-Before-Edit / Planner-Style Video Editing

**VideoCoF: Unified Video Editing with Temporal Reasoner**

- Primary sources checked: arXiv `2512.07469`, project page `videocof.github.io`, GitHub `knightyxp/VideoCoF`, CVPR 2026 supplemental/openaccess traces.
- Credibility: **A/B**. Multiple official sources; project/GitHub claim CVPR 2026 Highlight. Check final proceedings for citation metadata.
- What it actually does: enforces a "see, reason, then edit" process by predicting reasoning tokens / edit-region latents before target video tokens. It aims to avoid user masks while improving instruction-to-region alignment.
- Relevance: **high**. It is a strong template for discovering affected regions/latents before editing, which maps naturally to causal footprint discovery.
- Limitation for us: trained video editing model; not a plug-in T2V erasure wrapper.

**ChronoEdit**

- Primary sources checked: arXiv `2510.04290`, NVIDIA project page, GitHub `nv-tlabs/ChronoEdit`.
- Credibility: **A/B**. Strong institutional/project sources; GitHub claims ICLR 2026.
- What it actually does: reframes image editing as video generation and uses temporal reasoning tokens to imagine physically plausible editing trajectories, then drops reasoning tokens after early steps for efficiency.
- Relevance: **high conceptually**. It supports the idea that physically consistent editing needs an intermediate temporal reasoning stage.
- Limitation for us: image editing/world simulation, not video concept erasure; likely not directly reusable.

**IF-Edit**

- Primary sources checked: arXiv `2511.19435`, CVPR 2026 openaccess/search trace, author homepage.
- Credibility: **A/B**. arXiv and author page claim CVPR 2026; final citation should be checked.
- What it actually does: tuning-free image editing by repurposing image-to-video diffusion priors. Uses chain-of-thought prompt enhancement, temporal latent dropout, and self-consistent post-refinement.
- Relevance: **medium-high**. It is useful evidence that video diffusion priors can support reasoning-centric edits without full training.
- Limitation for us: image editing, not generated-video erasure; still closer to prompt/trajectory recipe than causal-footprint suppression.

**RVEDiT / Reasoning to Align**

- Primary sources checked: arXiv `2605.24674`.
- Credibility: **C/B**. arXiv exists, but no code/venue found in this pass.
- What it claims: DiT-native instruction video editing with MLLM-distilled editing tokens routed across shallow/deep blocks and attention alignment regularization.
- Relevance: **medium**. Interesting for "latent reasoning" and token routing.
- Limitation for us: trained model internals; too new to anchor our method unless stronger sources appear.

### Temporal Consistency / Inpainting Mechanics

**ProPainter**

- Primary sources checked: ICCV 2023 openaccess PDF, arXiv `2309.03897`, GitHub `sczhou/ProPainter`.
- Credibility: **A**.
- What it does: video inpainting using dual-domain propagation and mask-guided sparse video Transformer.
- Relevance: **medium**. Useful for post-generation repair or temporal inpainting mechanics.
- Limitation for us: not causal-aware and requires masks/input video.

**TokenFlow**

- Primary sources checked: ICLR 2024 proceedings, project page, GitHub `omerbt/TokenFlow`, arXiv `2307.10373`.
- Credibility: **A**.
- What it does: text-driven video editing by enforcing consistency in diffusion feature space through inter-frame correspondences; training-free.
- Relevance: **medium**. Useful for temporal consistency and feature propagation.
- Limitation for us: designed to preserve layout/motion, while our causal-footprint task sometimes requires changing downstream motion/effects.

**FateZero**

- Primary sources checked: ICCV 2023 / IEEE page, arXiv `2303.09535`, GitHub `ChenyangQiQi/FateZero`.
- Credibility: **A**.
- What it does: zero-shot text-based video editing using attention maps captured during inversion and fused during editing.
- Relevance: **medium**. Useful for attention/inversion mechanics.
- Limitation for us: not targeted at cause-effect removal; requires input video/inversion.

**Video-P2P**

- Primary sources checked: CVPR 2024 openaccess, arXiv `2303.04761`, GitHub `JIA-Lab-research/Video-P2P`.
- Credibility: **A**.
- What it does: real-world video editing via cross-attention control, inversion, optimized unconditional embeddings, and decoupled guidance.
- Relevance: **medium**. Useful if we later explore attention-control implementations.
- Limitation for us: input-video editing and attention-control; not direct T2V concept erasure or causal-footprint handling.

### Concept Erasure Baselines / Related Work

**VideoEraser**

- Primary sources checked: arXiv `2508.15314`, ACL Anthology EMNLP 2025 main paper list/PDF.
- Credibility: **A**.
- What it does: training-free T2V concept erasure using Selective Prompt Embedding Adjustment and Adversarial-Resilient Noise Guidance.
- Relevance: **required baseline**.
- Limitation for us: objective is target concept suppression, not downstream footprint consistency.

**SAFREE**

- Primary sources checked: ICLR 2025 OpenReview/proceedings PDF, project page.
- Credibility: **A**.
- What it does: training-free adaptive guard for safe T2I/T2V generation through embedding/latent filtering.
- Relevance: **required baseline / useful mechanism reference**.
- Limitation for us: no explicit causal footprint objective.

**T2VUnlearning**

- Primary sources checked: arXiv `2505.17550`, OpenReview PDF, GitHub link in paper.
- Credibility: **B/C for reproducibility**. Paper exists, but local repo inspection showed missing/placeholder training paths; treat implementation claims carefully.
- What it does: T2V concept erasure via negatively guided velocity prediction fine-tuning, prompt augmentation, localization, and preservation regularization.
- Relevance: **required baseline**.
- Limitation for us: training code/checkpoints are not fully reproducible from the public state we saw; no causal-footprint objective.

**CLEAR / Concept-Layer Alignment**

- Primary sources checked: arXiv `2605.25941`; secondary lists claim ICML 2026.
- Credibility: **B/C until proceedings verified**.
- What it does: identifies model depths where target concepts are more separable, then optimizes erasure around concept-layer alignment.
- Relevance: **method-internals reference** if we later choose layer/attention intervention.
- Limitation for us: target concept separation, not target-to-footprint causal propagation.

## Method Takeaways From Verified Batch 1

This section translates the literature check into method-design implications. It should guide prototypes after the three-model baseline closure.

### Takeaway 1: "Effect masks" are the missing object-removal concept we need to generalize

ROSE and EffectErase both formalize affected regions beyond the object mask. ROSE supervises side-effect regions with paired-video differences, while EffectErase uses paired object-present/object-absent videos and task-aware region guidance. VOID extends this idea from appearance effects to physical interactions by asking a VLM to identify affected regions.

For our setting, the analogous object is:

```text
target mask / concept region  -> direct evidence of C
effect mask / affected region -> causal footprint F(C)
```

But unlike object removal, we often do not have:

- an input video mask;
- paired factual/counterfactual videos;
- a consistent pixel region for abstract footprints;
- a single object-removal model shared across CogVideoX, ZeroScope, and Wan.

Therefore, the immediate method direction is not to copy ROSE/EffectErase. The transferable idea is to build a **causal affected-region or affected-concept planner**. It can output text descriptors first, and later masks if the generator/editing backend supports them.

### Takeaway 2: VOID validates our problem framing, but not our implementation setting

VOID is especially important because it explicitly says appearance-only object removal fails when the object causes physical interactions. This is almost the same failure pattern as causal-footprint leakage:

```text
object removed, interaction remains -> implausible video
target erased, footprint remains    -> implausible counterfactual erasure
```

The difference is the interface:

```text
VOID: existing video + object deletion + affected-region inpainting
ours: text-to-video concept erasure + no guaranteed input mask + multiple T2V backbones
```

This suggests the paper should position VOID/ROSE/EffectErase as adjacent object/effect editing work, not as direct baselines unless we add a post-generation repair track.

### Takeaway 3: Reasoning tokens / temporal reasoning are a strong conceptual bridge

VideoCoF and ChronoEdit both argue that precise editing needs an intermediate reasoning step. VideoCoF predicts edit-region latents before target video tokens; ChronoEdit denoises with temporal reasoning tokens to constrain physically plausible transformations. IF-Edit shows a lighter version: chain-of-thought prompt enhancement plus video-diffusion priors can improve reasoning-centric edits without fine-tuning.

For us, this motivates a two-stage method concept:

```text
Stage 1: reason about causal consequences
  input: source prompt, target concept, optional clean video/eval evidence
  output: affected footprint descriptors, no-cause replacement, optional regions

Stage 2: model-specific erasure/editing
  CogVideoX / ZeroScope / Wan each use the plan through the mechanism available in that backbone
```

This framework allows model-specific implementations while keeping one conceptual method: **reasoned counterfactual causal erasure**.

### Takeaway 4: Black-box counterfactual steering is credible, but may be too weak alone

Causally Steered Diffusion is useful because it treats video editing as counterfactual generation with causal effectiveness and minimality criteria. It is also black-box compatible, which is attractive across backbones.

However, it optimizes prompts rather than changing model internals. If our final method only optimizes prompt text, reviewers may still group it with advanced prompting. A stronger version should use black-box planning as the reasoning layer and then inject the plan through at least one nontrivial generation/editing mechanism:

- denoising latent steering;
- prompt embedding displacement with counterfactual comparison;
- attention/latent region guidance where supported;
- iterative critic loop with evaluator feedback.

### Takeaway 5: Temporal consistency methods preserve what we sometimes need to change

TokenFlow, FateZero, Video-P2P, and ProPainter are useful for keeping video edits coherent over time. But our task is not simply to preserve the source video. Sometimes the footprint is precisely the temporal content that must be changed.

For causal-footprint erasure, temporal consistency must be conditional:

```text
preserve unrelated background and camera motion
change target-caused downstream dynamics
```

This is a key distinction from standard video editing. It should appear in the method motivation and evaluation.

### Takeaway 6: Concept-erasure papers remain baselines, not the solution center

VideoEraser, SAFREE, T2VUnlearning, and CLEAR are essential for baseline and related-work positioning. But none of the verified concept-erasure sources explicitly optimizes:

```text
target absent AND causal footprint absent AND unrelated scene preserved
```

This is the gap our paper should keep emphasizing. Their methods may still inspire components:

- VideoEraser: prompt-embedding adjustment and noise guidance;
- SAFREE: training-free embedding/latent filtering;
- T2VUnlearning: preservation-vs-erasure loss structure;
- CLEAR: layer/depth choice for intervention.

But the objective must be upgraded from target erasure to counterfactual causal erasure.

### Takeaway 7: Temporal consistency can preserve the wrong thing

ProPainter, TokenFlow, FateZero, and Video-P2P give useful mechanisms for preserving motion, layout, and cross-frame identity. The catch is that causal-footprint erasure sometimes needs to change exactly the motion/effect that ordinary video editing tries to preserve.

For our task, temporal consistency must be conditional:

```text
preserve: background, camera motion, unrelated objects, unrelated dynamics
change: target-caused ripples, cracks, traces, deformation, particles, collision outcomes
```

Therefore, if we borrow TokenFlow-style feature propagation or ProPainter-style inpainting, it must be gated by an affected-footprint plan. Without such gating, temporal propagation may keep the leaked footprint stable across frames.

### Takeaway 8: Concept-erasure mechanisms are useful primitives, not enough as objectives

The verified concept-erasure papers suggest mechanisms we can reuse:

- VideoEraser: prompt-embedding adjustment and adversarial-resilient noise guidance.
- SAFREE: training-free text/latent filtering and subspace-style unsafe-concept suppression.
- T2VUnlearning: erasure/preservation objective design and localization regularization.
- CLEAR: choosing model depths where concept signals are more separable.

But these mechanisms need a new objective. A causal-footprint method cannot only reduce target-concept evidence; it must reduce the conditional evidence of downstream effects under target absence.

## Candidate Method Families After Batch 1

### Family A: Reasoned Causal-Set Guidance

Use an LLM/VLM to expand:

```text
C -> {direct target evidence, downstream footprint descriptors, no-cause replacement}
```

Then run model-specific target/footprint guidance. This is easiest and most portable, but it risks being viewed as prompt engineering unless paired with latent/denoising intervention or strong ablations.

### Family B: Counterfactual Denoising Steering

Construct factual and counterfactual conditions and steer denoising predictions away from the factual-minus-counterfactual direction. This is stronger than negative prompting and can potentially be implemented at scheduler/noise-prediction level for multiple backbones.

Minimum first test:

```text
model: ZeroScope or Wan
sample: 20-30 clean-valid cases
compare:
  target-only baseline
  target+footprint negative prompt
  counterfactual prompt only
  counterfactual denoising steering
metrics:
  target erased
  strict footprint leakage
  erased-clean
  quality/borderline
```

### Family C: Planner-Critic Iterative Erasure

Use the existing evaluator as a critic:

```text
generate erased video
if target_visible=no and footprint_visible=yes:
  ask planner to revise counterfactual plan
  regenerate or repair
```

This is highly portable across backbones but expensive and may be criticized as a system pipeline. It is still a strong practical method if we can show large leakage reductions.

### Family D: Post-Generation Effect Repair

Apply object/effect removal or video inpainting after target erasure. This is closest to VOID/ROSE/EffectErase but requires masks/regions and is less naturally a T2V concept-erasure method. It is best reserved for upper-bound experiments or figure-quality repair examples.

### Family E: Synthetic Counterfactual Adapter

Train a small adapter/LoRA on factual/counterfactual pairs. This is the most principled long-term solution if data is available, but it is not the best immediate path because training data and cross-backbone portability are hard.

## Immediate Survey Next Steps

1. Read VOID, ROSE, and EffectErase in detail, focusing on how they define and supervise affected regions.
2. Read VideoCoF and ChronoEdit for how reasoning tokens/plans are inserted into generation.
3. Inspect code availability for VOID/ROSE/EffectErase/VideoCoF enough to know whether post-generation repair or planner extraction is realistic.
4. Create a small `method_prototype_candidates.md` note with two concrete prototype plans:
   - ZeroScope/Wan counterfactual denoising steering;
   - black-box planner-critic loop using the existing VLM evaluator.

## Code / Prototype Feasibility Notes

### VOID

Code and checkpoints are available from the official Netflix repository and Hugging Face. The repository states that VOID is built on CogVideoX and uses two transformer checkpoints, Pass 1 for base inpainting and Pass 2 for warped-noise refinement. It also notes that Diffusers pipeline support is still a TODO.

Implication:

- feasible as a post-generation repair reference if we can provide input videos and masks;
- not immediately a clean T2V concept-erasure module;
- useful implementation idea: interaction-aware mask conditioning and a two-pass refinement design;
- likely requires substantial VRAM and video inpainting setup.

### ROSE

The official repository exposes inference arguments for `validation_videos`, `validation_masks`, prompts, frame length, and output directory. It also provides an interactive demo. The code is based on Wan2.1-Fun-1.3B-Inpaint and ProPainter.

Implication:

- feasible as an input-video inpainting / repair candidate;
- requires masks, so it needs a target/footprint localization module before it can be used on our generated videos;
- useful for studying effect-region supervision and reference-based erasing, but not directly a T2V generation-time method.

### EffectErase

The official repository provides quick-start instructions, model checkpoint download, and an inference script. It requires an input foreground/background video path and a mask path, with masks generated by SAM2.1. The project page describes VOR as triplet pairs of target-present videos, target-absent videos, and masks. Its method uses task-aware region guidance in DiT blocks and an effect consistency loss over aggregated attention maps.

Implication:

- strongest candidate if we want a post-generation repair upper bound;
- its task-aware region guidance is a concrete mechanism to borrow if we later train an adapter;
- not directly applicable without masks and a video-editing formulation;
- good citation for paired counterfactual object/effect data.

### VideoCoF

The official repository has inference/training code, released training code, model weights, Hugging Face model/demo/dataset links, and claims CVPR 2026 Highlight. The project emphasizes "see -> reason -> edit": predict reasoning/edit-region latents before generating target video tokens.

Implication:

- highly relevant for method concept, but less directly reusable because it is an editing model trained for its own architecture;
- possible adaptation route is conceptual: add a causal reasoning/planning stage before erasure;
- if code is runnable, it may serve as a qualitative editing baseline for input-video repair, not a direct T2V erasure method.

## Feasibility Ranking for Prototypes

| Prototype | Feasibility now | Uses external code? | Requires masks? | Cross-backbone? | Why |
|---|---:|---:|---:|---:|---|
| Causal-set negative prompt | Very high | No | No | Yes | Already fits our existing baseline adapters. |
| Counterfactual prompt baseline | Very high | No | No | Yes | Need only LLM/planner or template-generated no-cause prompt. |
| Counterfactual denoising steering | Medium | No | No | Medium | Requires modifying generation loop/noise prediction access; easiest first on ZeroScope or Wan. |
| Planner-critic loop | Medium | No | No | Yes | Uses our VLM evaluator as critic; expensive but black-box. |
| EffectErase/ROSE/VOID repair | Medium-low | Yes | Yes | Not naturally | Needs input video plus target/effect masks; useful as upper bound. |
| VideoCoF-style reasoning tokens | Low-medium | Yes/training | No user mask, but trained model-specific | Low | Strong concept but not easy to plug into our T2V backbones. |

Current practical recommendation after code feasibility check:

1. Keep object/effect removal methods as **related work and upper-bound repair options**.
2. For the first method prototype, focus on **counterfactual planning + generation-time steering** because it is closer to T2V erasure and does not require masks.
3. If we later need a visually strong qualitative repair demo, try EffectErase or VOID on a small subset where target/footprint masks are easy.

## Current Method Hypothesis: Minimal-Pair Causal Chain Steering

The active method hypothesis is now documented in
`docs/method_candidate_causal_chain_steering.md`.

The design revises the earlier naive CLS idea. Instead of comparing one broad
factual prompt against one broad counterfactual prompt, it decomposes the
erasure target into a cause-mechanism-footprint chain and estimates denoising
directions from controlled minimal prompt pairs. This is meant to address the
main reviewer risk: a monolithic counterfactual prompt can change background,
camera, lighting, and motion in addition to the causal footprint.

This is not yet a frozen method. Opus-style adversarial review highlighted that
minimal pairs are still prompt-derived, benchmark metadata weakens generality
claims, and global steering may erase too much without localization. The
immediate next step is therefore an MVP-0 mechanism probe, not a full method
launch: test whether minimal-pair denoising steering beats target-plus-footprint
prompting, monolithic counterfactual prompting, and random-direction steering
without causing quality collapse. If it does not, the line should pivot before
being written as the paper's main method.
