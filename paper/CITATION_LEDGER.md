# Citation ledger for Related Work

Verified 2026-09-10 from primary paper, proceedings, project, or repository
pages.  The ledger licenses narrow manuscript claims; it is not a priority
claim or an exhaustive survey.

| Key | Primary authority | Licensed use in the paper | Boundary that must remain explicit |
| --- | --- | --- | --- |
| `gandikota2023esd` | ICCV 2023 open proceedings | Negative-guidance weight fine-tuning as image concept erasure | Image concepts, not video events |
| `gandikota2024uce` | WACV 2024 open proceedings | Closed-form parameter editing as a second intervention family | Image-only; edits text-to-image projections rather than event roles |
| `yoon2025safree` | ICLR 2025 OpenReview and project | Training-free prompt/latent filtering for T2I and T2V | Safety concepts, not registered event roles |
| `xu2025videoeraser` | EMNLP 2025 ACL Anthology | Training-free T2V prompt/noise intervention | Target-concept presence, not a dependent footprint |
| `ye2025t2vunlearning` | arXiv v3 and official repository | Velocity-prediction unlearning with localization/preservation | Cite as preprint; our baseline is an adaptation |
| `wang2026latentdpo` | IEEE ICASSP 2026 | Matched-noise latent preference learning for concept-free trajectories | Paired preference does not imply source-role intervention |
| `wang2026erasesae` | arXiv v2 and ECCV 2026 accepted list | Sparse-feature localization for T2V concept erasure | Localizes target semantics, not downstream causal effects |
| `sadhu2021vidsrl` | CVPR 2021 open proceedings | Video events represented with verbs, role-bearing entities, and event relations | Predicts semantic roles for understanding; does not edit a generator or discover causal roles |
| `zhou2023propainter` | ICCV 2023 open proceedings | Masked video inpainting and temporal propagation | Requires an observed video and mask |
| `miao2025rose` | NeurIPS 2025 proceedings | Paired-data removal of shadows/reflections and other side effects | Input-video, reference-based erasing |
| `kushwaha2026objectwiper` | CVPR 2026 open proceedings | Training-free associated-effect localization and removal | Requires input video, object mask, and query tokens |
| `fu2026effecterase` | CVPR 2026 project and arXiv | Paired present/absent video supervision and reciprocal insertion/removal | Mask-conditioned video-to-video setting |
| `motamed2026void` | arXiv and ECCV 2026 accepted list | Object removal that changes downstream physical interactions | Input-video instance removal with affected-region guidance |
| `ekin2026beyondmasks` | arXiv and ECCV 2026 accepted list | Causal/physical removal evaluation with object and after-effect scores | Paired clean references for video removal, not model erasure |
| `spyrou2025causally` | arXiv and official code | Prompt optimization under an assumed causal graph for video counterfactuals | Input-video editing; graph supplied rather than discovered |
| `wang2026chain` | CVPR 2026 open proceedings | Event-centric causal progression for physically plausible generation | Improves event generation; does not erase a source role |

## Positioning licensed by this ledger

- Do not claim that this paper first notices residual effects, first removes an
  object and its consequences, or first evaluates object and after-effect
  disappearance separately.
- The supported distinction is the setting and suppression unit: model-level
  behavior in an open-ended text-to-video generator, conditioned on a
  registered causal-source role and paired with same-noun noncausal
  specificity.
- `distributional source-free target` must not be described as a pixel-aligned
  individual or uniquely true counterfactual.
- The empirical conclusion remains `role-conditioned behavior`; supplied event
  roles and behavioral tests do not establish causal discovery or an internal
  causal representation.

## Verified watch list, not cited in the compact draft

- ICE / *Now You See It, Now You Don't* (CVPR 2026 Findings) is relevant to
  erase/preserve subspace overlap but not event roles.
- ObjectDrop (ECCV 2024) is useful background for aligned object-present/absent
  image counterfactuals.
- SIRUS (`arXiv:2607.14194`), CLEAR (`arXiv:2605.25941`), and GenEraser
  (`arXiv:2605.30045`) remain useful monitoring items but are not needed for
  the current three-paragraph argument.
- UnderEraser (`arXiv:2604.01693`) and EffectLearner (`arXiv:2608.05565`)
  explicitly model object--effect relations; they overlap the cited
  effect-aware removal family and remain watch-list items rather than added
  citation density.
