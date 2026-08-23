# Seven-mechanism causal-role erasure: single-seed master protocol v2

Status: **pre-treatment scope amendment; prompt-refresh smoke frozen on
2026-08-23**.

Protocol ID: `causal_role_erasure_7m_single_seed_v2`.

This protocol supersedes `causal_role_erasure_8m_single_seed_v1` for the main
experiment. All terms of v1 remain binding except for the explicit overrides
below. The amendment was made after inspecting Wan Original capability videos
and before training any main adapter or generating any treatment output. It is
therefore a post-hoc benchmark-design decision, not confirmatory evidence.

## 1. Scope

The main experiment contains seven equal-status mechanisms:

1. water impact;
2. rigid collision;
3. brittle fracture;
4. powder impact;
5. elastic deformation;
6. material release; and
7. surface trace.

Airflow-mediated response is excluded. It produced only 4/24 task-relevant
Original videos in capability v2, whereas each retained mechanism produced at
least 12/24. Airflow is not replaced by a new mechanism. The complete frozen
eight-mechanism capability batch, its adjudicated scores, and its FAIL outcome
remain unchanged and reportable.

The seven-mechanism aggregate gives every retained mechanism weight `1/7`.
No retained mechanism may be dropped or down-weighted after any treatment
output is inspected.

Historical capability indices remain `[0, 1, 2, 3, 4, 6, 7]`; new contiguous
main-experiment indices are `[0, 1, 2, 3, 4, 5, 6]`. Existing release and trace
artifacts are never renamed or reindexed.

## 2. Prompt construction amendment

Water-impact and material-release ontology semantics are unchanged. The other
five mechanisms use one uniform prompt rule derived from their Original-only
failure modes:

- sentence one places a recognizable source and clean receiver together in a
  fixed-camera composition before the action;
- sentence two specifies one visible trigger, one simple persistent footprint,
  and a visible source end state;
- direct and natural wording remain equally explicit; the filler phrase
  `as a natural result` is not used;
- objects use high source/receiver contrast and familiar physical forms; and
- prompts do not require perfect removal, a delayed source entrance, or a
  transient effect to remain visible for the whole clip.

Mechanism-specific persistent footprints are:

| Mechanism | Main prompt correction |
| --- | --- |
| Rigid collision | Receiver falls flat and remains there; sliding is not required |
| Brittle fracture | Separate fragments remain and the impactor rests visibly among or beside them |
| Powder impact | A deep crater remains around a half-embedded visible ball; a dust puff is optional |
| Elastic deformation | The elastic bed remains deeply bowed beneath a resting visible ball |
| Surface trace | A deep shape-specific indentation remains while the stamp stays in frame |

These corrections change data construction, not the V4 optimization method,
training hyperparameters, evaluation metric, or baseline set.

## 3. One-time development smoke

Before the final ontology and formal case manifests are frozen, exactly two
new prompts per corrected mechanism are generated with the fixed seeds
`979000..979009`. This ten-video run is a development-only sanity check. It is
not pooled with capability v2 or reported as a paper result. Seeds are not
retried, and its decision rule is frozen in
`data/causal_role_erasure_7mechanism_prompt_refresh_smoke_v1.json`.

After this bounded check, prompt templates and ontology fields are frozen for
the main experiment. Formal Original eligibility remains the per-case rule
defined in v1, so weak base generations stay in the fixed denominator and
cannot be counted as erasure successes.

## 4. Updated experiment counts

- 7 matched-control adapters + 7 V4 adapters + 4 Water/Fracture
  identification controls = **18 Wan training runs**;
- 7 mechanisms x (24 causal + 18 specificity) = **294 semantic cases**;
- 294 cases x 8 unchanged physical method streams = **2,352 main videos**;
- the identification study adds **96** videos; and
- total formal generation is **2,448** videos.

The eight physical streams and all named baselines remain unchanged: Wan
Original, matched no-randomization control, V4, CogVideoX Original, Negative
Prompt, official VideoEraser, T2VUnlearning-adapted, and SAFREE-CogVideoX.

All 18 Wan runs retain training seed 26000, 200 updates, LoRA rank/alpha 16/16,
learning rate `5e-5`, target-teacher weight 4, preservation weight 4, and
inference scale 1.25. Water/Fracture identification controls, review procedure,
continuous CES/SU metrics, five headline contrasts, Holm correction, and
non-inferiority margins remain unchanged.

## 5. Evidence boundary

Capability v2 is development evidence used to define this pre-treatment scope.
It cannot subsequently be presented as an independent confirmation of the
seven-mechanism result. The confirmatory evidence is the frozen formal
evaluation generated after this amendment, with one seed per case and no
case replacement or scientific retry.
