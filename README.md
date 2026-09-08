# Video Causal Erasure

This repository studies causal-role erasure in text-to-video diffusion:
reducing a registered source and its downstream visual footprint while
preserving the receiver, scene quality, and noncausal uses of the same entity.

## Current Status

Updated 2026-09-08.  The active protocol is
`causal_role_erasure_7m_single_seed_v2`; the earlier single-mechanism
water-impact studies are historical development evidence.

The formal experiment contains seven equal-status mechanisms:

1. water impact;
2. rigid collision;
3. brittle fracture;
4. powder impact;
5. elastic deformation;
6. material release; and
7. surface trace.

Airflow was excluded by a frozen Original-capability amendment before formal
training or treatment generation.  It was not replaced by another mechanism.

The eight main streams are Wan Original, Matched Control, Source-slot
Randomized Counterfactual Distillation (SRCD; recorded as `V4` in frozen
artifacts), CogVideoX Original, Negative Prompt, VideoEraser (official
CogVideoX), T2VUnlearning-adapted (ours), and SAFREE-CogVideoX.
Here “(ours)” identifies the internal baseline adaptation, not the proposed
SRCD method.

Formal data construction, training, and generation are complete:

| Artifact | Frozen count or status |
| --- | ---: |
| Selected source-free targets | 7 x 178 = 1,246 |
| Shared preservation bindings | 36 |
| Wan method/control training runs | 18 eligible step-200 checkpoints |
| Adapted T2VUnlearning runs | 7 eligible step-100 checkpoints |
| Main-study videos | 2,352 |
| Identification-study additions | 96 |
| Total validated videos | 2,448 |

Two blinded, independently ordered passes of the same `gpt-5.6-luna` judge
are also complete: each covers 2,448/2,448 videos, satisfies the registered
response schema, and records zero scientific fallback scores.

Final human labeling has not started.  Its frozen initial queue
contains 3,633 atomic judgments covering 2,009 anonymous videos.  Until two
independent human reviews, disagreement adjudication, any required audit
expansion, and the canonical metric freeze are complete, no current numeric
result is final.

The existing generated tables are explicitly labeled
`POST_UNBLINDING_EXPLORATORY_PRELIMINARY`, `NOT_CANONICAL`, and
`WRITING_PREVIEW_ONLY_NOT_FOR_FINAL_SCIENTIFIC_CLAIMS`.  They may guide paper
structure and internal discussion but must not supply final paper claims.

## Paper Status

The manuscript uses the official ICLR 2027 template under `paper/`.
Section 3, Method 4.1--4.3, and Appendices A--C have draft prose.  Results,
final tables, and the abstract result sentence remain gated on human-canonical
scores.  The writing order and claim boundaries are in
[`paper/WRITING_PLAN.md`](paper/WRITING_PLAN.md).

## Start Here

- [`docs/PROJECT_HANDOFF.md`](docs/PROJECT_HANDOFF.md): authoritative current
  state, artifact locations, and next actions.
- [`docs/causal_role_erasure_7mechanism_protocol_v2.md`](docs/causal_role_erasure_7mechanism_protocol_v2.md):
  frozen seven-mechanism scope and main protocol.
- [`data/causal_role_erasure_7mechanism_main_v2/run_matrix.csv`](data/causal_role_erasure_7mechanism_main_v2/run_matrix.csv):
  the 18 Wan method/control training runs.
- [`paper/README.md`](paper/README.md): paper architecture, build commands, and
  submission checks.

## Immediate Next Actions

The method and GPU experiment are frozen.  Do not tune another checkpoint,
seed, mechanism weight, prompt set, or method variant on the formal data.

1. Freeze a provenance amendment that truthfully records the already-opened
   global method key while preserving process-local answer-key-blind human
   review and every registered scoring rule.  Add new versioned files/wrappers;
   do not edit any v1 evaluation-registry-bound implementation.
2. Implement and freeze the final human-canonical table exporter before
   viewing canonical metrics; the current exporter is preview-only.
3. Freeze sanitized reviewer instructions, then give each independent reviewer
   an isolated copy of only
   `../causal7m_formal_review_launch_v5/human_audit_stage0/public/`.  Never
   copy its parent directory or any `private/` directory.
4. Adjudicate every human disagreement and apply the registered greater-than-
   5% high-confidence-error expansion rule.
5. Freeze anonymous canonical scores, then freeze Original eligibility and
   shared-capability subsets.
6. Run the amended formal metric builder and the already-frozen final table
   exporter.
7. In parallel, continue paper sections that do not depend on final scores:
   evaluation details, reproducibility, experimental setup, and related work.

## Artifact Locations

The formal A100 project is
`/data/xiaohuang_workspace/ljc/Video-causal-v4`, with outputs under
`outputs/causal_role_erasure_7mechanism_main_v2`.  Frozen local media and
review artifacts live outside this Git checkout in the sibling workspaces:

- `../causal7m_formal_media_snapshot_v1`;
- `../causal7m_wan_original_snapshot_v1`; and
- `../causal7m_formal_review_launch_v5`.

Do not clean, overwrite, or regenerate these roots.  Model weights, videos,
caches, private answer keys, and review secrets remain outside Git.  Commit
source code, public manifests, paper text, and sanitized public aggregate
summaries only; never copy a private receipt verbatim into Git.

## Claim Boundary

The controlled method comparison is SRCD versus Matched on the same Wan
backbone.  CogVideoX baselines form a separately normalized benchmark block;
raw cross-backbone scores are descriptive.  The paper does not claim complete
source removal, one universal seven-mechanism adapter, discovery of an
internal causal representation, or superiority over every baseline unless
the final registered evidence licenses that statement.
