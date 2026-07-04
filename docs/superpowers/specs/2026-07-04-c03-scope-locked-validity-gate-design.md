# C0.3 Scope-Locked Validity Gate Design

## Goal

C0.3 is the decision gate after the C0.2 discrete factorial pilot. C0.2
generated correctly, but visual inspection found only one strong item family:
the garden-rake / soil-groove surface-trace case. This creates a reviewer
attack: if the next experiment keeps only the best-looking item after seeing
the outputs, the result becomes a demo rather than evidence for a method.

C0.3 prevents that by freezing the claim scope and success criteria before any
new generation pass. It is still not a causal-intervention method and not
evidence of an internal causal graph. It is a validity gate that decides
whether the project has a defensible target regime for C1.

## Design Alternatives

| Option | Description | Strength | Weakness |
| --- | --- | --- | --- |
| Broad pre-registered panel | Sample a larger diverse set of causal mechanisms and report the full pass rate. | Strongest against cherry-picking. | Higher compute and likely many failures with ZeroScope. |
| Single-item debug | Continue only with item 4 because it looked strongest. | Fastest path to a visually useful demo. | Scientifically weak; easy to attack as post-hoc selection. |
| Scope-locked panel | Define a narrow regime, then pre-register several unseen items in that regime and report all outcomes. | Balances cost and rigor; turns failures into boundary evidence. | Claim must stay narrow. |

The recommended path is the scope-locked panel. C0.3 should not claim that
counterfactual prompt grids work for video causal reasoning in general. It
should test whether a narrow regime is viable: low-entanglement rigid-object
surface traces.

## Target Regime

C0.3 targets only scenes that satisfy all of these predicates:

- The target is a rigid or tool-like object with a visually separable boundary.
- The interaction surface is simple and mostly static.
- The footprint is localized, persistent, and human-readable in a short clip.
- The footprint can plausibly appear without the target in a `footprint_only`
  cell.
- The target and footprint are named by different visual concepts, so the
  prompt does not force the model to render the target in order to render the
  footprint.
- The background and camera framing can remain stable under all four cells.

Examples inside scope:

- rake or comb leaving parallel grooves in soil or sand;
- marker, chalk, or brush leaving a simple line on a clean board or surface;
- tire or shoe leaving a single track on mud, snow, or sand;
- stamp or block leaving a simple imprint on clay.

Examples outside scope:

- fluids, splashes, ripples, smoke, powder clouds, and mist;
- deformable body parts pressing soft objects when the target and footprint
  tend to merge visually;
- complex diagrams or dense sketch surfaces where any black line looks like the
  footprint;
- scenes where removing the target changes the whole setting.

## Anti-Cherry-Pick Rules

C0.3 must obey these rules before new generation:

- Freeze the target regime in this spec.
- Freeze the candidate item list before inspecting new videos.
- Report every generated candidate in the denominator.
- Keep prior-seen C0.2 item 4 separate from the new denominator. It may be used
  as a debugging exemplar, but not as proof that the C0.3 panel passed.
- Do not replace failed items after seeing outputs.
- Record all prompt templates and all generated manifests.
- Treat failures from items 3 and 8 in C0.2 as substantive negative evidence
  for the broader regime, not as noise.

## C0.2 Audit Before C0.3

Before running a new C0.3 generation pass, the existing C0.2 blind review rows
should be scored. This turns the current visual impression into an auditable
record.

Use the existing review package:

```text
experiments/evaluation/c02_discrete_factorial_gate_20260704_real_s10_f16_240x432/blind_review.csv
experiments/evaluation/c02_discrete_factorial_gate_20260704_real_s10_f16_240x432/answer_key.csv
```

C0.2 has only three seeds per item, so its score is diagnostic rather than a
final pass/fail benchmark. A C0.2 item is marked `diagnostic_promising` only if
all four cells meet the expected target/footprint state for at least 2/3 seeds,
with no unresolved `uncertain` labels and with scene structure preserved in at
least 2/3 seeds per cell.

Expected C0.2 interpretation:

- item 4 may become a C1 debugging exemplar if the blind score confirms the
  spot-check;
- item 10 is at most a simplified-prompt follow-up;
- items 3 and 8 are rejected for this generator/prompt regime unless blind
  review strongly contradicts the spot-check.

## C0.3 Candidate Panel

C0.3 should use a new panel of candidates that were not selected after viewing
C0.2 outputs. The default panel size is:

- 8 to 12 candidate items;
- 5 seeds per item;
- 4 cells per seed;
- 160 to 240 videos total.

If GPU budget is tight, use 8 items. If the generator remains stable and the
candidate list is cheap to build, use 12 items. The item list should be stored
as a manifest before generation. Each row must include:

- `item_id`
- `target_concept`
- `surface_or_object`
- `causal_footprint`
- `scope_predicates_met`
- `prompt_template_id`
- `prior_seen=false`

The prior C0.2 item 4 can appear in a separate `debug_exemplar` split, but not
in the C0.3 denominator.

## Four-Cell Grid

C0.3 keeps the C0.2 four-cell structure:

| Cell | Target expected | Footprint expected | Role |
| --- | --- | --- | --- |
| `original` | yes | yes | target visibly causes footprint |
| `remove_target` | no | no | neither target nor footprint appears |
| `footprint_only` | no | yes | footprint appears without visible target |
| `target_only` | yes | no | target appears without contact or footprint |

All prompts must explicitly state both target and footprint state. If a cell
does not mention the footprint absence or presence, the item is invalid before
generation.

## Scoring Labels

Human review remains the primary judge. Fable or another model may advise on
method decisions, but it is not ground truth for visual labels.

Each video receives:

- `target_visible`: `present`, `absent`, or `uncertain`;
- `footprint_visible`: `present`, `absent`, or `uncertain`;
- `scene_structure_preserved`: `yes`, `no`, or `uncertain`;
- `generation_failure`: `yes` or `no`;
- `mode_collapse`: `yes` or `no`;
- `notes`.

`uncertain` does not count as success. If a second reviewer is available,
disagreements and uncertain rows may be adjudicated. Without a second reviewer,
all uncertain rows remain failures for pass/fail scoring.

## Acceptance Rule

An item passes the C0.3 validity gate only if all of these hold over five seeds:

- `original`: at least 4/5 seeds have target present and footprint present.
- `remove_target`: at least 4/5 seeds have target absent and footprint absent.
- `target_only`: at least 4/5 seeds have target present and footprint absent.
- `footprint_only`: at least 3/5 seeds have target absent and footprint
  present.
- Every cell preserves scene structure for at least 4/5 seeds.
- No cell has more than one generation failure or mode-collapse row.

Panel-level decision:

- If at least 3 new C0.3 denominator items pass, C1 may proceed inside the
  declared low-entanglement surface-trace scope.
- If 1 or 2 new items pass, C1 may proceed only as debugging or demonstration,
  not as a method claim.
- If 0 new items pass, stop ZeroScope C1 work and either switch backbone or
  build a controlled synthetic benchmark first.

The pass rate must be reported as `passed_items / generated_denominator_items`.

## Required Rejection Codes

Each failed or borderline item must receive one or more structured rejection
codes:

- `original_unreliable`
- `remove_target_failed`
- `target_only_preserves_footprint`
- `footprint_only_incoherent`
- `scene_drift`
- `cells_indistinguishable`
- `target_footprint_entangled`
- `generic_texture_confound`
- `generation_failure`
- `mode_collapse`
- `review_uncertain`

The C0.3 summary should count these codes. This makes negative results useful
instead of merely discarded.

## Evidence Needed For A Publishable Method Claim

C0.3 alone can support only a scoped generation-validity claim. A stronger
paper claim needs additional evidence:

- a pre-registered candidate denominator and full pass-rate report;
- blind human labels with either a second reviewer or explicit single-reviewer
  caveat;
- prompt logs and generation manifests for every item;
- negative reporting for failed candidates;
- later C1 intervention results only on items that passed C0.3;
- a controlled or synthetic benchmark with known masks or known causal
  structure;
- a consistency or re-insertion test showing that the method changes the
  intended target/footprint relation rather than only changing style.

Without these, the honest claim is: the protocol found a narrow set of visually
expressible target/footprint prompt factors, not that the video model performs
causal counterfactual reasoning.

## Outputs

C0.3 should produce:

- this scope-locking spec;
- a C0.2 blind-score audit over the existing 48 rows;
- a pre-registered C0.3 candidate manifest;
- a generation manifest for the new denominator panel;
- review CSV, answer key, review manifest, and frame strips;
- an item-level score summary with pass/fail and rejection codes;
- an experiment-log entry recording the denominator, pass rate, and decision.

Video and image files remain local generated artifacts unless explicitly
needed for sharing. CSV, JSON, and logs are safe to commit.

## Non-Claims

C0.3 does not prove causal reasoning, internal world-model structure, or
semantic factorization inside diffusion. It is a guardrail against premature
claims. Passing C0.3 means only that the generator and prompt family can express
the required four target/footprint states for a narrow, declared regime.
