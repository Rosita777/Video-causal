# C0.1 Seed-Matched Factorial Gate Design

## Goal

C0.1 is a gate before any C1 repair or verifier-guided search. It asks a
limited question: for a given text-to-video backbone and prompt family, can the
model express target presence and causal-footprint presence as separable prompt
factors?

This is not a repair method and not evidence about the model's internal causal
mechanism. It is a screening protocol that prevents later C1 experiments from
spending compute on cases where the base model cannot reliably render the
required target/footprint states.

## Name And Claim Boundary

Use the conservative name **seed-matched factorial prompt gate**.

Avoid calling the gate a causal intervention or a true counterfactual
experiment. The same seed controls only the initial noise; prompt changes can
still produce divergent denoising trajectories, composition changes, and
background drift. The safe claim is that C0.1 tests prompt-conditioned
expressibility under a fixed generation setting.

Safe claims:

- The protocol constructs a 2x2 target/footprint prompt grid with matched
  seeds.
- The protocol separates base-model expressibility screening from later repair.
- Items that fail the gate are excluded from C1 rather than counted as repair
  failures.
- Passing the gate means the item is suitable for a later intervention test,
  not that the model has learned an internal causal graph.

Unsafe claims:

- The prompt factors caused isolated internal changes in the denoising process.
- A passing item proves target and footprint are causally independent.
- A C1 success on ungated items is meaningful.
- Single-seed C0 outputs are enough for method claims.

## Four-Cell Grid

For each source item and each seed, generate four cells:

| Cell | Target expected | Footprint expected | Role |
| --- | --- | --- | --- |
| `original` | yes | yes | base causal scene |
| `remove_target` | no | no | target and consequence removed |
| `footprint_only` | no | yes | consequence without visible target |
| `target_only` | yes | no | target without consequence |

The prompts must be explicit about both factors. For example,
`remove_target` should say both that the target is absent and that the named
footprint is absent. `target_only` should preserve target visibility while
forbidding contact, impact, disturbance, and the named footprint.

## Pilot Scope

The immediate C0.1 pilot uses:

- Backbone: ZeroScope v2 576w.
- Items: 3 diverse source items from the current MVP-0 probe.
- Seeds: 5 seeds per item.
- Cells: 4 cells per seed.
- Total videos: 60.

This is large enough to test seed luck and prompt stability, but small enough
to inspect manually before scaling.

## Acceptance Gate

Each video receives two forced-choice human labels:

- `target_visible`: `present`, `absent`, or `uncertain`.
- `footprint_visible`: `present`, `absent`, or `uncertain`.

`uncertain` is not a soft success. It is a review failure for that cell unless
a second reviewer or adjudication resolves it to `present` or `absent`. This
avoids using a vague `weak` category as an escape hatch.

An item passes into C1 only if it satisfies all of the following over five
seeds:

- `original`: at least 4/5 seeds have target present and footprint present.
- `remove_target`: at least 4/5 seeds have target absent and footprint absent.
- `target_only`: at least 4/5 seeds have target present and footprint absent.
- `footprint_only`: at least 3/5 seeds have target absent and footprint
  present.

The `footprint_only` threshold is slightly looser because consequence-without-
cause prompts may be physically incoherent. That relaxation must be reported,
not hidden.

Reject an item if any of the following occurs:

- A cell repeatedly fails to generate, mode-collapses, or produces unrelated
  content.
- All four cells look essentially identical across seeds.
- The original cell does not reliably contain the target and footprint.
- `target_only` repeatedly preserves the footprint.
- Prompt changes cause large unrelated scene rewrites that make the four cells
  incomparable.

Scene drift is counted as a rejection reason when the cell changes the basic
scene category, camera framing, background object layout, or target location so
strongly that target/footprint presence can no longer be compared to the
original cell. Reviewers record this with a separate `scene_structure_preserved`
field. The item fails if scene structure is not preserved for at least 4/5
seeds in any required cell.

Cells are treated as indistinguishable when reviewers cannot identify which
cell is meant to contain the target, footprint, both, or neither after seeing
the four cells for a seed in random order. If this occurs for 4/5 seeds, the
item is rejected as prompt-deaf or semantically collapsed.

## Human Review Protocol

C0.1 uses human review as the first gate, not fable or another VLM as the
primary judge. Reviewers should see frame strips or short videos without the
cell name. They may see the target concept and footprint definition, but not
whether the clip is supposed to be `original`, `remove_target`,
`footprint_only`, or `target_only`.

The reviewer form should record:

- target visibility,
- footprint visibility,
- unrelated scene drift,
- whether the original scene structure is preserved,
- whether the four cells for a seed are semantically distinguishable,
- generation failure or mode collapse,
- notes for uncertain cases.

If two human reviewers are available, disagreements on present/absent cells
should be adjudicated manually before an item passes to C1. If only one reviewer
is available, C0.1 results should be described as a pilot screen rather than a
validated benchmark. A single-reviewer pass may identify candidate items, but
it does not authorize C1 claims. Before any C1 go-decision, a second reviewer
must confirm at least one passing item or the result must be explicitly labeled
single-reviewer provisional.

Structured rejection reasons are required:

- `original_unreliable`
- `remove_target_failed`
- `target_only_preserves_footprint`
- `footprint_only_incoherent`
- `scene_drift`
- `cells_indistinguishable`
- `generation_failure`
- `mode_collapse`
- `review_uncertain`

## Required Controls

C0.1 needs controls that answer common reviewer attacks:

- Multi-seed control: five seeds per cell prevent one lucky seed from passing.
- Explicit-factor control: every prompt must specify both target and footprint
  state.
- Base-validity control: original cells must pass before repair is considered.
- Negative reporting: failed and borderline items remain in the manifest and
  summary.
- Prompt log: all prompts used for all cells are stored verbatim.
- Rejection-code log: all failed and borderline decisions carry one or more
  structured rejection reasons.

Optional later controls:

- A second prompt template per cell to distinguish prompt-template artifacts
  from item expressibility.
- A second backbone, likely Wan, if ZeroScope yields too few passable items.

Before scaling, `footprint_only` prompts should receive a prompt-validity
check. If two reasonable `footprint_only` templates fail to produce the
footprint, the item is marked `footprint_only_incoherent` rather than treated
as ordinary model failure.

## C1 Anti-Prompt-Hacking Rules

C1 may only run on C0.1-passing items. It must keep the following rules:

- All candidate prompts and rejected attempts are logged.
- The verifier or LLM prompter cannot see hidden cell labels or hand-picked
  best frames.
- C1 outputs must be evaluated on held-out seeds, not only the seed used during
  search.
- C1 prompts must preserve the source scene and target/footprint wording within
  a bounded edit policy; full scene rewrites are not allowed.
- A known difficult or entangled item should be included as a negative control.
  If C1 claims success there, the run is treated as verifier exploitation until
  manually disproven.
- C1 success must be compared against matched prompt-only and random-search
  controls.

## Outputs

The C0.1 pilot should produce:

- a generation manifest with item, seed, cell, prompt, and expected states;
- generated videos or frame strips for human review;
- a blinded review CSV;
- a gate summary with per-item pass/fail counts;
- a short limitations note listing failed cells and rejected items.

## Decision Rule

After the 3-item pilot:

- If at least one item passes cleanly and another is borderline, implement C1 on
  only the passing item(s), subject to second-reviewer confirmation or explicit
  provisional labeling.
- If no item passes, do not run C1 on ZeroScope; either redesign prompts or move
  C0.1 to Wan.
- If all three pass, scale C0.1 to a 12-item screen before C1.

The 3-item pilot is a process test. It can show whether the protocol is
executable and whether any candidate item is worth C1. It cannot establish that
the gate is a validated benchmark.
