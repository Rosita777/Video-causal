# Fable Review: C0.1 Implementation Check

`claude-fable-5` reviewed the implemented C0.1 seed-matched factorial gate as a
method and engineering advisor, not as a video judge.

## Verdict

No blocking issue before the real 60-video pilot, as long as C0.1 is framed as
a generation-validity gate rather than a causal-intervention or internal-causal
reasoning result.

## Reviewed Implementation Facts

- The dry-run manifest expands 3 items into 5 seeds and 4 factorial cells.
- The four cells use expected target/footprint states: `original` yes/yes,
  `remove_target` no/no, `footprint_only` no/yes, and `target_only` yes/no.
- The blind review CSV is separated from the answer key.
- The blind review does not expose the variant label, expected labels, prompt,
  or raw video path.
- The scorer joins by `review_id`, treats `uncertain` as a failed cell, and
  records structured rejection reasons.
- The related pytest suite passes.

## Required Pilot Guardrails

1. Manually spot-check 10 to 15 generated videos before bulk review. Confirm
   that the cells are distinguishable, target/footprint manipulations are
   plausible, and there is no systematic prompt ignoring.
2. Track `uncertain` labels by reviewer and cell type. If more than about 30%
   of labels are uncertain, pause and revise the prompts or reviewer
   instructions instead of treating the item failures as ordinary rejects.
3. Add an inter-rater check on at least 10 overlapping review rows. If binary
   target/footprint agreement is poor, pause and recalibrate reviewers before
   scoring the full pilot.

## Framing Constraint

Passing C0.1 makes an item eligible for C1 intervention testing. It does not
prove causal independence, true counterfactual consistency, or internal causal
reasoning.
