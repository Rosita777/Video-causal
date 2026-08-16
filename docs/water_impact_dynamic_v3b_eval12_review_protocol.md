# Water-impact v3b eval12 blind-review protocol

Status: frozen on 2026-08-16 after generation and technical validation, but
before either independent reviewer returned any visual score. Eval12 is a
development gate, not the fresh paper-final main test.

## Blinding and evidence

- Public reviewer package:
  `experiments/water_impact_dynamic_eval12/v3b_target_prompt_teacher_blind_review_v3_public`
- Private key/provenance package:
  `experiments/water_impact_dynamic_eval12/v3b_target_prompt_teacher_blind_review_v3_private`
- The public package contains only the 24-row blind CSV, 12 anonymous
  composites, and 24 independently copied anonymous videos.
- Reviewers may inspect only the public package. They score every full 49-frame
  video and may use the paired seven-frame composite for reference.
- Target and footprint visibility: `0=absent`, `1=partial/weaker`, `2=clear`.
- Receiver preservation and video quality: `0=bad`, `1=partial`, `2=good`.

## Independent review and adjudication

Two reviewers score all 24 anonymous candidates independently. For each atomic
field, exact agreement becomes the canonical score. Every disagreement is sent
to a third blinded adjudicator, who sees only the disputed anonymous sample.
The canonical value is the three-reviewer majority; if the three values are
exactly `0`, `1`, and `2`, the canonical value is the median, `1`.

No reviewer or adjudicator may read the private answer key, provenance
manifest, source generation directories, prior scores, or method mappings
before the canonical review CSV has been frozen and hashed.

## Decision rule

The scorer applies the gate frozen in
`docs/water_impact_dynamic_v3b_target_prompt_teacher.md`: mechanism-positive
requires three usable control-target improvements, one reaching absence, two
generalization groups, and at least +3 target-suppression points on the
control-usable set. Promotion additionally requires at least 11 usable v3b
outputs, the registered receiver and quality floors, non-worse footprint
suppression on the control-usable set, and at least one strict success.

