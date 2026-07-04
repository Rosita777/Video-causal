# C0.2 Discrete Factorial Gate Design

## Goal

C0.2 is a follow-up to the C0.1 real 60-video pilot. C0.1 showed that the
protocol is executable, but the first three items were visually weak: water
ripples, mirror cracks, and net deformation were hard to separate from texture,
scene drift, or ordinary generation noise.

C0.2 keeps the same conservative claim boundary as C0.1. It is still a
generation-validity gate, not a causal intervention and not evidence of an
internal causal graph. The change is the item and prompt policy: choose more
discrete target/footprint pairs and use stricter four-cell prompt templates.

## Motivation From C0.1

The C0.1 spot-check found three failure patterns:

- Fluid ripples were too texture-like: target absence and footprint absence
  were visually ambiguous.
- Mirror cracks caused large scene and material drift across cells.
- Soccer net deformation had clearer target control, but footprint presence
  remained hard to judge.

These are not just bad samples. They show why the gate is needed: C1 should not
try to repair or intervene on items where the base generator cannot reliably
render the required states.

## Candidate Set

C0.2 starts from the existing 12-item MVP-0 probe manifest and selects items
with more discrete, localized, and human-readable footprints. The initial
candidate indices are:

| Index | Pair | Target | Footprint |
| --- | --- | --- | --- |
| 3 | `v2_particle_dispersion_makeup_brush_a_pink_powder_cloud_blooms_upward_bright_244` | makeup brush | pink powder cloud |
| 4 | `v2_surface_trace_garden_rake_parallel_grooves_remain_in_the_soil_side_123` | garden rake | parallel soil grooves |
| 8 | `v2_elastic_deformation_hand_a_deep_pillow_dent_forms_and_remains_slow_176` | hand | pillow dent |
| 10 | `v2_surface_trace_marker_pen_a_black_line_remains_on_the_whiteboa_bright_114` | marker pen | black line on whiteboard |

These pairs are chosen because the footprint should be more spatially discrete
than ripples, cracks, or net motion. If any selected item produces obvious
prompt incoherence in dry-run inspection, it can be replaced by another
particle, trace, or deformation item from the same source manifest.

## Four-Cell Grid

C0.2 preserves the C0.1 four-cell grid:

| Cell | Target expected | Footprint expected | Role |
| --- | --- | --- | --- |
| `original` | yes | yes | target causes visible footprint |
| `remove_target` | no | no | no target and no footprint |
| `footprint_only` | no | yes | footprint visible with no target |
| `target_only` | yes | no | target visible but no contact or footprint |

The important change is prompt specificity. Every cell must specify both target
state and footprint state. A cell should not rely on the base prompt leaving a
factor implicit.

## Prompt Template Policy

All four cells use the same scene anchor:

```text
A realistic fixed-camera close-up video of the same simple scene.
The background, camera framing, and surface stay consistent across the clip.
```

Cell templates:

```text
original:
The {target} is clearly visible and contacts the {surface_or_object}.
After contact, {footprint} is clearly visible.

remove_target:
No {target} is present. No visible cause is present.
The {surface_or_object} stays clean and unchanged.
There is no {footprint}.

footprint_only:
No {target} is present and no visible cause appears in the frame.
The {footprint} is clearly visible on the {surface_or_object}.
The scene otherwise stays the same.

target_only:
The {target} is clearly visible, but it is separated from the
{surface_or_object} and does not touch, strike, mark, press, disturb, or
change it. There is no {footprint}.
```

The implementation may fill `surface_or_object` from a small item-specific
override table. If no override exists, it should fall back to a conservative
phrase such as `surface` or `scene object`, but all C0.2 selected items should
receive explicit overrides.

Initial overrides:

| Target | Surface/object phrase |
| --- | --- |
| makeup brush | compact of pink powder |
| garden rake | smooth soil bed |
| hand | pillow surface |
| marker pen | whiteboard surface |

## Pilot Scope

The C0.2 pilot should be smaller than the first C0.1 real run but broader in
item diversity:

- Items: 4 selected candidate items.
- Seeds: 3 seeds per item.
- Cells: 4 cells per seed.
- Total videos: 48.
- Backbone: ZeroScope first, using the same lightweight generation settings as
  C0.1 unless a resource issue appears.

This pilot is a screening step. If at least two items look visually promising
in spot-check, the best items can be rerun with five seeds for a stricter
C0.1-compatible pass/fail score.

## Review And Spot-Check

C0.2 does not immediately require full blind scoring. The first pass is a
visual spot-check over contact sheets:

- one sheet per item showing all seeds and four cells;
- no fable or VLM as the primary judge;
- human notes about target clarity, footprint clarity, and scene drift.

An item is promising if:

- `original` usually shows both target and footprint;
- `remove_target` usually shows neither target nor footprint;
- `target_only` usually shows the target without the footprint;
- `footprint_only` usually shows the footprint without the target;
- the four cells preserve enough scene structure to compare.

An item should be rejected early if all four cells look similar, if the target
or footprint is not visually identifiable, or if prompt changes rewrite the
scene category.

## Outputs

C0.2 should produce:

- a filtered candidate manifest or runner option that selects the four item
  indices;
- a generation manifest with all 48 planned rows;
- local videos and frame strips;
- contact sheets for human spot-check;
- a short experiment-log entry that reports which items are promising,
  borderline, or rejected.

The video and image media remain local generated artifacts unless explicitly
needed for sharing. CSV, JSON, and log summaries are safe to commit.

## Decision Rule

After C0.2:

- If at least two items are visually promising, rerun those items with five
  seeds and the full blinded C0.1 scoring flow.
- If exactly one item is promising, use it only as a provisional C1 debugging
  target, not as evidence for a method claim.
- If no item is promising, stop ZeroScope C1 attempts and either redesign the
  prompt families or move the gate to a stronger video backbone.

## Non-Claims

C0.2 does not prove causal reasoning. It does not show that prompt factors map
to isolated latent factors. It only screens for prompt-conditioned
target/footprint expressibility under a controlled four-cell generation setup.
