# Water-impact v3c fresh-dev24 evaluation protocol

Status: stage 1 frozen on 2026-08-16 before v3c training or any eligible v3c
generation. Stage 2 is intentionally locked until the completed v3c
`checkpoint-000200` exists and passes the registered training sanity checks.
Eval12 is exhausted and is not eligible for this decision.

## Frozen partition

The source is the byte-exact 72-row
`data/water_impact_dynamic_v1/test_pairs.csv` (SHA-256
`7a8ad92df03a78e8a972a2df552e61554836e225f2a310efc8e906e9cf2d0036`).
The 12 rows already present in eval12 are excluded. Within each of the six
`generalization_group x prompt_variant` strata, candidates are ranked by
SHA-256 of

```text
26016001:<generalization_group>:<prompt_variant>:<pair_id>
```

The lowest four of each ten-row stratum form fresh-dev24; the remaining six
form sealed-final36. Output order follows `source_test_index`. This gives, per
generalization group, four direct and four natural fresh-development prompts
and six direct and six natural sealed-final prompts.

The stage-1 registry is
`data/water_impact_dynamic_v1/v3c_eval_split_registry.json`, SHA-256
`4f31a291e8ffca07da4bf057e9a86df72f656c03aab65bc06d4c3c155b72962a`.
Its validator recomputes the exact ranked membership and order; another
merely balanced 24/36 partition is rejected.

Sealed-final36 must not be generated, inspected, or scored unless every
fresh-dev24 gate check passes. The evaluation launcher intentionally exposes
no final36 command.

## Two-stage generation lock

Stage 1 binds the split, blank prompt files, blind seed, review semantics, and
all-or-nothing gate before training. The committed
`v3c_fresh_dev24_stage2_registration.template.json` contains explicit
`REQUIRED_AFTER_TRAINING_BEFORE_V3C_GENERATION` sentinels and cannot authorize
generation.

After training, `scripts/register_water_impact_dynamic_v3c_eval_stage2.py`
accepts only the registered v3c training protocol at step 200. It verifies the
objective and schedule, registration hash, passed 16-observation scale sanity,
loss/sigma arithmetic, checkpoint, LoRA weights, training state, and sanity
hashes. It then freezes their hashes in the real stage-2 registration using an
exclusive create.

Stage 2 also hashes every meaningful file in the Wan model directory in
ordered relative-path order. `.cache` metadata and temporary/lock files are
excluded; pipeline configs, tokenizer/scheduler files, transformer, VAE, text
encoder, and weights are included. Base, v3b, and v3c generation all recompute
that inventory and bind its digest plus the stage-1 and stage-2 hashes in
pre-generation sidecars. Thus changing a loaded backbone file invalidates all
three arms.

Commands after the corresponding prerequisites exist are:

```bash
bash scripts/run_water_impact_dynamic_v3c_fresh_dev24.sh preflight
bash scripts/run_water_impact_dynamic_v3c_fresh_dev24.sh register-stage2
bash scripts/run_water_impact_dynamic_v3c_fresh_dev24.sh stage2-preflight
bash scripts/run_water_impact_dynamic_v3c_fresh_dev24.sh original
bash scripts/run_water_impact_dynamic_v3c_fresh_dev24.sh v3b
bash scripts/run_water_impact_dynamic_v3c_fresh_dev24.sh v3c
bash scripts/run_water_impact_dynamic_v3c_fresh_dev24.sh compare
```

Every output directory is reserved by an exclusive `mkdir`; interrupted or
partial runs are not silently reused.

## Blinding and adjudication

The comparison contains v3b and v3c only, with matched prompts and seeds and
an Original reference row in each composite. The public package contains a
blank 48-row review CSV, 24 composites, and 48 independently copied anonymous
candidate videos. The private sibling contains only the answer key and the
hash-binding provenance manifest. No method label or source generation path is
present in the public package.

Two reviewers independently inspect all full 49-frame videos and score target
visibility, footprint visibility, receiver preservation, and video quality on
the frozen 0/1/2 rubric. Exact atomic agreement is canonical. Every atomic
disagreement is scored by a third blinded adjudicator. The canonical value is
the majority of three; an exact `0,1,2` split has median value 1. The scorer
requires exactly the disputed fields—no missing or unsolicited adjudications—
and records the hashes of both reviews and the adjudication file before
unblinding.

## All-or-nothing gate

Let `usable(x)` mean receiver and quality are both at least 1, and let
`C = {i: usable(v3b_i)}`. On C, an unusable v3c output contributes zero target
and footprint suppression points. Absent-target counts likewise include only
usable outputs, so collapse cannot create suppression credit.

V3c is promoted and final36 is unsealed only if every condition holds:

- v3b has at least 20 usable controls;
- valid v3c target suppression on C is at least v3b plus 6 points;
- at least six samples in C have a usable v3c target-visibility improvement;
- at least two improvements are clear-to-absent (`2 -> 0`) and span at least
  two generalization groups;
- v3c has at least two more usable absent-target outputs than v3b;
- v3c has at least 22 usable outputs;
- v3c receiver points are at least `max(38, v3b_receiver - 2)`;
- v3c quality points are at least `max(32, v3b_quality - 2)`;
- valid v3c footprint suppression on C is not below v3b;
- v3c has at least two strict `(0,0,2,2)` successes.

Failure of any single condition records v3c as a negative development
ablation. It does not authorize a teacher-weight, sigma-window, checkpoint,
seed, or schedule sweep.
