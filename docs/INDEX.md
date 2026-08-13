# Documentation Index

## Current

- [`PROJECT_HANDOFF.md`](PROJECT_HANDOFF.md): authoritative current state and
  handoff instructions.
- [`water_impact_dynamic_counterfactual_v1.md`](water_impact_dynamic_counterfactual_v1.md):
  current pair construction and data gate.
- [`water_impact_dynamic_sft_v1_training_2026-08-12.md`](water_impact_dynamic_sft_v1_training_2026-08-12.md):
  dynamic SFT v1 data-generation record.
- [`water_impact_dynamic_eval12_results_2026-08-13.md`](water_impact_dynamic_eval12_results_2026-08-13.md):
  current preliminary method/baseline comparison.
- [`baseline_setup.md`](baseline_setup.md): reusable baseline runner interfaces
  and dependency notes.

## Historical Experiment Records

These documents preserve earlier waterdrop, collision, fracture, powder, and
prompt-bank experiments. They are evidence and debugging history, not current
entry points:

- `waterdrop_*`
- `collision_*`
- `five_mechanism_*`
- `apple_flour_*`
- `protocol_v1_*`
- `fable_*`

When citing a historical result, include its date and do not silently merge its
metrics with the current dynamic water-impact eval.

## Planning and Recovery

- [`restart_plan_2026-07-29.md`](restart_plan_2026-07-29.md): restart decisions.
- [`recovery_status.md`](recovery_status.md): filesystem/repository recovery.
- [`current_open_questions.md`](current_open_questions.md): unresolved research
  questions.
- [`experiment_log.md`](experiment_log.md): chronological raw log; useful for
  forensic detail, but intentionally not a quick-start document.

## Rule

If a new experiment becomes the active paper direction, update
`PROJECT_HANDOFF.md`, this index, and `README.md` in the same commit. Mark the
previous direction historical instead of leaving two competing “current”
descriptions.
