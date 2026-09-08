# Documentation Index

## Current

- [`PROJECT_HANDOFF.md`](PROJECT_HANDOFF.md): authoritative current state and
  handoff instructions.
- [`causal_role_erasure_7mechanism_protocol_v2.md`](causal_role_erasure_7mechanism_protocol_v2.md):
  pre-treatment/freeze-time seven-mechanism contract. Its header intentionally
  records the state at freeze; current execution status lives in the handoff.
- [`causal_role_erasure_7mechanism_baseline_launch_v2.md`](causal_role_erasure_7mechanism_baseline_launch_v2.md):
  registered CogVideoX baseline implementations and launch contract.
- [`baseline_reproduction_research_2026-08-09.md`](baseline_reproduction_research_2026-08-09.md):
  official-versus-adapted baseline fidelity record.
- [`../paper/WRITING_PLAN.md`](../paper/WRITING_PLAN.md): paper order,
  claim-to-evidence matrix, and readiness gates.

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
- `water_impact_dynamic_v3*`
- `water_impact_dynamic_v4_dev72*`

When citing a historical result, include its date and do not silently merge its
metrics with the active seven-mechanism formal evaluation.

## Planning and Recovery

- [`restart_plan_2026-07-29.md`](restart_plan_2026-07-29.md): restart decisions.
- [`recovery_status.md`](recovery_status.md): filesystem/repository recovery.
- [`current_open_questions.md`](current_open_questions.md): historical
  June-2026 open questions; not a current task list.
- [`experiment_log.md`](experiment_log.md): chronological raw log; useful for
  forensic detail, but intentionally not a quick-start document.

## Rule

If a new experiment becomes the active paper direction, update
`PROJECT_HANDOFF.md`, this index, and `README.md` in the same commit. Mark the
previous direction historical instead of leaving two competing “current”
descriptions.
