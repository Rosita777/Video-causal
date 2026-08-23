# Eight-Mechanism Original Capability v2

This amendment registers a fresh Original-only capability screen. It does not
modify or reuse the v1 capability ontology and does not authorize training,
treatment generation, evaluation-case selection, or access to sealed formal
evaluation material.

## Frozen semantic batch

The batch contains eight mechanisms in canonical order. Each mechanism crosses
two fresh source identities with two fresh receiver identities under direct and
natural wording, producing eight semantic cases. Every case has three explicit
deterministic seeds, for 24 rows per mechanism and 192 rows overall. Seeds use
`940000 + 1000*mechanism_index + 10*combination_index + repetition_index`.

Every prompt contains exactly two sentences and at most 55 English words. The
first sentence states a simple receiver state. The second sentence contains one
source action and one visible result. Human actors and negative-prompt lists are
excluded. Field-mediated response uses non-contact airflow from a desk fan to
move either a paper pinwheel or a ribbon across a visible gap. Elastic
deformation uses only a black round trampoline and a blue square trampoline.
Frames 0--15 remain diagnostic metadata and are not a v2 eligibility gate.

## Generation contract

The formal Original batch uses Wan with 25 inference steps, guidance scale 5,
49 frames at 8 fps, 480 by 832 resolution, and bf16. Eight mechanism jobs run as
two fail-closed waves of four isolated GPU processes. Each process receives its
24 explicit canonical seeds. Existing outputs are never resumed or skipped.

## Stage and sealed-data boundary

Formal stage preparation requires an exact clean tracked snapshot plus exactly
the eight frozen public evidence files and the hash-bound v1 capability stage
registry as pre-existing untracked inputs. The v2 launcher then creates its own
exclusive stage registry. Every formal run reopens that registry by caller-
supplied SHA-256 and revalidates code, data, model, runtime, git state, public
evidence, and predecessor-stage bindings before reserving an output directory
and at every wave boundary.

No path containing `sealed` or `final36` is accepted. The launcher rejects such
paths before reading a manifest or prompt file, and formal actions require an
explicit unopened-sealed-data attestation.

## Output validation

Each mechanism must produce exactly 24 videos in canonical order. The frozen
Wan runtime uses PyAV to fully decode every video and requires one video stream,
49 decoded frames, 8 fps, and 480 by 832 dimensions. Every video is SHA-256
bound after initial validation and rehashed and decoded again before the final
192-video generation manifest is written exclusively.

The capability review gate is aggregate-only. A mechanism requires at least
12 eligible rows, at least six eligible direct rows, at least six eligible
natural rows, and eligible coverage of both sources and both receivers. All
eight mechanisms must pass.
