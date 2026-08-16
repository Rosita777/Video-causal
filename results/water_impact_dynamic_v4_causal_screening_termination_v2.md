# Water-impact dynamic v4 causal-screening termination (`v4_dev72_v2`)

Recorded: 2026-08-17

## Formal outcome

`preflight_dataset_invalid`

This is not `inconclusive_invalid_evaluation`. The registered outcome taxonomy
in `docs/water_impact_dynamic_v4_source_slot_randomization.md` reserves
`preflight_dataset_invalid` for a Stage-0/Stage-1 selector that cannot produce
a valid frozen dataset before an eligible checkpoint exists.
`inconclusive_invalid_evaluation` applies only after an eligible checkpoint
exists and an evaluation-validity condition fails. No v4 training was
authorized or launched, and no v4 checkpoint exists.

## Frozen causal-screening result

The committed Original-only screen completed for all 48 causal candidates.
The independent reviews, atomic disputes, third-reviewer adjudication,
canonical eligibility table, adjudication audit, and screening freeze manifest
were frozen successfully. The canonical table contains 24 eligible candidates
and 21 adjudicated atomic disagreements.

The frozen aggregate eligibility counts are:

| Registered cell | Eligible | Required |
|---|---:|---:|
| `holdout_source_new_receiver:direct` | 4 | 4 |
| `holdout_source_new_receiver:natural` | 1 | 4 |
| `holdout_source_seen_receiver:direct` | 6 | 4 |
| `holdout_source_seen_receiver:natural` | 8 | 4 |
| `seen_source_new_receiver:direct` | 4 | 4 |
| `seen_source_new_receiver:natural` | 1 | 4 |

The registered selector requires exactly four qualified cases from every one
of the six `group x prompt_variant` cells. Two cells contain only one eligible
candidate each. Therefore no 24-case constrained subset can satisfy the cell
quota, irrespective of the remaining global subset constraints. The selector
failed closed and produced no selector output, selected-24 manifest, or U72
manifest.

## Bound evidence

The termination decision is bound to the following immutable identifiers. No
private candidate identity, prompt, seed, row-level score, or media content is
included here.

| Artifact | Digest |
|---|---|
| Git implementation HEAD | `1d90956d72139b2fa97c1ce97c1dc608fac117e8` |
| Causal Stage-0 commitment wrapper | `29696ad8031bb164fe1c6819c8c382d7e4e828835f750f0d245e4877d4167b38` |
| Screening generation manifest | `1270e1bc01666c6b7e5fe5b4d3cb4c92b3a8722c2dd7f3873ccf7a0480bd0c18` |
| Screening raw manifest | `319c55858ed2f2cb55309dfcf4ca322bf41758d7c5b803f049e52d5ed5ec9bd6` |
| Screening package manifest | `9003e1582ee4ee3d5a0f899a5a0a5e87c9d6cfa8a83ed4262db19c757c0fa3e9` |
| Blank review template | `dcd61ff7666b214c173581bce8e84e0b7777df31a3a890dd28a93af451c5b888` |
| Reviewer A raw review | `cc441831271885f924b949e252dd4c9a528448a74cd399c48303677d31f8bfad` |
| Reviewer B raw review | `127b33cb92afa71e27696d6640de25c4bccda491d7383170799e406eeea72021` |
| Atomic dispute manifest | `9279aa23488d981ef42d6b88fd634e3476ac18452bb2125de46601572fea43a7` |
| Third-reviewer adjudication (J) | `7395630007015c8d2b0078c2a212a888fca2f454f119f7b9aa09b112050c44a3` |
| Canonical eligibility table | `8749020484ad927705ba1a2fbab969cd52acf669b85941e0824dda2236d811f7` |
| Adjudication audit | `4bd1e42b2587a62d4a72e221d86e542a9a4b2b8261f65e389a3e45b453747434` |
| Screening freeze manifest | `52ac630edd16d234742f4b3cdd75b840e2c2b3a3d2f73f23797fae91ecd59cb5` |
| Fail-closed selector stderr | `a2c7a536ffb72beaa4b110daf6b3da8b61312991a65ae8717b541fbf55d59731` |

Digests in the table are SHA-256 except the explicitly identified Git commit.

## Consequences

There is no causal Stage-1 commitment, specificity dataset, prompt sidecar,
training authorization, v4 training run, v4 checkpoint, treatment generation,
or v4 evaluation. Sealed-final36 remains unopened and must stay sealed.

`v4_dev72_v2` is exhausted as a dataset version. Do not retry its selector,
replace or reorder candidates, substitute prompts, sources, or receivers,
change any screening or evaluation seed, or create a reserve queue within this
version. Any future experiment would require a separately justified and
preregistered version; this termination does not authorize one.
