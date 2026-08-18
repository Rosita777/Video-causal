# Water-impact dynamic v4 pre-Stage-0 termination (`v4_dev72_v3`)

Recorded: 2026-08-18

## Outcome

`v4_dev72_v3` is scientifically exhausted before the pending Stage-0 data
freeze. The registered isolated identity-disjointness audit found two fresh
source normalized-head intersections with the frozen v2 candidate inventory.
Fresh-source intersections are required to be zero; the only registered
identity exceptions are eight original-source nodes and eight historical
receiver nodes.

This is an ontology/disjointness failure, not a static implementation defect
and not an evaluation result. No same-version source, head, ontology, graph,
exception, salt, or seed may be changed. No Original media was generated.

## Aggregate identity result

The formal identity auditor failed closed and published no standard report.
An aggregate-only diagnostic using the same committed comparison logic
reported:

| Registered intersection | Count |
|---|---:|
| `case_id` | 0 |
| `canonical_record` | 0 |
| `fresh_source_id` | 2 |
| `fresh_receiver_id` | 0 |
| `source_receiver_pair` | 0 |
| `source_receiver_variant_triple` | 0 |

The `fresh_source_id` aggregate was further decomposed without exposing any
identity value:

| Fresh-source comparison component | Count |
|---|---:|
| exact source ID | 0 |
| normalized full phrase | 0 |
| normalized head | 2 |

The two affected rows had only the normalized-head flag set. That still
violates the preregistered identity rule: both v2 and v3 head values are
normalized before comparison, and fresh sources are not covered by either
historical exception.

## Other prerequisite audits

- The construct-equivalence auditor failed closed and published no report.
  Template bytes, field-rule bytes, and qualification objects passed before
  the failure. The failure was the exact `cell_quota` representation: the
  frozen v2 input is a semantic string while the v3 input is a structured
  object. Semantic similarity is not an exact-byte pass and is not repaired
  after the identity failure.
- The forbidden-seed audit passed with relation `strict_superset`: 26,196 v2
  seeds and 26,201 v3 seeds. Its standard aggregate report SHA-256 is
  `5d7c02f02efb7fc1a5e8c8e5a65e080a309889f69d01800014bbe5733793fd11`.

## Bound evidence

No private identity, phrase, head value, prompt, seed value, salt, row, or
media content is included here.

| Artifact | Digest |
|---|---|
| Git implementation HEAD | `dcc0046c6fc484f6d54aeb3b3369f7c097723e79` |
| V3 preregistration | `614617d14ae5608ea6048b91b46e74fdf6ada25222ced140b451a959ca9a8247` |
| Model content inventory | `51e7199b99ee206934924ee043bd01b40ba413dfb60ef1e72682d36a10b46290` |
| Runtime registry | `9043adf7f823022b20711267ab9e28b9dfb72452273d26e27c131b92e65eff01` |
| Code registry | `804bb5bc7f74f813f93826fc63f10a8c4e6bfce1357a6d94c8ece8f611fa84ee` |
| Capacity model | `223392b9884e23e839000d9d95ffde8fc318f084187a61de71f02a166141fc33` |
| Static graph audit | `1db1e06264adb868c43aab12356413b091cca0431875cc60067698efd48a001a` |
| Prepared private inventory aggregate | `149713ef91362d41c4f6e97f18c01114395b07dec13932d57acd330bbbb354c6` |
| Forbidden-seed source audit | `5d7c02f02efb7fc1a5e8c8e5a65e080a309889f69d01800014bbe5733793fd11` |

The private preparation completed before the isolated audits: exact 19
mode-600, single-link regular files; 576 candidate records; and 1,728 unique
evaluation-seed audit records with zero screening or forbidden-seed
collisions. These retained artifacts are evidence only and do not constitute
a pending or authorized Stage 0.

## Machine-state boundary

The identity report is a prerequisite for the public holdout commitment, and
the holdout commitment is a prerequisite for the pending Stage-0 freeze.
Because identity failed, none of those artifacts exists. The current protocol
only authorizes a machine-readable scientific invalid outcome after the
pending freeze has been validated. Therefore no
`causal_preflight_dataset_invalid_v3.json` was emitted, and the internal
authorizer invalid helper was not called directly.

This distinction is intentional:

- scientific/version state: exhausted and not repairable within
  `v4_dev72_v3`;
- machine state: stopped before pending, with no schema-defined invalid JSON.

At termination, all of the following are absent: identity report, construct
report, holdout public commitment, cost calibration, pending Stage-0
commitment, authorizing Stage-0 wrapper, invalid-outcome JSON, Stage-1
commitment, Original media, review package, review scores, eligibility, and
selector outputs. Sealed-final36 remains unopened.

## Consequences

Do not rerun or weaken either failed auditor, edit or replace either
overlapping source, add an exception, rebuild the graph, change a prompt,
reuse a v3 row, resample a secret, launch cost calibration, publish pending or
Stage 0, generate media, or begin review.

Any future continuation requires a new dataset version with new ontologies,
case and receiver identities, media, and fully rotated salts and seeds. Its
preregistration must also define a machine-readable pre-pending scientific
termination path and use one exact canonical representation for the shared
cell-quota construct.
