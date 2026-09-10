# Causal-role-erasure reproducibility freeze

This directory defines the release-time evaluation environment for
`causal_role_erasure_7m_single_seed_v2`. It does not reconstruct or claim the
historical training, generation, or hosted-VLM environments. Frozen completed
artifacts remain the downstream authority for those already executed stages.

## Gate order

Run these gates before any human-canonical metric is opened:

1. Commit the amendment, v2 wrappers, table exporter, reviewer process code,
   reproducibility code, lock, reviewer instructions, and all release tests.
   The checkout must then be clean.
2. Create a fresh CPython 3.12 virtual environment outside the repository and
   install `evaluation-reproduction-v1.lock` without adding packages.
3. Run the nine-file release suite through `freeze-environment`. This writes a
   self-hashed `evaluation_environment_receipt_v1.json` outside the repository.
4. Run `build_causal_role_erasure_7mechanism_pre_metric_freeze_v2.py` from the
   same clean commit. Pass the successful environment receipt and a fresh
   output directory outside the checkout. The builder copies the receipt into
   that directory, live-rechecks the exact interpreter, distributions,
   platform, codecs, external tools, and complete 64-pass gate, inventories the
   18 trained-Wan receipts, freezes the full command ledger, builds the prefix
   artifact DAG, and verifies every bound commit, tree, blob, file hash, count,
   and DAG edge before publishing it.
5. Validate the amendment against the resulting
   `pre_metric_code_freeze_v2.json`. Only then prepare reviewer deliveries.
   Every delivery and process receipt binds that manifest's file hash,
   implementation commit/tree, and environment receipt. This establishes
   content-consistent ordering, not an externally signed timestamp.

The exact nine test paths and every pipeline command are emitted by
`build_causal_role_erasure_7mechanism_reproducibility_v1.py`; the command
ledger deliberately stores argument vectors and environment mappings rather
than shell strings. Secret values are represented only by named placeholders.

## Human review rounds

The initial round uses byte-identical copies of the frozen public audit
package. After both reviews and adjudication are frozen, the registered v1
expansion command always creates a final audit root. The delivery builder then
creates one deterministic projection per reviewer: it preserves only that
reviewer's own prior labels and blanks peer and adjudicator columns. Reviewers
score newly blank rows only. This second receipt is required even when no
expansion was triggered, in which case the reviewer returns the projected CSV
unchanged. Each reviewer explicitly attests no access to peer labels, answer
keys, method mappings, private directories, or the parent workspace. The final
receipt also binds the initial receipt whose completed-human file created the
expansion/no-expansion child. See
`docs/causal_role_erasure_7mechanism_human_review_v2.md`.

## Availability boundaries

- The local snapshot does not contain the 18 original training-receipt bytes.
  The inventory truthfully records that it derives their individually
  validated hashes and identities from the frozen, completed trained-Wan run
  manifest and aggregate; it does not claim to reopen those receipts.
- Six small upstream authority files currently remain on the persistent A100
  project. The pre-metric DAG records only their published 12-hex digest
  prefixes and marks them `declared_unavailable`. Final verification requires
  a full 64-hex digest for any artifact that remains unavailable.
- Human labels, adjudication, canonical scores, Original eligibility, formal
  metrics, final paper tables, and the post-metric release DAG remain pending
  until their registered stages are executed. Preview or post-unblinding
  preliminary tables are never accepted as substitutes.

After the human-canonical tables exist, run the ledger's
`artifact_dag_final_materialize` stage before `artifact_dag_final`. The builder
requires full SHA-256 values extending all six frozen unavailable-artifact
prefixes, replaces exactly the registered pending branch, binds the complete
pre-metric DAG and freeze as ancestry nodes, and writes
`FINAL_RELEASE_ROOT/artifact_dag_release_v1.json` exclusively. The following
verification stage rejects missing or extra mandatory nodes and revalidates
every final manifest/status link against the frozen prefix.

All generated receipts, reviewer deliveries, and pre-metric artifacts must be
written outside the Git checkout. The only final table output permitted inside
the repository is `paper/tables/human_canonical_v1`, produced by the frozen
exporter after canonical metrics exist.
