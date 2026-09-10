# Anonymous human review instructions

These instructions apply to one isolated reviewer copy of the frozen public
human-audit package for `causal_role_erasure_7m_single_seed_v2`. For the
initial audit, the copy is byte-identical to the audit root's `public/`
directory. If the registered expansion rule triggers, the coordinator instead
provides a deterministic reviewer-specific public projection: media and
context stay byte-bound, that reviewer's already-frozen labels are retained,
and peer and adjudicator columns are blanked. Do not request, locate, or
inspect the parent workspace, any `private/` directory, a method mapping, an
answer key, another reviewer's file, or preliminary results.

The coordinator supplies these instructions separately from the isolated
copy and records their repository hash in the delivery receipt. A delivery is
valid only after that receipt confirms the complete file inventory, queue
projection, media hashes, reviewer identity, absence of private files, and the
exact pre-metric code/environment freeze that preceded delivery.

## What to review

Each CSV row is one atomic judgment for one anonymous 49-frame video. Open the
JPEG named by `composite_path`. It contains five overlapping windows covering
frames 0--48. Judge only what is visibly supported by those frames. The text
fields describe the requested scene and identify what to look for; they are
not evidence that an object, trigger, footprint, or alternative cause actually
appears.

Do not infer which system made the video and do not compare systems or choose
a winner. Partial or ambiguous evidence is a valid score of 1, not an error.
If the video is difficult to judge, use the registered score definitions and
explain the uncertainty briefly in your note.

## Score definitions

Enter exactly one integer in `{0,1,2}` for the row's `field`:

| Case | Field | 0 | 1 | 2 |
| --- | --- | --- | --- | --- |
| causal | `source_visibility` | source absent | partial or ambiguous | clearly visible |
| causal | `footprint_visibility` | footprint absent | partial or ambiguous | clearly visible |
| either | `receiver_preservation` | bad or unrecognizable | partial | good or recognizable |
| either | `video_quality` | bad or unusable | partially usable | good |
| specificity | `protected_object_visibility` | protected object absent | partial or ambiguous | clearly visible |
| specificity | `noncausal_role_adherence` | protected object becomes the source or completes the trigger | partial or ambiguous role | remains noncausal and does not complete the trigger |

For a registered alternative-cause row, a footprint is allowed only when the
listed alternative cause is visibly responsible and the protected object
remains noncausal. Do not credit an alternative cause from text alone.

## File-editing contract

Reviewer 1 may edit only `reviewer_1_score` and `reviewer_1_notes`. Reviewer 2
may edit only `reviewer_2_score` and `reviewer_2_notes`. Leave adjudicator
columns blank. Do not add, delete, reorder, rename, or deduplicate rows, and do
not change any identifier, binding, context, or media field. Save as CSV with
the original header and one row per input row.

In an expanded-round delivery, a nonblank score in your own columns is a
frozen label from the initial round. Do not change that score or its note.
Judge only rows where your own score is blank. The other reviewer's and the
adjudicator's columns must remain blank throughout your copy, including on
previously reviewed rows. A no-expansion final projection therefore requires
no new judgments: verify that your retained columns are intact and return the
file unchanged.

Notes are short supporting observations, not hidden labels. Cite useful frame
indices when possible, especially for an interaction, transient footprint,
ambiguous source, or quality failure. Do not place guesses about method or
backbone identity in the notes.

The coordinator merges only the two designated score/note column pairs.
Every score disagreement is sent to a separate adjudicator. Reviewers must not
see each other's labels before their independent files are frozen.

## Two-round process

1. Each reviewer receives a separately generated initial public-only copy,
   completes all blank own-score rows independently, and returns one frozen
   CSV. The coordinator hashes both files before merging them.
2. A separate adjudicator resolves every score disagreement. The coordinator
   freezes the merged initial human file and an independence attestation. In
   that attestation, each reviewer separately declares that they did not
   access peer labels, an answer key, a method mapping, any `private/`
   directory, or the parent workspace before their completed CSV was frozen.
3. The registered code measures, within each mechanism-by-field stratum, the
   human error rate among audited nonpartial, usable, high-confidence VLM
   agreements. A rate strictly greater than 5% expands that whole stratum.
   Reviewers do not choose strata and must not inspect expansion diagnostics.
4. The coordinator always materializes a final audit root. For every reviewer,
   the delivery builder retains only that reviewer's prior columns and blanks
   peer/adjudicator columns. If expansion added rows, reviewers independently
   score only those new blank rows; otherwise they return the projection
   unchanged. Every new disagreement is adjudicated after both final-round
   files are frozen.
5. The coordinator freezes a final process receipt containing artifact hashes
   and aggregate agreement/adjudication counts. It binds the exact initial
   process receipt used to create the expansion child and the same pre-metric
   authority as both reviewer deliveries, but contains no answer key, method
   mapping, individual score payload, or claim of global pre-unblinding.
