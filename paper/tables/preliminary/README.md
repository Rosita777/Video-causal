# Preliminary tables - never final

This directory is reserved for explicitly labeled VLM-only drafting previews.
It is intentionally separate from tables/main and tables/appendix.

Rules:

- Every filename ends in _preliminary.tex.
- Every caption states that the values are VLM-only and non-canonical.
- No final section file may silently fall back to this directory.
- Human-canonical completion replaces the result source; it does not overwrite
  or relabel a preliminary artifact.

Current internal artifact:

- `results_writing_preview_v2_preliminary.tex` is a hand-authored,
  post-unblinding VLM-only narrative and table preview. It is loaded only when
  `\ifoutline` is true, carries the frozen preview-manifest digest, and is
  deliberately rejected by the submission gate. It must be removed rather
  than promoted when the generated human-canonical tables become available.
