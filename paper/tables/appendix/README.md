# Final appendix tables

This directory records the appendix-table interface contract.  The frozen
final exporter atomically materializes human-canonical result tables under
`../human_canonical_v1/appendix/`; it never hand-copies generated files into
this directory.  Those outputs cover complete results, registered contrasts,
M6 pairs, mechanism heterogeneity, and reproducibility.

Files must be generated from committed scripts and carry source hashes and
canonical status in LaTeX comments.
