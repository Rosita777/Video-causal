# Final main tables

This directory records the main-table interface contract.  The frozen final
exporter atomically materializes the actual generated files under
`../human_canonical_v1/main/`; it never hand-copies files into this directory.
Planned interfaces:

- tab_main_mechanisms.tex
- tab_main_macro_guardrails.tex
- tab_identification_controls.tex
- tab_generalization.tex

The frozen exporter keeps all four files in this generated interface. Under
the current nine-page plan, the compact identification and generalization
projections are generated for compatibility but are not rendered. Section 5.3
summarizes them in prose and points to the exporter's distinct, more complete
Appendix F1--F3 projections.

Each generated file must begin with comments recording the source result
manifest, source hash, row count, code commit, and canonical status. Generated
tables are never edited by hand.
