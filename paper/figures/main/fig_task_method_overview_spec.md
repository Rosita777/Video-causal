# Combined Figure 1: task and method overview

The user-approved reference-style slide replaces the former separate task
teaser and method figure. The Introduction includes it once, with label
fig:task-method-overview; Method refers back to the same figure.

## Composition and scientific meaning

- Upper tier: structured event, two factual prompt variants, and a common
  independently generated, screened source-free proxy. The ball/pebble
  prompts are schematic examples, not literal registered training rows.
- The source-free drawings describe the desired outcome. They are not
  measured model outputs or guarantees that every screened proxy is perfect.
- The noncausal ball denotes evaluation specificity, not hard-negative
  training data.
- Lower tier: shared noised target latent and timestep, frozen teacher
  conditioned on source-free text, separate mechanism-specific Matched/SRCD
  LoRA arms, and separate alternating erasure/preservation updates.
- The comparison isolates factual source identity. Targets, stochastic draws,
  objectives, schedule, initialization, and remaining training inputs are
  shared. Full fixed-field lists and the 100-of-178 active schedule stay in
  Method and Appendix C.

## Publication preparation

The approved 1280 by 580 pixel slide keeps its palette, pictograms, and
layout. Small labels are shortened and enlarged for a 5.5-inch figure.
The ten Lucide icons use the original SVG paths rather than raster previews.
The slide export must preserve searchable embedded-font text and vector
paths; the PDF must contain no raster images.

1. Run paper/scripts/prepare_task_method_figure.mjs on the approved PPTX,
   with RUNTIME_NODE_MODULES set to the available Artifact Tool runtime.
2. Validate the staged PPTX with the presentation finalizer.
3. Export through bundled LibreOffice with a task-local font configuration
   that resolves the source's Arial fonts.
4. Run paper/scripts/finalize_task_method_pdf.py to scale the vector page
   to 396 points and record sanitized source/output hashes.
5. Compile all three paper drivers and inspect the figure at manuscript scale.

The adjacent manifest records selected files, physical dimensions, fonts,
vector/raster checks, and SHA-256 values. Schematic selection is explicitly
not applicable. Future qualitative Figure 2 has a separate selection protocol.
