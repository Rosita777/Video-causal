# Waterdrop generalization expansion16 (2026-08-02)

This run expands the waterdrop training candidate pool with 16 fixed-seed prompts across four receiver families. Wan generation used two shards on GPUs 2 and 3, 49 frames per video, 8 fps, and 25 inference steps.

Automatic screening found 5 candidates with a measurable clean prefix. Manual review accepted all five because the causal order is visually clear: the receiver is stable first, the droplet arrives, and the footprint appears after contact.

Accepted pairs:

- `wdgen001`: stainless-steel saucepan -> splash/ripples
- `wdgen005`: sealed hardwood tabletop -> wet mark/splash
- `wdgen006`: clear acrylic sheet -> splash/ripples
- `wdgen008`: white paper towel -> spreading wet patch
- `wdgen014`: yellow cornmeal -> crater/damp particle mark

The resulting v2 split has 60 aligned pairs: 31 train candidates, 16 internal receiver holdouts, and 13 reserved external-eval overlaps. The five accepted pairs are training candidates only; they are not used to redefine the frozen evaluation set.

This is still a waterdrop-only generalization test. It tests whether one waterdrop adapter can handle unseen receivers and several footprint families. It does not claim transfer to other mechanisms such as glass breaking.
