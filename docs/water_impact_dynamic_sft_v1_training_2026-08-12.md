# Water-impact dynamic SFT v1 training run (2026-08-12)

The complete target generation produced 192 videos. No failed sample was
regenerated with another seed. Technical screening and full contact-sheet
review rejected 14 targets and retained 178.

Rejected targets contained one or more of the following: an unrecognizable
receiver, a human hand, an entering object, pouring water, an abrupt white
surface artifact, or almost no temporal motion. The exact row-level reasons are
recorded in `data/water_impact_dynamic_v1/train_targets_v1_screen_final.csv`.

The accepted set still covers all 8 training source objects and all 12 training
receivers. Counts are 88 direct prompts and 90 natural prompts.

The first training run is deliberately simple:

- Wan 2.1 T2V 1.3B backbone;
- factual causal prompt mapped to a separately generated dynamic
  counterfactual target;
- plain flow-matching LoRA SFT;
- rank 16, alpha 16, learning rate 1e-4;
- 300 optimization steps, checkpoint every 50 steps;
- no residual mask, activation gate, or paired loss.

The run uses `scripts/run_water_impact_dynamic_sft_v1.sh`. Video latents and
prompt embeddings are cached before optimization, so later hyperparameter runs
can reuse the expensive preprocessing stage.
