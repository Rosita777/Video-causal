# Causal Attention Dependency Probe Design

## Goal

Phase B+ showed that prompt-pair residual steering does not beat matched
random and orthogonal controls. The next method should stop treating prompt
embedding differences as causal directions and instead inspect the model's
white-box dependency path. This probe asks whether target tokens have
measurable cross-attention influence over video latent positions during
generation.

## Scope

This first B pass is diagnostic, not a repair claim. It will:

- identify target-token indices in the CLIP tokenizer,
- hook ZeroScope UNet cross-attention modules during generation,
- summarize attention mass assigned to target tokens versus matched
  non-target tokens,
- write a compact JSON/CSV report for one or a few MVP-0 items.

It will not yet mask attention, edit videos, or claim causal repair. Those are
only justified if the dependency probe finds a stable signal.

## Design

The probe wraps Diffusers attention processors with a recorder. For each
cross-attention call, the wrapper computes attention probabilities, records
summary statistics for selected token indices, and then returns the normal
attention output. To avoid huge memory use, it records only aggregated values:
mean attention mass for target tokens, comparison tokens, and all text tokens,
grouped by attention module and denoising step.

The script will run the same prompt rows used in MVP-0 but with a small
configuration first: one item, few steps, low resolution. Outputs are:

```text
attention_trace.jsonl
attention_summary.csv
generation_manifest.json
```

The first success criterion is instrumentation correctness: attention calls are
captured, token indices resolve to the intended target words, and generation
still completes. The second criterion is signal quality: target-token attention
should differ from matched comparison tokens in a repeatable way on at least
some layers or timestep bands. If target attention is flat or indistinguishable
from controls, the attention-intervention method should stop before masking.

## Controls

The probe reports three comparisons:

- target token indices,
- footprint token indices,
- matched non-target tokens from the same prompt.

Later intervention tests must include random matched token masks and random
matched layer/head masks. This first diagnostic only prepares those controls.

## Success Gate

B-probe is worth extending to masking only if:

- the recorder captures cross-attention from multiple UNet modules without
  breaking generation,
- target and footprint token indices are found for each tested prompt,
- at least one layer/timestep band shows a clear target-or-footprint attention
  elevation over comparison tokens.

If those conditions fail, the method is not yet paper-worthy and should shift
to counterfactual-pair training or verifier-guided selection instead.
