# Causal Attention Mask B2 Design

## Goal

Extend the Method B diagnostic probe into a small intervention test. B2 asks
whether suppressing cross-attention to causal-chain tokens changes generated
videos more specifically than matched random-token or matched module controls.

## Scope

B2 is still an exploratory method test, not a paper claim. It will generate a
small matrix over the first MVP-0 items and compare:

- `baseline`: no attention intervention,
- `target_mask`: suppress target-token attention,
- `footprint_mask`: suppress footprint-token attention,
- `chain_mask`: suppress target plus footprint tokens,
- `random_token_mask`: suppress the same number of non-chain text tokens,
- `comparison_token_mask`: suppress deterministic comparison tokens already
  used by the B probe.

The first run uses three items and a compact generation setting. A later run
may scale only if `chain_mask` beats matched controls under VLM review without
obvious global video collapse.

## Intervention

The existing `RecordingAttnProcessor` becomes an intervention-capable
processor. It computes attention probabilities normally, optionally rescales
selected token columns, renormalizes each attention row, records the resulting
text-conditioned attention mass, and continues the usual attention output.

The intervention applies only to cross-attention (`attn2`) and only to the
text-conditioned CFG half of the expanded video batch. The unconditional half
is left untouched. This preserves the baseline CFG structure and prevents
unconditional-prompt artifacts from contaminating the causal-token intervention.

The first intervention is multiplicative suppression:

```text
attention[:, :, selected_token_indices] *= mask_scale
attention = attention / attention.sum(dim=-1, keepdim=True)
```

with `mask_scale=0.0` by default for a hard ablation. The CLI also supports
weaker values such as `0.25`.

## Outputs

Each generated row writes:

```text
sample.mp4
attention_trace.jsonl
attention_summary.csv
```

The top-level manifest records the condition, selected token indices, mask
scale, seed, and output paths. The attention files are kept for all conditions,
including controls, so failures can be diagnosed as either no intervention
effect or excessive global attention damage.

## Controls And Success Gate

The control rows must match the number of masked token indices. `random_token`
selection is deterministic from `seed + probe_index` and excludes special,
target, and footprint tokens. `comparison_token_mask` uses the B probe's
deterministic comparison indices.

B2 is worth evaluating with fable only if:

- all rows generate without crashes,
- attention traces show lower selected-token mass after masking,
- chain masking does not simply destroy all videos,
- `chain_mask` visually differs from matched token controls on at least one
  item.

Even if these pass, the result remains exploratory until fable/VLM evaluation
shows chain-specific improvement over matched controls.
