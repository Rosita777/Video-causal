# Dynamic water-impact eval12: first main comparison

The first controlled comparison uses 12 prompts with identical seeds and Wan
2.1 T2V 1.3B generation settings for every method. The split contains unseen
source objects, unseen receivers, and samples where both are unseen.

Methods:

- Original;
- Negative Prompt;
- T2VUnlearning Wan proxy;
- VideoEraser Wan proxy;
- preservation-balanced dynamic SFT v2, checkpoint 200, LoRA scale 1.25.

Each output was reviewed from a seven-frame reference/output sheet. Target and
footprint visibility use `0=absent, 1=partial/weaker, 2=clear`. Receiver
preservation and video quality use `0=bad, 1=partial, 2=good`.

| Method | Target suppression | Footprint suppression | Receiver preservation | Video quality | Usable videos |
|---|---:|---:|---:|---:|---:|
| Original | 0.0 | 0.0 | 100.0 | 100.0 | 100.0 |
| Negative Prompt | 4.2 | 8.3 | 83.3 | 100.0 | 100.0 |
| T2VUnlearning | 91.7 | 100.0 | 0.0 | 0.0 | 0.0 |
| VideoEraser | 91.7 | 100.0 | 0.0 | 0.0 | 0.0 |
| Ours v2 | 4.2 | 37.5 | 83.3 | 95.8 | 91.7 |

These are preliminary manual scores on only 12 samples, not paper-final
numbers. They nevertheless establish the main trade-off. Negative Prompt
largely preserves generation quality but barely suppresses the causal
footprint. The T2VUnlearning and VideoEraser Wan proxies strongly suppress
visible target and footprint evidence by collapsing the requested scene, so
their apparent erasure does not produce usable outputs. Ours v2 improves
footprint suppression over Negative Prompt while preserving most receivers and
video quality, although source-object suppression remains weak.

The main evaluation should therefore report erasure and preservation jointly.
A footprint or target score alone would incorrectly reward scene collapse.

Artifacts:

- `experiments/water_impact_dynamic_eval12/manual_scores_v1.csv`;
- `experiments/water_impact_dynamic_eval12/manual_summary_v1.csv`;
- `scripts/record_water_impact_dynamic_eval12_manual.py`.
