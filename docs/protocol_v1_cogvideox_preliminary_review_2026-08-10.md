# Protocol v1 CogVideoX preliminary review

Status: preliminary quality diagnostics and a blinded 16-sample visual pilot. This is not the final semantic evaluation.

## Evaluated outputs

- Original CogVideoX-2B: 80 videos.
- Negative Prompt: 80 videos.
- VideoEraser official pipeline: 80 videos.
- T2VUnlearning-adapted: 80 videos using checkpoint 100.

The adapted T2VUnlearning checkpoint was changed from step 500 to step 100 for review because the step-500 output collapsed toward near-black videos. On the water-impact development sample, step 100 had mean RGB 149.55 while step 500 had mean RGB 15.37. Step 100 is the earliest common saved checkpoint and the only tested checkpoint without severe global degradation. It must be reported as an adapted reproduction, not as an official implementation.

## Automatic quality diagnostics

The diagnostic reads every frame of all 320 videos. It is a supporting quality check, not a semantic erasure metric.

- Original, Negative Prompt, and VideoEraser had 0/80 collapse flags.
- T2VUnlearning-adapted checkpoint 100 had 1/80 collapse flags.
- The collapsed T2V sample was `eval_brittle_fracture_unseen_unseen_03`.
- T2V showed especially large early-frame divergence for brittle fracture, indicating strong non-local side effects.
- VideoEraser also showed visible divergence from Original, especially on powder and water impact, but did not trigger the simple collapse rule.

Exact per-video and mechanism-level diagnostics are stored in:

- `experiments/protocol_v1/cogvideox_quality_ckpt100/per_video_quality.csv`
- `experiments/protocol_v1/cogvideox_quality_ckpt100/summary.csv`

## Blinded visual pilot

The pilot selected one row from each generalization group for each mechanism, for 16 source prompts total. Candidate order was randomized independently per sample.

Preliminary observations:

1. Water impact: Original generally shows the source and ripples. The three baselines usually retain the source, splash, or ripples; no consistent clean erasure was observed in the four pilot rows.
2. Rigid collision: Original frequently fails to show a clear receiver response. These rows must fail the Original capability gate and cannot support a causal-footprint erasure claim.
3. Brittle fracture: Original usually shows a complete impact-to-fracture chain. Baselines generally retain the striking object, cracks, or fragments. Some outputs reduce fracture while retaining the source object.
4. Powder impact: Original generally shows the source, plume, or crater. Baselines usually retain the object and at least one footprint cue.

These observations support running the full Original gate before computing erasure rates. They do not yet provide final method percentages.

## Required next evaluation step

1. Label all 80 Original videos for source visibility, footprint visibility, receiver correctness, and judgeability.
2. Restrict the main conditional erasure rate to Original-gate-positive samples.
3. Label all baseline outputs for source absence, footprint absence, receiver preservation, and quality.
4. Report unconditional failure rates alongside conditional rates so that base-model failures are not hidden.
5. Double-check all automatic labels with low confidence, all apparent successes, and a random sample of failures.
