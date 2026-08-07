# Experiment Protocol v1

Status: frozen for the first full-scale experiment batch.

## Claim and adapter unit

One adapter targets one causal mechanism family. It removes the source object participating in that mechanism and the downstream causal footprint, while preserving the receiver, background, and unrelated mechanisms.

Protocol v1 contains four mechanism families:

1. Water impact.
2. Rigid collision.
3. Brittle fracture.
4. Powder impact.

The adapter is mechanism-specific, not lexical-object-specific. For example, the water-impact adapter is expected to transfer from droplets to unseen stones or balls entering water. It is not claimed to selectively remove only one named object when several objects instantiate the same target mechanism.

## Frozen registry

The exact source objects, receivers, descriptions, clean states, footprints, and train/test membership are stored in `data/protocol_v1/registry.json`.

Each mechanism has:

- three training source objects;
- two held-out source objects;
- three training receivers;
- two held-out receivers.

Train and test source IDs are disjoint within each mechanism. Train and test receiver IDs are also disjoint within each mechanism. Reusing an object across different mechanisms is intentional: it tests whether an adapter follows the mechanism instead of merely reacting to an object word.

## Shared prompt-level dataset

Wan2.1-T2V-1.3B and CogVideoX-2B use the same protocol rows, prompt text, seed, object split, receiver split, frame count, and FPS. Each backbone renders its own video realization from those shared specifications. No sample is removed after seeing a backbone's result.

All factual prompts require a fixed camera and a clean initial interval. The source object is absent during the first two seconds and enters only afterward. This makes the clean prefix usable as an aligned counterfactual reference.

The fixed generation settings are:

- one seed per prompt;
- seed 12000 plus a fixed mechanism offset for evaluation;
- 49 frames;
- 8 FPS;
- first 16 frames reserved as the clean prefix.

## Erase training set

For each mechanism:

```text
3 training sources x 3 training receivers x 4 prompt variants = 36 erase rows
```

Across four mechanisms, Protocol v1 contains 144 erase rows.

The four training prompt variants are:

1. explicit full causal chain;
2. concise full causal chain;
3. contact-only wording without naming the footprint;
4. natural implicit wording.

No repeated seeds are used to inflate the dataset.

## Counterfactual construction

For every factual training video:

1. decode the first 16 clean frames;
2. compute their per-pixel temporal median;
3. repeat that clean reference frame for all 49 frames;
4. save the result as the aligned counterfactual video;
5. compute the latent factual/counterfactual residual;
6. smooth and normalize that residual to obtain the causal mask.

Negative-prompt generation and manual image editing are not used to construct the counterfactual target.

This construction is valid only for the fixed-camera, initially clean scenes defined by this protocol.

## Shared preservation bank

Protocol v1 includes 36 generic preservation prompts: 12 ordinary non-impact scenes with three wording variants each. They include static scenes and smooth motion but exclude all four target mechanisms. The same preservation bank is used by every adapter.

Other target mechanisms are not included as training preservation examples. Cross-mechanism preservation is evaluated directly, which avoids teaching the method only a small enumerated list of protected mechanisms.

Each adapter therefore trains with:

```text
36 erase rows + 36 preserve rows = 72 unique training rows
```

Erase and preserve roles are sampled in a balanced alternating schedule.

## Evaluation set

Each mechanism has 20 evaluation rows:

| Generalization group | Count |
| --- | ---: |
| Seen source, seen receiver | 5 |
| Unseen source, seen receiver | 5 |
| Seen source, unseen receiver | 5 |
| Unseen source, unseen receiver | 5 |

The full evaluation set contains 80 prompt-level rows. Evaluation wording is distinct from training wording. Wan and CogVideoX use the same rows and seeds. Base-generation failures remain in the manifest and cannot be counted as successful erasure.

## Frozen method configuration

The first full batch uses one shared configuration for all mechanisms. Per-mechanism tuning is not allowed in the main result.

- LoRA rank: 16.
- LoRA alpha: 16.
- learning rate: 1e-4.
- optimization steps: 150.
- causal mask weight: 8.0.
- background weight: 1.0.
- paired-separation weight: 1.0.
- pair margin: 0.05.
- factual redirect weight: 0.5.
- preserve weight: 8.0.
- inference LoRA scale: 0.75.
- objective: mask-weighted dual trajectory.

One adapter is trained for each mechanism and backbone, producing eight adapters in total.

## Compared methods

Every applicable evaluation row is generated with:

1. frozen original model;
2. negative prompting;
3. T2VUnlearning;
4. VideoEraser;
5. the proposed adapter.

Methods use the same prompt, seed, frame count, resolution, and backbone wherever their interfaces allow it. Baseline parameters are fixed per backbone and are not tuned per mechanism.

## Evaluation axes

The main evaluation reports:

1. source-object removal;
2. causal-footprint removal;
3. receiver and background preservation;
4. preservation on unrelated mechanisms;
5. base-model causal capability, reported separately from erasure.

Temporal activity suppression and early-frame MAE are supporting proxy metrics, not semantic proof of successful erasure. Human semantic review is required for the main object, footprint, and receiver judgments. VBench is deferred to the final general-quality evaluation.

## Change control

The generated manifests and their SHA-256 hashes are stored in `data/protocol_v1/manifest_summary.json`. Protocol v1 rows must not be edited after the Git tag is created. A methodological, data, split, prompt, seed, or hyperparameter change requires a new protocol version rather than modifying v1 in place.

Bug fixes that do not change the semantic experiment definition must be documented with the exact Git commit used for affected runs.
