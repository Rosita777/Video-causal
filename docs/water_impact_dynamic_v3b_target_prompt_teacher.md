# Water-impact dynamic v3b: target-prompt teacher consistency

Status: pre-registered development ablation. This protocol is frozen before
training or inspecting any v3b generation. Eval12 is a repeatedly used
development set and is not the paper-final main experiment.

## Question and single treatment

The seeded balanced control still leaves the source object visible. V3b tests
whether a source-free text condition can provide a more direct erase signal
without requiring a factual video or pixel-aligned counterfactual pair.

Both arms start from the same seeded LoRA initialization and use the same 200
step balanced schedule (100 erase and 100 preserve), base cache, noise/sigma
RNG, optimizer, and model. V3b is trained from scratch; it does not continue
from checkpoint 200.

For an erase row, with counterfactual target latent `z_cf`:

```text
z_t     = (1 - sigma) * z_cf + sigma * epsilon
student = LoRA(z_t, timestep, factual_prompt)
teacher = frozen_base(z_t, timestep, target_generation_prompt)
L_erase = MSE(student, epsilon - z_cf) + MSE(student, stopgrad(teacher))
```

The teacher term has frozen weight `1.0`. It is added to, not substituted for
or averaged with, the flow loss. The teacher runs with adapters disabled and
shares the exact noisy latent and timestep with the student. Preserve rows keep
the existing `4.0 * MSE(LoRA, frozen_base)` same-prompt objective. No factual
latent, residual mask, token gate, CFG, or pixel alignment is introduced.

The first 16 erase updates are a training-only scale sanity check. The mean
raw teacher/flow loss ratio must be finite and in `[0.1, 10.0]`; otherwise the
run is declared invalid before generation. This check cannot be used to tune
the weight, and no weight sweep is allowed.

## Frozen inputs and control

- train manifest SHA-256: `3d4d8cbf9244b1575357f0ac74380cd7cb6265df4d1a85bf450de4cac120aee4`
- 214-entry base-cache SHA-256: `4654ee67b8937d248963839b744e59f4f535f8be8b8eee421aef264fb3cd4d65`
- target-prompt binding SHA-256: `9b1cc6e5bbdbe60b8f9f8378dc0ea11fea2fe82d8c73d6f9afed3f72b2bd00cc`
- target-prompt rows / unique prompts: `178 / 24`
- Wan model revision: `0fad780a534b6463e45facd96134c9f345acfa5b`
- sidecar cache SHA-256: `6cf7ba0112d8df0e0a5253a7a943411fcfd85c56fc6df580446776b49d1ac9a9`
- sidecar manifest SHA-256: `c467d7f81ee22b2c4b1ff719537487fbfc808eacc98e730c3d24f0a17aca77cb`
- unique target-embedding SHA-256: `a15f5e910358d5e95bcdd995303abb7eb7e7302fd9ee649c4cfebf3b8f6b6330`
- LoRA rank / alpha: `16 / 16`
- learning rate: `5e-5`
- seed / steps: `26000 / 200`
- eval LoRA scale: `1.25`

The completed seeded-balanced control is reused and must remain byte-identical:

- checkpoint artifact: `e61eb33235da3ad68f08e31c451c6690db194bc9b3aa498df58194549955d7f0`
- training state: `91593e27a0bfde232c7cc344a1579e3f1c203d825bd3252e6160532a833f1142`
- eval12 generation manifest: `af1ada55eb56fe28261765e23b4b24b782d403736f6310216602fee069eecf1a`

The target-prompt embeddings live in a separate read-only sidecar. The frozen
214-entry base cache is never modified. The sidecar must contain exactly the
178 erase-row filenames, `(1, 226, 4096)` bfloat16 embeddings, no eval rows,
and no unexpected payloads. Its byte inventory, per-tensor hashes, repeated
prompt consistency, cache manifest, model revision, and prompt binding are all
validated before training.

## Paths and commands

All new artifacts are real paths under the clean v3 clone; they do not use the
legacy `outputs/water_impact_dynamic_v1` symlink for writes.

```text
outputs/water_impact_dynamic_v3b/teacher_prompt_cache_v1
outputs/water_impact_dynamic_v3b/adapter_target_prompt_teacher_v1
outputs/water_impact_dynamic_v3b/eval12_target_prompt_teacher_v1_ckpt200_scale1p25
```

Prepare the sidecar once, freeze its printed inventory hash in this document
and the launcher, then train:

```bash
CUDA_VISIBLE_DEVICES=1 bash scripts/run_water_impact_dynamic_sft_v3b_teacher.sh prepare
CUDA_VISIBLE_DEVICES=1 bash scripts/run_water_impact_dynamic_sft_v3b_teacher.sh train
```

Only `checkpoint-000200` may be evaluated. Training loss cannot select a
checkpoint or change the registered weight.

## Blinded eval12 development gate

The comparison is seeded-balanced versus v3b in a new paired, blinded review.
The reviewer scores target visibility, footprint visibility, receiver
preservation, and video quality on `0/1/2`. A video is usable when receiver and
quality are both at least 1. An unusable v3b video receives zero suppression
points in paired aggregate calculations.

Let `C` be the control-usable samples. V3b is mechanism-positive only if all of
the following hold:

- at least three samples in `C` with control target visibility `2` improve to
  v3b target visibility `0` or `1` while v3b remains usable;
- at least one of those reaches target visibility `0`;
- those improvements cover at least two generalization groups;
- on `C`, v3b target-suppression points are at least control plus 3.

Promotion as the new development operating point additionally requires:

- at least 11 of 12 v3b videos usable;
- receiver-preservation total at least `max(19, control - 1)`;
- video-quality total at least `max(16, control - 1)`;
- on `C`, v3b footprint-suppression points at least the control total;
- at least one strict success `(target, footprint, receiver, quality) = (0, 0, 2, 2)`.

The method is promoted only if every mechanism and preservation condition
passes. A pass freezes v3b and its weight, after which the actual main
experiment must use a fresh held-out test set with multiple seeds and expanded
baselines. A failure is recorded as a negative ablation and is not hidden by a
weight sweep on eval12.
