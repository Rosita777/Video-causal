# Water-impact dynamic v3b: target-prompt teacher consistency

Status: amended and re-frozen after a training-only scale-invalid pilot, before
any v3b generation. Eval12 is a repeatedly used development set and is not the
paper-final main experiment.

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
L_erase = MSE(student, epsilon - z_cf) + 4 * MSE(student, stopgrad(teacher))
```

The teacher term has frozen weight `4.0`. It is added to, not substituted for
or averaged with, the flow loss. The teacher runs with adapters disabled and
shares the exact noisy latent and timestep with the student. Preserve rows keep
the existing `4.0 * MSE(LoRA, frozen_base)` same-prompt objective. No factual
latent, residual mask, token gate, CFG, or pixel alignment is introduced.

For each of the first 16 erase updates, define
`r_i = L_teacher / L_flow` and
`s_i = 4 * sqrt(r_i)`. Because both losses are mean MSE over the same student
output, `s_i` is the exact teacher/flow output-gradient norm ratio before the
shared model Jacobian. All values and losses must be finite, `L_flow > 0`, the
arithmetic mean of `s_i` must be in `[0.20, 0.50]`, and `max(s_i) <= 1.0`.
The gate is written before the 16th erase optimizer update. Failure invalidates
the run before generation; it does not trigger another weight choice.

## Training-only scale amendment

The initially registered `lambda=1` run started from the expected LoRA hash and
stopped at global step 31 when its 16th erase observation produced mean raw
teacher/flow loss ratio `0.005843`, below its registered `0.1` floor. No v3b
video was generated or inspected. This pilot is permanently scale-invalid and
will never be evaluated or resumed. Its frozen evidence is:

- log SHA-256: `c0f35542d9be763ea4a446af773e0e22fe44913b019b89aca51588780f5719ba`
- checkpoint-25 LoRA SHA-256: `2ee9f08c83d291630c09efcdf5bf0f8ae082f7b23b4c6be0ed89de791377ff3b`
- checkpoint-25 state SHA-256: `d51fe90cedc168125e773f4c44ad458cc2baf84f409df6ed29f20cc09bcae854`

The recalibration uses only these training losses. The first 15 logged erase
updates have mean `sqrt(r_i)=0.07530` and median `0.07650`; the error-reported
16-update mean is consistent with the same scale. The single frozen rule is
`lambda = nearest_power_of_two(0.30 / mean_i(sqrt(r_i)))`, which gives `4`.
This targets an auxiliary output-gradient ratio near `0.30`. Lambda 32 was
explicitly rejected because it would yield a teacher-dominant ratio near 2.4.
No generated output or eval12 label informed this amendment, and there will be
no second calibration or weight sweep.

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
- expected initial LoRA SHA-256: `af163fcb6706c8403ffb1eaa9001cb2b9ac8ef86110663e8b20000961bb270a8`

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
outputs/water_impact_dynamic_v3b/adapter_target_prompt_teacher_v1  # invalid lambda=1 pilot
outputs/water_impact_dynamic_v3b/adapter_target_prompt_teacher_scale4_v1
outputs/water_impact_dynamic_v3b/eval12_target_prompt_teacher_scale4_v1_ckpt200_scale1p25
```

Prepare the sidecar once, freeze its printed inventory hash in this document
and the launcher, then train:

```bash
CUDA_VISIBLE_DEVICES=1 bash scripts/run_water_impact_dynamic_sft_v3b_teacher.sh prepare
CUDA_VISIBLE_DEVICES=1 bash scripts/run_water_impact_dynamic_sft_v3b_teacher.sh train-scale4
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
