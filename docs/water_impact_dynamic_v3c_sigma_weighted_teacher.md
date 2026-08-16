# Water-impact dynamic v3c: sigma-weighted target-prompt teacher

Status: frozen before training or generation. V3c is a single-factor
development ablation. It is not the paper-final main experiment, and eval12 is
not eligible for v3c selection.

## Question and only treatment

V3b improved footprint suppression but usually changed a clearly visible
source object only to partial visibility. V3c tests whether allocating the same
mean target-prompt-teacher budget toward high-noise diffusion states improves
source-object removal.

V3c is trained from scratch. It keeps v3b's manifest, cached latents, target
prompt embeddings, model revision, LoRA initialization, seed, RNG construction,
balanced erase/preserve sample order, optimizer, 200-step budget, and preserve
branch fixed. The only training-objective change is the erase-row teacher
schedule. For sigma sampled exactly as in v3b:

```text
z_t       = (1 - sigma) * z_cf + sigma * epsilon
student   = LoRA(z_t, timestep, factual_prompt)
teacher   = frozen_base(z_t, timestep, target_generation_prompt)
L_teacher = MSE(student, stopgrad(teacher))
L_erase   = L_flow + 4 * (2 * sigma) * L_teacher
```

The registered base weight is `4.0`; the per-step effective weight is
`8*sigma`. Since sigma is sampled uniformly on `[0, 1)`, `E[2*sigma] = 1` and
the expected teacher budget remains 4. This is a schedule ablation, not a
teacher-weight sweep. The frozen teacher still has adapters disabled, consumes
the same noisy latent and timestep as the factual-prompt student, and is
stop-gradient. No factual latent, causal mask, token gate, CFG, pixel alignment,
or additional loss is introduced. Preserve rows remain exactly
`4 * MSE(LoRA, frozen_base)` under the same prompt.

The dedicated CLI objective is
`--objective target_prompt_teacher_sigma_weighted`. The canonical v3b trainer,
v3b launcher, and v3b protocol document are immutable because their byte hashes
are bound by the completed v3b registration; v3c therefore uses a dedicated
trainer copy.

## Frozen inputs and invariants

- protocol: `water_impact_dynamic_v3c_sigma_weighted_target_prompt_teacher_v1`
- model: `models/Wan2.1-T2V-1.3B-Diffusers`
- model revision: `0fad780a534b6463e45facd96134c9f345acfa5b`
- ordered transformer inventory SHA-256: `fe0c20ba99318c4b6d28d153b839888bbcf8a7a856796979566c305788dc18ac`
- train manifest SHA-256: `3d4d8cbf9244b1575357f0ac74380cd7cb6265df4d1a85bf450de4cac120aee4`
- 214-entry base-cache SHA-256: `4654ee67b8937d248963839b744e59f4f535f8be8b8eee421aef264fb3cd4d65`
- 178-entry teacher-cache SHA-256: `6cf7ba0112d8df0e0a5253a7a943411fcfd85c56fc6df580446776b49d1ac9a9`
- teacher-cache manifest SHA-256: `c467d7f81ee22b2c4b1ff719537487fbfc808eacc98e730c3d24f0a17aca77cb`
- target-prompt binding SHA-256: `9b1cc6e5bbdbe60b8f9f8378dc0ea11fea2fe82d8c73d6f9afed3f72b2bd00cc`
- unique target-embedding SHA-256: `a15f5e910358d5e95bcdd995303abb7eb7e7302fd9ee649c4cfebf3b8f6b6330`
- seed / steps / learning rate: `26000 / 200 / 5e-5`
- LoRA rank / alpha: `16 / 16`
- erase / preserve updates: `100 / 100`, strictly alternating
- expected initial LoRA SHA-256: `af163fcb6706c8403ffb1eaa9001cb2b9ac8ef86110663e8b20000961bb270a8`
- expected initial noise/sigma RNG SHA-256: `49b65850c0793680efb3a7cfc023601e240f13acb78ddb3aa483794c68136704`
- expected final noise/sigma RNG SHA-256: `79ff6c9a3db46b02896073cc95e8d05d185e813c844475e14b1ae460dd61b33f`
- expected final sample-order SHA-256: `a0c819a58fbf3f5be184f7b7708d30134d2f9dc0ebc98d79415f19d1cacc87cb`
- eval LoRA scale, if the gate permits generation: `1.25`

The registration also binds the completed v3b checkpoint weights, training
state, run registration, and scale sanity by their frozen SHA-256 values. This
makes the reference arm and the single-factor relationship explicit.

The revision string is not accepted as model identity by itself. The launcher
and trainer independently recompute an ordered byte inventory using
`SHA256(path_utf8 || NUL || file_bytes || LF)` for each lexicographically
ordered entry:

| Model-relative path | Bytes | File SHA-256 |
|---|---:|---|
| `transformer/config.json` | 465 | `0b093fa072e9ff28763febe9b964ee582f566733a6d6709deb9dfba1bde16b81` |
| `transformer/diffusion_pytorch_model-00001-of-00002.safetensors` | 4,998,781,576 | `6d011927dbd2cc8afe53d57abab04a8fd86f615d83324770d985fb058ece3a24` |
| `transformer/diffusion_pytorch_model-00002-of-00002.safetensors` | 677,289,072 | `b92ec2309b1f239af6f746431815a881afcc938abb26a4f08d9a2fd6c892f872` |
| `transformer/diffusion_pytorch_model.safetensors.index.json` | 73,296 | `dcbcf3497134a3f50557ff069dd7d2c84b5c4d8c5932472f6bdb780fb4016589` |

The safetensors filenames referenced by the index must exactly equal the
actual top-level safetensors inventory. Missing, extra, renamed, resized, or
byte-modified files fail before model loading. Both the ordered records and
aggregate digest are bound into `run_registration.json` and the checkpoint
training state.

The dedicated v3c trainer is also write-disabled for the shared base cache:
`--rebuild-cache` and `--cache-only` are rejected during argument parsing. The
v3c main path enumerates an exact, complete existing cache inventory and never
calls the inherited cache builder. A missing or unexpected entry fails before
training without creating or rewriting a cache file.

## Mandatory pre-checkpoint scale sanity

For each of the first 16 erase updates, record the sampled sigma and raw loss
ratio `r_i = L_teacher / L_flow`, then compute

```text
g_i = 8 * sigma_i * sqrt(r_i)
```

All losses, sigmas, ratios, and `g_i` values must be finite; `L_flow > 0`,
`L_teacher >= 0`, and `0 <= sigma <= 1`. The arithmetic mean of the 16 `g_i`
values must lie in the inclusive interval `[0.20, 0.50]`, and their maximum
must be at most `1.0`. A frozen-RNG arithmetic replay predicted mean `0.2868`
and maximum `0.8582`; these predictions are diagnostic only and do not change
the registered bounds.

No checkpoint may be written before this sanity artifact passes. In
particular, the nominal checkpoint-25 write is deferred because the 16th erase
observation occurs at global step 31. A failure writes the immutable sanity
diagnostic and terminates before that update, before any checkpoint, and before
generation. It does not authorize another schedule, weight, sigma window, seed,
or checkpoint choice.

Only `checkpoint-000200` is eligible for generation. At step 200, the trainer
also requires the sample-order digest to equal the frozen v3b digest before
writing the checkpoint.

## Paths and commands

```text
scripts/train_wan_waterdrop_lora_v3c.py
scripts/run_water_impact_dynamic_sft_v3c_teacher.sh
outputs/water_impact_dynamic_v3c/adapter_target_prompt_teacher_sigma2_scale4_v1
```

The launcher first verifies all frozen input inventories and v3b reference
artifacts. `preflight` is read-only with respect to the v3c output path:

```bash
bash scripts/run_water_impact_dynamic_sft_v3c_teacher.sh preflight
CUDA_VISIBLE_DEVICES=1 bash scripts/run_water_impact_dynamic_sft_v3c_teacher.sh train
```

The training command reserves its output with an atomic, exclusive `mkdir` and
refuses to reuse or race on an existing path. `run_registration.json` binds the
trainer, launcher, this document, frozen inputs, exact training configuration,
v3b reference artifacts, schedule formula, sanity bounds, expected initial
LoRA hash, and expected sample order.

## Evaluation boundary

Do not generate or inspect v3c on eval12. Freeze the fresh-dev24/sealed-final36
split and its v3b control generations before any v3c video. V3c is compared
only against frozen v3b on fresh-dev24 under the separately preregistered blind
review and promotion gate. Only a complete fresh-dev24 pass can unseal final36
for the multi-seed paper main experiment. A failure is recorded as a negative
ablation without a weight, schedule, sigma-window, or checkpoint sweep.
