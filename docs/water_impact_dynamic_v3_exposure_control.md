# Water-impact dynamic v3 exposure control

## Purpose

The historical v2 run alternated erase and preservation rows for 200 steps. It therefore
used 100 erase updates and 100 preservation updates even though the manifest
contains 178 erase rows and 36 preservation rows. Under the fixed seed, only
100 of the 178 erase rows were visited.

The old trainer did not seed LoRA initialization, so the historical checkpoint
cannot serve as a strict one-factor control. Before adding a new loss, this
experiment reruns two arms with explicitly seeded, fingerprinted initialization:

- `balanced`: the historical 100 erase / 100 preservation schedule;
- `exposure`: 168 erase / 32 preservation updates from the manifest order.

Their only intended difference is sampling. The comparison tests whether weak
source-object deletion is partly an erase-data exposure problem.

The comparison builder fails closed unless both checkpoint states confirm the
frozen model and optimizer settings, matching seeded initialization hashes, the
expected `100 / 100` and `168 / 32` role counts, and the exact seeded sample
orders. It also recomputes each checkpoint artifact hash and verifies all
generation settings against the frozen eval12 protocol.

## Frozen comparison

The following stay identical to v2:

- Wan 2.1 T2V 1.3B Diffusers backbone;
- 178 accepted erase rows and 36 generic preservation rows;
- cached latents and prompt embeddings;
- LoRA rank / alpha `16 / 16`;
- learning rate `5e-5`;
- 200 optimizer steps and seed `26000`;
- plain flow-matching objective and preservation weight `4.0`;
- eval12 prompts, seeds, generation settings, and LoRA scale `1.25`.

The frozen training-manifest SHA-256 is
`3d4d8cbf9244b1575357f0ac74380cd7cb6265df4d1a85bf450de4cac120aee4`.
The launcher refuses to train if it differs.

The ordered, byte-level SHA-256 of the 214 frozen cache payloads is
`4654ee67b8937d248963839b744e59f4f535f8be8b8eee421aef264fb3cd4d65`.
Both launch and comparison preflights reject any content change.

The exposure arm removes `--balanced-roles`. The seeded first 200 rows contain
168 erase and 32 preservation updates, close to the manifest's empirical ratio,
and leave 10 erase plus 4 preservation rows unseen. Both training logs and
checkpoints record an initial-LoRA SHA-256; the two hashes must match.

## Commands

Prepare and validate the shared cache once:

```bash
bash scripts/run_water_impact_dynamic_sft_v3_ablation.sh prepare
```

Train the two arms, optionally in parallel on separate free GPUs:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_water_impact_dynamic_sft_v3_ablation.sh balanced
CUDA_VISIBLE_DEVICES=1 bash scripts/run_water_impact_dynamic_sft_v3_ablation.sh exposure
```

Generate both matched eval12 arms, then build the blinded comparison:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_water_impact_dynamic_v3_eval12.sh balanced
CUDA_VISIBLE_DEVICES=1 bash scripts/run_water_impact_dynamic_v3_eval12.sh exposure
bash scripts/run_water_impact_dynamic_v3_eval12.sh compare
```

Both launchers accept `WAN_PYTHON`, `WAN_MODEL`, and `WAN_DEVICE` overrides.

## Pre-registered decision rule

Score the same four atomic fields as v2 from seven-frame sheets, without using
training loss to select the result. Hide the method identity during review and
use a fixed random order for candidate sheets.

Call the ablation **source-positive** when all of the following hold:

- at least three samples that score target visibility `2` under the seeded
  balanced control improve to `1` or `0` under exposure while the exposure
  output remains usable (`receiver >= 1` and `quality >= 1`);
- at least one of those samples reaches target visibility `0`;
- target improvement appears in at least two generalization groups.

Promote it as the new operating point only if it is source-positive and:

- at least 11 of 12 outputs remain usable;
- exposure receiver-preservation total is at least `19 / 24` and no more than
  one point below the seeded balanced control;
- exposure video-quality total is at least `22 / 24` and no more than one
  point below the seeded balanced control;
- on the samples usable under the seeded balanced control, exposure's
  footprint-suppression points are at least `7` and no more than one point
  below the control. An unusable exposure output contributes zero points;
- at least one output is a strict success: target and footprint absent with
  receiver preservation and quality both scored `2`.

After annotation, unblind and apply the frozen formulas with:

```bash
models/.wan-runtime/bin/python scripts/score_water_impact_dynamic_v3_sampling.py \
  --review experiments/water_impact_dynamic_eval12/v3_sampling_blind_review/blind_review.csv \
  --answer-key experiments/water_impact_dynamic_eval12/v3_sampling_blind_review/answer_key.csv \
  --output-dir experiments/water_impact_dynamic_eval12/v3_sampling_scores
```

The scorer binds the answer key to the generated review manifest, requires all
12 sample indices and all 24 candidate rows, and checks gate-critical fields
such as the generalization group against the hidden key before unblinding.

If v3a fails this gate, do not combine it with other changes. The next isolated
ablation should target the erase loss directly, while retaining the seeded
balanced arm as the comparison baseline.
