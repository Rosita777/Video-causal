# Seven-mechanism baseline preparation and launch order

This stack prepares the frozen CogVideoX block for
`causal_role_erasure_7m_single_seed_v2`. It does not authorize replacing any
formal seed, case, mechanism, or named method.

The five CogVideoX streams are CogVideoX Original, Negative Prompt,
VideoEraser (official CogVideoX), T2VUnlearning-adapted (ours), and
SAFREE-CogVideoX. Together they contain `294 x 5 = 1,470` videos. The three Wan
streams are generated separately.

## 1. Build the incomplete asset/model snapshot

The old persistent project still contains the exact official VideoEraser
pipeline at commit `ba19cceb561dda916614e609759eb5c5b54f1c83`. The builder
copies it into the immutable registry directory and verifies pipeline SHA-256
`bd9e4052740eba1b37fbd31b879c4079f7139d2d23ff3a5811cab09b8321c0f4`.

```bash
cd /data/xiaohuang_workspace/ljc/Video-causal-v4
models/.wan-runtime/bin/python \
  scripts/build_causal_role_erasure_7mechanism_baseline_registry_v2.py \
  --project-root "$PWD" \
  --output-root outputs/causal_role_erasure_7mechanism_main_v2/baseline_registry_v2_pre_t2v
```

This first registry is expected to say `baseline_stack_incomplete`; it still
freezes the CogVideoX model inventory, runtime, formal-case SHA, and copied
VideoEraser source needed by the T2V preparation stage.

## 2. Freeze and train the seven adapted T2V operators

```bash
models/.wan-runtime/bin/python \
  scripts/build_causal_role_erasure_7mechanism_t2v_training_registry_v2.py \
  --project-root "$PWD" \
  --model-inventory \
    outputs/causal_role_erasure_7mechanism_main_v2/baseline_registry_v2_pre_t2v/model_inventory.json
```

The default registration selects 36 prompts per mechanism by a treatment-blind
SHA rule and precommits step 100 as the only eligible checkpoint. This retains
the historical non-collapsed adapted setting while eliminating the old
trainer's hard-coded input count and any post-generation checkpoint choice.

Validate all seven specs without loading a model:

```bash
for mechanism in water_impact rigid_collision brittle_fracture powder_impact \
  elastic_deformation material_release surface_trace; do
  models/.wan-runtime/bin/python \
    scripts/train_causal_role_erasure_7mechanism_t2v_adapted_v2.py \
    --project-root "$PWD" \
    --training-registry \
      outputs/causal_role_erasure_7mechanism_main_v2/t2v_training_specs_v2/t2v_training_registry.json \
    --mechanism "$mechanism" \
    --dry-run
done
```

Only after the dry run is clean, launch the registered training queue:

```bash
bash scripts/run_causal_role_erasure_7mechanism_t2v_training_queue_v2.sh
```

## 3. Freeze the restored SAFREE checkout

The official checkout at `baselines/external/SAFREE` is frozen at commit
`b8b2c3fa9d7f51c46f5a570170503fc98bd9c7ec`.  The SHA-256 of
`cogvideox/cogvideox_pipeline.py` is
`55185f5972ec0942e9ba653176c14c9ecd5ac0e19ef308d92e31439a10ce2609`.
The final builder must receive both exact values; no floating branch name is
accepted.

## 4. Build the complete formal-generation registry

Use a fresh final output directory after all seven T2V checkpoints exist:

```bash
models/.wan-runtime/bin/python \
  scripts/build_causal_role_erasure_7mechanism_baseline_registry_v2.py \
  --project-root "$PWD" \
  --safree-expected-commit b8b2c3fa9d7f51c46f5a570170503fc98bd9c7ec \
  --safree-expected-pipeline-sha256 \
    55185f5972ec0942e9ba653176c14c9ecd5ac0e19ef308d92e31439a10ce2609 \
  --t2v-registry \
    outputs/causal_role_erasure_7mechanism_main_v2/t2v_training_specs_v2/t2v_training_registry.json \
  --output-root \
    outputs/causal_role_erasure_7mechanism_main_v2/baseline_registry_v2_final \
  --require-complete
```

Formal generation must not begin unless this registry says
`formal_generation_authorized: true`.

## 5. CogVideoX Original and Negative Prompt

The runner reads every arbitrary frozen seed directly from `formal_cases.csv`.
One process per mechanism is the intended GPU shard. A dry run writes only the
deterministic plan:

```bash
models/.wan-runtime/bin/python \
  scripts/run_causal_role_erasure_7mechanism_cogvideox_controls_v2.py \
  --project-root "$PWD" \
  --baseline-registry \
    outputs/causal_role_erasure_7mechanism_main_v2/baseline_registry_v2_final/baseline_registry.json \
  --baseline cogvideox_original \
  --mechanism water_impact \
  --output-dir \
    outputs/causal_role_erasure_7mechanism_main_v2/formal_cogvideox/cogvideox_original/water_impact \
  --dry-run
```

Use `--baseline negative_prompt` for the paired Negative Prompt stream. After
the dry run freezes its output plan, replace `--dry-run` with `--resume` for
the formal launch. An interrupted run may also use `--resume`;
every existing video is re-decoded and hash-checked before it is skipped. Each
accepted file must contain exactly 49 frames at 8 FPS and 480x720, and receives
an individual immutable receipt.

## 6. Run all five CogVideoX streams

The unified queue dispatches all `5 x 7 = 35` stream/mechanism jobs and checks
all 1,470 videos through the method-specific formal runners.  Use one SAFREE
job at a time for the first formal run because its official pipeline is fp32:

```bash
models/.wan-runtime/bin/python \
  scripts/run_causal_role_erasure_7mechanism_baseline_queue_v2.py \
  --project-root "$PWD" \
  --baseline-registry \
    outputs/causal_role_erasure_7mechanism_main_v2/baseline_registry_v2_final/baseline_registry.json \
  --output-root \
    outputs/causal_role_erasure_7mechanism_main_v2/formal_baselines_v2 \
  --gpus 0,1,2,3 \
  --safree-max-concurrency 1 \
  --dry-run
```

Inspect the deterministic plan, then replace `--dry-run` with `--resume` to
execute that exact plan.  `--resume` is infrastructure-only: it revalidates
the complete child plan, manifest, receipts, media, and hashes; partial or
failed jobs are rejected in place.
