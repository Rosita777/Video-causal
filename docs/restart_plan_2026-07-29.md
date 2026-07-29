# Project Restart Plan

Updated: 2026-07-29

## Current Scope

The project studies a narrow failure in text-to-video concept erasure: after a target object is removed, a physical result caused by that object can remain visible.

The first restart milestone deliberately uses one simple causal chain:

```text
red rubber ball rolls -> ball hits four upright wooden blocks -> blocks topple
```

The erasure goal is to remove the red rubber ball and prevent the ball-caused toppling while preserving the wooden blocks and the rest of the scene.

## Decisions Made

- Use Wan2.1-T2V-1.3B as the first backbone.
- Keep CogVideoX-2B available as a later second backbone.
- Stop using ZeroScope as an active backbone because its base generation is too weak for reliable causal-chain output.
- Start with training-based erasure. The exact adapter target, prompt controls, and loss are still open and must not be presented as settled.
- Do not assume cross-object or cross-mechanism generalization before measuring it.
- Use one generated video per clean-source prompt for this initial screen.

## First Clean-Source Dataset

Prompt file:

```text
prompts/ball_blocks_clean_candidates50.txt
```

It contains 50 wording variants with fixed scene semantics:

- target: `red rubber ball`;
- receiver: four upright wooden blocks;
- motion: left to right;
- interaction: visible ball-block impact;
- result: all four blocks topple after impact;
- scene: fixed side camera, simple studio, flat gray floor.

Wan generation command is tracked in:

```text
scripts/run_ball_blocks_wan_clean.sh
```

Planned output directory:

```text
outputs/ball_blocks_clean_candidates50_wan21_t2v_1.3b_seed7000_step25_f49_480x832
```

The runner assigns one seed per prompt (`7000` through `7049`). This is one sample per prompt, not a multi-seed experiment.

## Clean-Source Review

A generated video passes only when the full video shows all of the following:

1. The red rubber ball is visible.
2. Four wooden blocks are upright before contact.
3. The ball visibly contacts the blocks.
4. The blocks topple only after contact.
5. No other visible event explains the toppling.
6. Video quality is sufficient to judge the sequence.

The clean-source review result, selected test set, manifests, metrics, and experiment summary must be committed to Git. Generated videos, model weights, caches, and local environments remain outside Git.

## Open Questions

These are intentionally unresolved:

- What prompt-only training data should the adapter use?
- What frozen-teacher target and preservation controls should define the training loss?
- Is one adapter specific to this object and causal chain, or can it generalize within a broader class?
- How should the erased output be judged beyond target removal and block preservation?

Answer these only after the base Wan model yields a reliable clean-source test set.

## Backup Policy

Commit and push after each meaningful milestone. Track:

- source code and launch scripts;
- prompts and structured dataset records;
- generation manifests and configs;
- human/VLM review tables;
- aggregate metrics and experiment summaries;
- small adapter metadata and hashes.

Do not commit:

- generated videos or frame images;
- base model weights or large adapter checkpoints;
- local Python/Conda environments and caches;
- API tokens or raw secrets.
