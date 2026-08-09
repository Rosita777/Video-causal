# Baseline Reproduction Research

## Short conclusion

The current repository implementations named `run_t2vunlearning_wan.py` and `run_videoeraser_wan.py` are Wan prompt/embedding proxies. They are useful smoke baselines, but they are not the official implementations.

For the final paper table:

- Negative Prompt can be reported directly as a prompt baseline.
- VideoEraser should be reproduced on CogVideoX using the authors' official pipeline, because the official method is training-free and explicitly supports CogVideoX.
- T2VUnlearning cannot be reproduced exactly for our four mechanisms from the public repository: the official repository provides CogVideoX/Hunyuan inference and released nudity adapters, but marks training code as TODO. A new implementation based on the paper would be an adaptation, not an official checkpoint reproduction.

## T2VUnlearning

Official paper: [arXiv:2505.17550](https://arxiv.org/abs/2505.17550). Official code: [VDIGPKU/T2VUnlearning](https://github.com/VDIGPKU/T2VUnlearning).

The method is an adapter-based fine-tuning procedure with three parts:

1. Negatively-guided velocity prediction: train the adapter toward a target velocity that moves away from the target-concept prediction.
2. QK/text-to-video attention localization: apply the adapter only inside the target concept mask.
3. Preservation regularization: match the frozen model on a non-target preservation concept.

The paper reports CogVideoX settings of rank 128, learning rate `1e-4`, 500 epochs, bf16, localization weight `alpha=1.0`, and preservation weight `beta=0.0`. The public README explicitly says training code is not released and provides only inference code plus released nudity erasure adapters. The released adapter is for nudity, so it cannot be used as a causal-mechanism baseline for water impact, collision, fracture, or powder impact.

An implementation from the paper would require a separate CogVideoX trainer, prompt augmentation, QK-mask extraction, and the Receler-style adapter. We should label that result `T2VUnlearning-adapted (ours)` unless the authors release matching checkpoints.

## VideoEraser

Official paper: [arXiv:2508.15314](https://arxiv.org/abs/2508.15314). Official code: [bluedream02/VideoEraser](https://github.com/bluedream02/VideoEraser).

VideoEraser is training-free and has an official CogVideoX implementation. It has two stages:

1. SPEA adjusts only prompt tokens identified as close to the erased concept embedding.
2. ARNG modifies the denoising noise guidance using the input prompt and target-concept predictions, with temporal-step and frame averaging.

The paper's default parameters are `alpha=0.01`, `w0=1000`, `sm=0.5`, `v0=0`, `beta=0.5`, and `theta=1`. The CogVideoX experiment uses 50 denoising steps, guidance scale 6, 720x480, and 50 frames. The repository's CogVideoX script uses 49 frames and the same DPM scheduler family.

For our protocol, the clean reproduction is to run the official CogVideoX pipeline with each mechanism phrase as `unsafe_concept`, using the same 80 prompts and fixed seeds. A one-prompt smoke test succeeded with the official pipeline unchanged under CogVideoX-2B, torch 2.6.0, and diffusers 0.33.1, so the formal run keeps the official implementation unmodified.

## Fair comparison protocol

The baseline prompt should be the mechanism concept, not a single training object. Otherwise the baseline receives a different task from our mechanism adapter. Use the same 80 prompts, seeds, frame count, resolution, and base model within each model family. Report separately:

- official CogVideoX VideoEraser;
- Negative Prompt;
- our Wan adapter;
- T2VUnlearning adapted to CogVideoX, only if we implement and clearly mark the adaptation.

Do not put the current Wan T2VUnlearning/VideoEraser proxies in the main table. They can remain as engineering smoke tests.
