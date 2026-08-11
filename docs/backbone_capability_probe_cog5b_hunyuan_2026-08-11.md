# CogVideoX-5B and HunyuanVideo capability probe (2026-08-11)

## Purpose

This probe checks whether stronger backbones improve causal-chain generation before changing the frozen main protocol. It is a capability diagnostic, not a method or baseline result.

## Models and runtime

- CogVideoX-5B: `THUDM/CogVideoX-5b`, stored remotely under `models/CogVideoX-5b`.
- HunyuanVideo: Diffusers conversion `hunyuanvideo-community/HunyuanVideo`, stored remotely under `models/HunyuanVideo-Diffusers`.
- Hardware: A100 80GB GPUs on `A100_sc`.
- Both models use one fixed seed per prompt.

Hunyuan inference support is implemented in `scripts/generate_hunyuanvideo_clean.py`. Shared five-mechanism probe prompts are in:

- `prompts/backbone_capability_probe_v0.prompts`
- `prompts/backbone_capability_probe_compact_v0.prompts`

The compact prompts are 58-63 CLIP tokens. This avoids Hunyuan's 77-token CLIP truncation while retaining the clean opening, source event, contact, downstream footprint, and fixed-camera constraints.

## CogVideoX-5B short-prompt probe

Five 49-frame, 480x720, 30-step samples were generated for water impact, rigid collision, brittle fracture, powder impact, and surface trace.

Strict clean-source results were 0/5:

| Mechanism | Observation |
| --- | --- |
| Water impact | Source and ripples appear, but ripples pre-exist and the sampled sequence does not show a clean source-before-effect transition. |
| Rigid collision | Near-solid-color failure. |
| Brittle fracture | Near-solid-color failure. |
| Powder impact | The ball and crater are present from the beginning; no clean transition. |
| Surface trace | Tracks pre-exist before the car becomes visible. |

This setting should not be used to conclude that CogVideoX-5B lacks causal capability. The official model guidance emphasizes prompt optimization, and the short prompts did not specify a clean temporal opening.

## Long and compact water prompts

The Protocol v1 water prompt was also tested. CogVideoX-5B accepted the full prompt but generated an effectively reversed trajectory: splash/ripple activity appears first and the water becomes quieter later.

Hunyuan produced a visually clear droplet/contact/ripple sequence, but its CLIP branch truncated the 105-token Protocol v1 prompt at 77 tokens. The truncated portion included the explicit downstream footprint clause, so this is not a valid final Hunyuan setting.

With the shared 63-token compact prompt:

- CogVideoX-5B, 49 frames, 480x720, 50 steps: frame 0 is clean, followed by the source/contact and expanding ripples. The causal order is valid, but the event begins around frame 6, so the fixed 16-frame clean-prefix requirement is not met.
- HunyuanVideo, 81 frames, 480x720, 30 steps: the droplet descent, contact, splash column, and expanding ripples are visually strong and correctly ordered. However, the droplet is already visible at frame 0, so the clean-prefix requirement is not met.

## Engineering observations

- CogVideoX-5B is strongly prompt-sensitive. Compact, explicitly ordered wording works better than either short event-only wording or the longer Protocol v1 wording in this smoke.
- HunyuanVideo gives the strongest visual causal event in this water probe.
- Direct Hunyuan 81-frame inference used roughly 62GB peak GPU memory and about 14 seconds per denoising step on an otherwise free A100.
- A 129-frame, 544x960 direct-load smoke reached roughly 79GB and about 69 seconds for its first step. It was stopped because a single 30-step smoke would take roughly half an hour. Offload or distributed inference is required for practical native-resolution batches.

## Method implication

Both stronger backbones are worth continued capability testing, especially HunyuanVideo. However, the current training construction assumes that the first 16 frames are source-free and can be repeated as the counterfactual target. This assumption did not hold reliably in either new backbone probe.

Before full adapter training, the project must choose one defensible cross-backbone construction:

1. replace the fixed 16-frame rule with a verified per-video clean-prefix boundary;
2. generate a separate clean counterfactual trajectory instead of deriving it from the factual prefix; or
3. use an initial-state-conditioned generation setup that guarantees a clean starting state.

This decision must be made before freezing the next main protocol version.
