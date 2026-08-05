# Five-mechanism evaluation smoke v0 results

## Setup

- Backbones: Wan2.1-T2V-1.3B and CogVideoX-2B.
- Ten prompts per backbone: two prompts for each of five mechanisms.
- One generation per prompt.
- Review basis: 12 evenly sampled frames from every full video.

## Semantic screen

| Mechanism | Wan strict valid | CogVideoX strict valid | Decision |
| --- | ---: | ---: | --- |
| Waterdrop impact | 2/2 | 1/2 | Keep |
| Red-ball collision | 1/2 plus 1 borderline | 0/2 | Simplify prompt |
| Steel-ball fracture | 2/2 | 0/2 | Keep for Wan; simplify for CogVideoX |
| Blue-ball particle impact | 0/2 plus 1 borderline | 0/2 | Make crater the primary visible footprint |
| Toy-car surface trace | 0/2 | 0/2 | Use higher-contrast surfaces and shorter prompts |

Wan supports three of the five provisional mechanisms in this first formulation. CogVideoX-2B supports only one of ten individual prompts. The main CogVideoX failures are missing causal actors, static target-only scenes, and nearly blank fracture/particle videos.

## Decision

Do not build training sets from v0 yet. Run a second shared-prompt smoke with substantially shorter prompts, higher-contrast particle/trace surfaces, and a clear persistent crater as the particle footprint. Preserve v0 as negative evidence for prompt and scene selection.
