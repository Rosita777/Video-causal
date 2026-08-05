# Five-mechanism simple-prompt smoke v1 results

## Change from v0

- Both backbones receive the same substantially shorter prompts.
- Particle and trace cases use higher-contrast dark or red surfaces.
- The particle footprint explicitly includes a persistent crater.
- CogVideoX-2B uses 50 denoising steps instead of 25 to rule out an insufficient step budget.

## Strict semantic screen

| Mechanism | Wan strict valid | CogVideoX strict valid | Compared with v0 |
| --- | ---: | ---: | --- |
| Waterdrop impact | 2/2 | 2/2 | Stable and improved on CogVideoX |
| Red-ball collision | 0/2 | 0/2 | Short prompts hurt Wan; use the longer collision template |
| Steel-ball fracture | 1/2 | 0/2 | Wan remains feasible; CogVideoX remains weak |
| Blue-ball particle impact | 1/2 | 0/2 plus 1 borderline | Clear crater formulation improves Wan |
| Toy-car surface trace | 0/2 | 0/2 | Tracks still appear before the car moves |

## Decision

Prompt length is not the main CogVideoX-2B blocker. Even with short prompts and 50 steps, only waterdrop produces a consistently valid causal chain. CogVideoX-2B should not be used as the second backbone for a five-mechanism main experiment unless the scope is reduced to model-specific subsets.

For Wan, retain mechanism-specific prompt styles: simple waterdrop, long collision, long fracture, and simple high-contrast particle impact. Replace the toy-car trace mechanism with a more generatable persistent-footprint scene before constructing training data.
