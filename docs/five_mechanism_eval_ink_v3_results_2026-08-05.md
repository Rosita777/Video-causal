# Ink droplet v3 prompt screen

## Change

The prompt explicitly required a fixed side view, a completely clean paper surface, a round blue droplet visible in the air, downward motion, contact, and a stain spreading only after contact.

## Result

| Backbone | Strict causal-chain valid |
| --- | ---: |
| Wan | 0/2 |
| CogVideoX-2B | 0/2 |

Both models still mostly place a blue mark directly on the paper and grow or change it over time. They do not reliably show a separate falling droplet before contact. CogVideoX-2B is especially unstable: one sample produces only a late ring.

## Decision

Do not use ink droplet to stain as a main five-mechanism training/evaluation scene. The scene is useful as a negative feasibility result, but it does not provide a clean causal chain for the current backbones. Keep the replacement of toy-car trace open and choose a mechanism with a visibly separable source object and receiver interaction, such as a ball hitting a soft but rigidly supported target, after a small smoke screen.
