# Method v1: dual-trajectory causal-footprint LoRA

## Scope

The method is a training-based Wan LoRA adapter for removing a source object and the downstream video effects caused by that object. Version 1 uses one adapter per causal mechanism family. It is trained from aligned factual/counterfactual pairs and is not yet claimed to be a universal adapter across unrelated mechanisms.

## Training pair

For each accepted fixed-camera causal video, construct an aligned counterfactual target from the clean prefix. The counterfactual keeps the camera, receiver, lighting, and background while removing the source object and its downstream footprint.

The factual and counterfactual latents define a soft causal residual mask:

```text
M = smooth_normalize(abs(z_factual - z_counterfactual))
```

The mask follows the source object and its downstream footprint through space and time without requiring manual per-frame masks.

## Dual trajectories

The adapter sees two denoising trajectories for the same prompt:

1. A counterfactual trajectory, trained toward the aligned clean target.
2. A factual trajectory that already contains the object and footprint, redirected toward the clean target inside `M`.

Outside `M`, both trajectories are distilled toward the frozen base model to preserve unrelated content and the pre-event scene.

The implemented objective is:

```text
L = L_remove + lambda_bg * L_background
    + lambda_pair * L_pair + lambda_redirect * L_redirect
```

`L_pair` explicitly separates the adapter prediction from the factual causal target inside `M`; `L_redirect` teaches the adapter to redirect a partially formed factual chain toward the counterfactual endpoint.

## Default operating point

- backbone: Wan2.1-T2V-1.3B
- LoRA: rank 16, alpha 16, learning rate 1e-4
- background weight: 1.0
- pair weight: 1.0
- redirect training weight: 0.05
- inference LoRA scale: 0.75 by default; 1.0 is the maximum-erasure setting

## Evidence and limitation

On the frozen waterdrop held-out evaluation, dual-trajectory training reached 85.44% causal-activity suppression versus 74.90% for plain paired LoRA. Inference scale 0.75 reached 81.69% while reducing early-frame drift below both the plain and scale-1.0 settings.

This establishes the v1 method direction, not perfect erasure. Final experiments must still report object removal, footprint removal, unrelated-object preservation, clean-scene preservation, and general video quality.
