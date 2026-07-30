# Water-Drop Backbone Probe (2026-07-30)

## Goal

Check whether Wan2.1-T2V-1.3B and CogVideoX-2B can generate both sides of a simple water-drop counterfactual pair before expanding the dataset or training an adapter.

The five receivers were a pond, water in a glass, ceramic tile, absorbent paper, and a leaf. Each factual and counterfactual pair used the same seed.

## Configuration

- Wan2.1-T2V-1.3B-Diffusers: 25 steps, guidance 5.0, 49 frames, 480x832, bf16.
- CogVideoX-2B: 50 steps, guidance 6.0, 49 frames, 480x720, fp16.
- Seeds: 8100-8104 for the five paired scenes.
- Initial prompts: `prompts/waterdrop_capability_factual5.txt` and `prompts/waterdrop_capability_counterfactual5.txt`.
- Outputs: `outputs/waterdrop_backbone_probe_seed8100/` (not tracked by Git).

## Initial Manual Review

This is a qualitative capability probe, not a final benchmark score.

| Model | Receiver | Result |
| --- | --- | --- |
| Wan | pond | Failed temporal order: ripples were visible before contact. |
| Wan | glass | The event appeared, but the liquid level and scene changed too much. |
| Wan | tile | A wet region existed before the falling drop. |
| Wan | paper | A spreading mark appeared, but the falling drop/contact was unclear. |
| Wan | leaf | Water was already present and the requested leaf bending was absent. |
| Cog | pond | Temporal order was plausible, but the falling drop was barely visible. |
| Cog | glass | The event was plausible, but the glass itself was not clearly preserved. |
| Cog | tile | Best factual result: visible drop, impact, then splash. |
| Cog | paper | Unusable due to severe overexposure. |
| Cog | leaf | Unusable due to artifacts and missing leaf deformation. |

Wan produced usable static counterfactual states for most receivers, but factual and counterfactual compositions were often substantially different. Cog counterfactuals were less reliable: the pond moved despite the still-water prompt, the glass was lost, and paper/leaf outputs contained strong artifacts.

## Temporal-Prompt Follow-Up

The first Wan failures could have been either a model-capacity limit or a prompt-ordering problem. A second probe explicitly required an unchanged receiver during the first two seconds, followed by a visible falling drop, contact, and an effect only after contact.

- Prompts: `prompts/waterdrop_temporal_order_probe3.txt`.
- Outputs: `outputs/waterdrop_temporal_order_probe_wan_seed8200/` (not tracked by Git).

Results:

- Pond: temporal order improved from failed to usable, although the post-impact water column was not fully realistic.
- Tile: temporal order improved from failed to clearly usable.
- Paper: remained unusable because of overexposure and weak object visibility.

## Decisions

1. Use Wan as the primary first training backbone; retain Cog as a later comparison unless its paired prompts improve.
2. Require an explicit temporal template: stable initial state, visible cause, visible contact, and effect only after contact.
3. Start dataset expansion with pond/liquid-surface and tile/hard-surface families.
4. Do not expand paper or leaf prompts until a small probe demonstrates reliable source generation.
5. Screen both the factual causal video and its counterfactual target. A valid factual video alone is insufficient for prompt-pair training.
6. Track target visibility, contact visibility, temporal order, receiver preservation, and physical quality separately.
