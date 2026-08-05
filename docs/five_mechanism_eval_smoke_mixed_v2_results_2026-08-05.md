# Five-mechanism mixed-prompt smoke v2 results

## Purpose

This smoke test keeps the best prompt style observed for each existing mechanism and replaces the unreliable toy-car trace mechanism with ink droplet to persistent stain. One generation per prompt is used for screening, with two prompts per mechanism and backbone.

The intended second backbone was CogVideoX-5B. The A100 server could not reach Hugging Face, so the available CogVideoX-2B checkpoint was used as a temporary fallback. These results must not be reported as CogVideoX-5B results.

## Strict semantic screen

| Mechanism | Wan strict valid | CogVideoX-2B strict valid | Reading |
| --- | ---: | ---: | --- |
| Waterdrop impact | 2/2 | 2/2 | Stable on both backbones |
| Red-ball collision | 0/2 plus 1 borderline | 0/2 | Receiver toppling remains unreliable |
| Steel-ball fracture | 2/2 | 2/2 | Strongest non-water mechanism in this run |
| Blue-ball particle impact | 0/2 plus 1 borderline | 0/2 | Craters often pre-exist or scatter is missing |
| Ink droplet stain | 1/2 plus 1 borderline | 0/2 plus 1 borderline | Much more feasible than toy-car trace, but prompts need a visibly separated falling droplet |

## Decision

Keep ink droplet to stain as the fifth mechanism candidate and retire toy-car trace. The replacement is useful because clean paper can be shown before contact and a persistent footprint can remain afterward. Its next prompt revision should enforce a side-view or oblique view so the falling droplet is visually separated from the paper before impact.

Do not scale training data for all five mechanisms yet. Waterdrop and fracture are ready for a larger clean-source screen. Collision, particle impact, and ink stain each need a small prompt revision and another smoke screen. CogVideoX-2B remains only a development fallback; a stronger second backbone is still required for the main experiment.
