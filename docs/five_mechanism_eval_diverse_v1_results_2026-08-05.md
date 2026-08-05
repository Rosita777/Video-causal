# Diverse-object smoke v1 results

## Purpose

This screen tests two non-ball source objects to reduce object bias in the planned mechanism set: a hammer or mallet fracturing glass, and an apple impacting flour. Each mechanism has two prompts and one generation per backbone.

## Strict semantic screen

| Mechanism | Wan strict valid | CogVideoX-2B strict valid | Borderline |
| --- | ---: | ---: | --- |
| Hammer or mallet fractures glass | 1/2 | 0/2 | None |
| Apple impacts flour | 0/2 | 0/2 | Wan 2/2 |

## Reading

Wan can generate non-ball causal chains. The wooden-mallet and glass-cup sample is fully usable: the receiver starts intact, the mallet contacts it, and cracking and fragments follow. Both apple samples show downward apple motion and a post-contact flour response, but the persistent indentation is unclear or the transition is visually unstable.

CogVideoX-2B fails all four strict screens. Its failures include blank videos, absent target objects, and footprints that already exist in the first frame. This is further evidence that CogVideoX-2B is not an adequate second backbone for the main multi-mechanism experiment.

## Decision

Replace the steel-ball fracture source with the wooden mallet while retaining glass fracture as the mechanism. Keep apple-to-flour as a promising diversity candidate, but run a second Wan prompt screen before using it for training data. The revised object set can therefore include a water droplet, one ball, a mallet, and an apple rather than three balls.
