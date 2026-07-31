# Waterdrop five-condition semantic review

Date: 2026-07-31

All 100 twelve-frame contact sheets were reviewed by condition. The temporal auto-screen labels were not used as semantic acceptance labels.

## Result

| Condition | Pass | Fail | Conclusion |
|---|---:|---:|---|
| Explicit causal | 20 | 0 | Stable |
| Implicit causal | 20 | 0 | Stable |
| Target only | 1 | 19 | Not usable with the current prompt design |
| Unrelated footprint | 18 | 2 | Mostly stable |
| Clean control | 19 | 1 | Stable |
| Total | 78 | 22 | Pilot only; not a final frozen test set |

## Main failure

The target-only prompt asks a water droplet to enter and remain suspended above the receiver without contact. Wan usually ignores the non-contact constraint and generates an impact plus a ripple or wet footprint. One sample (`wdfive0067`) satisfies the requested state; the other 19 do not provide a valid target-only control.

This is a base-model capability failure, not an erasure-method result. Keeping these invalid controls would make later evaluation uninterpretable.

## Other failures

- `wdfive0008`: no clear unrelated footprint is visible.
- `wdfive0018`: an unprompted object replaces/corrupts the intended receiver.
- `wdfive0049`: ripples appear in the clean control.

## Decision

Do not finalize this five-condition test100 as the paper evaluation set. Remove the unstable target-only condition rather than forcing a difficult state that the base model cannot reliably generate.

The next evaluation revision should use four conditions:

1. explicit causal;
2. implicit causal;
3. unrelated footprint preservation;
4. clean-scene preservation.

After excluding the three receiver groups with failures outside target-only, 17 complete receiver groups remain. Replace those three groups with three newly screened held-out receivers to obtain a balanced 20-receiver, four-condition test80.
