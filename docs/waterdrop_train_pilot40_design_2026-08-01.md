# Waterdrop training pilot40 design

This pilot uses 10 hard-surface receivers that do not occur in the frozen test100. Each receiver has four generated conditions: explicit causal, target only, unrelated footprint, and clean control.

The intended training mapping is:

| Input prompt condition | Desired SFT target video |
|---|---|
| Explicit causal | Same receiver's clean-control video |
| Target only | Same receiver's clean-control video |
| Unrelated footprint | Its own unrelated-footprint video |
| Clean control | Its own clean-control video |

This creates 40 training records from 20 unique desired target videos. All 40 condition videos are generated and screened so that base-model capability and later contrastive experiments can also be measured.

This is an engineering pilot, not the final paper-scale training set. It tests whether one waterdrop adapter can learn removal while preserving unrelated marks and clean surfaces. If the training loop works, the same schema can be expanded to more receivers and more footprint types.
