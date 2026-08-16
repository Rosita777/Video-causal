# Water-impact dynamic v3b eval12 result (2026-08-16)

## Conclusion

The target-prompt teacher treatment is a useful but negative development
ablation. It substantially improves source-object and causal-footprint
suppression while preserving the receiver, but it fails the two registered
complete-deletion conditions. It is therefore **not promoted** as the new
operating point, and its weight must not be tuned on eval12.

Eval12 has been used repeatedly for development. These numbers are not a
paper-final main experiment.

## Frozen treatment

- Evaluation protocol: `water_impact_dynamic_v3b_eval12_v1`
- Objective: factual-prompt LoRA student plus source-free frozen-base teacher
- Teacher weight: `4.0`
- Schedule: 200 steps, exactly 100 erase and 100 preserve
- Initial LoRA SHA-256:
  `af163fcb6706c8403ffb1eaa9001cb2b9ac8ef86110663e8b20000961bb270a8`
- Final LoRA SHA-256:
  `d3fecf26b7f1ca6c4a8f46c86850a47a7ec5a62762d0e0aa15c49363040875d3`
- Final training-state SHA-256:
  `0f9aa26e825f4f6f497b1312c507b685c054bc319f2f9f538e45eeb7a7908bea`
- Scale sanity: 16/16 observations, mean output-gradient ratio `0.2815118`,
  maximum `0.4673677`, registered gate passed
- Generation: checkpoint 200, LoRA scale 1.25, the frozen 12 prompts/seeds,
  25 steps, 49 frames, 480x832, 8 fps
- Treatment generation-manifest SHA-256:
  `f9d6ad4ee85a9ddbc2be0d20a5423d6338ee9bef166ad5e5d67bc5ea5954823c`

All 12 treatment videos decoded to exactly 49 non-empty 832x480 frames. The
training log contains 200 finite steps and no traceback or OOM.

## Blind review

Two reviewers independently inspected all 24 anonymous candidate videos in
full (49 frames each) plus the 12 seven-frame composites. They agreed on 82 of
96 atomic scores. A third blinded reviewer adjudicated the remaining 14 fields
using the rule frozen before either score set was returned.

The reviewer-facing directory contained only the blind CSV, 12 composites,
and 24 independently copied anonymous MP4 files. The answer key and provenance
manifest were stored in a separate private directory. Reviewers confirmed that
they did not access the private package or source run directories before the
canonical blind CSV was frozen.

- Canonical blind-review SHA-256:
  `13e84286417b807801a023ca0da72780b2b57ba369980adf3d67d077641fd842`
- Answer-key SHA-256:
  `17310d0f6538ed50f763750a4b6c7114453b8412332f72fe3542c1c047421bde`
- Private review-manifest SHA-256:
  `11ffff9e784949c424bfa8e4ee790014afdf2b8365a16066cebfad0898b19bc2`

The three raw reviewer files are archived as supporting provenance. They are
not inputs bound by the scoring gate, which starts from the frozen canonical
review:

- Reviewer A SHA-256:
  `df2a107e26bfe5c40f02cb379ce4148e64f403f34582733b276a2f17b88ce809`
- Reviewer B SHA-256:
  `5698f33ea7d9f3db15b33a205216f561f2f6e9ecff12655b75ed73ad23ec7c81`
- Adjudicator SHA-256:
  `8c22fe6353f3228d33be8d1355eff160bcd22d3e86f4c5253ee88fc604d9c7d5`

## Scores

Suppression points are `2 - visibility`; larger is better. Receiver and
quality points are direct sums; larger is better.

| Method | Usable | Receiver /24 | Quality /24 | Target suppression /24 | Footprint suppression /24 | Strict success |
|---|---:|---:|---:|---:|---:|---:|
| Seeded balanced control | 11/12 | 21 | 17 | 3 | 3 | 0 |
| V3b target-prompt teacher | 12/12 | 22 | 18 | 7 | 12 | 0 |

On the 11 control-usable samples used by the paired gate:

- target-suppression points improve from `2` to `6`;
- footprint-suppression points improve from `1` to `10`;
- three control-target-visibility-2 samples improve to visibility 1;
- those improvements cover unseen-receiver and both-unseen groups.

V3b also produces one target-visibility-0 output, sample 5, but its control is
already target visibility 1. It therefore does not satisfy the registered
requirement that at least one control-visibility-2 improvement reach 0. That
sample also has receiver and quality scores of 1, so it is not a strict
success.

## Registered gate

Passed:

- at least three usable target improvements;
- improvements in at least two generalization groups;
- target suppression at least control plus 3 on the control-usable set;
- at least 11 usable treatment videos;
- all absolute and control-relative receiver and quality floors;
- non-worse footprint suppression on the control-usable set.

On the paired control-usable set, receiver and quality are each tied
(`21 vs 21` and `17 vs 17`). Thus the preservation floors themselves pass;
the aggregate `preservation_positive` flag below is false because it also
requires a strict success.

Failed:

- no registered improvement reaches target visibility 0;
- strict success count is 0.

The resulting gate is:

```text
mechanism_positive = false
preservation_positive = false
promote_v3b_operating_point = false
```

Gate SHA-256:
`37b51b4a75f7734f04f8dde05f2e61cdcf2f9dfb78a0d5de092f80eb2b031336`.

Unblinded-score SHA-256:
`0280a39d28408199bb16a3e2032af719c878a4da0d476d68c60f82369c322961`.

Summary SHA-256:
`d2c798b36d69313ebb839637fdaad173931cf936f6a4d3060981c2ec2986cb82`.

## Decision

Record v3b as evidence that target-prompt teacher consistency moves the model
in the correct direction without sacrificing preservation. Do not run a
teacher-weight sweep or choose a different checkpoint on eval12. A subsequent
method change must be structural, must be registered as a single-factor
experiment, and should use a fresh development holdout before any paper-final
main test.

These are descriptive results from a 12-case, single-seed development gate;
they are neither a significance claim nor a paper-final result.
