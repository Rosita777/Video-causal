# Water-impact dynamic v3c fresh-dev24 result (2026-08-16)

## Conclusion

V3c is a valid but negative development ablation. Moving the v3b target-prompt
teacher budget toward high-noise diffusion states preserves usability and
scene quality, but it provides only a marginal gain over v3b and does not
solve complete source-object deletion.

V3c therefore **does not pass the preregistered gate**, is not promoted, and
does not authorize opening sealed-final36. No teacher-weight, sigma-window,
schedule, seed, checkpoint, or prompt sweep is authorized by this result.

This is a 24-case, single-seed development comparison. It is not the paper
main experiment.

## Frozen treatment and training validity

The only treatment relative to v3b is the erase-row teacher schedule:

```text
L_erase = L_flow + 4 * (2 * sigma) * L_target_prompt_teacher
```

The mean teacher budget remains 4 because `E[2*sigma] = 1`. The training
manifest, latent and prompt caches, target-prompt teacher cache, Wan backbone,
LoRA initialization, sample order, optimizer, preservation branch, and
200-step budget remain frozen to v3b.

- Training protocol:
  `water_impact_dynamic_v3c_sigma_weighted_target_prompt_teacher_v1`
- Steps: 200, exactly 100 erase and 100 preserve updates
- Initial LoRA SHA-256:
  `af163fcb6706c8403ffb1eaa9001cb2b9ac8ef86110663e8b20000961bb270a8`
- Final LoRA SHA-256:
  `74923eaa7c4ed6b5eb1c19eb9c05952ad107f46dd276addf8b3b5cf173ea5ed4`
- Final training-state SHA-256:
  `7bc20f56d2f568da711937d30ca44b6d6e689cd9227c0442cc37c5275fd9c63b`
- Checkpoint artifact SHA-256:
  `ef51810e951ef46ce2a465d15b474069405f26f132924098acd4b6610078d54c`
- Run-registration SHA-256:
  `dbfe87724777f58ad4713aca960102bd0a711b0be018e8bf6e6c7fff095b936b`
- Scale-sanity SHA-256:
  `7c88f0a917e9bb8382b009435ddeee915eb8e9b6ee2098e08ed4fe145f584f85`
- Training-log SHA-256:
  `e1ecf4ae506a3739c0efc7ec8ceaefd661060fdf46ef9911c48451a99e2231b9`

The mandatory first-16-erase safety gate passed before any eligible
checkpoint was written: mean weighted output-gradient ratio
`0.2923522001244061`, maximum `0.8582215358392717`, within the registered
`[0.20, 0.50]` mean and `<=1.0` maximum bounds. The log contains exactly 200
finite steps and no OOM or traceback. The final adapter contains 480 finite
bf16 tensors and 11,796,480 LoRA parameters.

## Fresh split and generation

Eval12 was exhausted during earlier development and was not used for this
decision. Before v3c generation, the remaining 60 rows were deterministically
partitioned into fresh-dev24 and sealed-final36. Fresh-dev24 contains eight
cases per generalization group and four direct plus four natural prompts per
group.

- Stage-1 split registry SHA-256:
  `4f31a291e8ffca07da4bf057e9a86df72f656c03aab65bc06d4c3c155b72962a`
- Fresh-dev24 CSV SHA-256:
  `3b286c4561e4a0671fad7d652fc4356978229f2b6630e4dd6ca661d85d3b9197`
- Sealed-final36 CSV SHA-256:
  `983f684c7a798b84f02ce8817b9cf5a3f8ccf45f0697d5553f207661dc023a4b`
- Stage-2 registration SHA-256:
  `a7e8be6a259d6afb12fed3f3bdb1823dbe9fa1ef6faaf40f074f34326b407696`

Original, frozen v3b, and v3c were generated for the same 24 prompts and
seeds at 25 diffusion steps, CFG 5, 49 frames, 480x832, 8 fps, bf16; v3b and
v3c both used LoRA scale 1.25.

- Original generation-manifest SHA-256:
  `80fa7b999c77b05c27ee069aa7785336cec5d2697ea8c9861a37b633a651eecf`
- V3b generation-manifest SHA-256:
  `0cf91f0cf3eba8e1586fdf70af932354ed12964d3f51b3f4d6f4c42ad6c579d0`
- V3c generation-manifest SHA-256:
  `50c6698a4107f85e5a5210553114b1906d7bcb9ca27e7f6ba5b0b39f87263b56`

All 72 generated videos decoded successfully to exactly 49 frames. The three
arms have disjoint paths, inodes, and content hashes. No sealed-final36
generation manifest or canonical output directory exists.

## Blind review and adjudication

Two independent blinded reviewers each inspected all 48 anonymous candidate
videos in full (49 frames each) and all 24 seven-frame composites. The public
review package contained only the blank CSV, composites, and independent
anonymous video copies. The answer key and provenance manifest were isolated
in a private sibling directory.

The reviewers agreed on 166 of 192 atomic fields. A third blinded reviewer
adjudicated exactly the 26 disagreements across 20 candidate videos: 22 fields
had a two-of-three majority and four exact `0/1/2` splits used the registered
median value 1.

- Blank public review SHA-256:
  `1ad48c94b7c91c0295275ac75ae9877aa74101520558f16eb6f8eb667624eb7d`
- Answer-key SHA-256:
  `7cb45b48514c76ab74d14e60af65f0c6c49bd6ce8d7e1123ec2c7d36630c05b3`
- Private review-manifest SHA-256:
  `f4ae07798313f53f31b42223a2578d4f1972e4b039c077f841268e5d590b2118`
- Reviewer A SHA-256:
  `0a28f9bf556772d6ca6d1f13db51785d9679013e2348c29489ec8fedce1b2334`
- Reviewer B SHA-256:
  `e645d2afbb3180437cd642e3f4f06ef7730f42daf4c110ca6ef34a8f360698d1`
- Adjudicator SHA-256:
  `3cc7db0bfe86b9aa9a7ba23dc5d8b92538741ebdb5ccb65b7ae7d01303afad03`
- Canonical blind-review SHA-256:
  `f3e1d3c6d545f59608a045a915ddddd3fc8d375ac8f19712aefccb1366af73c9`
- Adjudication-audit SHA-256:
  `7c2671c1619d406b38f48e94cb20dc16be2fadf2deddf8c20b46b9720924bbff`

## Scores

Suppression points are `2 - visibility`; larger is better. Receiver and
quality points are direct sums; larger is better.

| Method | Usable | Receiver /48 | Quality /48 | Target suppression /48 | Footprint suppression /48 | Usable absent targets | Strict |
|---|---:|---:|---:|---:|---:|---:|---:|
| V3b | 24/24 | 47 | 32 | 11 | 17 | 4 | 0 |
| V3c | 24/24 | 47 | 32 | 12 | 18 | 3 | 0 |

V3c gains only one target-suppression point and one footprint-suppression
point. It has four paired target improvements: three clear-to-partial changes
and one partial-to-absent change. There is no clear-to-absent improvement.
The number of usable absent-target outputs decreases from four to three.

The score summary SHA-256 is
`4e3456dd782ccde34b8b095da6ff3a8d78b4d1aa5855a0a41f7f9f51d2155885`.
The unblinded-score SHA-256 is
`2d1b568e57f46f438b910e1810b73a6814c706ce9ec316f4adb0951860086a25`.

## Registered gate

Passed:

- at least 20 usable v3b controls (`24`);
- v3c usable at least 22/24 (`24`);
- receiver and quality preservation floors (`47` and `32`, both tied);
- non-worse footprint suppression on the v3b-usable set (`18 vs 17`).

Failed:

- target-suppression gain is `+1`, below the required `+6`;
- four paired target improvements, below the required six;
- zero clear-to-absent improvements, below the required two;
- zero clear-to-absent generalization groups, below the required two;
- usable absent-target count changes `4 -> 3`, below the required `+2`;
- zero strict successes, below the required two.

The all-or-nothing decision is:

```text
promote_v3c_and_unseal_final36 = false
```

Gate SHA-256:
`be9a38dc4ca7b0d6da274ec1c11e58c020ed5f9e6f89598420573d12e3cd9a70`.

An independent recomputation rebuilt all 192 atomic canonical scores, the 26
adjudications, method assignments, summary, and gate without discrepancy.

## Decision

Record v3c as evidence that high-noise redistribution of the teacher term is
not enough to convert v3b's partial source suppression into reliable complete
deletion. It preserves the receiver and quality, but the treatment effect is
too small and fails every registered complete-deletion condition.

Keep sealed-final36 unopened. Do not run the paper main experiment with v3c,
and do not adaptively sweep the teacher weight, sigma schedule/window, seed,
checkpoint, or prompt on fresh-dev24. Any subsequent method must start from a
new structural hypothesis and a newly preregistered development protocol that
does not consume sealed-final36.
