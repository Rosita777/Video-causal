# Waterdrop generalization split v1

Date: 2026-08-02

## Purpose

This is a candidate receiver-held-out split for retraining the waterdrop
adapter. It is built from the 55 reviewed factual/counterfactual aligned pairs;
it is metadata only and does not regenerate videos.

## Split rule

- `train_candidate`: aligned pairs whose receiver does not overlap the frozen
  eval20 receiver families and is not part of the internal holdout.
- `internal_receiver_holdout`: five receiver families held out from training:
  glass tabletop, metal bucket, brown cardboard, kitchen sponge, and chalk dust.
- `reserved_external_eval_overlap`: receiver families already represented in
  the frozen eval20: pond, cup, tray, and cutting board.

The receiver matching is semantic, so obvious variants such as `ceramic cup`
and `porcelain teacup` are treated as the same family.

## Current counts

| Split | Pairs |
| --- | ---: |
| Train candidate | 26 |
| Internal receiver holdout | 16 |
| Reserved external eval overlap | 13 |
| Total aligned pairs | 55 |

The intended first training set has at least 30 causal pairs. Therefore **four
new accepted causal pairs are still needed** before retraining. The current
training candidate already covers four footprint families:

- splash + ripple: 10
- splash + wet mark: 7
- spreading wet patch: 7
- crater or damp particle mark: 2

## Important limitation

The labels are derived from receiver metadata and prompt descriptions; they are
not new human annotations. New candidates must still pass the same clean-prefix,
visible-contact, post-contact-footprint review before entering training.

The next data action is to generate more prompt-bank candidates on receivers not
in the external eval20, screen them, and accept at least four aligned pairs. The
split must then be rebuilt before the next adapter training run.
