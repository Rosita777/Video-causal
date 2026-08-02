# Collision prompt gate30 (2026-08-02)

Wan generated all 30 fixed-seed collision prompts. Automatic motion screening reported 16 candidates, 4 short-prefix reviews, and 10 no-clean-prefix rejects. Manual causal-order review was stricter and accepted 10 videos.

Accepted distribution:

- block stacks / domino-like objects: 4
- cups, bottles, and tins: 4
- upright game pieces / pegs: 2

The successful videos show a visible ball reaching an object before the object tips or moves. Common failures were duplicated balls, hand intrusion, objects moving before contact, and scenes where the ball moved but the receiver stayed unchanged.

This gate establishes that ball collision is viable as a second mechanism, but 10 pairs are not enough for the mechanism adapter. The next data round should reuse the successful physical layouts and avoid the failed prompt families before training.
