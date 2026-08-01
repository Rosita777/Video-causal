#!/usr/bin/env python3
"""Record manual pilot40 review without conflating generation and SFT usability."""

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPLICIT_ALIGNED = {0, 1, 2, 4, 5, 6, 8}
TARGET_ONLY_STRICT = {2, 6}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    rows = read_csv(ROOT / "data/waterdrop_train_pilot40.csv")
    review = []
    for row in rows:
        group_index = int(row["train_group_id"][-2:])
        condition = row["condition"]
        if condition == "explicit_causal":
            passed = group_index in EXPLICIT_ALIGNED
            note = (
                "causal event and footprint are visible; clean prefix supports aligned target"
                if passed
                else "causal event is visible but there are fewer than two reliable clean-prefix frames"
            )
            sft_usable = passed
        elif condition == "target_only":
            passed = group_index in TARGET_ONLY_STRICT
            note = (
                "one bead is present without a substantial footprint"
                if passed
                else "base generation adds delayed formation, impact, or footprint; prompt remains usable as erase conditioning"
            )
            sft_usable = group_index in EXPLICIT_ALIGNED
        elif condition == "unrelated_footprint":
            passed = True
            note = "dry unrelated ring is visible and the receiver is stable"
            sft_usable = group_index in EXPLICIT_ALIGNED
        else:
            passed = True
            note = "empty dry receiver remains stable"
            sft_usable = group_index in EXPLICIT_ALIGNED

        review.append(
            {
                "scene_id": row["scene_id"],
                "train_group_id": row["train_group_id"],
                "condition": condition,
                "receiver": row["receiver"],
                "generated_condition_pass": "yes" if passed else "no",
                "preliminary_sft_usable": "yes" if sft_usable else "no",
                "semantic_note": note,
                "review_basis": "12-frame contact sheet",
            }
        )

    output = ROOT / "data/waterdrop_train_pilot40_semantic_review.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(review[0]))
        writer.writeheader()
        writer.writerows(review)

    review_by_id = {row["scene_id"]: row for row in review}
    preliminary = read_csv(ROOT / "data/waterdrop_train_pilot40_sft_preliminary.csv")
    final_rows = []
    for row in preliminary:
        decision = review_by_id[row["scene_id"]]
        if decision["preliminary_sft_usable"] != "yes":
            continue
        final_rows.append(
            {
                **row,
                "semantic_status": "manual_sft_usable",
                "generated_condition_pass": decision["generated_condition_pass"],
            }
        )
    final_output = ROOT / "data/waterdrop_train_pilot40_sft_v0.csv"
    with final_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(final_rows[0]))
        writer.writeheader()
        writer.writerows(final_rows)
    print(
        f"wrote {len(review)} reviews; generated_pass="
        f"{sum(row['generated_condition_pass'] == 'yes' for row in review)}; "
        f"sft_usable={len(final_rows)}"
    )


if __name__ == "__main__":
    main()
