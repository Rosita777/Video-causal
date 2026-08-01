#!/usr/bin/env python3
"""Freeze the balanced five-condition waterdrop test100 manifest."""

import csv
from collections import Counter
from pathlib import Path
from typing import Optional


ROOT = Path(__file__).resolve().parents[1]
FINAL_FIELDS = [
    "test_index",
    "scene_id",
    "receiver_group_id",
    "source_scene_id",
    "receiver_id",
    "family",
    "receiver",
    "condition",
    "causal_footprint",
    "expected_base_target",
    "expected_base_footprint",
    "expected_erased_target",
    "expected_erased_footprint",
    "erase_instruction",
    "fixed_seed",
    "prompt",
    "video_path",
    "contact_sheet",
    "selection_source",
    "replaces_scene_id",
    "semantic_status",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def keyed(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["scene_id"]: row for row in rows}


def final_row(
    row: dict[str, str],
    media: dict[str, str],
    source: str,
    *,
    condition: Optional[str] = None,
    receiver_group_id: Optional[str] = None,
) -> dict[str, str]:
    result = {field: row.get(field, "") for field in FINAL_FIELDS}
    result.update(
        {
            "test_index": "",
            "condition": condition or row["condition"],
            "receiver_group_id": receiver_group_id or row.get("receiver_group_id", ""),
            "video_path": media["video_path"],
            "contact_sheet": media["contact_sheet"],
            "selection_source": source,
            "replaces_scene_id": row.get("replaced_scene_id", ""),
            "semantic_status": "pass",
        }
    )
    return result


def write_replacement_reviews() -> None:
    rows = read_csv(ROOT / "data/waterdrop_control_replacements12.csv")
    selected = {"wdcontrolfix000", "wdcontrolfix004"}
    review = []
    for row in rows:
        scene_id = row["scene_id"]
        is_unrelated = row["condition"] == "unrelated_footprint"
        review.append(
            {
                "scene_id": scene_id,
                "replaced_scene_id": row["replaced_scene_id"],
                "condition": row["condition"],
                "receiver": row["receiver"],
                "semantic_pass": "yes" if is_unrelated else "no",
                "selected_for_final100": "yes" if scene_id in selected else "no",
                "semantic_note": (
                    "visible unrelated ripples, no target droplet, receiver preserved"
                    if is_unrelated
                    else "ripples remain visible, so this is not a clean control"
                ),
                "review_basis": "12-frame contact sheet",
            }
        )
    write_csv(ROOT / "data/waterdrop_control_replacements12_semantic_review.csv", review)

    dry_rows = read_csv(ROOT / "data/waterdrop_clean_dry_step4.csv")
    dry_review = []
    for row in dry_rows:
        dry_review.append(
            {
                "scene_id": row["scene_id"],
                "replaced_scene_id": row["replaced_scene_id"],
                "condition": row["condition"],
                "receiver": row["receiver"],
                "semantic_pass": "yes",
                "selected_for_final100": "yes" if row["scene_id"] == "wdcleanfix001" else "no",
                "semantic_note": "dry empty surface remains stable with no target or footprint",
                "review_basis": "12-frame contact sheet",
            }
        )
    write_csv(ROOT / "data/waterdrop_clean_dry_step4_semantic_review.csv", dry_review)


def main() -> None:
    write_replacement_reviews()
    base = keyed(read_csv(ROOT / "data/waterdrop_five_condition_test100.csv"))
    base_review = read_csv(ROOT / "data/waterdrop_five_condition_test100_semantic_review.csv")
    rows = []
    for reviewed in base_review:
        condition = reviewed["condition"]
        if condition == "target_only" or reviewed["semantic_status"] != "pass":
            continue
        rows.append(final_row(base[reviewed["scene_id"]], reviewed, "original_test100"))

    control_rows = keyed(read_csv(ROOT / "data/waterdrop_control_replacements12.csv"))
    control_media = keyed(read_csv(ROOT / "data/waterdrop_control_replacements12_auto_screen.csv"))
    for scene_id in ["wdcontrolfix000", "wdcontrolfix004"]:
        rows.append(final_row(control_rows[scene_id], control_media[scene_id], "control_replacement12"))

    dry_rows = keyed(read_csv(ROOT / "data/waterdrop_clean_dry_step4.csv"))
    dry_media = keyed(read_csv(ROOT / "data/waterdrop_clean_dry_step4_auto_screen.csv"))
    rows.append(final_row(dry_rows["wdcleanfix001"], dry_media["wdcleanfix001"], "dry_step_replacement4"))

    target_rows = read_csv(ROOT / "data/waterdrop_target_only_final20.csv")
    target_media = {
        **keyed(read_csv(ROOT / "data/waterdrop_target_only_resting20_auto_screen.csv")),
        **keyed(read_csv(ROOT / "data/waterdrop_target_only_replenish30_auto_screen.csv")),
    }
    for index, row in enumerate(target_rows):
        rows.append(
            final_row(
                row,
                target_media[row["scene_id"]],
                "target_only_final20",
                condition="target_only",
                receiver_group_id=f"wdtargetonly{index:02d}",
            )
        )

    order = {
        "explicit_causal": 0,
        "implicit_causal": 1,
        "target_only": 2,
        "unrelated_footprint": 3,
        "clean_control": 4,
    }
    rows.sort(key=lambda row: (order[row["condition"]], row["scene_id"]))
    for index, row in enumerate(rows):
        row["test_index"] = f"{index:03d}"

    counts = Counter(row["condition"] for row in rows)
    expected = {condition: 20 for condition in order}
    if len(rows) != 100 or counts != expected:
        raise ValueError(f"invalid final set: total={len(rows)} counts={dict(counts)}")
    if len({row["scene_id"] for row in rows}) != 100:
        raise ValueError("duplicate scene IDs in final set")
    if any(not row["video_path"] for row in rows):
        raise ValueError("missing video path in final set")

    write_csv(ROOT / "data/waterdrop_test100_final.csv", rows)
    print(f"Wrote final test100: {dict(counts)}")


if __name__ == "__main__":
    main()
