#!/usr/bin/env python3
"""Record manual replenishment review and freeze the final target-only set."""

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data/waterdrop_target_only_replenish30.csv"
REVIEW_OUTPUT = ROOT / "data/waterdrop_target_only_replenish30_semantic_review.csv"
FINAL_OUTPUT = ROOT / "data/waterdrop_target_only_final20.csv"
FINAL_FIELDS = [
    "scene_id",
    "source_scene_id",
    "receiver_id",
    "family",
    "receiver",
    "condition",
    "split",
    "expected_base_target",
    "expected_base_footprint",
    "expected_erased_target",
    "expected_erased_footprint",
    "erase_instruction",
    "fixed_seed",
    "prompt",
    "selection_source",
]

PASS_IDS = {
    "wdreplenish002",
    "wdreplenish003",
    "wdreplenish004",
    "wdreplenish005",
    "wdreplenish011",
    "wdreplenish012",
    "wdreplenish013",
    "wdreplenish014",
    "wdreplenish023",
    "wdreplenish024",
    "wdreplenish025",
}

SELECTED_IDS = PASS_IDS - {"wdreplenish005"}

FAIL_REASONS = {
    "wdreplenish000": "bead grows substantially across the clip",
    "wdreplenish001": "bead is initially tiny and grows substantially",
    "wdreplenish006": "dark puddle and ring form instead of one clear bead",
    "wdreplenish007": "bead forms and grows after recording begins",
    "wdreplenish008": "new falling water enters near the end",
    "wdreplenish009": "bead grows substantially near the end",
    "wdreplenish010": "late impact-like deformation",
    "wdreplenish015": "bead changes shape substantially",
    "wdreplenish016": "surrounding ring and late falling water",
    "wdreplenish017": "persistent surrounding wet ring",
    "wdreplenish018": "surrounding ring and late falling water",
    "wdreplenish019": "surrounding ring and late falling water",
    "wdreplenish020": "new falling water enters near the end",
    "wdreplenish021": "new falling water enters near the end",
    "wdreplenish022": "persistent surrounding wet ring",
    "wdreplenish026": "surrounding ring and late impact",
    "wdreplenish027": "bead grows and receives a late impact",
    "wdreplenish028": "large shape changes and late incoming water",
    "wdreplenish029": "large shape changes across the clip",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def final_row(row: dict[str, str], source: str) -> dict[str, str]:
    return {
        field: source if field == "selection_source" else row.get(field, "")
        for field in FINAL_FIELDS
    }


def main() -> None:
    rows = read_csv(INPUT)
    reviewed = []
    for row in rows:
        scene_id = row["scene_id"]
        passed = scene_id in PASS_IDS
        reviewed.append(
            {
                "scene_id": scene_id,
                "receiver": row["receiver"],
                "prompt_variant": row["prompt_variant"],
                "semantic_pass": "yes" if passed else "no",
                "selected_for_final20": "yes" if scene_id in SELECTED_IDS else "no",
                "failure_reason": "" if passed else FAIL_REASONS[scene_id],
                "review_basis": "12-frame contact sheet",
            }
        )
    write_csv(REVIEW_OUTPUT, reviewed)

    original_rows = read_csv(ROOT / "data/waterdrop_target_only_resting20.csv")
    original_review = read_csv(ROOT / "data/waterdrop_target_only_resting20_semantic_review.csv")
    original_pass = {
        row["scene_id"] for row in original_review if row["semantic_pass"] == "yes"
    }
    final_rows = []
    for row in original_rows:
        if row["scene_id"] in original_pass:
            final_rows.append(final_row(row, "original_resting20"))
    for row in rows:
        if row["scene_id"] in SELECTED_IDS:
            final_rows.append(final_row(row, "replenish30"))
    if len(final_rows) != 20:
        raise ValueError(f"expected 20 final samples, got {len(final_rows)}")
    write_csv(FINAL_OUTPUT, final_rows)
    print(f"review pass={len(PASS_IDS)}/30; selected=10; final={len(final_rows)}")


if __name__ == "__main__":
    main()
