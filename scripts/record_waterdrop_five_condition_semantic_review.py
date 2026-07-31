#!/usr/bin/env python3
"""Record the visual semantic review of the five-condition waterdrop pilot."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


TARGET_ONLY_PASS = {"wdfive0067"}
TARGET_ONLY_MISSING = {"wdfive0007"}
TARGET_ONLY_WRONG_OBJECT = {"wdfive0017"}
UNRELATED_FAILURES = {
    "wdfive0008": "no_clear_unrelated_footprint",
    "wdfive0018": "receiver_corrupted_by_unprompted_object",
}
CLEAN_FAILURES = {"wdfive0049": "unexpected_ripples_in_clean_control"}


def decision(row: dict[str, str]) -> tuple[str, str]:
    scene_id = row["scene_id"]
    condition = row["condition"]
    if condition in {"explicit_causal", "implicit_causal"}:
        return "pass", "visible_droplet_contact_and_causal_footprint"
    if condition == "target_only":
        if scene_id in TARGET_ONLY_PASS:
            return "pass", "droplet_remains_visibly_separated_without_footprint"
        if scene_id in TARGET_ONLY_MISSING:
            return "fail", "target_droplet_missing"
        if scene_id in TARGET_ONLY_WRONG_OBJECT:
            return "fail", "target_rendered_as_wrong_object"
        return "fail", "droplet_contacts_receiver_or_creates_footprint"
    if condition == "unrelated_footprint":
        reason = UNRELATED_FAILURES.get(scene_id)
        return ("fail", reason) if reason else ("pass", "footprint_visible_without_target_droplet")
    if condition == "clean_control":
        reason = CLEAN_FAILURES.get(scene_id)
        return ("fail", reason) if reason else ("pass", "no_target_or_waterdrop_footprint_visible")
    raise ValueError(f"unknown condition: {condition}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    args = parser.parse_args()
    root = args.repo_root.resolve()
    source = root / "data/waterdrop_five_condition_test100_auto_screen.csv"
    output = root / "data/waterdrop_five_condition_test100_semantic_review.csv"
    with source.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    reviewed = []
    for row in rows:
        status, reason = decision(row)
        reviewed.append({**row, "semantic_status": status, "semantic_reason": reason})
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(reviewed[0]))
        writer.writeheader()
        writer.writerows(reviewed)
    counts = Counter((row["condition"], row["semantic_status"]) for row in reviewed)
    for condition in sorted({row["condition"] for row in reviewed}):
        print(condition, "pass", counts[(condition, "pass")], "fail", counts[(condition, "fail")])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
