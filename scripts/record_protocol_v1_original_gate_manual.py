#!/usr/bin/env python3
"""Record the 2026-08-10 manual CogVideoX Original capability gate."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


PASS = {
    # Water impact: 14/20.
    "eval_water_impact_seen_seen_00",
    "eval_water_impact_seen_seen_01",
    "eval_water_impact_seen_seen_04",
    "eval_water_impact_unseen_seen_03",
    "eval_water_impact_unseen_seen_04",
    "eval_water_impact_seen_unseen_00",
    "eval_water_impact_seen_unseen_01",
    "eval_water_impact_seen_unseen_02",
    "eval_water_impact_seen_unseen_03",
    "eval_water_impact_seen_unseen_04",
    "eval_water_impact_unseen_unseen_00",
    "eval_water_impact_unseen_unseen_02",
    "eval_water_impact_unseen_unseen_03",
    "eval_water_impact_unseen_unseen_04",
    # Brittle fracture: 11/20.
    "eval_brittle_fracture_seen_seen_00",
    "eval_brittle_fracture_seen_seen_01",
    "eval_brittle_fracture_seen_seen_04",
    "eval_brittle_fracture_unseen_seen_00",
    "eval_brittle_fracture_unseen_seen_01",
    "eval_brittle_fracture_unseen_seen_04",
    "eval_brittle_fracture_seen_unseen_00",
    "eval_brittle_fracture_seen_unseen_01",
    "eval_brittle_fracture_seen_unseen_04",
    "eval_brittle_fracture_unseen_unseen_00",
    "eval_brittle_fracture_unseen_unseen_04",
    # Powder impact: 5/20.
    "eval_powder_impact_seen_seen_03",
    "eval_powder_impact_seen_seen_04",
    "eval_powder_impact_unseen_seen_00",
    "eval_powder_impact_unseen_unseen_00",
    "eval_powder_impact_unseen_unseen_04",
}

ORDER_FAILURES = {
    "eval_water_impact_unseen_seen_00",
    "eval_water_impact_unseen_seen_01",
    "eval_water_impact_unseen_seen_02",
    "eval_water_impact_unseen_unseen_01",
    "eval_brittle_fracture_unseen_unseen_01",
    "eval_powder_impact_seen_seen_00",
    "eval_powder_impact_unseen_seen_01",
}

PARTIAL_FOOTPRINT = {
    "eval_water_impact_seen_seen_02",
    "eval_powder_impact_seen_seen_02",
    "eval_powder_impact_unseen_seen_02",
    "eval_powder_impact_unseen_seen_03",
    "eval_powder_impact_unseen_seen_04",
    "eval_powder_impact_seen_unseen_00",
    "eval_powder_impact_unseen_unseen_01",
}

PARTIAL_SOURCE = {
    "eval_water_impact_seen_seen_03",
    "eval_rigid_collision_seen_seen_04",
    "eval_rigid_collision_seen_unseen_01",
    "eval_rigid_collision_unseen_unseen_01",
}

NO_SOURCE = {"eval_rigid_collision_seen_seen_01"}


def footprint_label(sample_id: str) -> str:
    if sample_id in PARTIAL_FOOTPRINT:
        return "partial"
    if sample_id in PASS or sample_id in ORDER_FAILURES:
        return "yes"
    return "no"


def note_for(sample_id: str, source: str, footprint: str) -> str:
    if sample_id in PASS:
        return "source-contact-footprint chain visible in temporal order"
    if sample_id in ORDER_FAILURES:
        return "footprint begins before a clear source contact or the clean prefix is violated"
    if footprint == "partial":
        return "only part of the expected footprint is visible"
    if source == "no":
        return "source is absent and the expected receiver response is absent"
    return "source appears but the expected downstream receiver response is absent"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 80:
        raise ValueError(f"Expected 80 rows, found {len(rows)}")
    manifest_ids = {row["sample_id"] for row in rows}
    referenced = PASS | ORDER_FAILURES | PARTIAL_FOOTPRINT | PARTIAL_SOURCE | NO_SOURCE
    unknown = referenced - manifest_ids
    if unknown:
        raise ValueError(f"Unknown manually labeled IDs: {sorted(unknown)}")

    output = []
    for row in rows:
        sample_id = row["sample_id"]
        source = "no" if sample_id in NO_SOURCE else "partial" if sample_id in PARTIAL_SOURCE else "yes"
        footprint = footprint_label(sample_id)
        if sample_id in PASS:
            causal_order = "yes"
        elif sample_id in ORDER_FAILURES or footprint == "no":
            causal_order = "no"
        else:
            causal_order = "partial"
        output.append(
            {
                "sample_id": sample_id,
                "mechanism": row["mechanism"],
                "generalization_group": row["generalization_group"],
                "source_object": row["source_object"],
                "receiver": row["receiver"],
                "expected_footprint": row["expected_footprint"],
                "source_visible": source,
                "footprint_visible": footprint,
                "causal_order_valid": causal_order,
                "receiver_correct": "yes",
                "quality_ok": "yes",
                "gate_pass": "yes" if sample_id in PASS else "no",
                "notes": note_for(sample_id, source, footprint),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    counts = Counter(row["mechanism"] for row in output if row["gate_pass"] == "yes")
    print(f"Wrote {len(output)} Original gate labels to {args.output}")
    for mechanism in sorted({row["mechanism"] for row in output}):
        print(f"{mechanism}: {counts[mechanism]}/20")
    print(f"all: {sum(counts.values())}/80")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
