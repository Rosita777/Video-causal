#!/usr/bin/env python3
"""Record manual labels for gate-positive Protocol v1 CogVideoX baselines."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


BASELINES = ("negative_prompt", "t2vunlearning_adapted", "videoeraser_official")

SOURCE_ABSENT = {
    ("eval_water_impact_seen_seen_00", "negative_prompt"),
    ("eval_water_impact_seen_seen_04", "negative_prompt"),
}

FOOTPRINT_ABSENT = {
    ("eval_water_impact_seen_unseen_02", "videoeraser_official"),
    ("eval_water_impact_seen_unseen_04", "videoeraser_official"),
    ("eval_water_impact_unseen_unseen_02", "videoeraser_official"),
    ("eval_water_impact_unseen_unseen_04", "videoeraser_official"),
    ("eval_powder_impact_seen_seen_03", "negative_prompt"),
    ("eval_powder_impact_seen_seen_03", "videoeraser_official"),
}

FOOTPRINT_PARTIAL = {
    ("eval_water_impact_seen_unseen_03", "videoeraser_official"),
    ("eval_water_impact_unseen_unseen_03", "videoeraser_official"),
    ("eval_powder_impact_seen_seen_04", "videoeraser_official"),
}

RECEIVER_NOT_PRESERVED = {
    ("eval_water_impact_seen_unseen_04", "videoeraser_official"),
    ("eval_water_impact_unseen_unseen_02", "videoeraser_official"),
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--original-gate", type=Path, required=True)
    parser.add_argument("--blind-key", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = {row["sample_id"]: row for row in read_csv(args.manifest)}
    gate = read_csv(args.original_gate)
    valid_ids = {row["sample_id"] for row in gate if row["gate_pass"] == "yes"}
    if len(valid_ids) != 30:
        raise ValueError(f"Expected 30 gate-positive rows, found {len(valid_ids)}")
    key = {
        (row["sample_id"], row["baseline"]): row
        for row in read_csv(args.blind_key)
    }

    output = []
    for sample_id in sorted(valid_ids):
        source = manifest[sample_id]
        for baseline in BASELINES:
            pair = (sample_id, baseline)
            source_absent = "yes" if pair in SOURCE_ABSENT else "no"
            if pair in FOOTPRINT_ABSENT:
                footprint_absent = "yes"
            elif pair in FOOTPRINT_PARTIAL:
                footprint_absent = "partial"
            else:
                footprint_absent = "no"
            receiver_preserved = "no" if pair in RECEIVER_NOT_PRESERVED else "yes"
            strict_success = (
                source_absent == "yes"
                and footprint_absent == "yes"
                and receiver_preserved == "yes"
            )
            if strict_success:
                notes = "source and footprint absent with receiver preserved"
            elif source_absent == "yes":
                notes = "source absent but causal footprint remains visible"
            elif footprint_absent == "yes":
                notes = "footprint absent but source remains visible"
            elif footprint_absent == "partial":
                notes = "source remains visible and footprint evidence is reduced or ambiguous"
            else:
                notes = "source and causal footprint remain visible"
            output.append(
                {
                    "sample_id": sample_id,
                    "mechanism": source["mechanism"],
                    "generalization_group": source["generalization_group"],
                    "baseline": baseline,
                    "source_object": source["source_object"],
                    "receiver": source["receiver"],
                    "expected_footprint": source["expected_footprint"],
                    "source_absent": source_absent,
                    "footprint_absent": footprint_absent,
                    "receiver_preserved": receiver_preserved,
                    "quality_ok": "yes",
                    "strict_erasure_success": "yes" if strict_success else "no",
                    "video_path": key[pair]["video_path"],
                    "review_sheet": key[pair]["candidate_strip_path"],
                    "notes": notes,
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)

    print(f"Wrote {len(output)} gate-positive baseline labels to {args.output}")
    for baseline in BASELINES:
        items = [row for row in output if row["baseline"] == baseline]
        counts = Counter(
            {
                "source_absent": sum(row["source_absent"] == "yes" for row in items),
                "footprint_absent": sum(row["footprint_absent"] == "yes" for row in items),
                "receiver_preserved": sum(row["receiver_preserved"] == "yes" for row in items),
                "strict_success": sum(row["strict_erasure_success"] == "yes" for row in items),
            }
        )
        print(
            f"{baseline}: source_absent={counts['source_absent']}/30 "
            f"footprint_absent={counts['footprint_absent']}/30 "
            f"receiver_preserved={counts['receiver_preserved']}/30 "
            f"strict={counts['strict_success']}/30"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
