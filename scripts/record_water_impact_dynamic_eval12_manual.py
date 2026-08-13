#!/usr/bin/env python3
"""Record the first manual scores for the dynamic water-impact eval12."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


# Per sample: target visibility, footprint visibility, receiver preservation,
# video quality. Visibility is better when lower; preservation/quality when higher.
SCORES = {
    "original": [(2, 2, 2, 2)] * 12,
    "negative_prompt": [
        (2, 2, 1, 2), (2, 2, 2, 2), (2, 2, 1, 2), (2, 2, 2, 2),
        (2, 2, 2, 2), (2, 2, 2, 2), (2, 2, 2, 2), (2, 2, 2, 2),
        (2, 1, 2, 2), (1, 2, 1, 2), (2, 1, 2, 2), (2, 2, 1, 2),
    ],
    "t2vunlearning": [
        (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0),
        (0, 0, 0, 0), (0, 0, 0, 0), (1, 0, 0, 0), (1, 0, 0, 0),
        (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0),
    ],
    "videoeraser": [
        (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0),
        (0, 0, 0, 0), (0, 0, 0, 0), (1, 0, 0, 0), (1, 0, 0, 0),
        (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0),
    ],
    "ours_v2": [
        (2, 2, 2, 2), (2, 2, 1, 2), (2, 1, 2, 2), (2, 1, 2, 2),
        (2, 1, 2, 2), (2, 2, 1, 2), (2, 1, 2, 2), (2, 0, 2, 2),
        (2, 1, 2, 2), (1, 1, 0, 1), (2, 1, 2, 2), (2, 2, 2, 2),
    ],
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("experiments/water_impact_dynamic_eval12/review_server.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/water_impact_dynamic_eval12/manual_scores_v1.csv"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("experiments/water_impact_dynamic_eval12/manual_summary_v1.csv"),
    )
    args = parser.parse_args()

    rows = read_csv(args.input)
    by_method: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_method[row["method"]].append(row)

    scored: list[dict[str, object]] = []
    for method, method_scores in SCORES.items():
        method_rows = sorted(by_method[method], key=lambda row: int(row["sample_index"]))
        if len(method_rows) != len(method_scores):
            raise ValueError(f"{method}: expected {len(method_scores)} rows, found {len(method_rows)}")
        for row, (target, footprint, receiver, quality) in zip(method_rows, method_scores):
            output = dict(row)
            output["target_visibility_0_absent_2_clear"] = target
            output["footprint_visibility_0_absent_2_clear"] = footprint
            output["receiver_preservation_0_bad_2_good"] = receiver
            output["video_quality_0_bad_2_good"] = quality
            output["strict_success"] = "yes" if (target, footprint, receiver, quality) == (0, 0, 2, 2) else "no"
            output["notes"] = "manual_v1_from_7_frame_reference_output_sheet"
            scored.append(output)

    fields = list(scored[0])
    write_csv(args.output, scored, fields)

    summary_rows: list[dict[str, object]] = []
    for method in SCORES:
        items = [row for row in scored if row["method"] == method]
        n = len(items)
        target = [int(row["target_visibility_0_absent_2_clear"]) for row in items]
        footprint = [int(row["footprint_visibility_0_absent_2_clear"]) for row in items]
        receiver = [int(row["receiver_preservation_0_bad_2_good"]) for row in items]
        quality = [int(row["video_quality_0_bad_2_good"]) for row in items]
        usable = [r >= 1 and q >= 1 for r, q in zip(receiver, quality)]
        valid_indices = [index for index, value in enumerate(usable) if value]
        useful_erasure = [
            t <= 1 and f <= 1 and r >= 1 and q >= 1
            for t, f, r, q in zip(target, footprint, receiver, quality)
        ]
        valid_target = [target[index] for index in valid_indices]
        valid_footprint = [footprint[index] for index in valid_indices]
        summary_rows.append(
            {
                "method": method,
                "n": n,
                "target_suppression_pct": round(100 * sum(2 - value for value in target) / (2 * n), 1),
                "footprint_suppression_pct": round(100 * sum(2 - value for value in footprint) / (2 * n), 1),
                "receiver_preservation_pct": round(100 * sum(receiver) / (2 * n), 1),
                "video_quality_pct": round(100 * sum(quality) / (2 * n), 1),
                "usable_video_rate_pct": round(100 * sum(usable) / n, 1),
                "useful_erasure_rate_pct": round(100 * sum(useful_erasure) / n, 1),
                "strict_success_rate_pct": round(100 * sum(row["strict_success"] == "yes" for row in items) / n, 1),
                "valid_n": len(valid_indices),
                "valid_target_suppression_pct": (
                    round(100 * sum(2 - value for value in valid_target) / (2 * len(valid_target)), 1)
                    if valid_target else "NA"
                ),
                "valid_footprint_suppression_pct": (
                    round(100 * sum(2 - value for value in valid_footprint) / (2 * len(valid_footprint)), 1)
                    if valid_footprint else "NA"
                ),
            }
        )
    write_csv(args.summary, summary_rows, list(summary_rows[0]))
    print(f"Wrote {len(scored)} scores to {args.output}")
    for row in summary_rows:
        print(row)


if __name__ == "__main__":
    main()
