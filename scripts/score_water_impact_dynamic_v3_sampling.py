#!/usr/bin/env python3
"""Unblind, summarize, and gate the water-impact v3 sampling experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


METHODS = ("balanced", "exposure")
EXPECTED_SAMPLE_INDICES = set(range(12))
EXPECTED_EVAL_SHA256 = "dca68f8632e10ef83cc5f3867679c9cba54f4cbce96426db5db8c5214ac1ec1a"
EXPECTED_PROMPTS_SHA256 = "06dae57a0202e2d53e32fc02f9b26fd694237755a18f85bdd67c728bf706681c"
EXPECTED_TRAIN_SHA256 = "3d4d8cbf9244b1575357f0ac74380cd7cb6265df4d1a85bf450de4cac120aee4"
SCORE_FIELDS = (
    "target_visibility_0_absent_2_clear",
    "footprint_visibility_0_absent_2_clear",
    "receiver_preservation_0_bad_2_good",
    "video_quality_0_bad_2_good",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def review_binding_sha256(rows: list[dict[str, str]]) -> str:
    fields = (
        "review_id",
        "sample_index",
        "pair_id",
        "generalization_group",
        "candidate_code",
        "composite_path",
        "source_object",
        "receiver",
    )
    canonical = [
        {field: str(row[field]) for field in fields}
        for row in sorted(rows, key=lambda item: item["review_id"])
    ]
    return hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def score_value(row: dict[str, str], field: str) -> int:
    try:
        value = int(row[field])
    except (KeyError, ValueError) as exc:
        raise ValueError(f"{row.get('review_id', '<unknown>')}: invalid {field}") from exc
    if value not in {0, 1, 2}:
        raise ValueError(f"{row['review_id']}: {field} must be 0, 1, or 2")
    return value


def is_usable(row: dict[str, object]) -> bool:
    return int(row[SCORE_FIELDS[2]]) >= 1 and int(row[SCORE_FIELDS[3]]) >= 1


def is_strict(row: dict[str, object]) -> bool:
    return tuple(int(row[field]) for field in SCORE_FIELDS) == (0, 0, 2, 2)


def method_summary(method: str, rows: list[dict[str, object]]) -> dict[str, object]:
    n = len(rows)
    target = [int(row[SCORE_FIELDS[0]]) for row in rows]
    footprint = [int(row[SCORE_FIELDS[1]]) for row in rows]
    receiver = [int(row[SCORE_FIELDS[2]]) for row in rows]
    quality = [int(row[SCORE_FIELDS[3]]) for row in rows]
    usable = [is_usable(row) for row in rows]
    return {
        "method": method,
        "n": n,
        "usable_n": sum(usable),
        "receiver_points": sum(receiver),
        "quality_points": sum(quality),
        "target_suppression_points": sum(2 - value for value in target),
        "footprint_suppression_points": sum(2 - value for value in footprint),
        "strict_success_n": sum(is_strict(row) for row in rows),
    }


def validate_review_artifacts(
    review_manifest: dict[str, object], review_rows: list[dict[str, str]]
) -> None:
    expected_composites = review_manifest.get("composite_sha256")
    if not isinstance(expected_composites, dict) or len(expected_composites) != 12:
        raise ValueError("review manifest must contain exactly 12 composite hashes")
    composite_paths = {Path(row["composite_path"]) for row in review_rows}
    if len(composite_paths) != 12:
        raise ValueError("review rows must reference exactly 12 composites")
    expected_names = set(expected_composites)
    if {path.name for path in composite_paths} != expected_names:
        raise ValueError("review composite paths do not match review manifest")
    for path in composite_paths:
        if not path.is_file():
            raise FileNotFoundError(f"missing review composite: {path}")
        if file_sha256(path) != expected_composites[path.name]:
            raise ValueError(f"review composite hash mismatch: {path}")

    generation_manifests = review_manifest.get("generation_manifests")
    if not isinstance(generation_manifests, dict) or set(generation_manifests) != {
        "original",
        "balanced",
        "exposure",
    }:
        raise ValueError("review manifest generation provenance is incomplete")
    for label, record in generation_manifests.items():
        if not isinstance(record, dict):
            raise ValueError(f"invalid generation provenance for {label}")
        path = Path(str(record.get("path", "")))
        if not path.is_file() or file_sha256(path) != record.get("sha256"):
            raise ValueError(f"generation manifest hash mismatch: {label}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--answer-key", type=Path, required=True)
    parser.add_argument("--review-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.output_dir.exists():
        parser.error(f"refusing to overwrite score directory: {args.output_dir}")
    review_manifest_path = args.review_manifest or args.review.parent / "review_manifest.json"
    review_manifest = json.loads(review_manifest_path.read_text(encoding="utf-8"))
    expected_review_metadata = {
        "eval_csv_sha256": EXPECTED_EVAL_SHA256,
        "prompts_sha256": EXPECTED_PROMPTS_SHA256,
        "train_manifest_sha256": EXPECTED_TRAIN_SHA256,
        "sample_count": 12,
        "review_rows": 24,
    }
    for field, expected_value in expected_review_metadata.items():
        if review_manifest.get(field) != expected_value:
            raise ValueError(f"review manifest {field} mismatch")
    if review_manifest.get("answer_key_sha256") != file_sha256(args.answer_key):
        raise ValueError("answer-key hash does not match review manifest")
    review_rows = read_csv(args.review)
    key_rows = read_csv(args.answer_key)
    if len(review_rows) != 24 or len(key_rows) != 24:
        raise ValueError("eval12 scoring requires exactly 24 review and answer-key rows")
    if review_manifest.get("review_binding_sha256") != review_binding_sha256(review_rows):
        raise ValueError("immutable review binding does not match review manifest")
    validate_review_artifacts(review_manifest, review_rows)
    key_by_id = {row["review_id"]: row for row in key_rows}
    if len(key_by_id) != len(key_rows):
        raise ValueError("duplicate review_id in answer key")
    review_ids = [row["review_id"] for row in review_rows]
    if len(set(review_ids)) != len(review_ids):
        raise ValueError("duplicate review_id in review")
    if set(review_ids) != set(key_by_id):
        raise ValueError("review and answer-key IDs do not match")

    unblinded: list[dict[str, object]] = []
    for row in review_rows:
        key = key_by_id[row["review_id"]]
        for field in (
            "sample_index",
            "pair_id",
            "generalization_group",
            "candidate_code",
        ):
            if row[field] != key[field]:
                raise ValueError(f"{row['review_id']}: answer-key mismatch for {field}")
        method = key["method"]
        if method not in METHODS:
            raise ValueError(f"unexpected method: {method}")
        output: dict[str, object] = {
            "review_id": row["review_id"],
            "sample_index": int(row["sample_index"]),
            "pair_id": row["pair_id"],
            "generalization_group": row["generalization_group"],
            "method": method,
            "video_path": key["video_path"],
        }
        for field in SCORE_FIELDS:
            output[field] = score_value(row, field)
        output["usable"] = "yes" if is_usable(output) else "no"
        output["strict_success"] = "yes" if is_strict(output) else "no"
        output["notes"] = row.get("notes", "")
        unblinded.append(output)

    by_method: dict[str, list[dict[str, object]]] = defaultdict(list)
    by_sample: dict[int, dict[str, dict[str, object]]] = defaultdict(dict)
    for row in unblinded:
        method = str(row["method"])
        sample_index = int(row["sample_index"])
        by_method[method].append(row)
        if method in by_sample[sample_index]:
            raise ValueError(f"duplicate {method} row for sample {sample_index}")
        by_sample[sample_index][method] = row
    if set(by_method) != set(METHODS):
        raise ValueError(f"expected methods {METHODS}, found {sorted(by_method)}")
    if set(by_sample) != EXPECTED_SAMPLE_INDICES:
        raise ValueError("eval12 scoring requires sample indices 0 through 11")
    if any(len(by_method[method]) != 12 for method in METHODS):
        raise ValueError("eval12 scoring requires 12 rows per method")
    if any(set(pair) != set(METHODS) for pair in by_sample.values()):
        raise ValueError("each sample must contain balanced and exposure rows")

    for rows in by_method.values():
        rows.sort(key=lambda row: int(row["sample_index"]))
    summaries = [method_summary(method, by_method[method]) for method in METHODS]
    balanced_summary, exposure_summary = summaries

    improvements = []
    for sample_index, pair in sorted(by_sample.items()):
        balanced = pair["balanced"]
        exposure = pair["exposure"]
        if (
            int(balanced[SCORE_FIELDS[0]]) == 2
            and int(exposure[SCORE_FIELDS[0]]) <= 1
            and is_usable(exposure)
        ):
            improvements.append(
                {
                    "sample_index": sample_index,
                    "generalization_group": exposure["generalization_group"],
                    "exposure_target": int(exposure[SCORE_FIELDS[0]]),
                }
            )

    control_valid = [row for row in by_method["balanced"] if is_usable(row)]
    exposure_by_sample = {int(row["sample_index"]): row for row in by_method["exposure"]}
    control_footprint_points = sum(2 - int(row[SCORE_FIELDS[1]]) for row in control_valid)
    exposure_footprint_points = sum(
        2 - int(exposure_by_sample[int(row["sample_index"])][SCORE_FIELDS[1]])
        if is_usable(exposure_by_sample[int(row["sample_index"])])
        else 0
        for row in control_valid
    )

    checks = {
        "source_improvements_at_least_3": len(improvements) >= 3,
        "source_absent_at_least_1": any(
            row["exposure_target"] == 0 for row in improvements
        ),
        "source_improvement_groups_at_least_2": len(
            {str(row["generalization_group"]) for row in improvements}
        ) >= 2,
        "exposure_usable_at_least_11": int(exposure_summary["usable_n"]) >= 11,
        "receiver_absolute_floor": int(exposure_summary["receiver_points"]) >= 19,
        "receiver_relative_floor": int(exposure_summary["receiver_points"])
        >= int(balanced_summary["receiver_points"]) - 1,
        "quality_absolute_floor": int(exposure_summary["quality_points"]) >= 22,
        "quality_relative_floor": int(exposure_summary["quality_points"])
        >= int(balanced_summary["quality_points"]) - 1,
        "footprint_absolute_floor": exposure_footprint_points >= 7,
        "footprint_relative_floor": (
            exposure_footprint_points >= control_footprint_points - 1
        ),
        "strict_success_at_least_1": int(exposure_summary["strict_success_n"]) >= 1,
    }
    source_positive_names = (
        "source_improvements_at_least_3",
        "source_absent_at_least_1",
        "source_improvement_groups_at_least_2",
    )
    source_positive = all(checks[name] for name in source_positive_names)
    gate = {
        "source_positive": source_positive,
        "promote_operating_point": source_positive and all(checks.values()),
        "checks": checks,
        "source_improvements": improvements,
        "control_valid_sample_count": len(control_valid),
        "control_footprint_points": control_footprint_points,
        "exposure_footprint_points_on_control_valid": exposure_footprint_points,
        "input_provenance": {
            "review_sha256": file_sha256(args.review),
            "answer_key_sha256": file_sha256(args.answer_key),
            "review_manifest_sha256": file_sha256(review_manifest_path),
            "eval_csv_sha256": review_manifest["eval_csv_sha256"],
            "generation_manifests": review_manifest.get("generation_manifests", {}),
        },
    }

    args.output_dir.mkdir(parents=True)
    write_csv(
        args.output_dir / "unblinded_scores.csv",
        sorted(
            unblinded,
            key=lambda row: (int(row["sample_index"]), str(row["method"])),
        ),
    )
    write_csv(args.output_dir / "summary.csv", summaries)
    (args.output_dir / "gate.json").write_text(
        json.dumps(gate, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(gate, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
