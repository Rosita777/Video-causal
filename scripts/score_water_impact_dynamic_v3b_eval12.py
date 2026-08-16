#!/usr/bin/env python3
"""Unblind and apply the frozen v3b eval12 development gate."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from build_water_impact_dynamic_v3b_blind_review import (
    BLIND_SEED,
    PRIVATE_OUTPUT_DIR,
    PUBLIC_OUTPUT_DIR,
    review_binding_sha256,
)
from water_impact_dynamic_v3b_eval_protocol import (
    BALANCED_GENERATION_MANIFEST_SHA256,
    BALANCED_RUN,
    EVAL_CSV,
    EVAL_CSV_SHA256,
    FRAME_INDICES,
    METHODS,
    ORIGINAL_GENERATION_MANIFEST_SHA256,
    ORIGINAL_RUN,
    PROMPTS_SHA256,
    PROTOCOL,
    SCORE_FIELDS,
    TRAIN_MANIFEST,
    TRAIN_MANIFEST_SHA256,
    V3B_RUN,
    file_sha256,
    load_frozen_inputs,
    load_generation_run,
    resolve_path,
    validate_balanced_checkpoint,
    validate_training_caches,
    validate_v3b_checkpoint,
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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


def is_usable(row: dict[str, Any]) -> bool:
    return int(row[SCORE_FIELDS[2]]) >= 1 and int(row[SCORE_FIELDS[3]]) >= 1


def is_strict(row: dict[str, Any]) -> bool:
    return tuple(int(row[field]) for field in SCORE_FIELDS) == (0, 0, 2, 2)


def method_summary(method: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "method": method,
        "n": len(rows),
        "usable_n": sum(is_usable(row) for row in rows),
        "receiver_points": sum(int(row[SCORE_FIELDS[2]]) for row in rows),
        "quality_points": sum(int(row[SCORE_FIELDS[3]]) for row in rows),
        "target_suppression_points": sum(2 - int(row[SCORE_FIELDS[0]]) for row in rows),
        "footprint_suppression_points": sum(2 - int(row[SCORE_FIELDS[1]]) for row in rows),
        "strict_success_n": sum(is_strict(row) for row in rows),
    }


def validate_frozen_blind_assignment(
    project_root: Path,
    public_dir: Path,
    eval_rows: list[dict[str, str]],
    review_rows: list[dict[str, str]],
    key_rows: list[dict[str, str]],
) -> None:
    review_by_id = {row["review_id"]: row for row in review_rows}
    key_by_id = {row["review_id"]: row for row in key_rows}
    sample_order = list(range(12))
    random.Random(BLIND_SEED).shuffle(sample_order)
    expected_review_ids: set[str] = set()
    for review_position, sample_index in enumerate(sample_order):
        eval_row = eval_rows[sample_index]
        ordered_methods = list(METHODS)
        random.Random(f"{BLIND_SEED}:{eval_row['pair_id']}").shuffle(ordered_methods)
        expected_composite = (
            public_dir / "composites" / f"r{review_position:03d}.jpg"
        ).resolve(strict=True)
        for candidate_index, method in enumerate(ordered_methods):
            code = chr(ord("A") + candidate_index)
            review_id = f"r{review_position:03d}_{code}"
            expected_review_ids.add(review_id)
            review_row = review_by_id.get(review_id)
            key_row = key_by_id.get(review_id)
            if review_row is None or key_row is None:
                raise ValueError("review package does not match the frozen blind assignment")
            expected_fields = {
                "sample_index": str(sample_index),
                "pair_id": eval_row["pair_id"],
                "generalization_group": eval_row["generalization_group"],
                "candidate_code": code,
            }
            for field, expected in expected_fields.items():
                if review_row.get(field) != expected or key_row.get(field) != expected:
                    raise ValueError(f"{review_id}: frozen blind field mismatch for {field}")
            if review_row.get("source_object") != eval_row["source_object"]:
                raise ValueError(f"{review_id}: frozen source-object annotation mismatch")
            if review_row.get("receiver") != eval_row["receiver"]:
                raise ValueError(f"{review_id}: frozen receiver annotation mismatch")
            if key_row.get("method") != method:
                raise ValueError(f"{review_id}: frozen blind method assignment mismatch")
            actual_composite = resolve_path(
                project_root, str(review_row.get("composite_path", ""))
            ).resolve(strict=True)
            if actual_composite != expected_composite:
                raise ValueError(f"{review_id}: frozen composite-path assignment mismatch")
            expected_media = (
                public_dir / "media" / f"{review_id}.mp4"
            ).resolve(strict=True)
            actual_media = resolve_path(
                project_root, str(review_row.get("candidate_video_path", ""))
            ).resolve(strict=True)
            if actual_media != expected_media:
                raise ValueError(f"{review_id}: frozen candidate-video path mismatch")
    if set(review_by_id) != expected_review_ids or set(key_by_id) != expected_review_ids:
        raise ValueError("review package contains unexpected blinded IDs")


def validate_anonymous_media(
    project_root: Path,
    public_dir: Path,
    review_manifest: dict[str, Any],
    review_rows: list[dict[str, str]],
    key_rows: list[dict[str, str]],
    current_videos: dict[str, dict[int, Path]],
) -> None:
    if any(
        forbidden in review_rows[0]
        for forbidden in ("method", "video_path", "source_video_path")
    ):
        raise ValueError("blind review CSV exposes a hidden method or source-video field")
    records = review_manifest.get("anonymous_media_sha256")
    expected_names = {
        f"r{position:03d}_{code}.mp4"
        for position in range(12)
        for code in ("A", "B")
    }
    if not isinstance(records, dict) or set(records) != expected_names:
        raise ValueError("review manifest must bind exactly 24 anonymous candidate videos")
    media_dir = public_dir / "media"
    if not media_dir.is_dir() or media_dir.is_symlink():
        raise ValueError("anonymous media directory is missing or is a symlink")
    actual_files = list(media_dir.rglob("*.mp4"))
    expected_paths = {(media_dir / name).resolve(strict=True) for name in expected_names}
    if len(actual_files) != 24 or {
        path.resolve(strict=True) for path in actual_files
    } != expected_paths:
        raise ValueError("anonymous media directory does not contain the frozen 24-file inventory")
    review_by_id = {row["review_id"]: row for row in review_rows}
    key_by_id = {row["review_id"]: row for row in key_rows}
    for name in sorted(expected_names):
        review_id = name.removesuffix(".mp4")
        expected_path = media_dir / name
        review_path = resolve_path(
            project_root, review_by_id[review_id]["candidate_video_path"]
        )
        record = records[name]
        if not isinstance(record, dict):
            raise ValueError(f"{review_id}: invalid anonymous-media provenance")
        recorded_path = resolve_path(project_root, str(record.get("path", "")))
        if (
            review_path.resolve(strict=True) != expected_path.resolve(strict=True)
            or recorded_path.resolve(strict=True) != expected_path.resolve(strict=True)
        ):
            raise ValueError(f"{review_id}: anonymous-media path mismatch")
        if expected_path.is_symlink() or not expected_path.is_file() or expected_path.stat().st_size == 0:
            raise ValueError(f"{review_id}: anonymous media is missing, empty, or a symlink")
        actual_hash = file_sha256(expected_path)
        if record.get("sha256") != actual_hash:
            raise ValueError(f"{review_id}: anonymous-media hash mismatch")
        key = key_by_id[review_id]
        source = current_videos[key["method"]][int(key["sample_index"])]
        if expected_path.samefile(source):
            raise ValueError(f"{review_id}: anonymous media is not an independent file copy")
        if actual_hash != file_sha256(source):
            raise ValueError(f"{review_id}: anonymous media differs from answer-key source video")


def validate_review_artifacts(
    project_root: Path,
    public_dir: Path,
    manifest_path: Path,
    review_manifest: dict[str, Any],
    review_rows: list[dict[str, str]],
    key_rows: list[dict[str, str]],
) -> None:
    expected_metadata = {
        "protocol": PROTOCOL,
        "eval_csv": EVAL_CSV,
        "eval_csv_sha256": EVAL_CSV_SHA256,
        "prompts_sha256": PROMPTS_SHA256,
        "train_manifest": TRAIN_MANIFEST,
        "train_manifest_sha256": TRAIN_MANIFEST_SHA256,
        "blind_seed": BLIND_SEED,
        "frame_indices": list(FRAME_INDICES),
        "sample_count": 12,
        "review_rows": 24,
        "methods": list(METHODS),
    }
    for field, expected in expected_metadata.items():
        if review_manifest.get(field) != expected:
            raise ValueError(f"review manifest {field} does not match the frozen protocol")

    composite_hashes = review_manifest.get("composite_sha256")
    if not isinstance(composite_hashes, dict) or len(composite_hashes) != 12:
        raise ValueError("review manifest must bind exactly 12 composites")
    composite_paths = {resolve_path(project_root, row["composite_path"]) for row in review_rows}
    if len(composite_paths) != 12:
        raise ValueError("review rows must reference exactly 12 composites")
    expected_composite_root = (public_dir / "composites").resolve(strict=True)
    if {path.name for path in composite_paths} != set(composite_hashes):
        raise ValueError("review composite paths do not match the review manifest")
    for path in composite_paths:
        if not path.is_file():
            raise FileNotFoundError(f"missing review composite: {path}")
        try:
            path.resolve(strict=True).relative_to(expected_composite_root)
        except ValueError as exc:
            raise ValueError(f"review composite escapes its package: {path}") from exc
        if file_sha256(path) != composite_hashes[path.name]:
            raise ValueError(f"review composite hash mismatch: {path}")

    eval_rows, train_rows = load_frozen_inputs(project_root)
    validate_frozen_blind_assignment(
        project_root, public_dir, eval_rows, review_rows, key_rows
    )

    run_specs = {
        "original": (ORIGINAL_RUN, ORIGINAL_GENERATION_MANIFEST_SHA256),
        "balanced": (BALANCED_RUN, BALANCED_GENERATION_MANIFEST_SHA256),
        "v3b": (V3B_RUN, None),
    }
    current_manifests: dict[str, Path] = {}
    current_videos: dict[str, dict[int, Path]] = {}
    for label, (run_dir, frozen_hash) in run_specs.items():
        current_manifests[label], _, current_videos[label] = load_generation_run(
            project_root,
            run_dir,
            label,
            eval_rows,
            expected_manifest_sha256=frozen_hash,
        )
    recorded_manifests = review_manifest.get("generation_manifests")
    if not isinstance(recorded_manifests, dict) or set(recorded_manifests) != set(run_specs):
        raise ValueError("review manifest generation provenance is incomplete")
    for label, path in current_manifests.items():
        record = recorded_manifests[label]
        if not isinstance(record, dict):
            raise ValueError(f"invalid generation provenance for {label}")
        recorded_path = resolve_path(project_root, str(record.get("path", "")))
        if recorded_path.resolve(strict=True) != path.resolve(strict=True):
            raise ValueError(f"generation manifest path mismatch: {label}")
        if record.get("sha256") != file_sha256(path):
            raise ValueError(f"generation manifest hash mismatch: {label}")

    recorded_videos = review_manifest.get("video_sha256")
    if not isinstance(recorded_videos, dict) or set(recorded_videos) != set(run_specs):
        raise ValueError("review manifest video provenance is incomplete")
    for label, arm in current_videos.items():
        records = recorded_videos[label]
        if not isinstance(records, dict) or set(records) != {str(index) for index in range(12)}:
            raise ValueError(f"{label}: review manifest must bind 12 video hashes")
        for index, path in arm.items():
            record = records[str(index)]
            if not isinstance(record, dict):
                raise ValueError(f"{label}: invalid video provenance for index {index}")
            recorded_path = resolve_path(project_root, str(record.get("path", "")))
            if recorded_path.resolve(strict=True) != path.resolve(strict=True):
                raise ValueError(f"{label}: video path mismatch at index {index}")
            if record.get("sha256") != file_sha256(path):
                raise ValueError(f"{label}: video hash mismatch at index {index}")

    key_video_bindings = {
        (row["method"], int(row["sample_index"])): resolve_path(project_root, row["video_path"])
        for row in key_rows
    }
    if set(key_video_bindings) != {(method, index) for method in METHODS for index in range(12)}:
        raise ValueError("answer key does not bind one controlled video per method and sample")
    for (method, index), path in key_video_bindings.items():
        if path.resolve(strict=True) != current_videos[method][index].resolve(strict=True):
            raise ValueError(f"answer-key video mismatch: {method} sample {index}")
    validate_anonymous_media(
        project_root,
        public_dir,
        review_manifest,
        review_rows,
        key_rows,
        current_videos,
    )

    current_training = {
        "inputs": validate_training_caches(project_root, train_rows),
        "balanced": validate_balanced_checkpoint(project_root),
        "v3b": validate_v3b_checkpoint(project_root),
    }
    if review_manifest.get("training_provenance") != current_training:
        raise ValueError("review manifest training provenance no longer matches frozen artifacts")


def validate_public_private_inventory(
    project_root: Path,
    review_path: Path,
    answer_key_path: Path,
    manifest_path: Path,
) -> tuple[Path, Path]:
    public_dir = review_path.parent
    private_dir = answer_key_path.parent
    if private_dir != manifest_path.parent:
        raise ValueError("answer key and review manifest must share the private directory")
    if public_dir.parent.resolve() != private_dir.parent.resolve() or public_dir == private_dir:
        raise ValueError("public and private review directories must be distinct siblings")
    if public_dir.is_symlink() or not public_dir.is_dir():
        raise ValueError("public review directory is missing or is a symlink")
    if private_dir.is_symlink() or not private_dir.is_dir():
        raise ValueError("private review directory is missing or is a symlink")
    expected_public_root = resolve_path(project_root, PUBLIC_OUTPUT_DIR)
    expected_private_root = resolve_path(project_root, PRIVATE_OUTPUT_DIR)
    if (
        public_dir.resolve(strict=True) != expected_public_root.resolve(strict=True)
        or private_dir.resolve(strict=True) != expected_private_root.resolve(strict=True)
    ):
        raise ValueError("public/private review directories are outside the frozen v3 paths")
    expected_review = public_dir / "blind_review.csv"
    expected_key = private_dir / "answer_key.csv"
    expected_manifest = private_dir / "review_manifest.json"
    if (
        review_path.resolve(strict=True) != expected_review.resolve(strict=True)
        or answer_key_path.resolve(strict=True) != expected_key.resolve(strict=True)
        or manifest_path.resolve(strict=True) != expected_manifest.resolve(strict=True)
    ):
        raise ValueError("public/private review file paths do not match the frozen v3 layout")
    if any(path.is_symlink() for path in (review_path, answer_key_path, manifest_path)):
        raise ValueError("public/private CSV or manifest must not be a symlink")
    public_entries = {path.name: path for path in public_dir.iterdir()}
    if set(public_entries) != {"blind_review.csv", "composites", "media"}:
        raise ValueError("public review directory contains an unexpected entry")
    private_entries = {path.name: path for path in private_dir.iterdir()}
    if set(private_entries) != {"answer_key.csv", "review_manifest.json"}:
        raise ValueError("private review directory must contain exactly two frozen files")
    composite_dir = public_entries["composites"]
    media_dir = public_entries["media"]
    if (
        not composite_dir.is_dir()
        or composite_dir.is_symlink()
        or not media_dir.is_dir()
        or media_dir.is_symlink()
    ):
        raise ValueError("public composite/media entries must be real directories")
    expected_composites = {f"r{position:03d}.jpg" for position in range(12)}
    composite_entries = list(composite_dir.iterdir())
    if (
        {path.name for path in composite_entries} != expected_composites
        or any(not path.is_file() or path.is_symlink() for path in composite_entries)
    ):
        raise ValueError("public composites directory must contain exactly 12 real JPEG files")
    expected_media = {
        f"r{position:03d}_{code}.mp4"
        for position in range(12)
        for code in ("A", "B")
    }
    media_entries = list(media_dir.iterdir())
    if (
        {path.name for path in media_entries} != expected_media
        or any(not path.is_file() or path.is_symlink() for path in media_entries)
    ):
        raise ValueError("public media directory must contain exactly 24 real MP4 files")
    return public_dir, private_dir


def compute_gate(unblinded: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_sample: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in unblinded:
        method = str(row["method"])
        sample_index = int(row["sample_index"])
        if method in by_sample[sample_index]:
            raise ValueError(f"duplicate {method} row for sample {sample_index}")
        by_method[method].append(row)
        by_sample[sample_index][method] = row
    if set(by_method) != set(METHODS) or any(len(by_method[method]) != 12 for method in METHODS):
        raise ValueError("eval12 scoring requires exactly 12 rows per controlled method")
    if set(by_sample) != set(range(12)) or any(set(pair) != set(METHODS) for pair in by_sample.values()):
        raise ValueError("eval12 scoring requires balanced and v3b rows for samples 0 through 11")
    for rows in by_method.values():
        rows.sort(key=lambda row: int(row["sample_index"]))
    summaries = [method_summary(method, by_method[method]) for method in METHODS]
    summary_by_method = {str(row["method"]): row for row in summaries}
    balanced_summary = summary_by_method["balanced"]
    v3b_summary = summary_by_method["v3b"]

    control_usable = [row for row in by_method["balanced"] if is_usable(row)]
    v3b_by_sample = {int(row["sample_index"]): row for row in by_method["v3b"]}
    improvements: list[dict[str, Any]] = []
    for control in control_usable:
        sample_index = int(control["sample_index"])
        treatment = v3b_by_sample[sample_index]
        if (
            int(control[SCORE_FIELDS[0]]) == 2
            and int(treatment[SCORE_FIELDS[0]]) <= 1
            and is_usable(treatment)
        ):
            improvements.append(
                {
                    "sample_index": sample_index,
                    "generalization_group": treatment["generalization_group"],
                    "v3b_target_visibility": int(treatment[SCORE_FIELDS[0]]),
                }
            )

    control_target_points = sum(2 - int(row[SCORE_FIELDS[0]]) for row in control_usable)
    v3b_target_points = sum(
        2 - int(v3b_by_sample[int(row["sample_index"])][SCORE_FIELDS[0]])
        if is_usable(v3b_by_sample[int(row["sample_index"])])
        else 0
        for row in control_usable
    )
    control_footprint_points = sum(2 - int(row[SCORE_FIELDS[1]]) for row in control_usable)
    v3b_footprint_points = sum(
        2 - int(v3b_by_sample[int(row["sample_index"])][SCORE_FIELDS[1]])
        if is_usable(v3b_by_sample[int(row["sample_index"])])
        else 0
        for row in control_usable
    )
    checks = {
        "mechanism_improvements_at_least_3": len(improvements) >= 3,
        "mechanism_absent_at_least_1": any(
            row["v3b_target_visibility"] == 0 for row in improvements
        ),
        "mechanism_groups_at_least_2": len(
            {str(row["generalization_group"]) for row in improvements}
        )
        >= 2,
        "target_suppression_margin_at_least_3_on_control_usable": (
            v3b_target_points >= control_target_points + 3
        ),
        "v3b_usable_at_least_11": int(v3b_summary["usable_n"]) >= 11,
        "receiver_absolute_floor_19": int(v3b_summary["receiver_points"]) >= 19,
        "receiver_relative_floor_control_minus_1": int(v3b_summary["receiver_points"])
        >= int(balanced_summary["receiver_points"]) - 1,
        "quality_absolute_floor_16": int(v3b_summary["quality_points"]) >= 16,
        "quality_relative_floor_control_minus_1": int(v3b_summary["quality_points"])
        >= int(balanced_summary["quality_points"]) - 1,
        "footprint_not_worse_on_control_usable": (
            v3b_footprint_points >= control_footprint_points
        ),
        "strict_success_at_least_1": int(v3b_summary["strict_success_n"]) >= 1,
    }
    mechanism_names = (
        "mechanism_improvements_at_least_3",
        "mechanism_absent_at_least_1",
        "mechanism_groups_at_least_2",
        "target_suppression_margin_at_least_3_on_control_usable",
    )
    preservation_names = tuple(name for name in checks if name not in mechanism_names)
    mechanism_positive = all(checks[name] for name in mechanism_names)
    preservation_positive = all(checks[name] for name in preservation_names)
    gate = {
        "protocol": PROTOCOL,
        "mechanism_positive": mechanism_positive,
        "preservation_positive": preservation_positive,
        "promote_v3b_operating_point": mechanism_positive and preservation_positive,
        "checks": checks,
        "mechanism_improvements": improvements,
        "control_usable_sample_count": len(control_usable),
        "control_target_suppression_points_on_control_usable": control_target_points,
        "v3b_target_suppression_points_on_control_usable": v3b_target_points,
        "control_footprint_suppression_points_on_control_usable": control_footprint_points,
        "v3b_footprint_suppression_points_on_control_usable": v3b_footprint_points,
    }
    return summaries, gate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--answer-key", type=Path, required=True)
    parser.add_argument("--review-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error(f"refusing to overwrite score directory: {args.output_dir}")
    manifest_path = args.review_manifest
    public_dir, _ = validate_public_private_inventory(
        Path.cwd(), args.review, args.answer_key, manifest_path
    )
    review_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if review_manifest.get("answer_key_sha256") != file_sha256(args.answer_key):
        raise ValueError("answer-key hash does not match review manifest")
    review_rows = read_csv(args.review)
    key_rows = read_csv(args.answer_key)
    if len(review_rows) != 24 or len(key_rows) != 24:
        raise ValueError("v3b eval12 scoring requires exactly 24 review and answer-key rows")
    if review_manifest.get("review_binding_sha256") != review_binding_sha256(review_rows):
        raise ValueError("immutable review binding does not match review manifest")
    validate_review_artifacts(
        Path.cwd(), public_dir, manifest_path, review_manifest, review_rows, key_rows
    )

    key_by_id = {row["review_id"]: row for row in key_rows}
    if len(key_by_id) != 24:
        raise ValueError("duplicate review_id in answer key")
    review_ids = [row["review_id"] for row in review_rows]
    if len(set(review_ids)) != 24 or set(review_ids) != set(key_by_id):
        raise ValueError("review and answer-key IDs do not form the frozen 24-row set")
    unblinded: list[dict[str, Any]] = []
    for row in review_rows:
        key = key_by_id[row["review_id"]]
        for field in ("sample_index", "pair_id", "generalization_group", "candidate_code"):
            if row[field] != key[field]:
                raise ValueError(f"{row['review_id']}: answer-key mismatch for {field}")
        method = key["method"]
        if method not in METHODS:
            raise ValueError(f"unexpected method in answer key: {method}")
        output: dict[str, Any] = {
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

    summaries, gate = compute_gate(unblinded)
    gate["input_provenance"] = {
        "review_sha256": file_sha256(args.review),
        "answer_key_sha256": file_sha256(args.answer_key),
        "review_manifest_sha256": file_sha256(manifest_path),
        "eval_csv_sha256": EVAL_CSV_SHA256,
        "generation_manifests": review_manifest["generation_manifests"],
        "training_provenance": review_manifest["training_provenance"],
    }
    args.output_dir.mkdir(parents=True)
    write_csv(
        args.output_dir / "unblinded_scores.csv",
        sorted(unblinded, key=lambda row: (int(row["sample_index"]), str(row["method"]))),
    )
    write_csv(args.output_dir / "summary.csv", summaries)
    gate_path = args.output_dir / "gate.json"
    gate_path.write_text(json.dumps(gate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(gate, indent=2, ensure_ascii=False))
    print(f"gate SHA-256={file_sha256(gate_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
