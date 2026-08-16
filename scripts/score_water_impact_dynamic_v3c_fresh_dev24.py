#!/usr/bin/env python3
"""Merge blind reviews, unblind, and apply the frozen v3c fresh-dev24 gate."""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import water_impact_dynamic_v3c_eval_protocol as protocol
from build_water_impact_dynamic_v3c_blind_review import review_binding_sha256


SHORT_FIELDS = {
    "target": protocol.SCORE_FIELDS[0],
    "footprint": protocol.SCORE_FIELDS[1],
    "receiver": protocol.SCORE_FIELDS[2],
    "quality": protocol.SCORE_FIELDS[3],
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def score_value(row: dict[str, Any], field: str, label: str) -> int:
    try:
        value = int(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{label}: invalid score for {field}") from exc
    if value not in {0, 1, 2}:
        raise ValueError(f"{label}: {field} must be 0, 1, or 2")
    return value


def is_usable(row: dict[str, Any]) -> bool:
    return int(row[protocol.SCORE_FIELDS[2]]) >= 1 and int(row[protocol.SCORE_FIELDS[3]]) >= 1


def is_strict(row: dict[str, Any]) -> bool:
    return tuple(int(row[field]) for field in protocol.SCORE_FIELDS) == (0, 0, 2, 2)


def merge_blind_reviews(
    template_rows: list[dict[str, str]],
    reviewer_a: list[dict[str, str]],
    reviewer_b: list[dict[str, str]],
    adjudicator: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Apply exact agreement, otherwise majority-of-three/0-1-2 median semantics."""

    if len(template_rows) != 48 or len(reviewer_a) != 48 or len(reviewer_b) != 48:
        raise ValueError("template and both independent reviews must each contain 48 rows")
    template_by_id = {row["review_id"]: row for row in template_rows}
    a_by_id = {row["review_id"]: row for row in reviewer_a}
    b_by_id = {row["review_id"]: row for row in reviewer_b}
    if any(len(mapping) != 48 for mapping in (template_by_id, a_by_id, b_by_id)):
        raise ValueError("duplicate review_id in blind review inputs")
    if set(template_by_id) != set(a_by_id) or set(template_by_id) != set(b_by_id):
        raise ValueError("blind reviewer IDs do not match the frozen template")
    expected_columns = set(template_rows[0])
    if any(set(row) != expected_columns for row in reviewer_a + reviewer_b):
        raise ValueError("reviewer sheets must retain the exact blind template columns")
    metadata_fields = (
        "sample_index",
        "pair_id",
        "generalization_group",
        "candidate_code",
        "composite_path",
        "candidate_video_path",
        "source_object",
        "receiver",
    )
    disputes: set[tuple[str, str]] = set()
    for review_id, template in template_by_id.items():
        for reviewer in (a_by_id[review_id], b_by_id[review_id]):
            for field in metadata_fields:
                if reviewer.get(field) != template.get(field):
                    raise ValueError(f"{review_id}: reviewer metadata changed for {field}")
        for short, field in SHORT_FIELDS.items():
            a_value = score_value(a_by_id[review_id], field, f"reviewer A/{review_id}")
            b_value = score_value(b_by_id[review_id], field, f"reviewer B/{review_id}")
            if a_value != b_value:
                disputes.add((review_id, short))

    adjudication: dict[tuple[str, str], dict[str, str]] = {}
    for row in adjudicator:
        if set(row) != {"review_id", "field", "score", "brief_reason"}:
            raise ValueError("adjudicator sheet must contain only the frozen blind columns")
        key = (row.get("review_id", ""), row.get("field", ""))
        if key in adjudication:
            raise ValueError(f"duplicate adjudication: {key}")
        if key not in disputes:
            raise ValueError(f"adjudication is not an exact disputed atomic field: {key}")
        score_value(row, "score", f"adjudicator/{key[0]}/{key[1]}")
        adjudication[key] = row
    if set(adjudication) != disputes:
        missing = sorted(disputes - set(adjudication))
        raise ValueError(f"every atomic disagreement requires blinded adjudication: {missing}")

    canonical: list[dict[str, Any]] = []
    for template in template_rows:
        review_id = template["review_id"]
        output: dict[str, Any] = {
            field: template[field]
            for field in ("review_id", *metadata_fields)
        }
        adjudicated_fields: list[str] = []
        for short, field in SHORT_FIELDS.items():
            a_value = score_value(a_by_id[review_id], field, f"reviewer A/{review_id}")
            b_value = score_value(b_by_id[review_id], field, f"reviewer B/{review_id}")
            if a_value == b_value:
                output[field] = a_value
            else:
                third = score_value(
                    adjudication[(review_id, short)],
                    "score",
                    f"adjudicator/{review_id}/{short}",
                )
                output[field] = int(statistics.median((a_value, b_value, third)))
                adjudicated_fields.append(short)
        output["notes"] = (
            "two_reviewer_agreement"
            if not adjudicated_fields
            else f"adjudicated:{','.join(adjudicated_fields)}"
        )
        canonical.append(output)
    dispute_rows = [
        {
            "review_id": review_id,
            "field": short,
            "reviewer_a": score_value(a_by_id[review_id], SHORT_FIELDS[short], "reviewer A"),
            "reviewer_b": score_value(b_by_id[review_id], SHORT_FIELDS[short], "reviewer B"),
            "adjudicator": score_value(adjudication[(review_id, short)], "score", "adjudicator"),
            "canonical": next(
                int(row[SHORT_FIELDS[short]]) for row in canonical if row["review_id"] == review_id
            ),
        }
        for review_id, short in sorted(disputes)
    ]
    return canonical, dispute_rows


def method_summary(method: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "method": method,
        "n": len(rows),
        "usable_n": sum(is_usable(row) for row in rows),
        "receiver_points": sum(int(row[protocol.SCORE_FIELDS[2]]) for row in rows),
        "quality_points": sum(int(row[protocol.SCORE_FIELDS[3]]) for row in rows),
        "target_suppression_points": sum(2 - int(row[protocol.SCORE_FIELDS[0]]) for row in rows),
        "footprint_suppression_points": sum(
            2 - int(row[protocol.SCORE_FIELDS[1]]) for row in rows
        ),
        "usable_absent_target_n": sum(
            is_usable(row) and int(row[protocol.SCORE_FIELDS[0]]) == 0 for row in rows
        ),
        "strict_success_n": sum(is_strict(row) for row in rows),
    }


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
    if set(by_method) != set(protocol.METHODS) or any(
        len(by_method[method]) != 24 for method in protocol.METHODS
    ):
        raise ValueError("fresh-dev scoring requires exactly 24 rows per method")
    if set(by_sample) != set(range(24)) or any(
        set(pair) != set(protocol.METHODS) for pair in by_sample.values()
    ):
        raise ValueError("fresh-dev scoring requires paired v3b/v3c samples 0 through 23")
    for rows in by_method.values():
        rows.sort(key=lambda row: int(row["sample_index"]))
    summaries = [method_summary(method, by_method[method]) for method in protocol.METHODS]
    summary = {str(row["method"]): row for row in summaries}
    control = by_method["v3b"]
    treatment_by_sample = {int(row["sample_index"]): row for row in by_method["v3c"]}
    control_usable = [row for row in control if is_usable(row)]

    paired_improvements: list[dict[str, Any]] = []
    clear_to_absent: list[dict[str, Any]] = []
    for control_row in control_usable:
        sample_index = int(control_row["sample_index"])
        treatment = treatment_by_sample[sample_index]
        control_target = int(control_row[protocol.SCORE_FIELDS[0]])
        treatment_target = int(treatment[protocol.SCORE_FIELDS[0]])
        if is_usable(treatment) and treatment_target < control_target:
            record = {
                "sample_index": sample_index,
                "generalization_group": treatment["generalization_group"],
                "v3b_target_visibility": control_target,
                "v3c_target_visibility": treatment_target,
            }
            paired_improvements.append(record)
            if control_target == 2 and treatment_target == 0:
                clear_to_absent.append(record)

    def valid_points(field: str, method: str) -> int:
        rows_by_sample = {int(row["sample_index"]): row for row in by_method[method]}
        return sum(
            2 - int(rows_by_sample[int(control_row["sample_index"])][field])
            if is_usable(rows_by_sample[int(control_row["sample_index"])])
            else 0
            for control_row in control_usable
        )

    control_target = valid_points(protocol.SCORE_FIELDS[0], "v3b")
    treatment_target = valid_points(protocol.SCORE_FIELDS[0], "v3c")
    control_footprint = valid_points(protocol.SCORE_FIELDS[1], "v3b")
    treatment_footprint = valid_points(protocol.SCORE_FIELDS[1], "v3c")
    v3b_summary = summary["v3b"]
    v3c_summary = summary["v3c"]
    checks = {
        "control_usable_at_least_20": int(v3b_summary["usable_n"]) >= 20,
        "target_suppression_margin_at_least_6_on_C": treatment_target >= control_target + 6,
        "paired_target_improvements_at_least_6": len(paired_improvements) >= 6,
        "clear_to_absent_at_least_2": len(clear_to_absent) >= 2,
        "clear_to_absent_groups_at_least_2": len(
            {str(row["generalization_group"]) for row in clear_to_absent}
        )
        >= 2,
        "usable_absent_target_margin_at_least_2": int(v3c_summary["usable_absent_target_n"])
        >= int(v3b_summary["usable_absent_target_n"]) + 2,
        "v3c_usable_at_least_22": int(v3c_summary["usable_n"]) >= 22,
        "receiver_floor_max_38_control_minus_2": int(v3c_summary["receiver_points"])
        >= max(38, int(v3b_summary["receiver_points"]) - 2),
        "quality_floor_max_32_control_minus_2": int(v3c_summary["quality_points"])
        >= max(32, int(v3b_summary["quality_points"]) - 2),
        "footprint_nonworse_on_C": treatment_footprint >= control_footprint,
        "strict_success_at_least_2": int(v3c_summary["strict_success_n"]) >= 2,
    }
    gate = {
        "protocol": protocol.EVAL_PROTOCOL,
        "gate_spec": protocol.GATE_SPEC,
        "control_usable_set": [int(row["sample_index"]) for row in control_usable],
        "control_usable_n": len(control_usable),
        "v3b_target_suppression_points_on_C": control_target,
        "v3c_target_suppression_points_on_C": treatment_target,
        "v3b_footprint_suppression_points_on_C": control_footprint,
        "v3c_footprint_suppression_points_on_C": treatment_footprint,
        "paired_target_improvements": paired_improvements,
        "clear_to_absent_improvements": clear_to_absent,
        "checks": checks,
        "promote_v3c_and_unseal_final36": all(checks.values()),
    }
    return summaries, gate


def validate_package_inventory(
    project_root: Path, review_path: Path, answer_key_path: Path, manifest_path: Path
) -> tuple[Path, Path, dict[str, Any]]:
    public_dir = review_path.parent
    private_dir = answer_key_path.parent
    if private_dir != manifest_path.parent or public_dir == private_dir:
        raise ValueError("public and private review packages must be distinct")
    if public_dir.parent.resolve() != private_dir.parent.resolve():
        raise ValueError("public/private packages must be sibling directories")
    expected_public = protocol.resolve_path(project_root, protocol.PUBLIC_REVIEW_DIR)
    expected_private = protocol.resolve_path(project_root, protocol.PRIVATE_REVIEW_DIR)
    if public_dir.resolve() != expected_public.resolve() or private_dir.resolve() != expected_private.resolve():
        raise ValueError("review package is outside the frozen public/private paths")
    if any(path.is_symlink() for path in (public_dir, private_dir, review_path, answer_key_path, manifest_path)):
        raise ValueError("review package paths must not be symlinks")
    if {path.name for path in public_dir.iterdir()} != {"blind_review.csv", "composites", "media"}:
        raise ValueError("public package contains an unexpected entry")
    if {path.name for path in private_dir.iterdir()} != {"answer_key.csv", "review_manifest.json"}:
        raise ValueError("private package must contain only key and manifest")
    composites = list((public_dir / "composites").iterdir())
    media = list((public_dir / "media").iterdir())
    expected_composites = {f"r{index:03d}.jpg" for index in range(24)}
    expected_media = {
        f"r{index:03d}_{code}.mp4" for index in range(24) for code in ("A", "B")
    }
    if {path.name for path in composites} != expected_composites or {path.name for path in media} != expected_media:
        raise ValueError("public package must contain exactly 24 composites and 48 videos")
    if any(path.is_symlink() or not path.is_file() for path in composites + media):
        raise ValueError("public media must be real files")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol") != protocol.EVAL_PROTOCOL:
        raise ValueError("review manifest protocol mismatch")
    if manifest.get("answer_key_sha256") != protocol.file_sha256(answer_key_path):
        raise ValueError("answer-key hash mismatch")
    return public_dir, private_dir, manifest


def validate_review_artifacts(
    project_root: Path,
    public_dir: Path,
    manifest: dict[str, Any],
    template: list[dict[str, str]],
    key_rows: list[dict[str, str]],
    stage2_path: Path,
    stage2: dict[str, Any],
) -> None:
    eval_rows = protocol.read_csv(protocol.resolve_path(project_root, protocol.FRESH_DEV_CSV))
    registered_fresh_hash = protocol.validate_split_registration(project_root)["registered_files"][
        protocol.FRESH_DEV_CSV
    ]["sha256"]
    expected_metadata = {
        "protocol": protocol.EVAL_PROTOCOL,
        "split_registry": {"path": protocol.SPLIT_REGISTRY, "sha256": protocol.SPLIT_REGISTRY_SHA256},
        "fresh_dev_csv": protocol.FRESH_DEV_CSV,
        "fresh_dev_csv_sha256": registered_fresh_hash,
        "blind_seed": protocol.BLIND_SEED,
        "frame_indices": list(protocol.FRAME_INDICES),
        "sample_count": 24,
        "review_rows": 48,
        "methods": list(protocol.METHODS),
    }
    for field, expected in expected_metadata.items():
        if manifest.get(field) != expected:
            raise ValueError(f"review manifest {field} does not match frozen protocol")
    if len(template) != 48 or len(key_rows) != 48:
        raise ValueError("review template and answer key must each contain 48 rows")
    if any(field in template[0] for field in ("method", "video_path", "source_video_path")):
        raise ValueError("public review template leaks a method or source path")
    if any(row.get(field, "") != "" for row in template for field in protocol.SCORE_FIELDS):
        raise ValueError("frozen public review template must be blank before independent review")
    template_by_id = {row["review_id"]: row for row in template}
    key_by_id = {row["review_id"]: row for row in key_rows}
    if len(template_by_id) != 48 or len(key_by_id) != 48 or set(template_by_id) != set(key_by_id):
        raise ValueError("review template and key do not form the exact 48-ID set")

    expected_ids: set[str] = set()
    sample_order = list(range(24))
    random.Random(protocol.BLIND_SEED).shuffle(sample_order)
    for position, sample_index in enumerate(sample_order):
        sample = eval_rows[sample_index]
        methods = list(protocol.METHODS)
        random.Random(f"{protocol.BLIND_SEED}:{sample['pair_id']}").shuffle(methods)
        expected_composite = (public_dir / "composites" / f"r{position:03d}.jpg").resolve(strict=True)
        for candidate_index, method in enumerate(methods):
            code = chr(ord("A") + candidate_index)
            review_id = f"r{position:03d}_{code}"
            expected_ids.add(review_id)
            review = template_by_id.get(review_id)
            key = key_by_id.get(review_id)
            if review is None or key is None:
                raise ValueError("blind assignment is incomplete")
            expected_fields = {
                "sample_index": str(sample_index),
                "pair_id": sample["pair_id"],
                "generalization_group": sample["generalization_group"],
                "candidate_code": code,
            }
            for field, expected in expected_fields.items():
                if review.get(field) != expected or key.get(field) != expected:
                    raise ValueError(f"{review_id}: frozen blind assignment mismatch for {field}")
            if review.get("source_object") != sample["source_object"] or review.get("receiver") != sample["receiver"]:
                raise ValueError(f"{review_id}: frozen semantic annotation mismatch")
            if key.get("method") != method:
                raise ValueError(f"{review_id}: frozen method assignment mismatch")
            actual_composite = protocol.resolve_path(project_root, review["composite_path"])
            actual_media = protocol.resolve_path(project_root, review["candidate_video_path"])
            if actual_composite.resolve(strict=True) != expected_composite:
                raise ValueError(f"{review_id}: composite path mismatch")
            if actual_media.resolve(strict=True) != (public_dir / "media" / f"{review_id}.mp4").resolve(strict=True):
                raise ValueError(f"{review_id}: anonymous media path mismatch")
    if set(template_by_id) != expected_ids:
        raise ValueError("review package contains unexpected blinded IDs")

    recorded_composites = manifest.get("composite_sha256")
    recorded_anonymous = manifest.get("anonymous_media_sha256")
    expected_composite_names = {f"r{index:03d}.jpg" for index in range(24)}
    expected_media_names = {
        f"r{index:03d}_{code}.mp4" for index in range(24) for code in ("A", "B")
    }
    if not isinstance(recorded_composites, dict) or set(recorded_composites) != expected_composite_names:
        raise ValueError("review manifest must bind exactly 24 composites")
    if not isinstance(recorded_anonymous, dict) or set(recorded_anonymous) != expected_media_names:
        raise ValueError("review manifest must bind exactly 48 anonymous videos")
    for name, expected_hash in recorded_composites.items():
        path = public_dir / "composites" / name
        if protocol.file_sha256(path) != expected_hash:
            raise ValueError(f"composite hash mismatch: {name}")

    current_manifests: dict[str, Path] = {}
    current_videos: dict[str, dict[int, Path]] = {}
    model_inventory = stage2["generation_spec"]["model_artifact_inventory"]
    for label, run in (
        ("original", protocol.ORIGINAL_RUN),
        ("v3b", protocol.V3B_RUN),
        ("v3c", protocol.V3C_RUN),
    ):
        current_manifests[label], _, current_videos[label] = protocol.load_generation_run(
            project_root,
            run,
            label,
            eval_rows,
            stage2_path,
            stage2,
            model_inventory,
        )
    recorded_manifests = manifest.get("generation_manifests")
    recorded_videos = manifest.get("video_sha256")
    if not isinstance(recorded_manifests, dict) or set(recorded_manifests) != set(current_manifests):
        raise ValueError("review manifest generation provenance is incomplete")
    if not isinstance(recorded_videos, dict) or set(recorded_videos) != set(current_videos):
        raise ValueError("review manifest video provenance is incomplete")
    for label, manifest_path in current_manifests.items():
        record = recorded_manifests[label]
        if (
            not isinstance(record, dict)
            or protocol.resolve_path(project_root, str(record.get("path", ""))).resolve(strict=True)
            != manifest_path.resolve(strict=True)
            or record.get("sha256") != protocol.file_sha256(manifest_path)
        ):
            raise ValueError(f"generation manifest provenance mismatch: {label}")
        arm_records = recorded_videos[label]
        if not isinstance(arm_records, dict) or set(arm_records) != {str(index) for index in range(24)}:
            raise ValueError(f"{label}: video provenance must bind exactly 24 indices")
        for index, source in current_videos[label].items():
            item = arm_records[str(index)]
            if (
                not isinstance(item, dict)
                or protocol.resolve_path(project_root, str(item.get("path", ""))).resolve(strict=True)
                != source.resolve(strict=True)
                or item.get("sha256") != protocol.file_sha256(source)
            ):
                raise ValueError(f"{label}: source-video provenance mismatch at {index}")

    for name, record in recorded_anonymous.items():
        review_id = name.removesuffix(".mp4")
        anonymous = public_dir / "media" / name
        key = key_by_id[review_id]
        source = current_videos[key["method"]][int(key["sample_index"])]
        keyed_source = protocol.resolve_path(project_root, key["video_path"])
        if (
            not isinstance(record, dict)
            or protocol.resolve_path(project_root, str(record.get("path", ""))).resolve(strict=True)
            != anonymous.resolve(strict=True)
            or record.get("sha256") != protocol.file_sha256(anonymous)
            or protocol.file_sha256(anonymous) != protocol.file_sha256(source)
            or anonymous.samefile(source)
            or keyed_source.resolve(strict=True) != source.resolve(strict=True)
        ):
            raise ValueError(f"anonymous-media provenance mismatch: {name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-template", type=Path, required=True)
    parser.add_argument("--reviewer-a", type=Path, required=True)
    parser.add_argument("--reviewer-b", type=Path, required=True)
    parser.add_argument("--adjudicator", type=Path, required=True)
    parser.add_argument("--answer-key", type=Path, required=True)
    parser.add_argument("--review-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error(f"refusing to overwrite score directory: {args.output_dir}")
    project_root = Path.cwd()
    protocol.validate_split_registration(project_root)
    stage2_path, stage2 = protocol.load_stage2_registration(project_root)
    _, _, manifest = validate_package_inventory(
        project_root, args.review_template, args.answer_key, args.review_manifest
    )
    if manifest.get("split_registry") != {
        "path": protocol.SPLIT_REGISTRY,
        "sha256": protocol.SPLIT_REGISTRY_SHA256,
    }:
        raise ValueError("review manifest split binding mismatch")
    recorded_stage2 = manifest.get("stage2_registration")
    if not isinstance(recorded_stage2, dict) or recorded_stage2.get("sha256") != protocol.file_sha256(stage2_path):
        raise ValueError("review manifest stage-2 binding mismatch")
    if protocol.resolve_path(project_root, str(recorded_stage2.get("path", ""))).resolve(strict=True) != stage2_path.resolve(strict=True):
        raise ValueError("review manifest stage-2 path mismatch")
    if recorded_stage2.get("payload") != stage2:
        raise ValueError("review manifest stage-2 payload changed")
    template = read_csv(args.review_template)
    if manifest.get("review_binding_sha256") != review_binding_sha256(template):
        raise ValueError("review template binding mismatch")
    key_rows = read_csv(args.answer_key)
    validate_review_artifacts(
        project_root, args.review_template.parent, manifest, template, key_rows, stage2_path, stage2
    )
    canonical, disputes = merge_blind_reviews(
        template, read_csv(args.reviewer_a), read_csv(args.reviewer_b), read_csv(args.adjudicator)
    )
    key_by_id = {row["review_id"]: row for row in key_rows}
    if len(key_by_id) != 48 or set(key_by_id) != {row["review_id"] for row in canonical}:
        raise ValueError("answer key does not match canonical review IDs")
    unblinded: list[dict[str, Any]] = []
    for row in canonical:
        key = key_by_id[row["review_id"]]
        for field in ("sample_index", "pair_id", "generalization_group", "candidate_code"):
            if str(row[field]) != str(key[field]):
                raise ValueError(f"{row['review_id']}: answer-key mismatch for {field}")
        if key["method"] not in protocol.METHODS:
            raise ValueError("answer key contains an unexpected method")
        output = {
            "review_id": row["review_id"],
            "sample_index": int(row["sample_index"]),
            "pair_id": row["pair_id"],
            "generalization_group": row["generalization_group"],
            "method": key["method"],
            "video_path": key["video_path"],
            **{field: int(row[field]) for field in protocol.SCORE_FIELDS},
            "usable": "yes" if is_usable(row) else "no",
            "strict_success": "yes" if is_strict(row) else "no",
            "notes": row["notes"],
        }
        unblinded.append(output)
    summaries, gate = compute_gate(unblinded)
    gate["input_provenance"] = {
        "review_template_sha256": protocol.file_sha256(args.review_template),
        "reviewer_a_sha256": protocol.file_sha256(args.reviewer_a),
        "reviewer_b_sha256": protocol.file_sha256(args.reviewer_b),
        "adjudicator_sha256": protocol.file_sha256(args.adjudicator),
        "answer_key_sha256": protocol.file_sha256(args.answer_key),
        "review_manifest_sha256": protocol.file_sha256(args.review_manifest),
        "stage2_registration_sha256": protocol.file_sha256(stage2_path),
        "split_registry_sha256": protocol.SPLIT_REGISTRY_SHA256,
    }
    args.output_dir.mkdir(parents=True)
    write_csv(args.output_dir / "canonical_blind_review.csv", canonical)
    if disputes:
        write_csv(args.output_dir / "adjudication_audit.csv", disputes)
    else:
        (args.output_dir / "adjudication_audit.csv").write_text(
            "review_id,field,reviewer_a,reviewer_b,adjudicator,canonical\n",
            encoding="utf-8",
        )
    write_csv(
        args.output_dir / "unblinded_scores.csv",
        sorted(unblinded, key=lambda row: (int(row["sample_index"]), str(row["method"]))),
    )
    write_csv(args.output_dir / "summary.csv", summaries)
    gate_path = args.output_dir / "gate.json"
    gate_path.write_text(json.dumps(gate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(gate, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
