#!/usr/bin/env python3
"""Build isolated O/A/B full-video blind-review packages for v4 evaluation."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from PIL import Image, ImageDraw

import water_impact_dynamic_v4_eval_protocol as protocol


FRAME_INDICES = (0, 8, 16, 24, 32, 40, 48)


def _hash_rank(salt: str, value: str) -> str:
    if not salt:
        raise ValueError("blind-assignment salt must be nonempty")
    return hashlib.sha256(f"{salt}\0{value}".encode("utf-8")).hexdigest()


def derive_causal_ab_assignment(
    unit_rows: Sequence[Mapping[str, Any]], private_salt: str
) -> dict[str, str]:
    """Return unit_id -> v4 arm with exact block and within-case balance."""

    protocol.validate_causal_unit_manifest(unit_rows)
    by_cell: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in unit_rows:
        by_cell[(str(row["group"]), str(row["prompt_variant"]))].append(row)
    assignment: dict[str, str] = {}
    for cell, rows in sorted(by_cell.items()):
        by_replicate: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            by_replicate[int(row["replicate"])].append(row)
        if set(by_replicate) != {0, 1, 2} or any(len(values) != 4 for values in by_replicate.values()):
            raise ValueError(f"causal assignment block is not 4x3: {cell}")
        candidates: list[dict[str, str]] = []
        for choices in itertools.product(
            *(list(itertools.combinations(values, 2)) for _, values in sorted(by_replicate.items()))
        ):
            v4_a = {str(row["unit_id"]) for pair in choices for row in pair}
            case_a = Counter(
                str(row["semantic_case_id"]) for row in rows if str(row["unit_id"]) in v4_a
            )
            if set(case_a.values()) != {1, 2} or Counter(case_a.values()) != {1: 2, 2: 2}:
                continue
            candidate = {
                str(row["unit_id"]): "A" if str(row["unit_id"]) in v4_a else "B" for row in rows
            }
            candidates.append(candidate)
        if not candidates:
            raise ValueError(f"no valid causal A/B block assignment: {cell}")
        canonical = lambda item: json.dumps(item, sort_keys=True, separators=(",", ":"))
        chosen = min(candidates, key=lambda item: _hash_rank(private_salt, f"{cell}:{canonical(item)}"))
        assignment.update(chosen)
    validate_blocked_assignment(unit_rows, "causal", assignment)
    return assignment


def derive_specificity_ab_assignment(
    unit_rows: Sequence[Mapping[str, Any]], private_salt: str
) -> dict[str, str]:
    """Swap v4 A/B across each specificity case's two replicates."""

    # Full semantic validation occurs before this function in the CLI.  The
    # assignment itself only needs the exact 18x2 case/replicate inventory.
    by_case: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in unit_rows:
        by_case[str(row["specificity_case_id"])].append(row)
    if len(unit_rows) != 36 or len(by_case) != 18:
        raise ValueError("specificity assignment requires exactly 18 cases x 2 replicates")
    output: dict[str, str] = {}
    for case_id, rows in by_case.items():
        rows = sorted(rows, key=lambda row: int(row["replicate"]))
        if [int(row["replicate"]) for row in rows] != [0, 1]:
            raise ValueError(f"{case_id}: specificity replicates must be exactly 0,1")
        first = "A" if int(_hash_rank(private_salt, case_id), 16) % 2 == 0 else "B"
        output[str(rows[0]["unit_id"])] = first
        output[str(rows[1]["unit_id"])] = "B" if first == "A" else "A"
    validate_blocked_assignment(unit_rows, "specificity", output)
    return output


def validate_blocked_assignment(
    unit_rows: Sequence[Mapping[str, Any]], dataset: str, assignment: Mapping[str, str]
) -> None:
    expected_ids = {str(row["unit_id"]) for row in unit_rows}
    if set(assignment) != expected_ids or set(assignment.values()) - {"A", "B"}:
        raise ValueError("A/B assignment does not cover the exact unit inventory")
    if dataset == "causal":
        block_counts: Counter[tuple[str, str, int, str]] = Counter()
        case_counts: Counter[tuple[str, str]] = Counter()
        for row in unit_rows:
            code = assignment[str(row["unit_id"])]
            block_counts[(str(row["group"]), str(row["prompt_variant"]), int(row["replicate"]), code)] += 1
            case_counts[(str(row["semantic_case_id"]), code)] += 1
        for group in protocol.CAUSAL_GROUPS:
            for variant in protocol.PROMPT_VARIANTS:
                for replicate in range(3):
                    if block_counts[(group, variant, replicate, "A")] != 2 or block_counts[(group, variant, replicate, "B")] != 2:
                        raise ValueError("causal A/B assignment is not balanced within every block")
        for case_id in {str(row["semantic_case_id"]) for row in unit_rows}:
            if sorted((case_counts[(case_id, "A")], case_counts[(case_id, "B")])) != [1, 2]:
                raise ValueError("each causal case must put v4 in A for one or two replicates")
    elif dataset == "specificity":
        by_case: dict[str, list[str]] = defaultdict(list)
        cell_counts: Counter[tuple[str, str, str]] = Counter()
        for row in unit_rows:
            code = assignment[str(row["unit_id"])]
            by_case[str(row["specificity_case_id"])].append(code)
            cell_counts[(str(row["membership"]), str(row["prompt_variant"]), code)] += 1
        if any(sorted(codes) != ["A", "B"] for codes in by_case.values()):
            raise ValueError("each specificity case must swap v4 between A and B")
        for membership in protocol.SPECIFICITY_MEMBERSHIPS:
            for variant in protocol.PROMPT_VARIANTS:
                if cell_counts[(membership, variant, "A")] != 3 or cell_counts[(membership, variant, "B")] != 3:
                    raise ValueError("specificity membership/variant cells must be A/B balanced")
    else:
        raise ValueError("unknown blind-review dataset")


def _load_frames(path: Path) -> list[Image.Image]:
    import av

    selected: dict[int, Image.Image] = {}
    with av.open(str(path)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index in FRAME_INDICES:
                selected[index] = frame.to_image().convert("RGB")
            if index >= FRAME_INDICES[-1]:
                break
    if set(selected) != set(FRAME_INDICES):
        raise ValueError(f"video does not contain all frozen composite frames: {path}")
    return [selected[index] for index in FRAME_INDICES]


def _strip(frames: Sequence[Image.Image], label: str) -> Image.Image:
    frame_width = 208
    frame_height = round(frames[0].height * frame_width / frames[0].width)
    label_width = 120
    header_height = 24
    strip = Image.new("RGB", (label_width + frame_width * len(frames), header_height + frame_height), "white")
    draw = ImageDraw.Draw(strip)
    draw.text((8, header_height + 10), label, fill="black")
    for column, (frame_index, frame) in enumerate(zip(FRAME_INDICES, frames)):
        x = label_width + column * frame_width
        draw.text((x + 5, 5), f"frame {frame_index}", fill="black")
        strip.paste(frame.resize((frame_width, frame_height)), (x, header_height))
    return strip


def build_composite(output_path: Path, paths: Mapping[str, Path]) -> None:
    strips = [_strip(_load_frames(paths[code]), {"O": "Reference", "A": "Candidate A", "B": "Candidate B"}[code]) for code in protocol.ARM_CODES]
    composite = Image.new("RGB", (strips[0].width, sum(item.height for item in strips)), "white")
    y = 0
    for strip in strips:
        composite.paste(strip, (0, y))
        y += strip.height
    output_path.parent.mkdir(parents=True, exist_ok=True)
    composite.save(output_path, quality=92)


def build_screening_composite(output_path: Path, video_path: Path) -> None:
    """Build the frozen seven-frame Original-only screening strip."""

    strip = _strip(_load_frames(video_path), "Original screening")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    strip.save(output_path, quality=92)


def review_binding_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    canonical = [
        {key: str(value) for key, value in row.items() if key not in {*protocol.CAUSAL_SCORE_FIELDS, *protocol.CAUSAL_REFERENCE_ONLY_FIELDS, *protocol.SPECIFICITY_SCORE_FIELDS, "notes"}}
        for row in rows
    ]
    return protocol.canonical_json_sha256(canonical)


def derive_review_order(unit_rows: Sequence[Mapping[str, Any]], private_salt: str) -> list[str]:
    unit_ids = [str(row.get("unit_id", "")) for row in unit_rows]
    if any(not unit_id for unit_id in unit_ids) or len(set(unit_ids)) != len(unit_ids):
        raise ValueError("review order requires unique nonempty unit IDs")
    return sorted(
        unit_ids,
        key=lambda unit_id: (_hash_rank(private_salt, f"order:{unit_id}"), unit_id),
    )


def _atomic_write_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp.{os.getpid()}"
    if path.exists() or path.is_symlink() or temporary.exists():
        raise FileExistsError(f"refusing to overwrite frozen review commitment: {path}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _file_ref(path: Path) -> dict[str, str]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"review commitment input is missing or symlinked: {path}")
    return {"path": str(path), "sha256": protocol.file_sha256(path)}


def build_review_package_commitment(
    project_root: Path,
    *,
    dataset: str,
    unit_rows: Sequence[Mapping[str, Any]],
    unit_manifest_path: Path,
    template_path: Path,
    answer_key_path: Path,
    review_manifest_path: Path,
    checkpoint_eligibility_path: Path,
    generation_manifest_paths: Mapping[str, Path],
    private_salt: str,
    composite_hashes: Mapping[str, str],
    anonymous_hashes: Mapping[str, str],
    output_path: Path,
) -> dict[str, Any]:
    """Freeze every pre-review byte while the private answer key remains unopened."""

    if dataset not in protocol.DATASETS or set(generation_manifest_paths) != set(
        protocol.METHODS
    ):
        raise ValueError("review package commitment inventory is not exact")
    expected_units = protocol.UNIT_COUNTS[dataset]
    if len(unit_rows) != expected_units:
        raise ValueError("review package commitment unit count differs from protocol")
    if protocol.read_csv(unit_manifest_path) != [
        {key: str(value) for key, value in row.items()} for row in unit_rows
    ]:
        raise ValueError("review package units differ from the bound unit-manifest bytes")
    eligibility = protocol.validate_checkpoint_eligibility(
        project_root, checkpoint_eligibility_path
    )
    model_sha256 = eligibility.get("model_content_inventory_sha256")
    runtime_sha256 = eligibility.get("runtime_registry_sha256")
    code_sha256 = eligibility.get("training_code_registry_sha256")
    if not all(protocol.is_sha256(value) for value in (model_sha256, runtime_sha256, code_sha256)):
        raise ValueError("checkpoint eligibility lacks model/runtime/code inventory binding")
    code_path = protocol.resolve_path(project_root, protocol.TRAINING_CODE_REGISTRY)
    if not code_path.is_file() or code_path.is_symlink() or protocol.file_sha256(code_path) != code_sha256:
        raise ValueError("review package training-code registry bytes differ from checkpoint")
    protocol.validate_training_code_registry(project_root, code_path)
    protocol.validate_runtime_registry(
        protocol.resolve_path(project_root, protocol.RUNTIME_REGISTRY), runtime_sha256
    )
    generation_refs: dict[str, dict[str, str]] = {}
    for method in protocol.METHODS:
        path = generation_manifest_paths[method]
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("dataset") != dataset
            or payload.get("method") != method
            or payload.get("model_inventory_sha256") != model_sha256
            or payload.get("runtime_registry_sha256") != runtime_sha256
        ):
            raise ValueError(f"{method}: generation manifest differs from checkpoint inventories")
        generation_refs[method] = _file_ref(path)
    assignment = (
        derive_causal_ab_assignment(unit_rows, private_salt)
        if dataset == "causal"
        else derive_specificity_ab_assignment(unit_rows, private_salt)
    )
    review_order = derive_review_order(unit_rows, private_salt)
    payload = {
        "protocol": protocol.FINAL_REVIEW_PACKAGE_COMMITMENT_PROTOCOL,
        "dataset": dataset,
        "dataset_version": protocol.DATASET_VERSION,
        "status": "committed_before_blind_review",
        "assignment_salt_sha256": hashlib.sha256(
            private_salt.encode("utf-8")
        ).hexdigest(),
        "assignment_sha256": protocol.canonical_json_sha256(assignment),
        "review_order_sha256": protocol.canonical_json_sha256(review_order),
        "unit_manifest": {
            **_file_ref(unit_manifest_path),
            "canonical_sha256": protocol.canonical_json_sha256(
                [{key: str(value) for key, value in row.items()} for row in unit_rows]
            ),
            "row_count": expected_units,
        },
        "public_template": {
            **_file_ref(template_path),
            "row_count": 3 * expected_units,
        },
        "unopened_answer_key": {
            **_file_ref(answer_key_path),
            "row_count": 3 * expected_units,
        },
        "review_manifest": _file_ref(review_manifest_path),
        "checkpoint_eligibility": _file_ref(checkpoint_eligibility_path),
        "model_inventory_sha256": model_sha256,
        "runtime_registry_sha256": runtime_sha256,
        "training_code_registry": {
            "path": protocol.TRAINING_CODE_REGISTRY,
            "sha256": code_sha256,
        },
        "generation_manifests": generation_refs,
        "composite_sha256": dict(composite_hashes),
        "anonymous_media_sha256": dict(anonymous_hashes),
    }
    _atomic_write_new_json(output_path, payload)
    return payload


def build_review_package(
    *,
    project_root: Path,
    dataset: str,
    unit_rows: Sequence[Mapping[str, Any]],
    unit_manifest_path: Path,
    videos: Mapping[str, Mapping[str, Path]],
    generation_manifest_paths: Mapping[str, Path],
    checkpoint_eligibility_path: Path,
    public_dir: Path,
    private_dir: Path,
    package_commitment_path: Path,
    private_salt: str,
    composite_builder: Callable[[Path, Mapping[str, Path]], None] = build_composite,
) -> dict[str, Any]:
    if dataset not in protocol.DATASETS:
        raise ValueError("unsupported review dataset")
    if public_dir.exists() or private_dir.exists():
        raise FileExistsError("refusing to overwrite public/private review package")
    if public_dir.resolve() == private_dir.resolve() or public_dir.parent.resolve() != private_dir.parent.resolve():
        raise ValueError("public/private review packages must be distinct sibling directories")
    expected_ids = {str(row["unit_id"]) for row in unit_rows}
    if set(videos) != set(protocol.METHODS) or any(set(arm) != expected_ids for arm in videos.values()):
        raise ValueError("review videos do not cover exact O/v3b/v4 unit inventories")
    assignment = (
        derive_causal_ab_assignment(unit_rows, private_salt)
        if dataset == "causal"
        else derive_specificity_ab_assignment(unit_rows, private_salt)
    )
    row_by_id = {str(row["unit_id"]): row for row in unit_rows}
    review_order = derive_review_order(unit_rows, private_salt)
    public_dir.mkdir(parents=True)
    private_dir.mkdir(parents=True)
    public_rows: list[dict[str, Any]] = []
    key_rows: list[dict[str, Any]] = []
    composite_hashes: dict[str, str] = {}
    anonymous_hashes: dict[str, str] = {}
    for position, unit_id in enumerate(review_order):
        unit = row_by_id[unit_id]
        v4_code = assignment[unit_id]
        method_by_code = {
            "O": "original",
            v4_code: "v4",
            "B" if v4_code == "A" else "A": "v3b",
        }
        copied: dict[str, Path] = {}
        for code in protocol.ARM_CODES:
            method = method_by_code[code]
            source = videos[method][unit_id]
            anonymous = public_dir / "media" / f"r{position:03d}_{code}.mp4"
            anonymous.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, anonymous)
            if anonymous.is_symlink() or anonymous.samefile(source):
                raise ValueError("anonymous media must be an independent real-file copy")
            source_hash = protocol.file_sha256(source)
            anonymous_hash = protocol.file_sha256(anonymous)
            if source_hash != anonymous_hash:
                raise ValueError("anonymous copy differs from source video")
            copied[code] = anonymous
            anonymous_hashes[anonymous.name] = anonymous_hash
            review_id = f"r{position:03d}_{code}"
            score_fields = (
                (*protocol.CAUSAL_SCORE_FIELDS, *protocol.CAUSAL_REFERENCE_ONLY_FIELDS)
                if dataset == "causal"
                else protocol.SPECIFICITY_SCORE_FIELDS
            )
            public_rows.append(
                {
                    "review_id": review_id,
                    "anonymous_unit": f"r{position:03d}",
                    "arm_code": code,
                    "object_phrase": str(unit["source_phrase"]),
                    "receiver_description": str(unit["receiver"]),
                    "composite_path": str(public_dir / "composites" / f"r{position:03d}.jpg"),
                    "video_path": str(anonymous),
                    **{field: "" for field in score_fields},
                    "notes": "",
                }
            )
            key_rows.append(
                {
                    "review_id": review_id,
                    "unit_id": unit_id,
                    "case_id": str(unit["semantic_case_id"] if dataset == "causal" else unit["specificity_case_id"]),
                    "group": str(unit.get("group", "")),
                    "membership": str(unit.get("membership", "")),
                    "prompt_variant": str(unit["prompt_variant"]),
                    "replicate": int(unit["replicate"]),
                    "seed": int(unit["seed"]),
                    "arm_code": code,
                    "method": method,
                    "source_id": str(unit["source_id"]),
                    "source_phrase": str(unit["source_phrase"]),
                    "receiver_id": str(unit["receiver_id"]),
                    "receiver": str(unit["receiver"]),
                    "source_video_path": str(source),
                    "source_video_sha256": source_hash,
                    "anonymous_video_path": str(anonymous),
                    "anonymous_video_sha256": anonymous_hash,
                }
            )
        composite = public_dir / "composites" / f"r{position:03d}.jpg"
        composite_builder(composite, copied)
        if not composite.is_file() or composite.is_symlink() or composite.stat().st_size <= 0:
            raise ValueError("composite builder did not create a real nonempty file")
        composite_hashes[composite.name] = protocol.file_sha256(composite)
    protocol.validate_public_review_columns(public_rows)
    template_path = public_dir / "blind_review_v2.csv"
    answer_key_path = private_dir / "answer_key_v2.csv"
    protocol.write_csv(template_path, public_rows)
    protocol.write_csv(answer_key_path, key_rows)
    assignment_digest = protocol.canonical_json_sha256(assignment)
    manifest = {
        "protocol": protocol.BLIND_REVIEW_PROTOCOL,
        "dataset": dataset,
        "dataset_version": protocol.DATASET_VERSION,
        "unit_manifest_canonical_sha256": protocol.canonical_json_sha256(
            [{key: str(value) for key, value in row.items()} for row in unit_rows]
        ),
        "unit_count": protocol.UNIT_COUNTS[dataset],
        "review_row_count": 3 * protocol.UNIT_COUNTS[dataset],
        "assignment_algorithm": "sha256_blocked_exact_balance_v1",
        "assignment_salt_sha256": hashlib.sha256(private_salt.encode("utf-8")).hexdigest(),
        "assignment_sha256": assignment_digest,
        "review_order_algorithm": "sha256_salted_unit_id_v1",
        "review_binding_sha256": review_binding_sha256(public_rows),
        "public_template_sha256": protocol.file_sha256(template_path),
        "answer_key_sha256": protocol.file_sha256(answer_key_path),
        "checkpoint_eligibility": {
            "path": str(checkpoint_eligibility_path),
            "sha256": protocol.file_sha256(checkpoint_eligibility_path),
        },
        "generation_manifests": {
            method: {"path": str(path), "sha256": protocol.file_sha256(path)}
            for method, path in generation_manifest_paths.items()
        },
        "composite_sha256": composite_hashes,
        "anonymous_media_sha256": anonymous_hashes,
    }
    review_manifest_path = private_dir / "review_manifest_v2.json"
    review_manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    build_review_package_commitment(
        project_root,
        dataset=dataset,
        unit_rows=unit_rows,
        unit_manifest_path=unit_manifest_path,
        template_path=template_path,
        answer_key_path=answer_key_path,
        review_manifest_path=review_manifest_path,
        checkpoint_eligibility_path=checkpoint_eligibility_path,
        generation_manifest_paths=generation_manifest_paths,
        private_salt=private_salt,
        composite_hashes=composite_hashes,
        anonymous_hashes=anonymous_hashes,
        output_path=package_commitment_path,
    )
    return manifest


def validate_review_package_commitment(
    project_root: Path,
    *,
    dataset: str,
    commitment_path: Path,
    template_path: Path | None = None,
    answer_key_path: Path | None = None,
    review_manifest_path: Path | None = None,
    unit_manifest_path: Path | None = None,
    checkpoint_eligibility_path: Path | None = None,
) -> dict[str, Any]:
    """Validate the immutable package root without interpreting the answer key."""

    if not commitment_path.is_file() or commitment_path.is_symlink():
        raise FileNotFoundError("pre-review package commitment is missing or symlinked")
    payload = json.loads(commitment_path.read_text(encoding="utf-8"))
    expected_fields = {
        "protocol",
        "dataset",
        "dataset_version",
        "status",
        "assignment_salt_sha256",
        "assignment_sha256",
        "review_order_sha256",
        "unit_manifest",
        "public_template",
        "unopened_answer_key",
        "review_manifest",
        "checkpoint_eligibility",
        "model_inventory_sha256",
        "runtime_registry_sha256",
        "training_code_registry",
        "generation_manifests",
        "composite_sha256",
        "anonymous_media_sha256",
    }
    if not isinstance(payload, dict) or set(payload) != expected_fields:
        raise ValueError("pre-review package commitment fields are not exact")
    if (
        payload["protocol"] != protocol.FINAL_REVIEW_PACKAGE_COMMITMENT_PROTOCOL
        or payload["dataset"] != dataset
        or payload["dataset_version"] != protocol.DATASET_VERSION
        or payload["status"] != "committed_before_blind_review"
    ):
        raise ValueError("pre-review package commitment identity/status mismatch")
    for name in (
        "assignment_salt_sha256",
        "assignment_sha256",
        "review_order_sha256",
        "model_inventory_sha256",
        "runtime_registry_sha256",
    ):
        if not protocol.is_sha256(payload[name]):
            raise ValueError(f"pre-review package commitment/{name} is invalid")

    expected_units = protocol.UNIT_COUNTS[dataset]

    def validate_ref(
        name: str,
        record: Any,
        *,
        extra_fields: set[str] = set(),
        expected_path: Path | None = None,
    ) -> Path:
        if not isinstance(record, dict) or set(record) != {"path", "sha256", *extra_fields}:
            raise ValueError(f"pre-review package commitment/{name} ref is not exact")
        if not protocol.is_sha256(record["sha256"]):
            raise ValueError(f"pre-review package commitment/{name} hash is invalid")
        resolved = protocol.resolve_path(project_root, str(record["path"]))
        if (
            not resolved.is_file()
            or resolved.is_symlink()
            or protocol.file_sha256(resolved) != record["sha256"]
        ):
            raise ValueError(f"pre-review package commitment/{name} byte hash mismatch")
        if expected_path is not None and resolved.resolve() != expected_path.resolve():
            raise ValueError(f"pre-review package commitment/{name} path mismatch")
        return resolved

    unit_path = validate_ref(
        "unit_manifest",
        payload["unit_manifest"],
        extra_fields={"canonical_sha256", "row_count"},
        expected_path=unit_manifest_path,
    )
    if (
        payload["unit_manifest"]["row_count"] != expected_units
        or not protocol.is_sha256(payload["unit_manifest"]["canonical_sha256"])
    ):
        raise ValueError("pre-review package commitment unit metadata mismatch")
    units = protocol.read_csv(unit_path)
    if (
        len(units) != expected_units
        or protocol.canonical_json_sha256(units)
        != payload["unit_manifest"]["canonical_sha256"]
    ):
        raise ValueError("pre-review package commitment unit inventory mismatch")
    public_path = validate_ref(
        "public_template",
        payload["public_template"],
        extra_fields={"row_count"},
        expected_path=template_path,
    )
    if payload["public_template"]["row_count"] != 3 * expected_units:
        raise ValueError("pre-review package commitment public row count mismatch")
    public_rows = protocol.read_csv(public_path)
    protocol.validate_public_review_columns(public_rows)
    if len(public_rows) != 3 * expected_units:
        raise ValueError("pre-review package commitment public inventory mismatch")
    key_path = validate_ref(
        "unopened_answer_key",
        payload["unopened_answer_key"],
        extra_fields={"row_count"},
        expected_path=answer_key_path,
    )
    if payload["unopened_answer_key"]["row_count"] != 3 * expected_units:
        raise ValueError("pre-review package commitment key row count mismatch")
    manifest_path = validate_ref(
        "review_manifest",
        payload["review_manifest"],
        expected_path=review_manifest_path,
    )
    eligibility_path = validate_ref(
        "checkpoint_eligibility",
        payload["checkpoint_eligibility"],
        expected_path=checkpoint_eligibility_path,
    )
    eligibility = protocol.validate_checkpoint_eligibility(project_root, eligibility_path)
    if (
        eligibility.get("model_content_inventory_sha256")
        != payload["model_inventory_sha256"]
        or eligibility.get("runtime_registry_sha256")
        != payload["runtime_registry_sha256"]
        or eligibility.get("training_code_registry_sha256")
        != payload["training_code_registry"].get("sha256")
    ):
        raise ValueError("pre-review package inventories differ from checkpoint eligibility")
    code_ref = payload["training_code_registry"]
    code_path = validate_ref("training_code_registry", code_ref)
    if (
        code_ref["path"] != protocol.TRAINING_CODE_REGISTRY
        or code_path.resolve()
        != protocol.resolve_path(project_root, protocol.TRAINING_CODE_REGISTRY).resolve()
    ):
        raise ValueError("pre-review package code-registry path differs from protocol")
    protocol.validate_training_code_registry(project_root, code_path)
    protocol.validate_runtime_registry(
        protocol.resolve_path(project_root, protocol.RUNTIME_REGISTRY),
        payload["runtime_registry_sha256"],
    )
    generation_refs = payload["generation_manifests"]
    if not isinstance(generation_refs, dict) or set(generation_refs) != set(protocol.METHODS):
        raise ValueError("pre-review package generation refs are not exact")
    for method, record in generation_refs.items():
        generation_path = validate_ref(f"generation_manifests/{method}", record)
        generation = json.loads(generation_path.read_text(encoding="utf-8"))
        if (
            generation.get("dataset") != dataset
            or generation.get("method") != method
            or generation.get("model_inventory_sha256")
            != payload["model_inventory_sha256"]
            or generation.get("runtime_registry_sha256")
            != payload["runtime_registry_sha256"]
        ):
            raise ValueError(f"{method}: pre-review generation inventory mismatch")
    public_dir = public_path.parent
    expected_composites = {f"r{index:03d}.jpg" for index in range(expected_units)}
    expected_media = {
        f"r{index:03d}_{code}.mp4"
        for index in range(expected_units)
        for code in protocol.ARM_CODES
    }
    for name, expected_names, directory_name in (
        ("composite_sha256", expected_composites, "composites"),
        ("anonymous_media_sha256", expected_media, "media"),
    ):
        inventory = payload[name]
        if (
            not isinstance(inventory, dict)
            or set(inventory) != expected_names
            or any(not protocol.is_sha256(value) for value in inventory.values())
        ):
            raise ValueError(f"pre-review package {name} inventory is not exact")
        directory = public_dir / directory_name
        actual = {item.name: item for item in directory.iterdir()}
        if set(actual) != expected_names or any(
            item.is_symlink()
            or not item.is_file()
            or protocol.file_sha256(item) != inventory[item.name]
            for item in actual.values()
        ):
            raise ValueError(f"pre-review package {name} bytes changed")
    del key_path, manifest_path
    return payload


def validate_review_package(
    project_root: Path,
    *,
    dataset: str,
    unit_rows: Sequence[Mapping[str, Any]],
    template_path: Path,
    answer_key_path: Path,
    review_manifest_path: Path,
    package_commitment_path: Path,
    assignment_salt: str,
    unit_manifest_path: Path,
    checkpoint_eligibility_path: Path,
    decode: Callable[[Path], Mapping[str, int]] = protocol._default_decode,
) -> dict[str, Any]:
    """Recompute public/private isolation, answer key, assignment, and media bindings."""

    commitment = validate_review_package_commitment(
        project_root,
        dataset=dataset,
        commitment_path=package_commitment_path,
        template_path=template_path,
        answer_key_path=answer_key_path,
        review_manifest_path=review_manifest_path,
        unit_manifest_path=unit_manifest_path,
        checkpoint_eligibility_path=checkpoint_eligibility_path,
    )
    if hashlib.sha256(assignment_salt.encode("utf-8")).hexdigest() != commitment[
        "assignment_salt_sha256"
    ]:
        raise ValueError("opened assignment salt differs from pre-review commitment")
    expected_units = protocol.UNIT_COUNTS[dataset]
    expected_rows = 3 * expected_units
    if len(unit_rows) != expected_units:
        raise ValueError("review package unit inventory count differs from protocol")
    unit_by_id = {str(row.get("unit_id", "")): row for row in unit_rows}
    if len(unit_by_id) != expected_units or "" in unit_by_id:
        raise ValueError("review package unit IDs are duplicate or blank")
    expected_unit_sha256 = protocol.canonical_json_sha256(
        [{key: str(value) for key, value in row.items()} for row in unit_rows]
    )
    public_dir = template_path.parent
    private_dir = answer_key_path.parent
    if review_manifest_path.parent != private_dir or public_dir.parent.resolve() != private_dir.parent.resolve() or public_dir.resolve() == private_dir.resolve():
        raise ValueError("public/private review packages must be distinct sibling directories")
    if any(path.is_symlink() for path in (public_dir, private_dir, template_path, answer_key_path, review_manifest_path)):
        raise ValueError("review package paths must not be symlinks")
    if {path.name for path in public_dir.iterdir()} != {"blind_review_v2.csv", "composites", "media"}:
        raise ValueError("public review package contains an unexpected entry")
    if {path.name for path in private_dir.iterdir()} != {"answer_key_v2.csv", "review_manifest_v2.json"}:
        raise ValueError("private review package must contain only key and manifest")
    expected_composites = {f"r{index:03d}.jpg" for index in range(expected_units)}
    expected_media = {
        f"r{index:03d}_{code}.mp4"
        for index in range(expected_units)
        for code in protocol.ARM_CODES
    }
    composites = {path.name: path for path in (public_dir / "composites").iterdir()}
    media = {path.name: path for path in (public_dir / "media").iterdir()}
    if set(composites) != expected_composites or set(media) != expected_media:
        raise ValueError("public package does not contain the exact composite/video inventory")
    if any(path.is_symlink() or not path.is_file() for path in [*composites.values(), *media.values()]):
        raise ValueError("anonymous review media must be real files")
    template = protocol.read_csv(template_path)
    key_rows = protocol.read_csv(answer_key_path)
    protocol.validate_public_review_columns(template)
    if len(template) != expected_rows or len(key_rows) != expected_rows:
        raise ValueError("review template/key row count mismatch")
    if any(
        str(row.get(field, "")) != ""
        for row in template
        for field in (
            (*protocol.CAUSAL_SCORE_FIELDS, *protocol.CAUSAL_REFERENCE_ONLY_FIELDS)
            if dataset == "causal"
            else protocol.SPECIFICITY_SCORE_FIELDS
        )
    ):
        raise ValueError("frozen public review template is not blank")
    template_by_id = {row["review_id"]: row for row in template}
    key_by_id = {row["review_id"]: row for row in key_rows}
    if len(template_by_id) != expected_rows or len(key_by_id) != expected_rows or set(template_by_id) != set(key_by_id):
        raise ValueError("review template and key IDs are not exact")
    expected_key_fields = {
        "review_id",
        "unit_id",
        "case_id",
        "group",
        "membership",
        "prompt_variant",
        "replicate",
        "seed",
        "arm_code",
        "method",
        "source_id",
        "source_phrase",
        "receiver_id",
        "receiver",
        "source_video_path",
        "source_video_sha256",
        "anonymous_video_path",
        "anonymous_video_sha256",
    }
    if any(set(row) != expected_key_fields for row in key_rows):
        raise ValueError("answer key columns are not exact")
    manifest = json.loads(review_manifest_path.read_text(encoding="utf-8"))
    expected_manifest_fields = {
        "protocol",
        "dataset",
        "dataset_version",
        "unit_manifest_canonical_sha256",
        "unit_count",
        "review_row_count",
        "assignment_algorithm",
        "assignment_salt_sha256",
        "assignment_sha256",
        "review_order_algorithm",
        "review_binding_sha256",
        "public_template_sha256",
        "answer_key_sha256",
        "checkpoint_eligibility",
        "generation_manifests",
        "composite_sha256",
        "anonymous_media_sha256",
    }
    if not isinstance(manifest, dict) or set(manifest) != expected_manifest_fields:
        raise ValueError("review manifest fields are not exact")
    if (
        manifest["protocol"] != protocol.BLIND_REVIEW_PROTOCOL
        or manifest["dataset"] != dataset
        or manifest["dataset_version"] != protocol.DATASET_VERSION
        or manifest["unit_manifest_canonical_sha256"] != expected_unit_sha256
        or manifest["unit_count"] != expected_units
        or manifest["review_row_count"] != expected_rows
        or manifest["assignment_algorithm"] != "sha256_blocked_exact_balance_v1"
        or manifest["review_order_algorithm"] != "sha256_salted_unit_id_v1"
        or manifest["review_binding_sha256"] != review_binding_sha256(template)
        or manifest["public_template_sha256"] != protocol.file_sha256(template_path)
        or manifest["answer_key_sha256"] != protocol.file_sha256(answer_key_path)
    ):
        raise ValueError("review manifest protocol/template/key binding mismatch")
    if not protocol.is_sha256(manifest["assignment_salt_sha256"]) or not protocol.is_sha256(manifest["assignment_sha256"]):
        raise ValueError("review assignment digests are invalid")
    if (
        manifest["assignment_salt_sha256"] != commitment["assignment_salt_sha256"]
        or manifest["assignment_sha256"] != commitment["assignment_sha256"]
    ):
        raise ValueError("review manifest assignment differs from pre-review commitment")
    if manifest["composite_sha256"] != {
        name: protocol.file_sha256(path) for name, path in composites.items()
    }:
        raise ValueError("review composite byte inventory mismatch")
    if manifest["anonymous_media_sha256"] != {
        name: protocol.file_sha256(path) for name, path in media.items()
    }:
        raise ValueError("anonymous review-video byte inventory mismatch")
    if (
        manifest["composite_sha256"] != commitment["composite_sha256"]
        or manifest["anonymous_media_sha256"]
        != commitment["anonymous_media_sha256"]
    ):
        raise ValueError("review media differs from pre-review package commitment")
    checkpoint_ref = manifest["checkpoint_eligibility"]
    if checkpoint_ref != {
        "path": str(checkpoint_eligibility_path),
        "sha256": protocol.file_sha256(checkpoint_eligibility_path),
    }:
        raise ValueError("review package checkpoint-eligibility binding mismatch")
    generation_refs = manifest["generation_manifests"]
    if not isinstance(generation_refs, dict) or set(generation_refs) != set(protocol.METHODS):
        raise ValueError("review package generation-manifest refs are incomplete")
    generation_videos: dict[str, dict[str, Mapping[str, Any]]] = {}
    model_inventory_hashes: set[str] = set()
    runtime_registry_hashes: set[str] = set()
    all_source_paths: set[Path] = set()
    all_source_inodes: set[tuple[int, int]] = set()
    all_source_hashes: set[str] = set()
    eligibility_payload = json.loads(checkpoint_eligibility_path.read_text(encoding="utf-8"))
    for method, ref in generation_refs.items():
        if not isinstance(ref, dict) or set(ref) != {"path", "sha256"}:
            raise ValueError("generation-manifest reference is not exact")
        path = protocol.resolve_path(project_root, str(ref["path"]))
        if not path.is_file() or path.is_symlink() or protocol.file_sha256(path) != ref["sha256"]:
            raise ValueError(f"{method}: generation-manifest byte hash mismatch")
        payload = json.loads(path.read_text(encoding="utf-8"))
        expected_generation_fields = {
            "protocol",
            "dataset",
            "dataset_version",
            "method",
            "unit_manifest_canonical_sha256",
            "generation_spec",
            "model_inventory_sha256",
            "runtime_registry_sha256",
            "method_artifact",
            "videos",
        }
        if (
            not isinstance(payload, dict)
            or set(payload) != expected_generation_fields
            or
            payload.get("protocol") != protocol.GENERATION_MANIFEST_PROTOCOL
            or payload.get("dataset") != dataset
            or payload.get("dataset_version") != protocol.DATASET_VERSION
            or payload.get("method") != method
            or payload.get("unit_manifest_canonical_sha256")
            != manifest["unit_manifest_canonical_sha256"]
        ):
            raise ValueError(f"{method}: generation-manifest identity mismatch")
        if payload.get("generation_spec") != protocol.GENERATION_SPEC or not protocol.is_sha256(
            payload.get("model_inventory_sha256")
        ):
            raise ValueError(f"{method}: generation spec/model inventory mismatch")
        if payload["model_inventory_sha256"] != eligibility_payload.get(
            "model_content_inventory_sha256"
        ):
            raise ValueError(f"{method}: generation model inventory differs from checkpoint")
        method_artifact = payload.get("method_artifact")
        if method == "original":
            expected_method_artifact: dict[str, Any] = {"kind": "base_model"}
        elif method == "v3b":
            expected_method_artifact = {
                "kind": "lora_checkpoint",
                "path": protocol.V3B_CHECKPOINT,
                "sha256": protocol.V3B_CHECKPOINT_SHA256,
                "scale": 1.25,
                "step": 200,
            }
        else:
            expected_method_artifact = {
                "kind": "lora_checkpoint",
                "checkpoint_eligibility_path": str(checkpoint_eligibility_path),
                "checkpoint_eligibility_sha256": protocol.file_sha256(
                    checkpoint_eligibility_path
                ),
                "path": eligibility_payload.get("checkpoint", {}).get("path"),
                "weights_sha256": eligibility_payload.get("checkpoint", {}).get(
                    "weights_sha256"
                ),
                "scale": 1.25,
                "step": 200,
            }
        if method_artifact != expected_method_artifact:
            raise ValueError(f"{method}: generation method artifact mismatch")
        model_inventory_hashes.add(str(payload["model_inventory_sha256"]))
        if not protocol.is_sha256(payload.get("runtime_registry_sha256")):
            raise ValueError(f"{method}: generation runtime registry is not bound")
        if payload["runtime_registry_sha256"] != eligibility_payload.get(
            "runtime_registry_sha256"
        ):
            raise ValueError(f"{method}: generation runtime differs from checkpoint")
        runtime_registry_hashes.add(str(payload["runtime_registry_sha256"]))
        records = payload.get("videos")
        if not isinstance(records, list) or len(records) != expected_units:
            raise ValueError(f"{method}: generation video inventory count mismatch")
        generation_videos[method] = {str(record["unit_id"]): record for record in records}
        if len(generation_videos[method]) != expected_units:
            raise ValueError(f"{method}: duplicate generation unit ID")
        for index, record in enumerate(records):
            expected_record_fields = {
                "unit_id",
                "index",
                "path",
                "size_bytes",
                "sha256",
                "prompt_sha256",
                "seed",
                "frame_count",
                "width",
                "height",
                "fps_numerator",
                "fps_denominator",
            }
            if (
                not isinstance(record, dict)
                or set(record) != expected_record_fields
                or record["index"] != index
                or record["frame_count"] != protocol.FRAME_COUNT
                or record["width"] != protocol.WIDTH
                or record["height"] != protocol.HEIGHT
                or record["fps_numerator"] != protocol.FPS.numerator
                or record["fps_denominator"] != protocol.FPS.denominator
                or not protocol.is_sha256(record["sha256"])
                or not protocol.is_sha256(record["prompt_sha256"])
            ):
                raise ValueError(f"{method}: generation video record contract mismatch")
            unit = unit_by_id.get(str(record["unit_id"]))
            if (
                unit is None
                or record["seed"] != int(unit["seed"])
                or record["prompt_sha256"]
                != hashlib.sha256(str(unit["prompt"]).encode("utf-8")).hexdigest()
            ):
                raise ValueError(f"{method}: generation video differs from committed unit")
            source = protocol.resolve_path(project_root, str(record["path"]))
            if not source.is_file() or source.is_symlink():
                raise ValueError(f"{method}: generated source video missing")
            resolved = source.resolve(strict=True)
            inode = (source.stat().st_dev, source.stat().st_ino)
            digest = protocol.file_sha256(source)
            if (
                resolved in all_source_paths
                or inode in all_source_inodes
                or digest in all_source_hashes
            ):
                raise ValueError("generation source video path/inode/content reuse detected")
            all_source_paths.add(resolved)
            all_source_inodes.add(inode)
            all_source_hashes.add(digest)
            if record["size_bytes"] != source.stat().st_size or record["sha256"] != digest:
                raise ValueError(f"{method}: source video byte inventory mismatch")
            decoded = dict(decode(source))
            expected_decoded = {
                "frame_count": protocol.FRAME_COUNT,
                "width": protocol.WIDTH,
                "height": protocol.HEIGHT,
                "fps_numerator": protocol.FPS.numerator,
                "fps_denominator": protocol.FPS.denominator,
            }
            if decoded != expected_decoded:
                raise ValueError(f"{method}: full-video decode contract mismatch")
    if len(model_inventory_hashes) != 1:
        raise ValueError("O/v3b/v4 generation arms use different model inventories")
    if len(runtime_registry_hashes) != 1:
        raise ValueError("O/v3b/v4 generation arms use different runtime registries")
    runtime_sha256 = next(iter(runtime_registry_hashes))
    protocol.validate_runtime_registry(
        protocol.resolve_path(project_root, protocol.RUNTIME_REGISTRY), runtime_sha256
    )
    assignment: dict[str, str] = {}
    reconstructed: dict[str, dict[str, Any]] = {}
    units_seen: dict[str, set[str]] = defaultdict(set)
    for review_id, public in template_by_id.items():
        key = key_by_id[review_id]
        code = key["arm_code"]
        method = key["method"]
        unit_id = key["unit_id"]
        unit = unit_by_id.get(unit_id)
        if unit is None:
            raise ValueError("answer key references an unknown committed unit")
        case_field = "semantic_case_id" if dataset == "causal" else "specificity_case_id"
        expected_key_metadata = {
            "case_id": str(unit[case_field]),
            "group": str(unit.get("group", "")),
            "membership": str(unit.get("membership", "")),
            "prompt_variant": str(unit["prompt_variant"]),
            "replicate": str(unit["replicate"]),
            "seed": str(unit["seed"]),
            "source_id": str(unit["source_id"]),
            "source_phrase": str(unit["source_phrase"]),
            "receiver_id": str(unit["receiver_id"]),
            "receiver": str(unit["receiver"]),
        }
        if any(str(key[field]) != value for field, value in expected_key_metadata.items()):
            raise ValueError("answer-key metadata differs from committed unit")
        if code not in protocol.ARM_CODES or public["arm_code"] != code:
            raise ValueError("public/key arm-code mismatch")
        if code == "O" and method != "original" or code in {"A", "B"} and method not in protocol.CANDIDATE_METHODS:
            raise ValueError("answer key violates O/A/B method semantics")
        units_seen[unit_id].add(method)
        if method == "v4":
            assignment[unit_id] = code
        anonymous = protocol.resolve_path(project_root, key["anonymous_video_path"])
        source = protocol.resolve_path(project_root, key["source_video_path"])
        expected_anonymous = public_dir / "media" / f"{review_id}.mp4"
        generated = generation_videos[method].get(unit_id)
        if generated is None:
            raise ValueError("answer key references an unknown generated unit")
        generated_path = protocol.resolve_path(project_root, str(generated["path"]))
        if (
            anonymous.resolve(strict=True) != expected_anonymous.resolve(strict=True)
            or public["video_path"] != str(expected_anonymous)
            or source.resolve(strict=True) != generated_path.resolve(strict=True)
            or protocol.file_sha256(source) != key["source_video_sha256"]
            or key["source_video_sha256"] != generated.get("sha256")
            or protocol.file_sha256(anonymous) != key["anonymous_video_sha256"]
            or key["anonymous_video_sha256"] != key["source_video_sha256"]
            or anonymous.samefile(source)
        ):
            raise ValueError("answer key anonymous/source video binding mismatch")
        composite = public_dir / "composites" / f"{public['anonymous_unit']}.jpg"
        if Path(public["composite_path"]).resolve() != composite.resolve():
            raise ValueError("public composite path mismatch")
        record = reconstructed.setdefault(
            unit_id,
            {
                "unit_id": unit_id,
                "semantic_case_id" if dataset == "causal" else "specificity_case_id": key["case_id"],
                "group": key["group"],
                "membership": key["membership"],
                "prompt_variant": key["prompt_variant"],
                "replicate": int(key["replicate"]),
            },
        )
        if any(
            str(record.get(field, ""))
            != str(
                {
                    "semantic_case_id": key["case_id"],
                    "specificity_case_id": key["case_id"],
                    "group": key["group"],
                    "membership": key["membership"],
                    "prompt_variant": key["prompt_variant"],
                    "replicate": int(key["replicate"]),
                }.get(field, record.get(field, ""))
            )
            for field in record
            if field != "unit_id"
        ):
            raise ValueError("answer-key semantic metadata differs across O/A/B")
    if len(reconstructed) != expected_units or any(methods != set(protocol.METHODS) for methods in units_seen.values()):
        raise ValueError("answer key does not contain exact O/v3b/v4 triples")
    validate_blocked_assignment(list(reconstructed.values()), dataset, assignment)
    if manifest["assignment_sha256"] != protocol.canonical_json_sha256(assignment):
        raise ValueError("review manifest assignment digest mismatch")
    expected_assignment = (
        derive_causal_ab_assignment(unit_rows, assignment_salt)
        if dataset == "causal"
        else derive_specificity_ab_assignment(unit_rows, assignment_salt)
    )
    expected_order = derive_review_order(unit_rows, assignment_salt)
    if (
        assignment != expected_assignment
        or protocol.canonical_json_sha256(expected_assignment)
        != commitment["assignment_sha256"]
        or protocol.canonical_json_sha256(expected_order)
        != commitment["review_order_sha256"]
    ):
        raise ValueError("answer key assignment/order does not recompute from opened salt")
    for position, unit_id in enumerate(expected_order):
        review_ids = {f"r{position:03d}_{code}" for code in protocol.ARM_CODES}
        if {key_by_id[review_id]["unit_id"] for review_id in review_ids} != {unit_id}:
            raise ValueError("answer key review order does not recompute from opened salt")
    manifest = dict(manifest)
    manifest["_validated_model_inventory_sha256"] = next(iter(model_inventory_hashes))
    manifest["_validated_runtime_registry_sha256"] = runtime_sha256
    manifest["_validated_source_video_sha256"] = sorted(all_source_hashes)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=protocol.DATASETS, required=True)
    parser.add_argument("--unit-manifest", type=Path, required=True)
    parser.add_argument("--causal-selected", type=Path)
    parser.add_argument("--generation-manifest-original", type=Path, required=True)
    parser.add_argument("--generation-manifest-v3b", type=Path, required=True)
    parser.add_argument("--generation-manifest-v4", type=Path, required=True)
    parser.add_argument("--assignment-salt-file", type=Path, required=True)
    parser.add_argument("--training-authorization", type=Path, default=Path(protocol.TRAINING_AUTHORIZATION))
    parser.add_argument("--checkpoint-eligibility", type=Path, default=Path(protocol.CHECKPOINT_ELIGIBILITY))
    parser.add_argument("--public-dir", type=Path, required=True)
    parser.add_argument("--private-dir", type=Path, required=True)
    parser.add_argument("--package-commitment", type=Path, required=True)
    args = parser.parse_args()
    protocol.reject_sealed_final36_path(*[value for value in vars(args).values() if isinstance(value, Path)])
    project_root = Path.cwd()
    protocol.validate_training_authorization(
        project_root,
        expected_gate_spec=protocol.GATE_SPEC,
        authorization_path=args.training_authorization,
    )
    protocol.validate_checkpoint_eligibility(project_root, args.checkpoint_eligibility)
    units = protocol.read_csv(args.unit_manifest)
    if args.dataset == "causal":
        protocol.validate_causal_unit_manifest(units)
    else:
        if args.causal_selected is None:
            parser.error("specificity review requires --causal-selected")
        causal_cases = protocol.read_csv(args.causal_selected)
        protocol.validate_specificity_unit_manifest(units, causal_cases=causal_cases)
    manifests = {
        "original": args.generation_manifest_original,
        "v3b": args.generation_manifest_v3b,
        "v4": args.generation_manifest_v4,
    }
    videos = protocol.validate_generation_bundle(
        project_root,
        dataset=args.dataset,
        unit_rows=units,
        manifest_paths=manifests,
        checkpoint_eligibility_path=args.checkpoint_eligibility,
    )
    salt = args.assignment_salt_file.read_text(encoding="utf-8").strip()
    build_review_package(
        project_root=project_root,
        dataset=args.dataset,
        unit_rows=units,
        unit_manifest_path=args.unit_manifest,
        videos=videos,
        generation_manifest_paths=manifests,
        checkpoint_eligibility_path=args.checkpoint_eligibility,
        public_dir=args.public_dir,
        private_dir=args.private_dir,
        package_commitment_path=args.package_commitment,
        private_salt=salt,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
