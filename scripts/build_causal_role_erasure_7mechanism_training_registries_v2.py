#!/usr/bin/env python3
"""Build and finalize registry-bound seven-mechanism training inputs.

``build-inputs`` freezes selected targets, the shared preservation bank, arm
mappings, and seven cache-input registries.  ``finalize-runs`` is deliberately
separate: it emits the 18 frozen trainer run specs only after the real cache
manifests and matched-tensor receipts can be validated.  No command loads a
generative model or starts GPU work.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import random
import re
import shutil
import subprocess
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import build_causal_role_erasure_7mechanism_main_v2 as main_data
import prepare_causal_role_erasure_7mechanism_training_cache_v2 as cache_contract
import train_wan_causal_role_lora_v2 as trainer_contract
from build_water_impact_dynamic_pairs_v1 import factual_prompt as water_factual_prompt


PROTOCOL_VERSION = main_data.PROTOCOL_VERSION
REGISTRY_PROTOCOL = cache_contract.REGISTRY_PROTOCOL
RUN_SPEC_PROTOCOL = trainer_contract.RUN_SPEC_PROTOCOL
BUILD_PROTOCOL = "causal_role_erasure_7mechanism_training_registries_v2"
BUILD_STATUS = "training_inputs_frozen_pre_cache"
FINAL_STATUS = "training_run_specs_frozen_after_cache_validation"
MECHANISM_ORDER = main_data.MECHANISM_ORDER
ARMS = trainer_contract.ARMS
CACHE_ARM = trainer_contract.ARM_CACHE_NAMES
ERASE_ROWS = 178
PRESERVE_ROWS = 36
BASE_ROWS = 214
MEDIA_CONTRACT = {"decoded_frames": 49, "fps": "8/1", "height": 480, "width": 832}
HEX64 = re.compile(r"^[0-9a-f]{64}$")
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,199}$")
SEALED = ("final36", "sealed-final", "sealed_final")

SELECTED_FIELDS = (
    "protocol_version", "mechanism", "selected_index", "candidate_id",
    "global_index", "source_id", "source_object", "receiver_id", "receiver",
    "prompt_style", "factual_prompt", "target_prompt", "seed",
    "target_video_path", "target_video_sha256",
)
PRESERVE_FIELDS = (
    "protocol_version", "preserve_index", "preserve_id", "prompt",
    "target_video_path", "target_video_sha256",
)
MAPPING_FIELDS = (
    "protocol_version", "mechanism", "arm", "selected_index", "candidate_id",
    "student_prompt", "assigned_source_id", "assigned_source_object",
    "mapping_ordinal", "active_erase_ordinal", "source_bank_index",
    "assigned_source_membership", "original_source_id", "original_source_object",
    "receiver_id", "receiver", "prompt_style", "original_factual_prompt",
    "assignment_salt_sha256", "mapping_rule",
)
GENERIC_MAPPING_FIELDS = (
    "protocol_version", "mechanism", "arm", "selected_index", "candidate_id",
    "student_prompt", "assigned_source_id", "assigned_source_object",
    "original_source_id", "original_source_object", "receiver_id", "receiver",
    "prompt_style", "control_rule",
)

PYAV_PROBE_CODE = r"""
import av, json, sys
from fractions import Fraction
path = sys.argv[1]
with av.open(path) as container:
    videos = [stream for stream in container.streams if stream.type == 'video']
    audios = [stream for stream in container.streams if stream.type == 'audio']
    if len(videos) != 1 or audios:
        raise SystemExit('expected exactly one video stream and no audio')
    stream = videos[0]
    rate = stream.average_rate or stream.guessed_rate
    count = sum(1 for _ in container.decode(video=stream.index))
    print(json.dumps({
        'decoded_frames': count,
        'fps': '' if rate is None else f'{Fraction(rate).numerator}/{Fraction(rate).denominator}',
        'height': int(stream.height),
        'width': int(stream.width),
    }, sort_keys=True))
"""


class RegistryError(ValueError):
    """Fail-closed registry construction error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RegistryError(message)


def reject_sealed(*values: str | Path) -> None:
    for value in values:
        require(
            not any(token in str(value).casefold() for token in SEALED),
            f"sealed/final36 path is forbidden: {value}",
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def require_sha256(value: Any, label: str) -> str:
    require(isinstance(value, str) and HEX64.fullmatch(value) is not None, f"{label} is not SHA-256")
    return value


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def csv_bytes(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(fields), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field, "") for field in fields})
    return stream.getvalue().encode("utf-8")


def regular_file(path: Path, label: str) -> None:
    require(path.is_file() and not path.is_symlink(), f"{label} missing or symlinked: {path}")


def resolve_path(project_root: Path, value: str | Path, *, strict: bool = True) -> Path:
    reject_sealed(value)
    path = Path(value)
    path = path if path.is_absolute() else project_root / path
    result = path.resolve(strict=strict)
    reject_sealed(result)
    return result


def canonical_path(project_root: Path, path: Path) -> str:
    resolved = path.resolve(strict=False)
    try:
        return resolved.relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def require_distinct_nonnested(paths: Mapping[str, Path]) -> None:
    resolved = {name: path.resolve(strict=False) for name, path in paths.items()}
    for left_name, left in resolved.items():
        for right_name, right in resolved.items():
            if left_name >= right_name:
                continue
            require(
                left != right and left not in right.parents and right not in left.parents,
                f"{left_name} and {right_name} must be distinct and nonnested",
            )


def verify_input(path: Path, expected_sha256: str, label: str) -> None:
    regular_file(path, label)
    require_sha256(expected_sha256, f"{label} expected hash")
    require(sha256_file(path) == expected_sha256, f"{label} byte hash mismatch")


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    regular_file(path, "CSV input")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    require(fields and all(None not in row for row in rows), f"invalid CSV structure: {path}")
    return fields, [{field: str(row[field]) for field in fields} for row in rows]


def artifact_ref(project_root: Path, path: Path, row_count: int) -> dict[str, Any]:
    regular_file(path, "artifact reference")
    return {
        "path": canonical_path(project_root, path),
        "sha256": sha256_file(path),
        "row_count": row_count,
    }


def artifact_ref_bytes(project_root: Path, final_path: Path, raw: bytes, row_count: int) -> dict[str, Any]:
    return {
        "path": canonical_path(project_root, final_path),
        "sha256": sha256_bytes(raw),
        "row_count": row_count,
    }


def _int(value: str, label: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise RegistryError(f"{label} must be an integer") from exc
    return result


def probe_media(runtime_python: Path, path: Path) -> dict[str, Any]:
    environment = os.environ.copy()
    environment.pop("PYTHONHOME", None)
    environment.pop("PYTHONPATH", None)
    environment.update(PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1", PYTHONSAFEPATH="1")
    result = subprocess.run(
        [str(runtime_python), "-c", PYAV_PROBE_CODE, str(path)],
        text=True,
        capture_output=True,
        check=False,
        env=environment,
    )
    require(result.returncode == 0, f"cannot decode preservation media {path}: {result.stderr.strip()}")
    payload = json.loads(result.stdout)
    require(payload == MEDIA_CONTRACT, f"preservation media contract mismatch: {path}")
    return payload


def _publish_fresh(root: Path, payloads: Mapping[Path, bytes]) -> None:
    reject_sealed(root)
    require(not root.exists() and not root.is_symlink(), f"output root must be fresh: {root}")
    root.parent.mkdir(parents=True, exist_ok=True)
    require(not root.parent.is_symlink(), "output parent is symlinked")
    partial = root.with_name(f".{root.name}.partial-{os.getpid()}")
    require(not partial.exists(), f"stale partial output exists: {partial}")
    partial.mkdir(mode=0o700)
    try:
        for final_path, raw in payloads.items():
            relative = final_path.relative_to(root)
            target = partial / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            descriptor = os.open(target, flags, 0o600)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
        os.replace(partial, root)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise


def load_ontology(path: Path) -> tuple[dict[str, Any], dict[str, Mapping[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(payload, dict), "ontology registry must be an object")
    main_data.validate_ontology_payload(payload)
    require(payload["protocol_version"] == PROTOCOL_VERSION, "ontology protocol mismatch")
    return payload, {row["mechanism"]: row for row in payload["mechanisms"]}


def validate_run_matrix(path: Path) -> list[dict[str, str]]:
    fields, rows = read_csv(path)
    require(tuple(fields) == main_data.RUN_FIELDS, "run-matrix columns/order changed")
    require(len(rows) == 18, "run matrix must contain 18 rows")
    require([_int(row["run_index"], "run_index") for row in rows] == list(range(18)), "run indices invalid")
    require(len({row["run_id"] for row in rows}) == 18, "run IDs repeat")
    require(Counter(row["arm"] for row in rows) == Counter({"matched_control": 7, "V4": 7, "generic_paraphrase": 2, "bystander_token": 2}), "run arms invalid")
    for row in rows:
        require(row["mechanism"] in MECHANISM_ORDER and row["arm"] in ARMS, "run identity invalid")
        require(
            row["training_seed"] == "26000"
            and row["updates"] == "200"
            and row["erase_updates"] == row["preserve_updates"] == "100"
            and row["selected_erase_rows_required"] == "178"
            and row["preserve_rows"] == "36"
            and row["lora_rank"] == row["lora_alpha"] == "16"
            and row["learning_rate"] == "5e-5"
            and row["target_teacher_weight"] == row["preservation_weight"] == "4"
            and row["checkpoint"] == "200"
            and row["inference_scale"] == "1.25",
            f"run configuration drift: {row['run_id']}",
        )
    return rows


def validate_selected_targets(
    path: Path,
    *,
    project_root: Path,
    ontologies: Mapping[str, Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    fields, rows = read_csv(path)
    required = {
        *main_data.TARGET_FIELDS,
        "eligible", "selection_hash", "selection_rank",
        "selected_video_path", "selected_video_sha256", "selected_media",
    }
    require(required <= set(fields), f"selected targets missing columns: {sorted(required - set(fields))}")
    require(len(rows) == len(MECHANISM_ORDER) * ERASE_ROWS, "selected target count must be 1246")
    grouped: dict[str, list[dict[str, Any]]] = {}
    cursor = 0
    seen_paths: set[Path] = set()
    seen_ids: set[str] = set()
    for mechanism_index, mechanism in enumerate(MECHANISM_ORDER):
        block = rows[cursor : cursor + ERASE_ROWS]
        cursor += ERASE_ROWS
        require(len(block) == ERASE_ROWS and {row["mechanism"] for row in block} == {mechanism}, f"{mechanism}: selected block/order invalid")
        ontology = ontologies[mechanism]
        original = {row["source_id"]: row for row in ontology["original_training_sources"]}
        train_receivers = {row["receiver_id"]: row for row in ontology["train_receivers"]}
        output: list[dict[str, Any]] = []
        for selected_index, row in enumerate(block):
            candidate_id = row["candidate_id"]
            require(SAFE_ID.fullmatch(candidate_id) is not None and candidate_id not in seen_ids, f"unsafe/duplicate candidate ID: {candidate_id}")
            seen_ids.add(candidate_id)
            require(row["protocol_version"] == PROTOCOL_VERSION, f"{candidate_id}: protocol mismatch")
            require(_int(row["mechanism_index"], "mechanism_index") == mechanism_index, f"{candidate_id}: mechanism index mismatch")
            require(row["eligible"] == "yes", f"{candidate_id}: selected row is not eligible")
            expected_selection_hash = hashlib.sha256(f"target-select-v1|{candidate_id}".encode("utf-8")).hexdigest()
            require(row["selection_hash"] == expected_selection_hash, f"{candidate_id}: selection hash mismatch")
            require(_int(row["selection_rank"], "selection_rank") == selected_index + 1, f"{candidate_id}: selection rank mismatch")
            require(row["source_id"] in original, f"{candidate_id}: source is outside original training eight")
            require(row["source_object"] == original[row["source_id"]]["canonical_phrase"], f"{candidate_id}: source phrase mismatch")
            require(row["receiver_id"] in train_receivers, f"{candidate_id}: receiver outside training twelve")
            require(row["receiver"] == train_receivers[row["receiver_id"]]["canonical_phrase"], f"{candidate_id}: receiver phrase mismatch")
            require(row["prompt_style"] in main_data.PROMPT_STYLES, f"{candidate_id}: prompt style invalid")
            rebuilt = rebuild_factual_prompt(mechanism, row["source_object"], row, ontology)
            require(rebuilt == row["factual_prompt"], f"{candidate_id}: factual prompt is not canonical")
            path_value = row["selected_video_path"]
            video_path = resolve_path(project_root, path_value)
            regular_file(video_path, f"selected target {candidate_id}")
            require(video_path not in seen_paths, f"selected target path repeats: {video_path}")
            seen_paths.add(video_path)
            expected_sha = require_sha256(row["selected_video_sha256"], f"{candidate_id} video")
            require(sha256_file(video_path) == expected_sha, f"{candidate_id}: selected target bytes changed")
            media = json.loads(row["selected_media"])
            require(media == MEDIA_CONTRACT, f"{candidate_id}: selected media declaration mismatch")
            output.append(
                {
                    "protocol_version": PROTOCOL_VERSION,
                    "mechanism": mechanism,
                    "selected_index": selected_index,
                    "candidate_id": candidate_id,
                    "global_index": _int(row["global_index"], "global_index"),
                    "source_id": row["source_id"],
                    "source_object": row["source_object"],
                    "receiver_id": row["receiver_id"],
                    "receiver": row["receiver"],
                    "prompt_style": row["prompt_style"],
                    "factual_prompt": row["factual_prompt"],
                    "target_prompt": row["target_prompt"],
                    "seed": _int(row["seed"], "seed"),
                    "target_video_path": canonical_path(project_root, video_path),
                    "target_video_sha256": expected_sha,
                }
            )
        require(
            [row["candidate_id"] for row in block]
            == [
                row["candidate_id"]
                for row in sorted(
                    block,
                    key=lambda item: (
                        item["selection_hash"], item["candidate_id"]
                    ),
                )
            ],
            f"{mechanism}: selected targets are not in frozen selection-hash order",
        )
        require(len({row["global_index"] for row in output}) == ERASE_ROWS, f"{mechanism}: selected global indices repeat")
        grouped[mechanism] = output
    return grouped


def rebuild_factual_prompt(
    mechanism: str,
    source_object: str,
    selected_row: Mapping[str, Any],
    ontology: Mapping[str, Any],
) -> str:
    if mechanism == "water_impact":
        return water_factual_prompt(
            source_object,
            str(selected_row["receiver"]),
            str(selected_row["prompt_style"]),
        )
    receiver = next(
        row for row in ontology["train_receivers"]
        if row["receiver_id"] == selected_row["receiver_id"]
    )
    return main_data.factual_prompt(
        mechanism,
        source_object,
        receiver,
        str(selected_row["prompt_style"]),
        eval_wording=False,
    )


def discover_preserve_video(root: Path, index: int, preserve_id: str) -> Path:
    nested = root / f"prompt_{index:03d}" / "videos"
    candidates: list[Path] = []
    if nested.is_dir() and not nested.is_symlink():
        candidates.extend(sorted(nested.glob("*.mp4")))
    candidates.extend(
        path for path in (
            root / f"{index:03d}_{preserve_id}.mp4",
            root / f"{preserve_id}.mp4",
        )
        if path.exists() or path.is_symlink()
    )
    unique = list(dict.fromkeys(path.resolve(strict=False) for path in candidates))
    require(len(unique) == 1, f"preserve row {index} must resolve to exactly one MP4, found {len(unique)}")
    return unique[0]


def build_preserve36(
    manifest_path: Path,
    media_root: Path,
    *,
    project_root: Path,
    runtime_python: Path,
    media_probe: Callable[[Path, Path], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    fields, rows = read_csv(manifest_path)
    required = {
        "protocol_version", "sample_id", "training_role", "mechanism",
        "prompt", "seed", "num_frames", "fps",
    }
    require(required <= set(fields), "legacy preserve manifest schema is incomplete")
    require(len(rows) == PRESERVE_ROWS, "legacy preserve manifest must contain 36 rows")
    require(media_root.is_dir() and not media_root.is_symlink(), "preserve media root is missing/symlinked")
    output: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_paths: set[Path] = set()
    for index, row in enumerate(rows):
        preserve_id = row["sample_id"]
        require(SAFE_ID.fullmatch(preserve_id) is not None and preserve_id not in seen_ids, f"preserve row {index}: ID invalid/repeated")
        seen_ids.add(preserve_id)
        require(
            row["protocol_version"] == "v1"
            and row["training_role"] == "preserve"
            and row["mechanism"] == "generic_preservation"
            and row["seed"] == "12000"
            and row["num_frames"] == "49"
            and row["fps"] == "8",
            f"preserve row {index}: frozen legacy semantics changed",
        )
        require(row["prompt"].strip(), f"preserve row {index}: prompt empty")
        video_path = discover_preserve_video(media_root, index, preserve_id)
        regular_file(video_path, f"preserve media {index}")
        require(video_path not in seen_paths, f"preserve media repeats: {video_path}")
        seen_paths.add(video_path)
        require(dict(media_probe(runtime_python, video_path)) == MEDIA_CONTRACT, f"preserve media {index}: decoded contract mismatch")
        output.append(
            {
                "protocol_version": PROTOCOL_VERSION,
                "preserve_index": index,
                "preserve_id": preserve_id,
                "prompt": row["prompt"],
                "target_video_path": canonical_path(project_root, video_path),
                "target_video_sha256": sha256_file(video_path),
            }
        )
    return output


def build_schedule() -> list[tuple[str, int]]:
    schedule = trainer_contract.build_schedule()
    require(
        trainer_contract.schedule_sha256(schedule)
        == trainer_contract.EXPECTED_SCHEDULE_SHA256,
        "unified training schedule digest changed",
    )
    require(
        Counter(role for role, _index in schedule)
        == Counter({"erase": 100, "preserve": 100}),
        "training schedule role counts changed",
    )
    return schedule


def _ranked_bank(bank: Sequence[Mapping[str, Any]], salt: str) -> list[Mapping[str, Any]]:
    ranked = [
        (
            hashlib.sha256(
                f"{salt}\0source-permute-7m-v2\0{source['source_id']}".encode("utf-8")
            ).hexdigest(),
            source,
        )
        for source in bank
    ]
    require(len({rank for rank, _source in ranked}) == 64, "source permutation ranks collide")
    return [source for _rank, source in sorted(ranked, key=lambda item: item[0])]


def _repair_derangement(
    assignments: list[Mapping[str, Any]],
    ordered_rows: Sequence[tuple[int, Mapping[str, Any]]],
    *,
    salt: str,
) -> None:
    for position, (_selected_index, selected) in enumerate(ordered_rows):
        assigned = assignments[position]
        if (
            assigned["source_id"] != selected["source_id"]
            and assigned["canonical_phrase"] != selected["source_object"]
        ):
            continue
        partition = (
            range(0, 100) if position < 100 else range(100, ERASE_ROWS)
        )
        candidates: list[tuple[str, int]] = []
        for other in partition:
            if other == position:
                continue
            other_selected = ordered_rows[other][1]
            other_assigned = assignments[other]
            if (
                other_assigned["source_id"] == selected["source_id"]
                or other_assigned["canonical_phrase"] == selected["source_object"]
                or assigned["source_id"] == other_selected["source_id"]
                or assigned["canonical_phrase"] == other_selected["source_object"]
            ):
                continue
            rank = hashlib.sha256(
                f"{salt}\0source-swap-7m-v2\0{position}\0{other}".encode("utf-8")
            ).hexdigest()
            candidates.append((rank, other))
        require(candidates, f"cannot repair source assignment at mapping position {position}")
        other = min(candidates)[1]
        assignments[position], assignments[other] = assignments[other], assignments[position]
    require(
        all(
            assigned["source_id"] != selected["source_id"]
            and assigned["canonical_phrase"] != selected["source_object"]
            for assigned, (_index, selected) in zip(assignments, ordered_rows)
        ),
        "source derangement repair left an original-source assignment",
    )


def build_v4_mapping(
    mechanism: str,
    selected: Sequence[Mapping[str, Any]],
    ontology: Mapping[str, Any],
    *,
    ontology_sha256: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    require(len(selected) == ERASE_ROWS, f"{mechanism}: selected178 missing")
    bank = ontology["source_bank64"]
    require(len(bank) == 64 and [row["bank_index"] for row in bank] == list(range(64)), f"{mechanism}: source bank order invalid")
    schedule = build_schedule()
    active_indices = [index for role, index in schedule if role == "erase"]
    require(len(active_indices) == 100 and len(set(active_indices)) == 100, "active100 erase rows are not unique")
    remaining = [index for index in range(ERASE_ROWS) if index not in set(active_indices)]
    ordered_indices = active_indices + remaining
    ordered_rows = [(index, selected[index]) for index in ordered_indices]
    salt = hashlib.sha256(
        f"source-assignment-7m-v2|{PROTOCOL_VERSION}|{mechanism}|{ontology_sha256}".encode("utf-8")
    ).hexdigest()
    permutation = _ranked_bank(bank, salt)
    assignments = [permutation[ordinal % 64] for ordinal in range(ERASE_ROWS)]
    _repair_derangement(assignments, ordered_rows, salt=salt)
    active_counts = Counter(row["source_id"] for row in assignments[:100])
    full_counts = Counter(row["source_id"] for row in assignments)
    require(
        len(active_counts) == 64
        and set(active_counts.values()) == {1, 2}
        and max(active_counts.values()) - min(active_counts.values()) == 1,
        f"{mechanism}: active100 source balance invalid",
    )
    require(
        len(full_counts) == 64
        and set(full_counts.values()) == {2, 3}
        and max(full_counts.values()) - min(full_counts.values()) == 1,
        f"{mechanism}: full178 source balance invalid",
    )
    mapping_ordinal = {selected_index: ordinal for ordinal, selected_index in enumerate(ordered_indices)}
    active_ordinal = {selected_index: ordinal for ordinal, selected_index in enumerate(active_indices)}
    assigned_by_selected = {
        selected_index: assigned
        for (selected_index, _row), assigned in zip(ordered_rows, assignments)
    }
    records: list[dict[str, Any]] = []
    for selected_index, row in enumerate(selected):
        assigned = assigned_by_selected[selected_index]
        student_prompt = rebuild_factual_prompt(
            mechanism, assigned["canonical_phrase"], row, ontology
        )
        require(student_prompt != row["factual_prompt"], f"{mechanism}/{row['candidate_id']}: V4 prompt unchanged")
        record = {
            "protocol_version": PROTOCOL_VERSION,
            "mechanism": mechanism,
            "arm": "v4",
            "selected_index": selected_index,
            "candidate_id": row["candidate_id"],
            "student_prompt": student_prompt,
            "assigned_source_id": assigned["source_id"],
            "assigned_source_object": assigned["canonical_phrase"],
            "mapping_ordinal": mapping_ordinal[selected_index],
            "active_erase_ordinal": active_ordinal.get(selected_index, ""),
            "source_bank_index": assigned["bank_index"],
            "assigned_source_membership": assigned["membership"],
            "original_source_id": row["source_id"],
            "original_source_object": row["source_object"],
            "receiver_id": row["receiver_id"],
            "receiver": row["receiver"],
            "prompt_style": row["prompt_style"],
            "original_factual_prompt": row["factual_prompt"],
            "assignment_salt_sha256": hashlib.sha256(salt.encode("ascii")).hexdigest(),
            "mapping_rule": "active100_then_remaining_sha256_bank_permutation_deranged_within_partition_v2",
        }
        require(tuple(record) == MAPPING_FIELDS, "V4 mapping field order changed")
        records.append(record)
    metadata = {
        "mechanism": mechanism,
        "source_assignment_salt_sha256": hashlib.sha256(salt.encode("ascii")).hexdigest(),
        "schedule_sha256": trainer_contract.EXPECTED_SCHEDULE_SHA256,
        "active100_selected_indices_sha256": sha256_bytes(
            canonical_json_bytes(active_indices)
        ),
        "active100_mapping_sha256": sha256_bytes(
            canonical_json_bytes(
                [records[index] for index in active_indices]
            )
        ),
        "full178_mapping_sha256": sha256_bytes(canonical_json_bytes(records)),
        "active_source_count_min": min(active_counts.values()),
        "active_source_count_max": max(active_counts.values()),
        "full_source_count_min": min(full_counts.values()),
        "full_source_count_max": max(full_counts.values()),
        "active_source_counts": dict(sorted(active_counts.items())),
        "full_source_counts": dict(sorted(full_counts.items())),
        "no_original_source_assignments": True,
    }
    return records, metadata


def build_identification_mappings(
    mechanism: str,
    selected: Sequence[Mapping[str, Any]],
    ontology: Mapping[str, Any],
    v4_records: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    require(mechanism in {"water_impact", "brittle_fracture"}, "identification mapping mechanism invalid")
    generic: list[dict[str, Any]] = []
    bystander: list[dict[str, Any]] = []
    for index, (row, v4) in enumerate(zip(selected, v4_records)):
        opposite_style = "natural" if row["prompt_style"] == "direct" else "direct"
        generic_row = dict(row)
        generic_row["prompt_style"] = opposite_style
        generic_prompt = rebuild_factual_prompt(
            mechanism, row["source_object"], generic_row, ontology
        )
        require(generic_prompt != row["factual_prompt"], f"{mechanism}/{row['candidate_id']}: generic paraphrase unchanged")
        generic_record = {
            "protocol_version": PROTOCOL_VERSION,
            "mechanism": mechanism,
            "arm": "generic_paraphrase",
            "selected_index": index,
            "candidate_id": row["candidate_id"],
            "student_prompt": generic_prompt,
            "assigned_source_id": "",
            "assigned_source_object": "",
            "original_source_id": row["source_id"],
            "original_source_object": row["source_object"],
            "receiver_id": row["receiver_id"],
            "receiver": row["receiver"],
            "prompt_style": row["prompt_style"],
            "control_rule": "same_original_causal_source_identity_opposite_registered_training_wording",
        }
        require(tuple(generic_record) == GENERIC_MAPPING_FIELDS, "generic mapping field order changed")
        generic.append(generic_record)

        assigned_phrase = str(v4["assigned_source_object"])
        bystander_prompt = (
            f"{row['factual_prompt']} Separately, {assigned_phrase} remains motionless "
            "in the far background and never contacts or affects the receiver."
        )
        require(
            assigned_phrase != row["source_object"]
            and main_data.normalize(assigned_phrase)
            in main_data.normalize(bystander_prompt)
            and main_data.normalize(str(row["source_object"]))
            in main_data.normalize(bystander_prompt),
            f"{mechanism}/{row['candidate_id']}: bystander identity rule failed",
        )
        bystander_record = {
            "protocol_version": PROTOCOL_VERSION,
            "mechanism": mechanism,
            "arm": "bystander",
            "selected_index": index,
            "candidate_id": row["candidate_id"],
            "student_prompt": bystander_prompt,
            "assigned_source_id": v4["assigned_source_id"],
            "assigned_source_object": assigned_phrase,
            "original_source_id": row["source_id"],
            "original_source_object": row["source_object"],
            "receiver_id": row["receiver_id"],
            "receiver": row["receiver"],
            "prompt_style": row["prompt_style"],
            "control_rule": "V4_identity_and_frequency_as_noncausal_bystander_true_source_unchanged",
        }
        require(tuple(bystander_record) == GENERIC_MAPPING_FIELDS, "bystander mapping field order changed")
        bystander.append(bystander_record)
    require(
        Counter(row["assigned_source_id"] for row in bystander)
        == Counter(row["assigned_source_id"] for row in v4_records),
        f"{mechanism}: bystander/V4 identity frequencies differ",
    )
    return generic, bystander


def validate_model_runtime_inputs(
    *,
    project_root: Path,
    model_root: Path,
    model_inventory: Path,
    runtime_root: Path,
    runtime_registry: Path,
    python_executable: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    require(model_root.is_dir() and not model_root.is_symlink(), "model root missing/symlinked")
    require(runtime_root.is_dir() and not runtime_root.is_symlink(), "runtime root missing/symlinked")
    regular_file(python_executable, "runtime Python")
    require(os.access(python_executable, os.X_OK), "runtime Python is not executable")
    model_payload = json.loads(model_inventory.read_text(encoding="utf-8"))
    runtime_payload = json.loads(runtime_registry.read_text(encoding="utf-8"))
    require(isinstance(model_payload, dict) and isinstance(model_payload.get("files"), list), "model inventory schema invalid")
    require(isinstance(runtime_payload, dict), "runtime registry schema invalid")
    model_ref = artifact_ref(project_root, model_inventory, len(model_payload["files"]))
    runtime_ref = artifact_ref(project_root, runtime_registry, 1)
    binding = {
        "project_root": str(project_root),
        "model_root": str(model_root),
        "model_inventory": {**model_ref, "path": str(model_inventory)},
        "runtime_root": str(runtime_root),
        "runtime_registry": {**runtime_ref, "path": str(runtime_registry)},
        "python_executable": str(python_executable),
    }
    cache_contract.validate_model_inventory(binding, live=False)
    cache_contract.validate_runtime(binding, live=False)
    return model_ref, runtime_ref


def build_inputs(
    *,
    project_root: Path,
    selected_targets: Path,
    ontology_registry: Path,
    run_matrix: Path,
    preserve_manifest: Path,
    preserve_media_root: Path,
    model_root: Path,
    model_inventory: Path,
    runtime_root: Path,
    runtime_registry: Path,
    python_executable: Path,
    output_root: Path,
    cache_output_root: Path,
    run_spec_output_root: Path,
    media_probe: Callable[[Path, Path], Mapping[str, Any]] = probe_media,
) -> dict[str, Any]:
    reject_sealed(
        selected_targets, ontology_registry, run_matrix, preserve_manifest,
        preserve_media_root, model_root, model_inventory, runtime_root,
        runtime_registry, python_executable, output_root, cache_output_root,
        run_spec_output_root,
    )
    require(not cache_output_root.exists() and not cache_output_root.is_symlink(), "cache output root must be fresh")
    require(not run_spec_output_root.exists() and not run_spec_output_root.is_symlink(), "run-spec output root must be fresh")
    require_distinct_nonnested(
        {
            "training_inputs": output_root,
            "training_caches": cache_output_root,
            "run_specs": run_spec_output_root,
        }
    )
    ontology_payload, ontologies = load_ontology(ontology_registry)
    run_rows = validate_run_matrix(run_matrix)
    selected = validate_selected_targets(
        selected_targets, project_root=project_root, ontologies=ontologies
    )
    preserve = build_preserve36(
        preserve_manifest,
        preserve_media_root,
        project_root=project_root,
        runtime_python=python_executable,
        media_probe=media_probe,
    )
    model_ref, runtime_ref = validate_model_runtime_inputs(
        project_root=project_root,
        model_root=model_root,
        model_inventory=model_inventory,
        runtime_root=runtime_root,
        runtime_registry=runtime_registry,
        python_executable=python_executable,
    )
    ontology_sha256 = sha256_file(ontology_registry)
    payloads: dict[Path, bytes] = {}
    preserve_path = output_root / "preserve36.csv"
    preserve_raw = csv_bytes(preserve, PRESERVE_FIELDS)
    payloads[preserve_path] = preserve_raw
    preserve_ref = artifact_ref_bytes(project_root, preserve_path, preserve_raw, PRESERVE_ROWS)
    mechanism_refs: dict[str, Any] = {}
    cache_registry_refs: dict[str, Any] = {}
    for mechanism_index, mechanism in enumerate(MECHANISM_ORDER):
        selected_path = (
            output_root / "selected_by_mechanism"
            / f"{mechanism_index:02d}_{mechanism}_selected178.csv"
        )
        selected_raw = csv_bytes(selected[mechanism], SELECTED_FIELDS)
        payloads[selected_path] = selected_raw
        selected_ref = artifact_ref_bytes(
            project_root, selected_path, selected_raw, ERASE_ROWS
        )
        v4_records, mapping_metadata = build_v4_mapping(
            mechanism,
            selected[mechanism],
            ontologies[mechanism],
            ontology_sha256=ontology_sha256,
        )
        v4_path = (
            output_root / "mappings" / f"{mechanism_index:02d}_{mechanism}_v4.csv"
        )
        v4_raw = csv_bytes(v4_records, MAPPING_FIELDS)
        payloads[v4_path] = v4_raw
        v4_ref = artifact_ref_bytes(project_root, v4_path, v4_raw, ERASE_ROWS)
        mapping_registry = {
            "schema_version": 1,
            "protocol": "causal_role_erasure_7mechanism_source_mapping_v2",
            "protocol_version": PROTOCOL_VERSION,
            "status": "frozen",
            "mechanism": mechanism,
            "selected178": selected_ref,
            "source_bank64_sha256": sha256_bytes(
                canonical_json_bytes(ontologies[mechanism]["source_bank64"])
            ),
            "mapping_csv": v4_ref,
            **mapping_metadata,
        }
        mapping_registry_path = (
            output_root / "mapping_registries"
            / f"{mechanism_index:02d}_{mechanism}_v4.json"
        )
        mapping_registry_raw = canonical_json_bytes(mapping_registry)
        payloads[mapping_registry_path] = mapping_registry_raw

        arm_refs: dict[str, Any] = {"matched": None, "v4": v4_ref}
        identification_refs: dict[str, Any] = {}
        if mechanism in {"water_impact", "brittle_fracture"}:
            generic, bystander = build_identification_mappings(
                mechanism, selected[mechanism], ontologies[mechanism], v4_records
            )
            for arm, records in (
                ("generic_paraphrase", generic),
                ("bystander", bystander),
            ):
                path = (
                    output_root / "mappings"
                    / f"{mechanism_index:02d}_{mechanism}_{arm}.csv"
                )
                raw = csv_bytes(records, GENERIC_MAPPING_FIELDS)
                payloads[path] = raw
                ref = artifact_ref_bytes(project_root, path, raw, ERASE_ROWS)
                arm_refs[arm] = ref
                identification_refs[arm] = ref

        cache_registry = {
            "schema_version": 1,
            "protocol": REGISTRY_PROTOCOL,
            "protocol_version": PROTOCOL_VERSION,
            "status": "frozen",
            "mechanism": mechanism,
            "selected178": selected_ref,
            "preserve36": preserve_ref,
            "model_root": canonical_path(project_root, model_root),
            "model_inventory": model_ref,
            "runtime_root": canonical_path(project_root, runtime_root),
            "runtime_registry": runtime_ref,
            "python_executable": canonical_path(project_root, python_executable),
            "arms": arm_refs,
        }
        cache_registry_path = (
            output_root / "cache_registries"
            / f"{mechanism_index:02d}_{mechanism}.json"
        )
        cache_registry_raw = canonical_json_bytes(cache_registry)
        payloads[cache_registry_path] = cache_registry_raw
        cache_registry_ref = artifact_ref_bytes(
            project_root, cache_registry_path, cache_registry_raw, 1
        )
        cache_registry_refs[mechanism] = cache_registry_ref
        mechanism_refs[mechanism] = {
            "selected178": selected_ref,
            "v4_mapping": v4_ref,
            "v4_mapping_registry": artifact_ref_bytes(
                project_root, mapping_registry_path, mapping_registry_raw, 1
            ),
            "identification_mappings": identification_refs,
            "cache_registry": cache_registry_ref,
            "expected_cache_layout": {
                "base": canonical_path(project_root, cache_output_root / mechanism / "base"),
                "teacher": canonical_path(project_root, cache_output_root / mechanism / "teacher"),
                "arms": {
                    cache_arm: canonical_path(
                        project_root,
                        cache_output_root / mechanism / "arms" / cache_arm,
                    )
                    for cache_arm in arm_refs
                },
            },
        }

    registry = {
        "schema_version": 1,
        "protocol": BUILD_PROTOCOL,
        "protocol_version": PROTOCOL_VERSION,
        "status": BUILD_STATUS,
        "fresh_only": True,
        "inputs": {
            "selected_targets": artifact_ref(
                project_root, selected_targets, len(MECHANISM_ORDER) * ERASE_ROWS
            ),
            "ontology_registry": artifact_ref(project_root, ontology_registry, 7),
            "run_matrix": artifact_ref(project_root, run_matrix, 18),
            "legacy_preserve_manifest": artifact_ref(
                project_root, preserve_manifest, PRESERVE_ROWS
            ),
            "preserve_media_root": canonical_path(project_root, preserve_media_root),
            "model_root": canonical_path(project_root, model_root),
            "model_inventory": model_ref,
            "runtime_root": canonical_path(project_root, runtime_root),
            "runtime_registry": runtime_ref,
            "python_executable": canonical_path(project_root, python_executable),
        },
        "counts": {
            "mechanisms": 7,
            "selected_targets": 1_246,
            "selected_per_mechanism": 178,
            "preserve_rows": 36,
            "source_bank_per_mechanism": 64,
            "active_erase_updates": 100,
            "training_runs": 18,
        },
        "preserve36": preserve_ref,
        "mechanisms": mechanism_refs,
        "cache_registries": cache_registry_refs,
        "run_matrix": [
            {
                "run_id": row["run_id"],
                "mechanism": row["mechanism"],
                "arm": row["arm"],
                "output_dir": row["output_dir"],
            }
            for row in run_rows
        ],
        "cache_output_root": canonical_path(project_root, cache_output_root),
        "run_spec_output_root": canonical_path(project_root, run_spec_output_root),
        "run_specs_emitted": False,
    }
    registry_path = output_root / "training_input_registry.json"
    payloads[registry_path] = canonical_json_bytes(registry)
    _publish_fresh(output_root, payloads)
    return {
        **registry,
        "registry": canonical_path(project_root, registry_path),
        "registry_sha256": sha256_file(registry_path),
    }


def load_input_registry(
    project_root: Path, path: Path, expected_sha256: str
) -> dict[str, Any]:
    verify_input(path, expected_sha256, "training input registry")
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version", "protocol", "protocol_version", "status",
        "fresh_only", "inputs", "counts", "preserve36", "mechanisms",
        "cache_registries", "run_matrix", "cache_output_root",
        "run_spec_output_root", "run_specs_emitted",
    }
    require(isinstance(payload, dict) and set(payload) == required, "training input registry fields are not exact")
    require(
        payload["schema_version"] == 1
        and payload["protocol"] == BUILD_PROTOCOL
        and payload["protocol_version"] == PROTOCOL_VERSION
        and payload["status"] == BUILD_STATUS
        and payload["fresh_only"] is True
        and payload["run_specs_emitted"] is False,
        "training input registry identity/status invalid",
    )
    require(payload["counts"] == {
        "mechanisms": 7,
        "selected_targets": 1_246,
        "selected_per_mechanism": 178,
        "preserve_rows": 36,
        "source_bank_per_mechanism": 64,
        "active_erase_updates": 100,
        "training_runs": 18,
    }, "training input registry counts changed")
    require(set(payload["mechanisms"]) == set(MECHANISM_ORDER), "mechanism registry set invalid")
    require(set(payload["cache_registries"]) == set(MECHANISM_ORDER), "cache registry set invalid")
    require(len(payload["run_matrix"]) == 18, "registered run matrix length invalid")
    for mechanism, ref in payload["cache_registries"].items():
        path_value = resolve_path(project_root, ref["path"])
        verify_input(path_value, ref["sha256"], f"{mechanism} cache registry")
    return payload


def inspect_cache(
    project_root: Path,
    cache_dir: Path,
    *,
    mode: str,
    mechanism: str,
    arm: str | None,
    expected_rows: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    require(cache_dir.is_dir() and not cache_dir.is_symlink(), f"cache directory missing/symlinked: {cache_dir}")
    manifest_path = cache_dir / "cache_manifest.json"
    regular_file(manifest_path, "cache manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest.get("row_count") == expected_rows, f"{mechanism}/{mode}/{arm}: cache row count invalid")
    inventory = require_sha256(
        manifest.get("ordered_inventory_sha256"),
        f"{mechanism}/{mode}/{arm} ordered inventory",
    )
    record = {
        "path": canonical_path(project_root, cache_dir),
        "manifest_sha256": sha256_file(manifest_path),
        "ordered_inventory_sha256": inventory,
        "row_count": expected_rows,
    }
    validated = trainer_contract.validate_cache(
        {**record, "path": str(cache_dir)},
        mode=mode,
        mechanism=mechanism,
        arm=arm,
    )
    return record, validated


def build_matched_receipt(
    mechanism: str,
    base_record: Mapping[str, Any],
    base_validation: Mapping[str, Any],
    arm_record: Mapping[str, Any],
    arm_validation: Mapping[str, Any],
) -> dict[str, Any]:
    base_files = base_validation["manifest"]["files"][:ERASE_ROWS]
    arm_files = arm_validation["manifest"]["files"]
    require(len(base_files) == len(arm_files) == ERASE_ROWS, f"{mechanism}: matched receipt inventory invalid")
    items: list[dict[str, Any]] = []
    for index, (base, arm) in enumerate(zip(base_files, arm_files)):
        require(base["row_id"] == arm["row_id"], f"{mechanism}: matched row identity mismatch at {index}")
        require(
            base["prompt_embeds_sha256"] == arm["prompt_embeds_sha256"],
            f"{mechanism}: matched prompt tensors differ at {index}",
        )
        items.append(
            {
                "selected_index": index,
                "candidate_id": base["row_id"],
                "base_prompt_embeds_sha256": base["prompt_embeds_sha256"],
                "arm_prompt_embeds_sha256": arm["prompt_embeds_sha256"],
                "tensor_equal": True,
            }
        )
    return {
        "protocol_version": PROTOCOL_VERSION,
        "status": "passed",
        "mechanism": mechanism,
        "arm": "matched_control",
        "row_count": ERASE_ROWS,
        "base_cache_manifest_sha256": base_record["manifest_sha256"],
        "arm_cache_manifest_sha256": arm_record["manifest_sha256"],
        "items": items,
    }


CacheInspector = Callable[
    [Path, Path, str, str, Any, int],
    tuple[dict[str, Any], dict[str, Any]],
]


def default_cache_inspector(
    project_root: Path,
    cache_dir: Path,
    mode: str,
    mechanism: str,
    arm: str | None,
    expected_rows: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    return inspect_cache(
        project_root,
        cache_dir,
        mode=mode,
        mechanism=mechanism,
        arm=arm,
        expected_rows=expected_rows,
    )


def finalize_runs(
    *,
    project_root: Path,
    input_registry_path: Path,
    input_registry_sha256: str,
    output_root: Path,
    cache_inspector: CacheInspector = default_cache_inspector,
) -> dict[str, Any]:
    registry = load_input_registry(
        project_root, input_registry_path, input_registry_sha256
    )
    expected_output = resolve_path(
        project_root, registry["run_spec_output_root"], strict=False
    )
    require(output_root.resolve(strict=False) == expected_output, "run-spec output root differs from input registry")
    require(not output_root.exists() and not output_root.is_symlink(), "run-spec output root must be fresh")
    cache_root = resolve_path(
        project_root, registry["cache_output_root"], strict=False
    )
    input_model = registry["inputs"]
    payloads: dict[Path, bytes] = {}
    caches: dict[str, dict[str, Any]] = {}
    for mechanism in MECHANISM_ORDER:
        mechanism_root = cache_root / mechanism
        base_record, base_validation = cache_inspector(
            project_root, mechanism_root / "base", "prepare-base", mechanism,
            None, BASE_ROWS,
        )
        teacher_record, teacher_validation = cache_inspector(
            project_root, mechanism_root / "teacher", "prepare-teacher",
            mechanism, None, ERASE_ROWS,
        )
        arms: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
        registered_arms = registry["mechanisms"][mechanism][
            "expected_cache_layout"
        ]["arms"]
        for cache_arm in registered_arms:
            arm_record, arm_validation = cache_inspector(
                project_root,
                mechanism_root / "arms" / cache_arm,
                "prepare-arm",
                mechanism,
                cache_arm,
                ERASE_ROWS,
            )
            arms[cache_arm] = (arm_record, arm_validation)
        caches[mechanism] = {
            "base": (base_record, base_validation),
            "teacher": (teacher_record, teacher_validation),
            "arms": arms,
        }

    run_spec_refs: dict[str, Any] = {}
    for run_index, run in enumerate(registry["run_matrix"]):
        mechanism = run["mechanism"]
        external_arm = run["arm"]
        require(external_arm in ARMS, f"run arm invalid: {external_arm}")
        cache_arm = CACHE_ARM[external_arm]
        require(cache_arm in caches[mechanism]["arms"], f"cache missing for {run['run_id']}")
        base_record, base_validation = caches[mechanism]["base"]
        teacher_record, _teacher_validation = caches[mechanism]["teacher"]
        arm_record, arm_validation = caches[mechanism]["arms"][cache_arm]
        receipt_ref: dict[str, Any] | None = None
        if external_arm == "matched_control":
            receipt = build_matched_receipt(
                mechanism,
                base_record,
                base_validation,
                arm_record,
                arm_validation,
            )
            receipt_path = (
                output_root / "matched_receipts"
                / f"{MECHANISM_ORDER.index(mechanism):02d}_{mechanism}.json"
            )
            receipt_raw = canonical_json_bytes(receipt)
            payloads[receipt_path] = receipt_raw
            receipt_ref = artifact_ref_bytes(
                project_root, receipt_path, receipt_raw, ERASE_ROWS
            )
        training_output = resolve_path(
            project_root, run["output_dir"], strict=False
        )
        require(
            not training_output.exists() and not training_output.is_symlink(),
            f"training output must be fresh: {training_output}",
        )
        spec = {
            "schema_version": 1,
            "protocol": RUN_SPEC_PROTOCOL,
            "protocol_version": PROTOCOL_VERSION,
            "status": "frozen",
            "run_id": run["run_id"],
            "mechanism": mechanism,
            "arm": external_arm,
            "base_cache": base_record,
            "teacher_cache": teacher_record,
            "arm_cache": arm_record,
            "model_root": input_model["model_root"],
            "model_inventory": input_model["model_inventory"],
            "runtime_root": input_model["runtime_root"],
            "runtime_registry": input_model["runtime_registry"],
            "python_executable": input_model["python_executable"],
            "training_config": trainer_contract.EXPECTED_CONFIG,
            "output_dir": canonical_path(project_root, training_output),
            "matched_tensor_equality_receipt": receipt_ref,
        }
        spec_path = output_root / "run_specs" / f"{run_index:02d}_{run['run_id']}.json"
        spec_raw = canonical_json_bytes(spec)
        payloads[spec_path] = spec_raw
        run_spec_refs[run["run_id"]] = artifact_ref_bytes(
            project_root, spec_path, spec_raw, 1
        )
    require(len(run_spec_refs) == 18, "run-spec count must be 18")
    final_registry = {
        "schema_version": 1,
        "protocol": BUILD_PROTOCOL,
        "protocol_version": PROTOCOL_VERSION,
        "status": FINAL_STATUS,
        "fresh_only": True,
        "input_registry": artifact_ref(
            project_root, input_registry_path, 1
        ),
        "cache_output_root": canonical_path(project_root, cache_root),
        "run_specs": run_spec_refs,
        "run_spec_count": 18,
        "matched_receipt_count": 7,
    }
    final_registry_path = output_root / "run_spec_registry.json"
    payloads[final_registry_path] = canonical_json_bytes(final_registry)
    _publish_fresh(output_root, payloads)
    try:
        for ref in run_spec_refs.values():
            path = resolve_path(project_root, ref["path"])
            trainer_contract.load_run_spec(
                project_root, path, ref["sha256"]
            )
    except BaseException:
        # This root was created by this invocation and has not been handed to
        # a trainer yet.  A failed post-publication contract check must not
        # leave a seemingly frozen run-spec package behind.
        shutil.rmtree(output_root)
        raise
    return {
        **final_registry,
        "registry": canonical_path(project_root, final_registry_path),
        "registry_sha256": sha256_file(final_registry_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build-inputs")
    for name in (
        "selected-targets", "ontology-registry", "run-matrix",
        "preserve-manifest", "model-inventory", "runtime-registry",
    ):
        build.add_argument(f"--{name}", type=Path, required=True)
        build.add_argument(f"--{name}-sha256", required=True)
    build.add_argument("--preserve-media-root", type=Path, required=True)
    build.add_argument("--model-root", type=Path, required=True)
    build.add_argument("--runtime-root", type=Path, required=True)
    build.add_argument("--python-executable", type=Path, required=True)
    build.add_argument("--output-root", type=Path, required=True)
    build.add_argument("--cache-output-root", type=Path, required=True)
    build.add_argument("--run-spec-output-root", type=Path, required=True)

    finalize = subparsers.add_parser("finalize-runs")
    finalize.add_argument("--input-registry", type=Path, required=True)
    finalize.add_argument("--input-registry-sha256", required=True)
    finalize.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        project_root = args.project_root.resolve(strict=True)
        require(project_root.is_dir(), "project root is not a directory")
        if args.command == "build-inputs":
            input_specs = (
                ("selected_targets", "selected_targets_sha256", "selected targets"),
                ("ontology_registry", "ontology_registry_sha256", "ontology registry"),
                ("run_matrix", "run_matrix_sha256", "run matrix"),
                ("preserve_manifest", "preserve_manifest_sha256", "preserve manifest"),
                ("model_inventory", "model_inventory_sha256", "model inventory"),
                ("runtime_registry", "runtime_registry_sha256", "runtime registry"),
            )
            resolved: dict[str, Path] = {}
            for path_name, hash_name, label in input_specs:
                path = resolve_path(project_root, getattr(args, path_name))
                verify_input(path, getattr(args, hash_name), label)
                resolved[path_name] = path
            result = build_inputs(
                project_root=project_root,
                selected_targets=resolved["selected_targets"],
                ontology_registry=resolved["ontology_registry"],
                run_matrix=resolved["run_matrix"],
                preserve_manifest=resolved["preserve_manifest"],
                preserve_media_root=resolve_path(project_root, args.preserve_media_root),
                model_root=resolve_path(project_root, args.model_root),
                model_inventory=resolved["model_inventory"],
                runtime_root=resolve_path(project_root, args.runtime_root),
                runtime_registry=resolved["runtime_registry"],
                python_executable=resolve_path(project_root, args.python_executable),
                output_root=resolve_path(project_root, args.output_root, strict=False),
                cache_output_root=resolve_path(
                    project_root, args.cache_output_root, strict=False
                ),
                run_spec_output_root=resolve_path(
                    project_root, args.run_spec_output_root, strict=False
                ),
            )
        else:
            result = finalize_runs(
                project_root=project_root,
                input_registry_path=resolve_path(
                    project_root, args.input_registry
                ),
                input_registry_sha256=args.input_registry_sha256,
                output_root=resolve_path(
                    project_root, args.output_root, strict=False
                ),
            )
    except (
        OSError,
        RegistryError,
        ValueError,
        RuntimeError,
        json.JSONDecodeError,
        subprocess.SubprocessError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
