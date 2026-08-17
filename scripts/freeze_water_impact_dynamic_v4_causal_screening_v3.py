#!/usr/bin/env python3
"""Validate, adjudicate, and transactionally freeze v4_dev72_v3 screening.

This entry point never selects cases and never publishes Stage 1.  The
``derive-disputes`` command is public-only.  The ``freeze`` command accepts no
free artifact paths: every input and output is derived from the registered
project/private roots and exact v3 basenames.
"""

from __future__ import annotations

import argparse
import ctypes
import csv
import errno
import hashlib
import io
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
import time
from pathlib import Path
from statistics import median
from typing import Any, Callable, Iterable, Mapping, Sequence

try:
    import water_impact_dynamic_v4_eval_protocol_v3 as protocol
    import build_water_impact_dynamic_v4_causal_candidates_v3 as candidate_builder
    import run_water_impact_dynamic_v4_causal_screening_v3 as screening_runner
except ModuleNotFoundError:  # imported as scripts.freeze_...
    from scripts import water_impact_dynamic_v4_eval_protocol_v3 as protocol
    from scripts import build_water_impact_dynamic_v4_causal_candidates_v3 as candidate_builder
    from scripts import run_water_impact_dynamic_v4_causal_screening_v3 as screening_runner


PACKAGE_PROTOCOL = screening_runner.PACKAGE_COMMITMENT_PROTOCOL
PUBLIC_PACKAGE_PROTOCOL = screening_runner.PUBLIC_PACKAGE_PROTOCOL
PRIVATE_PACKAGE_PROTOCOL = screening_runner.PRIVATE_PACKAGE_PROTOCOL
GENERATION_PROTOCOL = screening_runner.GENERATION_MANIFEST_PROTOCOL
RAW_INVENTORY_PROTOCOL = screening_runner.RAW_INVENTORY_PROTOCOL
CANDIDATE_BINDING_PROTOCOL = screening_runner.CANDIDATE_BINDING_PROTOCOL
ANONYMOUS_INVENTORY_PROTOCOL = screening_runner.ANONYMOUS_INVENTORY_PROTOCOL
COMPOSITE_INVENTORY_PROTOCOL = screening_runner.COMPOSITE_INVENTORY_PROTOCOL
FREEZE_PROTOCOL = "water_impact_dynamic_v4_causal_screening_freeze_v3"

PUBLIC_PACKAGE_DIR = "causal_original_screening_review_public_v3"
PRIVATE_PACKAGE_DIR = "causal_original_screening_review_private_v3"
BLIND_INPUT_DIR = "causal_original_screening_blind_inputs_v3"
GENERATION_DIR = "causal_original_screening_generation_v3"
FREEZE_PARENT = "causal_stage1_execution_v3"
FREEZE_DIR = "freeze"

PUBLIC_MANIFEST = screening_runner.PUBLIC_MANIFEST_BASENAME
PRIVATE_MANIFEST = screening_runner.PRIVATE_MANIFEST_BASENAME
PACKAGE_COMMITMENT = screening_runner.PACKAGE_COMMITMENT_BASENAME
TEMPLATE = screening_runner.REVIEW_TEMPLATE_BASENAME
REVIEW_A = "screening_review_a_576_v3.csv"
REVIEW_B = "screening_review_b_576_v3.csv"
DISPUTES = "screening_dispute_manifest_v3.csv"
ADJUDICATION = "screening_adjudication_v3.csv"
ANSWER_KEY = screening_runner.ANSWER_KEY_BASENAME
RAW_INVENTORY = screening_runner.RAW_INVENTORY_BASENAME
GENERATION_MANIFEST = screening_runner.GENERATION_MANIFEST_BASENAME
CANDIDATE_BINDING = screening_runner.CANDIDATE_BINDING_BASENAME
ANONYMOUS_INVENTORY = screening_runner.ANONYMOUS_INVENTORY_BASENAME
COMPOSITE_INVENTORY = screening_runner.COMPOSITE_INVENTORY_BASENAME

ELIGIBILITY_OUT = "eligibility_table_576_v3.csv"
AUDIT_OUT = "screening_adjudication_audit_v3.csv"
FREEZE_MANIFEST_OUT = "screening_freeze_manifest_v3.json"

SCORE_FIELDS = (
    "source_visibility",
    "footprint_visibility",
    "receiver",
    "quality",
    "causal_link",
)
PUBLIC_REVIEW_HEADER = (
    "review_id",
    "candidate_video_path",
    "candidate_video_sha256",
    "composite_path",
    "composite_sha256",
    *SCORE_FIELDS,
    "notes",
)
DISPUTE_HEADER = ("review_id", "field")
ADJUDICATION_HEADER = ("review_id", "field", "score", "brief_reason")
ANSWER_KEY_HEADER = (
    "review_id",
    "candidate_index",
    "case_id",
    "raw_video_sha256",
    "anonymous_video_sha256",
    "composite_sha256",
)
ELIGIBILITY_HEADER = (
    "candidate_id",
    "semantic_case_id",
    "group",
    "prompt_variant",
    *SCORE_FIELDS,
    "eligible",
)
AUDIT_HEADER = (
    "candidate_id",
    "review_id",
    "field",
    "reviewer_a",
    "reviewer_b",
    "adjudicator",
    "canonical",
)

MEDIA_EXPECTED = {
    "frame_count": 49,
    "width": 832,
    "height": 480,
    "fps_numerator": 8,
    "fps_denominator": 1,
}
REVIEW_ID_PATTERN = re.compile(r"s[0-9]{3}\Z")
PUBLICATION_MIN_SETTLE_SECONDS = 5.0
PUBLICATION_VISIBILITY_TIMEOUT_SECONDS = 30.0


def _record(path: Path, row_count: int | None) -> dict[str, Any]:
    return {
        "sha256": protocol.sha256_file(path),
        "size_bytes": path.stat().st_size,
        "row_count": row_count,
    }


def _exact_record(value: Any, label: str, row_count: int | None) -> Mapping[str, Any]:
    record = protocol.require_exact_keys(
        value, {"sha256", "size_bytes", "row_count"}, label
    )
    protocol.require(protocol.is_hex64(record["sha256"]), f"{label}: hash invalid")
    protocol.require(
        isinstance(record["size_bytes"], int)
        and not isinstance(record["size_bytes"], bool)
        and record["size_bytes"] > 0,
        f"{label}: size invalid",
    )
    protocol.require(record["row_count"] == row_count, f"{label}: row count invalid")
    return record


def _read_csv_bytes(raw: bytes, expected_header: Sequence[str], label: str) -> list[dict[str, str]]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label}: CSV is not UTF-8") from exc
    reader = csv.DictReader(io.StringIO(text, newline=""))
    protocol.require(tuple(reader.fieldnames or ()) == tuple(expected_header), f"{label}: header/order mismatch")
    rows = [dict(row) for row in reader]
    protocol.require(all(tuple(row) == tuple(expected_header) for row in rows), f"{label}: row fields changed")
    return rows


def _csv_bytes(rows: Sequence[Mapping[str, Any]], header: Sequence[str]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output,
        fieldnames=list(header),
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def _real_root(path: Path, label: str) -> Path:
    protocol.reject_forbidden_path(path)
    lexical = protocol._canonical_lexical_absolute(path)
    protocol._require_no_symlink_components(lexical)
    for component in lexical.parts:
        folded = component.casefold()
        protocol.require(
            re.search(r"(?:^|[_-])v[12](?:$|[_-])", folded) is None,
            f"{label}: v1/v2 path ancestry is forbidden",
        )
    info = os.lstat(lexical)
    protocol.require(stat.S_ISDIR(info.st_mode), f"{label}: must be a real directory")
    return lexical.resolve(strict=True)


def _open_regular_at(root_fd: int, relative: str, label: str) -> tuple[int, os.stat_result]:
    parts = Path(relative).parts
    protocol.require(parts and not Path(relative).is_absolute() and ".." not in parts, f"{label}: relative path invalid")
    descriptor = os.dup(root_fd)
    try:
        for part in parts[:-1]:
            child = os.open(
                part,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = child
        leaf = os.open(
            parts[-1],
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=descriptor,
        )
        info = os.fstat(leaf)
        protocol.require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1, f"{label}: leaf must be single-link regular file")
        return leaf, info
    finally:
        os.close(descriptor)


def _read_fd(descriptor: int) -> bytes:
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)


def _public_file_bytes(root: Path, relative: str, label: str) -> bytes:
    root_fd = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0))
    try:
        descriptor, _ = _open_regular_at(root_fd, relative, label)
        try:
            return _read_fd(descriptor)
        finally:
            os.close(descriptor)
    finally:
        os.close(root_fd)


def _private_file_bytes(private_root: Path, relative: str, label: str) -> bytes:
    _private_exact(private_root, relative)
    return _public_file_bytes(private_root, relative, label)


def _validate_public_dir_inventory(root: Path, *, reviews: bool, dispute: bool) -> None:
    expected = {
        PUBLIC_MANIFEST,
        TEMPLATE,
        ANONYMOUS_INVENTORY,
        COMPOSITE_INVENTORY,
        "media",
        "composites",
    }
    if reviews:
        expected |= {REVIEW_A, REVIEW_B}
    if dispute:
        expected.add(DISPUTES)
    actual = {entry.name for entry in os.scandir(root)}
    protocol.require(actual == expected, "public review root inventory is not exact")
    for name in ("media", "composites"):
        path = root / name
        info = os.lstat(path)
        protocol.require(stat.S_ISDIR(info.st_mode) and not path.is_symlink(), f"public {name} is not a real directory")
    protocol.require(
        {entry.name for entry in os.scandir(root / "media")}
        == {f"s{index:03d}.mp4" for index in range(protocol.CANDIDATE_COUNT)},
        "public media child inventory is not exact",
    )
    protocol.require(
        {entry.name for entry in os.scandir(root / "composites")}
        == {f"s{index:03d}.jpg" for index in range(protocol.CANDIDATE_COUNT)},
        "public composite child inventory is not exact",
    )


def _score(value: str, label: str) -> int:
    protocol.require(value in {"0", "1", "2"}, f"{label}: score must be 0, 1, or 2")
    return int(value)


def _validate_template(rows: Sequence[Mapping[str, str]]) -> None:
    protocol.require(len(rows) == protocol.CANDIDATE_COUNT, "public template must contain 576 rows")
    expected_ids = [f"s{index:03d}" for index in range(protocol.CANDIDATE_COUNT)]
    protocol.require([row["review_id"] for row in rows] == expected_ids, "public review IDs/order differ")
    for index, row in enumerate(rows):
        protocol.require(REVIEW_ID_PATTERN.fullmatch(row["review_id"]) is not None, "public review ID invalid")
        protocol.require(row["candidate_video_path"] == f"media/s{index:03d}.mp4", "anonymous media path is not canonical")
        protocol.require(row["composite_path"] == f"composites/s{index:03d}.jpg", "composite path is not canonical")
        protocol.require(
            protocol.is_hex64(row["candidate_video_sha256"])
            and protocol.is_hex64(row["composite_sha256"]),
            "public review media hash is invalid",
        )
        protocol.require(all(row[field] == "" for field in (*SCORE_FIELDS, "notes")), "public template is not blank")


def _validate_reviews(
    template: Sequence[Mapping[str, str]],
    left: Sequence[Mapping[str, str]],
    right: Sequence[Mapping[str, str]],
) -> list[dict[str, str]]:
    protocol.require(len(left) == len(right) == len(template) == protocol.CANDIDATE_COUNT, "review sheets do not cover 576 rows")
    disputes: list[dict[str, str]] = []
    for frozen, a, b in zip(template, left, right):
        protocol.require(a["review_id"] == b["review_id"] == frozen["review_id"], "review ID/order drift")
        for field in PUBLIC_REVIEW_HEADER[:5]:
            protocol.require(a[field] == b[field] == frozen[field], f"review changed public metadata: {field}")
        for reviewer, label in ((a, "review A"), (b, "review B")):
            for field in SCORE_FIELDS:
                _score(reviewer[field], f"{label}/{frozen['review_id']}/{field}")
        for field in SCORE_FIELDS:
            if a[field] != b[field]:
                disputes.append({"review_id": frozen["review_id"], "field": field})
    return disputes


def _atomic_public_csv(root: Path, name: str, rows: Sequence[Mapping[str, Any]], header: Sequence[str]) -> str:
    protocol.require(name == DISPUTES, "public output basename is not registered")
    raw = _csv_bytes(rows, header)
    root_fd = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0))
    temporary = f".{name}.tmp.{os.getpid()}"
    descriptor = -1
    published = False
    temporary_created = False
    temporary_identity: tuple[int, int] | None = None
    try:
        try:
            os.stat(name, dir_fd=root_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError(f"refusing to overwrite public dispute manifest: {name}")
        descriptor = os.open(
            temporary,
            os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=root_fd,
        )
        temporary_created = True
        opened = os.fstat(descriptor)
        temporary_identity = (opened.st_dev, opened.st_ino)
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            protocol.require(written > 0, "public dispute write made no progress")
            offset += written
        os.fsync(descriptor)
        protocol.require(_read_fd(descriptor) == raw, "public dispute writeback mismatch")
        os.link(temporary, name, src_dir_fd=root_fd, dst_dir_fd=root_fd, follow_symlinks=False)
        os.fsync(root_fd)
        published = True
        return hashlib.sha256(raw).hexdigest()
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if not published:
            try:
                output_info = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                if temporary_identity == (output_info.st_dev, output_info.st_ino):
                    os.unlink(name, dir_fd=root_fd)
        if temporary_created:
            try:
                temporary_info = os.stat(
                    temporary, dir_fd=root_fd, follow_symlinks=False
                )
            except FileNotFoundError:
                pass
            else:
                if temporary_identity == (
                    temporary_info.st_dev,
                    temporary_info.st_ino,
                ):
                    os.unlink(temporary, dir_fd=root_fd)
        os.close(root_fd)


def _parse_json_bytes(raw: bytes, label: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label}: invalid canonical JSON") from exc
    protocol.require(isinstance(payload, dict), f"{label}: root must be an object")
    return payload


def validate_public_review_root(
    root: Path, *, reviews: bool, dispute: bool
) -> tuple[
    Mapping[str, Any],
    Mapping[str, Any],
    Mapping[str, Any],
    list[dict[str, str]],
    list[dict[str, str]] | None,
    list[dict[str, str]] | None,
]:
    root = _real_root(root, "public review root")
    _validate_public_dir_inventory(root, reviews=reviews, dispute=dispute)
    manifest_raw = _public_file_bytes(root, PUBLIC_MANIFEST, "public manifest")
    manifest = _parse_json_bytes(manifest_raw, "public manifest")
    screening_runner._validate_public_package_payload(manifest)
    anonymous_raw = _public_file_bytes(
        root, ANONYMOUS_INVENTORY, "anonymous video inventory"
    )
    anonymous = screening_runner.validate_anonymous_inventory(
        _parse_json_bytes(anonymous_raw, "anonymous video inventory")
    )
    composite_raw = _public_file_bytes(
        root, COMPOSITE_INVENTORY, "composite inventory"
    )
    composite = screening_runner.validate_composite_inventory(
        _parse_json_bytes(composite_raw, "composite inventory")
    )
    protocol.require(
        manifest["anonymous_video_inventory_sha256"]
        == hashlib.sha256(anonymous_raw).hexdigest()
        and manifest["composite_inventory_sha256"]
        == hashlib.sha256(composite_raw).hexdigest(),
        "public package inventory hash binding mismatch",
    )
    review_ids = [f"s{index:03d}" for index in range(protocol.CANDIDATE_COUNT)]
    expected_order = protocol.sha256_bytes(protocol.canonical_json_bytes(review_ids))
    protocol.require(
        manifest["review_order_sha256"]
        == anonymous["review_order_sha256"]
        == composite["review_order_sha256"]
        == expected_order,
        "public review order binding mismatch",
    )
    template_raw = _public_file_bytes(root, TEMPLATE, "public template")
    protocol.require(
        hashlib.sha256(template_raw).hexdigest()
        == manifest["review_template_sha256"],
        "public template bytes differ from manifest",
    )
    template = _read_csv_bytes(template_raw, PUBLIC_REVIEW_HEADER, "public template")
    _validate_template(template)
    for index, (media_record, composite_record, template_row) in enumerate(
        zip(anonymous["videos"], composite["composites"], template)
    ):
        review_id = f"s{index:03d}"
        media_raw = _public_file_bytes(
            root, f"media/{review_id}.mp4", f"public anonymous media/{review_id}"
        )
        image_raw = _public_file_bytes(
            root,
            f"composites/{review_id}.jpg",
            f"public composite/{review_id}",
        )
        protocol.require(
            media_record["review_id"] == composite_record["review_id"] == review_id
            and media_record["sha256"] == hashlib.sha256(media_raw).hexdigest()
            and media_record["size_bytes"] == len(media_raw)
            and composite_record["sha256"] == hashlib.sha256(image_raw).hexdigest()
            and composite_record["size_bytes"] == len(image_raw),
            "public media bytes differ from inventory",
        )
        protocol.require(
            template_row["candidate_video_sha256"] == media_record["sha256"]
            and template_row["composite_sha256"] == composite_record["sha256"],
            "public review row is not bound to media inventories",
        )
    left = right = None
    if reviews:
        left = _read_csv_bytes(_public_file_bytes(root, REVIEW_A, "review A"), PUBLIC_REVIEW_HEADER, "review A")
        right = _read_csv_bytes(_public_file_bytes(root, REVIEW_B, "review B"), PUBLIC_REVIEW_HEADER, "review B")
        _validate_reviews(template, left, right)
    return manifest, anonymous, composite, template, left, right


def derive_disputes(project_root: Path, public_root: Path) -> tuple[int, str]:
    project_root = protocol.validate_project_root(project_root)
    protocol.validate_v2_public_inputs(project_root)
    code_path = project_root / protocol.CODE_REGISTRY
    code_payload = protocol.load_json(code_path, project_root=project_root)
    protocol.validate_code_registry(code_payload, project_root)
    root = _real_root(public_root, "public review root")
    root_fd = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        try:
            os.stat(DISPUTES, dir_fd=root_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError(f"refusing to overwrite public dispute manifest: {DISPUTES}")
    finally:
        os.close(root_fd)
    manifest, _, _, template, left, right = validate_public_review_root(
        root, reviews=True, dispute=False
    )
    stage0_path, _, stage0_sha = _load_stage0(project_root)
    del stage0_path
    protocol.require(
        manifest["stage0_registry_sha256"] == stage0_sha,
        "public review package is not bound to current Stage-0",
    )
    assert left is not None and right is not None
    disputes = _validate_reviews(template, left, right)
    digest = _atomic_public_csv(root, DISPUTES, disputes, DISPUTE_HEADER)
    after_manifest, _, _, after_template, after_left, after_right = (
        validate_public_review_root(root, reviews=True, dispute=True)
    )
    protocol.require(
        after_manifest == manifest
        and after_template == template
        and after_left == left
        and after_right == right
        and _read_csv_bytes(
            _public_file_bytes(root, DISPUTES, "published dispute manifest"),
            DISPUTE_HEADER,
            "published dispute manifest",
        )
        == disputes
        and protocol.sha256_file(project_root / protocol.STAGE0_REGISTRY)
        == stage0_sha,
        "public dispute inputs changed during publication",
    )
    protocol.validate_v2_public_inputs(project_root)
    code_payload = protocol.load_json(code_path, project_root=project_root)
    protocol.validate_code_registry(code_payload, project_root)
    return len(disputes), digest


def _runtime_decode(path: Path) -> Mapping[str, int]:
    decoded, _ = screening_runner._decode_video(
        path, collect_composite_frames=False
    )
    return decoded


def _private_exact(private_root: Path, relative: str, basename: str | None = None) -> Path:
    path = private_root / relative
    if basename is not None:
        protocol.require(path.name == basename, f"private basename must be exactly {basename}")
    protocol.validate_private_path(private_root, path)
    return path


def _validate_private_directory(private_root: Path, relative: str, expected: set[str]) -> Path:
    path = private_root / relative
    protocol.reject_forbidden_path(path)
    lexical = protocol._require_no_symlink_components(path)
    info = os.lstat(lexical)
    protocol.require(stat.S_ISDIR(info.st_mode) and stat.S_IMODE(info.st_mode) == 0o700, f"private directory must be mode 700: {relative}")
    protocol.require({entry.name for entry in os.scandir(lexical)} == expected, f"private directory inventory differs: {relative}")
    return lexical.resolve(strict=True)


def _load_private_json(private_root: Path, relative: str) -> tuple[Path, Mapping[str, Any]]:
    path = _private_exact(private_root, relative)
    return path, _parse_json_bytes(
        _private_file_bytes(private_root, relative, f"private JSON {relative}"),
        f"private JSON {relative}",
    )


def _verify_stage0_artifact(
    stage0: Mapping[str, Any], name: str, path: Path, expected_rows: int | None
) -> None:
    record = stage0["artifacts"].get(name)
    protocol.require(isinstance(record, Mapping), f"Stage-0 artifact missing: {name}")
    protocol.require(record["sha256"] == protocol.sha256_file(path) and record["size_bytes"] == path.stat().st_size and record["row_count"] == expected_rows, f"Stage-0 artifact bytes differ: {name}")


def _load_stage0(project_root: Path) -> tuple[Path, Mapping[str, Any], str]:
    root = protocol.validate_project_root(project_root)
    path = root / protocol.STAGE0_REGISTRY
    protocol.validate_runtime_read_path(root, path, allow_v2=False)
    payload = protocol.load_json(path, project_root=root, allow_v2=False)
    protocol.validate_commitment_registry(payload, stage=0)
    return path, payload, protocol.sha256_file(path)


def _validate_raw_inventory(
    private_root: Path,
    payload: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    *,
    decode: Callable[[Path], Mapping[str, int]],
) -> tuple[list[Mapping[str, Any]], dict[str, Path]]:
    screening_runner.validate_raw_video_inventory(payload)
    rows = payload["videos"]
    videos: dict[str, Path] = {}
    inodes: set[tuple[int, int]] = set()
    for index, (row, candidate) in enumerate(zip(rows, candidates)):
        candidate_id = str(candidate["case_id"])
        expected_path = f"{GENERATION_DIR}/videos/{row['video_name']}"
        protocol.require(
            row["case_id"] == candidate_id
            and row["index"] == index
            and Path(row["video_name"]).suffix.casefold() == ".mp4",
            "raw video candidate/order/path mismatch",
        )
        protocol.require(row["prompt_sha256"] == hashlib.sha256(str(candidate["canonical_prompt"]).encode("utf-8")).hexdigest(), "raw video prompt hash mismatch")
        video = _private_exact(private_root, expected_path)
        info = video.stat()
        inode = (info.st_dev, info.st_ino)
        raw = _private_file_bytes(private_root, expected_path, f"raw video/{candidate_id}")
        digest = hashlib.sha256(raw).hexdigest()
        protocol.require(inode not in inodes, "raw video path/inode reused")
        inodes.add(inode)
        protocol.require(row["size_bytes"] == info.st_size == len(raw) and row["sha256"] == digest, "raw video byte binding mismatch")
        protocol.require(
            {name: row[name] for name in MEDIA_EXPECTED} == MEDIA_EXPECTED
            and dict(decode(video)) == MEDIA_EXPECTED,
            "raw video decode contract mismatch",
        )
        protocol.require(
            _private_file_bytes(private_root, expected_path, f"raw video/{candidate_id}")
            == raw,
            "raw video changed during decode",
        )
        videos[candidate_id] = video
    protocol.require(len(videos) == protocol.CANDIDATE_COUNT, "raw video IDs repeat")
    return rows, videos


def _validate_generation_manifest(
    payload: Mapping[str, Any],
    *,
    stage0_sha256: str,
    candidate_manifest_sha256: str,
    raw_payload: Mapping[str, Any],
) -> None:
    screening_runner.validate_generation_manifest(payload)
    protocol.require(payload["stage0_registry_sha256"] == stage0_sha256 and payload["candidate_manifest_sha256"] == candidate_manifest_sha256, "generation manifest Stage-0/candidate binding mismatch")
    protocol.require(
        payload["videos"] == raw_payload["videos"],
        "generation manifest is not exact raw-inventory projection",
    )


def _validate_private_package_manifest(payload: Mapping[str, Any], stage0_sha256: str) -> None:
    screening_runner.validate_private_package_manifest(payload)
    protocol.require(
        payload["stage0_registry_sha256"] == stage0_sha256,
        "private package Stage-0 binding mismatch",
    )


PACKAGE_ARTIFACT_ROWS = {
    "screening_generation_manifest_576": 576,
    "screening_raw_video_inventory_576": 576,
    "screening_candidate_binding_576": 576,
    "screening_anonymous_video_inventory_576": 576,
    "screening_composite_inventory_576": 576,
    "screening_public_package_manifest_576": 576,
    "screening_private_package_manifest_576": 576,
    "screening_package_commitment": None,
    "screening_review_template_576": 576,
}


def _validate_package_commitment(payload: Mapping[str, Any], stage0_sha256: str) -> None:
    screening_runner.validate_package_commitment(payload)
    protocol.require(
        payload["stage0_registry_sha256"] == stage0_sha256,
        "package commitment Stage-0 binding mismatch",
    )


def _require_record_bytes(record: Mapping[str, Any], raw: bytes, label: str) -> None:
    protocol.require(record["sha256"] == hashlib.sha256(raw).hexdigest() and record["size_bytes"] == len(raw), f"{label}: committed bytes differ")


def _validate_registered_public_state(
    project_root: Path, stage0: Mapping[str, Any]
) -> dict[str, Any]:
    protocol.validate_v2_public_inputs(project_root)
    code_path = project_root / protocol.CODE_REGISTRY
    code_payload = protocol.load_json(code_path, project_root=project_root)
    screening_runner.authorizer.validate_code_registry_full(
        code_payload, project_root
    )
    _verify_stage0_artifact(
        stage0, "eval_code_registry", code_path, None
    )
    model_path = project_root / screening_runner.authorizer.MODEL_INVENTORY_PATH
    model_payload = protocol.load_json(model_path, project_root=project_root)
    _verify_stage0_artifact(stage0, "model_content_inventory", model_path, None)
    model_content_sha = screening_runner.authorizer._validate_model_inventory(
        model_payload, project_root
    )
    runtime_path = project_root / screening_runner.authorizer.RUNTIME_REGISTRY_PATH
    protocol.load_json(runtime_path, project_root=project_root)
    _verify_stage0_artifact(stage0, "runtime_registry", runtime_path, None)
    pending_path = project_root / protocol.STAGE0_PUBLIC
    protocol.validate_runtime_read_path(project_root, pending_path)
    return {
        "code_registry_sha256": protocol.sha256_file(code_path),
        "generator_sha256": code_payload["artifacts"]["generator"]["sha256"],
        "model_content_inventory_sha256": model_content_sha,
        "runtime_registry_sha256": protocol.sha256_file(runtime_path),
        "pending_commitment_sha256": protocol.sha256_file(pending_path),
    }


def _hash_record_from_bytes(raw: bytes) -> dict[str, Any]:
    return {"sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw)}


def _assert_file_bindings(
    private_root: Path, bindings: Sequence[tuple[str, bytes]]
) -> None:
    for relative, expected in bindings:
        protocol.require(
            _private_file_bytes(private_root, relative, f"bound file {relative}")
            == expected,
            f"bound file changed before publication: {relative}",
        )


def _validate_package(
    project_root: Path,
    private_root: Path,
    *,
    decode: Callable[[Path], Mapping[str, int]],
) -> dict[str, Any]:
    stage0_path, stage0, stage0_sha = _load_stage0(project_root)
    registered = _validate_registered_public_state(project_root, stage0)
    graph_path, graph = _load_private_json(private_root, "causal_stage0_candidate_graph_private576_v3.json")
    candidate_path, candidates_payload = _load_private_json(private_root, "causal_stage0_candidates_private576_v3.json")
    candidate_builder.validate_candidate_projection(graph, candidates_payload)
    _verify_stage0_artifact(stage0, "candidate_graph_576", graph_path, protocol.CANDIDATE_COUNT)
    _verify_stage0_artifact(stage0, "candidate_manifest_576", candidate_path, protocol.CANDIDATE_COUNT)
    selection_binding_path = _private_exact(
        private_root, "causal_selection_binding_v3.json"
    )
    _verify_stage0_artifact(
        stage0, "selection_binding", selection_binding_path, None
    )
    candidates = candidates_payload["candidates"]

    expected_root = set(screening_runner.authorizer.PRIVATE_INPUTS.values()) | {
        "causal_selection_binding_v3.json",
        screening_runner.CUDA_LOCK_BASENAME,
        GENERATION_DIR,
        PUBLIC_PACKAGE_DIR,
        PRIVATE_PACKAGE_DIR,
        BLIND_INPUT_DIR,
        FREEZE_PARENT,
    }
    protocol.require(
        {entry.name for entry in os.scandir(private_root)} == expected_root,
        "PRIVATE_V3_ROOT inventory is not exact for screening freeze",
    )
    generation_expected = {
        ".run_reservation_v3.json",
        "execution_started_v3.json",
        "generator_output_v3.log",
        "prompts.txt",
        screening_runner.GENERIC_MANIFEST_BASENAME,
        RAW_INVENTORY,
        GENERATION_MANIFEST,
        "execution_succeeded_v3.json",
        "videos",
    }
    _validate_private_directory(private_root, GENERATION_DIR, generation_expected)
    raw_path, raw_payload = _load_private_json(
        private_root, f"{GENERATION_DIR}/{RAW_INVENTORY}"
    )
    generation_path, generation_payload = _load_private_json(
        private_root, f"{GENERATION_DIR}/{GENERATION_MANIFEST}"
    )
    screening_runner.validate_raw_video_inventory(raw_payload)
    screening_runner.validate_generation_manifest(generation_payload)
    expected_video_names = {row["video_name"] for row in raw_payload["videos"]}
    _validate_private_directory(
        private_root, f"{GENERATION_DIR}/videos", expected_video_names
    )
    public_dir = _validate_private_directory(
        private_root,
        PUBLIC_PACKAGE_DIR,
        {
            PUBLIC_MANIFEST,
            TEMPLATE,
            ANONYMOUS_INVENTORY,
            COMPOSITE_INVENTORY,
            "media",
            "composites",
        },
    )
    _validate_private_directory(
        private_root,
        f"{PUBLIC_PACKAGE_DIR}/media",
        {f"s{index:03d}.mp4" for index in range(protocol.CANDIDATE_COUNT)},
    )
    _validate_private_directory(
        private_root,
        f"{PUBLIC_PACKAGE_DIR}/composites",
        {f"s{index:03d}.jpg" for index in range(protocol.CANDIDATE_COUNT)},
    )
    private_dir = _validate_private_directory(
        private_root,
        PRIVATE_PACKAGE_DIR,
        {ANSWER_KEY, CANDIDATE_BINDING, PRIVATE_MANIFEST, PACKAGE_COMMITMENT},
    )
    del private_dir

    raw_rows, source_videos = _validate_raw_inventory(
        private_root, raw_payload, candidates, decode=decode
    )
    _validate_generation_manifest(
        generation_payload,
        stage0_sha256=stage0_sha,
        candidate_manifest_sha256=protocol.sha256_file(candidate_path),
        raw_payload=raw_payload,
    )
    protocol.require(
        raw_payload["stage0_registry_sha256"] == stage0_sha
        and raw_payload["candidate_manifest_sha256"]
        == protocol.sha256_file(candidate_path)
        and raw_payload["generation_spec_sha256"]
        == stage0["artifacts"]["screening_generation_spec"]["sha256"],
        "raw video inventory provenance binding mismatch",
    )
    stage0_generation_bindings = {
        "generation_spec_sha256": "screening_generation_spec",
        "screening_seed_sha256": "screening_seed",
        "selection_binding_sha256": "selection_binding",
        "candidate_graph_sha256": "candidate_graph_576",
    }
    for manifest_field, artifact_name in stage0_generation_bindings.items():
        protocol.require(
            generation_payload[manifest_field]
            == stage0["artifacts"][artifact_name]["sha256"],
            f"generation manifest differs from Stage-0 artifact: {artifact_name}",
        )
    protocol.require(
        generation_payload["model_content_inventory_sha256"]
        == registered["model_content_inventory_sha256"]
        and generation_payload["runtime_registry_sha256"]
        == registered["runtime_registry_sha256"]
        and generation_payload["code_registry_sha256"]
        == registered["code_registry_sha256"]
        and generation_payload["generator_sha256"]
        == registered["generator_sha256"],
        "generation registry provenance differs",
    )
    support_names = {
        "cuda_lock_sha256": screening_runner.CUDA_LOCK_BASENAME,
        "run_reservation_sha256": f"{GENERATION_DIR}/.run_reservation_v3.json",
        "execution_started_sha256": f"{GENERATION_DIR}/execution_started_v3.json",
        "generator_log_sha256": f"{GENERATION_DIR}/generator_output_v3.log",
        "prompt_file_sha256": f"{GENERATION_DIR}/prompts.txt",
        "generic_generation_manifest_sha256": (
            f"{GENERATION_DIR}/{screening_runner.GENERIC_MANIFEST_BASENAME}"
        ),
        "raw_video_inventory_sha256": f"{GENERATION_DIR}/{RAW_INVENTORY}",
    }
    file_bindings: list[tuple[str, bytes]] = [
        (
            "causal_stage0_candidate_graph_private576_v3.json",
            _private_file_bytes(
                private_root,
                "causal_stage0_candidate_graph_private576_v3.json",
                "candidate graph",
            ),
        ),
        (
            "causal_stage0_candidates_private576_v3.json",
            _private_file_bytes(
                private_root,
                "causal_stage0_candidates_private576_v3.json",
                "candidate manifest",
            ),
        ),
        (
            "causal_selection_binding_v3.json",
            _private_file_bytes(
                private_root,
                "causal_selection_binding_v3.json",
                "selection binding",
            ),
        ),
    ]
    for field, relative in support_names.items():
        raw = _private_file_bytes(private_root, relative, field)
        protocol.require(
            generation_payload[field] == hashlib.sha256(raw).hexdigest(),
            f"generation support hash differs: {field}",
        )
        file_bindings.append((relative, raw))

    public_manifest, anonymous_inventory, composite_inventory, template, _, _ = validate_public_review_root(
        public_dir, reviews=False, dispute=False
    )
    protocol.require(public_manifest["stage0_registry_sha256"] == stage0_sha, "public package Stage-0 binding mismatch")
    protocol.require(
        public_manifest["generation_manifest_sha256"]
        == protocol.sha256_file(generation_path),
        "public package generation binding mismatch",
    )
    public_manifest_path = _private_exact(
        private_root, f"{PUBLIC_PACKAGE_DIR}/{PUBLIC_MANIFEST}"
    )
    template_path = _private_exact(private_root, f"{PUBLIC_PACKAGE_DIR}/{TEMPLATE}")
    answer_path = _private_exact(private_root, f"{PRIVATE_PACKAGE_DIR}/{ANSWER_KEY}")
    answer_raw = _private_file_bytes(
        private_root, f"{PRIVATE_PACKAGE_DIR}/{ANSWER_KEY}", "screening answer key"
    )
    answer = _read_csv_bytes(answer_raw, ANSWER_KEY_HEADER, "screening answer key")
    protocol.require(len(answer) == protocol.CANDIDATE_COUNT, "screening answer key must contain 576 rows")
    candidate_binding_path, candidate_binding = _load_private_json(
        private_root, f"{PRIVATE_PACKAGE_DIR}/{CANDIDATE_BINDING}"
    )
    screening_runner.validate_candidate_binding(candidate_binding)
    raw_by_id = {row["case_id"]: row for row in raw_rows}
    answer_by_review: dict[str, Mapping[str, str]] = {}
    anonymous_inodes: set[tuple[int, int]] = set()
    composite_inodes: set[tuple[int, int]] = set()
    raw_inventory_map: dict[str, dict[str, Any]] = {}
    anonymous_map: dict[str, dict[str, Any]] = {}
    composite_map: dict[str, dict[str, Any]] = {}
    for index, (key, binding, candidate, public_row, media_record, composite_record) in enumerate(
        zip(
            answer,
            candidate_binding["rows"],
            candidates,
            template,
            anonymous_inventory["videos"],
            composite_inventory["composites"],
        )
    ):
        review_id = f"s{index:03d}"
        candidate_id = str(candidate["case_id"])
        protocol.require(
            key["review_id"] == binding["review_id"] == public_row["review_id"] == review_id
            and key["candidate_index"] == str(index)
            and key["case_id"] == candidate_id
            and binding["candidate"] == candidate,
            "answer key/candidate binding mismatch",
        )
        expected_source = f"{GENERATION_DIR}/videos/{raw_by_id[candidate_id]['video_name']}"
        expected_anonymous = f"{PUBLIC_PACKAGE_DIR}/media/{review_id}.mp4"
        expected_composite = f"{PUBLIC_PACKAGE_DIR}/composites/{review_id}.jpg"
        source = source_videos[candidate_id]
        anonymous = _private_exact(private_root, expected_anonymous)
        composite = _private_exact(private_root, expected_composite)
        source_raw = _private_file_bytes(private_root, expected_source, "raw source")
        anonymous_raw = _private_file_bytes(private_root, expected_anonymous, "anonymous video")
        composite_raw = _private_file_bytes(private_root, expected_composite, "composite")
        source_record = _hash_record_from_bytes(source_raw)
        anonymous_record = _hash_record_from_bytes(anonymous_raw)
        composite_bytes_record = _hash_record_from_bytes(composite_raw)
        protocol.require(
            key["raw_video_sha256"]
            == binding["raw_video_sha256"]
            == raw_by_id[candidate_id]["sha256"]
            == source_record["sha256"],
            "answer key raw source hash mismatch",
        )
        protocol.require(
            key["anonymous_video_sha256"]
            == binding["anonymous_video_sha256"]
            == media_record["sha256"]
            == anonymous_record["sha256"]
            == source_record["sha256"],
            "anonymous copy does not byte-match raw source",
        )
        protocol.require(
            key["composite_sha256"]
            == binding["composite_sha256"]
            == composite_record["sha256"]
            == composite_bytes_record["sha256"],
            "composite answer-key hash mismatch",
        )
        protocol.require(
            media_record["size_bytes"] == anonymous_record["size_bytes"]
            and composite_record["size_bytes"] == composite_bytes_record["size_bytes"],
            "public media size binding mismatch",
        )
        protocol.require(dict(decode(anonymous)) == MEDIA_EXPECTED, "anonymous video decode contract mismatch")
        source_inode = (source.stat().st_dev, source.stat().st_ino)
        anonymous_inode = (anonymous.stat().st_dev, anonymous.stat().st_ino)
        composite_inode = (composite.stat().st_dev, composite.stat().st_ino)
        protocol.require(anonymous_inode != source_inode and anonymous_inode not in anonymous_inodes and composite_inode not in composite_inodes, "public package reuses an inode")
        anonymous_inodes.add(anonymous_inode)
        composite_inodes.add(composite_inode)
        answer_by_review[review_id] = key
        raw_inventory_map[review_id] = source_record
        anonymous_map[review_id] = anonymous_record
        composite_map[review_id] = composite_bytes_record
        file_bindings.extend(
            (
                (expected_source, source_raw),
                (expected_anonymous, anonymous_raw),
                (expected_composite, composite_raw),
            )
        )
    protocol.require(len(answer_by_review) == protocol.CANDIDATE_COUNT, "answer-key review IDs repeat")

    private_manifest_path, private_manifest = _load_private_json(
        private_root, f"{PRIVATE_PACKAGE_DIR}/{PRIVATE_MANIFEST}"
    )
    _validate_private_package_manifest(private_manifest, stage0_sha)
    expected_private_hashes = {
        "selection_binding_sha256": stage0["artifacts"]["selection_binding"]["sha256"],
        "candidate_graph_sha256": protocol.sha256_file(graph_path),
        "candidate_manifest_sha256": protocol.sha256_file(candidate_path),
        "generation_spec_sha256": stage0["artifacts"]["screening_generation_spec"]["sha256"],
        "generation_manifest_sha256": protocol.sha256_file(generation_path),
        "public_manifest_sha256": protocol.sha256_file(public_manifest_path),
        "answer_key_sha256": protocol.sha256_file(answer_path),
        "raw_video_inventory_sha256": protocol.sha256_file(raw_path),
        "review_order_sha256": public_manifest["review_order_sha256"],
        "review_template_sha256": protocol.sha256_file(template_path),
        "candidate_binding_sha256": protocol.sha256_file(candidate_binding_path),
        "anonymous_video_inventory_sha256": protocol.sha256_file(
            public_dir / ANONYMOUS_INVENTORY
        ),
        "composite_inventory_sha256": protocol.sha256_file(
            public_dir / COMPOSITE_INVENTORY
        ),
    }
    protocol.require(
        private_manifest["stage0_registry_sha256"] == stage0_sha
        and all(private_manifest[key] == value for key, value in expected_private_hashes.items())
        and private_manifest["raw_media"] == raw_inventory_map
        and private_manifest["anonymous_media"] == anonymous_map
        and private_manifest["composites"] == composite_map,
        "private package manifest byte binding mismatch",
    )
    commitment_path, commitment = _load_private_json(
        private_root, f"{PRIVATE_PACKAGE_DIR}/{PACKAGE_COMMITMENT}"
    )
    _validate_package_commitment(commitment, stage0_sha)
    success_relative = f"{GENERATION_DIR}/execution_succeeded_v3.json"
    _, success_status = _load_private_json(private_root, success_relative)
    protocol.require_exact_keys(
        success_status,
        {
            "protocol",
            "dataset_version",
            "status",
            "stage0_registry_sha256",
            "failure_phase",
            "reason_code",
            "generation_manifest_sha256",
            "package_commitment_sha256",
        },
        "screening execution success status",
    )
    protocol.require(
        success_status
        == {
            "protocol": screening_runner.STATUS_PROTOCOL,
            "dataset_version": protocol.DATASET_VERSION,
            "status": "succeeded",
            "stage0_registry_sha256": stage0_sha,
            "failure_phase": None,
            "reason_code": None,
            "generation_manifest_sha256": protocol.sha256_file(
                generation_path
            ),
            "package_commitment_sha256": protocol.sha256_file(
                commitment_path
            ),
        },
        "screening execution success status binding mismatch",
    )
    file_bindings.append(
        (
            success_relative,
            _private_file_bytes(
                private_root, success_relative, "screening execution status"
            ),
        )
    )
    artifact_paths = {
        "screening_generation_manifest_576": generation_path,
        "screening_raw_video_inventory_576": raw_path,
        "screening_candidate_binding_576": candidate_binding_path,
        "screening_anonymous_video_inventory_576": public_dir
        / ANONYMOUS_INVENTORY,
        "screening_composite_inventory_576": public_dir
        / COMPOSITE_INVENTORY,
        "screening_public_package_manifest_576": public_manifest_path,
        "screening_private_package_manifest_576": private_manifest_path,
        "screening_package_commitment": commitment_path,
        "screening_review_template_576": template_path,
    }
    expected_commitment_hashes = {
        **expected_private_hashes,
        **registered,
        "private_manifest_sha256": protocol.sha256_file(private_manifest_path),
        "cuda_lock_sha256": generation_payload["cuda_lock_sha256"],
        "run_reservation_sha256": generation_payload["run_reservation_sha256"],
        "execution_started_sha256": generation_payload["execution_started_sha256"],
        "generator_log_sha256": generation_payload["generator_log_sha256"],
        "prompt_file_sha256": generation_payload["prompt_file_sha256"],
        "generic_generation_manifest_sha256": generation_payload[
            "generic_generation_manifest_sha256"
        ],
    }
    protocol.require(
        all(commitment[key] == value for key, value in expected_commitment_hashes.items())
        and commitment["raw_media"] == raw_inventory_map
        and commitment["anonymous_media"] == anonymous_map
        and commitment["composites"] == composite_map,
        "package commitment byte/provenance binding mismatch",
    )
    for relative in (
        f"{GENERATION_DIR}/{GENERATION_MANIFEST}",
        f"{GENERATION_DIR}/{RAW_INVENTORY}",
        f"{PUBLIC_PACKAGE_DIR}/{PUBLIC_MANIFEST}",
        f"{PUBLIC_PACKAGE_DIR}/{TEMPLATE}",
        f"{PUBLIC_PACKAGE_DIR}/{ANONYMOUS_INVENTORY}",
        f"{PUBLIC_PACKAGE_DIR}/{COMPOSITE_INVENTORY}",
        f"{PRIVATE_PACKAGE_DIR}/{ANSWER_KEY}",
        f"{PRIVATE_PACKAGE_DIR}/{CANDIDATE_BINDING}",
        f"{PRIVATE_PACKAGE_DIR}/{PRIVATE_MANIFEST}",
        f"{PRIVATE_PACKAGE_DIR}/{PACKAGE_COMMITMENT}",
    ):
        file_bindings.append(
            (relative, _private_file_bytes(private_root, relative, relative))
        )
    protocol.require(protocol.sha256_file(stage0_path) == stage0_sha, "Stage-0 wrapper changed during package validation")
    artifact_bytes = {
        name: _private_file_bytes(
            private_root,
            path.relative_to(private_root).as_posix(),
            f"freeze artifact {name}",
        )
        for name, path in artifact_paths.items()
    }
    return {
        "stage0_path": stage0_path,
        "stage0": stage0,
        "stage0_sha256": stage0_sha,
        "graph_path": graph_path,
        "candidate_path": candidate_path,
        "candidates": candidates,
        "generation_path": generation_path,
        "raw_path": raw_path,
        "public_manifest_path": public_manifest_path,
        "public_manifest": public_manifest,
        "template_path": template_path,
        "template": template,
        "private_manifest_path": private_manifest_path,
        "answer_path": answer_path,
        "answer": answer,
        "answer_by_review": answer_by_review,
        "candidate_binding_path": candidate_binding_path,
        "anonymous_inventory_path": public_dir / ANONYMOUS_INVENTORY,
        "composite_inventory_path": public_dir / COMPOSITE_INVENTORY,
        "commitment_path": commitment_path,
        "commitment": commitment,
        "committed_paths": artifact_paths,
        "artifact_bytes": artifact_bytes,
        "file_bindings": file_bindings,
        "registered_public_state": registered,
    }


def _merge_reviews(
    candidates: Sequence[Mapping[str, Any]],
    template: Sequence[Mapping[str, str]],
    left: Sequence[Mapping[str, str]],
    right: Sequence[Mapping[str, str]],
    dispute_rows: Sequence[Mapping[str, str]],
    adjudication_rows: Sequence[Mapping[str, str]],
    answer_by_review: Mapping[str, Mapping[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    expected_disputes = _validate_reviews(template, left, right)
    protocol.require([dict(row) for row in dispute_rows] == expected_disputes, "dispute manifest is not exact disagreement set")
    expected_keys = {(row["review_id"], row["field"]) for row in expected_disputes}
    protocol.require(len(expected_keys) == len(expected_disputes), "dispute keys repeat")
    adjudication: dict[tuple[str, str], Mapping[str, str]] = {}
    for row in adjudication_rows:
        key = (row["review_id"], row["field"])
        protocol.require(key in expected_keys and key not in adjudication, "adjudication contains unexpected/duplicate key")
        _score(row["score"], f"adjudication/{key[0]}/{key[1]}")
        protocol.require(row["brief_reason"].strip(), "adjudication reason is blank")
        adjudication[key] = row
    protocol.require(set(adjudication) == expected_keys, "adjudication does not cover every-only disagreement")
    candidate_by_id = {str(row["case_id"]): row for row in candidates}
    left_by_id = {row["review_id"]: row for row in left}
    right_by_id = {row["review_id"]: row for row in right}
    eligibility: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    for public_row in template:
        review_id = public_row["review_id"]
        candidate_id = answer_by_review[review_id]["case_id"]
        candidate = candidate_by_id[candidate_id]
        scores: dict[str, int] = {}
        for field in SCORE_FIELDS:
            a = _score(left_by_id[review_id][field], f"review A/{review_id}/{field}")
            b = _score(right_by_id[review_id][field], f"review B/{review_id}/{field}")
            if a == b:
                canonical = a
            else:
                c = _score(adjudication[(review_id, field)]["score"], f"adjudication/{review_id}/{field}")
                canonical = int(median((a, b, c)))
                audit.append(
                    {
                        "candidate_id": candidate_id,
                        "review_id": review_id,
                        "field": field,
                        "reviewer_a": a,
                        "reviewer_b": b,
                        "adjudicator": c,
                        "canonical": canonical,
                    }
                )
            scores[field] = canonical
        eligible = (
            scores["source_visibility"] == 2
            and scores["footprint_visibility"] >= 1
            and scores["receiver"] >= 1
            and scores["quality"] >= 1
            and scores["causal_link"] == 2
        )
        eligibility.append(
            {
                "candidate_id": candidate_id,
                "semantic_case_id": candidate_id,
                "group": candidate["group"],
                "prompt_variant": candidate["prompt_variant"],
                **scores,
                "eligible": "yes" if eligible else "no",
            }
        )
    protocol.require(len(eligibility) == protocol.CANDIDATE_COUNT and len(audit) == len(expected_disputes), "canonical review output counts differ")
    return eligibility, audit


def _write_private_file(path: Path, raw: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(descriptor)
    finally:
        os.close(descriptor)


class ScreeningTerminalFailure(RuntimeError):
    """A formal one-shot screening failure whose aggregate outcome was frozen."""

    def __init__(self, payload: Mapping[str, Any]):
        super().__init__(str(payload["reason_code"]))
        self.payload = dict(payload)


def _record_bytes(raw: bytes, row_count: int | None) -> dict[str, Any]:
    return {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
        "row_count": row_count,
    }


def _rename_directory_noreplace(source: Path, target: Path) -> None:
    source_info = os.lstat(source)
    protocol.require(
        stat.S_ISDIR(source_info.st_mode) and not stat.S_ISLNK(source_info.st_mode),
        "freeze staging source must be a real directory",
    )
    source_identity = (source_info.st_dev, source_info.st_ino)
    expected_snapshot = _tree_integrity_snapshot(source)
    libc = ctypes.CDLL(None, use_errno=True)
    source_raw = os.fsencode(source)
    target_raw = os.fsencode(target)
    result: int
    if hasattr(libc, "renameat2"):
        renameat2 = libc.renameat2
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        result = renameat2(-100, source_raw, -100, target_raw, 1)
    elif hasattr(libc, "renamex_np"):
        renamex = libc.renamex_np
        renamex.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        renamex.restype = ctypes.c_int
        result = renamex(source_raw, target_raw, 0x00000004)
    else:
        raise RuntimeError("platform lacks atomic no-replace directory rename")
    if result != 0:
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise FileExistsError(error, os.strerror(error), target)
        if error in {errno.EINVAL, errno.ENOSYS, errno.EOPNOTSUPP}:
            _portable_reserved_directory_rename(source, target)
        else:
            raise OSError(error, os.strerror(error), target)
    _wait_for_stable_freeze_visibility(
        target=target,
        expected_identity=source_identity,
        expected_snapshot=expected_snapshot,
    )


def _tree_integrity_snapshot(root: Path) -> tuple[dict[str, Any], ...]:
    root_info = os.lstat(root)
    protocol.require(
        stat.S_ISDIR(root_info.st_mode) and not stat.S_ISLNK(root_info.st_mode),
        "freeze visibility root must be a real directory",
    )
    records: list[dict[str, Any]] = []
    for entry in sorted(
        root.rglob("*"), key=lambda value: value.relative_to(root).as_posix()
    ):
        info = os.lstat(entry)
        protocol.require(
            not stat.S_ISLNK(info.st_mode),
            "freeze visibility snapshot contains a symlink",
        )
        relative = entry.relative_to(root).as_posix()
        if stat.S_ISDIR(info.st_mode):
            records.append(
                {
                    "path": relative,
                    "kind": "directory",
                    "mode": stat.S_IMODE(info.st_mode),
                }
            )
        elif stat.S_ISREG(info.st_mode):
            protocol.require(
                info.st_nlink == 1,
                "freeze visibility snapshot contains a hardlink",
            )
            records.append(
                {
                    "path": relative,
                    "kind": "file",
                    "mode": stat.S_IMODE(info.st_mode),
                    "size_bytes": info.st_size,
                    "sha256": protocol.sha256_file(entry),
                }
            )
        else:
            raise ValueError("freeze visibility snapshot contains a non-regular entry")
    return tuple(records)


def _wait_for_stable_freeze_visibility(
    *,
    target: Path,
    expected_identity: tuple[int, int],
    expected_snapshot: Sequence[Mapping[str, Any]],
) -> None:
    started = time.monotonic()
    deadline = started + PUBLICATION_VISIBILITY_TIMEOUT_SECONDS
    consecutive = 0
    while time.monotonic() < deadline:
        try:
            info = os.lstat(target)
            matches = (
                stat.S_ISDIR(info.st_mode)
                and not stat.S_ISLNK(info.st_mode)
                and stat.S_IMODE(info.st_mode) == 0o700
                and (info.st_dev, info.st_ino) == expected_identity
                and _tree_integrity_snapshot(target)
                == tuple(dict(record) for record in expected_snapshot)
            )
        except (FileNotFoundError, OSError, ValueError):
            matches = False
        consecutive = consecutive + 1 if matches else 0
        if (
            consecutive >= 3
            and time.monotonic() - started >= PUBLICATION_MIN_SETTLE_SECONDS
        ):
            return
        time.sleep(0.25)
    raise OSError("published freeze did not reach stable filesystem visibility")


def _portable_reserved_directory_rename(source: Path, target: Path) -> None:
    """Publish a directory on filesystems lacking no-replace rename.

    The shared private-root mutex serializes cooperating publishers.  Reserving
    the exact target as an owned empty directory prevents an unguarded
    check-then-rename fallback from overwriting a pre-existing target.
    """

    protocol.require(
        source.parent == target.parent
        and source.name not in {"", ".", ".."}
        and target.name not in {"", ".", ".."},
        "portable freeze rename requires sibling simple paths",
    )
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    parent_fd = os.open(source.parent, directory_flags)
    source_fd = -1
    marker_fd = -1
    marker_created = False
    rename_completed = False
    marker_identity: tuple[int, int] | None = None
    try:
        source_fd = os.open(source.name, directory_flags, dir_fd=parent_fd)
        source_info = os.fstat(source_fd)
        protocol.require(
            stat.S_ISDIR(source_info.st_mode),
            "freeze staging source must be a real directory",
        )
        source_identity = (source_info.st_dev, source_info.st_ino)
        os.mkdir(target.name, mode=0o700, dir_fd=parent_fd)
        marker_created = True
        marker_fd = os.open(target.name, directory_flags, dir_fd=parent_fd)
        marker_info = os.fstat(marker_fd)
        marker_identity = (marker_info.st_dev, marker_info.st_ino)
        protocol.require(
            stat.S_ISDIR(marker_info.st_mode)
            and not stat.S_ISLNK(marker_info.st_mode)
            and stat.S_IMODE(marker_info.st_mode) == 0o700
            and os.listdir(marker_fd) == [],
            "freeze target reservation is invalid",
        )
        os.fsync(parent_fd)
        current_source = os.stat(
            source.name, dir_fd=parent_fd, follow_symlinks=False
        )
        current_marker = os.stat(
            target.name, dir_fd=parent_fd, follow_symlinks=False
        )
        protocol.require(
            (current_source.st_dev, current_source.st_ino) == source_identity
            and (current_marker.st_dev, current_marker.st_ino) == marker_identity
            and (os.fstat(source_fd).st_dev, os.fstat(source_fd).st_ino)
            == source_identity
            and (os.fstat(marker_fd).st_dev, os.fstat(marker_fd).st_ino)
            == marker_identity
            and os.listdir(marker_fd) == [],
            "freeze source/target reservation changed before rename",
        )
        os.rename(
            source.name,
            target.name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
        )
        rename_completed = True
        pinned_source = os.fstat(source_fd)
        pinned_marker = os.fstat(marker_fd)
        protocol.require(
            (pinned_source.st_dev, pinned_source.st_ino) == source_identity
            and (pinned_marker.st_dev, pinned_marker.st_ino) == marker_identity,
            "pinned freeze source/reservation inode changed during rename",
        )
    finally:
        cleanup_error: BaseException | None = None
        if marker_created and not rename_completed and marker_identity is not None:
            try:
                pinned_marker = os.fstat(marker_fd)
                try:
                    path_marker = os.stat(
                        target.name, dir_fd=parent_fd, follow_symlinks=False
                    )
                except FileNotFoundError:
                    path_marker = None
                if (
                    path_marker is not None
                    and (pinned_marker.st_dev, pinned_marker.st_ino)
                    == marker_identity
                    and (path_marker.st_dev, path_marker.st_ino)
                    == marker_identity
                ):
                    protocol.require(
                        os.listdir(marker_fd) == [],
                        "freeze reservation became nonempty; refusing cleanup",
                    )
                    os.rmdir(target.name, dir_fd=parent_fd)
            except BaseException as exc:
                cleanup_error = exc
        if marker_fd >= 0:
            os.close(marker_fd)
        if source_fd >= 0:
            os.close(source_fd)
        os.close(parent_fd)
        if cleanup_error is not None:
            raise cleanup_error


def _remove_owned_directory(path: Path, identity: tuple[int, int]) -> None:
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return
    if (
        stat.S_ISDIR(info.st_mode)
        and not stat.S_ISLNK(info.st_mode)
        and (info.st_dev, info.st_ino) == identity
    ):
        shutil.rmtree(path)


def _require_no_terminal_outcome(project_root: Path) -> None:
    for relative, label in (
        (protocol.INVALID_OUTCOME, "invalid outcome"),
        (protocol.STAGE1_REGISTRY, "Stage-1 registry"),
    ):
        path = project_root / relative
        if os.path.lexists(path):
            protocol._require_no_symlink_components(path)
            raise FileExistsError(f"{label} already exists; v3 may not retry")


def _publish_invalid_outcome(
    *,
    project_root: Path,
    stage0_sha256: str,
    failure_phase: str,
    reason_code: str,
    generation_sha256: str | None,
    commitment_sha256: str | None,
) -> Mapping[str, Any]:
    bound = {
        "stage0_registry": stage0_sha256,
        "screening_generation_manifest": generation_sha256,
        "screening_package_commitment": commitment_sha256,
        "screening_freeze_manifest": None,
        "canonical_eligibility": None,
        "selector_stderr": None,
    }
    payload = {
        "protocol": protocol.INVALID_OUTCOME_PROTOCOL,
        "dataset": protocol.DATASET,
        "dataset_version": protocol.DATASET_VERSION,
        "status": "preflight_dataset_invalid",
        "failure_phase": failure_phase,
        "reason_code": reason_code,
        "stage0_registry_sha256": stage0_sha256,
        "candidate_count": protocol.CANDIDATE_COUNT,
        "eligible_count": None,
        "cell_eligible_counts": None,
        "selector_output_created": False,
        "unit_manifest_created": False,
        "stage1_registry_created": False,
        "sealed_final36_status": "unopened",
        "bound_artifacts": bound,
    }
    protocol.validate_invalid_outcome(
        payload, expected_stage0_sha256=stage0_sha256
    )
    output = project_root / protocol.INVALID_OUTCOME
    protocol.write_json_exclusive_atomic(output, payload, mode=0o644)
    observed = protocol.load_json(output, project_root=project_root)
    protocol.require(observed == payload, "invalid outcome publication mismatch")
    return payload


def _validate_freeze_manifest(payload: Mapping[str, Any]) -> None:
    protocol.require_exact_keys(
        payload,
        {
            "protocol",
            "dataset_version",
            "status",
            "stage0_registry_sha256",
            "package_commitment_sha256",
            "candidate_count",
            "eligible_count",
            "dispute_count",
            "artifacts",
        },
        "screening freeze manifest",
    )
    protocol.require(payload["protocol"] == FREEZE_PROTOCOL and payload["dataset_version"] == protocol.DATASET_VERSION and payload["status"] == "frozen_before_selection", "freeze manifest protocol/status mismatch")
    protocol.require(payload["candidate_count"] == protocol.CANDIDATE_COUNT, "freeze candidate count invalid")
    protocol.require(isinstance(payload["eligible_count"], int) and not isinstance(payload["eligible_count"], bool) and 0 <= payload["eligible_count"] <= protocol.CANDIDATE_COUNT, "freeze eligible count invalid")
    protocol.require(isinstance(payload["dispute_count"], int) and not isinstance(payload["dispute_count"], bool) and 0 <= payload["dispute_count"] <= 2880, "freeze dispute count invalid")
    protocol.require(protocol.is_hex64(payload["stage0_registry_sha256"]) and protocol.is_hex64(payload["package_commitment_sha256"]), "freeze root hash invalid")
    expected = {
        "screening_generation_manifest_576": 576,
        "screening_raw_video_inventory_576": 576,
        "screening_candidate_binding_576": 576,
        "screening_anonymous_video_inventory_576": 576,
        "screening_composite_inventory_576": 576,
        "screening_public_package_manifest_576": 576,
        "screening_private_package_manifest_576": 576,
        "screening_package_commitment": None,
        "screening_review_template_576": 576,
        "screening_review_a_576": 576,
        "screening_review_b_576": 576,
        "screening_dispute_manifest": payload["dispute_count"],
        "screening_adjudication": payload["dispute_count"],
        "screening_adjudication_audit": payload["dispute_count"],
        "eligibility_table_576": 576,
    }
    artifacts = payload["artifacts"]
    protocol.require(isinstance(artifacts, dict) and set(artifacts) == set(expected), "freeze artifact inventory differs")
    for name, rows in expected.items():
        _exact_record(artifacts[name], f"freeze/{name}", rows)


def _freeze_transaction(
    *,
    project_root: Path,
    private_root: Path,
    package: Mapping[str, Any],
    left: Sequence[Mapping[str, str]],
    right: Sequence[Mapping[str, str]],
    dispute_rows: Sequence[Mapping[str, str]],
    adjudication_rows: Sequence[Mapping[str, str]],
    blind_bytes: Mapping[str, bytes],
) -> Mapping[str, Any]:
    eligibility, audit = _merge_reviews(
        package["candidates"],
        package["template"],
        left,
        right,
        dispute_rows,
        adjudication_rows,
        package["answer_by_review"],
    )
    eligibility_raw = _csv_bytes(eligibility, ELIGIBILITY_HEADER)
    audit_raw = _csv_bytes(audit, AUDIT_HEADER)
    execution_parent = private_root / FREEZE_PARENT
    protocol.reject_forbidden_path(execution_parent)
    execution_parent = protocol._require_no_symlink_components(execution_parent)
    info = os.lstat(execution_parent)
    protocol.require(
        stat.S_ISDIR(info.st_mode) and stat.S_IMODE(info.st_mode) == 0o700,
        "freeze execution parent must be mode 700",
    )
    protocol.require(
        not any(os.scandir(execution_parent)),
        "freeze execution parent must be empty",
    )
    final = execution_parent / FREEZE_DIR
    protocol.require(not os.path.lexists(final), "freeze directory already exists")
    temporary = Path(tempfile.mkdtemp(prefix=".freeze-v3-", dir=execution_parent))
    os.chmod(temporary, 0o700)
    temporary_info = os.lstat(temporary)
    owned = (temporary_info.st_dev, temporary_info.st_ino)
    published = False
    try:
        eligibility_path = temporary / ELIGIBILITY_OUT
        audit_path = temporary / AUDIT_OUT
        _write_private_file(eligibility_path, eligibility_raw)
        _write_private_file(audit_path, audit_raw)
        dispute_count = len(dispute_rows)
        artifacts = {
            name: _record_bytes(raw, PACKAGE_ARTIFACT_ROWS[name])
            for name, raw in package["artifact_bytes"].items()
        }
        review_rows = {
            "screening_review_a_576": 576,
            "screening_review_b_576": 576,
            "screening_dispute_manifest": dispute_count,
            "screening_adjudication": dispute_count,
        }
        for name, rows in review_rows.items():
            artifacts[name] = _record_bytes(blind_bytes[name], rows)
        artifacts["screening_adjudication_audit"] = _record_bytes(
            audit_raw, dispute_count
        )
        artifacts["eligibility_table_576"] = _record_bytes(
            eligibility_raw, protocol.CANDIDATE_COUNT
        )
        commitment_digest = artifacts["screening_package_commitment"][
            "sha256"
        ]
        freeze_payload = {
            "protocol": FREEZE_PROTOCOL,
            "dataset_version": protocol.DATASET_VERSION,
            "status": "frozen_before_selection",
            "stage0_registry_sha256": package["stage0_sha256"],
            "package_commitment_sha256": commitment_digest,
            "candidate_count": protocol.CANDIDATE_COUNT,
            "eligible_count": sum(
                row["eligible"] == "yes" for row in eligibility
            ),
            "dispute_count": dispute_count,
            "artifacts": artifacts,
        }
        _validate_freeze_manifest(freeze_payload)
        freeze_raw = protocol.canonical_json_bytes(freeze_payload)
        freeze_manifest_path = temporary / FREEZE_MANIFEST_OUT
        _write_private_file(freeze_manifest_path, freeze_raw)
        protocol.require(
            eligibility_path.read_bytes() == eligibility_raw
            and audit_path.read_bytes() == audit_raw
            and freeze_manifest_path.read_bytes() == freeze_raw,
            "transaction output bytes changed",
        )
        _assert_file_bindings(private_root, package["file_bindings"])
        for basename, raw in (
            (REVIEW_A, blind_bytes["screening_review_a_576"]),
            (REVIEW_B, blind_bytes["screening_review_b_576"]),
            (DISPUTES, blind_bytes["screening_dispute_manifest"]),
            (ADJUDICATION, blind_bytes["screening_adjudication"]),
        ):
            protocol.require(
                _private_file_bytes(
                    private_root,
                    f"{BLIND_INPUT_DIR}/{basename}",
                    f"blind input {basename}",
                )
                == raw,
                f"blind input changed before publication: {basename}",
            )
        protocol.require(
            protocol.sha256_file(package["stage0_path"])
            == package["stage0_sha256"],
            "Stage-0 wrapper changed before freeze publication",
        )
        protocol.require(
            _validate_registered_public_state(project_root, package["stage0"])
            == package["registered_public_state"],
            "registered public state changed before freeze publication",
        )
        _require_no_terminal_outcome(project_root)
        protocol.require(
            {entry.name for entry in os.scandir(temporary)}
            == {ELIGIBILITY_OUT, AUDIT_OUT, FREEZE_MANIFEST_OUT},
            "transaction directory inventory differs",
        )
        for entry in os.scandir(temporary):
            item = Path(entry.path)
            item_info = os.lstat(item)
            protocol.require(
                stat.S_ISREG(item_info.st_mode)
                and item_info.st_nlink == 1
                and stat.S_IMODE(item_info.st_mode) == 0o600,
                "transaction output mode/link invalid",
            )
        staging_fd = os.open(
            temporary,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            os.fsync(staging_fd)
        finally:
            os.close(staging_fd)
        _rename_directory_noreplace(temporary, final)
        parent_fd = os.open(
            execution_parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        published = True
        return freeze_payload
    finally:
        if not published:
            _remove_owned_directory(temporary, owned)
            _remove_owned_directory(final, owned)


def _freeze_screening_locked(
    project_root: Path,
    private_root: Path,
    *,
    decode: Callable[[Path], Mapping[str, int]] = _runtime_decode,
) -> Mapping[str, Any]:
    project_root = protocol.validate_project_root(project_root)
    _require_no_terminal_outcome(project_root)
    stage0_path, _, stage0_sha = _load_stage0(project_root)
    del stage0_path
    generation_sha: str | None = None
    commitment_sha: str | None = None
    try:
        private_root = _real_root(private_root, "PRIVATE_V3_ROOT")
        root_info = os.lstat(private_root)
        protocol.require(
            stat.S_IMODE(root_info.st_mode) == 0o700,
            "PRIVATE_V3_ROOT mode must be 700",
        )
        _, generation_probe = _load_private_json(
            private_root, f"{GENERATION_DIR}/{GENERATION_MANIFEST}"
        )
        screening_runner.validate_generation_manifest(generation_probe)
        protocol.require(
            generation_probe["stage0_registry_sha256"] == stage0_sha,
            "generation manifest Stage-0 binding mismatch",
        )
        generation_sha = protocol.sha256_file(
            private_root / GENERATION_DIR / GENERATION_MANIFEST
        )
    except BaseException:
        payload = _publish_invalid_outcome(
            project_root=project_root,
            stage0_sha256=stage0_sha,
            failure_phase="original_generation",
            reason_code="screening_generation_incomplete",
            generation_sha256=None,
            commitment_sha256=None,
        )
        raise ScreeningTerminalFailure(payload) from None
    try:
        package = _validate_package(project_root, private_root, decode=decode)
        commitment_sha = hashlib.sha256(
            package["artifact_bytes"]["screening_package_commitment"]
        ).hexdigest()
    except BaseException:
        payload = _publish_invalid_outcome(
            project_root=project_root,
            stage0_sha256=stage0_sha,
            failure_phase="screening_package",
            reason_code="screening_package_integrity_failure",
            generation_sha256=generation_sha,
            commitment_sha256=None,
        )
        raise ScreeningTerminalFailure(payload) from None
    try:
        blind_path = private_root / BLIND_INPUT_DIR
        protocol._require_no_symlink_components(blind_path)
        blind_info = os.lstat(blind_path)
        protocol.require(
            stat.S_ISDIR(blind_info.st_mode)
            and stat.S_IMODE(blind_info.st_mode) == 0o700,
            "blind input directory is invalid",
        )
        blind_inventory = {entry.name for entry in os.scandir(blind_path)}
        protocol.require(
            blind_inventory
            in (
                {REVIEW_A, REVIEW_B, DISPUTES},
                {REVIEW_A, REVIEW_B, DISPUTES, ADJUDICATION},
            ),
            "blind review inventory is not exact",
        )
        review_a_raw = _private_file_bytes(
            private_root, f"{BLIND_INPUT_DIR}/{REVIEW_A}", "review A"
        )
        review_b_raw = _private_file_bytes(
            private_root, f"{BLIND_INPUT_DIR}/{REVIEW_B}", "review B"
        )
        dispute_raw = _private_file_bytes(
            private_root, f"{BLIND_INPUT_DIR}/{DISPUTES}", "dispute manifest"
        )
        left = _read_csv_bytes(review_a_raw, PUBLIC_REVIEW_HEADER, "review A")
        right = _read_csv_bytes(review_b_raw, PUBLIC_REVIEW_HEADER, "review B")
        dispute_rows = _read_csv_bytes(
            dispute_raw, DISPUTE_HEADER, "dispute manifest"
        )
        protocol.require(
            dispute_rows == _validate_reviews(package["template"], left, right),
            "dispute manifest is not exact disagreement set",
        )
    except BaseException:
        payload = _publish_invalid_outcome(
            project_root=project_root,
            stage0_sha256=stage0_sha,
            failure_phase="screening_review",
            reason_code="screening_review_coverage_failure",
            generation_sha256=generation_sha,
            commitment_sha256=commitment_sha,
        )
        raise ScreeningTerminalFailure(payload) from None
    try:
        _validate_private_directory(
            private_root,
            BLIND_INPUT_DIR,
            {REVIEW_A, REVIEW_B, DISPUTES, ADJUDICATION},
        )
        adjudication_raw = _private_file_bytes(
            private_root,
            f"{BLIND_INPUT_DIR}/{ADJUDICATION}",
            "adjudication",
        )
        adjudication_rows = _read_csv_bytes(
            adjudication_raw, ADJUDICATION_HEADER, "adjudication"
        )
        blind_bytes = {
            "screening_review_a_576": review_a_raw,
            "screening_review_b_576": review_b_raw,
            "screening_dispute_manifest": dispute_raw,
            "screening_adjudication": adjudication_raw,
        }
        return _freeze_transaction(
            project_root=project_root,
            private_root=private_root,
            package=package,
            left=left,
            right=right,
            dispute_rows=dispute_rows,
            adjudication_rows=adjudication_rows,
            blind_bytes=blind_bytes,
        )
    except BaseException:
        payload = _publish_invalid_outcome(
            project_root=project_root,
            stage0_sha256=stage0_sha,
            failure_phase="screening_freeze",
            reason_code="screening_adjudication_integrity_failure",
            generation_sha256=generation_sha,
            commitment_sha256=commitment_sha,
        )
        raise ScreeningTerminalFailure(payload) from None


def freeze_screening(
    project_root: Path,
    private_root: Path,
    *,
    decode: Callable[[Path], Mapping[str, int]] = _runtime_decode,
) -> Mapping[str, Any]:
    validated_project = protocol.validate_project_root(project_root)
    validated_private = _real_root(private_root, "PRIVATE_V3_ROOT")
    root_info = os.lstat(validated_private)
    protocol.require(
        stat.S_IMODE(root_info.st_mode) == 0o700,
        "PRIVATE_V3_ROOT mode must be 700",
    )
    with screening_runner._screening_mutex(validated_private):
        return _freeze_screening_locked(
            validated_project, validated_private, decode=decode
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    subparsers = parser.add_subparsers(dest="command", required=True)
    derive = subparsers.add_parser(
        "derive-disputes",
        help="derive the exact public disagreement set without private access",
    )
    derive.add_argument("--project-root", type=Path, required=True)
    derive.add_argument("--public-root", type=Path, required=True)
    freeze = subparsers.add_parser(
        "freeze",
        help="validate the committed package and transactionally freeze reviews",
    )
    freeze.add_argument("--project-root", type=Path, required=True)
    freeze.add_argument("--private-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "derive-disputes":
        count, digest = derive_disputes(args.project_root, args.public_root)
        print(
            protocol.canonical_json_bytes(
                {
                    "status": "public_disputes_derived",
                    "dispute_count": count,
                    "sha256": digest,
                }
            ).decode("ascii"),
            end="",
        )
        return 0
    try:
        payload = freeze_screening(args.project_root, args.private_root)
    except ScreeningTerminalFailure as exc:
        print(
            protocol.canonical_json_bytes(
                {
                    "status": "preflight_dataset_invalid",
                    "failure_phase": exc.payload["failure_phase"],
                    "reason_code": exc.payload["reason_code"],
                    "invalid_outcome_sha256": protocol.sha256_file(
                        Path(args.project_root) / protocol.INVALID_OUTCOME
                    ),
                }
            ).decode("ascii"),
            end="",
        )
        return 2
    print(
        protocol.canonical_json_bytes(
            {
                "status": "frozen_before_selection",
                "candidate_count": payload["candidate_count"],
                "eligible_count": payload["eligible_count"],
                "dispute_count": payload["dispute_count"],
                "freeze_manifest_sha256": protocol.sha256_file(
                    Path(args.private_root)
                    / FREEZE_PARENT
                    / FREEZE_DIR
                    / FREEZE_MANIFEST_OUT
                ),
            }
        ).decode("ascii"),
        end="",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
