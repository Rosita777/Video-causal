#!/usr/bin/env python3
"""Build and canonicalize the frozen 192-video capability review.

This is an infrastructure adapter between the frozen generation manifest and
the already-frozen review scorer.  It never changes the review rubric.  The
workflow is deliberately split into three fail-closed stages:

``build``
    Verify the exact generation/video inventory, render all 49 frames in the
    five registered overlapping temporal panels, and create two independently
    ordered anonymous review assignments.

``plan-adjudication``
    Verify completed reviewer-A and reviewer-B files and create a blind queue
    containing every disagreed atomic field.

``finalize``
    Verify the blind third-review file, resolve booleans by majority and
    ordinals by median, and emit the canonical adjudicated CSV in the exact
    order required by the frozen scorer.

Every formal output root must be absent.  Files are written into a fresh
partial directory and exposed with one final rename; nothing is overwritten.
"""

from __future__ import annotations

import argparse
import ctypes
import csv
import errno
import functools
import hashlib
import io
import json
import os
import re
import secrets
import shutil
import stat
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


WORKFLOW_VERSION = "causal_role_erasure_8mechanism_capability_review_workflow_v1"
MANIFEST_PROTOCOL = "causal_role_erasure_8mechanism_capability_v1"
REVIEW_PROTOCOL = "causal_role_erasure_8mechanism_capability_review_v1"
EXPECTED_ROWS = 192
EXPECTED_GENERATION_MANIFEST_SHA256 = (
    "e6075a60e1bd7650bb37e5ccfd8dceecb82eec2c2dab40520abdd098078ee769"
)
EXPECTED_CANONICAL_SHA256 = (
    "6d425076a7156aabc9695e6ab4fbe7cf85922261d1282c067fd56d176d05031d"
)
EXPECTED_REVIEW_TEMPLATE_SHA256 = (
    "286bb5a1dbe3cceb606ec46e0f69202e93af5afd4de1978e83835292b67230ce"
)
EXPECTED_REVIEW_RUBRIC_SHA256 = (
    "aa88f142891312a404832128ba4688beb1b537d1cfc70f8ccf395a840ef9405f"
)
EXPECTED_REVIEW_FREEZE_SHA256 = (
    "d3a331982d1c0a89146edcaf78fcb63fc12476299b942a5344b079f8a6deb3a8"
)

DEFAULT_CANONICAL = Path(
    "data/causal_role_erasure_8mechanism_capability_v1_manifest.canonical.json"
)
DEFAULT_REVIEW_TEMPLATE = Path(
    "data/causal_role_erasure_8mechanism_capability_v1_review_template.csv"
)
DEFAULT_REVIEW_RUBRIC = Path(
    "data/causal_role_erasure_8mechanism_capability_v1_review_rubric.json"
)
DEFAULT_REVIEW_FREEZE = Path(
    "data/causal_role_erasure_8mechanism_capability_v1_review_freeze.json"
)

PANEL_RANGES = ((0, 12), (9, 21), (18, 30), (27, 39), (36, 48))
BOOLEAN_FIELDS = ("decodable",)
ORDINAL_FIELDS = (
    "clean_prefix",
    "source_after16",
    "trigger_visible",
    "footprint_after_trigger",
    "receiver_recognizable",
    "fixed_camera",
    "quality",
)
SCORE_FIELDS = BOOLEAN_FIELDS + ORDINAL_FIELDS
FROZEN_REVIEW_FIELDS = (
    "review_id",
    "generation_id",
    "case_id",
    "mechanism",
    "source_id",
    "receiver_id",
    "prompt_style",
    "seed",
    "manifest_row_sha256",
    "video_binding_key",
    *SCORE_FIELDS,
    "reviewer_notes",
)
ASSIGNMENT_CONTEXT_FIELDS = (
    "assignment_position",
    "anonymous_review_id",
    "composite_path",
    "prompt",
    "source_object",
    "receiver",
    "receiver_clean_state",
    "trigger_definition",
    "expected_footprint",
)
ASSIGNMENT_FIELDS = ASSIGNMENT_CONTEXT_FIELDS + SCORE_FIELDS + ("reviewer_notes",)
BINDING_FIELDS = (
    "anonymous_review_id",
    "assignment_position",
    "review_id",
    "generation_id",
    "case_id",
    "mechanism",
    "source_id",
    "receiver_id",
    "prompt_style",
    "seed",
    "manifest_row_sha256",
    "video_binding_key",
    "video_path",
    "video_sha256",
    "composite_path",
    "composite_sha256",
)
ADJUDICATION_CONTEXT_FIELDS = (
    "adjudication_position",
    "adjudication_id",
    "composite_path",
    "prompt",
    "source_object",
    "receiver",
    "receiver_clean_state",
    "trigger_definition",
    "expected_footprint",
    "requested_fields",
)
ADJUDICATION_FIELDS = ADJUDICATION_CONTEXT_FIELDS + SCORE_FIELDS + ("reviewer_notes",)
ADJUDICATION_BINDING_FIELDS = (
    "adjudication_id",
    "adjudication_position",
    "review_id",
    "generation_id",
    "requested_fields",
    "composite_path",
    "composite_sha256",
    "reviewer_a_values",
    "reviewer_b_values",
)

_ACTIVE_PARTIALS: dict[Path, tuple[int, int]] = {}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def reject_sealed_path(*paths: Path) -> None:
    for path in paths:
        text = path.as_posix().casefold()
        require(
            "final36" not in text and "sealed-final" not in text,
            "sealed-final36 paths are forbidden in capability review",
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def pretty_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def regular_file(path: Path, label: str) -> None:
    reject_sealed_path(path)
    require(path.is_file() and not path.is_symlink(), f"{label} is missing or symlinked: {path}")
    resolved = path.resolve(strict=True)
    reject_sealed_path(resolved)
    require(resolved == Path(os.path.abspath(path)), f"{label} contains a symlink component: {path}")


def require_sha256(value: str, label: str) -> None:
    require(
        isinstance(value, str)
        and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
        f"{label} is not a lowercase SHA-256 digest",
    )


def file_ref(path: Path) -> dict[str, Any]:
    regular_file(path, "referenced file")
    return {"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}


def relative_file_ref(root: Path, path: Path) -> dict[str, Any]:
    """Return a ref that remains valid after a fresh partial root is renamed."""

    regular_file(path, "referenced package file")
    relative = path.relative_to(root).as_posix()
    return {"path": relative, "sha256": sha256_file(path), "size_bytes": path.stat().st_size}


def read_json_object(path: Path, label: str) -> dict[str, Any]:
    regular_file(path, label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot parse {label}: {path}") from exc
    require(isinstance(value, dict), f"{label} must be an object")
    return value


def read_json_list(path: Path, label: str) -> list[dict[str, str]]:
    regular_file(path, label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot parse {label}: {path}") from exc
    require(isinstance(value, list) and all(isinstance(row, dict) for row in value), f"{label} must be an array of objects")
    rows = [{str(key): str(item) for key, item in row.items()} for row in value]
    return rows


def read_csv_exact(path: Path, fields: Sequence[str], label: str) -> list[dict[str, str]]:
    regular_file(path, label)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        require(tuple(reader.fieldnames or ()) == tuple(fields), f"{label} columns are not exact")
        rows = list(reader)
    require(all(None not in row for row in rows), f"{label} contains overflow columns")
    return [{field: str(row[field]) for field in fields} for row in rows]


def csv_bytes(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(fields), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: str(row.get(field, "")) for field in fields})
    return stream.getvalue().encode("utf-8")


def write_bytes_exclusive(path: Path, raw: bytes, mode: int = 0o644) -> None:
    reject_sealed_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, mode)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def prepare_partial_root(output_root: Path) -> Path:
    reject_sealed_path(output_root)
    require(not output_root.exists() and not output_root.is_symlink(), f"output root already exists: {output_root}")
    parent = output_root.parent
    require(parent.is_dir() and not parent.is_symlink(), f"output parent is unsafe or missing: {parent}")
    partial = parent / f".{output_root.name}.partial-{os.getpid()}"
    require(not partial.exists() and not partial.is_symlink(), f"partial output already exists: {partial}")
    os.mkdir(partial, 0o755)
    info = os.lstat(partial)
    _ACTIVE_PARTIALS[partial] = (info.st_dev, info.st_ino)
    return partial


def _safe_cleanup_partial(path: Path) -> None:
    identity = _ACTIVE_PARTIALS.pop(path, None)
    if identity is None or not path.exists() or path.is_symlink():
        return
    info = os.lstat(path)
    if not stat.S_ISDIR(info.st_mode) or (info.st_dev, info.st_ino) != identity:
        return
    require(".partial-" in path.name and path.name.startswith("."), "refusing to clean an unrecognized partial path")
    shutil.rmtree(path)


def rollback_fresh_partials(function):
    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        before = set(_ACTIVE_PARTIALS)
        try:
            return function(*args, **kwargs)
        except BaseException:
            for path in set(_ACTIVE_PARTIALS) - before:
                _safe_cleanup_partial(path)
            raise

    return wrapped


def _rename_directory_noreplace(source: Path, destination: Path) -> None:
    """Atomically expose a directory without ever replacing a destination."""

    libc = ctypes.CDLL(None, use_errno=True)
    source_raw = os.fsencode(source)
    destination_raw = os.fsencode(destination)
    if hasattr(libc, "renameat2"):
        result = libc.renameat2(-100, source_raw, -100, destination_raw, 1)
    elif hasattr(libc, "renamex_np"):
        result = libc.renamex_np(source_raw, destination_raw, 0x00000004)
    else:  # Refuse to weaken the no-overwrite contract on an unknown platform.
        raise OSError(errno.ENOTSUP, "atomic no-replace directory rename is unavailable")
    if result != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(destination))


def _transfer_tree_exclusive(source: Path, destination: Path) -> None:
    """Move an owned tree into an owned reservation without replacing entries."""

    info = os.lstat(source)
    require(not stat.S_ISLNK(info.st_mode), "partial output contains a symlink")
    if stat.S_ISDIR(info.st_mode):
        os.mkdir(destination, stat.S_IMODE(info.st_mode))
        for child in sorted(source.iterdir(), key=lambda item: item.name):
            _transfer_tree_exclusive(child, destination / child.name)
        os.rmdir(source)
        return
    require(stat.S_ISREG(info.st_mode), "partial output contains a non-regular entry")
    try:
        os.link(source, destination, follow_symlinks=False)
    except OSError as exc:
        if exc.errno not in {errno.EPERM, errno.ENOTSUP, errno.EOPNOTSUPP, errno.EXDEV}:
            raise
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(destination, flags, stat.S_IMODE(info.st_mode))
        try:
            with source.open("rb") as source_handle, os.fdopen(descriptor, "wb", closefd=True) as destination_handle:
                shutil.copyfileobj(source_handle, destination_handle, length=1024 * 1024)
                destination_handle.flush()
                os.fsync(destination_handle.fileno())
        except BaseException:
            try:
                destination.unlink()
            except FileNotFoundError:
                pass
            raise
        require(sha256_file(source) == sha256_file(destination), "exclusive transfer changed file bytes")
    source.unlink()


def _publish_via_reserved_directory(partial: Path, output_root: Path) -> None:
    """QuarkFS-compatible fail-closed publication when rename flags are unsupported."""

    partial_info = os.lstat(partial)
    os.mkdir(output_root, stat.S_IMODE(partial_info.st_mode))
    output_identity: tuple[int, int] | None = None
    try:
        output_info = os.lstat(output_root)
        output_identity = (output_info.st_dev, output_info.st_ino)
        marker = output_root / ".incomplete"
        write_bytes_exclusive(marker, b"review package publication incomplete\n", mode=0o600)
        for child in sorted(partial.iterdir(), key=lambda item: item.name):
            _transfer_tree_exclusive(child, output_root / child.name)
        os.rmdir(partial)
        marker.unlink()
        directory_fd = os.open(output_root, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        if output_identity is not None and output_root.exists() and not output_root.is_symlink():
            current = os.lstat(output_root)
            if (current.st_dev, current.st_ino) == output_identity:
                shutil.rmtree(output_root)
        raise


def expose_partial_root(partial: Path, output_root: Path) -> None:
    identity = _ACTIVE_PARTIALS.get(partial)
    require(identity is not None, "partial output is not owned by this process")
    info = os.lstat(partial)
    require((info.st_dev, info.st_ino) == identity, "partial output identity changed")
    try:
        _rename_directory_noreplace(partial, output_root)
    except OSError as exc:
        if exc.errno not in {errno.EINVAL, errno.ENOTSUP, errno.EOPNOTSUPP, errno.EXDEV}:
            raise
        _publish_via_reserved_directory(partial, output_root)
    _ACTIVE_PARTIALS.pop(partial, None)


def deterministic_order(generation_ids: Sequence[str], pass_name: str) -> list[str]:
    require(pass_name in {"reviewer_a", "reviewer_b", "third"}, "unknown review pass")
    require(len(generation_ids) == EXPECTED_ROWS and len(set(generation_ids)) == EXPECTED_ROWS, "generation IDs are not exact")
    salt = f"{WORKFLOW_VERSION}:{pass_name}:frozen-order-v1"
    return sorted(
        generation_ids,
        key=lambda value: (hashlib.sha256(f"{salt}\0{value}".encode("utf-8")).hexdigest(), value),
    )


def panel_ranges(frame_count: int) -> tuple[tuple[int, int], ...]:
    require(frame_count == 49, "capability review requires exactly 49 frames")
    covered = {index for start, end in PANEL_RANGES for index in range(start, end + 1)}
    require(covered == set(range(49)), "frozen panel ranges do not cover all 49 frames")
    return PANEL_RANGES


def load_video_frames(path: Path) -> list[Any]:
    import av

    with av.open(str(path)) as container:
        streams = [stream for stream in container.streams if stream.type == "video"]
        require(len(streams) == 1, f"video must contain exactly one video stream: {path}")
        stream = streams[0]
        rate = stream.average_rate or stream.guessed_rate
        frames = [frame.to_image().convert("RGB") for frame in container.decode(video=stream.index)]
    require(len(frames) == 49, f"video must decode exactly 49 frames: {path}")
    require(stream.width == 832 and stream.height == 480, f"video dimensions differ from 832x480: {path}")
    require(rate is not None and Fraction(rate) == Fraction(8, 1), f"video frame rate differs from 8 fps: {path}")
    return frames


def render_full49_composite(frames: Sequence[Any], output_path: Path) -> None:
    from PIL import Image, ImageDraw

    require(len(frames) == 49, "composite input must contain exactly 49 frames")
    frame_width = 192
    frame_height = round(frames[0].height * frame_width / frames[0].width)
    columns = 7
    rows = 2
    label_height = 20
    header_height = 26
    panels = []
    for panel_index, (start, end) in enumerate(panel_ranges(len(frames))):
        panel = Image.new(
            "RGB",
            (columns * frame_width, header_height + rows * (frame_height + label_height)),
            "white",
        )
        draw = ImageDraw.Draw(panel)
        draw.text((4, 4), f"panel {panel_index + 1}: frames {start}-{end}", fill="black")
        for local_index, frame_index in enumerate(range(start, end + 1)):
            x = (local_index % columns) * frame_width
            y = header_height + (local_index // columns) * (frame_height + label_height)
            draw.text((x + 3, y + 2), f"f{frame_index:03d}", fill="black")
            panel.paste(frames[frame_index].resize((frame_width, frame_height)), (x, y + label_height))
        panels.append(panel)
    composite = Image.new("RGB", (panels[0].width, sum(panel.height for panel in panels)), "white")
    offset = 0
    for panel in panels:
        composite.paste(panel, (0, offset))
        offset += panel.height
    output_path.parent.mkdir(parents=True, exist_ok=True)
    require(not output_path.exists() and not output_path.is_symlink(), f"composite already exists: {output_path}")
    with output_path.open("xb") as handle:
        composite.save(handle, format="JPEG", quality=92, optimize=True)


def validate_frozen_inputs(
    canonical_path: Path,
    template_path: Path,
    rubric_path: Path,
    freeze_path: Path,
) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, Any]]:
    reject_sealed_path(canonical_path, template_path, rubric_path, freeze_path)
    expected = {
        canonical_path: EXPECTED_CANONICAL_SHA256,
        template_path: EXPECTED_REVIEW_TEMPLATE_SHA256,
        rubric_path: EXPECTED_REVIEW_RUBRIC_SHA256,
        freeze_path: EXPECTED_REVIEW_FREEZE_SHA256,
    }
    for path, digest in expected.items():
        regular_file(path, "frozen review input")
        require(sha256_file(path) == digest, f"frozen review input hash mismatch: {path}")
    canonical = read_json_list(canonical_path, "canonical capability manifest")
    template = read_csv_exact(template_path, FROZEN_REVIEW_FIELDS, "frozen review template")
    rubric = read_json_object(rubric_path, "frozen review rubric")
    require(len(canonical) == len(template) == EXPECTED_ROWS, "frozen review inputs must contain exactly 192 rows")
    require(rubric.get("review_protocol_version") == REVIEW_PROTOCOL, "review rubric protocol mismatch")
    require(
        [row["generation_id"] for row in canonical] == [row["generation_id"] for row in template],
        "canonical/template generation order differs",
    )
    require(all(not row[field] for row in template for field in (*SCORE_FIELDS, "reviewer_notes")), "frozen review template is not blank")
    return canonical, template, rubric


def validate_generation_manifest(
    generation_path: Path,
    expected_sha256: str,
    canonical: Sequence[Mapping[str, str]],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    reject_sealed_path(generation_path)
    require_sha256(expected_sha256, "expected generation-manifest SHA-256")
    regular_file(generation_path, "generation manifest")
    require(sha256_file(generation_path) == expected_sha256, "generation-manifest SHA-256 mismatch")
    payload = read_json_object(generation_path, "generation manifest")
    require(payload.get("protocol_version") == MANIFEST_PROTOCOL, "generation protocol mismatch")
    require(payload.get("status") == "frozen_after_exact_media_validation", "generation manifest is not frozen")
    require(payload.get("video_binding_key") == "generation_id", "generation video binding key mismatch")
    require(payload.get("video_count") == EXPECTED_ROWS, "generation video count mismatch")
    require(payload.get("canonical_manifest_sha256") == EXPECTED_CANONICAL_SHA256, "generation canonical binding mismatch")
    items = payload.get("items")
    require(isinstance(items, list) and len(items) == EXPECTED_ROWS, "generation items are not exactly 192")
    by_id: dict[str, dict[str, Any]] = {}
    for index, (item, row) in enumerate(zip(items, canonical)):
        require(isinstance(item, dict), f"generation item {index} is not an object")
        generation_id = str(item.get("generation_id", ""))
        require(generation_id == row["generation_id"], f"generation item {index} order/ID mismatch")
        require(item.get("canonical_row_index") == index, f"generation item {index} row index mismatch")
        require(str(item.get("mechanism", "")) == row["mechanism"], f"generation item {index} mechanism mismatch")
        require(int(item.get("seed", -1)) == int(row["seed"]), f"generation item {index} seed mismatch")
        require(generation_id not in by_id, "duplicate generation ID")
        require_sha256(str(item.get("video_sha256", "")), f"video SHA for {generation_id}")
        require(
            item.get("media")
            == {"decoded_frames": 49, "fps": "8/1", "height": 480, "width": 832},
            f"generation media contract mismatch for {generation_id}",
        )
        by_id[generation_id] = item
    return payload, by_id


def _composite_name(generation_id: str) -> str:
    digest = hashlib.sha256(
        f"{WORKFLOW_VERSION}:anonymous-composite\0{generation_id}".encode("utf-8")
    ).hexdigest()
    return f"c_{digest[:24]}.jpg"


def reviewer_instructions_payload(rubric: Mapping[str, Any]) -> dict[str, Any]:
    """Publish scoring semantics without exposing IDs, grouping, or gate counts."""

    return {
        "schema_version": 1,
        "review_protocol_version": rubric["review_protocol_version"],
        "task": "independent_full_49_frame_atomic_review",
        "independence": {
            "do_not_access_other_reviewer_assignment_or_scores": True,
            "do_not_access_private_bindings_or_canonical_manifest": True,
            "prompt_text_is_context_not_visual_evidence": True,
        },
        "video_contract": rubric["video_contract"],
        "atomic_fields": rubric["atomic_fields"],
        "eligibility_operator": rubric["eligibility"]["operator"],
        "required_atomic_values": rubric["eligibility"]["required_values"],
        "notes_policy": "reviewer_notes are private audit evidence and never enter the gate",
    }


def _assignment_rows(
    pass_name: str,
    order: Sequence[str],
    canonical_by_id: Mapping[str, Mapping[str, str]],
    template_by_id: Mapping[str, Mapping[str, str]],
    rubric: Mapping[str, Any],
    video_by_id: Mapping[str, Mapping[str, Any]],
    composite_by_id: Mapping[str, Mapping[str, str]],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    assignments: list[dict[str, str]] = []
    bindings: list[dict[str, str]] = []
    prefix = "a" if pass_name == "reviewer_a" else "b"
    mechanisms = rubric["mechanisms"]
    for position, generation_id in enumerate(order):
        canonical = canonical_by_id[generation_id]
        template = template_by_id[generation_id]
        video = video_by_id[generation_id]
        composite = composite_by_id[generation_id]
        anonymous_id = f"{prefix}{position:03d}"
        context = {
            "assignment_position": str(position),
            "anonymous_review_id": anonymous_id,
            "composite_path": composite["path"],
            "prompt": canonical["prompt"],
            "source_object": canonical["source_object"],
            "receiver": canonical["receiver"],
            "receiver_clean_state": canonical["receiver_clean_state"],
            "trigger_definition": str(mechanisms[canonical["mechanism"]]["trigger_definition"]),
            "expected_footprint": canonical["expected_footprint"],
            **{field: "" for field in SCORE_FIELDS},
            "reviewer_notes": "",
        }
        assignments.append(context)
        bindings.append(
            {
                "anonymous_review_id": anonymous_id,
                "assignment_position": str(position),
                **{field: template[field] for field in FROZEN_REVIEW_FIELDS[:10]},
                "video_path": str(video["video_path"]),
                "video_sha256": str(video["video_sha256"]),
                "composite_path": composite["path"],
                "composite_sha256": composite["sha256"],
            }
        )
    return assignments, bindings


@rollback_fresh_partials
def build_package(
    *,
    project_root: Path,
    generation_manifest: Path,
    expected_generation_sha256: str,
    canonical_path: Path,
    template_path: Path,
    rubric_path: Path,
    freeze_path: Path,
    output_root: Path,
    frame_loader: Callable[[Path], list[Any]] = load_video_frames,
    composite_renderer: Callable[[Sequence[Any], Path], None] = render_full49_composite,
) -> dict[str, Any]:
    canonical, template, rubric = validate_frozen_inputs(
        canonical_path, template_path, rubric_path, freeze_path
    )
    generation_payload, video_by_id = validate_generation_manifest(
        generation_manifest, expected_generation_sha256, canonical
    )
    partial = prepare_partial_root(output_root)
    public = partial / "public"
    private = partial / "private"
    public.mkdir()
    private.mkdir(mode=0o700)
    os.chmod(private, 0o700)
    composites = private / "composites"
    composites.mkdir(mode=0o700)
    os.chmod(composites, 0o700)

    composite_by_id: dict[str, dict[str, str]] = {}
    for row in canonical:
        generation_id = row["generation_id"]
        item = video_by_id[generation_id]
        video_path = Path(str(item["video_path"]))
        reject_sealed_path(video_path)
        regular_file(video_path, f"generated video {generation_id}")
        require(video_path.resolve(strict=True) == video_path, f"video path contains a symlink component: {generation_id}")
        require(video_path.stat().st_size == int(item["size_bytes"]), f"video size drift: {generation_id}")
        require(sha256_file(video_path) == item["video_sha256"], f"video hash drift: {generation_id}")
        frames = frame_loader(video_path)
        relative = Path("private/composites") / _composite_name(generation_id)
        composite_path = partial / relative
        composite_renderer(frames, composite_path)
        regular_file(composite_path, f"composite {generation_id}")
        require(sha256_file(video_path) == item["video_sha256"], f"video changed while rendering: {generation_id}")
        composite_by_id[generation_id] = {
            "path": relative.as_posix(),
            "sha256": sha256_file(composite_path),
        }

    generation_ids = [row["generation_id"] for row in canonical]
    canonical_by_id = {row["generation_id"]: row for row in canonical}
    template_by_id = {row["generation_id"]: row for row in template}
    artifacts: dict[str, dict[str, Any]] = {}
    orders: dict[str, list[str]] = {}
    for pass_name in ("reviewer_a", "reviewer_b"):
        order = deterministic_order(generation_ids, pass_name)
        orders[pass_name] = order
        delivery = public / pass_name
        delivery.mkdir()
        media_dir = delivery / "media"
        media_dir.mkdir()
        delivered_composites: dict[str, dict[str, str]] = {}
        prefix = "a" if pass_name == "reviewer_a" else "b"
        for generation_id in order:
            source_record = composite_by_id[generation_id]
            source_path = partial / source_record["path"]
            anonymous_name = f"{prefix}_{secrets.token_hex(16)}.jpg"
            destination = media_dir / anonymous_name
            with source_path.open("rb") as source_handle:
                write_bytes_exclusive(destination, source_handle.read())
            require(sha256_file(destination) == source_record["sha256"], "delivered composite copy differs")
            delivered_composites[generation_id] = {
                "path": destination.relative_to(partial).as_posix(),
                "sha256": source_record["sha256"],
            }
        assignments, bindings = _assignment_rows(
            pass_name,
            order,
            canonical_by_id,
            template_by_id,
            rubric,
            video_by_id,
            delivered_composites,
        )
        assignment_path = delivery / "assignment.csv"
        instructions_path = delivery / "instructions.json"
        binding_path = private / f"{pass_name}_binding.csv"
        write_bytes_exclusive(assignment_path, csv_bytes(assignments, ASSIGNMENT_FIELDS))
        write_bytes_exclusive(
            instructions_path,
            pretty_json_bytes(reviewer_instructions_payload(rubric)),
        )
        write_bytes_exclusive(binding_path, csv_bytes(bindings, BINDING_FIELDS), mode=0o600)
        artifacts[f"{pass_name}_assignment"] = relative_file_ref(partial, assignment_path)
        artifacts[f"{pass_name}_instructions"] = relative_file_ref(partial, instructions_path)
        artifacts[f"{pass_name}_binding"] = relative_file_ref(partial, binding_path)
    require(orders["reviewer_a"] != orders["reviewer_b"], "reviewer orders unexpectedly match")

    composite_inventory = [
        {
            "generation_id": generation_id,
            "path": composite_by_id[generation_id]["path"],
            "sha256": composite_by_id[generation_id]["sha256"],
            "video_sha256": str(video_by_id[generation_id]["video_sha256"]),
        }
        for generation_id in generation_ids
    ]
    inventory_path = private / "composite_inventory.json"
    write_bytes_exclusive(inventory_path, pretty_json_bytes(composite_inventory), mode=0o600)
    artifacts["composite_inventory"] = relative_file_ref(partial, inventory_path)

    manifest = {
        "schema_version": 1,
        "workflow_version": WORKFLOW_VERSION,
        "workflow_implementation": file_ref(Path(__file__).resolve()),
        "status": "committed_before_independent_review",
        "generation_manifest": file_ref(generation_manifest),
        "frozen_inputs": {
            "canonical_manifest": file_ref(canonical_path),
            "review_template": file_ref(template_path),
            "review_rubric": file_ref(rubric_path),
            "review_freeze": file_ref(freeze_path),
        },
        "video_count": EXPECTED_ROWS,
        "video_binding_key": "generation_id",
        "panel_ranges_inclusive": [list(item) for item in PANEL_RANGES],
        "all_49_frames_rendered": True,
        "reviewer_orders": {
            name: {
                "algorithm": "sha256_salted_generation_id_v1",
                "ordered_generation_ids_sha256": hashlib.sha256(canonical_json_bytes(order)).hexdigest(),
            }
            for name, order in orders.items()
        },
        "artifacts": artifacts,
        "composites": composite_inventory,
        "scope": {
            "original_only": True,
            "training_authorized": False,
            "treatment_generation_authorized": False,
        },
        "generation_status": generation_payload["status"],
    }
    manifest_path = partial / "review_package_manifest.json"
    write_bytes_exclusive(manifest_path, pretty_json_bytes(manifest))
    expose_partial_root(partial, output_root)
    return {"manifest": str(output_root / manifest_path.name), "sha256": sha256_file(output_root / manifest_path.name), **manifest}


def _resolve_ref(package_root: Path, record: Mapping[str, Any], label: str) -> Path:
    require(set(record) == {"path", "sha256", "size_bytes"}, f"{label} reference fields are not exact")
    path = Path(str(record["path"]))
    if not path.is_absolute():
        candidate = package_root / path
        path = candidate if candidate.exists() else path
    regular_file(path, label)
    require(path.stat().st_size == int(record["size_bytes"]), f"{label} size drift")
    require(sha256_file(path) == record["sha256"], f"{label} hash drift")
    return path


def verify_package(package_root: Path, expected_manifest_sha256: str | None = None) -> dict[str, Any]:
    reject_sealed_path(package_root)
    require(not (package_root / ".incomplete").exists(), "review package publication is incomplete")
    manifest_path = package_root / "review_package_manifest.json"
    if expected_manifest_sha256 is not None:
        require_sha256(expected_manifest_sha256, "expected review-package manifest SHA-256")
        regular_file(manifest_path, "review package manifest")
        require(sha256_file(manifest_path) == expected_manifest_sha256, "review-package manifest SHA-256 mismatch")
    manifest = read_json_object(manifest_path, "review package manifest")
    require(manifest.get("workflow_version") == WORKFLOW_VERSION, "review package workflow mismatch")
    require(manifest.get("status") == "committed_before_independent_review", "review package status mismatch")
    require(manifest.get("video_count") == EXPECTED_ROWS, "review package video count mismatch")
    require(manifest.get("panel_ranges_inclusive") == [list(item) for item in PANEL_RANGES], "review package panel ranges mismatch")
    require(manifest.get("all_49_frames_rendered") is True, "review package does not attest all frames")
    private_root = package_root / "private"
    require(private_root.is_dir() and not private_root.is_symlink(), "review package private root is unsafe")
    require(stat.S_IMODE(os.lstat(private_root).st_mode) == 0o700, "review package private root mode is not 0700")
    implementation = _resolve_ref(
        package_root,
        manifest["workflow_implementation"],
        "review workflow implementation",
    )
    require(implementation.resolve() == Path(__file__).resolve(), "review package binds a different workflow implementation")
    _resolve_ref(package_root, manifest["generation_manifest"], "generation manifest")
    for name, record in manifest["frozen_inputs"].items():
        _resolve_ref(package_root, record, f"frozen input {name}")
    for name, record in manifest["artifacts"].items():
        _resolve_ref(package_root, record, f"review artifact {name}")
    composites = manifest.get("composites")
    require(isinstance(composites, list) and len(composites) == EXPECTED_ROWS, "review composite inventory mismatch")
    for record in composites:
        require(set(record) == {"generation_id", "path", "sha256", "video_sha256"}, "composite record fields mismatch")
        path = package_root / str(record["path"])
        regular_file(path, "review composite")
        require(sha256_file(path) == record["sha256"], "review composite hash drift")
    master_by_id = {str(record["generation_id"]): str(record["sha256"]) for record in composites}
    rubric = read_json_object(
        _resolve_ref(package_root, manifest["frozen_inputs"]["review_rubric"], "review rubric"),
        "review rubric",
    )
    delivered_sets: list[set[Path]] = []
    for pass_name in ("reviewer_a", "reviewer_b"):
        instructions = read_json_object(
            _artifact_path(package_root, manifest, f"{pass_name}_instructions"),
            f"{pass_name} public instructions",
        )
        require(instructions == reviewer_instructions_payload(rubric), f"{pass_name} public instructions drift")
        assignment = read_csv_exact(
            _artifact_path(package_root, manifest, f"{pass_name}_assignment"),
            ASSIGNMENT_FIELDS,
            f"{pass_name} assignment",
        )
        binding = read_csv_exact(
            _artifact_path(package_root, manifest, f"{pass_name}_binding"),
            BINDING_FIELDS,
            f"{pass_name} binding",
        )
        binding_path = _artifact_path(package_root, manifest, f"{pass_name}_binding")
        require(stat.S_IMODE(os.lstat(binding_path).st_mode) == 0o600, f"{pass_name} binding mode is not 0600")
        require(len(assignment) == len(binding) == EXPECTED_ROWS, f"{pass_name} delivery count mismatch")
        delivered: set[Path] = set()
        expected_media_root = (package_root / "public" / pass_name / "media").resolve(strict=True)
        for public_row, private_row in zip(assignment, binding):
            require(public_row["anonymous_review_id"] == private_row["anonymous_review_id"], f"{pass_name} delivery ID mismatch")
            require(public_row["composite_path"] == private_row["composite_path"], f"{pass_name} delivery path mismatch")
            delivered_path = package_root / private_row["composite_path"]
            regular_file(delivered_path, f"{pass_name} delivered composite")
            require(delivered_path.parent.resolve(strict=True) == expected_media_root, f"{pass_name} media escaped delivery root")
            observed = sha256_file(delivered_path)
            require(observed == private_row["composite_sha256"], f"{pass_name} delivered composite hash drift")
            require(observed == master_by_id[private_row["generation_id"]], f"{pass_name} delivered composite differs from master")
            require(delivered_path not in delivered, f"{pass_name} duplicate delivered composite path")
            delivered.add(delivered_path)
        delivered_sets.append(delivered)
    require(delivered_sets[0].isdisjoint(delivered_sets[1]), "reviewer deliveries are cross-pass linkable by path")
    return manifest


def _artifact_path(package_root: Path, manifest: Mapping[str, Any], name: str) -> Path:
    return _resolve_ref(package_root, manifest["artifacts"][name], f"review artifact {name}")


def validate_completed_review(
    completed_path: Path,
    blank_assignment_path: Path,
    label: str,
) -> list[dict[str, str]]:
    completed = read_csv_exact(completed_path, ASSIGNMENT_FIELDS, label)
    blank = read_csv_exact(blank_assignment_path, ASSIGNMENT_FIELDS, "blank assignment")
    require(len(completed) == len(blank) == EXPECTED_ROWS, f"{label} must contain exactly 192 rows")
    for index, (actual, expected) in enumerate(zip(completed, blank)):
        for field in ASSIGNMENT_CONTEXT_FIELDS:
            require(actual[field] == expected[field], f"{label} row {index} changed frozen context {field}")
        for field in BOOLEAN_FIELDS:
            require(actual[field] in {"0", "1"}, f"{label} row {index} has invalid {field}")
        for field in ORDINAL_FIELDS:
            require(actual[field] in {"0", "1", "2"}, f"{label} row {index} has invalid {field}")
    return completed


def _scores_by_generation(
    completed: Sequence[Mapping[str, str]],
    binding: Sequence[Mapping[str, str]],
) -> dict[str, dict[str, str]]:
    require(len(completed) == len(binding) == EXPECTED_ROWS, "completed/binding row count mismatch")
    output: dict[str, dict[str, str]] = {}
    for scores, bound in zip(completed, binding):
        require(scores["anonymous_review_id"] == bound["anonymous_review_id"], "review/binding anonymous ID mismatch")
        generation_id = bound["generation_id"]
        require(generation_id not in output, "duplicate generation ID in completed review")
        output[generation_id] = {field: scores[field] for field in SCORE_FIELDS}
    return output


@rollback_fresh_partials
def plan_adjudication(
    *,
    package_root: Path,
    expected_package_manifest_sha256: str,
    reviewer_a_completed: Path,
    reviewer_b_completed: Path,
    output_root: Path,
) -> dict[str, Any]:
    manifest = verify_package(package_root, expected_package_manifest_sha256)
    a_assignment = _artifact_path(package_root, manifest, "reviewer_a_assignment")
    b_assignment = _artifact_path(package_root, manifest, "reviewer_b_assignment")
    a_binding_path = _artifact_path(package_root, manifest, "reviewer_a_binding")
    b_binding_path = _artifact_path(package_root, manifest, "reviewer_b_binding")
    a_rows = validate_completed_review(reviewer_a_completed, a_assignment, "reviewer A completed review")
    b_rows = validate_completed_review(reviewer_b_completed, b_assignment, "reviewer B completed review")
    a_binding = read_csv_exact(a_binding_path, BINDING_FIELDS, "reviewer A binding")
    b_binding = read_csv_exact(b_binding_path, BINDING_FIELDS, "reviewer B binding")
    a_scores = _scores_by_generation(a_rows, a_binding)
    b_scores = _scores_by_generation(b_rows, b_binding)
    require(set(a_scores) == set(b_scores) and len(a_scores) == EXPECTED_ROWS, "A/B reviews cover different videos")

    canonical_path = _resolve_ref(package_root, manifest["frozen_inputs"]["canonical_manifest"], "canonical manifest")
    rubric_path = _resolve_ref(package_root, manifest["frozen_inputs"]["review_rubric"], "review rubric")
    canonical = read_json_list(canonical_path, "canonical manifest")
    rubric = read_json_object(rubric_path, "review rubric")
    canonical_by_id = {row["generation_id"]: row for row in canonical}
    composite_by_id = {record["generation_id"]: record for record in manifest["composites"]}

    disagreements = {
        generation_id: [field for field in SCORE_FIELDS if a_scores[generation_id][field] != b_scores[generation_id][field]]
        for generation_id in a_scores
    }
    disagreement_ids = [generation_id for generation_id, fields in disagreements.items() if fields]
    order = [generation_id for generation_id in deterministic_order(list(a_scores), "third") if generation_id in set(disagreement_ids)]
    partial = prepare_partial_root(output_root)
    public = partial / "public"
    private = partial / "private"
    public.mkdir()
    private.mkdir(mode=0o700)
    os.chmod(private, 0o700)
    third_delivery = public / "third_reviewer"
    third_delivery.mkdir()
    third_media = third_delivery / "media"
    third_media.mkdir()
    queue: list[dict[str, str]] = []
    bindings: list[dict[str, str]] = []
    for position, generation_id in enumerate(order):
        canonical_row = canonical_by_id[generation_id]
        adjudication_id = f"t{position:03d}"
        requested = ";".join(disagreements[generation_id])
        composite = composite_by_id[generation_id]
        source_composite = package_root / str(composite["path"])
        delivered_composite = third_media / f"t_{secrets.token_hex(16)}.jpg"
        with source_composite.open("rb") as source_handle:
            write_bytes_exclusive(delivered_composite, source_handle.read())
        require(sha256_file(delivered_composite) == composite["sha256"], "third-review composite copy differs")
        queue.append(
            {
                "adjudication_position": str(position),
                "adjudication_id": adjudication_id,
                "composite_path": delivered_composite.relative_to(partial).as_posix(),
                "prompt": canonical_row["prompt"],
                "source_object": canonical_row["source_object"],
                "receiver": canonical_row["receiver"],
                "receiver_clean_state": canonical_row["receiver_clean_state"],
                "trigger_definition": str(rubric["mechanisms"][canonical_row["mechanism"]]["trigger_definition"]),
                "expected_footprint": canonical_row["expected_footprint"],
                "requested_fields": requested,
                **{field: "" for field in SCORE_FIELDS},
                "reviewer_notes": "",
            }
        )
        bindings.append(
            {
                "adjudication_id": adjudication_id,
                "adjudication_position": str(position),
                "review_id": next(row["review_id"] for row in a_binding if row["generation_id"] == generation_id),
                "generation_id": generation_id,
                "requested_fields": requested,
                "composite_path": delivered_composite.relative_to(partial).as_posix(),
                "composite_sha256": str(composite["sha256"]),
                "reviewer_a_values": json.dumps(a_scores[generation_id], sort_keys=True, separators=(",", ":")),
                "reviewer_b_values": json.dumps(b_scores[generation_id], sort_keys=True, separators=(",", ":")),
            }
        )
    queue_path = third_delivery / "assignment.csv"
    instructions_path = third_delivery / "instructions.json"
    binding_path = private / "adjudication_binding.csv"
    write_bytes_exclusive(queue_path, csv_bytes(queue, ADJUDICATION_FIELDS))
    third_instructions = reviewer_instructions_payload(rubric)
    third_instructions["task"] = "blind_atomic_adjudication_for_requested_disagreement_fields_only"
    third_instructions["fill_only_requested_fields"] = True
    write_bytes_exclusive(instructions_path, pretty_json_bytes(third_instructions))
    write_bytes_exclusive(binding_path, csv_bytes(bindings, ADJUDICATION_BINDING_FIELDS), mode=0o600)
    adjudication_manifest = {
        "schema_version": 1,
        "workflow_version": WORKFLOW_VERSION,
        "workflow_implementation": file_ref(Path(__file__).resolve()),
        "status": "committed_before_blind_third_review",
        "review_package_manifest": file_ref(package_root / "review_package_manifest.json"),
        "reviewer_a_completed": file_ref(reviewer_a_completed),
        "reviewer_b_completed": file_ref(reviewer_b_completed),
        "third_assignment": relative_file_ref(partial, queue_path),
        "third_instructions": relative_file_ref(partial, instructions_path),
        "adjudication_binding": relative_file_ref(partial, binding_path),
        "videos_with_disagreement": len(order),
        "atomic_disagreement_count": sum(len(disagreements[generation_id]) for generation_id in order),
        "resolution": {"boolean": "majority_of_three", "ordinal": "median_of_three"},
    }
    manifest_path = partial / "adjudication_manifest.json"
    write_bytes_exclusive(manifest_path, pretty_json_bytes(adjudication_manifest))
    expose_partial_root(partial, output_root)
    return {"manifest": str(output_root / manifest_path.name), "sha256": sha256_file(output_root / manifest_path.name), **adjudication_manifest}


def verify_adjudication_root(
    adjudication_root: Path,
    package_root: Path,
    expected_manifest_sha256: str,
) -> dict[str, Any]:
    reject_sealed_path(adjudication_root)
    require(not (adjudication_root / ".incomplete").exists(), "adjudication publication is incomplete")
    manifest_path = adjudication_root / "adjudication_manifest.json"
    require_sha256(expected_manifest_sha256, "expected adjudication-manifest SHA-256")
    regular_file(manifest_path, "adjudication manifest")
    require(sha256_file(manifest_path) == expected_manifest_sha256, "adjudication-manifest SHA-256 mismatch")
    manifest = read_json_object(manifest_path, "adjudication manifest")
    require(manifest.get("workflow_version") == WORKFLOW_VERSION, "adjudication workflow mismatch")
    require(manifest.get("status") == "committed_before_blind_third_review", "adjudication status mismatch")
    private_root = adjudication_root / "private"
    require(private_root.is_dir() and not private_root.is_symlink(), "adjudication private root is unsafe")
    require(stat.S_IMODE(os.lstat(private_root).st_mode) == 0o700, "adjudication private root mode is not 0700")
    implementation = _resolve_ref(
        adjudication_root,
        manifest["workflow_implementation"],
        "adjudication workflow implementation",
    )
    require(implementation.resolve() == Path(__file__).resolve(), "adjudication binds a different workflow implementation")
    package_ref = manifest["review_package_manifest"]
    package_path = _resolve_ref(adjudication_root, package_ref, "bound review package manifest")
    require(package_path.resolve() == (package_root / "review_package_manifest.json").resolve(), "adjudication binds a different package")
    for name in (
        "reviewer_a_completed",
        "reviewer_b_completed",
        "third_assignment",
        "third_instructions",
        "adjudication_binding",
    ):
        _resolve_ref(adjudication_root, manifest[name], f"adjudication artifact {name}")
    binding_path = _resolve_ref(adjudication_root, manifest["adjudication_binding"], "adjudication binding")
    require(stat.S_IMODE(os.lstat(binding_path).st_mode) == 0o600, "adjudication binding mode is not 0600")
    assignment_path = _resolve_ref(adjudication_root, manifest["third_assignment"], "third assignment")
    assignment = read_csv_exact(assignment_path, ADJUDICATION_FIELDS, "third assignment")
    binding = read_csv_exact(binding_path, ADJUDICATION_BINDING_FIELDS, "adjudication binding")
    require(len(assignment) == len(binding) == int(manifest["videos_with_disagreement"]), "third delivery count mismatch")
    package_manifest = read_json_object(package_path, "bound review package manifest")
    master_by_id = {
        str(record["generation_id"]): str(record["sha256"])
        for record in package_manifest["composites"]
    }
    expected_media_root = (adjudication_root / "public/third_reviewer/media").resolve(strict=True)
    delivered: set[Path] = set()
    for public_row, private_row in zip(assignment, binding):
        require(public_row["adjudication_id"] == private_row["adjudication_id"], "third delivery ID mismatch")
        require(public_row["composite_path"] == private_row["composite_path"], "third delivery path mismatch")
        delivered_path = adjudication_root / private_row["composite_path"]
        regular_file(delivered_path, "third delivered composite")
        require(delivered_path.parent.resolve(strict=True) == expected_media_root, "third media escaped delivery root")
        observed = sha256_file(delivered_path)
        require(observed == private_row["composite_sha256"], "third delivered composite hash drift")
        require(observed == master_by_id[private_row["generation_id"]], "third delivered composite differs from master")
        require(delivered_path not in delivered, "duplicate third delivered composite path")
        delivered.add(delivered_path)
    rubric_path = _resolve_ref(
        package_root,
        package_manifest["frozen_inputs"]["review_rubric"],
        "review rubric",
    )
    rubric = read_json_object(rubric_path, "review rubric")
    expected_instructions = reviewer_instructions_payload(rubric)
    expected_instructions["task"] = "blind_atomic_adjudication_for_requested_disagreement_fields_only"
    expected_instructions["fill_only_requested_fields"] = True
    actual_instructions = read_json_object(
        _resolve_ref(adjudication_root, manifest["third_instructions"], "third instructions"),
        "third instructions",
    )
    require(actual_instructions == expected_instructions, "third public instructions drift")
    return manifest


def validate_third_completed(
    completed_path: Path,
    assignment_path: Path,
) -> list[dict[str, str]]:
    completed = read_csv_exact(completed_path, ADJUDICATION_FIELDS, "third completed review")
    blank = read_csv_exact(assignment_path, ADJUDICATION_FIELDS, "third blank assignment")
    require(len(completed) == len(blank), "third completed row count mismatch")
    for index, (actual, expected) in enumerate(zip(completed, blank)):
        for field in ADJUDICATION_CONTEXT_FIELDS:
            require(actual[field] == expected[field], f"third review row {index} changed frozen context {field}")
        requested = set(actual["requested_fields"].split(";")) if actual["requested_fields"] else set()
        require(requested and requested <= set(SCORE_FIELDS), f"third review row {index} requested fields invalid")
        for field in SCORE_FIELDS:
            if field not in requested:
                require(actual[field] == "", f"third review row {index} filled non-requested {field}")
            elif field in BOOLEAN_FIELDS:
                require(actual[field] in {"0", "1"}, f"third review row {index} invalid {field}")
            else:
                require(actual[field] in {"0", "1", "2"}, f"third review row {index} invalid {field}")
    return completed


def median_of_three(left: str, right: str, third: str) -> str:
    return str(sorted((int(left), int(right), int(third)))[1])


@rollback_fresh_partials
def finalize_reviews(
    *,
    package_root: Path,
    expected_package_manifest_sha256: str,
    adjudication_root: Path,
    expected_adjudication_manifest_sha256: str,
    third_completed: Path,
    output_root: Path,
) -> dict[str, Any]:
    package = verify_package(package_root, expected_package_manifest_sha256)
    adjudication = verify_adjudication_root(
        adjudication_root,
        package_root,
        expected_adjudication_manifest_sha256,
    )
    a_completed_path = _resolve_ref(adjudication_root, adjudication["reviewer_a_completed"], "reviewer A completed")
    b_completed_path = _resolve_ref(adjudication_root, adjudication["reviewer_b_completed"], "reviewer B completed")
    a_assignment = _artifact_path(package_root, package, "reviewer_a_assignment")
    b_assignment = _artifact_path(package_root, package, "reviewer_b_assignment")
    a_rows = validate_completed_review(a_completed_path, a_assignment, "reviewer A completed review")
    b_rows = validate_completed_review(b_completed_path, b_assignment, "reviewer B completed review")
    a_binding = read_csv_exact(_artifact_path(package_root, package, "reviewer_a_binding"), BINDING_FIELDS, "reviewer A binding")
    b_binding = read_csv_exact(_artifact_path(package_root, package, "reviewer_b_binding"), BINDING_FIELDS, "reviewer B binding")
    a_scores = _scores_by_generation(a_rows, a_binding)
    b_scores = _scores_by_generation(b_rows, b_binding)

    third_assignment = _resolve_ref(adjudication_root, adjudication["third_assignment"], "third assignment")
    third_rows = validate_third_completed(third_completed, third_assignment)
    third_binding = read_csv_exact(
        _resolve_ref(adjudication_root, adjudication["adjudication_binding"], "adjudication binding"),
        ADJUDICATION_BINDING_FIELDS,
        "adjudication binding",
    )
    require(len(third_rows) == len(third_binding), "third review/binding row count mismatch")
    third_scores: dict[str, dict[str, str]] = {}
    for scores, binding in zip(third_rows, third_binding):
        require(scores["adjudication_id"] == binding["adjudication_id"], "third review/binding ID mismatch")
        third_scores[binding["generation_id"]] = {
            field: scores[field] for field in binding["requested_fields"].split(";")
        }

    template_path = _resolve_ref(package_root, package["frozen_inputs"]["review_template"], "review template")
    template = read_csv_exact(template_path, FROZEN_REVIEW_FIELDS, "review template")
    canonical_rows: list[dict[str, str]] = []
    resolved_disagreements = 0
    for blank in template:
        generation_id = blank["generation_id"]
        output = dict(blank)
        for field in SCORE_FIELDS:
            left = a_scores[generation_id][field]
            right = b_scores[generation_id][field]
            if left == right:
                output[field] = left
            else:
                require(generation_id in third_scores and field in third_scores[generation_id], f"missing third score for {generation_id}/{field}")
                output[field] = median_of_three(left, right, third_scores[generation_id][field])
                resolved_disagreements += 1
        output["reviewer_notes"] = ""
        canonical_rows.append(output)

    require(resolved_disagreements == int(adjudication["atomic_disagreement_count"]), "resolved disagreement count mismatch")
    partial = prepare_partial_root(output_root)
    os.chmod(partial, 0o700)
    canonical_path = partial / "canonical_adjudicated.csv"
    write_bytes_exclusive(canonical_path, csv_bytes(canonical_rows, FROZEN_REVIEW_FIELDS), mode=0o600)
    generation_manifest_path = _resolve_ref(package_root, package["generation_manifest"], "generation manifest")
    registry = {
        "schema_version": 1,
        "workflow_version": WORKFLOW_VERSION,
        "workflow_implementation": file_ref(Path(__file__).resolve()),
        "status": "frozen_canonical_adjudication",
        "review_package_manifest": file_ref(package_root / "review_package_manifest.json"),
        "adjudication_manifest": file_ref(adjudication_root / "adjudication_manifest.json"),
        "reviewer_a_completed": file_ref(a_completed_path),
        "reviewer_b_completed": file_ref(b_completed_path),
        "third_completed": file_ref(third_completed),
        "canonical_adjudicated": relative_file_ref(partial, canonical_path),
        "generation_manifest": file_ref(generation_manifest_path),
        "video_binding_inventory_sha256": hashlib.sha256(
            canonical_json_bytes(
                [
                    {
                        "generation_id": record["generation_id"],
                        "video_sha256": record["video_sha256"],
                    }
                    for record in package["composites"]
                ]
            )
        ).hexdigest(),
        "row_count": EXPECTED_ROWS,
        "resolved_atomic_disagreements": resolved_disagreements,
        "training_authorized": False,
        "treatment_generation_authorized": False,
    }
    registry_path = partial / "review_run_registry.json"
    write_bytes_exclusive(registry_path, pretty_json_bytes(registry), mode=0o600)
    expose_partial_root(partial, output_root)
    return {"registry": str(output_root / registry_path.name), "sha256": sha256_file(output_root / registry_path.name), **registry}


def verify_final_root(
    final_root: Path,
    expected_registry_sha256: str,
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    reject_sealed_path(final_root)
    require(not (final_root / ".incomplete").exists(), "final review publication is incomplete")
    registry_path = final_root / "review_run_registry.json"
    require_sha256(expected_registry_sha256, "expected review-run registry SHA-256")
    regular_file(registry_path, "review-run registry")
    require(sha256_file(registry_path) == expected_registry_sha256, "review-run registry SHA-256 mismatch")
    require(final_root.is_dir() and stat.S_IMODE(os.lstat(final_root).st_mode) == 0o700, "final review root mode is not 0700")
    registry = read_json_object(registry_path, "review-run registry")
    require(registry.get("workflow_version") == WORKFLOW_VERSION, "final workflow mismatch")
    require(registry.get("status") == "frozen_canonical_adjudication", "final review status mismatch")
    require(registry.get("row_count") == EXPECTED_ROWS, "final review row count mismatch")
    implementation = _resolve_ref(final_root, registry["workflow_implementation"], "final workflow implementation")
    require(implementation.resolve() == Path(__file__).resolve(), "final review binds a different workflow implementation")
    canonical_path = _resolve_ref(final_root, registry["canonical_adjudicated"], "canonical adjudicated review")
    require(canonical_path.parent.resolve() == final_root.resolve(), "canonical adjudicated review escaped final root")
    require(stat.S_IMODE(os.lstat(canonical_path).st_mode) == 0o600, "canonical adjudicated mode is not 0600")
    package_manifest_path = _resolve_ref(final_root, registry["review_package_manifest"], "review package manifest")
    package_root = package_manifest_path.parent
    package = verify_package(package_root, str(registry["review_package_manifest"]["sha256"]))
    adjudication_manifest_path = _resolve_ref(final_root, registry["adjudication_manifest"], "adjudication manifest")
    verify_adjudication_root(
        adjudication_manifest_path.parent,
        package_root,
        str(registry["adjudication_manifest"]["sha256"]),
    )
    for name in (
        "reviewer_a_completed",
        "reviewer_b_completed",
        "third_completed",
        "generation_manifest",
    ):
        _resolve_ref(final_root, registry[name], f"final bound artifact {name}")
    expected_video_binding = hashlib.sha256(
        canonical_json_bytes(
            [
                {
                    "generation_id": record["generation_id"],
                    "video_sha256": record["video_sha256"],
                }
                for record in package["composites"]
            ]
        )
    ).hexdigest()
    require(registry["video_binding_inventory_sha256"] == expected_video_binding, "final video binding digest mismatch")
    return registry, canonical_path, package


@rollback_fresh_partials
def score_bound_review(
    *,
    final_root: Path,
    expected_review_run_registry_sha256: str,
    output_root: Path,
) -> dict[str, Any]:
    import causal_role_erasure_8mechanism_capability_review_v1 as frozen_scorer

    registry, canonical_adjudicated, package = verify_final_root(
        final_root,
        expected_review_run_registry_sha256,
    )
    canonical_manifest = _resolve_ref(
        final_root,
        package["frozen_inputs"]["canonical_manifest"],
        "canonical capability manifest",
    )
    review_template = _resolve_ref(final_root, package["frozen_inputs"]["review_template"], "review template")
    review_rubric = _resolve_ref(final_root, package["frozen_inputs"]["review_rubric"], "review rubric")
    review_freeze = _resolve_ref(final_root, package["frozen_inputs"]["review_freeze"], "review freeze")
    manifest_rows = frozen_scorer.load_manifest(canonical_manifest)
    blank_rows = frozen_scorer.verify_frozen_review_inputs(
        manifest_rows,
        review_template_path=review_template,
        review_rubric_path=review_rubric,
        review_freeze_path=review_freeze,
    )
    adjudicated = frozen_scorer._read_csv_exact(
        canonical_adjudicated,
        frozen_scorer.REVIEW_FIELDS,
        "canonical adjudicated review",
    )
    gate = frozen_scorer.score_adjudicated_rows(
        adjudicated,
        blank_rows,
        adjudicated_sha256=sha256_file(canonical_adjudicated),
        review_freeze_sha256=sha256_file(review_freeze),
    )
    partial = prepare_partial_root(output_root)
    gate_path = partial / "aggregate_gate.json"
    write_bytes_exclusive(gate_path, pretty_json_bytes(gate))
    bound_registry = {
        "schema_version": 1,
        "workflow_version": WORKFLOW_VERSION,
        "status": "frozen_bound_capability_gate",
        "workflow_implementation": file_ref(Path(__file__).resolve()),
        "frozen_scorer_implementation": file_ref(Path(frozen_scorer.__file__).resolve()),
        "review_run_registry": file_ref(final_root / "review_run_registry.json"),
        "canonical_adjudicated": file_ref(canonical_adjudicated),
        "generation_manifest": registry["generation_manifest"],
        "aggregate_gate": relative_file_ref(partial, gate_path),
        "capability_gate_status": gate["status"],
        "training_authorized": False,
        "treatment_generation_authorized": False,
    }
    bound_path = partial / "bound_gate_registry.json"
    write_bytes_exclusive(bound_path, pretty_json_bytes(bound_registry))
    expose_partial_root(partial, output_root)
    return {
        "registry": str(output_root / bound_path.name),
        "sha256": sha256_file(output_root / bound_path.name),
        **bound_registry,
    }


def resolve_project_path(project_root: Path, value: Path) -> Path:
    reject_sealed_path(value)
    resolved = value.resolve() if value.is_absolute() else (project_root / value).resolve()
    reject_sealed_path(resolved)
    return resolved


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="build two anonymous full-49-frame assignments")
    build.add_argument("--generation-manifest", type=Path, required=True)
    build.add_argument("--expected-generation-manifest-sha256", required=True)
    build.add_argument("--canonical-manifest", type=Path, default=DEFAULT_CANONICAL)
    build.add_argument("--review-template", type=Path, default=DEFAULT_REVIEW_TEMPLATE)
    build.add_argument("--review-rubric", type=Path, default=DEFAULT_REVIEW_RUBRIC)
    build.add_argument("--review-freeze", type=Path, default=DEFAULT_REVIEW_FREEZE)
    build.add_argument("--output-root", type=Path, required=True)
    build.add_argument("--dry-run", action="store_true")

    adjudicate = subparsers.add_parser("plan-adjudication", help="build the blind third-review disagreement queue")
    adjudicate.add_argument("--package-root", type=Path, required=True)
    adjudicate.add_argument("--expected-package-manifest-sha256", required=True)
    adjudicate.add_argument("--reviewer-a-completed", type=Path, required=True)
    adjudicate.add_argument("--reviewer-b-completed", type=Path, required=True)
    adjudicate.add_argument("--output-root", type=Path, required=True)

    finalize = subparsers.add_parser("finalize", help="merge A/B/third scores into canonical adjudication")
    finalize.add_argument("--package-root", type=Path, required=True)
    finalize.add_argument("--expected-package-manifest-sha256", required=True)
    finalize.add_argument("--adjudication-root", type=Path, required=True)
    finalize.add_argument("--expected-adjudication-manifest-sha256", required=True)
    finalize.add_argument("--third-completed", type=Path, required=True)
    finalize.add_argument("--output-root", type=Path, required=True)

    score = subparsers.add_parser("score", help="verify the frozen review registry before running the frozen aggregate scorer")
    score.add_argument("--final-root", type=Path, required=True)
    score.add_argument("--expected-review-run-registry-sha256", required=True)
    score.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    reject_sealed_path(args.project_root)
    project_root = args.project_root.resolve()
    reject_sealed_path(project_root)
    require(project_root.is_dir(), f"project root is not a directory: {project_root}")
    try:
        if args.command == "build":
            generation = resolve_project_path(project_root, args.generation_manifest)
            canonical = resolve_project_path(project_root, args.canonical_manifest)
            template = resolve_project_path(project_root, args.review_template)
            rubric = resolve_project_path(project_root, args.review_rubric)
            freeze = resolve_project_path(project_root, args.review_freeze)
            output = resolve_project_path(project_root, args.output_root)
            reject_sealed_path(generation, canonical, template, rubric, freeze, output)
            if not args.dry_run:
                require(
                    args.expected_generation_manifest_sha256 == EXPECTED_GENERATION_MANIFEST_SHA256,
                    "formal v1 package must bind the registered generation-manifest SHA-256",
                )
            canonical_rows, _, _ = validate_frozen_inputs(canonical, template, rubric, freeze)
            validate_generation_manifest(generation, args.expected_generation_manifest_sha256, canonical_rows)
            if args.dry_run:
                print(
                    json.dumps(
                        {
                            "status": "dry_run_validated_no_output_written",
                            "generation_manifest_sha256": args.expected_generation_manifest_sha256,
                            "rows": EXPECTED_ROWS,
                            "panel_ranges_inclusive": PANEL_RANGES,
                            "reviewer_a_order_sha256": hashlib.sha256(canonical_json_bytes(deterministic_order([row["generation_id"] for row in canonical_rows], "reviewer_a"))).hexdigest(),
                            "reviewer_b_order_sha256": hashlib.sha256(canonical_json_bytes(deterministic_order([row["generation_id"] for row in canonical_rows], "reviewer_b"))).hexdigest(),
                        },
                        indent=2,
                    )
                )
                return 0
            result = build_package(
                project_root=project_root,
                generation_manifest=generation,
                expected_generation_sha256=args.expected_generation_manifest_sha256,
                canonical_path=canonical,
                template_path=template,
                rubric_path=rubric,
                freeze_path=freeze,
                output_root=output,
            )
        elif args.command == "plan-adjudication":
            result = plan_adjudication(
                package_root=resolve_project_path(project_root, args.package_root),
                expected_package_manifest_sha256=args.expected_package_manifest_sha256,
                reviewer_a_completed=resolve_project_path(project_root, args.reviewer_a_completed),
                reviewer_b_completed=resolve_project_path(project_root, args.reviewer_b_completed),
                output_root=resolve_project_path(project_root, args.output_root),
            )
        elif args.command == "finalize":
            result = finalize_reviews(
                package_root=resolve_project_path(project_root, args.package_root),
                expected_package_manifest_sha256=args.expected_package_manifest_sha256,
                adjudication_root=resolve_project_path(project_root, args.adjudication_root),
                expected_adjudication_manifest_sha256=args.expected_adjudication_manifest_sha256,
                third_completed=resolve_project_path(project_root, args.third_completed),
                output_root=resolve_project_path(project_root, args.output_root),
            )
        else:
            result = score_bound_review(
                final_root=resolve_project_path(project_root, args.final_root),
                expected_review_run_registry_sha256=args.expected_review_run_registry_sha256,
                output_root=resolve_project_path(project_root, args.output_root),
            )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"ERROR: {exc}", file=os.sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
