#!/usr/bin/env python3
"""Build and finalize blinded screening for seven-mechanism target candidates.

This tool is deliberately limited to *training-target* screening.  It does not
call a VLM, inspect formal-evaluation outputs, or start generation.  The build
stage binds the frozen 1,344-row candidate graph to the exact 1,152 newly
generated (non-Water) videos, verifies every byte/media contract, renders all
49 frames in five overlapping panels, and creates two independently ordered
anonymous review deliveries.  Later stages plan field-level third review and
freeze a deterministic 178-target selection per mechanism.
"""

from __future__ import annotations

import argparse
import csv
import functools
import hashlib
import io
import json
import os
import secrets
import shutil
import statistics
import sys
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


PROTOCOL_VERSION = "causal_role_erasure_7m_single_seed_v2"
WORKFLOW_VERSION = "causal_role_erasure_7mechanism_target_screening_v2"

MECHANISM_ORDER = (
    "water_impact",
    "rigid_collision",
    "brittle_fracture",
    "powder_impact",
    "elastic_deformation",
    "material_release",
    "surface_trace",
)
HISTORICAL_CAPABILITY_INDICES = (0, 1, 2, 3, 4, 6, 7)
WATER_MECHANISM = "water_impact"
NEW_MECHANISMS = MECHANISM_ORDER[1:]
ROWS_PER_MECHANISM = 192
EXPECTED_CANDIDATES = 7 * ROWS_PER_MECHANISM
EXPECTED_NEW_VIDEOS = 6 * ROWS_PER_MECHANISM
SELECTED_PER_MECHANISM = 178

FRAME_COUNT = 49
FPS = Fraction(8, 1)
HEIGHT = 480
WIDTH = 832
PANEL_RANGES = ((0, 12), (9, 21), (18, 30), (27, 39), (36, 48))

CANDIDATE_REQUIRED_FIELDS = (
    "protocol_version",
    "candidate_id",
    "global_index",
    "mechanism_index",
    "historical_capability_index",
    "mechanism",
    "mechanism_name",
    "source_index",
    "source_id",
    "source_object",
    "receiver_index",
    "receiver_id",
    "receiver",
    "prompt_style",
    "factual_prompt",
    "target_prompt",
    "expected_trigger",
    "expected_footprint",
    "expected_counterfactual_state",
    "seed",
    "num_frames",
    "fps",
    "target_origin",
    "prompt_shard",
    "prompt_shard_index",
)
CANDIDATE_OPTIONAL_PASSTHROUGH_FIELDS = (
    "height",
    "width",
    "seed_strategy",
    "intended_use",
    "historical_pair_id",
    "historical_screen_status",
    "target_video_path",
    "excluded_content_phrase",
)

GENERATION_AGGREGATE_REQUIRED_FIELDS = (
    "schema_version",
    "protocol_version",
    "status",
    "target_candidates_sha256",
    "video_binding_key",
    "expected_videos",
    "validated_videos",
    "media_contract",
    "items",
)
GENERATION_ITEM_FIELDS = (
    "global_index",
    "candidate_id",
    "mechanism",
    "seed",
    "prompt",
    "video_path",
    "video_sha256",
    "size_bytes",
    "media",
)
MEDIA_FIELDS = ("decoded_frames", "fps", "height", "width")
RUNNER_NATIVE_AGGREGATE_FIELDS = (
    "schema_version",
    "protocol_id",
    "runner_id",
    "status",
    "target_candidates_sha256",
    "water_reuse_manifest_sha256",
    "water_reuse_rows",
    "baseline",
    "generation",
    "video_count",
    "items",
)
RUNNER_NATIVE_ITEM_FIELDS = (
    "global_index",
    "candidate_id",
    "mechanism",
    "prompt_shard_index",
    "target_prompt",
    "seed",
    "video_path",
    "video_sha256",
    "size_bytes",
    "media",
)
RUNNER_NATIVE_MEDIA_FIELDS = ("decoded_frames", "fps", "width", "height")
RUNNER_NATIVE_ID = "causal_role_erasure_7mechanism_targets_runner_v2"
WATER_REUSE_MANIFEST_SHA256 = (
    "406e4b2d06c80e415dca05bda2262924e07c3aea67398b5caca3687fca7c140d"
)
RUNNER_NATIVE_GENERATION = {
    "baseline": "negative_prompt",
    "num_inference_steps": 25,
    "guidance_scale": 5.0,
    "num_frames": 49,
    "fps": 8,
    "height": 480,
    "width": 832,
    "dtype": "bf16",
    "skip_existing": False,
    "resume": False,
    "per_prompt_seeds": "explicit_from_target_candidates_csv",
}

SCORE_FIELDS = (
    "receiver_recognizable",
    "source_absent",
    "trigger_absent",
    "footprint_absent",
    "quality",
    "natural_motion",
)
SCORE_TEMPLATE_FIELDS = ("anonymous_review_id", *SCORE_FIELDS, "reviewer_notes")
ASSIGNMENT_FIELDS = (
    "assignment_position",
    "anonymous_review_id",
    "mechanism",
    "mechanism_name",
    "composite_path",
    "target_prompt",
    "receiver",
    "source_object",
    "expected_trigger",
    "expected_footprint",
    "expected_counterfactual_state",
    "excluded_content_phrase",
)
BINDING_FIELDS = (
    "anonymous_review_id",
    "pass_name",
    "candidate_id",
    "mechanism",
    "candidate_row_sha256",
    "video_path",
    "video_sha256",
    "composite_path",
    "composite_sha256",
)
ADJUDICATION_ASSIGNMENT_FIELDS = (
    "adjudication_position",
    "adjudication_id",
    "mechanism",
    "mechanism_name",
    "composite_path",
    "target_prompt",
    "receiver",
    "source_object",
    "expected_trigger",
    "expected_footprint",
    "expected_counterfactual_state",
    "excluded_content_phrase",
    "requested_fields",
)
ADJUDICATION_SCORE_FIELDS = (
    "adjudication_id",
    "requested_fields",
    *SCORE_FIELDS,
    "reviewer_notes",
)
ADJUDICATION_BINDING_FIELDS = (
    "adjudication_id",
    "candidate_id",
    "requested_fields",
    "reviewer_a_scores",
    "reviewer_b_scores",
    "composite_path",
    "composite_sha256",
)

SELECTION_HASH_CONTEXT = "target-select-v1"
ORDER_HASH_CONTEXT = (
    "causal_role_erasure_7mechanism_target_screening_v2:review_order_v1"
)


class ScreeningError(ValueError):
    """A fail-closed protocol validation error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ScreeningError(message)


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


def canonical_row_sha256(row: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(dict(row))).hexdigest()


def regular_file(path: Path, label: str) -> None:
    require(path.is_file() and not path.is_symlink(), f"{label} missing or symlinked: {path}")


def file_ref(path: Path) -> dict[str, Any]:
    regular_file(path, "referenced file")
    return {"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}


def relative_file_ref(root: Path, path: Path) -> dict[str, Any]:
    regular_file(path, "package artifact")
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    regular_file(path, "CSV input")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        require(fields, f"CSV has no header: {path}")
        rows = list(reader)
    require(all(None not in row for row in rows), f"CSV contains overflow columns: {path}")
    return fields, [{field: str(row[field]) for field in fields} for row in rows]


def csv_bytes(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(fields), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: str(row.get(field, "")) for field in fields})
    return stream.getvalue().encode("utf-8")


def write_bytes_exclusive(path: Path, raw: bytes, mode: int = 0o644) -> None:
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


def resolve_project_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _int(value: str, label: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ScreeningError(f"{label} is not an integer") from exc


def load_and_validate_candidates(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    fields, rows = read_csv(path)
    require(
        tuple(fields[: len(CANDIDATE_REQUIRED_FIELDS)]) == CANDIDATE_REQUIRED_FIELDS,
        "candidate canonical columns/order differ from the frozen screening interface",
    )
    require(len(rows) == EXPECTED_CANDIDATES, f"expected {EXPECTED_CANDIDATES} candidates")
    require(len({row["candidate_id"] for row in rows}) == len(rows), "duplicate candidate_id")
    require(all(row["candidate_id"] for row in rows), "blank candidate_id")

    new_seeds: set[int] = set()
    for global_index, row in enumerate(rows):
        mechanism_index = global_index // ROWS_PER_MECHANISM
        local_index = global_index % ROWS_PER_MECHANISM
        source_index = local_index // 24
        receiver_index = (local_index % 24) // 2
        variant_index = local_index % 2
        mechanism = MECHANISM_ORDER[mechanism_index]
        expected_id = (
            f"tgt7m_m{mechanism_index:02d}_s{source_index:02d}_"
            f"r{receiver_index:02d}_v{variant_index}"
        )
        require(row["protocol_version"] == PROTOCOL_VERSION, f"row {global_index}: protocol mismatch")
        require(_int(row["global_index"], "global_index") == global_index, f"row {global_index}: global order mismatch")
        require(_int(row["mechanism_index"], "mechanism_index") == mechanism_index, f"row {global_index}: mechanism index mismatch")
        require(_int(row["historical_capability_index"], "historical_capability_index") == HISTORICAL_CAPABILITY_INDICES[mechanism_index], f"row {global_index}: historical index mismatch")
        require(row["mechanism"] == mechanism, f"row {global_index}: mechanism order mismatch")
        require(row["candidate_id"] == expected_id, f"row {global_index}: candidate ID mismatch")
        require(_int(row["source_index"], "source_index") == source_index, f"row {global_index}: source index mismatch")
        require(_int(row["receiver_index"], "receiver_index") == receiver_index, f"row {global_index}: receiver index mismatch")
        require(row["prompt_style"] == ("direct" if variant_index == 0 else "natural"), f"row {global_index}: prompt style mismatch")
        require(row["factual_prompt"] and row["target_prompt"] and row["factual_prompt"] != row["target_prompt"], f"row {global_index}: invalid prompts")
        require(all(row[field].strip() for field in ("source_id", "source_object", "receiver_id", "receiver", "expected_trigger", "expected_footprint", "expected_counterfactual_state")), f"row {global_index}: blank ontology field")
        require(_int(row["num_frames"], "num_frames") == FRAME_COUNT, f"row {global_index}: frame-count mismatch")
        require(Fraction(row["fps"]) == FPS, f"row {global_index}: fps mismatch")

        if mechanism == WATER_MECHANISM:
            require(row["target_origin"] == "water_v1_reuse", f"row {global_index}: Water origin mismatch")
            require(row["prompt_shard"] == row["prompt_shard_index"] == "", f"row {global_index}: Water must not enter new-generation shards")
        else:
            require(row["target_origin"] == "new_generation_v2", f"row {global_index}: new target origin mismatch")
            require(row["prompt_shard"].strip(), f"row {global_index}: missing prompt shard")
            require(_int(row["prompt_shard_index"], "prompt_shard_index") == local_index, f"row {global_index}: prompt shard order mismatch")
            seed = _int(row["seed"], "seed")
            expected_seed = 1_100_000 + 10_000 * mechanism_index + 100 * source_index + 2 * receiver_index + variant_index
            require(seed == expected_seed, f"row {global_index}: seed formula mismatch")
            require(seed not in new_seeds, f"row {global_index}: duplicate new-generation seed")
            new_seeds.add(seed)

    require(len(new_seeds) == EXPECTED_NEW_VIDEOS, "new-generation seed inventory is incomplete")
    registry_path = path.parent / "build_registry.json"
    if registry_path.is_file() and not registry_path.is_symlink():
        registry = _load_json_object(registry_path, "candidate build registry")
        require(registry.get("protocol_version") == PROTOCOL_VERSION, "candidate build registry protocol mismatch")
        require(registry.get("counts", {}).get("target_candidates") == EXPECTED_CANDIDATES, "candidate build registry count mismatch")
        artifact_hashes = registry.get("artifact_sha256", {})
        bound_hashes = [
            digest
            for name, digest in artifact_hashes.items()
            if Path(str(name)).name == path.name
        ]
        require(bound_hashes == [sha256_file(path)], "candidate bytes differ from adjacent build registry")
        require(
            registry.get("selection_contract", {}).get("if_eligible_above_178")
            == "select ascending SHA256 of target-select-v1 pipe candidate_id",
            "candidate build registry selection contract mismatch",
        )
        require(
            registry.get("water_reuse", {}).get("generation_manifest_sha256")
            == WATER_REUSE_MANIFEST_SHA256,
            "candidate build registry Water generation binding mismatch",
        )
    return fields, rows


def expected_media() -> dict[str, Any]:
    return {
        "decoded_frames": FRAME_COUNT,
        "fps": f"{FPS.numerator}/{FPS.denominator}",
        "height": HEIGHT,
        "width": WIDTH,
    }


def load_and_validate_generation_aggregate(
    path: Path,
    *,
    candidates_path: Path,
    candidate_rows: Sequence[Mapping[str, str]],
    project_root: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    regular_file(path, "generation aggregate")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ScreeningError("cannot parse generation aggregate") from exc
    require(isinstance(payload, dict), "generation aggregate must be an object")
    native_runner_schema = set(payload) == set(RUNNER_NATIVE_AGGREGATE_FIELDS)
    if native_runner_schema:
        require(payload["schema_version"] == 1, "runner-native schema version mismatch")
        require(payload["protocol_id"] == PROTOCOL_VERSION, "generation aggregate protocol mismatch")
        require(payload["runner_id"] == RUNNER_NATIVE_ID, "runner-native runner ID mismatch")
        require(
            payload["status"] == "frozen_after_exact_1152_video_validation",
            "runner-native generation aggregate is not frozen after exact validation",
        )
        require(
            payload["water_reuse_manifest_sha256"] == WATER_REUSE_MANIFEST_SHA256
            and payload["water_reuse_rows"] == ROWS_PER_MECHANISM,
            "runner-native Water reuse binding mismatch",
        )
        require(payload["baseline"] == "negative_prompt", "runner-native baseline mismatch")
        require(payload["generation"] == RUNNER_NATIVE_GENERATION, "runner-native generation config mismatch")
        require(payload["video_count"] == EXPECTED_NEW_VIDEOS, "runner-native video count mismatch")
    else:
        require(set(GENERATION_AGGREGATE_REQUIRED_FIELDS) <= set(payload), "generation aggregate fields are incomplete")
        require(payload["protocol_version"] == PROTOCOL_VERSION, "generation aggregate protocol mismatch")
        require(payload["status"] == "completed", "generation aggregate is not completed")
        require(payload["video_binding_key"] == "candidate_id", "generation aggregate binding key mismatch")
        require(payload["expected_videos"] == payload["validated_videos"] == EXPECTED_NEW_VIDEOS, "generation aggregate video counts mismatch")
        require(payload["media_contract"] == expected_media(), "generation aggregate media contract mismatch")
    require(payload["target_candidates_sha256"] == sha256_file(candidates_path), "generation aggregate binds different candidate bytes")
    items = payload["items"]
    require(isinstance(items, list) and len(items) == EXPECTED_NEW_VIDEOS, "generation aggregate item count mismatch")

    new_rows = [row for row in candidate_rows if row["target_origin"] == "new_generation_v2"]
    by_id: dict[str, dict[str, Any]] = {}
    seen_paths: set[Path] = set()
    for expected_row, raw_item in zip(new_rows, items):
        require(isinstance(raw_item, dict), f"{expected_row['candidate_id']}: generation item is not an object")
        if native_runner_schema:
            require(set(raw_item) == set(RUNNER_NATIVE_ITEM_FIELDS), f"{expected_row['candidate_id']}: runner-native item schema mismatch")
            require(isinstance(raw_item["media"], dict) and set(raw_item["media"]) == set(RUNNER_NATIVE_MEDIA_FIELDS), f"{expected_row['candidate_id']}: runner-native media schema mismatch")
            require(raw_item["prompt_shard_index"] == _int(expected_row["prompt_shard_index"], "prompt_shard_index"), f"{expected_row['candidate_id']}: prompt shard index mismatch")
            require(raw_item["target_prompt"] == expected_row["target_prompt"], f"{expected_row['candidate_id']}: target prompt mismatch")
            item = {
                "global_index": raw_item["global_index"],
                "candidate_id": raw_item["candidate_id"],
                "mechanism": raw_item["mechanism"],
                "seed": raw_item["seed"],
                "prompt": raw_item["target_prompt"],
                "video_path": raw_item["video_path"],
                "video_sha256": raw_item["video_sha256"],
                "size_bytes": raw_item["size_bytes"],
                "media": {
                    "decoded_frames": raw_item["media"]["decoded_frames"],
                    "fps": raw_item["media"]["fps"],
                    "height": raw_item["media"]["height"],
                    "width": raw_item["media"]["width"],
                },
            }
        else:
            require(tuple(raw_item) == GENERATION_ITEM_FIELDS, f"{expected_row['candidate_id']}: generation item schema mismatch")
            require(isinstance(raw_item["media"], dict) and tuple(raw_item["media"]) == MEDIA_FIELDS, f"{expected_row['candidate_id']}: media schema mismatch")
            item = raw_item
        require(item["global_index"] == _int(expected_row["global_index"], "global_index"), f"{expected_row['candidate_id']}: global index mismatch")
        require(item["candidate_id"] == expected_row["candidate_id"], "generation/candidate order mismatch")
        require(item["mechanism"] == expected_row["mechanism"], f"{expected_row['candidate_id']}: mechanism mismatch")
        require(item["seed"] == _int(expected_row["seed"], "seed"), f"{expected_row['candidate_id']}: seed mismatch")
        require(item["prompt"] == expected_row["target_prompt"], f"{expected_row['candidate_id']}: target prompt mismatch")
        require(item["media"] == expected_media(), f"{expected_row['candidate_id']}: declared media mismatch")
        video_path = resolve_project_path(project_root, str(item["video_path"]))
        regular_file(video_path, f"{expected_row['candidate_id']} video")
        require(video_path.suffix.lower() == ".mp4", f"{expected_row['candidate_id']}: video is not MP4")
        require(video_path not in seen_paths, f"{expected_row['candidate_id']}: duplicate video path")
        seen_paths.add(video_path)
        if expected_row.get("target_video_path"):
            # ``target_video_path`` is the frozen logical candidate-ID path.
            # ``generate_wan_clean.py`` canonically emits a slug filename, so
            # bind the registered mechanism/videos directory while treating
            # the fully validated generation-manifest path as the canonical
            # physical filename.  This mirrors the runner's single-use path
            # recovery policy without weakening any ID/order/prompt/seed/hash
            # or media check above/below.
            logical_path = resolve_project_path(
                project_root, expected_row["target_video_path"]
            )
            require(
                logical_path.suffix.lower() == ".mp4"
                and logical_path.parent.resolve() == video_path.parent.resolve(),
                f"{expected_row['candidate_id']}: logical target_video_path escaped registered videos directory",
            )
        require(item["size_bytes"] == video_path.stat().st_size and item["size_bytes"] > 0, f"{expected_row['candidate_id']}: video size mismatch")
        require(item["video_sha256"] == sha256_file(video_path), f"{expected_row['candidate_id']}: video SHA-256 mismatch")
        by_id[expected_row["candidate_id"]] = dict(item, _resolved_video_path=str(video_path))
    require(len(by_id) == EXPECTED_NEW_VIDEOS, "generation item IDs are duplicate")
    return payload, by_id


def load_video_frames(path: Path) -> tuple[list[Any], dict[str, Any]]:
    try:
        import av
    except ImportError as exc:
        raise ScreeningError("PyAV is required to validate/render target videos") from exc
    with av.open(str(path)) as container:
        video_streams = [stream for stream in container.streams if stream.type == "video"]
        audio_streams = [stream for stream in container.streams if stream.type == "audio"]
        require(len(video_streams) == 1 and not audio_streams, f"video must contain one video stream and no audio: {path}")
        stream = video_streams[0]
        rate = stream.average_rate or stream.guessed_rate
        frames = [frame.to_image().convert("RGB") for frame in container.decode(video=stream.index)]
        media = {
            "decoded_frames": len(frames),
            "fps": f"{Fraction(rate).numerator}/{Fraction(rate).denominator}" if rate is not None else "",
            "height": int(stream.height),
            "width": int(stream.width),
        }
    require(media == expected_media(), f"decoded media contract mismatch: {path}")
    return frames, media


def render_full49_composite(frames: Sequence[Any], output_path: Path) -> None:
    from PIL import Image, ImageDraw

    require(len(frames) == FRAME_COUNT, "composite requires exactly 49 frames")
    covered = {index for start, end in PANEL_RANGES for index in range(start, end + 1)}
    require(covered == set(range(FRAME_COUNT)), "panel ranges do not cover all frames")
    frame_width = 192
    frame_height = round(frames[0].height * frame_width / frames[0].width)
    columns = 7
    label_height = 20
    header_height = 26
    panels = []
    for panel_index, (start, end) in enumerate(PANEL_RANGES):
        panel = Image.new("RGB", (columns * frame_width, header_height + 2 * (frame_height + label_height)), "white")
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
    require(not output_path.exists(), f"refusing to overwrite composite: {output_path}")
    with output_path.open("xb") as handle:
        composite.save(handle, format="JPEG", quality=92, optimize=True)


def review_order(rows: Sequence[Mapping[str, str]], pass_name: str) -> list[Mapping[str, str]]:
    require(pass_name in {"reviewer_a", "reviewer_b"}, "unknown review pass")
    output: list[Mapping[str, str]] = []
    for mechanism in NEW_MECHANISMS:
        mechanism_rows = [row for row in rows if row["mechanism"] == mechanism]
        require(len(mechanism_rows) == ROWS_PER_MECHANISM, f"{mechanism}: review inventory mismatch")
        output.extend(
            sorted(
                mechanism_rows,
                key=lambda row: (
                    hashlib.sha256(
                        f"{ORDER_HASH_CONTEXT}\0{pass_name}\0{mechanism}\0{row['candidate_id']}".encode("utf-8")
                    ).hexdigest(),
                    row["candidate_id"],
                ),
            )
        )
    return output


def reviewer_instructions() -> dict[str, Any]:
    return {
        "workflow_version": WORKFLOW_VERSION,
        "task": "independent_full_49_frame_training_target_screening",
        "method_outputs_or_formal_eval": False,
        "panel_ranges_inclusive": [list(value) for value in PANEL_RANGES],
        "fields": {
            "receiver_recognizable": {"0": "absent or unrecognizable", "1": "partial or uncertain", "2": "clearly recognizable"},
            "source_absent": {"0": "source clearly present", "1": "partial or uncertain source cue", "2": "source absent in all frames"},
            "trigger_absent": {"0": "registered trigger clearly occurs", "1": "partial or uncertain trigger cue", "2": "registered trigger absent"},
            "footprint_absent": {"0": "registered footprint clearly present", "1": "partial or uncertain footprint cue", "2": "registered footprint absent"},
            "quality": {"0": "unusable or badly broken", "1": "partially usable", "2": "good visual quality"},
            "natural_motion": {"0": "frozen or implausibly broken", "1": "limited but plausible temporal motion", "2": "clear plausible source-free motion"},
        },
        "independence": "do not access the other pass assignment or scores",
        "output": "fill only the separately supplied scoring_template.csv",
    }


def _fresh_partial(output_root: Path) -> Path:
    require(not output_root.exists() and not output_root.is_symlink(), f"output root already exists: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    partial = output_root.parent / f".{output_root.name}.partial-{os.getpid()}"
    require(not partial.exists(), f"partial output already exists: {partial}")
    partial.mkdir()
    return partial


def _publish_partial(partial: Path, output_root: Path) -> None:
    require(not output_root.exists(), f"output root appeared during build: {output_root}")
    os.rename(partial, output_root)


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    regular_file(path, label)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ScreeningError(f"cannot parse {label}: {path}") from exc
    require(isinstance(payload, dict), f"{label} must be an object")
    return payload


def _resolve_ref(root: Path, ref: Mapping[str, Any], label: str) -> Path:
    require(set(ref) == {"path", "sha256", "size_bytes"}, f"{label} reference schema mismatch")
    path = Path(str(ref["path"]))
    if not path.is_absolute():
        path = root / path
    regular_file(path, label)
    require(path.stat().st_size == int(ref["size_bytes"]), f"{label} size changed")
    require(sha256_file(path) == ref["sha256"], f"{label} SHA-256 changed")
    return path


def verify_review_package(package_root: Path, expected_sha256: str | None = None) -> dict[str, Any]:
    manifest_path = package_root / "review_package_manifest.json"
    regular_file(manifest_path, "review package manifest")
    if expected_sha256 is not None:
        require(sha256_file(manifest_path) == expected_sha256, "review package manifest SHA-256 mismatch")
    manifest = _load_json_object(manifest_path, "review package manifest")
    require(manifest.get("workflow_version") == WORKFLOW_VERSION, "review package workflow mismatch")
    require(manifest.get("status") == "committed_before_independent_review", "review package status mismatch")
    require(manifest.get("nonwater_candidate_count") == EXPECTED_NEW_VIDEOS, "review package count mismatch")
    require(manifest.get("score_fields") == list(SCORE_FIELDS), "review package score schema mismatch")
    candidates_path = _resolve_ref(package_root, manifest["candidate_manifest"], "candidate manifest")
    _, candidates = load_and_validate_candidates(candidates_path)
    candidate_by_id = {row["candidate_id"]: row for row in candidates}
    _resolve_ref(package_root, manifest["generation_aggregate"], "generation aggregate")
    for pass_name in ("reviewer_a", "reviewer_b"):
        artifacts = manifest.get("artifacts", {}).get(pass_name, {})
        require(set(artifacts) == {"assignment", "scoring_template", "instructions", "binding"}, f"{pass_name}: artifact inventory mismatch")
        for name, ref in artifacts.items():
            _resolve_ref(package_root, ref, f"{pass_name}/{name}")
    masters = manifest.get("master_composites")
    require(isinstance(masters, dict) and len(masters) == EXPECTED_NEW_VIDEOS, "master composite inventory mismatch")
    for candidate_id, ref in masters.items():
        _resolve_ref(package_root, ref, f"master composite/{candidate_id}")
    for pass_name in ("reviewer_a", "reviewer_b"):
        assignment_fields, assignments = read_csv(
            _artifact_path(package_root, manifest, pass_name, "assignment")
        )
        template_fields, templates = read_csv(
            _artifact_path(package_root, manifest, pass_name, "scoring_template")
        )
        binding_fields, bindings = read_csv(
            _artifact_path(package_root, manifest, pass_name, "binding")
        )
        require(tuple(assignment_fields) == ASSIGNMENT_FIELDS, f"{pass_name}: assignment columns differ")
        require(tuple(template_fields) == SCORE_TEMPLATE_FIELDS, f"{pass_name}: template columns differ")
        require(tuple(binding_fields) == BINDING_FIELDS, f"{pass_name}: binding columns differ")
        require(len(assignments) == len(templates) == len(bindings) == EXPECTED_NEW_VIDEOS, f"{pass_name}: delivery count mismatch")
        seen_ids: set[str] = set()
        seen_candidates: set[str] = set()
        for assignment, template, binding in zip(assignments, templates, bindings):
            review_id = assignment["anonymous_review_id"]
            candidate_id = binding["candidate_id"]
            require(review_id == template["anonymous_review_id"] == binding["anonymous_review_id"], f"{pass_name}: public/private review ID mismatch")
            require(review_id not in seen_ids and candidate_id not in seen_candidates, f"{pass_name}: duplicate delivery binding")
            seen_ids.add(review_id)
            seen_candidates.add(candidate_id)
            row = candidate_by_id.get(candidate_id)
            require(row is not None and row["target_origin"] == "new_generation_v2", f"{pass_name}: binding references unknown candidate")
            require(binding["candidate_row_sha256"] == canonical_row_sha256(row), f"{pass_name}/{candidate_id}: candidate binding drift")
            require(assignment["mechanism"] == binding["mechanism"] == row["mechanism"], f"{pass_name}/{candidate_id}: mechanism binding mismatch")
            require(assignment["composite_path"] == binding["composite_path"], f"{pass_name}/{candidate_id}: composite path mismatch")
            delivered = package_root / binding["composite_path"]
            regular_file(delivered, f"{pass_name}/{candidate_id} delivered composite")
            require(sha256_file(delivered) == binding["composite_sha256"], f"{pass_name}/{candidate_id}: delivered composite drift")
            require(binding["composite_sha256"] == masters[candidate_id]["sha256"], f"{pass_name}/{candidate_id}: delivered/master mismatch")
            require(all(template[field] == "" for field in (*SCORE_FIELDS, "reviewer_notes")), f"{pass_name}: scoring template is not blank")
        require(len(seen_candidates) == EXPECTED_NEW_VIDEOS, f"{pass_name}: candidate inventory incomplete")
    return manifest


def _artifact_path(package_root: Path, package: Mapping[str, Any], pass_name: str, name: str) -> Path:
    return _resolve_ref(package_root, package["artifacts"][pass_name][name], f"{pass_name}/{name}")


def validate_completed_scores(
    completed_path: Path,
    template_path: Path,
    label: str,
) -> list[dict[str, str]]:
    completed_fields, completed = read_csv(completed_path)
    template_fields, template = read_csv(template_path)
    require(tuple(completed_fields) == SCORE_TEMPLATE_FIELDS, f"{label}: completed score columns differ")
    require(tuple(template_fields) == SCORE_TEMPLATE_FIELDS, f"{label}: template score columns differ")
    require(len(completed) == len(template) == EXPECTED_NEW_VIDEOS, f"{label}: score row count mismatch")
    for index, (actual, blank) in enumerate(zip(completed, template)):
        require(actual["anonymous_review_id"] == blank["anonymous_review_id"], f"{label} row {index}: review ID/order changed")
        require(all(blank[field] == "" for field in (*SCORE_FIELDS, "reviewer_notes")), f"{label}: frozen template is not blank")
        for field in SCORE_FIELDS:
            require(actual[field] in {"0", "1", "2"}, f"{label} row {index}: invalid {field}")
    return completed


def _scores_by_candidate(
    scores: Sequence[Mapping[str, str]],
    bindings: Sequence[Mapping[str, str]],
) -> dict[str, dict[str, str]]:
    require(len(scores) == len(bindings), "score/binding count mismatch")
    output: dict[str, dict[str, str]] = {}
    for score, binding in zip(scores, bindings):
        require(score["anonymous_review_id"] == binding["anonymous_review_id"], "score/binding review ID mismatch")
        candidate_id = binding["candidate_id"]
        require(candidate_id not in output, "duplicate candidate score binding")
        output[candidate_id] = {field: score[field] for field in SCORE_FIELDS}
    require(len(output) == EXPECTED_NEW_VIDEOS, "score binding inventory incomplete")
    return output


def _read_bindings(package_root: Path, package: Mapping[str, Any], pass_name: str) -> list[dict[str, str]]:
    fields, rows = read_csv(_artifact_path(package_root, package, pass_name, "binding"))
    require(tuple(fields) == BINDING_FIELDS and len(rows) == EXPECTED_NEW_VIDEOS, f"{pass_name}: binding schema/count mismatch")
    return rows


def build_review_package(
    *,
    project_root: Path,
    candidates_path: Path,
    generation_aggregate_path: Path,
    output_root: Path,
    frame_loader: Callable[[Path], tuple[list[Any], dict[str, Any]]] = load_video_frames,
    renderer: Callable[[Sequence[Any], Path], None] = render_full49_composite,
) -> dict[str, Any]:
    candidate_fields, candidates = load_and_validate_candidates(candidates_path)
    aggregate, generated = load_and_validate_generation_aggregate(
        generation_aggregate_path,
        candidates_path=candidates_path,
        candidate_rows=candidates,
        project_root=project_root,
    )
    new_rows = [row for row in candidates if row["target_origin"] == "new_generation_v2"]
    partial = _fresh_partial(output_root)
    try:
        private = partial / "private"
        public = partial / "public"
        master = private / "master_composites"
        private.mkdir(mode=0o700)
        public.mkdir()
        master.mkdir()
        master_records: dict[str, dict[str, Any]] = {}
        for row in new_rows:
            candidate_id = row["candidate_id"]
            item = generated[candidate_id]
            video_path = Path(item["_resolved_video_path"])
            frames, media = frame_loader(video_path)
            require(media == expected_media(), f"{candidate_id}: live media differs from aggregate")
            composite_path = master / f"{candidate_id}.jpg"
            renderer(frames, composite_path)
            master_records[candidate_id] = {
                "path": composite_path.relative_to(partial).as_posix(),
                "sha256": sha256_file(composite_path),
                "size_bytes": composite_path.stat().st_size,
            }

        a_order = review_order(new_rows, "reviewer_a")
        b_order = review_order(new_rows, "reviewer_b")
        for mechanism in NEW_MECHANISMS:
            a_ids = [row["candidate_id"] for row in a_order if row["mechanism"] == mechanism]
            b_ids = [row["candidate_id"] for row in b_order if row["mechanism"] == mechanism]
            require(a_ids != b_ids, f"{mechanism}: independent review orders unexpectedly coincide")

        orders = {"reviewer_a": a_order, "reviewer_b": b_order}
        package_artifacts: dict[str, Any] = {}
        for pass_name, prefix in (("reviewer_a", "a"), ("reviewer_b", "b")):
            delivery = public / pass_name
            media_dir = delivery / "media"
            media_dir.mkdir(parents=True)
            assignments: list[dict[str, str]] = []
            templates: list[dict[str, str]] = []
            bindings: list[dict[str, str]] = []
            for position, row in enumerate(orders[pass_name]):
                candidate_id = row["candidate_id"]
                anonymous_id = f"{prefix}{position:04d}"
                delivered = media_dir / f"{prefix}_{secrets.token_hex(16)}.jpg"
                source = partial / master_records[candidate_id]["path"]
                shutil.copyfile(source, delivered)
                require(sha256_file(delivered) == master_records[candidate_id]["sha256"], "anonymous composite copy drift")
                assignments.append(
                    {
                        "assignment_position": str(position),
                        "anonymous_review_id": anonymous_id,
                        "mechanism": row["mechanism"],
                        "mechanism_name": row["mechanism_name"],
                        "composite_path": delivered.relative_to(partial).as_posix(),
                        "target_prompt": row["target_prompt"],
                        "receiver": row["receiver"],
                        "source_object": row["source_object"],
                        "expected_trigger": row["expected_trigger"],
                        "expected_footprint": row["expected_footprint"],
                        "expected_counterfactual_state": row["expected_counterfactual_state"],
                        "excluded_content_phrase": row.get("excluded_content_phrase", ""),
                    }
                )
                templates.append({"anonymous_review_id": anonymous_id, **{field: "" for field in SCORE_FIELDS}, "reviewer_notes": ""})
                item = generated[candidate_id]
                bindings.append(
                    {
                        "anonymous_review_id": anonymous_id,
                        "pass_name": pass_name,
                        "candidate_id": candidate_id,
                        "mechanism": row["mechanism"],
                        "candidate_row_sha256": canonical_row_sha256(row),
                        "video_path": str(item["video_path"]),
                        "video_sha256": str(item["video_sha256"]),
                        "composite_path": delivered.relative_to(partial).as_posix(),
                        "composite_sha256": master_records[candidate_id]["sha256"],
                    }
                )
            assignment_path = delivery / "assignment.csv"
            template_path = delivery / "scoring_template.csv"
            instructions_path = delivery / "instructions.json"
            binding_path = private / f"{pass_name}_binding.csv"
            write_bytes_exclusive(assignment_path, csv_bytes(assignments, ASSIGNMENT_FIELDS))
            write_bytes_exclusive(template_path, csv_bytes(templates, SCORE_TEMPLATE_FIELDS))
            write_bytes_exclusive(instructions_path, json.dumps(reviewer_instructions(), indent=2, ensure_ascii=False).encode("utf-8") + b"\n")
            write_bytes_exclusive(binding_path, csv_bytes(bindings, BINDING_FIELDS), mode=0o600)
            package_artifacts[pass_name] = {
                "assignment": relative_file_ref(partial, assignment_path),
                "scoring_template": relative_file_ref(partial, template_path),
                "instructions": relative_file_ref(partial, instructions_path),
                "binding": relative_file_ref(partial, binding_path),
            }

        manifest = {
            "schema_version": 1,
            "workflow_version": WORKFLOW_VERSION,
            "status": "committed_before_independent_review",
            "candidate_manifest": file_ref(candidates_path),
            "candidate_fields": candidate_fields,
            "generation_aggregate": file_ref(generation_aggregate_path),
            "generation_aggregate_binding_sha256": hashlib.sha256(canonical_json_bytes(aggregate)).hexdigest(),
            "nonwater_candidate_count": EXPECTED_NEW_VIDEOS,
            "panel_ranges_inclusive": [list(value) for value in PANEL_RANGES],
            "score_fields": list(SCORE_FIELDS),
            "order_algorithm": "per_mechanism_sha256_frozen_context_v1",
            "master_composites": master_records,
            "artifacts": package_artifacts,
        }
        manifest_path = partial / "review_package_manifest.json"
        write_bytes_exclusive(manifest_path, json.dumps(manifest, indent=2, ensure_ascii=False).encode("utf-8") + b"\n")
        _publish_partial(partial, output_root)
        return {**manifest, "manifest": str(output_root / "review_package_manifest.json"), "sha256": sha256_file(output_root / "review_package_manifest.json")}
    except BaseException:
        if partial.exists() and partial.name.startswith(f".{output_root.name}.partial-"):
            shutil.rmtree(partial)
        raise


def plan_adjudication(
    *,
    package_root: Path,
    expected_package_sha256: str | None,
    reviewer_a_completed: Path,
    reviewer_b_completed: Path,
    output_root: Path,
) -> dict[str, Any]:
    package = verify_review_package(package_root, expected_package_sha256)
    a_rows = validate_completed_scores(
        reviewer_a_completed,
        _artifact_path(package_root, package, "reviewer_a", "scoring_template"),
        "reviewer A",
    )
    b_rows = validate_completed_scores(
        reviewer_b_completed,
        _artifact_path(package_root, package, "reviewer_b", "scoring_template"),
        "reviewer B",
    )
    a_bindings = _read_bindings(package_root, package, "reviewer_a")
    b_bindings = _read_bindings(package_root, package, "reviewer_b")
    a_scores = _scores_by_candidate(a_rows, a_bindings)
    b_scores = _scores_by_candidate(b_rows, b_bindings)
    require(set(a_scores) == set(b_scores), "A/B candidate inventories differ")

    candidates_path = _resolve_ref(package_root, package["candidate_manifest"], "candidate manifest")
    _, candidates = load_and_validate_candidates(candidates_path)
    candidate_by_id = {row["candidate_id"]: row for row in candidates}
    disagreements = {
        candidate_id: [
            field for field in SCORE_FIELDS
            if a_scores[candidate_id][field] != b_scores[candidate_id][field]
        ]
        for candidate_id in a_scores
    }
    disagreement_ids = [candidate_id for candidate_id, fields in disagreements.items() if fields]
    disagreement_ids.sort(
        key=lambda candidate_id: (
            hashlib.sha256(f"{ORDER_HASH_CONTEXT}\0third\0{candidate_id}".encode("utf-8")).hexdigest(),
            candidate_id,
        )
    )

    partial = _fresh_partial(output_root)
    try:
        public = partial / "public" / "third_reviewer"
        private = partial / "private"
        media_dir = public / "media"
        media_dir.mkdir(parents=True)
        private.mkdir(mode=0o700)
        assignments: list[dict[str, str]] = []
        templates: list[dict[str, str]] = []
        bindings: list[dict[str, str]] = []
        for position, candidate_id in enumerate(disagreement_ids):
            row = candidate_by_id[candidate_id]
            adjudication_id = f"t{position:04d}"
            requested = ";".join(disagreements[candidate_id])
            master_ref = package["master_composites"][candidate_id]
            master_path = _resolve_ref(package_root, master_ref, f"master composite/{candidate_id}")
            delivered = media_dir / f"t_{secrets.token_hex(16)}.jpg"
            shutil.copyfile(master_path, delivered)
            require(sha256_file(delivered) == master_ref["sha256"], "third composite copy drift")
            assignments.append(
                {
                    "adjudication_position": str(position),
                    "adjudication_id": adjudication_id,
                    "mechanism": row["mechanism"],
                    "mechanism_name": row["mechanism_name"],
                    "composite_path": delivered.relative_to(partial).as_posix(),
                    "target_prompt": row["target_prompt"],
                    "receiver": row["receiver"],
                    "source_object": row["source_object"],
                    "expected_trigger": row["expected_trigger"],
                    "expected_footprint": row["expected_footprint"],
                    "expected_counterfactual_state": row["expected_counterfactual_state"],
                    "excluded_content_phrase": row.get("excluded_content_phrase", ""),
                    "requested_fields": requested,
                }
            )
            templates.append(
                {
                    "adjudication_id": adjudication_id,
                    "requested_fields": requested,
                    **{field: "" for field in SCORE_FIELDS},
                    "reviewer_notes": "",
                }
            )
            bindings.append(
                {
                    "adjudication_id": adjudication_id,
                    "candidate_id": candidate_id,
                    "requested_fields": requested,
                    "reviewer_a_scores": json.dumps(a_scores[candidate_id], sort_keys=True, separators=(",", ":")),
                    "reviewer_b_scores": json.dumps(b_scores[candidate_id], sort_keys=True, separators=(",", ":")),
                    "composite_path": delivered.relative_to(partial).as_posix(),
                    "composite_sha256": master_ref["sha256"],
                }
            )
        assignment_path = public / "assignment.csv"
        template_path = public / "scoring_template.csv"
        instructions_path = public / "instructions.json"
        binding_path = private / "adjudication_binding.csv"
        write_bytes_exclusive(assignment_path, csv_bytes(assignments, ADJUDICATION_ASSIGNMENT_FIELDS))
        write_bytes_exclusive(template_path, csv_bytes(templates, ADJUDICATION_SCORE_FIELDS))
        instructions = reviewer_instructions()
        instructions.update(
            {
                "task": "field_level_third_review_for_A_B_disagreements_only",
                "fill_only_requested_fields": True,
            }
        )
        write_bytes_exclusive(instructions_path, json.dumps(instructions, indent=2, ensure_ascii=False).encode("utf-8") + b"\n")
        write_bytes_exclusive(binding_path, csv_bytes(bindings, ADJUDICATION_BINDING_FIELDS), mode=0o600)
        manifest = {
            "schema_version": 1,
            "workflow_version": WORKFLOW_VERSION,
            "status": "committed_before_field_level_third_review",
            "review_package_manifest": file_ref(package_root / "review_package_manifest.json"),
            "reviewer_a_completed": file_ref(reviewer_a_completed),
            "reviewer_b_completed": file_ref(reviewer_b_completed),
            "videos_with_disagreement": len(disagreement_ids),
            "atomic_disagreement_count": sum(len(disagreements[value]) for value in disagreement_ids),
            "resolution": "per_field_median_of_three",
            "third_assignment": relative_file_ref(partial, assignment_path),
            "third_scoring_template": relative_file_ref(partial, template_path),
            "third_instructions": relative_file_ref(partial, instructions_path),
            "adjudication_binding": relative_file_ref(partial, binding_path),
        }
        manifest_path = partial / "adjudication_manifest.json"
        write_bytes_exclusive(manifest_path, json.dumps(manifest, indent=2, ensure_ascii=False).encode("utf-8") + b"\n")
        _publish_partial(partial, output_root)
        return {**manifest, "manifest": str(output_root / "adjudication_manifest.json"), "sha256": sha256_file(output_root / "adjudication_manifest.json")}
    except BaseException:
        if partial.exists() and partial.name.startswith(f".{output_root.name}.partial-"):
            shutil.rmtree(partial)
        raise


def verify_adjudication_root(
    adjudication_root: Path,
    package_root: Path,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    manifest_path = adjudication_root / "adjudication_manifest.json"
    regular_file(manifest_path, "adjudication manifest")
    if expected_sha256 is not None:
        require(sha256_file(manifest_path) == expected_sha256, "adjudication manifest SHA-256 mismatch")
    manifest = _load_json_object(manifest_path, "adjudication manifest")
    require(manifest.get("workflow_version") == WORKFLOW_VERSION, "adjudication workflow mismatch")
    require(manifest.get("status") == "committed_before_field_level_third_review", "adjudication status mismatch")
    package_path = _resolve_ref(adjudication_root, manifest["review_package_manifest"], "bound review package")
    require(package_path.resolve() == (package_root / "review_package_manifest.json").resolve(), "adjudication binds another review package")
    for name in (
        "reviewer_a_completed",
        "reviewer_b_completed",
        "third_assignment",
        "third_scoring_template",
        "third_instructions",
        "adjudication_binding",
    ):
        _resolve_ref(adjudication_root, manifest[name], name)
    return manifest


def validate_third_completed(
    completed_path: Path,
    template_path: Path,
) -> list[dict[str, str]]:
    completed_fields, completed = read_csv(completed_path)
    template_fields, template = read_csv(template_path)
    require(tuple(completed_fields) == ADJUDICATION_SCORE_FIELDS, "third completed columns differ")
    require(tuple(template_fields) == ADJUDICATION_SCORE_FIELDS, "third template columns differ")
    require(len(completed) == len(template), "third completed row count mismatch")
    for index, (actual, blank) in enumerate(zip(completed, template)):
        require(actual["adjudication_id"] == blank["adjudication_id"], f"third row {index}: ID/order changed")
        require(actual["requested_fields"] == blank["requested_fields"], f"third row {index}: requested fields changed")
        requested = set(blank["requested_fields"].split(";")) if blank["requested_fields"] else set()
        require(requested and requested <= set(SCORE_FIELDS), f"third row {index}: requested fields invalid")
        for field in SCORE_FIELDS:
            if field in requested:
                require(actual[field] in {"0", "1", "2"}, f"third row {index}: invalid {field}")
            else:
                require(actual[field] == "", f"third row {index}: non-requested {field} must remain blank")
    return completed


def validate_water_reuse(
    *,
    candidates: Sequence[Mapping[str, str]],
    screen_csv: Path,
    selected_csv: Path,
    project_root: Path,
    water_media_root: Path | None = None,
    frame_loader: Callable[[Path], tuple[list[Any], dict[str, Any]]] = load_video_frames,
) -> tuple[list[Mapping[str, str]], dict[str, dict[str, Any]]]:
    media_root = project_root if water_media_root is None else water_media_root
    require(media_root.is_dir(), f"Water media root missing: {media_root}")
    water = [row for row in candidates if row["mechanism"] == WATER_MECHANISM]
    require(len(water) == ROWS_PER_MECHANISM, "Water candidate inventory mismatch")
    screen_fields, screen = read_csv(screen_csv)
    selected_fields, selected = read_csv(selected_csv)
    for field in ("pair_id", "video_path", "final_status"):
        require(field in screen_fields, f"Water screen CSV missing {field}")
    for field in ("pair_id", "desired_target_video"):
        require(field in selected_fields, f"Water selected CSV missing {field}")
    require(len(screen) == ROWS_PER_MECHANISM, "Water screen CSV must contain 192 rows")
    require(len(selected) == SELECTED_PER_MECHANISM, "Water selected CSV must contain 178 rows")
    screen_by_pair = {row["pair_id"]: row for row in screen}
    require(len(screen_by_pair) == ROWS_PER_MECHANISM, "Water screen pair IDs are duplicate")
    accepted_pairs = {row["pair_id"] for row in screen if row["final_status"] == "accept"}
    selected_by_pair = {row["pair_id"]: row for row in selected}
    require(len(accepted_pairs) == SELECTED_PER_MECHANISM, "Water historical accept count differs from 178")
    require(set(selected_by_pair) == accepted_pairs, "Water selected manifest differs from historical accepts")

    water_by_pair: dict[str, Mapping[str, str]] = {}
    for row in water:
        pair_id = row.get("historical_pair_id", "")
        require(pair_id and pair_id in screen_by_pair and pair_id not in water_by_pair, f"{row['candidate_id']}: invalid historical Water binding")
        historical = screen_by_pair[pair_id]
        require(row.get("historical_screen_status", "") == historical["final_status"], f"{row['candidate_id']}: Water screen status mismatch")
        require(row.get("target_video_path", "") == historical["video_path"], f"{row['candidate_id']}: Water target path mismatch")
        water_by_pair[pair_id] = row
    require(set(water_by_pair) == set(screen_by_pair), "Water candidate/screen inventories differ")

    selected_rows = [water_by_pair[pair_id] for pair_id in accepted_pairs]
    media_by_candidate: dict[str, dict[str, Any]] = {}
    for row in selected_rows:
        pair_id = row.get("historical_pair_id", "")
        path = resolve_project_path(media_root, row.get("target_video_path", ""))
        regular_file(path, f"Water target/{pair_id}")
        require(resolve_project_path(media_root, selected_by_pair[pair_id]["desired_target_video"]).resolve() == path.resolve(), f"Water selected path mismatch: {pair_id}")
        _, media = frame_loader(path)
        require(media == expected_media(), f"Water target media mismatch: {pair_id}")
        media_by_candidate[row["candidate_id"]] = {
            "video_path": str(path),
            "video_sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
            "media": media,
        }
    return selected_rows, media_by_candidate


def finalize_screening(
    *,
    project_root: Path,
    package_root: Path,
    expected_package_sha256: str | None,
    adjudication_root: Path,
    expected_adjudication_sha256: str | None,
    third_completed: Path,
    water_screen_csv: Path,
    water_selected_csv: Path,
    water_media_root: Path | None,
    output_root: Path,
    water_frame_loader: Callable[[Path], tuple[list[Any], dict[str, Any]]] = load_video_frames,
) -> dict[str, Any]:
    package = verify_review_package(package_root, expected_package_sha256)
    adjudication = verify_adjudication_root(adjudication_root, package_root, expected_adjudication_sha256)
    a_path = _resolve_ref(adjudication_root, adjudication["reviewer_a_completed"], "reviewer A completed")
    b_path = _resolve_ref(adjudication_root, adjudication["reviewer_b_completed"], "reviewer B completed")
    a_rows = validate_completed_scores(a_path, _artifact_path(package_root, package, "reviewer_a", "scoring_template"), "reviewer A")
    b_rows = validate_completed_scores(b_path, _artifact_path(package_root, package, "reviewer_b", "scoring_template"), "reviewer B")
    a_scores = _scores_by_candidate(a_rows, _read_bindings(package_root, package, "reviewer_a"))
    b_scores = _scores_by_candidate(b_rows, _read_bindings(package_root, package, "reviewer_b"))

    third_template = _resolve_ref(adjudication_root, adjudication["third_scoring_template"], "third template")
    third_rows = validate_third_completed(third_completed, third_template)
    binding_fields, third_bindings = read_csv(_resolve_ref(adjudication_root, adjudication["adjudication_binding"], "adjudication binding"))
    require(tuple(binding_fields) == ADJUDICATION_BINDING_FIELDS, "adjudication binding columns differ")
    require(len(third_rows) == len(third_bindings), "third score/binding count mismatch")
    third_by_candidate: dict[str, dict[str, str]] = {}
    for score, binding in zip(third_rows, third_bindings):
        require(score["adjudication_id"] == binding["adjudication_id"], "third score/binding ID mismatch")
        third_by_candidate[binding["candidate_id"]] = {
            field: score[field] for field in binding["requested_fields"].split(";")
        }

    candidates_path = _resolve_ref(package_root, package["candidate_manifest"], "candidate manifest")
    candidate_fields, candidates = load_and_validate_candidates(candidates_path)
    candidate_build_registry = candidates_path.parent / "build_registry.json"
    if candidate_build_registry.is_file() and not candidate_build_registry.is_symlink():
        build_registry = _load_json_object(
            candidate_build_registry, "candidate build registry"
        )
        require(
            build_registry.get("input_sha256", {}).get("water_screen")
            == sha256_file(water_screen_csv),
            "Water screen bytes differ from candidate build registry",
        )
    candidate_by_id = {row["candidate_id"]: row for row in candidates}
    aggregate_path = _resolve_ref(package_root, package["generation_aggregate"], "generation aggregate")
    _, generated = load_and_validate_generation_aggregate(
        aggregate_path,
        candidates_path=candidates_path,
        candidate_rows=candidates,
        project_root=project_root,
    )

    canonical_scores: list[dict[str, Any]] = []
    score_by_candidate: dict[str, dict[str, int]] = {}
    for candidate_id in [row["candidate_id"] for row in candidates if row["target_origin"] == "new_generation_v2"]:
        final: dict[str, int] = {}
        resolution: list[str] = []
        for field in SCORE_FIELDS:
            left = int(a_scores[candidate_id][field])
            right = int(b_scores[candidate_id][field])
            if left == right:
                final[field] = left
            else:
                require(candidate_id in third_by_candidate and field in third_by_candidate[candidate_id], f"missing third score for {candidate_id}/{field}")
                final[field] = int(statistics.median((left, right, int(third_by_candidate[candidate_id][field]))))
                resolution.append(field)
        score_by_candidate[candidate_id] = final
        canonical_scores.append(
            {
                "candidate_id": candidate_id,
                "mechanism": candidate_by_id[candidate_id]["mechanism"],
                **final,
                "eligible": "yes" if eligible_from_scores(final) else "no",
                "resolution": "A_B_agreement" if not resolution else "median_of_three:" + ";".join(resolution),
            }
        )
    require(len(canonical_scores) == EXPECTED_NEW_VIDEOS, "canonical score inventory incomplete")

    water_selected, water_media = validate_water_reuse(
        candidates=candidates,
        screen_csv=water_screen_csv,
        selected_csv=water_selected_csv,
        project_root=project_root,
        water_media_root=water_media_root,
        frame_loader=water_frame_loader,
    )
    selected_by_mechanism, eligible_counts = select_new_targets(
        candidates, score_by_candidate
    )
    selected_by_mechanism[WATER_MECHANISM] = sorted(
        water_selected,
        key=lambda row: (selection_hash(row["candidate_id"]), row["candidate_id"]),
    )
    eligible_counts[WATER_MECHANISM] = len(water_selected)

    partial = _fresh_partial(output_root)
    try:
        selected_dir = partial / "selected_by_mechanism"
        selected_dir.mkdir()
        canonical_path = partial / "canonical_nonwater_scores.csv"
        canonical_fields = ("candidate_id", "mechanism", *SCORE_FIELDS, "eligible", "resolution")
        write_bytes_exclusive(canonical_path, csv_bytes(canonical_scores, canonical_fields), mode=0o600)
        appended_fields = (
            *SCORE_FIELDS,
            "eligible",
            "eligibility_source",
            "selection_hash",
            "selection_rank",
            "selected_video_path",
            "selected_video_sha256",
            "selected_size_bytes",
            "selected_media",
        )
        selected_fields = (*candidate_fields, *appended_fields)
        all_selected: list[dict[str, Any]] = []
        mechanism_refs: dict[str, Any] = {}
        for mechanism_index, mechanism in enumerate(MECHANISM_ORDER):
            output_rows: list[dict[str, Any]] = []
            for rank, row in enumerate(selected_by_mechanism[mechanism], start=1):
                candidate_id = row["candidate_id"]
                if mechanism == WATER_MECHANISM:
                    scores: Mapping[str, Any] = {field: "" for field in SCORE_FIELDS}
                    media_record = water_media[candidate_id]
                    source = "historical_water_v1_accept"
                else:
                    scores = score_by_candidate[candidate_id]
                    item = generated[candidate_id]
                    media_record = {
                        "video_path": item["video_path"],
                        "video_sha256": item["video_sha256"],
                        "size_bytes": item["size_bytes"],
                        "media": item["media"],
                    }
                    source = "canonical_A_B_or_median_review"
                output_rows.append(
                    {
                        **row,
                        **scores,
                        "eligible": "yes",
                        "eligibility_source": source,
                        "selection_hash": selection_hash(candidate_id),
                        "selection_rank": str(rank),
                        "selected_video_path": media_record["video_path"],
                        "selected_video_sha256": media_record["video_sha256"],
                        "selected_size_bytes": str(media_record["size_bytes"]),
                        "selected_media": json.dumps(media_record["media"], sort_keys=True, separators=(",", ":")),
                    }
                )
            require(len(output_rows) == SELECTED_PER_MECHANISM, f"{mechanism}: selected count mismatch")
            path = selected_dir / f"{mechanism_index:02d}_{mechanism}.csv"
            write_bytes_exclusive(path, csv_bytes(output_rows, selected_fields), mode=0o600)
            mechanism_refs[mechanism] = relative_file_ref(partial, path)
            all_selected.extend(output_rows)
        require(len(all_selected) == len(MECHANISM_ORDER) * SELECTED_PER_MECHANISM, "all-selected count mismatch")
        all_path = partial / "selected_targets.csv"
        write_bytes_exclusive(all_path, csv_bytes(all_selected, selected_fields), mode=0o600)
        summary = {
            "schema_version": 1,
            "workflow_version": WORKFLOW_VERSION,
            "status": "selected_after_canonical_blind_screening",
            "eligibility_rule": {
                "all_fields_at_least": 1,
                "source_absent": 2,
                "trigger_absent": 2,
                "footprint_absent": 2,
            },
            "selection_order": "sha256('target-select-v1|' + candidate_id)",
            "selection_hash_context": SELECTION_HASH_CONTEXT,
            "eligible_counts": eligible_counts,
            "selected_counts": {mechanism: SELECTED_PER_MECHANISM for mechanism in MECHANISM_ORDER},
            "review_package_manifest": file_ref(package_root / "review_package_manifest.json"),
            "adjudication_manifest": file_ref(adjudication_root / "adjudication_manifest.json"),
            "third_completed": file_ref(third_completed),
            "water_historical_screen": file_ref(water_screen_csv),
            "water_historical_selected": file_ref(water_selected_csv),
            "canonical_nonwater_scores": relative_file_ref(partial, canonical_path),
            "selected_targets": relative_file_ref(partial, all_path),
            "selected_by_mechanism": mechanism_refs,
            "selected_total": len(all_selected),
        }
        summary_path = partial / "selection_registry.json"
        write_bytes_exclusive(summary_path, json.dumps(summary, indent=2, ensure_ascii=False).encode("utf-8") + b"\n", mode=0o600)
        _publish_partial(partial, output_root)
        return {**summary, "registry": str(output_root / "selection_registry.json"), "sha256": sha256_file(output_root / "selection_registry.json")}
    except BaseException:
        if partial.exists() and partial.name.startswith(f".{output_root.name}.partial-"):
            shutil.rmtree(partial)
        raise


def selection_hash(candidate_id: str) -> str:
    return hashlib.sha256(f"{SELECTION_HASH_CONTEXT}|{candidate_id}".encode("utf-8")).hexdigest()


def eligible_from_scores(scores: Mapping[str, int]) -> bool:
    return (
        all(int(scores[field]) >= 1 for field in SCORE_FIELDS)
        and all(int(scores[field]) == 2 for field in ("source_absent", "trigger_absent", "footprint_absent"))
    )


def select_new_targets(
    candidates: Sequence[Mapping[str, str]],
    score_by_candidate: Mapping[str, Mapping[str, int]],
) -> tuple[dict[str, list[Mapping[str, str]]], dict[str, int]]:
    """Select exactly 178 eligible rows per non-Water mechanism by frozen hash."""

    selected: dict[str, list[Mapping[str, str]]] = {}
    counts: dict[str, int] = {}
    for mechanism in NEW_MECHANISMS:
        mechanism_rows = [row for row in candidates if row["mechanism"] == mechanism]
        require(len(mechanism_rows) == ROWS_PER_MECHANISM, f"{mechanism}: candidate inventory mismatch")
        eligible: list[Mapping[str, str]] = []
        for row in mechanism_rows:
            candidate_id = row["candidate_id"]
            require(candidate_id in score_by_candidate, f"{candidate_id}: canonical score missing")
            if eligible_from_scores(score_by_candidate[candidate_id]):
                eligible.append(row)
        eligible.sort(key=lambda row: (selection_hash(row["candidate_id"]), row["candidate_id"]))
        counts[mechanism] = len(eligible)
        require(
            len(eligible) >= SELECTED_PER_MECHANISM,
            f"{mechanism}: only {len(eligible)} eligible targets; need {SELECTED_PER_MECHANISM}",
        )
        selected[mechanism] = eligible[:SELECTED_PER_MECHANISM]
    return selected, counts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="verify media and build two anonymous review deliveries")
    build.add_argument("--candidates", type=Path, required=True)
    build.add_argument("--generation-aggregate", type=Path, required=True)
    build.add_argument("--output-root", type=Path, required=True)
    adjudicate = subparsers.add_parser(
        "plan-adjudication",
        help="verify independent A/B scores and build the field-level third-review queue",
    )
    adjudicate.add_argument("--package-root", type=Path, required=True)
    adjudicate.add_argument("--expected-package-sha256")
    adjudicate.add_argument("--reviewer-a-completed", type=Path, required=True)
    adjudicate.add_argument("--reviewer-b-completed", type=Path, required=True)
    adjudicate.add_argument("--output-root", type=Path, required=True)

    finalize = subparsers.add_parser(
        "finalize",
        help="median-resolve A/B disagreements and freeze 178 selected targets per mechanism",
    )
    finalize.add_argument("--package-root", type=Path, required=True)
    finalize.add_argument("--expected-package-sha256")
    finalize.add_argument("--adjudication-root", type=Path, required=True)
    finalize.add_argument("--expected-adjudication-sha256")
    finalize.add_argument("--third-completed", type=Path, required=True)
    finalize.add_argument(
        "--water-screen-csv",
        type=Path,
        default=Path("data/water_impact_dynamic_v1/train_targets_v1_screen_final.csv"),
    )
    finalize.add_argument(
        "--water-selected-csv",
        type=Path,
        default=Path("data/water_impact_dynamic_v1/train_dynamic_sft_manifest.csv"),
    )
    finalize.add_argument(
        "--water-media-root",
        type=Path,
        help="Root used to resolve legacy Water video paths; defaults to --project-root.",
    )
    finalize.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    project_root = args.project_root.resolve()
    require(project_root.is_dir(), f"project root missing: {project_root}")
    try:
        if args.command == "build":
            result = build_review_package(
                project_root=project_root,
                candidates_path=resolve_project_path(project_root, str(args.candidates)),
                generation_aggregate_path=resolve_project_path(project_root, str(args.generation_aggregate)),
                output_root=resolve_project_path(project_root, str(args.output_root)),
            )
        elif args.command == "plan-adjudication":
            result = plan_adjudication(
                package_root=resolve_project_path(project_root, str(args.package_root)),
                expected_package_sha256=args.expected_package_sha256,
                reviewer_a_completed=resolve_project_path(project_root, str(args.reviewer_a_completed)),
                reviewer_b_completed=resolve_project_path(project_root, str(args.reviewer_b_completed)),
                output_root=resolve_project_path(project_root, str(args.output_root)),
            )
        elif args.command == "finalize":
            result = finalize_screening(
                project_root=project_root,
                package_root=resolve_project_path(project_root, str(args.package_root)),
                expected_package_sha256=args.expected_package_sha256,
                adjudication_root=resolve_project_path(project_root, str(args.adjudication_root)),
                expected_adjudication_sha256=args.expected_adjudication_sha256,
                third_completed=resolve_project_path(project_root, str(args.third_completed)),
                water_screen_csv=resolve_project_path(project_root, str(args.water_screen_csv)),
                water_selected_csv=resolve_project_path(project_root, str(args.water_selected_csv)),
                water_media_root=(
                    project_root
                    if args.water_media_root is None
                    else args.water_media_root.resolve()
                ),
                output_root=resolve_project_path(project_root, str(args.output_root)),
            )
        else:  # pragma: no cover - argparse prevents this
            raise ScreeningError("unknown command")
    except (OSError, ScreeningError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
