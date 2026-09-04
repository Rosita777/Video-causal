#!/usr/bin/env python3
"""Freeze the seven-mechanism generation ledger and blinded review package.

This is deliberately a final-stage, fail-closed builder.  It accepts only the
four completed generation manifests registered by the experiment, rejoins
every generated video to the frozen case tables, verifies the bytes and media,
and writes a fresh package atomically.  Public artifacts contain semantic
review instructions and anonymous composite paths only.  Stream, backbone and
source-video provenance are confined to mode-0600 private keys.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import hmac
import json
import os
import shutil
import stat
import tempfile
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


PROTOCOL_VERSION = "causal_role_erasure_7m_single_seed_v2"
PACKAGE_PROTOCOL = "causal_role_erasure_7m_anonymous_review_v1"
LEDGER_PROTOCOL = "causal_role_erasure_7m_generation_ledger_v1"
SCHEMA_VERSION = 1

MECHANISMS = (
    "water_impact",
    "rigid_collision",
    "brittle_fracture",
    "powder_impact",
    "elastic_deformation",
    "material_release",
    "surface_trace",
)
MAIN_STREAMS = (
    "wan_original",
    "matched_control",
    "V4",
    "cogvideox_original",
    "negative_prompt",
    "videoeraser_official",
    "t2vunlearning_adapted",
    "safree_cogvideox",
)
IDENTIFICATION_STREAMS = ("generic_paraphrase", "bystander_token")
COGVIDEOX_STREAMS = (
    "cogvideox_original",
    "negative_prompt",
    "videoeraser_official",
    "t2vunlearning_adapted",
    "safree_cogvideox",
)
ORIGINAL_STREAMS = ("wan_original", "cogvideox_original")

FORMAL_CASES = 294
IDENTIFICATION_CASES = 48
MAIN_VIDEO_COUNT = 2352
IDENTIFICATION_VIDEO_COUNT = 96
TOTAL_VIDEO_COUNT = 2448

PANEL_WINDOWS: tuple[tuple[int, int], ...] = (
    (0, 12),
    (9, 21),
    (18, 30),
    (27, 39),
    (36, 48),
)
THUMB_WIDTH = 112
THUMB_HEIGHT = 64
PANEL_HEADER_HEIGHT = 20
FRAME_LABEL_HEIGHT = 14
PANEL_GAP = 6
COMPOSITE_WIDTH = 13 * THUMB_WIDTH
COMPOSITE_HEIGHT = len(PANEL_WINDOWS) * (
    PANEL_HEADER_HEIGHT + THUMB_HEIGHT + FRAME_LABEL_HEIGHT + PANEL_GAP
)
JPEG_QUALITY = 82
MAX_COMPOSITE_BYTES = 3_145_728

CAUSAL_SCORE_FIELDS = (
    "source_visibility",
    "footprint_visibility",
    "receiver_preservation",
    "video_quality",
)
SPECIFICITY_SCORE_FIELDS = (
    "protected_object_visibility",
    "noncausal_role_adherence",
    "receiver_preservation",
    "video_quality",
)
PUBLIC_ASSIGNMENT_FIELDS = (
    "anonymous_review_id",
    "case_kind",
    "mechanism_name",
    "prompt",
    "source_object",
    "receiver",
    "expected_trigger",
    "expected_footprint",
    "expected_counterfactual_state",
    "protected_object",
    "specificity_subtype",
    "acceptable_alternative_cause",
    "composite_path",
    "composite_sha256",
    "assignment_sha256",
)


class ReviewPackageError(ValueError):
    """The frozen inputs or requested output violate the review protocol."""


@dataclass(frozen=True)
class DecodedVideo:
    frames: tuple[Any, ...]
    video_streams: int
    audio_streams: int
    decoded_frames: int
    fps: str
    width: int
    height: int

    def media(self) -> dict[str, Any]:
        return {
            "video_streams": self.video_streams,
            "audio_streams": self.audio_streams,
            "decoded_frames": self.decoded_frames,
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
        }


Decoder = Callable[[Path], DecodedVideo]
Renderer = Callable[[Sequence[Any], Path], dict[str, Any]]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReviewPackageError(message)


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def regular_file(path: Path, label: str) -> None:
    require(path.is_file() and not path.is_symlink(), f"{label} missing or symlinked: {path}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_bytes(path: Path, value: bytes, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(value)
    path.chmod(mode)


def write_json(path: Path, value: Any, mode: int = 0o644) -> None:
    write_bytes(path, canonical_json_bytes(value), mode)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]], mode: int = 0o644) -> None:
    body = b"".join(canonical_json_bytes(dict(row)) for row in rows)
    write_bytes(path, body, mode)


def load_json(path: Path, label: str) -> dict[str, Any]:
    regular_file(path, label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReviewPackageError(f"{label} is not readable JSON: {path}") from exc
    require(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def load_csv(path: Path, label: str) -> list[dict[str, str]]:
    regular_file(path, label)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames is not None, f"{label} is missing a CSV header")
        return [dict(row) for row in reader]


def _integer(value: Any, label: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ReviewPackageError(f"{label} must be an integer") from exc


def _hex_sha(value: Any, label: str) -> str:
    text = str(value)
    require(len(text) == 64 and all(char in "0123456789abcdef" for char in text), f"{label} is not a lowercase SHA-256")
    return text


def resolve_input(project_root: Path, path: Path) -> Path:
    candidate = path if path.is_absolute() else project_root / path
    regular_file(candidate, str(path))
    resolved = candidate.resolve()
    return resolved


def validate_formal_cases(rows: Sequence[Mapping[str, str]]) -> dict[str, dict[str, str]]:
    require(len(rows) == FORMAL_CASES, "formal case table must contain exactly 294 rows")
    require(
        [_integer(row.get("global_case_index"), "global_case_index") for row in rows]
        == list(range(FORMAL_CASES)),
        "formal global_case_index must be exactly 0..293",
    )
    case_ids = [str(row.get("case_id", "")) for row in rows]
    seeds = [_integer(row.get("seed"), f"{case_ids[index]} seed") for index, row in enumerate(rows)]
    require(len(set(case_ids)) == FORMAL_CASES and all(case_ids), "formal case IDs repeat or are empty")
    require(len(set(seeds)) == FORMAL_CASES and all(seed > 0 for seed in seeds), "formal seeds repeat or are nonpositive")
    by_id: dict[str, dict[str, str]] = {}
    for row in rows:
        case_id = str(row["case_id"])
        require(row.get("protocol_version") == PROTOCOL_VERSION, f"{case_id}: protocol version changed")
        require(row.get("mechanism") in MECHANISMS, f"{case_id}: invalid mechanism")
        require(row.get("case_kind") in ("causal", "specificity"), f"{case_id}: invalid case_kind")
        require(
            (_integer(row.get("num_frames"), f"{case_id} num_frames"), _integer(row.get("fps"), f"{case_id} fps"))
            == (49, 8),
            f"{case_id}: temporal media contract changed",
        )
        for field in (
            "mechanism_name",
            "prompt",
            "source_object",
            "receiver",
            "expected_trigger",
            "expected_footprint",
            "expected_counterfactual_state",
        ):
            require(str(row.get(field, "")).strip(), f"{case_id}: missing {field}")
        if row["case_kind"] == "specificity":
            require(str(row.get("protected_object", "")).strip(), f"{case_id}: missing protected_object")
            require(
                row.get("specificity_subtype")
                in ("same_noun_noncausal", "role_swap_or_near_causal", "same_footprint_alternative_cause"),
                f"{case_id}: invalid specificity_subtype",
            )
            if row.get("specificity_subtype") == "same_footprint_alternative_cause":
                require(
                    str(row.get("acceptable_alternative_cause", "")).strip(),
                    f"{case_id}: same-footprint specificity case is missing alternative cause",
                )
            else:
                require(
                    not str(row.get("acceptable_alternative_cause", "")).strip(),
                    f"{case_id}: non-alternative-cause specificity case has an unexpected alternative cause",
                )
        else:
            require(
                not str(row.get("protected_object", "")).strip()
                and not str(row.get("specificity_subtype", "")).strip()
                and not str(row.get("acceptable_alternative_cause", "")).strip(),
                f"{case_id}: causal case contains specificity-only fields",
            )
        by_id[case_id] = dict(row)
    require(
        Counter(row["mechanism"] for row in rows) == {mechanism: 42 for mechanism in MECHANISMS},
        "formal mechanism balance must be 42 cases per mechanism",
    )
    for mechanism in MECHANISMS:
        block = [row for row in rows if row["mechanism"] == mechanism]
        require(
            Counter(row["case_kind"] for row in block) == {"causal": 24, "specificity": 18},
            f"{mechanism}: expected 24 causal and 18 specificity cases",
        )
    return by_id


def validate_identification_cases(
    rows: Sequence[Mapping[str, str]], formal_by_id: Mapping[str, Mapping[str, str]]
) -> tuple[dict[str, dict[str, str]], dict[str, int]]:
    require(len(rows) == IDENTIFICATION_CASES, "identification table must contain exactly 48 rows")
    require(
        [_integer(row.get("identification_case_index"), "identification_case_index") for row in rows]
        == list(range(IDENTIFICATION_CASES)),
        "identification_case_index must be exactly 0..47",
    )
    by_id: dict[str, dict[str, str]] = {}
    index_by_id: dict[str, int] = {}
    for index, row in enumerate(rows):
        case_id = str(row.get("case_id", ""))
        require(case_id in formal_by_id and case_id not in by_id, f"identification join repeats or is absent: {case_id}")
        formal = formal_by_id[case_id]
        require(row.get("protocol_version") == PROTOCOL_VERSION, f"{case_id}: identification protocol changed")
        require(row.get("included_streams") == "matched_control,V4", f"{case_id}: included streams changed")
        require(
            row.get("additional_streams") == "generic_paraphrase,bystander_token",
            f"{case_id}: identification streams changed",
        )
        for field in ("mechanism", "case_kind", "seed"):
            require(str(row.get(field)) == str(formal.get(field)), f"{case_id}: identification {field} mismatch")
        by_id[case_id] = dict(row)
        index_by_id[case_id] = index
    require(
        Counter(row["mechanism"] for row in rows) == {"water_impact": 24, "brittle_fracture": 24},
        "identification table must contain 24 water and 24 fracture cases",
    )
    require(
        Counter((row["mechanism"], row["case_kind"]) for row in rows)
        == {
            ("water_impact", "causal"): 12,
            ("water_impact", "specificity"): 12,
            ("brittle_fracture", "causal"): 12,
            ("brittle_fracture", "specificity"): 12,
        },
        "identification 12/12 causal/specificity balance changed",
    )
    return by_id, index_by_id


def validate_manifest_envelopes(
    *,
    wan_original: Mapping[str, Any],
    trained_wan: Mapping[str, Any],
    cog_core: Mapping[str, Any],
    safree: Mapping[str, Any],
) -> None:
    require(wan_original.get("schema_version") == 1, "Wan Original schema changed")
    require(wan_original.get("protocol_id") == PROTOCOL_VERSION, "Wan Original protocol changed")
    require(
        wan_original.get("runner_id") == "causal_role_erasure_7mechanism_wan_original_v2",
        "Wan Original runner changed",
    )
    require(
        wan_original.get("status") == "frozen_after_exact_294_video_validation"
        and wan_original.get("video_count") == FORMAL_CASES,
        "Wan Original manifest is not the exact frozen 294-video artifact",
    )

    require(trained_wan.get("schema_version") == 1, "trained-Wan schema changed")
    require(trained_wan.get("protocol_id") == PROTOCOL_VERSION, "trained-Wan protocol changed")
    require(
        trained_wan.get("runner_id") == "causal_role_erasure_7mechanism_trained_wan_eval_v2",
        "trained-Wan runner changed",
    )
    require(
        trained_wan.get("status") == "frozen_after_exact_684_video_validation"
        and trained_wan.get("video_count") == 684,
        "trained-Wan manifest is not the exact frozen 684-video artifact",
    )

    for label, manifest, streams, count in (
        ("CogVideoX core", cog_core, list(COGVIDEOX_STREAMS[:4]), 1176),
        ("SAFREE", safree, ["safree_cogvideox"], 294),
    ):
        require(manifest.get("schema_version") == 1, f"{label} schema changed")
        require(
            manifest.get("protocol") == "causal_role_erasure_7mechanism_baseline_queue_v2",
            f"{label} queue protocol changed",
        )
        require(manifest.get("protocol_version") == PROTOCOL_VERSION, f"{label} experiment protocol changed")
        require(manifest.get("selected_streams") == streams, f"{label} stream inventory changed")
        require(
            manifest.get("status") == "frozen_after_full_media_validation"
            and manifest.get("video_count") == count,
            f"{label} manifest is not the exact frozen {count}-video artifact",
        )


def validate_manifest_input_bindings(
    *,
    project_root: Path,
    formal_path: Path,
    identification_path: Path,
    manifests: Mapping[str, Mapping[str, Any]],
) -> None:
    formal_sha = sha256_file(formal_path)
    for label in ("wan_original", "trained_wan", "cog_core", "safree"):
        inputs = manifests[label].get("inputs")
        require(isinstance(inputs, dict), f"{label}: missing frozen input bindings")
        require(inputs.get("formal_cases_sha256") == formal_sha, f"{label}: formal case SHA binding mismatch")
        registered = Path(str(inputs.get("formal_cases", "")))
        registered = (registered if registered.is_absolute() else project_root / registered).resolve()
        require(registered == formal_path, f"{label}: formal case path binding mismatch")
    trained_inputs = manifests["trained_wan"]["inputs"]
    require(
        trained_inputs.get("identification_subset_sha256") == sha256_file(identification_path),
        "trained_wan: identification subset SHA binding mismatch",
    )
    registered_identification = Path(str(trained_inputs.get("identification_subset", "")))
    registered_identification = (
        registered_identification
        if registered_identification.is_absolute()
        else project_root / registered_identification
    ).resolve()
    require(
        registered_identification == identification_path,
        "trained_wan: identification subset path binding mismatch",
    )


def _manifest_items(manifest: Mapping[str, Any], count: int, label: str) -> list[dict[str, Any]]:
    items = manifest.get("items")
    require(isinstance(items, list) and len(items) == count, f"{label} items must contain exactly {count} rows")
    require(all(isinstance(item, dict) for item in items), f"{label} contains a non-object item")
    return [dict(item) for item in items]


def _expected_declared_media(stream: str) -> dict[str, Any]:
    width = 720 if stream in COGVIDEOX_STREAMS else 832
    base: dict[str, Any] = {"decoded_frames": 49, "fps": "8/1", "height": 480, "width": width}
    if stream not in COGVIDEOX_STREAMS:
        base.update({"video_streams": 1, "audio_streams": 0})
    return base


def _normalize_item(
    item: Mapping[str, Any],
    *,
    stream: str,
    partition: str,
    source_manifest: str,
    formal_by_id: Mapping[str, Mapping[str, str]],
    identification_by_id: Mapping[str, Mapping[str, str]],
    identification_index: Mapping[str, int],
    project_root: Path,
    seen_paths: set[Path],
    hash_cache: dict[tuple[int, int, int, int], str],
) -> dict[str, Any]:
    case_id = str(item.get("case_id", ""))
    allowed = formal_by_id if partition == "main" else identification_by_id
    require(case_id in allowed, f"{source_manifest}/{stream}: unexpected case_id {case_id}")
    case = formal_by_id[case_id]
    require(str(item.get("mechanism")) == case["mechanism"], f"{stream}/{case_id}: mechanism mismatch")
    require(_integer(item.get("seed"), f"{stream}/{case_id} seed") == int(case["seed"]), f"{stream}/{case_id}: seed mismatch")
    if "formal_global_index" in item:
        require(
            _integer(item["formal_global_index"], f"{stream}/{case_id} formal_global_index")
            == int(case["global_case_index"]),
            f"{stream}/{case_id}: global index mismatch",
        )
    if source_manifest == "wan_original":
        require(item.get("stream") == stream, f"{stream}/{case_id}: stream field mismatch")
    if source_manifest in ("cog_core", "safree"):
        require(item.get("stream") == stream, f"{stream}/{case_id}: stream field mismatch")
    if source_manifest == "trained_wan":
        require(item.get("arm") == stream, f"{stream}/{case_id}: arm field mismatch")

    media = item.get("media")
    require(isinstance(media, dict), f"{stream}/{case_id}: missing media receipt")
    for field, expected in _expected_declared_media(stream).items():
        require(media.get(field) == expected, f"{stream}/{case_id}: declared media {field} mismatch")

    source_path = Path(str(item.get("video_path", "")))
    require(source_path.is_absolute(), f"{stream}/{case_id}: video path must be absolute")
    regular_file(source_path, f"{stream}/{case_id} video")
    source_path = source_path.resolve()
    try:
        source_path.relative_to(project_root)
    except ValueError as exc:
        raise ReviewPackageError(f"{stream}/{case_id}: video is outside project root") from exc
    require(source_path not in seen_paths, f"source video path is reused: {source_path}")
    seen_paths.add(source_path)
    observed_size = source_path.stat().st_size
    require(observed_size == _integer(item.get("size_bytes"), f"{stream}/{case_id} size_bytes") and observed_size > 0, f"{stream}/{case_id}: file size mismatch")
    expected_sha = _hex_sha(item.get("video_sha256"), f"{stream}/{case_id} video_sha256")
    st = source_path.stat()
    cache_key = (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)
    observed_sha = hash_cache.get(cache_key)
    if observed_sha is None:
        observed_sha = sha256_file(source_path)
        hash_cache[cache_key] = observed_sha
    require(observed_sha == expected_sha, f"{stream}/{case_id}: video SHA-256 mismatch")

    return {
        "evaluation_partition": partition,
        "stream": stream,
        "case_id": case_id,
        "formal_global_index": int(case["global_case_index"]),
        "identification_case_index": None if partition == "main" else identification_index[case_id],
        "mechanism": case["mechanism"],
        "mechanism_name": case["mechanism_name"],
        "case_kind": case["case_kind"],
        "seed": int(case["seed"]),
        "prompt": case["prompt"],
        "source_object": case["source_object"],
        "receiver": case["receiver"],
        "expected_trigger": case["expected_trigger"],
        "expected_footprint": case["expected_footprint"],
        "expected_counterfactual_state": case["expected_counterfactual_state"],
        "protected_object": case.get("protected_object", ""),
        "specificity_subtype": case.get("specificity_subtype", ""),
        "acceptable_alternative_cause": case.get("acceptable_alternative_cause", ""),
        "source_manifest": source_manifest,
        "source_video_path": str(source_path),
        "video_sha256": observed_sha,
        "size_bytes": observed_size,
        "declared_media": dict(media),
    }


def build_generation_ledger(
    *,
    project_root: Path,
    formal_rows: Sequence[Mapping[str, str]],
    identification_rows: Sequence[Mapping[str, str]],
    wan_original: Mapping[str, Any],
    trained_wan: Mapping[str, Any],
    cog_core: Mapping[str, Any],
    safree: Mapping[str, Any],
) -> list[dict[str, Any]]:
    formal_by_id = validate_formal_cases(formal_rows)
    identification_by_id, identification_index = validate_identification_cases(identification_rows, formal_by_id)
    validate_manifest_envelopes(
        wan_original=wan_original, trained_wan=trained_wan, cog_core=cog_core, safree=safree
    )
    sources = (
        ("wan_original", wan_original, 294, "stream"),
        ("trained_wan", trained_wan, 684, "arm"),
        ("cog_core", cog_core, 1176, "stream"),
        ("safree", safree, 294, "stream"),
    )
    seen_paths: set[Path] = set()
    hash_cache: dict[tuple[int, int, int, int], str] = {}
    normalized: list[dict[str, Any]] = []
    for source_label, manifest, count, stream_field in sources:
        for item in _manifest_items(manifest, count, source_label):
            stream = str(item.get(stream_field, ""))
            if source_label == "wan_original":
                require(stream == "wan_original", "Wan Original item has an unexpected stream")
            elif source_label == "trained_wan":
                require(stream in ("matched_control", "V4", *IDENTIFICATION_STREAMS), f"unexpected trained-Wan arm: {stream}")
            elif source_label == "cog_core":
                require(stream in COGVIDEOX_STREAMS[:4], f"unexpected core stream: {stream}")
            else:
                require(stream == "safree_cogvideox", f"unexpected SAFREE stream: {stream}")
            partition = "identification" if stream in IDENTIFICATION_STREAMS else "main"
            normalized.append(
                _normalize_item(
                    item,
                    stream=stream,
                    partition=partition,
                    source_manifest=source_label,
                    formal_by_id=formal_by_id,
                    identification_by_id=identification_by_id,
                    identification_index=identification_index,
                    project_root=project_root,
                    seen_paths=seen_paths,
                    hash_cache=hash_cache,
                )
            )

    require(len(normalized) == TOTAL_VIDEO_COUNT, "unified ledger must contain exactly 2448 videos")
    counts = Counter((row["evaluation_partition"], row["stream"]) for row in normalized)
    require(
        counts == Counter(
            {**{("main", stream): FORMAL_CASES for stream in MAIN_STREAMS}, **{("identification", stream): IDENTIFICATION_CASES for stream in IDENTIFICATION_STREAMS}}
        ),
        "unified stream inventory is not 8x294 main plus 2x48 identification",
    )
    formal_ids = set(formal_by_id)
    identification_ids = set(identification_by_id)
    for stream in MAIN_STREAMS:
        rows = [row for row in normalized if row["evaluation_partition"] == "main" and row["stream"] == stream]
        require({row["case_id"] for row in rows} == formal_ids, f"{stream}: main case coverage mismatch")
        require(Counter(row["mechanism"] for row in rows) == {mechanism: 42 for mechanism in MECHANISMS}, f"{stream}: mechanism balance mismatch")
    for stream in IDENTIFICATION_STREAMS:
        rows = [row for row in normalized if row["evaluation_partition"] == "identification" and row["stream"] == stream]
        require({row["case_id"] for row in rows} == identification_ids, f"{stream}: identification coverage mismatch")
    stream_order = {stream: index for index, stream in enumerate((*MAIN_STREAMS, *IDENTIFICATION_STREAMS))}
    normalized.sort(
        key=lambda row: (
            0 if row["evaluation_partition"] == "main" else 1,
            stream_order[row["stream"]],
            row["formal_global_index"] if row["evaluation_partition"] == "main" else row["identification_case_index"],
        )
    )
    for index, row in enumerate(normalized):
        row["ledger_index"] = index
        row["ledger_id"] = f"glr_{index:04d}"
        row["ledger_record_sha256"] = sha256_bytes(canonical_json_bytes(row))
    require(sum(row["evaluation_partition"] == "main" for row in normalized) == MAIN_VIDEO_COUNT, "main ledger count is not 2352")
    require(sum(row["evaluation_partition"] == "identification" for row in normalized) == IDENTIFICATION_VIDEO_COUNT, "identification ledger count is not 96")
    return normalized


def decode_video_pyav(path: Path) -> DecodedVideo:
    try:
        import av
    except ImportError as exc:
        raise ReviewPackageError(
            "PyAV is required for formal media validation (run with the registered Wan runtime)"
        ) from exc
    try:
        with av.open(str(path)) as container:
            video_streams = [stream for stream in container.streams if stream.type == "video"]
            audio_streams = [stream for stream in container.streams if stream.type == "audio"]
            require(len(video_streams) == 1, f"{path}: expected exactly one video stream")
            stream = video_streams[0]
            rate = stream.average_rate or stream.guessed_rate
            frames = tuple(frame.to_image().convert("RGB") for frame in container.decode(video=stream.index))
            fps = "" if rate is None else f"{Fraction(rate).numerator}/{Fraction(rate).denominator}"
            return DecodedVideo(
                frames=frames,
                video_streams=len(video_streams),
                audio_streams=len(audio_streams),
                decoded_frames=len(frames),
                fps=fps,
                width=int(stream.width),
                height=int(stream.height),
            )
    except ReviewPackageError:
        raise
    except Exception as exc:
        raise ReviewPackageError(f"{path}: media decode failed: {exc}") from exc


def decode_video_opencv_for_test(path: Path) -> DecodedVideo:
    """CPU-only test helper; formal CLI intentionally requires PyAV/audio probing."""
    try:
        import cv2
        from PIL import Image
    except ImportError as exc:
        raise ReviewPackageError("OpenCV and Pillow are required by the test decoder") from exc
    capture = cv2.VideoCapture(str(path))
    frames: list[Any] = []
    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_value = capture.get(cv2.CAP_PROP_FPS)
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    finally:
        capture.release()
    rate = Fraction(str(fps_value)).limit_denominator(1000)
    return DecodedVideo(
        frames=tuple(frames),
        video_streams=1,
        audio_streams=0,
        decoded_frames=len(frames),
        fps=f"{rate.numerator}/{rate.denominator}",
        width=width,
        height=height,
    )


def panel_frame_indices() -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(range(start, end + 1)) for start, end in PANEL_WINDOWS)


def render_composite(frames: Sequence[Any], output_path: Path) -> dict[str, Any]:
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise ReviewPackageError("Pillow is required to render review composites") from exc
    require(len(frames) == 49, "composite renderer requires exactly 49 decoded frames")
    canvas = Image.new("RGB", (COMPOSITE_WIDTH, COMPOSITE_HEIGHT), "white")
    draw = ImageDraw.Draw(canvas)
    row_height = PANEL_HEADER_HEIGHT + THUMB_HEIGHT + FRAME_LABEL_HEIGHT + PANEL_GAP
    for panel_index, (start, end) in enumerate(PANEL_WINDOWS):
        y = panel_index * row_height
        draw.rectangle((0, y, COMPOSITE_WIDTH, y + PANEL_HEADER_HEIGHT - 1), fill=(24, 24, 24))
        draw.text((5, y + 4), f"Temporal panel {panel_index + 1}/5 | frames {start:02d}-{end:02d}", fill="white")
        for column, frame_index in enumerate(range(start, end + 1)):
            image = frames[frame_index].convert("RGB")
            image.thumbnail((THUMB_WIDTH, THUMB_HEIGHT), Image.Resampling.LANCZOS)
            tile = Image.new("RGB", (THUMB_WIDTH, THUMB_HEIGHT), "white")
            tile.paste(image, ((THUMB_WIDTH - image.width) // 2, (THUMB_HEIGHT - image.height) // 2))
            x = column * THUMB_WIDTH
            canvas.paste(tile, (x, y + PANEL_HEADER_HEIGHT))
            draw.text((x + 3, y + PANEL_HEADER_HEIGHT + THUMB_HEIGHT + 1), f"f{frame_index:02d}", fill=(35, 35, 35))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(
        output_path,
        format="JPEG",
        quality=JPEG_QUALITY,
        optimize=False,
        progressive=False,
        subsampling=2,
    )
    size = output_path.stat().st_size
    require(size <= MAX_COMPOSITE_BYTES, f"composite exceeds Copilot image limit: {size} > {MAX_COMPOSITE_BYTES}")
    return {
        "width": canvas.width,
        "height": canvas.height,
        "quality": JPEG_QUALITY,
        "size_bytes": size,
        "sha256": sha256_file(output_path),
        "used_frame_indices": [list(indices) for indices in panel_frame_indices()],
    }


def _verify_decoded_media(row: Mapping[str, Any], decoded: DecodedVideo) -> None:
    expected_width = 720 if row["stream"] in COGVIDEOX_STREAMS else 832
    expected = {
        "video_streams": 1,
        "audio_streams": 0,
        "decoded_frames": 49,
        "fps": "8/1",
        "width": expected_width,
        "height": 480,
    }
    require(decoded.media() == expected, f"{row['ledger_id']}: physical media mismatch: {decoded.media()} != {expected}")


def _anonymous_id(blind_key: bytes, ledger_record_sha256: str) -> str:
    digest = hmac.new(blind_key, b"review-id-v1\0" + ledger_record_sha256.encode("ascii"), hashlib.sha256).hexdigest()
    return "rv_" + digest[:32]


def _score_template(assignment: Mapping[str, Any]) -> dict[str, Any]:
    fields = CAUSAL_SCORE_FIELDS if assignment["case_kind"] == "causal" else SPECIFICITY_SCORE_FIELDS
    return {
        "anonymous_review_id": assignment["anonymous_review_id"],
        "assignment_sha256": assignment["assignment_sha256"],
        "case_kind": assignment["case_kind"],
        "scores": {field: None for field in fields},
        "confidence": {field: None for field in fields},
        "evidence_frames": {field: [] for field in fields},
        "evidence_observations": {field: [] for field in fields},
        "unusable_reason": "",
        "status": "pending",
    }


def _public_assignment(row: Mapping[str, Any], anonymous_id: str, composite_sha256: str) -> dict[str, Any]:
    assignment: dict[str, Any] = {
        "anonymous_review_id": anonymous_id,
        "case_kind": row["case_kind"],
        "mechanism_name": row["mechanism_name"],
        "prompt": row["prompt"],
        "source_object": row["source_object"],
        "receiver": row["receiver"],
        "expected_trigger": row["expected_trigger"],
        "expected_footprint": row["expected_footprint"],
        "expected_counterfactual_state": row["expected_counterfactual_state"],
        "protected_object": row["protected_object"],
        "specificity_subtype": row["specificity_subtype"],
        "acceptable_alternative_cause": row["acceptable_alternative_cause"],
        "composite_path": f"media/{anonymous_id}.jpg",
        "composite_sha256": composite_sha256,
    }
    assignment["assignment_sha256"] = sha256_bytes(canonical_json_bytes(assignment))
    require(tuple(assignment) == PUBLIC_ASSIGNMENT_FIELDS, "internal public-assignment field order changed")
    return assignment


def _backbone_family(stream: str) -> str:
    return "cogvideox" if stream in COGVIDEOX_STREAMS else "wan"


def _private_key_row(row: Mapping[str, Any], anonymous_id: str, composite: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "anonymous_review_id": anonymous_id,
        "ledger_id": row["ledger_id"],
        "ledger_record_sha256": row["ledger_record_sha256"],
        "evaluation_partition": row["evaluation_partition"],
        "stream": row["stream"],
        "backbone_family": _backbone_family(str(row["stream"])),
        "case_id": row["case_id"],
        "formal_global_index": row["formal_global_index"],
        "identification_case_index": row["identification_case_index"],
        "mechanism": row["mechanism"],
        "case_kind": row["case_kind"],
        "seed": row["seed"],
        "video_sha256": row["video_sha256"],
        "source_video_path": row["source_video_path"],
        "source_manifest": row["source_manifest"],
        "composite_sha256": composite["sha256"],
        "composite_size_bytes": composite["size_bytes"],
    }


def _shuffle_rows(rows: Sequence[dict[str, Any]], blind_key: bytes, pass_id: str) -> list[dict[str, Any]]:
    def rank(row: Mapping[str, Any]) -> bytes:
        return hmac.new(
            blind_key,
            f"review-order-{pass_id}-v1\0{row['anonymous_review_id']}".encode("utf-8"),
            hashlib.sha256,
        ).digest()

    return sorted(rows, key=rank)


def _assert_public_is_blind(pass_root: Path, assignments: Sequence[Mapping[str, Any]]) -> None:
    banned_fields = {"stream", "arm", "backbone", "video_path", "source_video_path", "case_id", "seed"}
    banned_tokens = tuple((*MAIN_STREAMS, *IDENTIFICATION_STREAMS))
    for row in assignments:
        require(set(row) == set(PUBLIC_ASSIGNMENT_FIELDS), "public assignment contains an unexpected field")
        require(not (set(row) & banned_fields), "public assignment leaks a private field")
        composite = Path(str(row["composite_path"]))
        require(not composite.is_absolute() and composite.parts[:1] == ("media",) and ".." not in composite.parts, "public composite path escapes pass media")
        for field, value in row.items():
            if field in ("composite_path", "composite_sha256", "assignment_sha256", "anonymous_review_id"):
                continue
            text = str(value)
            require(not any(token in text for token in banned_tokens), f"public assignment leaks a stream token in {field}")
    public_text = (pass_root / "assignments.jsonl").read_text(encoding="utf-8")
    require("source_video_path" not in public_text and '"stream"' not in public_text and '"backbone"' not in public_text, "serialized public assignment leaks provenance")


def _build_into(
    *,
    staging: Path,
    ledger: list[dict[str, Any]],
    blind_key: bytes,
    input_bindings: Mapping[str, Mapping[str, Any]],
    decoder: Decoder,
    renderer: Renderer,
) -> dict[str, Any]:
    public_root = staging / "public"
    private_root = staging / "private"
    pass_roots = {"A": public_root / "pass_a", "B": public_root / "pass_b"}
    public_root.mkdir(mode=0o755)
    private_root.mkdir(mode=0o700)
    for root in pass_roots.values():
        (root / "media").mkdir(parents=True, mode=0o755)

    decoded_cache: dict[str, DecodedVideo] = {}
    rendered_cache: dict[str, tuple[Path, dict[str, Any]]] = {}
    assignments: list[dict[str, Any]] = []
    key_rows: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    for row in ledger:
        anonymous_id = _anonymous_id(blind_key, str(row["ledger_record_sha256"]))
        require(anonymous_id not in used_ids, "anonymous review ID collision")
        used_ids.add(anonymous_id)
        target_a = pass_roots["A"] / "media" / f"{anonymous_id}.jpg"
        cached = rendered_cache.get(str(row["video_sha256"]))
        if cached is None:
            decoded = decoded_cache.get(str(row["video_sha256"]))
            if decoded is None:
                decoded = decoder(Path(str(row["source_video_path"])))
                decoded_cache[str(row["video_sha256"])] = decoded
            _verify_decoded_media(row, decoded)
            composite = renderer(decoded.frames, target_a)
            require(
                composite.get("width") == COMPOSITE_WIDTH
                and composite.get("height") == COMPOSITE_HEIGHT
                and composite.get("quality") == JPEG_QUALITY,
                f"{row['ledger_id']}: composite geometry/quality changed",
            )
            require(
                composite.get("used_frame_indices") == [list(indices) for indices in panel_frame_indices()],
                f"{row['ledger_id']}: composite did not use the frozen frame windows",
            )
            regular_file(target_a, f"{anonymous_id} composite")
            require(target_a.stat().st_size == composite.get("size_bytes"), f"{anonymous_id}: composite size receipt mismatch")
            require(target_a.stat().st_size <= MAX_COMPOSITE_BYTES, f"{anonymous_id}: composite exceeds image limit")
            require(sha256_file(target_a) == composite.get("sha256"), f"{anonymous_id}: composite SHA receipt mismatch")
            rendered_cache[str(row["video_sha256"])] = (target_a, dict(composite))
        else:
            source_composite, composite = cached
            cached_decoded = decoded_cache.get(str(row["video_sha256"]))
            require(cached_decoded is not None, "internal decoded-media cache is incomplete")
            _verify_decoded_media(row, cached_decoded)
            os.link(source_composite, target_a)
        target_b = pass_roots["B"] / "media" / f"{anonymous_id}.jpg"
        os.link(target_a, target_b)
        require(sha256_file(target_b) == composite["sha256"], f"{anonymous_id}: pass B composite differs")
        assignments.append(_public_assignment(row, anonymous_id, str(composite["sha256"])))
        key_rows.append(_private_key_row(row, anonymous_id, composite))

    require(len(assignments) == TOTAL_VIDEO_COUNT and len(used_ids) == TOTAL_VIDEO_COUNT, "anonymous ID inventory is not exact")
    assignment_by_id = {row["anonymous_review_id"]: row for row in assignments}
    pass_orders: dict[str, list[str]] = {}
    for pass_id, root in pass_roots.items():
        ordered = _shuffle_rows(assignments, blind_key, pass_id)
        pass_orders[pass_id] = [str(row["anonymous_review_id"]) for row in ordered]
        scores = [_score_template(row) for row in ordered]
        write_jsonl(root / "assignments.jsonl", ordered)
        write_jsonl(root / "scores.jsonl", scores)
        pass_manifest = {
            "protocol": PACKAGE_PROTOCOL,
            "schema_version": SCHEMA_VERSION,
            "pass_id": pass_id,
            "item_count": TOTAL_VIDEO_COUNT,
            "panel_windows": [list(window) for window in PANEL_WINDOWS],
            "assignments": {"path": "assignments.jsonl", "sha256": sha256_file(root / "assignments.jsonl")},
            "blank_scores": {"path": "scores.jsonl", "sha256": sha256_file(root / "scores.jsonl")},
            "composite_contract": {
                "path_base": "pass_root",
                "directory": "media",
                "format": "jpeg",
                "width": COMPOSITE_WIDTH,
                "height": COMPOSITE_HEIGHT,
                "jpeg_quality": JPEG_QUALITY,
                "max_bytes": MAX_COMPOSITE_BYTES,
                "one_image_per_assignment": True,
            },
            "ordering_commitment": sha256_bytes(
                hmac.new(blind_key, f"review-order-{pass_id}-v1".encode("ascii"), hashlib.sha256).digest()
            ),
        }
        write_json(root / "pass_manifest.json", pass_manifest)
        _assert_public_is_blind(root, ordered)
    require(pass_orders["A"] != pass_orders["B"], "A and B review orders are not independent")
    require(set(pass_orders["A"]) == set(pass_orders["B"]) == set(assignment_by_id), "A/B anonymous inventories differ")

    full_key = sorted(key_rows, key=lambda row: row["anonymous_review_id"])
    original_key = [row for row in full_key if row["stream"] in ORIGINAL_STREAMS]
    require(len(full_key) == TOTAL_VIDEO_COUNT, "full private key count changed")
    require(len(original_key) == 588, "Original-only private key must contain exactly 588 rows")
    ledger_path = private_root / "generation_ledger.jsonl"
    original_key_path = private_root / "original_only_key.jsonl"
    full_key_path = private_root / "full_key.jsonl"
    write_jsonl(ledger_path, ledger, mode=0o600)
    write_jsonl(original_key_path, original_key, mode=0o600)
    write_jsonl(full_key_path, full_key, mode=0o600)
    for path in (ledger_path, original_key_path, full_key_path):
        require(stat.S_IMODE(path.stat().st_mode) == 0o600, f"private artifact mode is not 0600: {path.name}")
    commitments = {
        "protocol": PACKAGE_PROTOCOL,
        "schema_version": SCHEMA_VERSION,
        "commitment_scheme": "sha256(canonical-jsonl-bytes)",
        "tier_1_original_only": {"row_count": 588, "sha256": sha256_file(original_key_path)},
        "tier_2_full": {"row_count": TOTAL_VIDEO_COUNT, "sha256": sha256_file(full_key_path)},
        "generation_ledger": {"row_count": TOTAL_VIDEO_COUNT, "sha256": sha256_file(ledger_path)},
        "blind_key_sha256": sha256_bytes(blind_key),
    }
    write_json(public_root / "key_commitments.json", commitments)
    receipt = {
        "protocol": LEDGER_PROTOCOL,
        "schema_version": SCHEMA_VERSION,
        "status": "frozen_after_exact_2448_media_and_blinding_validation",
        "main_video_count": MAIN_VIDEO_COUNT,
        "identification_video_count": IDENTIFICATION_VIDEO_COUNT,
        "total_video_count": TOTAL_VIDEO_COUNT,
        "main_streams": list(MAIN_STREAMS),
        "identification_streams": list(IDENTIFICATION_STREAMS),
        "inputs": dict(input_bindings),
        "private_artifacts": {
            "generation_ledger": {"path": "generation_ledger.jsonl", "sha256": sha256_file(ledger_path)},
            "original_only_key": {"path": "original_only_key.jsonl", "sha256": sha256_file(original_key_path)},
            "full_key": {"path": "full_key.jsonl", "sha256": sha256_file(full_key_path)},
        },
        "public_artifacts": {
            "pass_a_manifest_sha256": sha256_file(pass_roots["A"] / "pass_manifest.json"),
            "pass_b_manifest_sha256": sha256_file(pass_roots["B"] / "pass_manifest.json"),
            "key_commitments_sha256": sha256_file(public_root / "key_commitments.json"),
        },
    }
    write_json(private_root / "package_receipt.json", receipt, mode=0o600)
    require(stat.S_IMODE(private_root.stat().st_mode) == 0o700, "private directory mode is not 0700")
    return receipt


def build_review_package(
    *,
    project_root: Path,
    formal_cases_path: Path,
    identification_path: Path,
    wan_original_manifest_path: Path,
    trained_wan_manifest_path: Path,
    cog_core_manifest_path: Path,
    safree_manifest_path: Path,
    blind_key_path: Path,
    output_dir: Path,
    decoder: Decoder = decode_video_pyav,
    renderer: Renderer = render_composite,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    require(project_root.is_dir(), f"project root is not a directory: {project_root}")
    inputs = {
        "formal_cases": resolve_input(project_root, formal_cases_path),
        "identification_subset": resolve_input(project_root, identification_path),
        "wan_original_manifest": resolve_input(project_root, wan_original_manifest_path),
        "trained_wan_manifest": resolve_input(project_root, trained_wan_manifest_path),
        "cog_core_manifest": resolve_input(project_root, cog_core_manifest_path),
        "safree_manifest": resolve_input(project_root, safree_manifest_path),
    }
    blind_key = resolve_input(project_root, blind_key_path).read_bytes()
    require(len(blind_key) >= 32, "blind key file must contain at least 32 bytes")
    output_dir = output_dir if output_dir.is_absolute() else project_root / output_dir
    require(not output_dir.exists() and not output_dir.is_symlink(), f"output directory already exists: {output_dir}")

    formal_rows = load_csv(inputs["formal_cases"], "formal cases")
    identification_rows = load_csv(inputs["identification_subset"], "identification subset")
    manifests = {
        "wan_original": load_json(inputs["wan_original_manifest"], "Wan Original manifest"),
        "trained_wan": load_json(inputs["trained_wan_manifest"], "trained-Wan manifest"),
        "cog_core": load_json(inputs["cog_core_manifest"], "CogVideoX core manifest"),
        "safree": load_json(inputs["safree_manifest"], "SAFREE manifest"),
    }
    validate_manifest_input_bindings(
        project_root=project_root,
        formal_path=inputs["formal_cases"],
        identification_path=inputs["identification_subset"],
        manifests=manifests,
    )
    ledger = build_generation_ledger(
        project_root=project_root,
        formal_rows=formal_rows,
        identification_rows=identification_rows,
        wan_original=manifests["wan_original"],
        trained_wan=manifests["trained_wan"],
        cog_core=manifests["cog_core"],
        safree=manifests["safree"],
    )
    input_bindings: dict[str, dict[str, Any]] = {}
    for label, path in inputs.items():
        try:
            recorded_path = str(path.relative_to(project_root))
        except ValueError:
            recorded_path = str(path)
        input_bindings[label] = {"path": recorded_path, "sha256": sha256_file(path)}
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent))
    staging.chmod(0o755)
    try:
        receipt = _build_into(
            staging=staging,
            ledger=ledger,
            blind_key=blind_key,
            input_bindings=input_bindings,
            decoder=decoder,
            renderer=renderer,
        )
        require(not output_dir.exists(), f"output directory appeared during build: {output_dir}")
        os.replace(staging, output_dir)
        return receipt
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--formal-cases",
        type=Path,
        default=Path("data/causal_role_erasure_7mechanism_main_v2/formal_cases.csv"),
    )
    parser.add_argument(
        "--identification-subset",
        type=Path,
        default=Path("data/causal_role_erasure_7mechanism_main_v2/identification_subset.csv"),
    )
    parser.add_argument("--wan-original-manifest", type=Path, required=True)
    parser.add_argument("--trained-wan-manifest", type=Path, required=True)
    parser.add_argument("--cog-core-manifest", type=Path, required=True)
    parser.add_argument("--safree-manifest", type=Path, required=True)
    parser.add_argument("--blind-key-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        receipt = build_review_package(
            project_root=args.project_root,
            formal_cases_path=args.formal_cases,
            identification_path=args.identification_subset,
            wan_original_manifest_path=args.wan_original_manifest,
            trained_wan_manifest_path=args.trained_wan_manifest,
            cog_core_manifest_path=args.cog_core_manifest,
            safree_manifest_path=args.safree_manifest,
            blind_key_path=args.blind_key_file,
            output_dir=args.output_dir,
        )
    except ReviewPackageError as exc:
        parser.exit(2, f"review package refused: {exc}\n")
    print(
        f"Frozen {receipt['total_video_count']} anonymous review assignments "
        f"({receipt['main_video_count']} main + {receipt['identification_video_count']} identification)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
