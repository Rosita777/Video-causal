#!/usr/bin/env python3
"""Deterministically select the private v4 causal/specificity evaluation sets.

Run this only in the isolated evaluator environment.  The script contains no
ontology entries and writes no public prompt text; its outputs are private
artifacts later bound by the Stage-1 commitment registry.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import stat
import statistics
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import water_impact_dynamic_v4_eval_protocol as protocol
from build_water_impact_dynamic_v4_blind_review import build_screening_composite


SCREENING_FREEZE_PROTOCOL = "water_impact_dynamic_v4_screening_freeze_v2"
SCREENING_PACKAGE_PROTOCOL = "water_impact_dynamic_v4_screening_review_package_v2"
PUBLIC_SCREENING_ADJUDICATION_COLUMNS = (
    "review_id",
    "field",
    "score",
    "brief_reason",
)
CAUSAL_SCREENING_FIELDS = {
    "source": "source_visibility_0_absent_2_clear",
    "footprint": "footprint_visibility_0_absent_2_clear",
    "receiver": "receiver_preservation_0_bad_2_good",
    "quality": "video_quality_0_bad_2_good",
    "causal_link": "causal_link_0_absent_2_clear",
}
SPECIFICITY_SCREENING_FIELDS = {
    "protected": "protected_object_visibility_0_absent_2_clear",
    "receiver": "receiver_preservation_0_bad_2_good",
    "quality": "video_quality_0_bad_2_good",
    "adherence": "noncausal_role_adherence_0_bad_2_good",
}


def _screening_fields(dataset: str) -> dict[str, str]:
    if dataset == "causal":
        return CAUSAL_SCREENING_FIELDS
    if dataset == "specificity":
        return SPECIFICITY_SCREENING_FIELDS
    raise ValueError("unknown screening dataset")


def validate_screening_generation_manifest(
    project_root: Path,
    *,
    dataset: str,
    normalized_candidates: Sequence[Mapping[str, Any]],
    candidate_manifest_path: Path,
    screening_seed_path: Path,
    generation_spec_path: Path,
    stage0_registry_path: Path,
    generation_manifest_path: Path,
    decode: Callable[[Path], Mapping[str, int]] = protocol._default_decode,
    verify_model_bytes: bool = True,
) -> dict[str, Path]:
    """Recompute the exact Stage-0 Original generation/video inventory."""

    expected_n = protocol.CANDIDATE_COUNTS[dataset]
    if len(normalized_candidates) != expected_n:
        raise ValueError("screening generation candidate count differs from protocol")
    seed_text = screening_seed_path.read_text(encoding="ascii")
    if not re.fullmatch(r"(?:0|[1-9][0-9]*)\n", seed_text):
        raise ValueError("screening seed opening is not canonical decimal plus LF")
    seed = int(seed_text[:-1])
    spec = json.loads(generation_spec_path.read_text(encoding="utf-8"))
    runtime_ref = spec.get("runtime_registry") if isinstance(spec, dict) else None
    if (
        not isinstance(spec, dict)
        or set(spec)
        != {
            "protocol",
            "status",
            "model_inventory_sha256",
            "runtime_registry",
            "generation_spec",
            "source_mode",
        }
        or spec["protocol"] != protocol.GENERATION_SPEC_PROTOCOL
        or spec["status"] != "frozen_before_original_render"
        or spec["source_mode"] != "Original_screening_then_matched_O_v3b_v4"
        or spec["generation_spec"] != protocol.GENERATION_SPEC
        or not isinstance(runtime_ref, dict)
        or set(runtime_ref) != {"path", "sha256"}
        or runtime_ref["path"] != protocol.RUNTIME_REGISTRY
        or not protocol.is_sha256(runtime_ref["sha256"])
    ):
        raise ValueError("screening generation spec differs from exact protocol")
    protocol.validate_runtime_registry(
        protocol.resolve_path(project_root, protocol.RUNTIME_REGISTRY),
        runtime_ref["sha256"],
    )
    model_sha256 = spec.get("model_inventory_sha256")
    if model_sha256 != protocol.FROZEN_MODEL_CONTENT_INVENTORY_SHA256:
        raise ValueError("screening generation spec differs from frozen full-model inventory")
    if verify_model_bytes:
        actual_model = protocol.model_artifact_inventory(
            project_root, protocol.GENERATION_SPEC["model"]
        )
        if actual_model["sha256"] != model_sha256:
            raise ValueError("screening generation model inventory drifted")
    payload = json.loads(generation_manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != {
        "protocol",
        "dataset",
        "dataset_version",
        "method",
        "stage0_registry_sha256",
        "candidate_manifest_sha256",
        "screening_seed_sha256",
        "model_inventory_sha256",
        "runtime_registry_sha256",
        "raw_generation_manifest",
        "generation_spec",
        "videos",
    }:
        raise ValueError("screening generation manifest fields are not exact")
    if (
        payload["protocol"] != "water_impact_dynamic_v4_screening_generation_v2"
        or payload["dataset"] != dataset
        or payload["dataset_version"] != protocol.DATASET_VERSION
        or payload["method"] != "original"
        or payload["stage0_registry_sha256"] != protocol.file_sha256(stage0_registry_path)
        or payload["candidate_manifest_sha256"]
        != protocol.file_sha256(candidate_manifest_path)
        or payload["screening_seed_sha256"] != protocol.file_sha256(screening_seed_path)
        or payload["model_inventory_sha256"] != model_sha256
        or payload["runtime_registry_sha256"]
        != spec.get("runtime_registry", {}).get("sha256")
        or payload["generation_spec"] != protocol.GENERATION_SPEC
    ):
        raise ValueError("screening generation manifest provenance differs from Stage-0")
    raw_ref = payload["raw_generation_manifest"]
    if not isinstance(raw_ref, dict) or set(raw_ref) != {"path", "sha256"}:
        raise ValueError("screening raw-generation reference is not exact")
    raw_path = protocol.resolve_path(project_root, str(raw_ref["path"]))
    if (
        not raw_path.is_file()
        or raw_path.is_symlink()
        or protocol.file_sha256(raw_path) != raw_ref["sha256"]
        or raw_path.name != "generation_manifest.json"
        or generation_manifest_path.name != "v4_screening_generation_manifest_v2.json"
        or generation_manifest_path.parent.resolve() != raw_path.parent.resolve()
    ):
        raise ValueError("screening raw-generation manifest byte binding mismatch")
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != {
        "created_at_utc",
        "baseline",
        "pipeline",
        "model",
        "dry_run",
        "prompts",
        "generation",
        "items",
    }:
        raise ValueError("raw screening generator manifest fields are not exact")
    prompt_path = raw_path.parent / "prompts.txt"
    expected_prompt_text = "".join(
        f"{row['prompt']} | {row['source_phrase']} | registered v4 evaluation\n"
        for row in normalized_candidates
    )
    if (
        not isinstance(raw["created_at_utc"], str)
        or not raw["created_at_utc"].strip()
        or raw["baseline"] != "clean"
        or raw["pipeline"] != "WanPipeline"
        or raw["model"] != protocol.GENERATION_SPEC["model"]
        or raw["dry_run"] is not False
        or raw["prompts"] != str(prompt_path)
        or not prompt_path.is_file()
        or prompt_path.is_symlink()
        or prompt_path.read_text(encoding="utf-8") != expected_prompt_text
    ):
        raise ValueError("raw screening pipeline/prompt reservation differs from protocol")
    expected_generation = {
        "baseline": "clean",
        "seed": 42,
        "seeds": [seed] * expected_n,
        "num_inference_steps": 25,
        "guidance_scale": 5.0,
        "num_frames": 49,
        "fps": 8,
        "height": 480,
        "width": 832,
        "dtype": "bf16",
        "device": "cuda",
        "enable_model_cpu_offload": False,
        "enable_sequential_cpu_offload": False,
        "vae_slicing": True,
        "vae_tiling": True,
        "prompt_encode_device_policy": "cpu_when_offloaded_else_selected_device",
        "lora_path": None,
        "lora_sha256": None,
        "lora_scale": 1.0,
        "activation_gate_dir": None,
        "persistent_activation_gate": False,
        "lora_target_phrases": [],
        "attention_gate_dir": None,
        "attention_suppression_phrases": [],
        "attention_suppression_strength": 20.0,
    }
    if raw["generation"] != expected_generation:
        raise ValueError("raw screening generator configuration differs from protocol")
    reservation_path = raw_path.parent / ".run_reservation_v2.json"
    reservation = json.loads(reservation_path.read_text(encoding="utf-8"))
    reservation_fields = {
        "protocol",
        "dataset",
        "dataset_version",
        "method",
        "stage0_registry_sha256",
        "candidate_manifest_sha256",
        "screening_seed_sha256",
        "model_inventory_sha256",
        "runtime_registry_sha256",
    }
    if reservation != {field: payload[field] for field in reservation_fields}:
        raise ValueError("screening run reservation differs from final generation provenance")
    raw_items = raw["items"]
    if not isinstance(raw_items, list) or len(raw_items) != expected_n:
        raise ValueError("raw screening generator item inventory differs from protocol")
    records = payload["videos"]
    if not isinstance(records, list) or len(records) != expected_n:
        raise ValueError("screening generation video count differs from protocol")
    videos: dict[str, Path] = {}
    paths: set[Path] = set()
    inodes: set[tuple[int, int]] = set()
    hashes: set[str] = set()
    expected_media = {
        "frame_count": protocol.FRAME_COUNT,
        "width": protocol.WIDTH,
        "height": protocol.HEIGHT,
        "fps_numerator": protocol.FPS.numerator,
        "fps_denominator": protocol.FPS.denominator,
    }
    for index, (candidate, record) in enumerate(zip(normalized_candidates, records)):
        if not isinstance(record, dict) or set(record) != {
            "unit_id",
            "index",
            "path",
            "size_bytes",
            "sha256",
            "prompt_sha256",
            "seed",
            *expected_media,
        }:
            raise ValueError("screening generation video record fields are not exact")
        candidate_id = str(candidate["candidate_id"])
        expected_unit = f"screen_{dataset[0]}_{index:03d}"
        path = protocol.resolve_path(project_root, str(record["path"]))
        raw_item = raw_items[index]
        if not isinstance(raw_item, dict) or set(raw_item) != {
            "index",
            "prompt",
            "target_concept",
            "expected_effect",
            "seed",
            "video_path",
        } or (
            raw_item["index"] != index
            or raw_item["prompt"] != str(candidate["prompt"])
            or raw_item["target_concept"] != str(candidate["source_phrase"])
            or raw_item["expected_effect"] != "registered v4 evaluation"
            or raw_item["seed"] != seed
            or str(raw_item["video_path"]) != str(record["path"])
        ):
            raise ValueError("raw screening generator item differs from candidate/order")
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError("screening generation source video is missing")
        if path.parent.resolve() != (raw_path.parent / "videos").resolve():
            raise ValueError("screening generation video escapes the reserved raw run")
        resolved = path.resolve(strict=True)
        inode = (path.stat().st_dev, path.stat().st_ino)
        digest = protocol.file_sha256(path)
        if resolved in paths or inode in inodes or digest in hashes:
            raise ValueError("screening generation reuses a path/inode/content hash")
        paths.add(resolved)
        inodes.add(inode)
        hashes.add(digest)
        if (
            record["unit_id"] != expected_unit
            or record["index"] != index
            or record["seed"] != seed
            or record["prompt_sha256"]
            != hashlib.sha256(str(candidate["prompt"]).encode("utf-8")).hexdigest()
            or record["size_bytes"] != path.stat().st_size
            or record["sha256"] != digest
            or {key: record[key] for key in expected_media} != expected_media
            or dict(decode(path)) != expected_media
        ):
            raise ValueError("screening generation video differs from candidate/media contract")
        videos[candidate_id] = path
    if len(videos) != expected_n:
        raise ValueError("screening generation candidate IDs are duplicate")
    if {path.name for path in raw_path.parent.iterdir()} != {
        ".run_reservation_v2.json",
        "prompts.txt",
        "generation_manifest.json",
        "videos",
        "v4_screening_generation_manifest_v2.json",
    }:
        raise ValueError("screening raw generation directory inventory is not exact")
    if set((raw_path.parent / "videos").iterdir()) != set(videos.values()):
        raise ValueError("screening raw video inventory is not exact")
    return videos


def build_screening_review_package(
    *,
    project_root: Path,
    dataset: str,
    normalized_candidates: Sequence[Mapping[str, Any]],
    candidate_manifest_path: Path,
    screening_seed_path: Path,
    generation_spec_path: Path,
    stage0_registry_path: Path,
    generation_manifest_path: Path,
    public_dir: Path,
    private_dir: Path,
    composite_builder: Callable[[Path, Path], None] = build_screening_composite,
    decode: Callable[[Path], Mapping[str, int]] = protocol._default_decode,
    verify_model_bytes: bool = True,
) -> dict[str, Any]:
    if public_dir.exists() or private_dir.exists():
        raise FileExistsError("refusing to overwrite screening public/private package")
    if (
        public_dir.resolve() == private_dir.resolve()
        or public_dir.parent.resolve() != private_dir.parent.resolve()
    ):
        raise ValueError("screening public/private packages must be distinct siblings")
    videos = validate_screening_generation_manifest(
        project_root,
        dataset=dataset,
        normalized_candidates=normalized_candidates,
        candidate_manifest_path=candidate_manifest_path,
        screening_seed_path=screening_seed_path,
        generation_spec_path=generation_spec_path,
        stage0_registry_path=stage0_registry_path,
        generation_manifest_path=generation_manifest_path,
        decode=decode,
        verify_model_bytes=verify_model_bytes,
    )
    public_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(prefix=".v4-screening-package-", dir=public_dir.parent)
    )
    work_public = staging_root / "public"
    work_private = staging_root / "private"
    published: list[Path] = []
    try:
        work_public.mkdir()
        work_private.mkdir()
        (work_public / "media").mkdir()
        (work_public / "composites").mkdir()
        fields = tuple(_screening_fields(dataset).values())
        public_rows: list[dict[str, Any]] = []
        key_rows: list[dict[str, Any]] = []
        internal_rows: list[dict[str, Any]] = []
        media_hashes: dict[str, str] = {}
        composite_hashes: dict[str, str] = {}
        seed = int(screening_seed_path.read_text(encoding="ascii").strip())
        for index, candidate in enumerate(normalized_candidates):
            review_id = f"s{index:03d}"
            candidate_id = str(candidate["candidate_id"])
            source = videos[candidate_id]
            anonymous = work_public / "media" / f"{review_id}.mp4"
            final_anonymous = public_dir / "media" / anonymous.name
            shutil.copyfile(source, anonymous)
            if anonymous.is_symlink() or anonymous.samefile(source):
                raise ValueError("screening anonymous media must be an independent copy")
            source_sha = protocol.file_sha256(source)
            anonymous_sha = protocol.file_sha256(anonymous)
            if source_sha != anonymous_sha:
                raise ValueError("screening anonymous video differs from generated source")
            composite = work_public / "composites" / f"{review_id}.jpg"
            final_composite = public_dir / "composites" / composite.name
            composite_builder(composite, anonymous)
            if not composite.is_file() or composite.is_symlink() or composite.stat().st_size <= 0:
                raise ValueError("screening composite builder produced no real image")
            public_rows.append(
                {
                    "review_id": review_id,
                    "object_phrase": str(candidate["source_phrase"]),
                    "receiver_description": str(candidate["receiver"]),
                    "video_path": str(final_anonymous),
                    "composite_path": str(final_composite),
                    **{field: "" for field in fields},
                    "notes": "",
                }
            )
            key_rows.append(
                {
                    "review_id": review_id,
                    "candidate_id": candidate_id,
                    "source_video_path": str(source),
                    "source_video_sha256": source_sha,
                    "anonymous_video_path": str(final_anonymous),
                    "anonymous_video_sha256": anonymous_sha,
                }
            )
            internal_rows.append(
                {
                    **dict(candidate),
                    "unit_id": f"screen_{dataset[0]}_{index:03d}",
                    "seed": seed,
                    "screening_video_path": str(source),
                    "screening_video_sha256": source_sha,
                }
            )
            media_hashes[anonymous.name] = anonymous_sha
            composite_hashes[composite.name] = protocol.file_sha256(composite)
        work_public_template = work_public / "screening_review_v2.csv"
        work_answer_key = work_private / "screening_answer_key_v2.csv"
        work_candidates = work_private / "screening_candidates_v2.csv"
        protocol.write_csv(work_public_template, public_rows)
        protocol.write_csv(work_answer_key, key_rows)
        protocol.write_csv(work_candidates, internal_rows)
        public_template = public_dir / work_public_template.name
        answer_key = private_dir / work_answer_key.name
        candidates_path = private_dir / work_candidates.name
        manifest = {
            "protocol": SCREENING_PACKAGE_PROTOCOL,
            "dataset": dataset,
            "dataset_version": protocol.DATASET_VERSION,
            "stage0_registry": {
                "path": str(stage0_registry_path),
                "sha256": protocol.file_sha256(stage0_registry_path),
            },
            "raw_candidate_manifest": {
                "path": str(candidate_manifest_path),
                "sha256": protocol.file_sha256(candidate_manifest_path),
            },
            "generation_manifest": {
                "path": str(generation_manifest_path),
                "sha256": protocol.file_sha256(generation_manifest_path),
            },
            "candidate_projection": {
                "path": str(candidates_path),
                "sha256": protocol.file_sha256(work_candidates),
            },
            "public_template": {
                "path": str(public_template),
                "sha256": protocol.file_sha256(work_public_template),
            },
            "answer_key": {
                "path": str(answer_key),
                "sha256": protocol.file_sha256(work_answer_key),
            },
            "anonymous_media_sha256": media_hashes,
            "composite_sha256": composite_hashes,
        }
        (work_private / "screening_package_manifest_v2.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.rename(work_private, private_dir)
        published.append(private_dir)
        os.rename(work_public, public_dir)
        published.append(public_dir)
        return manifest
    except BaseException:
        for path in reversed(published):
            shutil.rmtree(path)
        raise
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)


def validate_screening_review_package(
    project_root: Path,
    *,
    dataset: str,
    manifest_path: Path,
    private_root: Path,
    candidate_manifest_path: Path,
    canonical_templates_path: Path,
    screening_seed_path: Path,
    generation_spec_path: Path,
    decode: Callable[[Path], Mapping[str, int]] = protocol._default_decode,
    verify_model_bytes: bool = True,
) -> tuple[dict[str, Any], list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    if not private_root.is_dir() or private_root.is_symlink():
        raise ValueError("screening package private root must be a real directory")
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise FileNotFoundError("screening package manifest is missing")
    try:
        manifest_path.resolve(strict=True).relative_to(private_root.resolve(strict=True))
    except ValueError as exc:
        raise ValueError("screening package manifest escapes private root") from exc
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or set(manifest) != {
        "protocol",
        "dataset",
        "dataset_version",
        "stage0_registry",
        "raw_candidate_manifest",
        "generation_manifest",
        "candidate_projection",
        "public_template",
        "answer_key",
        "anonymous_media_sha256",
        "composite_sha256",
    }:
        raise ValueError("screening package manifest fields are not exact")
    if (
        manifest["protocol"] != SCREENING_PACKAGE_PROTOCOL
        or manifest["dataset"] != dataset
        or manifest["dataset_version"] != protocol.DATASET_VERSION
    ):
        raise ValueError("screening package protocol/dataset mismatch")
    refs: dict[str, Path] = {}
    for name in (
        "stage0_registry",
        "raw_candidate_manifest",
        "generation_manifest",
        "candidate_projection",
        "public_template",
        "answer_key",
    ):
        ref = manifest[name]
        if not isinstance(ref, dict) or set(ref) != {"path", "sha256"}:
            raise ValueError(f"screening package/{name}: reference is not exact")
        path = protocol.resolve_path(project_root, str(ref["path"]))
        if not path.is_file() or path.is_symlink() or protocol.file_sha256(path) != ref["sha256"]:
            raise ValueError(f"screening package/{name}: byte hash mismatch")
        refs[name] = path
    resolved_private = private_root.resolve(strict=True)
    for name in (
        "raw_candidate_manifest",
        "generation_manifest",
        "candidate_projection",
        "answer_key",
    ):
        try:
            refs[name].resolve(strict=True).relative_to(resolved_private)
        except ValueError as exc:
            raise ValueError(f"screening package/{name}: escapes private root") from exc
    expected_stage0 = protocol.resolve_path(
        project_root,
        protocol.CAUSAL_STAGE0 if dataset == "causal" else protocol.SPECIFICITY_STAGE0,
    )
    if refs["stage0_registry"].resolve() != expected_stage0.resolve():
        raise ValueError("screening package Stage-0 registry path differs from protocol")
    stage0 = protocol.validate_commitment_registry(
        refs["stage0_registry"], dataset=dataset, stage=0
    )
    if refs["raw_candidate_manifest"].resolve() != candidate_manifest_path.resolve():
        raise ValueError("screening package points at a different raw candidate manifest")
    candidate_commitment = stage0["artifacts"][
        f"candidate_manifest_{protocol.CANDIDATE_COUNTS[dataset]}"
    ]
    if (
        protocol.file_sha256(candidate_manifest_path) != candidate_commitment["sha256"]
        or candidate_manifest_path.stat().st_size != candidate_commitment["size_bytes"]
    ):
        raise ValueError("screening package raw candidate bytes differ from Stage-0")
    if protocol.file_sha256(screening_seed_path) != stage0["artifacts"]["screening_seed"]["sha256"]:
        raise ValueError("screening package seed bytes differ from Stage-0")
    if protocol.file_sha256(generation_spec_path) != stage0["artifacts"]["screening_generation_spec"]["sha256"]:
        raise ValueError("screening package generation spec differs from Stage-0")
    normalized = protocol.load_normalized_candidate_manifest(
        candidate_manifest_path,
        dataset=dataset,
        canonical_templates_path=canonical_templates_path,
    )
    videos = validate_screening_generation_manifest(
        project_root,
        dataset=dataset,
        normalized_candidates=normalized,
        candidate_manifest_path=candidate_manifest_path,
        screening_seed_path=screening_seed_path,
        generation_spec_path=generation_spec_path,
        stage0_registry_path=refs["stage0_registry"],
        generation_manifest_path=refs["generation_manifest"],
        decode=decode,
        verify_model_bytes=verify_model_bytes,
    )
    private_dir = manifest_path.parent
    public_dir = refs["public_template"].parent
    if (
        refs["candidate_projection"].parent != private_dir
        or refs["answer_key"].parent != private_dir
        or public_dir.parent.resolve() != private_dir.parent.resolve()
        or public_dir.resolve() == private_dir.resolve()
        or any(path.is_symlink() for path in (public_dir, private_dir))
    ):
        raise ValueError("screening public/private directories are not isolated siblings")
    if {path.name for path in private_dir.iterdir()} != {
        "screening_answer_key_v2.csv",
        "screening_candidates_v2.csv",
        "screening_package_manifest_v2.json",
    } or {path.name for path in public_dir.iterdir()} != {
        "screening_review_v2.csv",
        "media",
        "composites",
    }:
        raise ValueError("screening package file inventory is not exact")
    candidates = protocol.read_csv(refs["candidate_projection"])
    template = protocol.read_csv(refs["public_template"])
    key_rows = protocol.read_csv(refs["answer_key"])
    expected_n = protocol.CANDIDATE_COUNTS[dataset]
    if any(len(rows) != expected_n for rows in (candidates, template, key_rows)):
        raise ValueError("screening package row inventory is not exact")
    fields = tuple(_screening_fields(dataset).values())
    expected_public_fields = {
        "review_id",
        "object_phrase",
        "receiver_description",
        "video_path",
        "composite_path",
        *fields,
        "notes",
    }
    expected_key_fields = {
        "review_id",
        "candidate_id",
        "source_video_path",
        "source_video_sha256",
        "anonymous_video_path",
        "anonymous_video_sha256",
    }
    if any(set(row) != expected_public_fields for row in template) or any(
        set(row) != expected_key_fields for row in key_rows
    ):
        raise ValueError("screening public/key columns are not exact")
    if any(str(row[field]) for row in template for field in (*fields, "notes")):
        raise ValueError("screening public template scores/notes are not blank")
    if set(template[0]) & protocol.FORBIDDEN_PUBLIC_FIELDS:
        raise ValueError("screening public package leaks a private field")
    key_by_review = {row["review_id"]: row for row in key_rows}
    candidate_by_id = {row["candidate_id"]: row for row in candidates}
    if len(key_by_review) != expected_n or len(candidate_by_id) != expected_n:
        raise ValueError("screening package IDs are duplicate")
    seed = int(screening_seed_path.read_text(encoding="ascii").strip())
    expected_internal: list[dict[str, str]] = []
    media_paths = {path.name: path for path in (public_dir / "media").iterdir()}
    composite_paths = {path.name: path for path in (public_dir / "composites").iterdir()}
    if set(media_paths) != {f"s{index:03d}.mp4" for index in range(expected_n)} or set(
        composite_paths
    ) != {f"s{index:03d}.jpg" for index in range(expected_n)}:
        raise ValueError("screening anonymous media/composite inventory is not exact")
    for index, (raw, public) in enumerate(zip(normalized, template)):
        review_id = f"s{index:03d}"
        candidate_id = str(raw["candidate_id"])
        key = key_by_review.get(review_id)
        if key is None or key["candidate_id"] != candidate_id or public["review_id"] != review_id:
            raise ValueError("screening public/key/candidate order binding mismatch")
        source = videos[candidate_id]
        anonymous = media_paths[f"{review_id}.mp4"]
        composite = composite_paths[f"{review_id}.jpg"]
        source_sha = protocol.file_sha256(source)
        anonymous_sha = protocol.file_sha256(anonymous)
        if (
            public["object_phrase"] != str(raw["source_phrase"])
            or public["receiver_description"] != str(raw["receiver"])
            or Path(public["video_path"]).resolve() != anonymous.resolve()
            or Path(public["composite_path"]).resolve() != composite.resolve()
            or Path(key["source_video_path"]).resolve() != source.resolve()
            or Path(key["anonymous_video_path"]).resolve() != anonymous.resolve()
            or key["source_video_sha256"] != source_sha
            or key["anonymous_video_sha256"] != anonymous_sha
            or source_sha != anonymous_sha
            or anonymous.samefile(source)
            or dict(decode(anonymous))
            != {
                "frame_count": protocol.FRAME_COUNT,
                "width": protocol.WIDTH,
                "height": protocol.HEIGHT,
                "fps_numerator": protocol.FPS.numerator,
                "fps_denominator": protocol.FPS.denominator,
            }
        ):
            raise ValueError("screening anonymous media/key binding mismatch")
        expected_internal.append(
            {
                **{field: str(value) for field, value in raw.items()},
                "unit_id": f"screen_{dataset[0]}_{index:03d}",
                "seed": str(seed),
                "screening_video_path": str(source),
                "screening_video_sha256": source_sha,
            }
        )
    if candidates != expected_internal:
        raise ValueError("screening candidate projection differs from frozen raw JSON")
    if manifest["anonymous_media_sha256"] != {
        name: protocol.file_sha256(path) for name, path in media_paths.items()
    } or manifest["composite_sha256"] != {
        name: protocol.file_sha256(path) for name, path in composite_paths.items()
    }:
        raise ValueError("screening public media/composite byte inventory mismatch")
    return manifest, candidates, template, key_rows


def derive_screening_disputes(
    dataset: str,
    candidates: Sequence[Mapping[str, str]],
    reviewer_a: Sequence[Mapping[str, str]],
    reviewer_b: Sequence[Mapping[str, str]],
) -> list[dict[str, str]]:
    candidate_map, a_map, b_map = _validate_screening_reviews(
        dataset, candidates, reviewer_a, reviewer_b
    )
    fields = _screening_fields(dataset)
    return [
        {"candidate_id": candidate_id, "field": short}
        for candidate_id in sorted(candidate_map)
        for short, field in fields.items()
        if _score(a_map[candidate_id], field) != _score(b_map[candidate_id], field)
    ]


def _public_screening_columns(dataset: str) -> tuple[str, ...]:
    return (
        "review_id",
        "object_phrase",
        "receiver_description",
        "video_path",
        "composite_path",
        *_screening_fields(dataset).values(),
        "notes",
    )


def _absolute_lexical_path(path: Path) -> Path:
    """Normalize ``.``/``..`` without following any filesystem link."""

    return Path(os.path.abspath(os.fspath(path)))


def _open_real_directory(path: Path, *, label: str) -> int:
    """Open an existing directory while rejecting a symlink in every component."""

    absolute = _absolute_lexical_path(path)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = -1
    try:
        descriptor = os.open(absolute.anchor, flags)
        for component in absolute.parts[1:]:
            child = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise ValueError(f"{label} must be a real directory")
        return descriptor
    except BaseException as exc:
        if descriptor >= 0:
            os.close(descriptor)
        if isinstance(exc, (FileNotFoundError, ValueError)):
            raise
        raise ValueError(
            f"{label} must be an existing real directory with no symlink ancestor"
        ) from exc


def _directory_descriptor_matches_path(
    descriptor: int,
    path: Path,
    *,
    label: str,
) -> bool:
    """Re-open the real path and compare it with an already pinned directory."""

    try:
        probe = _open_real_directory(path, label=label)
    except (OSError, ValueError):
        return False
    try:
        opened = os.fstat(descriptor)
        current = os.fstat(probe)
        return (opened.st_dev, opened.st_ino) == (current.st_dev, current.st_ino)
    finally:
        os.close(probe)


def _open_real_file_beneath(
    root_descriptor: int,
    relative_path: Path,
    *,
    label: str,
) -> int:
    """Open one regular file beneath a pinned root without following links."""

    if relative_path.is_absolute() or not relative_path.parts:
        raise ValueError(f"screening {label} must be a file below the public root")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    directory_flags = flags | getattr(os, "O_DIRECTORY", 0)
    directory_descriptor = os.dup(root_descriptor)
    descriptor = -1
    try:
        for component in relative_path.parts[:-1]:
            child = os.open(component, directory_flags, dir_fd=directory_descriptor)
            os.close(directory_descriptor)
            directory_descriptor = child
        descriptor = os.open(
            relative_path.name,
            flags | getattr(os, "O_NONBLOCK", 0),
            dir_fd=directory_descriptor,
        )
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            os.close(descriptor)
            descriptor = -1
            raise ValueError(f"screening {label} must be a real regular file")
        return descriptor
    except BaseException as exc:
        if descriptor >= 0:
            os.close(descriptor)
        if isinstance(exc, ValueError):
            raise
        if isinstance(exc, FileNotFoundError):
            raise FileNotFoundError(f"screening {label} is missing") from exc
        raise ValueError(
            f"screening {label} must be a real file with no symlink component"
        ) from exc
    finally:
        os.close(directory_descriptor)


def _read_exact_public_screening_csv(
    *,
    root_descriptor: int,
    public_root: Path,
    resolved_public_root: Path,
    path: Path,
    dataset: str,
    label: str,
) -> list[dict[str, str]]:
    absolute = _absolute_lexical_path(path)
    try:
        relative = absolute.relative_to(public_root)
    except ValueError as exc:
        raise ValueError(f"screening {label} must be contained by --public-root") from exc
    if not relative.parts:
        raise ValueError(f"screening {label} must be a file below --public-root")
    resolved = absolute.resolve(strict=True)
    protocol.reject_sealed_final36_path(resolved)
    try:
        resolved.relative_to(resolved_public_root)
    except ValueError as exc:
        raise ValueError(f"screening {label} resolves outside --public-root") from exc
    expected = os.stat(absolute, follow_symlinks=False)
    if stat.S_ISLNK(expected.st_mode):
        raise ValueError(f"screening {label} must have no symlink component")
    if not stat.S_ISREG(expected.st_mode):
        raise ValueError(f"screening {label} must be a real regular file")
    descriptor = _open_real_file_beneath(root_descriptor, relative, label=label)
    with os.fdopen(descriptor, "r", newline="", encoding="utf-8") as handle:
        opened = os.fstat(handle.fileno())
        if (expected.st_dev, expected.st_ino) != (opened.st_dev, opened.st_ino):
            raise ValueError(f"screening {label} changed during validation")
        reader = csv.DictReader(handle)
        expected_header = _public_screening_columns(dataset)
        if tuple(reader.fieldnames or ()) != expected_header:
            raise ValueError(f"screening {label} header is not exact")
        rows = list(reader)
        after = os.fstat(handle.fileno())
        if (
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
        ) != (
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise ValueError(f"screening {label} changed while it was read")
        return rows


def _read_csv_with_exact_header(
    path: Path,
    *,
    expected_header: Sequence[str],
    label: str,
) -> list[dict[str, str]]:
    """Read a CSV only when its raw header has the exact registered order."""

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != tuple(expected_header):
            raise ValueError(f"screening {label} header is not exact")
        return list(reader)


def derive_public_screening_disputes(
    dataset: str,
    template_rows: Sequence[Mapping[str, str]],
    reviewer_a: Sequence[Mapping[str, str]],
    reviewer_b: Sequence[Mapping[str, str]],
) -> list[dict[str, str]]:
    """Derive the exact anonymous disagreement set without opening private data."""

    if dataset not in protocol.DATASETS:
        raise ValueError("unknown screening dataset")
    expected_n = protocol.CANDIDATE_COUNTS[dataset]
    if any(len(rows) != expected_n for rows in (template_rows, reviewer_a, reviewer_b)):
        raise ValueError(
            f"{dataset}: public template and both reviews must each contain {expected_n} rows"
        )
    score_fields = tuple(_screening_fields(dataset).values())
    expected_columns = set(_public_screening_columns(dataset))
    for label, rows in (
        ("public template", template_rows),
        ("review A", reviewer_a),
        ("review B", reviewer_b),
    ):
        if any(set(row) != expected_columns for row in rows):
            raise ValueError(f"screening {label} columns are not exact")
    if any(
        str(row[field])
        for row in template_rows
        for field in (*score_fields, "notes")
    ):
        raise ValueError("screening public template scores/notes are not blank")

    expected_ids = [f"s{index:03d}" for index in range(expected_n)]
    if [str(row["review_id"]) for row in template_rows] != expected_ids:
        raise ValueError("screening public template review IDs/order are not exact")
    template = {str(row["review_id"]): row for row in template_rows}
    left = {str(row["review_id"]): row for row in reviewer_a}
    right = {str(row["review_id"]): row for row in reviewer_b}
    expected_id_set = set(expected_ids)
    if (
        set(template) != expected_id_set
        or set(left) != expected_id_set
        or set(right) != expected_id_set
        or len(left) != expected_n
        or len(right) != expected_n
    ):
        raise ValueError("screening public review IDs are duplicate or differ from template")
    metadata = expected_columns - set(score_fields) - {"notes"}
    disputes: list[dict[str, str]] = []
    for review_id in expected_ids:
        frozen = template[review_id]
        for reviewer in (left[review_id], right[review_id]):
            if any(str(reviewer[field]) != str(frozen[field]) for field in metadata):
                raise ValueError("screening reviewer changed blinded public metadata")
            for field in score_fields:
                _score(reviewer, field)
        for short, field in _screening_fields(dataset).items():
            if _score(left[review_id], field) != _score(right[review_id], field):
                disputes.append({"review_id": review_id, "field": short})
    return disputes


def _validate_screening_reviews(
    dataset: str,
    candidates: Sequence[Mapping[str, str]],
    reviewer_a: Sequence[Mapping[str, str]],
    reviewer_b: Sequence[Mapping[str, str]],
) -> tuple[
    dict[str, Mapping[str, str]],
    dict[str, Mapping[str, str]],
    dict[str, Mapping[str, str]],
]:
    expected_n = protocol.CANDIDATE_COUNTS[dataset]
    if any(len(rows) != expected_n for rows in (candidates, reviewer_a, reviewer_b)):
        raise ValueError(f"{dataset}: candidate and screening reviews must each contain {expected_n} rows")
    mappings = []
    for label, rows in (("candidate", candidates), ("review A", reviewer_a), ("review B", reviewer_b)):
        mapping = {str(row.get("candidate_id", "")): row for row in rows}
        if len(mapping) != expected_n or "" in mapping:
            raise ValueError(f"{dataset}: duplicate or blank candidate ID in {label}")
        mappings.append(mapping)
    candidate_map, a_map, b_map = mappings
    if set(candidate_map) != set(a_map) or set(candidate_map) != set(b_map):
        raise ValueError("screening reviewer IDs differ from frozen candidate manifest")
    fields = set(_screening_fields(dataset).values())
    metadata = set(candidates[0])
    expected_review_columns = metadata | fields | {"notes"}
    if any(set(row) != metadata for row in candidates):
        raise ValueError("candidate manifest columns are inconsistent")
    if any(set(row) != expected_review_columns for row in [*reviewer_a, *reviewer_b]):
        raise ValueError("screening sheets must contain exact candidate metadata, scores, and notes")
    for candidate_id, frozen in candidate_map.items():
        for reviewer in (a_map[candidate_id], b_map[candidate_id]):
            for field in metadata:
                if str(reviewer[field]) != str(frozen[field]):
                    raise ValueError(f"{candidate_id}: screening reviewer changed metadata {field}")
            for field in fields:
                _score(reviewer, field)
    return candidate_map, a_map, b_map


def merge_screening_reviews(
    dataset: str,
    candidates: Sequence[Mapping[str, str]],
    reviewer_a: Sequence[Mapping[str, str]],
    reviewer_b: Sequence[Mapping[str, str]],
    dispute_rows: Sequence[Mapping[str, str]],
    adjudication_rows: Sequence[Mapping[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidate_map, a_map, b_map = _validate_screening_reviews(
        dataset, candidates, reviewer_a, reviewer_b
    )
    expected = derive_screening_disputes(dataset, candidates, reviewer_a, reviewer_b)
    if [dict(row) for row in dispute_rows] != expected:
        raise ValueError("screening dispute template is not the exact atomic disagreement set")
    keys = {(row["candidate_id"], row["field"]) for row in expected}
    adjudication: dict[tuple[str, str], Mapping[str, str]] = {}
    for row in adjudication_rows:
        if set(row) != {"candidate_id", "field", "score", "brief_reason"}:
            raise ValueError("screening adjudication columns are not exact")
        key = (str(row["candidate_id"]), str(row["field"]))
        if key not in keys or key in adjudication:
            raise ValueError(f"unexpected or duplicate screening adjudication: {key}")
        _score(row, "score")
        if not str(row["brief_reason"]).strip():
            raise ValueError("screening adjudication requires a brief blinded reason")
        adjudication[key] = row
    if set(adjudication) != keys:
        raise ValueError("every screening disagreement requires blinded adjudication")
    fields = _screening_fields(dataset)
    canonical: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    for frozen in candidates:
        candidate_id = str(frozen["candidate_id"])
        output: dict[str, Any] = dict(frozen)
        for short, field in fields.items():
            left = _score(a_map[candidate_id], field)
            right = _score(b_map[candidate_id], field)
            if left == right:
                output[field] = left
            else:
                third = _score(adjudication[(candidate_id, short)], "score")
                output[field] = int(statistics.median((left, right, third)))
                audit.append(
                    {
                        "candidate_id": candidate_id,
                        "field": short,
                        "reviewer_a": left,
                        "reviewer_b": right,
                        "adjudicator": third,
                        "canonical": output[field],
                    }
                )
        output["eligible"] = (
            "yes"
            if (causal_eligible(output) if dataset == "causal" else specificity_eligible(output))
            else "no"
        )
        canonical.append(output)
    return canonical, audit


def _freeze_unbound_screening_rows(
    *,
    dataset: str,
    candidate_path: Path,
    reviewer_a_path: Path,
    reviewer_b_path: Path,
    dispute_path: Path,
    adjudication_path: Path,
    canonical_path: Path,
    audit_path: Path,
    freeze_manifest_path: Path,
) -> dict[str, Any]:
    for path in (canonical_path, audit_path, freeze_manifest_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite screening freeze artifact: {path}")
    if not dispute_path.is_file() or dispute_path.is_symlink():
        raise FileNotFoundError(
            "screening dispute template is missing; derive it before freezing"
        )
    if not adjudication_path.is_file() or adjudication_path.is_symlink():
        raise FileNotFoundError(
            "screening adjudication is missing; complete it before freezing"
        )
    candidates = protocol.read_csv(candidate_path)
    reviewer_a = protocol.read_csv(reviewer_a_path)
    reviewer_b = protocol.read_csv(reviewer_b_path)
    canonical, audit = merge_screening_reviews(
        dataset,
        candidates,
        reviewer_a,
        reviewer_b,
        protocol.read_csv(dispute_path),
        protocol.read_csv(adjudication_path),
    )
    canonical_path.parent.mkdir(parents=True, exist_ok=False)
    protocol.write_csv(canonical_path, canonical)
    protocol.write_csv(
        audit_path,
        audit,
        fieldnames=(
            "candidate_id",
            "field",
            "reviewer_a",
            "reviewer_b",
            "adjudicator",
            "canonical",
        ),
    )
    artifacts = {
        "candidate_manifest": candidate_path,
        "reviewer_a": reviewer_a_path,
        "reviewer_b": reviewer_b_path,
        "dispute_template": dispute_path,
        "adjudication": adjudication_path,
        "canonical_eligibility": canonical_path,
        "adjudication_audit": audit_path,
    }
    payload = {
        "protocol": SCREENING_FREEZE_PROTOCOL,
        "dataset": dataset,
        "dataset_version": protocol.DATASET_VERSION,
        "status": "frozen_before_selection",
        "artifacts": {
            name: {"path": str(path), "sha256": protocol.file_sha256(path)}
            for name, path in artifacts.items()
        },
    }
    freeze_manifest_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return payload


def _validate_unbound_screening_freeze(
    project_root: Path, path: Path, *, dataset: str, private_root: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if set(payload) != {
        "protocol",
        "dataset",
        "dataset_version",
        "status",
        "artifacts",
    }:
        raise ValueError("screening freeze manifest fields are not exact")
    if (
        payload["protocol"] != SCREENING_FREEZE_PROTOCOL
        or payload["dataset"] != dataset
        or payload["dataset_version"] != protocol.DATASET_VERSION
        or payload["status"] != "frozen_before_selection"
    ):
        raise ValueError("screening scores were not frozen before selection")
    required = {
        "candidate_manifest",
        "reviewer_a",
        "reviewer_b",
        "dispute_template",
        "adjudication",
        "canonical_eligibility",
        "adjudication_audit",
    }
    if not isinstance(payload["artifacts"], dict) or set(payload["artifacts"]) != required:
        raise ValueError("screening freeze artifact inventory is not exact")
    paths: dict[str, Path] = {}
    if not private_root.is_dir() or private_root.is_symlink():
        raise ValueError("screening evaluator root must be a real private directory")
    resolved_private = private_root.resolve(strict=True)
    for name, record in payload["artifacts"].items():
        if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
            raise ValueError(f"screening freeze/{name}: record is not exact")
        artifact = protocol.resolve_path(project_root, str(record["path"]))
        if not artifact.is_file() or artifact.is_symlink() or protocol.file_sha256(artifact) != record["sha256"]:
            raise ValueError(f"screening freeze/{name}: byte hash mismatch")
        try:
            artifact.resolve(strict=True).relative_to(resolved_private)
        except ValueError as exc:
            raise ValueError(f"screening freeze/{name}: artifact escapes private root") from exc
        paths[name] = artifact
    candidates = protocol.read_csv(paths["candidate_manifest"])
    canonical, audit = merge_screening_reviews(
        dataset,
        candidates,
        protocol.read_csv(paths["reviewer_a"]),
        protocol.read_csv(paths["reviewer_b"]),
        protocol.read_csv(paths["dispute_template"]),
        protocol.read_csv(paths["adjudication"]),
    )
    frozen = protocol.read_csv(paths["canonical_eligibility"])
    typed = [
        {
            **dict(row),
            **{field: int(row[field]) for field in _screening_fields(dataset).values()},
        }
        for row in frozen
    ]
    if canonical != typed:
        raise ValueError("canonical screening eligibility does not recompute")
    frozen_audit = protocol.read_csv(paths["adjudication_audit"])
    typed_audit = [
        {
            **dict(row),
            **{
                field: int(row[field])
                for field in ("reviewer_a", "reviewer_b", "adjudicator", "canonical")
            },
        }
        for row in frozen_audit
    ]
    if audit != typed_audit:
        raise ValueError("screening adjudication audit does not recompute")
    return payload, canonical


def _unblind_screening_sheets(
    *,
    dataset: str,
    candidates: Sequence[Mapping[str, str]],
    template_rows: Sequence[Mapping[str, str]],
    key_rows: Sequence[Mapping[str, str]],
    reviewer_a: Sequence[Mapping[str, str]],
    reviewer_b: Sequence[Mapping[str, str]],
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    expected_n = protocol.CANDIDATE_COUNTS[dataset]
    if any(
        len(rows) != expected_n
        for rows in (candidates, template_rows, key_rows, reviewer_a, reviewer_b)
    ):
        raise ValueError("screening public reviews do not cover the exact candidate inventory")
    template = {row["review_id"]: row for row in template_rows}
    keys = {row["review_id"]: row for row in key_rows}
    left = {row.get("review_id", ""): row for row in reviewer_a}
    right = {row.get("review_id", ""): row for row in reviewer_b}
    candidates_by_id = {row["candidate_id"]: row for row in candidates}
    if (
        len(template) != expected_n
        or len(keys) != expected_n
        or len(left) != expected_n
        or len(right) != expected_n
    ):
        raise ValueError("screening review/key IDs are duplicate or inconsistent")
    if not (set(template) == set(keys) == set(left) == set(right)):
        raise ValueError("screening review/key IDs differ")
    score_fields = set(_screening_fields(dataset).values())
    columns = set(template_rows[0])
    metadata = columns - score_fields - {"notes"}
    if any(set(row) != columns for row in [*template_rows, *reviewer_a, *reviewer_b]):
        raise ValueError("screening public review columns changed")
    output_a: list[dict[str, str]] = []
    output_b: list[dict[str, str]] = []
    disputes: list[dict[str, str]] = []
    for review_id in sorted(template):
        frozen = template[review_id]
        key = keys[review_id]
        candidate = candidates_by_id.get(key["candidate_id"])
        if candidate is None:
            raise ValueError("screening key references an unknown normalized candidate")
        for reviewer in (left[review_id], right[review_id]):
            if any(str(reviewer[field]) != str(frozen[field]) for field in metadata):
                raise ValueError("screening reviewer changed blinded public metadata")
            for field in score_fields:
                _score(reviewer, field)
        output_a.append(
            {
                **dict(candidate),
                **{field: left[review_id][field] for field in score_fields},
                "notes": left[review_id]["notes"],
            }
        )
        output_b.append(
            {
                **dict(candidate),
                **{field: right[review_id][field] for field in score_fields},
                "notes": right[review_id]["notes"],
            }
        )
        for short, field in _screening_fields(dataset).items():
            if _score(left[review_id], field) != _score(right[review_id], field):
                disputes.append({"review_id": review_id, "field": short})
    return output_a, output_b, disputes


def freeze_screening_reviews(
    *,
    project_root: Path,
    dataset: str,
    package_manifest_path: Path,
    private_root: Path,
    candidate_manifest_path: Path,
    canonical_templates_path: Path,
    screening_seed_path: Path,
    generation_spec_path: Path,
    reviewer_a_path: Path,
    reviewer_b_path: Path,
    dispute_path: Path,
    adjudication_path: Path,
    canonical_path: Path,
    audit_path: Path,
    freeze_manifest_path: Path,
    decode: Callable[[Path], Mapping[str, int]] = protocol._default_decode,
    verify_model_bytes: bool = True,
) -> dict[str, Any]:
    if not private_root.is_dir() or private_root.is_symlink():
        raise ValueError("screening evaluator root must be a real private directory")
    resolved_private = private_root.resolve(strict=True)
    if canonical_path.parent.resolve() != audit_path.parent.resolve():
        raise ValueError("canonical screening eligibility and audit must be siblings")
    for label, target in (
        ("canonical eligibility", canonical_path),
        ("adjudication audit", audit_path),
        ("freeze manifest", freeze_manifest_path),
    ):
        try:
            target.resolve().relative_to(resolved_private)
        except ValueError as exc:
            raise ValueError(f"screening {label} escapes private root") from exc
    for path in (canonical_path, audit_path, freeze_manifest_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite screening freeze artifact: {path}")
    if not dispute_path.is_file() or dispute_path.is_symlink():
        raise FileNotFoundError(
            "screening dispute template is missing; run derive-disputes first"
        )
    if not adjudication_path.is_file() or adjudication_path.is_symlink():
        raise FileNotFoundError(
            "screening adjudication is missing; complete it before freezing"
        )
    package, candidates, template, key_rows = validate_screening_review_package(
        project_root,
        dataset=dataset,
        manifest_path=package_manifest_path,
        private_root=private_root,
        candidate_manifest_path=candidate_manifest_path,
        canonical_templates_path=canonical_templates_path,
        screening_seed_path=screening_seed_path,
        generation_spec_path=generation_spec_path,
        decode=decode,
        verify_model_bytes=verify_model_bytes,
    )
    reviewer_a = protocol.read_csv(reviewer_a_path)
    reviewer_b = protocol.read_csv(reviewer_b_path)
    internal_a, internal_b, expected_disputes = _unblind_screening_sheets(
        dataset=dataset,
        candidates=candidates,
        template_rows=template,
        key_rows=key_rows,
        reviewer_a=reviewer_a,
        reviewer_b=reviewer_b,
    )
    public_disputes = protocol.read_csv(dispute_path)
    if public_disputes != expected_disputes:
        raise ValueError("screening public dispute template is not the exact disagreement set")
    key_by_review = {row["review_id"]: row for row in key_rows}
    internal_disputes = [
        {
            "candidate_id": key_by_review[row["review_id"]]["candidate_id"],
            "field": row["field"],
        }
        for row in public_disputes
    ]
    public_adjudication = _read_csv_with_exact_header(
        adjudication_path,
        expected_header=PUBLIC_SCREENING_ADJUDICATION_COLUMNS,
        label="public adjudication",
    )
    if any(
        set(row) != set(PUBLIC_SCREENING_ADJUDICATION_COLUMNS)
        for row in public_adjudication
    ):
        raise ValueError("screening public adjudication columns are not exact")
    expected_keys = {(row["review_id"], row["field"]) for row in expected_disputes}
    actual_keys = {(row["review_id"], row["field"]) for row in public_adjudication}
    if actual_keys != expected_keys or len(public_adjudication) != len(expected_keys):
        raise ValueError("screening public adjudication does not cover every disagreement")
    internal_adjudication = [
        {
            "candidate_id": key_by_review[row["review_id"]]["candidate_id"],
            "field": row["field"],
            "score": row["score"],
            "brief_reason": row["brief_reason"],
        }
        for row in public_adjudication
    ]
    canonical, audit = merge_screening_reviews(
        dataset,
        candidates,
        internal_a,
        internal_b,
        internal_disputes,
        internal_adjudication,
    )
    canonical_path.parent.mkdir(parents=True, exist_ok=False)
    protocol.write_csv(canonical_path, canonical)
    protocol.write_csv(
        audit_path,
        audit,
        fieldnames=(
            "candidate_id",
            "field",
            "reviewer_a",
            "reviewer_b",
            "adjudicator",
            "canonical",
        ),
    )
    artifacts = {
        "screening_package": package_manifest_path,
        "reviewer_a": reviewer_a_path,
        "reviewer_b": reviewer_b_path,
        "dispute_template": dispute_path,
        "adjudication": adjudication_path,
        "canonical_eligibility": canonical_path,
        "adjudication_audit": audit_path,
    }
    payload = {
        "protocol": SCREENING_FREEZE_PROTOCOL,
        "dataset": dataset,
        "dataset_version": protocol.DATASET_VERSION,
        "status": "frozen_before_selection",
        "package_binding_sha256": protocol.canonical_json_sha256(package),
        "artifacts": {
            name: {"path": str(path), "sha256": protocol.file_sha256(path)}
            for name, path in artifacts.items()
        },
    }
    freeze_manifest_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return payload


def validate_screening_freeze(
    project_root: Path,
    path: Path,
    *,
    dataset: str,
    private_root: Path,
    candidate_manifest_path: Path,
    canonical_templates_path: Path,
    screening_seed_path: Path,
    generation_spec_path: Path,
    decode: Callable[[Path], Mapping[str, int]] = protocol._default_decode,
    verify_model_bytes: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not private_root.is_dir() or private_root.is_symlink():
        raise ValueError("screening evaluator root must be a real private directory")
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError("screening freeze manifest is missing")
    resolved_private = private_root.resolve(strict=True)
    try:
        path.resolve(strict=True).relative_to(resolved_private)
    except ValueError as exc:
        raise ValueError("screening freeze manifest escapes private root") from exc
    payload = json.loads(path.read_text(encoding="utf-8"))
    if set(payload) != {
        "protocol",
        "dataset",
        "dataset_version",
        "status",
        "package_binding_sha256",
        "artifacts",
    }:
        raise ValueError("screening freeze manifest fields are not exact")
    if (
        payload["protocol"] != SCREENING_FREEZE_PROTOCOL
        or payload["dataset"] != dataset
        or payload["dataset_version"] != protocol.DATASET_VERSION
        or payload["status"] != "frozen_before_selection"
    ):
        raise ValueError("screening scores were not frozen before selection")
    required = {
        "screening_package",
        "reviewer_a",
        "reviewer_b",
        "dispute_template",
        "adjudication",
        "canonical_eligibility",
        "adjudication_audit",
    }
    if not isinstance(payload["artifacts"], dict) or set(payload["artifacts"]) != required:
        raise ValueError("screening freeze artifact inventory is not exact")
    paths: dict[str, Path] = {}
    for name, record in payload["artifacts"].items():
        if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
            raise ValueError(f"screening freeze/{name}: reference is not exact")
        artifact = protocol.resolve_path(project_root, str(record["path"]))
        if not artifact.is_file() or artifact.is_symlink() or protocol.file_sha256(artifact) != record["sha256"]:
            raise ValueError(f"screening freeze/{name}: byte hash mismatch")
        paths[name] = artifact
    for name in ("screening_package", "canonical_eligibility", "adjudication_audit"):
        try:
            paths[name].resolve(strict=True).relative_to(resolved_private)
        except ValueError as exc:
            raise ValueError(f"screening freeze/{name}: escapes private root") from exc
    package, candidates, template, key_rows = validate_screening_review_package(
        project_root,
        dataset=dataset,
        manifest_path=paths["screening_package"],
        private_root=private_root,
        candidate_manifest_path=candidate_manifest_path,
        canonical_templates_path=canonical_templates_path,
        screening_seed_path=screening_seed_path,
        generation_spec_path=generation_spec_path,
        decode=decode,
        verify_model_bytes=verify_model_bytes,
    )
    if payload["package_binding_sha256"] != protocol.canonical_json_sha256(package):
        raise ValueError("screening freeze package binding mismatch")
    internal_a, internal_b, expected_disputes = _unblind_screening_sheets(
        dataset=dataset,
        candidates=candidates,
        template_rows=template,
        key_rows=key_rows,
        reviewer_a=protocol.read_csv(paths["reviewer_a"]),
        reviewer_b=protocol.read_csv(paths["reviewer_b"]),
    )
    if protocol.read_csv(paths["dispute_template"]) != expected_disputes:
        raise ValueError("screening frozen dispute set does not recompute")
    key_by_review = {row["review_id"]: row for row in key_rows}
    internal_disputes = [
        {
            "candidate_id": key_by_review[row["review_id"]]["candidate_id"],
            "field": row["field"],
        }
        for row in expected_disputes
    ]
    public_adjudication = _read_csv_with_exact_header(
        paths["adjudication"],
        expected_header=PUBLIC_SCREENING_ADJUDICATION_COLUMNS,
        label="frozen adjudication",
    )
    if any(
        set(row) != set(PUBLIC_SCREENING_ADJUDICATION_COLUMNS)
        for row in public_adjudication
    ):
        raise ValueError("screening frozen adjudication columns are not exact")
    expected_public_keys = {
        (row["review_id"], row["field"]) for row in expected_disputes
    }
    actual_public_keys = {
        (row["review_id"], row["field"]) for row in public_adjudication
    }
    if (
        actual_public_keys != expected_public_keys
        or len(public_adjudication) != len(expected_public_keys)
    ):
        raise ValueError("screening frozen adjudication set is not exact")
    internal_adjudication = [
        {
            "candidate_id": key_by_review[row["review_id"]]["candidate_id"],
            "field": row["field"],
            "score": row["score"],
            "brief_reason": row["brief_reason"],
        }
        for row in public_adjudication
    ]
    canonical, audit = merge_screening_reviews(
        dataset,
        candidates,
        internal_a,
        internal_b,
        internal_disputes,
        internal_adjudication,
    )
    frozen = protocol.read_csv(paths["canonical_eligibility"])
    typed = [
        {
            **dict(row),
            **{field: int(row[field]) for field in _screening_fields(dataset).values()},
        }
        for row in frozen
    ]
    if canonical != typed:
        raise ValueError("canonical screening eligibility does not recompute from package")
    frozen_audit = protocol.read_csv(paths["adjudication_audit"])
    typed_audit = [
        {
            **dict(row),
            **{
                field: int(row[field])
                for field in ("reviewer_a", "reviewer_b", "adjudicator", "canonical")
            },
        }
        for row in frozen_audit
    ]
    if audit != typed_audit:
        raise ValueError("screening adjudication audit does not recompute from package")
    return payload, canonical


def _score(row: Mapping[str, Any], field: str) -> int:
    try:
        value = int(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{row.get('candidate_id', '<candidate>')}: invalid {field}") from exc
    if value not in {0, 1, 2}:
        raise ValueError(f"{row.get('candidate_id', '<candidate>')}: {field} must be 0, 1, or 2")
    return value


RANK_DOMAINS = {
    "causal": "causal-selector-v2",
    # This domain is executable but does not authorize specificity data; the
    # isolated specificity Stage-0 registry must independently commit it.
    "specificity": "specificity-selector-v2",
}
CAUSAL_CANONICAL_RECORD_FIELDS = {
    "case_id",
    "group",
    "prompt_variant",
    "source_membership",
    "source_id",
    "source_phrase",
    "source_head_lemma",
    "source_physical_audit_status",
    "receiver_membership",
    "receiver_id",
    "receiver_phrase",
    "canonical_prompt",
}
SPECIFICITY_CANONICAL_RECORD_FIELDS = {
    "case_id",
    "membership",
    "prompt_variant",
    "source_id",
    "source_phrase",
    "source_head_lemma",
    "receiver_id",
    "receiver_phrase",
    "causal_case_id",
    "template_id",
    "canonical_prompt",
}


def selection_rank(
    row: Mapping[str, Any], private_salt: str, *, dataset: str | None = None
) -> str:
    if not private_salt:
        raise ValueError("selector salt must be nonempty")
    if dataset is None:
        dataset = "causal" if "group" in row else "specificity" if "membership" in row else ""
    if dataset not in protocol.DATASETS:
        raise ValueError("candidate row does not identify a causal/specificity rank domain")
    canonical_fields = (
        CAUSAL_CANONICAL_RECORD_FIELDS
        if dataset == "causal"
        else SPECIFICITY_CANONICAL_RECORD_FIELDS
    )
    if not canonical_fields <= set(row) or "canonical_record_sha256" not in row:
        raise ValueError("candidate does not contain the exact frozen canonical record fields")
    canonical_record = {key: row[key] for key in canonical_fields}
    canonical_bytes = (
        json.dumps(
            canonical_record,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        + "\n"
    ).encode("utf-8")
    if hashlib.sha256(canonical_bytes).hexdigest() != str(row["canonical_record_sha256"]):
        raise ValueError("candidate canonical record changed before SHA ranking")
    digest = hashlib.sha256()
    digest.update(RANK_DOMAINS[dataset].encode("utf-8"))
    digest.update(b"\0")
    digest.update(private_salt.encode("utf-8"))
    digest.update(b"\0")
    digest.update(canonical_bytes)
    return digest.hexdigest()


def _with_case_ids(rows: Sequence[Mapping[str, Any]], dataset: str) -> list[dict[str, Any]]:
    id_field = "semantic_case_id" if dataset == "causal" else "specificity_case_id"
    output: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        if not str(item.get(id_field, "")):
            item[id_field] = str(item["candidate_id"])
        output.append(item)
    return output


def causal_eligible(row: Mapping[str, Any]) -> bool:
    return (
        _score(row, "source_visibility_0_absent_2_clear") == 2
        and _score(row, "footprint_visibility_0_absent_2_clear") >= 1
        and _score(row, "receiver_preservation_0_bad_2_good") >= 1
        and _score(row, "video_quality_0_bad_2_good") >= 1
        and _score(row, "causal_link_0_absent_2_clear") == 2
    )


def specificity_eligible(row: Mapping[str, Any]) -> bool:
    return (
        _score(row, "protected_object_visibility_0_absent_2_clear") == 2
        and _score(row, "receiver_preservation_0_bad_2_good") >= 1
        and _score(row, "video_quality_0_bad_2_good") >= 1
        and _score(row, "noncausal_role_adherence_0_bad_2_good") == 2
    )


def _first_lexicographic_feasible_subset(
    ranked: Sequence[dict[str, Any]],
    *,
    selected_n: int,
    cell_of: Callable[[Mapping[str, Any]], tuple[str, str]],
    target_per_cell: Mapping[tuple[str, str], int],
    compatible: Callable[[Sequence[Mapping[str, Any]], Mapping[str, Any]], bool],
    final_validator: Callable[[Sequence[Mapping[str, Any]]], None],
) -> list[dict[str, Any]]:
    """Return the feasible set with lexicographically smallest rank tuple.

    Candidates are visited in increasing SHA rank, with the include branch
    searched first.  The first solution is therefore exactly the registered
    lexicographic optimum; there is no reserve queue or post-hoc replacement.
    """

    cell_suffix: list[Counter[tuple[str, str]]] = [Counter() for _ in range(len(ranked) + 1)]
    for index in range(len(ranked) - 1, -1, -1):
        cell_suffix[index] = cell_suffix[index + 1].copy()
        cell_suffix[index][cell_of(ranked[index])] += 1

    chosen: list[dict[str, Any]] = []
    counts: Counter[tuple[str, str]] = Counter()

    def possible(index: int) -> bool:
        if len(chosen) > selected_n or len(chosen) + len(ranked) - index < selected_n:
            return False
        for cell, target in target_per_cell.items():
            if counts[cell] > target or counts[cell] + cell_suffix[index][cell] < target:
                return False
        return True

    def search(index: int) -> list[dict[str, Any]] | None:
        if not possible(index):
            return None
        if index == len(ranked):
            if len(chosen) != selected_n or any(counts[cell] != target for cell, target in target_per_cell.items()):
                return None
            try:
                final_validator(chosen)
            except (KeyError, TypeError, ValueError):
                return None
            return [dict(row) for row in chosen]

        candidate = ranked[index]
        cell = cell_of(candidate)
        if counts[cell] < target_per_cell.get(cell, 0) and compatible(chosen, candidate):
            chosen.append(candidate)
            counts[cell] += 1
            result = search(index + 1)
            if result is not None:
                return result
            counts[cell] -= 1
            chosen.pop()
        return search(index + 1)

    result = search(0)
    if result is None:
        raise ValueError("no feasible frozen subset; this dataset version is invalid")
    return result


def select_causal_cases(
    candidate_rows: Sequence[Mapping[str, Any]], private_salt: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(candidate_rows) != 48:
        raise ValueError("causal candidate pool must contain exactly 48 rows")
    ids = [str(row.get("candidate_id", "")) for row in candidate_rows]
    if any(not value for value in ids) or len(set(ids)) != 48:
        raise ValueError("causal candidate IDs must be nonempty and unique")
    expected_cells = {
        (group, variant) for group in protocol.CAUSAL_GROUPS for variant in protocol.PROMPT_VARIANTS
    }
    pool_counts = Counter((str(row.get("group")), str(row.get("prompt_variant"))) for row in candidate_rows)
    if set(pool_counts) != expected_cells or any(value != 8 for value in pool_counts.values()):
        raise ValueError("causal pool must contain eight candidates per group/variant cell")
    for row in candidate_rows:
        group = str(row["group"])
        membership = str(row.get("source_membership", ""))
        expected_membership = "original_source" if group == "seen_source_new_receiver" else "holdout_source"
        if membership != expected_membership:
            raise ValueError("causal group/source-membership mismatch")
    eligibility = [
        {
            "candidate_id": str(row["candidate_id"]),
            "group": str(row["group"]),
            "prompt_variant": str(row["prompt_variant"]),
            "eligible": "yes" if causal_eligible(row) else "no",
            "selection_rank_sha256": selection_rank(row, private_salt, dataset="causal"),
        }
        for row in candidate_rows
    ]
    ranks = [row["selection_rank_sha256"] for row in eligibility]
    if len(set(ranks)) != len(ranks):
        raise ValueError("causal selection rank tie invalidates the data version")
    eligible = [dict(row) for row in candidate_rows if causal_eligible(row)]
    eligible_counts = Counter((str(row["group"]), str(row["prompt_variant"])) for row in eligible)
    if any(eligible_counts[cell] < 4 for cell in expected_cells):
        raise ValueError("causal pool has fewer than four eligible cases in a cell")
    for row in eligible:
        row["selection_rank_sha256"] = selection_rank(row, private_salt, dataset="causal")
    ranked = sorted(eligible, key=lambda row: (row["selection_rank_sha256"], str(row["candidate_id"])))

    def compatible(chosen: Sequence[Mapping[str, Any]], candidate: Mapping[str, Any]) -> bool:
        if any(str(row["receiver_id"]) == str(candidate["receiver_id"]) for row in chosen):
            return False
        group = str(candidate["group"])
        if group in protocol.HOLDOUT_GROUPS and any(
            str(row["group"]) in protocol.HOLDOUT_GROUPS
            and str(row["source_head_lemma"]) == str(candidate["source_head_lemma"])
            for row in chosen
        ):
            return False
        if group == "seen_source_new_receiver" and any(
            str(row["group"]) == group and str(row["source_id"]) == str(candidate["source_id"])
            for row in chosen
        ):
            return False
        return True

    selected = _first_lexicographic_feasible_subset(
        ranked,
        selected_n=24,
        cell_of=lambda row: (str(row["group"]), str(row["prompt_variant"])),
        target_per_cell={cell: 4 for cell in expected_cells},
        compatible=compatible,
        final_validator=lambda rows: protocol.validate_causal_selected_cases(
            _with_case_ids(rows, "causal")
        ),
    )
    return selected, eligibility


def select_specificity_cases(
    candidate_rows: Sequence[Mapping[str, Any]],
    *,
    private_salt: str,
    causal_cases: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(candidate_rows) != 36:
        raise ValueError("specificity candidate pool must contain exactly 36 rows")
    ids = [str(row.get("candidate_id", "")) for row in candidate_rows]
    if any(not value for value in ids) or len(set(ids)) != 36:
        raise ValueError("specificity candidate IDs must be nonempty and unique")
    expected_pool = {
        ("original_source", "direct"): 4,
        ("original_source", "natural"): 4,
        ("new_bank_source", "direct"): 6,
        ("new_bank_source", "natural"): 6,
        ("holdout_source", "direct"): 8,
        ("holdout_source", "natural"): 8,
    }
    pool_counts = Counter((str(row.get("membership")), str(row.get("prompt_variant"))) for row in candidate_rows)
    if dict(pool_counts) != expected_pool:
        raise ValueError("specificity candidate pool cell inventory differs from protocol")
    eligibility = [
        {
            "candidate_id": str(row["candidate_id"]),
            "membership": str(row["membership"]),
            "prompt_variant": str(row["prompt_variant"]),
            "eligible": "yes" if specificity_eligible(row) else "no",
            "selection_rank_sha256": selection_rank(
                row, private_salt, dataset="specificity"
            ),
        }
        for row in candidate_rows
    ]
    ranks = [row["selection_rank_sha256"] for row in eligibility]
    if len(set(ranks)) != len(ranks):
        raise ValueError("specificity selection rank tie invalidates the data version")
    eligible = [dict(row) for row in candidate_rows if specificity_eligible(row)]
    target_cells = {
        (membership, variant): 3
        for membership in protocol.SPECIFICITY_MEMBERSHIPS
        for variant in protocol.PROMPT_VARIANTS
    }
    eligible_counts = Counter((str(row["membership"]), str(row["prompt_variant"])) for row in eligible)
    if any(eligible_counts[cell] < 3 for cell in target_cells):
        raise ValueError("specificity pool has fewer than three eligible cases in a cell")
    for row in eligible:
        row["selection_rank_sha256"] = selection_rank(
            row, private_salt, dataset="specificity"
        )
    ranked = sorted(eligible, key=lambda row: (row["selection_rank_sha256"], str(row["candidate_id"])))

    def compatible(chosen: Sequence[Mapping[str, Any]], candidate: Mapping[str, Any]) -> bool:
        if any(str(row["source_head_lemma"]) == str(candidate["source_head_lemma"]) for row in chosen):
            return False
        causal_id = str(candidate.get("causal_case_id", ""))
        if causal_id and any(str(row.get("causal_case_id", "")) == causal_id for row in chosen):
            return False
        return True

    selected = _first_lexicographic_feasible_subset(
        ranked,
        selected_n=18,
        cell_of=lambda row: (str(row["membership"]), str(row["prompt_variant"])),
        target_per_cell=target_cells,
        compatible=compatible,
        final_validator=lambda rows: protocol.validate_specificity_selected_cases(
            _with_case_ids(rows, "specificity"), causal_cases=causal_cases
        ),
    )
    return selected, eligibility


def _private_case_rows(rows: Sequence[Mapping[str, Any]], dataset: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    id_field = "semantic_case_id" if dataset == "causal" else "specificity_case_id"
    for position, row in enumerate(rows):
        allowed = (
            {
                "candidate_id",
                "semantic_case_id",
                "group",
                "source_membership",
                "prompt_variant",
                "source_id",
                "source_phrase",
                "source_head_lemma",
                "source_physical_audit_status",
                "receiver_id",
                "receiver",
                "prompt",
            }
            if dataset == "causal"
            else {
                "candidate_id",
                "specificity_case_id",
                "membership",
                "prompt_variant",
                "source_id",
                "source_phrase",
                "source_head_lemma",
                "receiver_id",
                "receiver",
                "causal_case_id",
                "template_id",
                "prompt",
            }
        )
        output = {key: value for key, value in row.items() if key in allowed}
        if not str(output.get(id_field, "")):
            output[id_field] = str(output.get("candidate_id") or f"{dataset[0]}case_{position:03d}")
        result.append(output)
    return result


def revalidate_stage1_derivation(
    project_root: Path,
    *,
    dataset: str,
    private_root: Path,
    opened_paths: Mapping[str, Path],
    causal_cases: Sequence[Mapping[str, Any]] | None = None,
    causal_units: Sequence[Mapping[str, Any]] | None = None,
    decode: Callable[[Path], Mapping[str, int]] = protocol._default_decode,
    verify_model_bytes: bool = True,
) -> dict[str, Any]:
    """Re-run screening merge, qualification, global selection, seeds, and M."""

    required = {
        f"candidate_manifest_{protocol.CANDIDATE_COUNTS[dataset]}",
        "canonical_templates",
        "screening_seed",
        "screening_generation_spec",
        "selector_salt",
        "evaluation_seed_salt",
        "forbidden_seed_inventory",
        "screening_freeze_manifest",
        f"eligibility_table_{protocol.CANDIDATE_COUNTS[dataset]}",
        f"selector_output_{protocol.CASE_COUNTS[dataset]}",
        (
            "selected_case_manifest_24"
            if dataset == "causal"
            else "selected_case_manifest_18"
        ),
        "unit_manifest_U_72" if dataset == "causal" else "unit_manifest_W_36",
    }
    if dataset == "specificity":
        required.add("holdout_mapping_M_6")
    if not required <= set(opened_paths):
        raise ValueError("Stage-1 semantic revalidation opening is incomplete")
    candidate_path = opened_paths[
        f"candidate_manifest_{protocol.CANDIDATE_COUNTS[dataset]}"
    ]
    _, candidates = validate_screening_freeze(
        project_root,
        opened_paths["screening_freeze_manifest"],
        dataset=dataset,
        private_root=private_root,
        candidate_manifest_path=candidate_path,
        canonical_templates_path=opened_paths["canonical_templates"],
        screening_seed_path=opened_paths["screening_seed"],
        generation_spec_path=opened_paths["screening_generation_spec"],
        decode=decode,
        verify_model_bytes=verify_model_bytes,
    )
    selector_salt = opened_paths["selector_salt"].read_text(encoding="ascii").strip()
    evaluation_salt = opened_paths["evaluation_seed_salt"].read_text(
        encoding="ascii"
    ).strip()
    screening_seed_text = opened_paths["screening_seed"].read_text(encoding="ascii")
    if not re.fullmatch(r"(?:0|[1-9][0-9]*)\n", screening_seed_text):
        raise ValueError("Stage-1 semantic revalidation screening seed is noncanonical")
    screening_seed = int(screening_seed_text[:-1])
    forbidden = protocol.validate_forbidden_seed_inventory(
        opened_paths["forbidden_seed_inventory"], dataset=dataset
    )
    if dataset == "causal":
        selected, eligibility = select_causal_cases(candidates, selector_salt)
        selected = _private_case_rows(selected, "causal")
        protocol.validate_causal_selected_cases(selected)
    elif dataset == "specificity":
        if causal_cases is None or causal_units is None:
            raise ValueError("specificity Stage-1 revalidation requires causal selected24/U72")
        protocol.validate_causal_selected_cases(causal_cases)
        protocol.validate_causal_unit_manifest(causal_units)
        causal_seed_set = {int(row["seed"]) for row in causal_units}
        if not causal_seed_set <= forbidden:
            raise ValueError("specificity forbidden inventory omits causal U seeds")
        selected, eligibility = select_specificity_cases(
            candidates,
            private_salt=selector_salt,
            causal_cases=causal_cases,
        )
        selected = _private_case_rows(selected, "specificity")
        protocol.validate_specificity_selected_cases(
            selected, causal_cases=causal_cases
        )
    else:
        raise ValueError("unknown Stage-1 semantic revalidation dataset")
    units = protocol.derive_unit_rows(
        selected,
        dataset=dataset,
        private_salt=evaluation_salt,
        forbidden_seeds={*forbidden, screening_seed},
    )

    def string_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
        return [{key: str(value) for key, value in row.items()} for row in rows]

    eligibility_name = f"eligibility_table_{protocol.CANDIDATE_COUNTS[dataset]}"
    selected_name = (
        "selected_case_manifest_24"
        if dataset == "causal"
        else "selected_case_manifest_18"
    )
    unit_name = "unit_manifest_U_72" if dataset == "causal" else "unit_manifest_W_36"
    if protocol.read_csv(opened_paths[eligibility_name]) != string_rows(eligibility):
        raise ValueError("committed eligibility table does not recompute from screening freeze")
    if protocol.read_csv(opened_paths[selected_name]) != string_rows(selected):
        raise ValueError("committed selected-case manifest is not the lexicographic optimum")
    if protocol.read_csv(opened_paths[unit_name]) != string_rows(units):
        raise ValueError("committed unit manifest seeds do not recompute from Stage-0 salt")

    mapping: list[dict[str, Any]] | None = None
    if dataset == "specificity":
        assert causal_cases is not None
        mapping = [
            {
                "specificity_case_id": row["specificity_case_id"],
                "causal_case_id": row["causal_case_id"],
                "source_id": row["source_id"],
                "source_phrase": row["source_phrase"],
                "receiver_id": row["receiver_id"],
                "receiver": row["receiver"],
            }
            for row in selected
            if row["membership"] == "holdout_source"
        ]
        protocol.validate_holdout_mapping(
            mapping,
            causal_cases=causal_cases,
            specificity_cases=selected,
        )
        if protocol.read_csv(opened_paths["holdout_mapping_M_6"]) != string_rows(mapping):
            raise ValueError("committed M6 mapping does not recompute from selected specificity cases")

    summary = {
        "protocol": protocol.PROTOCOL,
        "dataset": dataset,
        "dataset_version": protocol.DATASET_VERSION,
        "candidate_count": len(candidates),
        "eligible_count": sum(row["eligible"] == "yes" for row in eligibility),
        "selected_count": len(selected),
        "unit_count": len(units),
        "selection_rank_tuple": [
            next(
                row["selection_rank_sha256"]
                for row in eligibility
                if row["candidate_id"] == selected_row["candidate_id"]
            )
            for selected_row in selected
        ],
    }
    committed_summary = json.loads(
        opened_paths[f"selector_output_{protocol.CASE_COUNTS[dataset]}"].read_text(
            encoding="utf-8"
        )
    )
    if committed_summary != summary:
        raise ValueError("committed selector summary/rank tuple does not recompute")
    return {
        "selected": selected,
        "units": units,
        "eligibility": eligibility,
        "mapping": mapping,
        "summary": summary,
    }


def _atomic_write_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically publish one commitment without replacing existing bytes."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp.{os.getpid()}"
    if path.exists() or path.is_symlink() or temporary.exists():
        raise FileExistsError(f"refusing to overwrite frozen Stage-1 registry: {path}")
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


def _atomic_write_new_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    fieldnames: Sequence[str],
) -> str:
    """Atomically publish one CSV without replacing an existing path."""

    absolute = _absolute_lexical_path(path)
    protocol.reject_sealed_final36_path(absolute)
    if not absolute.name or len(set(fieldnames)) != len(fieldnames):
        raise ValueError("dispute output path/columns are not exact")
    resolved_parent = absolute.parent.resolve(strict=True)
    protocol.reject_sealed_final36_path(resolved_parent, resolved_parent / absolute.name)
    parent_descriptor = _open_real_directory(
        absolute.parent, label="screening dispute output parent"
    )
    temporary_name = f".{absolute.name}.tmp.{os.getpid()}"
    temporary_descriptor = -1
    temporary_identity: tuple[int, int] | None = None
    try:
        if not _directory_descriptor_matches_path(
            parent_descriptor,
            absolute.parent,
            label="screening dispute output parent",
        ):
            raise ValueError("screening dispute output parent changed during validation")
        try:
            os.stat(absolute.name, dir_fd=parent_descriptor, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError(
                f"refusing to overwrite frozen dispute template: {absolute}"
            )
        flags = (
            os.O_RDWR
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        temporary_descriptor = os.open(
            temporary_name,
            flags,
            0o600,
            dir_fd=parent_descriptor,
        )
        opened_temporary = os.fstat(temporary_descriptor)
        temporary_identity = (opened_temporary.st_dev, opened_temporary.st_ino)
        with os.fdopen(
            temporary_descriptor,
            "w",
            newline="",
            encoding="utf-8",
            closefd=False,
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(fieldnames),
                lineterminator="\n",
                extrasaction="raise",
            )
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(temporary_descriptor)
        os.lseek(temporary_descriptor, 0, os.SEEK_SET)
        digest = hashlib.sha256()
        for chunk in iter(lambda: os.read(temporary_descriptor, 1024 * 1024), b""):
            digest.update(chunk)
        temporary_stat = os.fstat(temporary_descriptor)
        named_temporary_stat = os.stat(
            temporary_name, dir_fd=parent_descriptor, follow_symlinks=False
        )
        if (temporary_stat.st_dev, temporary_stat.st_ino) != (
            named_temporary_stat.st_dev,
            named_temporary_stat.st_ino,
        ):
            raise RuntimeError("screening dispute temporary changed before publication")
        if not _directory_descriptor_matches_path(
            parent_descriptor,
            absolute.parent,
            label="screening dispute output parent",
        ):
            raise ValueError("screening dispute output parent changed before publication")
        try:
            os.link(
                temporary_name,
                absolute.name,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except FileExistsError as exc:
            raise FileExistsError(
                f"refusing to overwrite frozen dispute template: {absolute}"
            ) from exc
        published_stat = os.stat(
            absolute.name, dir_fd=parent_descriptor, follow_symlinks=False
        )
        if (temporary_stat.st_dev, temporary_stat.st_ino) != (
            published_stat.st_dev,
            published_stat.st_ino,
        ):
            raise RuntimeError("screening dispute output changed during publication")
        if not _directory_descriptor_matches_path(
            parent_descriptor,
            absolute.parent,
            label="screening dispute output parent",
        ):
            current_output = os.stat(
                absolute.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            if (temporary_stat.st_dev, temporary_stat.st_ino) == (
                current_output.st_dev,
                current_output.st_ino,
            ):
                os.unlink(absolute.name, dir_fd=parent_descriptor)
            raise ValueError("screening dispute output parent changed during publication")
        return digest.hexdigest()
    finally:
        try:
            if temporary_descriptor >= 0:
                try:
                    named = os.stat(
                        temporary_name,
                        dir_fd=parent_descriptor,
                        follow_symlinks=False,
                    )
                except FileNotFoundError:
                    pass
                else:
                    if temporary_identity == (named.st_dev, named.st_ino):
                        os.unlink(temporary_name, dir_fd=parent_descriptor)
        finally:
            if temporary_descriptor >= 0:
                os.close(temporary_descriptor)
            os.close(parent_descriptor)


def _publish_stage1_registry(
    project_root: Path,
    *,
    dataset: str,
    stage0_registry_path: Path,
    freeze_manifest_path: Path,
    freeze_payload: Mapping[str, Any],
    output_dir: Path,
    stage1_output: Path,
) -> dict[str, Any]:
    """Publish the exact Stage-1 digest wrapper after selection has succeeded."""

    expected_output = protocol.resolve_path(
        project_root,
        protocol.CAUSAL_STAGE1 if dataset == "causal" else protocol.SPECIFICITY_STAGE1,
    )
    if stage1_output.resolve() != expected_output.resolve():
        raise ValueError("Stage-1 output path differs from the standard protocol path")
    freeze_artifacts = freeze_payload.get("artifacts")
    if not isinstance(freeze_artifacts, Mapping):
        raise ValueError("Stage-1 requires a validated screening freeze")
    for name, record in freeze_artifacts.items():
        if not isinstance(record, Mapping) or set(record) != {"path", "sha256"}:
            raise ValueError(f"Stage-1 screening freeze/{name}: ref is not exact")
        current = protocol.resolve_path(project_root, str(record["path"]))
        if (
            not current.is_file()
            or current.is_symlink()
            or protocol.file_sha256(current) != record["sha256"]
        ):
            raise ValueError(f"Stage-1 screening freeze/{name}: bytes changed")

    def frozen_path(name: str) -> Path:
        record = freeze_artifacts.get(name)
        if not isinstance(record, Mapping):
            raise ValueError(f"Stage-1 screening freeze is missing {name}")
        return protocol.resolve_path(project_root, str(record.get("path", "")))

    package_path = frozen_path("screening_package")
    package = json.loads(package_path.read_text(encoding="utf-8"))
    generation_ref = package.get("generation_manifest")
    answer_ref = package.get("answer_key")
    if (
        not isinstance(generation_ref, Mapping)
        or not isinstance(answer_ref, Mapping)
        or set(generation_ref) != {"path", "sha256"}
        or set(answer_ref) != {"path", "sha256"}
    ):
        raise ValueError("Stage-1 screening package refs are not exact")
    generation_path = protocol.resolve_path(project_root, str(generation_ref["path"]))
    answer_path = protocol.resolve_path(project_root, str(answer_ref["path"]))
    if (
        protocol.file_sha256(generation_path) != generation_ref["sha256"]
        or protocol.file_sha256(answer_path) != answer_ref["sha256"]
    ):
        raise ValueError("Stage-1 screening package reference bytes changed")
    selector_paths = {
        "eligibility": output_dir / "eligibility_v2.csv",
        "summary": output_dir / "selector_output_v2.json",
        "selected": output_dir / "selected_cases_v2.csv",
        "units": output_dir / "unit_manifest_v2.csv",
    }
    for name, path in selector_paths.items():
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"Stage-1 selector artifact is missing: {name}")
    artifact_paths = {
        "screening_generation_manifest": generation_path,
        "screening_candidate_binding": answer_path,
        "screening_review_a": frozen_path("reviewer_a"),
        "screening_review_b": frozen_path("reviewer_b"),
        "screening_dispute_template": frozen_path("dispute_template"),
        "screening_adjudication": frozen_path("adjudication"),
        "screening_freeze_manifest": freeze_manifest_path,
        (
            "eligibility_table_48" if dataset == "causal" else "eligibility_table_36"
        ): selector_paths["eligibility"],
        (
            "selector_output_24" if dataset == "causal" else "selector_output_18"
        ): selector_paths["summary"],
        (
            "selected_case_manifest_24"
            if dataset == "causal"
            else "selected_case_manifest_18"
        ): selector_paths["selected"],
        (
            "unit_manifest_U_72" if dataset == "causal" else "unit_manifest_W_36"
        ): selector_paths["units"],
    }
    if dataset == "specificity":
        artifact_paths["holdout_mapping_M_6"] = output_dir / "holdout_mapping_M_v2.csv"
    expected_names = set(protocol.STAGE_ARTIFACTS[(dataset, 1)])
    if set(artifact_paths) != expected_names:
        raise ValueError("Stage-1 artifact inventory differs from protocol")
    artifacts: dict[str, dict[str, Any]] = {}
    for name, path in artifact_paths.items():
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"Stage-1 artifact missing: {name}")
        row_count = protocol.EXPECTED_COMMITMENT_ROW_COUNTS.get(
            (dataset, 1, name)
        )
        if row_count is not None and protocol._structured_row_count(path, row_count) != row_count:
            raise ValueError(f"Stage-1/{name}: row count differs from protocol")
        artifacts[name] = {
            "sha256": protocol.file_sha256(path),
            "size_bytes": path.stat().st_size,
            "row_count": row_count,
        }
    payload = {
        "protocol": protocol.COMMITMENT_PROTOCOL,
        "dataset": dataset,
        "dataset_version": protocol.DATASET_VERSION,
        "stage": 1,
        "status": "committed",
        "sealed_final36_status": "unopened",
        "stage0_registry_sha256": protocol.file_sha256(stage0_registry_path),
        "artifacts": artifacts,
    }
    _atomic_write_new_json(stage1_output, payload)
    return protocol.validate_commitment_registry(
        stage1_output,
        dataset=dataset,
        stage=1,
        expected_stage0_sha256=protocol.file_sha256(stage0_registry_path),
    )


def _cmd_derive_disputes(args: argparse.Namespace) -> int:
    protocol.reject_sealed_final36_path(
        args.public_root,
        args.template,
        args.reviewer_a,
        args.reviewer_b,
        args.output,
    )
    public_root = _absolute_lexical_path(args.public_root)
    resolved_public_root = public_root.resolve(strict=True)
    protocol.reject_sealed_final36_path(resolved_public_root)
    root_descriptor = _open_real_directory(
        public_root, label="screening public root"
    )
    try:
        resolved_root_stat = os.stat(resolved_public_root, follow_symlinks=False)
        opened_root_stat = os.fstat(root_descriptor)
        if (resolved_root_stat.st_dev, resolved_root_stat.st_ino) != (
            opened_root_stat.st_dev,
            opened_root_stat.st_ino,
        ):
            raise ValueError("screening public root changed during validation")
        if not _directory_descriptor_matches_path(
            root_descriptor,
            public_root,
            label="screening public root",
        ):
            raise ValueError("screening public root changed during validation")
        rows = {
            label: _read_exact_public_screening_csv(
                root_descriptor=root_descriptor,
                public_root=public_root,
                resolved_public_root=resolved_public_root,
                path=path,
                dataset=args.dataset,
                label=label,
            )
            for label, path in (
                ("public template", args.template),
                ("review A", args.reviewer_a),
                ("review B", args.reviewer_b),
            )
        }
    finally:
        os.close(root_descriptor)
    disputes = derive_public_screening_disputes(
        args.dataset,
        rows["public template"],
        rows["review A"],
        rows["review B"],
    )
    output_sha256 = _atomic_write_new_csv(
        args.output,
        disputes,
        fieldnames=("review_id", "field"),
    )
    print(
        json.dumps(
            {
                "status": "disputes_derived",
                "dataset": args.dataset,
                "review_count": protocol.CANDIDATE_COUNTS[args.dataset],
                "dispute_count": len(disputes),
                "output": str(args.output),
                "sha256": output_sha256,
            },
            sort_keys=True,
        )
    )
    return 0


def _cmd_freeze_screening(args: argparse.Namespace) -> int:
    protocol.reject_sealed_final36_path(
        *[value for value in vars(args).values() if isinstance(value, Path)]
    )
    freeze_screening_reviews(
        project_root=Path.cwd(),
        dataset=args.dataset,
        package_manifest_path=args.screening_package_manifest,
        private_root=args.private_root,
        candidate_manifest_path=args.candidate_manifest,
        canonical_templates_path=args.canonical_templates,
        screening_seed_path=args.screening_seed_file,
        generation_spec_path=args.generation_spec,
        reviewer_a_path=args.reviewer_a,
        reviewer_b_path=args.reviewer_b,
        dispute_path=args.dispute_template,
        adjudication_path=args.adjudication,
        canonical_path=args.canonical_eligibility,
        audit_path=args.adjudication_audit,
        freeze_manifest_path=args.freeze_manifest,
    )
    return 0


def _cmd_select(args: argparse.Namespace) -> int:
    protocol.reject_sealed_final36_path(
        args.screening_freeze_manifest,
        args.selector_salt_file,
        args.evaluation_seed_salt_file,
        args.output_dir,
        args.stage1_output,
        args.stage0_registry,
        args.private_root,
        args.candidate_manifest,
        *(tuple(value for value in (
            args.source_ontology,
            args.source_split,
            args.holdout_registry,
            args.receiver_ontology,
        ) if value is not None)),
        *(() if args.new_bank_assignment is None else (args.new_bank_assignment,)),
        args.canonical_templates,
        args.field_normalization,
        args.render_configuration,
        args.selection_rules,
        args.stage0_secrets,
        args.root_bundle,
        args.generation_spec,
        args.screening_seed_file,
        args.selection_binding,
        *(
            tuple(
                value
                for value in (
                    args.causal_selected,
                    args.causal_stage0_registry,
                    args.causal_stage1_registry,
                    args.causal_unit_manifest,
                )
                if value is not None
            )
        ),
    )
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite private selector output: {args.output_dir}")
    project_root = Path.cwd()
    expected_stage0 = protocol.resolve_path(
        project_root,
        protocol.CAUSAL_STAGE0 if args.dataset == "causal" else protocol.SPECIFICITY_STAGE0,
    )
    if args.stage0_registry.resolve() != expected_stage0.resolve():
        raise ValueError("selector Stage-0 registry path differs from protocol")
    stage0 = protocol.validate_commitment_registry(
        args.stage0_registry, dataset=args.dataset, stage=0
    )
    protocol.validate_selection_contract_opening(
        project_root,
        dataset=args.dataset,
        stage0_registry=stage0,
        private_root=args.private_root,
        candidate_manifest_path=args.candidate_manifest,
        canonical_templates_path=args.canonical_templates,
        field_rules_path=args.field_normalization,
        render_configuration_path=args.render_configuration,
        selection_rules_path=args.selection_rules,
        secrets_path=args.stage0_secrets,
        root_bundle_path=args.root_bundle,
        generation_spec_path=args.generation_spec,
        screening_seed_path=args.screening_seed_file,
        selector_salt_path=args.selector_salt_file,
        evaluation_seed_salt_path=args.evaluation_seed_salt_file,
        forbidden_seed_inventory_path=args.forbidden_seeds,
        selection_binding_path=args.selection_binding,
        source_ontology_path=args.source_ontology,
        source_split_path=args.source_split,
        holdout_registry_path=args.holdout_registry,
        receiver_ontology_path=args.receiver_ontology,
        new_bank_assignment_path=args.new_bank_assignment,
        causal_stage0_registry_path=args.causal_stage0_registry,
        causal_stage1_registry_path=args.causal_stage1_registry,
        causal_selected_path=args.causal_selected,
        causal_unit_manifest_path=args.causal_unit_manifest,
    )
    resolved_private = args.private_root.resolve(strict=True)
    if not args.private_root.is_dir() or args.private_root.is_symlink():
        raise ValueError("selector private root must be a real directory")
    for name, path in (
        ("selector_salt", args.selector_salt_file),
        ("evaluation_seed_salt", args.evaluation_seed_salt_file),
        ("forbidden_seed_inventory", args.forbidden_seeds),
    ):
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"selector/{name}: private artifact missing")
        try:
            path.resolve(strict=True).relative_to(resolved_private)
        except ValueError as exc:
            raise ValueError(f"selector/{name}: artifact escapes private root") from exc
        if protocol.file_sha256(path) != stage0["artifacts"][name]["sha256"]:
            raise ValueError(f"selector/{name}: bytes differ from Stage-0 commitment")
    selector_salt = args.selector_salt_file.read_text(encoding="utf-8").strip()
    seed_salt = args.evaluation_seed_salt_file.read_text(encoding="utf-8").strip()
    freeze_payload, candidates = validate_screening_freeze(
        project_root,
        args.screening_freeze_manifest,
        dataset=args.dataset,
        private_root=args.private_root,
        candidate_manifest_path=args.candidate_manifest,
        canonical_templates_path=args.canonical_templates,
        screening_seed_path=args.screening_seed_file,
        generation_spec_path=args.generation_spec,
    )
    forbidden = protocol.validate_forbidden_seed_inventory(
        args.forbidden_seeds, dataset=args.dataset
    )
    screening_seed_text = args.screening_seed_file.read_text(encoding="ascii")
    if not re.fullmatch(r"(?:0|[1-9][0-9]*)\n", screening_seed_text):
        raise ValueError("screening seed opening is not canonical decimal plus LF")
    screening_seed = int(screening_seed_text[:-1])
    if args.dataset == "causal":
        selected, eligibility = select_causal_cases(candidates, selector_salt)
        selected = _private_case_rows(selected, "causal")
        protocol.validate_causal_selected_cases(selected)
    else:
        if any(
            value is None
            for value in (
                args.causal_selected,
                args.causal_stage0_registry,
                args.causal_stage1_registry,
                args.causal_unit_manifest,
            )
        ):
            raise ValueError(
                "specificity selection requires causal Stage0/1, selected24, and U72"
            )
        if (
            args.causal_stage0_registry.resolve()
            != protocol.resolve_path(project_root, protocol.CAUSAL_STAGE0).resolve()
            or args.causal_stage1_registry.resolve()
            != protocol.resolve_path(project_root, protocol.CAUSAL_STAGE1).resolve()
        ):
            raise ValueError("specificity selector causal registry paths differ from protocol")
        causal0_sha = protocol.file_sha256(args.causal_stage0_registry)
        protocol.validate_commitment_registry(
            args.causal_stage0_registry, dataset="causal", stage=0
        )
        causal1 = protocol.validate_commitment_registry(
            args.causal_stage1_registry,
            dataset="causal",
            stage=1,
            expected_stage0_sha256=causal0_sha,
        )
        for label, path, artifact in (
            ("causal selected24", args.causal_selected, "selected_case_manifest_24"),
            ("causal U72", args.causal_unit_manifest, "unit_manifest_U_72"),
        ):
            if not path.is_file() or path.is_symlink():
                raise FileNotFoundError(f"specificity selector/{label}: file missing")
            try:
                path.resolve(strict=True).relative_to(resolved_private)
            except ValueError as exc:
                raise ValueError(f"specificity selector/{label}: escapes private root") from exc
            if protocol.file_sha256(path) != causal1["artifacts"][artifact]["sha256"]:
                raise ValueError(f"specificity selector/{label}: commitment mismatch")
        causal = protocol.read_csv(args.causal_selected)
        causal_units = protocol.read_csv(args.causal_unit_manifest)
        protocol.validate_causal_selected_cases(causal)
        protocol.validate_causal_unit_manifest(causal_units)
        causal_seeds = {int(row["seed"]) for row in causal_units}
        if not causal_seeds <= set(forbidden):
            raise ValueError("specificity forbidden-seed inventory does not contain all U seeds")
        selected, eligibility = select_specificity_cases(
            candidates, private_salt=selector_salt, causal_cases=causal
        )
        selected = _private_case_rows(selected, "specificity")
        protocol.validate_specificity_selected_cases(selected, causal_cases=causal)
    units = protocol.derive_unit_rows(
        selected,
        dataset=args.dataset,
        private_salt=seed_salt,
        forbidden_seeds={*forbidden, screening_seed},
    )
    args.output_dir.mkdir(parents=True)
    protocol.write_csv(args.output_dir / "eligibility_v2.csv", eligibility)
    protocol.write_csv(args.output_dir / "selected_cases_v2.csv", selected)
    protocol.write_csv(args.output_dir / "unit_manifest_v2.csv", units)
    if args.dataset == "specificity":
        mapping = [
            {
                "specificity_case_id": row["specificity_case_id"],
                "causal_case_id": row["causal_case_id"],
                "source_id": row["source_id"],
                "source_phrase": row["source_phrase"],
                "receiver_id": row["receiver_id"],
                "receiver": row["receiver"],
            }
            for row in selected
            if row["membership"] == "holdout_source"
        ]
        protocol.validate_holdout_mapping(
            mapping,
            causal_cases=causal,
            specificity_cases=selected,
        )
        protocol.write_csv(args.output_dir / "holdout_mapping_M_v2.csv", mapping)
    summary = {
        "protocol": protocol.PROTOCOL,
        "dataset": args.dataset,
        "dataset_version": protocol.DATASET_VERSION,
        "candidate_count": len(candidates),
        "eligible_count": sum(row["eligible"] == "yes" for row in eligibility),
        "selected_count": len(selected),
        "unit_count": len(units),
        "selection_rank_tuple": [
            next(
                row["selection_rank_sha256"]
                for row in eligibility
                if row["candidate_id"] == selected_row["candidate_id"]
            )
            for selected_row in selected
        ],
    }
    (args.output_dir / "selector_output_v2.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _publish_stage1_registry(
        project_root,
        dataset=args.dataset,
        stage0_registry_path=args.stage0_registry,
        freeze_manifest_path=args.screening_freeze_manifest,
        freeze_payload=freeze_payload,
        output_dir=args.output_dir,
        stage1_output=args.stage1_output,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    derive = sub.add_parser(
        "derive-disputes",
        help="derive the anonymous screening disagreement set from public reviews only",
    )
    derive.add_argument("--dataset", choices=protocol.DATASETS, required=True)
    derive.add_argument("--public-root", type=Path, required=True)
    derive.add_argument("--template", type=Path, required=True)
    derive.add_argument("--reviewer-a", type=Path, required=True)
    derive.add_argument("--reviewer-b", type=Path, required=True)
    derive.add_argument("--output", type=Path, required=True)
    derive.set_defaults(func=_cmd_derive_disputes)
    freeze = sub.add_parser(
        "freeze-screening", help="merge and freeze Original-only screening before selection"
    )
    freeze.add_argument("--dataset", choices=protocol.DATASETS, required=True)
    freeze.add_argument("--screening-package-manifest", type=Path, required=True)
    freeze.add_argument("--private-root", type=Path, required=True)
    freeze.add_argument("--candidate-manifest", type=Path, required=True)
    freeze.add_argument("--canonical-templates", type=Path, required=True)
    freeze.add_argument("--screening-seed-file", type=Path, required=True)
    freeze.add_argument("--generation-spec", type=Path, required=True)
    freeze.add_argument("--reviewer-a", type=Path, required=True)
    freeze.add_argument("--reviewer-b", type=Path, required=True)
    freeze.add_argument("--dispute-template", type=Path, required=True)
    freeze.add_argument("--adjudication", type=Path, required=True)
    freeze.add_argument("--canonical-eligibility", type=Path, required=True)
    freeze.add_argument("--adjudication-audit", type=Path, required=True)
    freeze.add_argument("--freeze-manifest", type=Path, required=True)
    freeze.set_defaults(func=_cmd_freeze_screening)
    select = sub.add_parser(
        "select", help="select only from a hash-bound canonical screening freeze"
    )
    select.add_argument("--dataset", choices=protocol.DATASETS, required=True)
    select.add_argument("--stage0-registry", type=Path, required=True)
    select.add_argument("--private-root", type=Path, required=True)
    select.add_argument("--candidate-manifest", type=Path, required=True)
    select.add_argument("--new-bank-assignment", type=Path)
    select.add_argument("--source-ontology", type=Path)
    select.add_argument("--source-split", type=Path)
    select.add_argument("--holdout-registry", type=Path)
    select.add_argument("--receiver-ontology", type=Path)
    select.add_argument("--canonical-templates", type=Path, required=True)
    select.add_argument("--field-normalization", type=Path, required=True)
    select.add_argument("--render-configuration", type=Path, required=True)
    select.add_argument("--selection-rules", type=Path, required=True)
    select.add_argument("--stage0-secrets", type=Path, required=True)
    select.add_argument("--root-bundle", type=Path, required=True)
    select.add_argument("--generation-spec", type=Path, required=True)
    select.add_argument("--screening-seed-file", type=Path, required=True)
    select.add_argument("--selection-binding", type=Path, required=True)
    select.add_argument("--screening-freeze-manifest", type=Path, required=True)
    select.add_argument("--selector-salt-file", type=Path, required=True)
    select.add_argument("--evaluation-seed-salt-file", type=Path, required=True)
    select.add_argument("--causal-selected", type=Path)
    select.add_argument("--causal-stage0-registry", type=Path)
    select.add_argument("--causal-stage1-registry", type=Path)
    select.add_argument("--causal-unit-manifest", type=Path)
    select.add_argument("--forbidden-seeds", type=Path, required=True)
    select.add_argument("--output-dir", type=Path, required=True)
    select.add_argument("--stage1-output", type=Path, required=True)
    select.set_defaults(func=_cmd_select)
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
