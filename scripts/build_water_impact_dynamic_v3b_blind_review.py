#!/usr/bin/env python3
"""Build the frozen two-arm blinded review for balanced control versus v3b."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import shutil
from pathlib import Path
from typing import Any, Callable

from PIL import Image, ImageDraw

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
    validate_balanced_checkpoint,
    validate_training_caches,
    validate_v3b_checkpoint,
)


BLIND_SEED = 26016
PUBLIC_OUTPUT_DIR = Path(
    "experiments/water_impact_dynamic_eval12/v3b_target_prompt_teacher_blind_review_v3_public"
)
PRIVATE_OUTPUT_DIR = Path(
    "experiments/water_impact_dynamic_eval12/v3b_target_prompt_teacher_blind_review_v3_private"
)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def review_binding_sha256(rows: list[dict[str, Any]]) -> str:
    fields = (
        "review_id",
        "sample_index",
        "pair_id",
        "generalization_group",
        "candidate_code",
        "composite_path",
        "candidate_video_path",
        "source_object",
        "receiver",
    )
    canonical = [
        {field: str(row[field]) for field in fields}
        for row in sorted(rows, key=lambda item: str(item["review_id"]))
    ]
    return hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def load_frames(path: Path) -> list[Image.Image]:
    import av

    selected: dict[int, Image.Image] = {}
    with av.open(str(path)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index in FRAME_INDICES:
                selected[index] = frame.to_image().convert("RGB")
            if index >= FRAME_INDICES[-1]:
                break
    missing = [index for index in FRAME_INDICES if index not in selected]
    if missing:
        raise ValueError(f"missing frozen review frames {missing}: {path}")
    return [selected[index] for index in FRAME_INDICES]


def make_strip(frames: list[Image.Image], label: str) -> Image.Image:
    frame_width = 208
    frame_height = round(frames[0].height * frame_width / frames[0].width)
    label_width = 120
    header_height = 24
    strip = Image.new(
        "RGB",
        (label_width + frame_width * len(frames), header_height + frame_height),
        "white",
    )
    draw = ImageDraw.Draw(strip)
    draw.text((8, header_height + 10), label, fill="black")
    for column, (frame_index, frame) in enumerate(zip(FRAME_INDICES, frames, strict=True)):
        x = label_width + column * frame_width
        draw.text((x + 5, 5), f"frame {frame_index}", fill="black")
        strip.paste(frame.resize((frame_width, frame_height)), (x, header_height))
    return strip


def build_composite(
    output_path: Path,
    pair_id: str,
    generalization_group: str,
    ordered_methods: list[str],
    paths: dict[str, Path],
) -> None:
    rows = [("Original reference", load_frames(paths["original"]))]
    rows.extend(
        (f"Candidate {chr(ord('A') + index)}", load_frames(paths[method]))
        for index, method in enumerate(ordered_methods)
    )
    strips = [make_strip(frames, label) for label, frames in rows]
    title_height = 42
    composite = Image.new(
        "RGB",
        (strips[0].width, title_height + sum(strip.height for strip in strips)),
        "white",
    )
    draw = ImageDraw.Draw(composite)
    draw.text((8, 7), pair_id, fill="black")
    draw.text((8, 23), f"Generalization: {generalization_group}", fill="black")
    y = title_height
    for strip in strips:
        composite.paste(strip, (0, y))
        y += strip.height
    output_path.parent.mkdir(parents=True, exist_ok=True)
    composite.save(output_path, quality=92)


def build_review_package(
    *,
    eval_rows: list[dict[str, str]],
    videos: dict[str, dict[int, Path]],
    manifest_paths: dict[str, Path],
    training_provenance: dict[str, dict[str, Any]],
    public_dir: Path,
    private_dir: Path,
    blind_seed: int = BLIND_SEED,
    composite_builder: Callable[[Path, str, str, list[str], dict[str, Path]], None] = build_composite,
) -> dict[str, Any]:
    for label, directory in (("public", public_dir), ("private", private_dir)):
        if directory.exists():
            raise FileExistsError(
                f"refusing to overwrite {label} review directory: {directory}"
            )
    if set(videos) != {"original", *METHODS}:
        raise ValueError("review inputs must contain original, balanced, and v3b videos")
    for label, arm in videos.items():
        if set(arm) != set(range(12)):
            raise ValueError(f"{label}: review inputs must contain indices 0 through 11")
    resolved_by_arm = {
        label: {path.resolve(strict=True) for path in arm.values()}
        for label, arm in videos.items()
    }
    labels = tuple(resolved_by_arm)
    for left_index, left in enumerate(labels):
        for right in labels[left_index + 1 :]:
            if resolved_by_arm[left] & resolved_by_arm[right]:
                raise ValueError(f"{left} and {right} reference overlapping video files")

    public_dir.mkdir(parents=True)
    private_dir.mkdir(parents=True)
    sample_order = list(range(12))
    random.Random(blind_seed).shuffle(sample_order)
    review_rows: list[dict[str, Any]] = []
    key_rows: list[dict[str, Any]] = []
    for review_position, sample_index in enumerate(sample_order):
        sample = eval_rows[sample_index]
        pair_id = sample["pair_id"]
        ordered_methods = list(METHODS)
        random.Random(f"{blind_seed}:{pair_id}").shuffle(ordered_methods)
        composite_path = public_dir / "composites" / f"r{review_position:03d}.jpg"
        sample_paths = {label: arm[sample_index] for label, arm in videos.items()}
        composite_builder(
            composite_path,
            pair_id,
            sample["generalization_group"],
            ordered_methods,
            sample_paths,
        )
        if not composite_path.is_file() or composite_path.stat().st_size == 0:
            raise FileNotFoundError(f"composite builder did not create {composite_path}")
        for candidate_index, method in enumerate(ordered_methods):
            candidate_code = chr(ord("A") + candidate_index)
            review_id = f"r{review_position:03d}_{candidate_code}"
            candidate_video_path = public_dir / "media" / f"{review_id}.mp4"
            candidate_video_path.parent.mkdir(parents=True, exist_ok=True)
            if candidate_video_path.exists() or candidate_video_path.is_symlink():
                raise FileExistsError(
                    f"refusing to overwrite anonymous candidate media: {candidate_video_path}"
                )
            shutil.copyfile(sample_paths[method], candidate_video_path)
            if (
                not candidate_video_path.is_file()
                or candidate_video_path.is_symlink()
                or candidate_video_path.stat().st_size == 0
                or candidate_video_path.samefile(sample_paths[method])
            ):
                raise ValueError(
                    f"anonymous candidate media is not an independent file copy: "
                    f"{candidate_video_path}"
                )
            if file_sha256(candidate_video_path) != file_sha256(sample_paths[method]):
                raise ValueError(
                    f"anonymous candidate media differs from its source: {candidate_video_path}"
                )
            review_rows.append(
                {
                    "review_id": review_id,
                    "sample_index": sample_index,
                    "pair_id": pair_id,
                    "generalization_group": sample["generalization_group"],
                    "candidate_code": candidate_code,
                    "composite_path": str(composite_path),
                    "candidate_video_path": str(candidate_video_path),
                    "source_object": sample["source_object"],
                    "receiver": sample["receiver"],
                    **{field: "" for field in SCORE_FIELDS},
                    "notes": "",
                }
            )
            key_rows.append(
                {
                    "review_id": review_id,
                    "sample_index": sample_index,
                    "pair_id": pair_id,
                    "generalization_group": sample["generalization_group"],
                    "candidate_code": candidate_code,
                    "method": method,
                    "video_path": str(sample_paths[method]),
                }
            )

    review_path = public_dir / "blind_review.csv"
    answer_key_path = private_dir / "answer_key.csv"
    write_csv(review_path, review_rows)
    write_csv(answer_key_path, sorted(key_rows, key=lambda row: str(row["review_id"])))
    review_manifest: dict[str, Any] = {
        "protocol": PROTOCOL,
        "eval_csv": EVAL_CSV,
        "eval_csv_sha256": EVAL_CSV_SHA256,
        "prompts_sha256": PROMPTS_SHA256,
        "train_manifest": TRAIN_MANIFEST,
        "train_manifest_sha256": TRAIN_MANIFEST_SHA256,
        "blind_seed": blind_seed,
        "frame_indices": list(FRAME_INDICES),
        "sample_count": 12,
        "review_rows": 24,
        "methods": list(METHODS),
        "answer_key_sha256": file_sha256(answer_key_path),
        "review_binding_sha256": review_binding_sha256(review_rows),
        "training_provenance": training_provenance,
        "video_sha256": {
            label: {
                str(index): {"path": str(path), "sha256": file_sha256(path)}
                for index, path in sorted(arm.items())
            }
            for label, arm in videos.items()
        },
        "anonymous_media_sha256": {
            path.name: {"path": str(path), "sha256": file_sha256(path)}
            for path in sorted((public_dir / "media").glob("*.mp4"))
        },
        "composite_sha256": {
            path.name: file_sha256(path)
            for path in sorted((public_dir / "composites").glob("*.jpg"))
        },
        "generation_manifests": {
            label: {"path": str(path), "sha256": file_sha256(path)}
            for label, path in manifest_paths.items()
        },
    }
    manifest_path = private_dir / "review_manifest.json"
    manifest_path.write_text(
        json.dumps(review_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"Built public/private v3 package for 12 pairs; "
        f"private review manifest SHA-256={file_sha256(manifest_path)}"
    )
    return review_manifest


def main() -> int:
    project_root = Path.cwd()
    for label, directory in (
        ("public", PUBLIC_OUTPUT_DIR),
        ("private", PRIVATE_OUTPUT_DIR),
    ):
        if directory.exists():
            raise SystemExit(
                f"refusing to overwrite {label} review directory: {directory}"
            )
    eval_rows, train_rows = load_frozen_inputs(project_root)
    run_specs = {
        "original": (ORIGINAL_RUN, ORIGINAL_GENERATION_MANIFEST_SHA256),
        "balanced": (BALANCED_RUN, BALANCED_GENERATION_MANIFEST_SHA256),
        "v3b": (V3B_RUN, None),
    }
    manifest_paths: dict[str, Path] = {}
    videos: dict[str, dict[int, Path]] = {}
    for label, (run_dir, expected_hash) in run_specs.items():
        manifest_paths[label], _, videos[label] = load_generation_run(
            project_root,
            run_dir,
            label,
            eval_rows,
            expected_manifest_sha256=expected_hash,
        )
    training_provenance = {
        "inputs": validate_training_caches(project_root, train_rows),
        "balanced": validate_balanced_checkpoint(project_root),
        "v3b": validate_v3b_checkpoint(project_root),
    }
    build_review_package(
        eval_rows=eval_rows,
        videos=videos,
        manifest_paths=manifest_paths,
        training_provenance=training_provenance,
        public_dir=PUBLIC_OUTPUT_DIR,
        private_dir=PRIVATE_OUTPUT_DIR,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
