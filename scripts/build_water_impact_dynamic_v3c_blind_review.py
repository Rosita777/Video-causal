#!/usr/bin/env python3
"""Build the isolated 24-pair v3b-versus-v3c blinded review package."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import shutil
from pathlib import Path
from typing import Any, Callable

from PIL import Image, ImageDraw

import water_impact_dynamic_v3c_eval_protocol as protocol


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
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
        for row in sorted(rows, key=lambda row: str(row["review_id"]))
    ]
    return hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def load_frames(path: Path) -> list[Image.Image]:
    import av

    selected: dict[int, Image.Image] = {}
    with av.open(str(path)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index in protocol.FRAME_INDICES:
                selected[index] = frame.to_image().convert("RGB")
            if index >= protocol.FRAME_INDICES[-1]:
                break
    missing = [index for index in protocol.FRAME_INDICES if index not in selected]
    if missing:
        raise ValueError(f"missing frozen review frames {missing}: {path}")
    return [selected[index] for index in protocol.FRAME_INDICES]


def _strip(frames: list[Image.Image], label: str) -> Image.Image:
    frame_width = 208
    frame_height = round(frames[0].height * frame_width / frames[0].width)
    label_width = 120
    header_height = 24
    strip = Image.new(
        "RGB", (label_width + frame_width * len(frames), header_height + frame_height), "white"
    )
    draw = ImageDraw.Draw(strip)
    draw.text((8, header_height + 10), label, fill="black")
    for column, (frame_index, frame) in enumerate(
        zip(protocol.FRAME_INDICES, frames)
    ):
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
    strips = [_strip(frames, label) for label, frames in rows]
    title_height = 42
    composite = Image.new(
        "RGB", (strips[0].width, title_height + sum(strip.height for strip in strips)), "white"
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
    generation_manifests: dict[str, Path],
    stage2_path: Path,
    stage2_payload: dict[str, Any],
    public_dir: Path,
    private_dir: Path,
    blind_seed: int = protocol.BLIND_SEED,
    composite_builder: Callable[[Path, str, str, list[str], dict[str, Path]], None] = build_composite,
) -> dict[str, Any]:
    if len(eval_rows) != 24:
        raise ValueError("fresh-dev review requires exactly 24 rows")
    for label, directory in (("public", public_dir), ("private", private_dir)):
        if directory.exists():
            raise FileExistsError(f"refusing to overwrite {label} review directory: {directory}")
    if set(videos) != {"original", *protocol.METHODS}:
        raise ValueError("review requires original, v3b, and v3c video arms")
    for label, arm in videos.items():
        if set(arm) != set(range(24)):
            raise ValueError(f"{label}: expected video indices 0 through 23")
    resolved = {
        label: {path.resolve(strict=True) for path in arm.values()}
        for label, arm in videos.items()
    }
    labels = tuple(resolved)
    for left_index, left in enumerate(labels):
        for right in labels[left_index + 1 :]:
            if resolved[left] & resolved[right]:
                raise ValueError(f"{left}/{right} video inventories overlap")

    public_dir.mkdir(parents=True)
    private_dir.mkdir(parents=True)
    sample_order = list(range(24))
    random.Random(blind_seed).shuffle(sample_order)
    review_rows: list[dict[str, Any]] = []
    key_rows: list[dict[str, Any]] = []
    for review_position, sample_index in enumerate(sample_order):
        sample = eval_rows[sample_index]
        methods = list(protocol.METHODS)
        random.Random(f"{blind_seed}:{sample['pair_id']}").shuffle(methods)
        composite_path = public_dir / "composites" / f"r{review_position:03d}.jpg"
        sample_paths = {label: arm[sample_index] for label, arm in videos.items()}
        composite_builder(
            composite_path,
            sample["pair_id"],
            sample["generalization_group"],
            methods,
            sample_paths,
        )
        if not composite_path.is_file() or composite_path.stat().st_size == 0:
            raise FileNotFoundError(f"composite was not created: {composite_path}")
        for candidate_index, method in enumerate(methods):
            code = chr(ord("A") + candidate_index)
            review_id = f"r{review_position:03d}_{code}"
            anonymous = public_dir / "media" / f"{review_id}.mp4"
            anonymous.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(sample_paths[method], anonymous)
            if anonymous.is_symlink() or anonymous.samefile(sample_paths[method]):
                raise ValueError("anonymous media must be an independent real file copy")
            if protocol.file_sha256(anonymous) != protocol.file_sha256(sample_paths[method]):
                raise ValueError("anonymous media copy differs from its source")
            review_rows.append(
                {
                    "review_id": review_id,
                    "sample_index": sample_index,
                    "pair_id": sample["pair_id"],
                    "generalization_group": sample["generalization_group"],
                    "candidate_code": code,
                    "composite_path": str(composite_path),
                    "candidate_video_path": str(anonymous),
                    "source_object": sample["source_object"],
                    "receiver": sample["receiver"],
                    **{field: "" for field in protocol.SCORE_FIELDS},
                    "notes": "",
                }
            )
            key_rows.append(
                {
                    "review_id": review_id,
                    "sample_index": sample_index,
                    "pair_id": sample["pair_id"],
                    "generalization_group": sample["generalization_group"],
                    "candidate_code": code,
                    "method": method,
                    "video_path": str(sample_paths[method]),
                }
            )

    review_path = public_dir / "blind_review.csv"
    answer_key = private_dir / "answer_key.csv"
    write_csv(review_path, review_rows)
    write_csv(answer_key, sorted(key_rows, key=lambda row: str(row["review_id"])))
    manifest: dict[str, Any] = {
        "protocol": protocol.EVAL_PROTOCOL,
        "split_registry": {
            "path": protocol.SPLIT_REGISTRY,
            "sha256": protocol.SPLIT_REGISTRY_SHA256,
        },
        "fresh_dev_csv": protocol.FRESH_DEV_CSV,
        "fresh_dev_csv_sha256": protocol.file_sha256(
            protocol.resolve_path(Path.cwd(), protocol.FRESH_DEV_CSV)
        ),
        "blind_seed": blind_seed,
        "frame_indices": list(protocol.FRAME_INDICES),
        "sample_count": 24,
        "review_rows": 48,
        "methods": list(protocol.METHODS),
        "answer_key_sha256": protocol.file_sha256(answer_key),
        "review_binding_sha256": review_binding_sha256(review_rows),
        "stage2_registration": {
            "path": str(stage2_path),
            "sha256": protocol.file_sha256(stage2_path),
            "payload": stage2_payload,
        },
        "generation_manifests": {
            label: {"path": str(path), "sha256": protocol.file_sha256(path)}
            for label, path in generation_manifests.items()
        },
        "video_sha256": {
            label: {
                str(index): {"path": str(path), "sha256": protocol.file_sha256(path)}
                for index, path in sorted(arm.items())
            }
            for label, arm in videos.items()
        },
        "anonymous_media_sha256": {
            path.name: {"path": str(path), "sha256": protocol.file_sha256(path)}
            for path in sorted((public_dir / "media").glob("*.mp4"))
        },
        "composite_sha256": {
            path.name: protocol.file_sha256(path)
            for path in sorted((public_dir / "composites").glob("*.jpg"))
        },
    }
    manifest_path = private_dir / "review_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    project_root = Path.cwd()
    protocol.validate_split_registration(project_root)
    stage2_path, stage2 = protocol.load_stage2_registration(project_root)
    eval_rows = protocol.read_csv(protocol.resolve_path(project_root, protocol.FRESH_DEV_CSV))
    manifests: dict[str, Path] = {}
    videos: dict[str, dict[int, Path]] = {}
    for label, run in (
        ("original", protocol.ORIGINAL_RUN),
        ("v3b", protocol.V3B_RUN),
        ("v3c", protocol.V3C_RUN),
    ):
        manifests[label], _, videos[label] = protocol.load_generation_run(
            project_root,
            run,
            label,
            eval_rows,
            stage2_path,
            stage2,
            stage2["generation_spec"]["model_artifact_inventory"],
        )
    manifest = build_review_package(
        eval_rows=eval_rows,
        videos=videos,
        generation_manifests=manifests,
        stage2_path=stage2_path,
        stage2_payload=stage2,
        public_dir=protocol.resolve_path(project_root, protocol.PUBLIC_REVIEW_DIR),
        private_dir=protocol.resolve_path(project_root, protocol.PRIVATE_REVIEW_DIR),
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
