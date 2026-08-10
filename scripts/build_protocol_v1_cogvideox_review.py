#!/usr/bin/env python3
"""Build blinded Protocol v1 CogVideoX review sheets and annotation tables."""

from __future__ import annotations

import argparse
import csv
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import av
from PIL import Image, ImageDraw


FRAME_INDICES = (0, 8, 16, 24, 32, 40, 48)
BASELINES = (
    "negative_prompt",
    "t2vunlearning_adapted",
    "videoeraser_official",
)
ROOTS = {
    "original": "eval_cogvideox_original",
    "negative_prompt": "eval_cogvideox_negative_prompt",
    "t2vunlearning_adapted": "eval_t2vunlearning_adapted",
    "videoeraser_official": "videoeraser_official",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def index_videos(protocol_root: Path, mechanism: str, baseline: str) -> dict[str, Path]:
    manifest_path = protocol_root / ROOTS[baseline] / mechanism / "generation_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        str(item["sample_id"]): Path(str(item["video_path"]))
        for item in manifest["items"]
    }


def load_frames(path: Path) -> list[Image.Image]:
    selected: dict[int, Image.Image] = {}
    with av.open(str(path)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index in FRAME_INDICES:
                selected[index] = frame.to_image().convert("RGB")
            if index >= FRAME_INDICES[-1]:
                break
    missing = [index for index in FRAME_INDICES if index not in selected]
    if missing:
        raise ValueError(f"Missing frames {missing}: {path}")
    return [selected[index] for index in FRAME_INDICES]


def make_strip(frames: list[Image.Image], label: str | None = None) -> Image.Image:
    frame_width = 192
    frame_height = round(frames[0].height * frame_width / frames[0].width)
    label_width = 120 if label else 0
    header_height = 24
    strip = Image.new(
        "RGB",
        (label_width + frame_width * len(frames), header_height + frame_height),
        "white",
    )
    draw = ImageDraw.Draw(strip)
    if label:
        draw.text((8, header_height + 10), label, fill="black")
    for column, (frame_index, frame) in enumerate(zip(FRAME_INDICES, frames)):
        x = label_width + column * frame_width
        draw.text((x + 5, 5), f"frame {frame_index}", fill="black")
        strip.paste(frame.resize((frame_width, frame_height)), (x, header_height))
    return strip


def process_sample(
    row: dict[str, str],
    paths: dict[str, dict[str, Path]],
    output_dir: Path,
    blind_seed: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    sample_id = row["sample_id"]
    frames = {
        baseline: load_frames(paths[baseline][sample_id])
        for baseline in ("original", *BASELINES)
    }
    sample_dir = output_dir / "strips" / row["mechanism"] / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    strip_paths = {}
    for baseline, baseline_frames in frames.items():
        strip_path = sample_dir / f"{baseline}.jpg"
        make_strip(baseline_frames).save(strip_path, quality=92)
        strip_paths[baseline] = strip_path

    shuffled = list(BASELINES)
    random.Random(f"{blind_seed}:{sample_id}").shuffle(shuffled)
    candidate_codes = {baseline: chr(ord("A") + index) for index, baseline in enumerate(shuffled)}
    composite_rows = [("Original reference", frames["original"])] + [
        (f"Candidate {candidate_codes[baseline]}", frames[baseline]) for baseline in shuffled
    ]
    row_images = [make_strip(item_frames, label) for label, item_frames in composite_rows]
    title_height = 42
    composite = Image.new(
        "RGB",
        (row_images[0].width, title_height + sum(image.height for image in row_images)),
        "white",
    )
    draw = ImageDraw.Draw(composite)
    draw.text((8, 7), f"{sample_id} | {row['generalization_group']}", fill="black")
    draw.text((8, 23), f"Mechanism: {row['mechanism']}", fill="black")
    y = title_height
    for image in row_images:
        composite.paste(image, (0, y))
        y += image.height
    composite_path = output_dir / "composites" / row["mechanism"] / f"{sample_id}.jpg"
    composite_path.parent.mkdir(parents=True, exist_ok=True)
    composite.save(composite_path, quality=92)

    review_rows = []
    key_rows = []
    for baseline in shuffled:
        code = candidate_codes[baseline]
        review_rows.append(
            {
                "sample_id": sample_id,
                "mechanism": row["mechanism"],
                "generalization_group": row["generalization_group"],
                "candidate_code": code,
                "composite_path": str(composite_path),
                "source_object": row["source_object"],
                "receiver": row["receiver"],
                "expected_footprint": row["expected_footprint"],
                "source_absent": "",
                "footprint_absent": "",
                "receiver_preserved": "",
                "quality_ok": "",
                "notes": "",
            }
        )
        key_rows.append(
            {
                "sample_id": sample_id,
                "candidate_code": code,
                "baseline": baseline,
                "video_path": str(paths[baseline][sample_id]),
                "candidate_strip_path": str(strip_paths[baseline]),
                "reference_strip_path": str(strip_paths["original"]),
            }
        )
    return review_rows, key_rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--protocol-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--blind-seed", type=int, default=20260810)
    args = parser.parse_args()

    rows = read_csv(args.manifest)
    mechanisms = sorted({row["mechanism"] for row in rows})
    paths = {baseline: {} for baseline in ("original", *BASELINES)}
    for baseline in paths:
        for mechanism in mechanisms:
            paths[baseline].update(index_videos(args.protocol_root, mechanism, baseline))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        results = list(
            executor.map(
                lambda row: process_sample(
                    row, paths, args.output_dir, args.blind_seed
                ),
                rows,
            )
        )
    review_rows = [item for result, _ in results for item in result]
    key_rows = [item for _, result in results for item in result]
    write_csv(
        args.output_dir / "blind_review.csv",
        review_rows,
        [
            "sample_id",
            "mechanism",
            "generalization_group",
            "candidate_code",
            "composite_path",
            "source_object",
            "receiver",
            "expected_footprint",
            "source_absent",
            "footprint_absent",
            "receiver_preserved",
            "quality_ok",
            "notes",
        ],
    )
    write_csv(
        args.output_dir / "blind_key.csv",
        key_rows,
        [
            "sample_id",
            "candidate_code",
            "baseline",
            "video_path",
            "candidate_strip_path",
            "reference_strip_path",
        ],
    )
    original_rows = [
        {
            "sample_id": row["sample_id"],
            "mechanism": row["mechanism"],
            "generalization_group": row["generalization_group"],
            "source_object": row["source_object"],
            "receiver": row["receiver"],
            "expected_footprint": row["expected_footprint"],
            "reference_strip_path": str(
                args.output_dir
                / "strips"
                / row["mechanism"]
                / row["sample_id"]
                / "original.jpg"
            ),
            "source_visible": "",
            "footprint_visible": "",
            "receiver_correct": "",
            "quality_ok": "",
            "notes": "",
        }
        for row in rows
    ]
    write_csv(
        args.output_dir / "original_gate_review.csv",
        original_rows,
        list(original_rows[0]),
    )
    print(
        f"Built {len(rows)} composites and {len(review_rows)} blinded candidate rows "
        f"in {args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
