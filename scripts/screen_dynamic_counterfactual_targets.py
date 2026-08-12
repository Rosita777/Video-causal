#!/usr/bin/env python3
"""Screen dynamic counterfactual targets and build temporal contact sheets."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


FIELDS = [
    "pair_index",
    "pair_id",
    "source_id",
    "receiver_id",
    "receiver",
    "prompt_variant",
    "seed",
    "video_path",
    "frame_count",
    "fps",
    "mean_adjacent_mae",
    "first_last_mae",
    "max_adjacent_mae",
    "max_to_median_ratio",
    "technical_status",
    "semantic_status",
    "final_status",
    "contact_sheet",
    "notes",
]


def load_video(path: Path) -> tuple[np.ndarray, float]:
    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate else 0.0
        frames = np.stack(
            [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
        )
    return frames, fps


def temporal_metrics(frames: np.ndarray) -> dict[str, float]:
    values = frames.astype(np.float32)
    adjacent = np.abs(np.diff(values, axis=0)).mean(axis=(1, 2, 3))
    median = float(np.median(adjacent))
    return {
        "mean_adjacent_mae": float(adjacent.mean()),
        "first_last_mae": float(np.abs(values[-1] - values[0]).mean()),
        "max_adjacent_mae": float(adjacent.max()),
        "max_to_median_ratio": float(adjacent.max() / max(median, 1e-6)),
    }


def technical_status(frame_count: int, fps: float, metrics: dict[str, float]) -> str:
    if frame_count != 49 or abs(fps - 8.0) > 0.1:
        return "reject_invalid_video"
    if metrics["mean_adjacent_mae"] < 0.08 or metrics["first_last_mae"] < 0.35:
        return "reject_nearly_static"
    if metrics["max_to_median_ratio"] > 4.0 and metrics["max_adjacent_mae"] > 4.0:
        return "review_abrupt_transition"
    return "candidate"


def make_sheet(frames: np.ndarray, output: Path, title: str, status: str) -> None:
    indices = [0, 8, 16, 24, 32, 40, 48]
    cell_width = 208
    cell_height = round(frames.shape[1] * cell_width / frames.shape[2])
    header = 44
    sheet = Image.new("RGB", (cell_width * len(indices), header + cell_height), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((6, 5), f"{title} | {status}", fill="black")
    for column, index in enumerate(indices):
        draw.text((column * cell_width + 5, 24), f"f{index:02d}", fill="black")
        image = Image.fromarray(frames[index]).resize((cell_width, cell_height))
        sheet.paste(image, (column * cell_width, header))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=90)


def portable(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--generation-manifest", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--sheet-dir", type=Path, required=True)
    args = parser.parse_args()

    root = args.repo_root.resolve()
    pairs_path = args.pairs if args.pairs.is_absolute() else root / args.pairs
    manifest_path = (
        args.generation_manifest
        if args.generation_manifest.is_absolute()
        else root / args.generation_manifest
    )
    output_csv = args.output_csv if args.output_csv.is_absolute() else root / args.output_csv
    sheet_dir = args.sheet_dir if args.sheet_dir.is_absolute() else root / args.sheet_dir

    with pairs_path.open(newline="", encoding="utf-8") as handle:
        pairs = list(csv.DictReader(handle))
    generation = json.loads(manifest_path.read_text(encoding="utf-8"))
    items = {int(item["index"]): item for item in generation["items"]}
    if len(pairs) != len(items):
        raise ValueError(f"Pair count {len(pairs)} != generated item count {len(items)}")

    rows: list[dict[str, str]] = []
    for index, pair in enumerate(pairs):
        item = items[index]
        video = Path(str(item["video_path"]))
        if not video.is_absolute():
            video = root / video
        frames, fps = load_video(video)
        metrics = temporal_metrics(frames)
        status = technical_status(len(frames), fps, metrics)
        sheet = sheet_dir / f"{index:03d}_{pair['receiver_id']}_{pair['prompt_variant']}.jpg"
        make_sheet(frames, sheet, f"{index:03d} | {pair['receiver_id']}", status)
        rows.append(
            {
                "pair_index": str(index),
                "pair_id": pair["pair_id"],
                "source_id": pair["source_id"],
                "receiver_id": pair["receiver_id"],
                "receiver": pair["receiver"],
                "prompt_variant": pair["prompt_variant"],
                "seed": pair["seed"],
                "video_path": portable(video, root),
                "frame_count": str(len(frames)),
                "fps": f"{fps:g}",
                **{key: f"{value:.4f}" for key, value in metrics.items()},
                "technical_status": status,
                "semantic_status": "",
                "final_status": "",
                "contact_sheet": portable(sheet, root),
                "notes": "",
            }
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Screened {len(rows)} targets: {dict(Counter(row['technical_status'] for row in rows))}")
    print(f"CSV: {output_csv}")
    print(f"Sheets: {sheet_dir}")


if __name__ == "__main__":
    main()
