#!/usr/bin/env python3
"""Measure and visualize base-vs-adapter collision validation outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw


FRAME_INDICES = [0, 8, 16, 24, 32, 40, 48]


def load_video(path: Path) -> np.ndarray:
    frames = iio.imread(path, plugin="pyav")
    if frames.ndim != 4 or len(frames) < 49:
        raise ValueError(f"Unexpected video shape: {path} -> {frames.shape}")
    return frames[:, :, :, :3]


def read_items(output_dir: Path) -> list[dict[str, object]]:
    manifest = json.loads((output_dir / "generation_manifest.json").read_text(encoding="utf-8"))
    return list(manifest["items"])


def make_sheet(base: np.ndarray, adapted: np.ndarray, title: str, output: Path) -> None:
    cell_width = 208
    cell_height = round(base.shape[1] * cell_width / base.shape[2])
    label_width = 100
    header_height = 44
    sheet = Image.new("RGB", (label_width + cell_width * 7, header_height + cell_height * 2), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 8), title, fill="black")
    for column, frame_index in enumerate(FRAME_INDICES):
        draw.text((label_width + column * cell_width + 6, 25), f"frame {frame_index}", fill="black")
    for row, (label, frames) in enumerate((("base", base), ("adapter", adapted))):
        y = header_height + row * cell_height
        draw.text((8, y + 8), label, fill="black")
        for column, frame_index in enumerate(FRAME_INDICES):
            image = Image.fromarray(frames[frame_index]).resize((cell_width, cell_height))
            sheet.paste(image, (label_width + column * cell_width, y))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=92)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--sheet-dir", type=Path, required=True)
    args = parser.parse_args()

    base_items = read_items(args.base_dir)
    adapter_items = read_items(args.adapter_dir)
    if len(base_items) != len(adapter_items):
        raise ValueError("Base and adapter manifests have different item counts")

    rows = []
    for base_item, adapter_item in zip(base_items, adapter_items, strict=True):
        index = int(base_item["index"])
        if base_item["prompt"] != adapter_item["prompt"] or base_item["seed"] != adapter_item["seed"]:
            raise ValueError(f"Prompt or seed mismatch at index {index}")
        base_path = Path(str(base_item["video_path"]))
        adapter_path = Path(str(adapter_item["video_path"]))
        base = load_video(base_path)
        adapted = load_video(adapter_path)
        base_float = base.astype(np.float32) / 255.0
        adapted_float = adapted.astype(np.float32) / 255.0
        base_change = float(np.abs(base_float[32:49].mean(0) - base_float[:8].mean(0)).mean())
        adapter_change = float(np.abs(adapted_float[32:49].mean(0) - adapted_float[:8].mean(0)).mean())
        suppression = 100.0 * (1.0 - adapter_change / base_change) if base_change else 0.0
        early_divergence = float(np.abs(adapted_float[:17] - base_float[:17]).mean())
        late_divergence = float(np.abs(adapted_float[32:49] - base_float[32:49]).mean())
        sheet_path = args.sheet_dir / f"collision_validation_{index:02d}.jpg"
        make_sheet(base, adapted, f"validation {index} seed {base_item['seed']}", sheet_path)
        rows.append(
            {
                "index": index,
                "seed": base_item["seed"],
                "target_concept": base_item["target_concept"],
                "base_post_change_mae": f"{base_change:.8f}",
                "adapter_post_change_mae": f"{adapter_change:.8f}",
                "post_motion_suppression_percent": f"{suppression:.2f}",
                "early_base_adapter_mae": f"{early_divergence:.8f}",
                "late_base_adapter_mae": f"{late_divergence:.8f}",
                "base_video": str(base_path),
                "adapter_video": str(adapter_path),
                "contact_sheet": str(sheet_path),
            }
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    mean_suppression = float(np.mean([float(row["post_motion_suppression_percent"]) for row in rows]))
    mean_early = float(np.mean([float(row["early_base_adapter_mae"]) for row in rows]))
    print(f"count={len(rows)} mean_post_motion_suppression={mean_suppression:.2f}% mean_early_mae={mean_early:.8f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
