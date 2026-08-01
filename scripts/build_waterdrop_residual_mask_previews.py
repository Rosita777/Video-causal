#!/usr/bin/env python3
"""Visualize pixel residual masks before implementing their latent-space loss."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from build_static_counterfactual_pair import load_video


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def soft_mask(frame: np.ndarray, target: np.ndarray) -> np.ndarray:
    difference = np.abs(frame.astype(np.float32) - target.astype(np.float32)).mean(axis=2) / 255.0
    mask = np.clip((difference - 0.015) / (0.12 - 0.015), 0.0, 1.0)
    image = Image.fromarray(np.uint8(mask * 255)).filter(ImageFilter.GaussianBlur(radius=5))
    return np.asarray(image, dtype=np.float32) / 255.0


def overlay(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    result = frame.astype(np.float32).copy()
    alpha = (0.55 * mask)[..., None]
    red = np.zeros_like(result)
    red[..., 0] = 255
    return np.uint8(result * (1 - alpha) + red * alpha)


def contact_sheet(
    factual: np.ndarray,
    target: np.ndarray,
    masks: np.ndarray,
    output: Path,
) -> None:
    indices = np.linspace(0, len(factual) - 1, 6, dtype=int)
    width = 240
    height = round(factual.shape[1] * width / factual.shape[2])
    label_height = 24
    rows = 4
    sheet = Image.new("RGB", (width * len(indices), (height + label_height) * rows), "white")
    draw = ImageDraw.Draw(sheet)
    labels = ["factual", "counterfactual", "residual mask", "overlay"]
    for column, frame_index in enumerate(indices):
        views = [
            factual[frame_index],
            target[frame_index],
            np.repeat(np.uint8(masks[frame_index, ..., None] * 255), 3, axis=2),
            overlay(factual[frame_index], masks[frame_index]),
        ]
        for row, view in enumerate(views):
            x = column * width
            y = row * (height + label_height)
            draw.text((x + 6, y + 5), f"{labels[row]} f{frame_index}", fill="black")
            sheet.paste(Image.fromarray(view).resize((width, height)), (x, y + label_height))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=94)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/waterdrop_train_pilot40_sft_preliminary.csv"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/waterdrop_train_pilot40_residual_mask_previews"),
    )
    args = parser.parse_args()
    root = args.repo_root.resolve()
    manifest = args.manifest if args.manifest.is_absolute() else root / args.manifest
    output_root = args.output_root if args.output_root.is_absolute() else root / args.output_root

    metrics = []
    for row in read_csv(manifest):
        if row["residual_mask_enabled"] != "yes":
            continue
        factual, factual_fps = load_video(root / row["residual_mask_factual_video"])
        target, _ = load_video(root / row["residual_mask_target_video"])
        if factual.shape != target.shape:
            raise ValueError(f"shape mismatch for {row['scene_id']}: {factual.shape} vs {target.shape}")
        masks = np.stack([soft_mask(frame, target_frame) for frame, target_frame in zip(factual, target)])
        group_dir = output_root / row["train_group_id"]
        group_dir.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(
            group_dir / "residual_mask.mp4",
            fps=factual_fps,
            codec="libx264",
            quality=8,
            macro_block_size=None,
        )
        try:
            for mask in masks:
                writer.append_data(np.repeat(np.uint8(mask[..., None] * 255), 3, axis=2))
        finally:
            writer.close()
        contact_sheet(factual, target, masks, group_dir / "residual_mask_contact_sheet.jpg")
        changed = (masks > 0.2).mean(axis=(1, 2))
        metrics.append(
            {
                "train_group_id": row["train_group_id"],
                "scene_id": row["scene_id"],
                "mean_changed_fraction": f"{changed.mean():.8f}",
                "peak_changed_fraction": f"{changed.max():.8f}",
                "first_frame_changed_fraction": f"{changed[0]:.8f}",
                "contact_sheet": str(
                    (group_dir / "residual_mask_contact_sheet.jpg").relative_to(root)
                ),
            }
        )
    output_csv = root / "data/waterdrop_train_pilot40_residual_mask_metrics.csv"
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    print(f"wrote {len(metrics)} residual-mask previews and {output_csv}")


if __name__ == "__main__":
    main()
