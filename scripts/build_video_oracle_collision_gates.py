#!/usr/bin/env python3
"""Build approximate collision gates from red-object and local-motion cues."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw
import torch


LATENT_FRAMES = 13
GRID_HEIGHT = 30
GRID_WIDTH = 52


def resize_frames(frames: np.ndarray) -> torch.Tensor:
    indices = np.linspace(0, len(frames) - 1, LATENT_FRAMES).round().astype(int)
    resized = [
        np.asarray(Image.fromarray(frames[index, :, :, :3]).resize((GRID_WIDTH, GRID_HEIGHT)))
        for index in indices
    ]
    return torch.from_numpy(np.stack(resized)).float() / 255.0


def dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return mask
    kernel = 2 * radius + 1
    return torch.nn.functional.max_pool2d(
        mask[:, None], kernel_size=kernel, stride=1, padding=radius
    )[:, 0]


def build_gate(frames: torch.Tensor) -> tuple[torch.Tensor, dict[str, float | int]]:
    red = frames[..., 0] - torch.maximum(frames[..., 1], frames[..., 2])
    red_mask = (red > 0.14) & (frames[..., 0] > 0.32)
    persistent_red = red_mask[:3].float().mean(dim=0) >= 0.67
    persistent_red = dilate(persistent_red[None].float(), 1)[0] > 0
    red_mask &= ~persistent_red
    red_pixels = red_mask.flatten(1).sum(1)
    visible = torch.where(red_pixels >= 2)[0]
    if len(visible) == 0:
        return torch.zeros((LATENT_FRAMES, GRID_HEIGHT, GRID_WIDTH)), {
            "target_detected": 0,
            "start_frame": -1,
            "coverage": 0.0,
        }

    motion = torch.zeros((LATENT_FRAMES, GRID_HEIGHT, GRID_WIDTH))
    motion[1:] = (frames[1:] - frames[:-1]).abs().mean(dim=-1)
    positive_motion = motion[motion > 0]
    threshold = max(0.035, float(torch.quantile(positive_motion, 0.82)))
    motion_mask = motion >= threshold

    start = int(visible[0])
    gate = torch.zeros_like(motion)
    previous = torch.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=torch.bool)
    for frame_index in range(start, LATENT_FRAMES):
        object_region = dilate(red_mask[frame_index][None].float(), 2)[0] > 0
        reachable = dilate(previous[None].float(), 4)[0] > 0
        footprint = motion_mask[frame_index] & reachable
        current = object_region | footprint
        gate[frame_index] = dilate(current[None].float(), 2)[0]
        previous = current | (dilate(previous[None].float(), 1)[0] > 0)
    return gate.clamp(0, 1), {
        "target_detected": 1,
        "start_frame": start,
        "motion_threshold": threshold,
        "coverage": float(gate.mean()),
    }


def save_preview(frames: torch.Tensor, gate: torch.Tensor, output: Path) -> None:
    scale = 4
    cell_width = GRID_WIDTH * scale
    cell_height = GRID_HEIGHT * scale
    selected = [0, 2, 4, 6, 8, 10, 12]
    canvas = Image.new("RGB", (cell_width * len(selected), cell_height + 24), "white")
    draw = ImageDraw.Draw(canvas)
    for column, frame_index in enumerate(selected):
        image = Image.fromarray((frames[frame_index].numpy() * 255).astype(np.uint8)).resize(
            (cell_width, cell_height)
        )
        overlay = np.zeros((GRID_HEIGHT, GRID_WIDTH, 4), dtype=np.uint8)
        active = gate[frame_index].numpy() > 0.5
        overlay[active] = [255, 0, 0, 100]
        mask_image = Image.fromarray(overlay, mode="RGBA").resize((cell_width, cell_height))
        image = Image.alpha_composite(image.convert("RGBA"), mask_image).convert("RGB")
        canvas.paste(image, (column * cell_width, 24))
        draw.text((column * cell_width + 4, 5), f"frame {frame_index}", fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=92)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    manifest = json.loads((args.base_dir / "generation_manifest.json").read_text(encoding="utf-8"))
    items = manifest["items"][: args.limit] if args.limit is not None else manifest["items"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for item in items:
        index = int(item["index"])
        video_path = Path(str(item["video_path"]))
        sampled = resize_frames(iio.imread(video_path, plugin="pyav"))
        gate, metrics = build_gate(sampled)
        torch.save(
            {"index": index, "video_path": str(video_path), "gate": gate.to(torch.float16), **metrics},
            args.output_dir / f"{index:03d}.pt",
        )
        save_preview(sampled, gate, args.output_dir / "previews" / f"{index:03d}.jpg")
        summary.append({"index": index, "video_path": str(video_path), **metrics})
        print(f"index={index:03d} detected={metrics['target_detected']} coverage={metrics['coverage']:.4f}")
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
