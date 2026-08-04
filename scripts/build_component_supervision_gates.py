#!/usr/bin/env python3
"""Build collision object/receiver supervision gates from reviewed videos.

This is a small prototype for the collision experiment. The red target is
localized in RGB, then the existing causal gate is used to assign the remaining
causal area to the receiver. The training loss treats the two regions as
separate normalized terms, so a large receiver cannot hide a missed target.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def read_frames(path: Path, count: int) -> list[Image.Image]:
    import av

    with av.open(str(path)) as container:
        frames = [frame.to_image().convert("RGB") for frame in container.decode(video=0)]
    if len(frames) < count:
        raise ValueError(f"{path} has {len(frames)} frames; expected at least {count}")
    indices = np.linspace(0, len(frames) - 1, count).round().astype(int)
    return [frames[index] for index in indices]


def red_object_mask(frames: list[Image.Image], out_hw: tuple[int, int]) -> torch.Tensor:
    rgb = torch.from_numpy(np.stack([np.asarray(frame) for frame in frames])).float()
    red, green, blue = rgb.unbind(dim=-1)
    mask = ((red > 100) & (red - green > 32) & (red - blue > 24)).float()
    mask = mask[:, None]
    mask = torch.nn.functional.interpolate(mask, size=out_hw, mode="bilinear", align_corners=False)
    mask = torch.nn.functional.max_pool2d(mask, kernel_size=5, stride=1, padding=2)
    return mask[:, 0].clamp(0.0, 1.0)


def video_difference_mask(
    factual: list[Image.Image], target: list[Image.Image], out_shape: tuple[int, int, int]
) -> torch.Tensor:
    factual_rgb = torch.from_numpy(np.stack([np.asarray(frame) for frame in factual])).float()
    target_rgb = torch.from_numpy(np.stack([np.asarray(frame) for frame in target])).float()
    difference = (factual_rgb - target_rgb).abs().mean(dim=-1)
    difference = torch.nn.functional.interpolate(
        difference[:, None], size=out_shape[-2:], mode="bilinear", align_corners=False
    )
    low = torch.quantile(difference, 0.75)
    high = torch.quantile(difference, 0.97)
    difference = ((difference - low) / (high - low).clamp_min(1e-6)).clamp(0.0, 1.0)
    difference = torch.nn.functional.interpolate(
        difference.permute(1, 0, 2, 3).unsqueeze(0),
        size=out_shape,
        mode="trilinear",
        align_corners=False,
    )[0, 0]
    return torch.nn.functional.max_pool3d(
        difference[None, None], kernel_size=(3, 5, 5), stride=1, padding=(1, 2, 2)
    )[0, 0].clamp(0.0, 1.0)


def build_one(row: dict[str, str], project_root: Path, causal_dir: Path, frames: int) -> dict[str, object]:
    payload = torch.load(causal_dir / f"{row['scene_id']}.pt", map_location="cpu", weights_only=True)
    causal = payload["gate"].float()
    if causal.ndim != 3:
        raise ValueError(f"Expected [T,H,W] causal gate, got {tuple(causal.shape)}")
    factual = project_root / row["residual_mask_factual_video"]
    target = project_root / row["desired_target_video"]
    factual_frames = read_frames(factual, frames)
    target_frames = read_frames(target, frames)
    object_gate = red_object_mask(factual_frames, causal.shape[-2:])
    if object_gate.shape[0] != causal.shape[0]:
        object_gate = torch.nn.functional.interpolate(
            object_gate[None, None], size=causal.shape, mode="trilinear", align_corners=False
        )[0, 0]
    object_gate = object_gate * (causal > 0.1).float()
    object_gate = torch.nn.functional.max_pool3d(
        object_gate[None, None], kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)
    )[0, 0].clamp(0.0, 1.0)
    object_exclusion = torch.nn.functional.max_pool3d(
        object_gate[None, None], kernel_size=(3, 5, 5), stride=1, padding=(1, 2, 2)
    )[0, 0].clamp(0.0, 1.0)
    receiver_gate = video_difference_mask(factual_frames, target_frames, tuple(causal.shape))
    receiver_gate = receiver_gate * (1.0 - object_exclusion).clamp(0.0, 1.0)
    activation_gate = torch.maximum(object_gate, receiver_gate)
    return {
        "scene_id": row["scene_id"],
        "gate": activation_gate.to(torch.float16),
        "object_gate": object_gate.to(torch.float16),
        "receiver_gate": receiver_gate.to(torch.float16),
        "causal_gate": causal.to(torch.float16),
        "object_mean": float(object_gate.mean()),
        "receiver_mean": float(receiver_gate.mean()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--causal-gate-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = Path.cwd()
    with args.manifest.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["training_role"] == "erase"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for row in rows:
        result = build_one(row, root, args.causal_gate_dir, args.num_frames)
        summary.append({k: result[k] for k in ("scene_id", "object_mean", "receiver_mean")})
        if not args.dry_run:
            torch.save(result, args.output_dir / f"{row['scene_id']}.pt")
    print(f"Built component supervision for {len(rows)} scenes")
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
