#!/usr/bin/env python3
"""Build automatic inference gates from target color and local video motion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import av
import numpy as np
import torch


def read_video(path: Path) -> torch.Tensor:
    with av.open(str(path)) as container:
        frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
    return torch.from_numpy(np.stack(frames)).float()


def build_gate(
    video: torch.Tensor,
    output_shape: tuple[int, int, int],
    motion_quantile: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    red, green, blue = video.unbind(dim=-1)
    target = (
        (red > 140)
        & (red - green > 60)
        & (red - blue > 50)
        & (red > 1.5 * green)
        & (red > 1.5 * blue)
        & (green < 100)
        & (blue < 100)
    ).float()

    motion = torch.zeros_like(red)
    motion[1:] = (video[1:] - video[:-1]).abs().mean(dim=-1)
    spatial_size = output_shape[-2:]
    motion = torch.nn.functional.interpolate(
        motion[:, None], size=spatial_size, mode="bilinear", align_corners=False
    )
    positive = motion[motion > 0]
    threshold = torch.quantile(positive, motion_quantile) if positive.numel() else torch.tensor(0.0)
    motion = ((motion - threshold) / threshold.clamp_min(1.0)).clamp(0.0, 1.0)
    target = torch.nn.functional.interpolate(
        target[:, None], size=spatial_size, mode="bilinear", align_corners=False
    )

    def resize_time(value: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.interpolate(
            value.permute(1, 0, 2, 3).unsqueeze(0),
            size=output_shape,
            mode="trilinear",
            align_corners=False,
        )[0, 0]

    target = resize_time(target)
    motion = resize_time(motion)
    target = torch.nn.functional.max_pool3d(
        target[None, None], kernel_size=(3, 5, 5), stride=1, padding=(1, 2, 2)
    )[0, 0]
    motion = torch.nn.functional.max_pool3d(
        motion[None, None], kernel_size=(3, 3, 3), stride=1, padding=1
    )[0, 0]
    gate = torch.maximum(target, motion).clamp(0.0, 1.0)
    return gate, target.clamp(0.0, 1.0), motion.clamp(0.0, 1.0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=13)
    parser.add_argument("--height", type=int, default=30)
    parser.add_argument("--width", type=int, default=52)
    parser.add_argument("--motion-quantile", type=float, default=0.92)
    args = parser.parse_args()
    if not 0.0 < args.motion_quantile < 1.0:
        parser.error("--motion-quantile must be between zero and one")
    items = json.loads(args.generation_manifest.read_text(encoding="utf-8"))["items"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    means = []
    for item in items:
        index = int(item["index"])
        gate, target, motion = build_gate(
            read_video(Path(item["video_path"])),
            (args.frames, args.height, args.width),
            args.motion_quantile,
        )
        torch.save(
            {
                "gate": gate.to(torch.float16),
                "target_gate": target.to(torch.float16),
                "motion_gate": motion.to(torch.float16),
                "source_video": item["video_path"],
                "motion_quantile": args.motion_quantile,
            },
            args.output_dir / f"{index:03d}.pt",
        )
        means.append(round(float(gate.mean()), 4))
    print(f"Built {len(items)} automatic inference gates; means={means}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
