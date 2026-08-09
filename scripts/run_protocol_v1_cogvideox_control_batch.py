#!/usr/bin/env python3
"""Generate Original or Negative Prompt CogVideoX controls for Protocol v1."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--mechanism", required=True)
    parser.add_argument("--baseline", choices=("original", "negative_prompt"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=6.0)
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    import torch
    from diffusers import CogVideoXDPMScheduler, CogVideoXPipeline
    from diffusers.utils import export_to_video

    with args.manifest.open(newline="", encoding="utf-8") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row["mechanism"] == args.mechanism
        ]
    if len(rows) != 20:
        raise ValueError(f"Expected 20 rows for {args.mechanism}, found {len(rows)}")

    pipe = CogVideoXPipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    pipe.scheduler = CogVideoXDPMScheduler.from_config(
        pipe.scheduler.config, timestep_spacing="trailing"
    )
    pipe.to("cuda")
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    items = []
    for row in rows:
        video_path = args.output_dir / "videos" / f"{row['sample_id']}.mp4"
        negative_prompt = row["target_concept"] if args.baseline == "negative_prompt" else None
        items.append(
            {
                "sample_id": row["sample_id"],
                "mechanism": row["mechanism"],
                "generalization_group": row["generalization_group"],
                "prompt": row["prompt"],
                "negative_prompt": negative_prompt,
                "seed": int(row["seed"]),
                "video_path": str(video_path),
            }
        )
        if args.skip_existing and video_path.exists():
            print(f"Skipping {row['sample_id']}", flush=True)
            continue
        video_path.parent.mkdir(parents=True, exist_ok=True)
        frames = pipe(
            prompt=row["prompt"],
            negative_prompt=negative_prompt,
            num_videos_per_prompt=1,
            num_inference_steps=args.steps,
            num_frames=args.num_frames,
            height=args.height,
            width=args.width,
            use_dynamic_cfg=True,
            guidance_scale=args.guidance_scale,
            generator=torch.Generator().manual_seed(int(row["seed"])),
        ).frames[0]
        export_to_video(frames, str(video_path), fps=args.fps)
        print(f"Finished {row['sample_id']}: {video_path}", flush=True)

    generation_manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline": args.baseline,
        "model": str(args.model),
        "mechanism": args.mechanism,
        "generation": {
            "num_inference_steps": args.steps,
            "guidance_scale": args.guidance_scale,
            "num_frames": args.num_frames,
            "height": args.height,
            "width": args.width,
            "fps": args.fps,
            "dtype": "bfloat16",
            "scheduler": "CogVideoXDPMScheduler trailing",
        },
        "items": items,
    }
    (args.output_dir / "generation_manifest.json").write_text(
        json.dumps(generation_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
