#!/usr/bin/env python3
"""Run the official VideoEraser CogVideoX pipeline on Protocol v1."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--mechanism", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=6.0)
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    cog_root = (args.official_root / "CogVideoX").resolve()
    sys.path.insert(0, str(cog_root))
    import torch
    from diffusers import CogVideoXDPMScheduler
    from diffusers.utils import export_to_video
    from cogvideox_pipeline import CogVideoXPipeline

    with args.manifest.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["mechanism"] == args.mechanism]
    if len(rows) != 20:
        raise ValueError(f"Expected 20 rows for {args.mechanism}, found {len(rows)}")

    pipe = CogVideoXPipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    pipe.to("cuda")
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    items = []
    for row in rows:
        video_path = args.output_dir / "videos" / f"{row['sample_id']}.mp4"
        item = {
            "sample_id": row["sample_id"],
            "mechanism": row["mechanism"],
            "generalization_group": row["generalization_group"],
            "prompt": row["prompt"],
            "target_concept": row["target_concept"],
            "seed": int(row["seed"]),
            "video_path": str(video_path),
        }
        items.append(item)
        if args.skip_existing and video_path.exists():
            print(f"Skipping {row['sample_id']}", flush=True)
            continue
        video_path.parent.mkdir(parents=True, exist_ok=True)
        result = pipe(
            prompt=row["prompt"],
            num_videos_per_prompt=1,
            num_inference_steps=args.steps,
            num_frames=args.num_frames,
            use_dynamic_cfg=True,
            guidance_scale=args.guidance_scale,
            generator=torch.Generator().manual_seed(int(row["seed"])),
            concept=row["target_concept"],
        ).frames[0]
        export_to_video(result, str(video_path), fps=args.fps)
        print(f"Finished {row['sample_id']}: {video_path}", flush=True)

    commit_file = args.official_root / "OFFICIAL_COMMIT"
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline": "videoeraser_official",
        "official_repository": "https://github.com/bluedream02/VideoEraser",
        "official_commit": commit_file.read_text().strip() if commit_file.exists() else "unknown",
        "model": str(args.model),
        "mechanism": args.mechanism,
        "generation": {
            "num_inference_steps": args.steps,
            "guidance_scale": args.guidance_scale,
            "num_frames": args.num_frames,
            "fps": args.fps,
            "dtype": "bfloat16",
            "scheduler": "CogVideoXDPMScheduler trailing",
            "official_pipeline_unmodified": True,
        },
        "items": items,
    }
    (args.output_dir / "generation_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
