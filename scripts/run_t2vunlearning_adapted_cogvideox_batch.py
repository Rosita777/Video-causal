#!/usr/bin/env python3
"""Generate Protocol v1 videos with a T2VUnlearning-adapted CogVideoX adapter."""

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
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=6.0)
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    import torch
    import torch.nn as nn
    from diffusers import CogVideoXDPMScheduler, CogVideoXPipeline
    from diffusers.models.attention import Attention
    from diffusers.utils import export_to_video

    with args.manifest.open(newline="", encoding="utf-8") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row["mechanism"] == args.mechanism
        ]
    if len(rows) != 20:
        raise ValueError(f"Expected 20 rows for {args.mechanism}, found {len(rows)}")
    if args.limit is not None:
        rows = rows[: args.limit]

    config_path = args.checkpoint / "eraser_config.json"
    weights_path = args.checkpoint / "eraser_weights.pt"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    rank = int(config["eraser_rank"])

    class AdapterEraser(nn.Module):
        def __init__(self, dim: int, adapter_rank: int):
            super().__init__()
            self.down = nn.Linear(dim, adapter_rank)
            self.act = nn.GELU()
            self.up = nn.Linear(adapter_rank, dim)

        def forward(self, hidden_states):
            dtype = hidden_states.dtype
            return self.up(self.act(self.down(hidden_states.float()))).to(dtype)

    class CogVideoXWithEraser(nn.Module):
        def __init__(self, attention: Attention, adapter_rank: int):
            super().__init__()
            self.attn = attention
            self.adapter = AdapterEraser(attention.to_v.weight.shape[-1], adapter_rank)

        def forward(
            self, hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs
        ):
            hidden_states, encoder_hidden_states = self.attn(
                hidden_states, encoder_hidden_states, attention_mask, **kwargs
            )
            hidden_states = hidden_states + self.adapter(hidden_states)
            return hidden_states, encoder_hidden_states

    pipe = CogVideoXPipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    pipe.scheduler = CogVideoXDPMScheduler.from_config(
        pipe.scheduler.config, timestep_spacing="trailing"
    )
    wrappers = []
    for block in pipe.transformer.transformer_blocks:
        block.attn1 = CogVideoXWithEraser(block.attn1, rank)
        wrappers.append(block.attn1)

    saved = torch.load(weights_path, map_location="cpu", weights_only=True)
    expected_keys = {
        f"transformer_blocks.{index}.attn1.adapter" for index in range(len(wrappers))
    }
    if set(saved) != expected_keys:
        raise ValueError("Adapter checkpoint keys do not match the CogVideoX blocks")
    for index, wrapper in enumerate(wrappers):
        wrapper.adapter.load_state_dict(
            saved[f"transformer_blocks.{index}.attn1.adapter"]
        )

    pipe.to("cuda")
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    items = []
    for row in rows:
        video_path = args.output_dir / "videos" / f"{row['sample_id']}.mp4"
        items.append(
            {
                "sample_id": row["sample_id"],
                "mechanism": row["mechanism"],
                "generalization_group": row["generalization_group"],
                "prompt": row["prompt"],
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
            num_videos_per_prompt=1,
            num_inference_steps=args.steps,
            num_frames=args.num_frames,
            use_dynamic_cfg=True,
            guidance_scale=args.guidance_scale,
            generator=torch.Generator().manual_seed(int(row["seed"])),
        ).frames[0]
        export_to_video(frames, str(video_path), fps=args.fps)
        print(f"Finished {row['sample_id']}: {video_path}", flush=True)

    generation_manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline": "t2vunlearning_adapted",
        "model": str(args.model),
        "mechanism": args.mechanism,
        "checkpoint": str(args.checkpoint),
        "generation": {
            "num_inference_steps": args.steps,
            "guidance_scale": args.guidance_scale,
            "num_frames": args.num_frames,
            "fps": args.fps,
            "dtype": "bfloat16",
            "scheduler": "CogVideoXDPMScheduler trailing",
            "adapter_rank": rank,
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
