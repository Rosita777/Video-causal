#!/usr/bin/env python3
"""Plan or run the required baseline suite for a CogVideoX case set."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


REQUIRED_BASELINES = [
    "negative_prompt",
    "safree_cogvideox",
    "videoeraser",
    "t2vunlearning",
]


BLOCKED_ADAPTERS = {
    "safree_cogvideox": {
        "status": "blocked_missing_adapter",
        "missing": [
            "scripts/adapters/run_safree_cogvideox.py",
            "SAFREE projection/attention-intervention specification for CogVideoXPipeline",
        ],
    },
    "videoeraser": {
        "status": "blocked_missing_adapter",
        "missing": [
            "baselines/external/VideoEraser",
            "scripts/adapters/run_videoeraser_cogvideox.py",
        ],
    },
    "t2vunlearning": {
        "status": "blocked_missing_adapter",
        "missing": [
            "baselines/external/T2VUnlearning",
            "scripts/adapters/run_t2vunlearning_cogvideox.py",
            "T2VUnlearning training/adaptation config for the target concept",
        ],
    },
}


def build_negative_prompt_command(args: argparse.Namespace, output_dir: Path) -> list[str]:
    command = [
        sys.executable,
        "scripts/generate_cogvideox_clean.py",
        "--baseline",
        "negative_prompt",
        "--prompts",
        str(args.prompts),
        "--output-dir",
        str(output_dir),
        "--model",
        args.model,
        "--seed",
        str(args.seed),
        "--steps",
        str(args.steps),
        "--guidance-scale",
        str(args.guidance_scale),
        "--num-frames",
        str(args.num_frames),
        "--fps",
        str(args.fps),
    ]
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    if args.enable_model_cpu_offload:
        command.append("--enable-model-cpu-offload")
    if args.enable_sequential_cpu_offload:
        command.append("--enable-sequential-cpu-offload")
    if args.vae_slicing:
        command.append("--vae-slicing")
    if args.vae_tiling:
        command.append("--vae-tiling")
    return command


def build_jobs(args: argparse.Namespace) -> list[dict[str, object]]:
    selected = args.baseline or REQUIRED_BASELINES
    jobs: list[dict[str, object]] = []
    for baseline in selected:
        output_dir = args.output_root / baseline
        if baseline == "negative_prompt":
            jobs.append(
                {
                    "baseline": baseline,
                    "status": "ready",
                    "output_dir": str(output_dir),
                    "command": build_negative_prompt_command(args, output_dir),
                }
            )
            continue
        blocked = BLOCKED_ADAPTERS[baseline]
        jobs.append(
            {
                "baseline": baseline,
                "status": blocked["status"],
                "output_dir": str(output_dir),
                "missing": blocked["missing"],
            }
        )
    return jobs


def write_suite_manifest(args: argparse.Namespace, jobs: list[dict[str, object]]) -> Path:
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dry_run": args.dry_run,
        "prompts": str(args.prompts),
        "model": args.model,
        "generation": {
            "seed": args.seed,
            "num_inference_steps": args.steps,
            "guidance_scale": args.guidance_scale,
            "num_frames": args.num_frames,
            "fps": args.fps,
            "limit": args.limit,
        },
        "jobs": jobs,
    }
    out = args.output_root / "suite_manifest.json"
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


def run_ready_jobs(jobs: list[dict[str, object]]) -> None:
    for job in jobs:
        if job["status"] != "ready":
            print(f"Blocked: {job['baseline']} ({job['status']})")
            continue
        print(f"Running: {job['baseline']}")
        subprocess.run(job["command"], check=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", action="append", choices=REQUIRED_BASELINES)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model", default="models/CogVideoX-2b")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=6.0)
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--enable-model-cpu-offload", action="store_true")
    parser.add_argument("--enable-sequential-cpu-offload", action="store_true")
    parser.add_argument("--vae-slicing", action="store_true")
    parser.add_argument("--vae-tiling", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.steps <= 0:
        parser.error("--steps must be positive")
    if args.num_frames <= 0:
        parser.error("--num-frames must be positive")
    if args.fps <= 0:
        parser.error("--fps must be positive")

    jobs = build_jobs(args)
    manifest = write_suite_manifest(args, jobs)
    print(f"Baseline suite manifest written: {manifest}")
    if not args.dry_run:
        run_ready_jobs(jobs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
