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


def safree_pipeline_path(args: argparse.Namespace) -> Path:
    return args.safree_root / "cogvideox" / "cogvideox_pipeline.py"


def videoeraser_runner_path(args: argparse.Namespace) -> Path:
    return args.videoeraser_root / "ModelScope" / "inference.py"


def t2vunlearning_required_paths(args: argparse.Namespace) -> list[Path]:
    return [
        args.t2vunlearning_root / "test_cogvideo.py",
        args.t2vunlearning_root / "receler" / "concept_reg_cogvideo.py",
    ]


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
        "--dtype",
        args.dtype,
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


def build_safree_command(args: argparse.Namespace, output_dir: Path) -> list[str]:
    command = [
        sys.executable,
        "scripts/adapters/run_safree_cogvideox.py",
        "--prompts",
        str(args.prompts),
        "--output-dir",
        str(output_dir),
        "--model",
        args.model,
        "--safree-root",
        str(args.safree_root),
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
        "--dtype",
        args.dtype,
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


def build_videoeraser_command(args: argparse.Namespace, output_dir: Path) -> list[str]:
    command = [
        sys.executable,
        "scripts/adapters/run_videoeraser_cogvideox.py",
        "--prompts",
        str(args.prompts),
        "--output-dir",
        str(output_dir),
        "--model",
        args.model,
        "--videoeraser-root",
        str(args.videoeraser_root),
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
        "--dtype",
        args.dtype,
    ]
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    return command


def build_t2vunlearning_command(args: argparse.Namespace, output_dir: Path) -> list[str]:
    command = [
        sys.executable,
        "scripts/adapters/run_t2vunlearning_cogvideox.py",
        "--prompts",
        str(args.prompts),
        "--output-dir",
        str(output_dir),
        "--model",
        args.model,
        "--t2vunlearning-root",
        str(args.t2vunlearning_root),
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
        "--dtype",
        args.dtype,
    ]
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
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
        if baseline == "safree_cogvideox":
            pipeline_path = safree_pipeline_path(args)
            if not pipeline_path.is_file():
                jobs.append(
                    {
                        "baseline": baseline,
                        "status": "blocked_missing_external",
                        "output_dir": str(output_dir),
                        "missing": [str(pipeline_path)],
                    }
                )
                continue
            jobs.append(
                {
                    "baseline": baseline,
                    "status": "ready",
                    "output_dir": str(output_dir),
                    "command": build_safree_command(args, output_dir),
                }
            )
            continue
        if baseline == "videoeraser":
            runner_path = videoeraser_runner_path(args)
            if not runner_path.is_file():
                jobs.append(
                    {
                        "baseline": baseline,
                        "status": "blocked_missing_external",
                        "output_dir": str(output_dir),
                        "missing": [str(runner_path)],
                    }
                )
                continue
            jobs.append(
                {
                    "baseline": baseline,
                    "status": "ready",
                    "output_dir": str(output_dir),
                    "command": build_videoeraser_command(args, output_dir),
                }
            )
            continue
        if baseline == "t2vunlearning":
            missing = [path for path in t2vunlearning_required_paths(args) if not path.is_file()]
            if missing:
                jobs.append(
                    {
                        "baseline": baseline,
                        "status": "blocked_missing_external",
                        "output_dir": str(output_dir),
                        "missing": [str(path) for path in missing],
                    }
                )
                continue
            jobs.append(
                {
                    "baseline": baseline,
                    "status": "ready",
                    "output_dir": str(output_dir),
                    "command": build_t2vunlearning_command(args, output_dir),
                }
            )
            continue
        raise ValueError(f"Unsupported baseline: {baseline}")
    return jobs


def write_suite_manifest(args: argparse.Namespace, jobs: list[dict[str, object]]) -> Path:
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dry_run": args.dry_run,
        "parallel": args.parallel,
        "prompts": str(args.prompts),
        "model": args.model,
        "generation": {
            "seed": args.seed,
            "num_inference_steps": args.steps,
            "guidance_scale": args.guidance_scale,
            "num_frames": args.num_frames,
            "fps": args.fps,
            "dtype": args.dtype,
            "limit": args.limit,
        },
        "external": {
            "safree_root": str(args.safree_root),
            "safree_pipeline": str(safree_pipeline_path(args)),
            "videoeraser_root": str(args.videoeraser_root),
            "videoeraser_runner": str(videoeraser_runner_path(args)),
            "t2vunlearning_root": str(args.t2vunlearning_root),
            "t2vunlearning_required": [str(path) for path in t2vunlearning_required_paths(args)],
        },
        "jobs": jobs,
    }
    out = args.output_root / "suite_manifest.json"
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


def run_ready_jobs(jobs: list[dict[str, object]], *, parallel: bool) -> None:
    ready_jobs = []
    for job in jobs:
        if job["status"] != "ready":
            print(f"Blocked: {job['baseline']} ({job['status']})")
            continue
        ready_jobs.append(job)

    if parallel:
        processes = []
        for job in ready_jobs:
            print(f"Starting: {job['baseline']}")
            processes.append((job, subprocess.Popen(job["command"])))
        failures = []
        for job, process in processes:
            return_code = process.wait()
            if return_code != 0:
                failures.append((job["baseline"], return_code))
        if failures:
            details = ", ".join(f"{name}={code}" for name, code in failures)
            raise SystemExit(f"Baseline job failures: {details}")
        return

    for job in ready_jobs:
        print(f"Running: {job['baseline']}")
        subprocess.run(job["command"], check=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", action="append", choices=REQUIRED_BASELINES)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model", default="models/CogVideoX-2b")
    parser.add_argument("--safree-root", type=Path, default=Path("baselines/external/SAFREE"))
    parser.add_argument("--videoeraser-root", type=Path, default=Path("baselines/external/VideoEraser"))
    parser.add_argument("--t2vunlearning-root", type=Path, default=Path("baselines/external/T2VUnlearning"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=6.0)
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--enable-model-cpu-offload", action="store_true")
    parser.add_argument("--enable-sequential-cpu-offload", action="store_true")
    parser.add_argument("--vae-slicing", action="store_true")
    parser.add_argument("--vae-tiling", action="store_true")
    parser.add_argument("--parallel", action="store_true")
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
        run_ready_jobs(jobs, parallel=args.parallel)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
