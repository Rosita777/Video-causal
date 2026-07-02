#!/usr/bin/env python3
"""Run or dry-run Phase A alpha/window sweeps for the MVP-0 ZeroScope probe."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNNER = PROJECT_ROOT / "scripts" / "adapters" / "run_mvp0_zeroscope_probe.py"
DEFAULT_CONDITIONS = [
    "target_negative",
    "target_footprint_negative",
    "full_chain_steering",
    "random_direction",
    "orthogonal_semantic",
]


class CompletedCell:
    def __init__(self, returncode: int, stdout: str, stderr: str) -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def parse_alpha_grid(value: str) -> list[float]:
    try:
        alphas = [float(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("alpha values must be numbers") from exc
    if not alphas:
        raise argparse.ArgumentTypeError("alpha grid must not be empty")
    if any(alpha <= 0 for alpha in alphas):
        raise argparse.ArgumentTypeError("alpha values must be positive")
    return alphas


def parse_window(value: str) -> tuple[int, int]:
    left, sep, right = value.partition(":")
    if not sep:
        raise argparse.ArgumentTypeError("window must be START:END")
    try:
        start = int(left)
        end = int(right)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("window bounds must be integers") from exc
    if start < 0 or end < start:
        raise argparse.ArgumentTypeError("expected 0 <= START <= END")
    return start, end


def parse_window_grid(value: str) -> list[tuple[int, int]]:
    windows = [parse_window(part.strip()) for part in value.split(",") if part.strip()]
    if not windows:
        raise argparse.ArgumentTypeError("window grid must not be empty")
    return windows


def alpha_slug(alpha: float) -> str:
    return f"{alpha:.2f}".rstrip("0").rstrip(".").replace(".", "p")


def build_runner_argv(
    args: argparse.Namespace,
    alpha: float,
    window: tuple[int, int],
    cell_dir: Path,
    conditions: list[str] | None = None,
) -> list[str]:
    argv = [
        sys.executable,
        str(RUNNER),
        "--probe-manifest",
        str(args.probe_manifest),
        "--output-dir",
        str(cell_dir),
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
        "--height",
        str(args.height),
        "--width",
        str(args.width),
        "--dtype",
        args.dtype,
        "--device",
        args.device,
        "--limit-items",
        str(args.limit_items),
        "--alpha",
        str(alpha),
        "--timestep-window",
        f"{window[0]}:{window[1]}",
        "--strict-prompt-length",
    ]
    active_conditions = conditions if conditions is not None else list(args.condition)
    for condition in active_conditions:
        argv.extend(["--condition", condition])
    if args.enable_model_cpu_offload:
        argv.append("--enable-model-cpu-offload")
    if args.enable_sequential_cpu_offload:
        argv.append("--enable-sequential-cpu-offload")
    if args.vae_slicing:
        argv.append("--vae-slicing")
    if args.dry_run:
        argv.append("--dry-run")
    return argv


def build_sweep_cells(args: argparse.Namespace) -> list[dict[str, object]]:
    conditions = list(args.condition) if args.condition else list(DEFAULT_CONDITIONS)
    cells: list[dict[str, object]] = []
    for alpha in args.alpha_grid:
        for window in args.timestep_window_grid:
            cell_id = f"alpha_{alpha_slug(alpha)}_window_{window[0]}_{window[1]}"
            cell_dir = args.output_dir / cell_id
            cells.append(
                {
                    "cell_id": cell_id,
                    "alpha": alpha,
                    "timestep_window": [window[0], window[1]],
                    "conditions": conditions,
                    "output_dir": str(cell_dir),
                    "runner_argv": build_runner_argv(args, alpha, window, cell_dir, conditions),
                    "status": "planned",
                }
            )
    return cells


def run_subprocess(cmd: list[str], cwd: Path, text: bool, capture_output: bool) -> CompletedCell:
    result = subprocess.run(cmd, cwd=cwd, text=text, capture_output=capture_output)
    return CompletedCell(result.returncode, result.stdout, result.stderr)


def write_sweep_manifest(
    args: argparse.Namespace,
    cells: list[dict[str, object]],
    executed_cells: int,
) -> Path:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "phase_a_conservative_parameter_sweep",
        "probe_manifest": str(args.probe_manifest),
        "model": args.model,
        "dry_run": args.dry_run,
        "total_cells": len(cells),
        "executed_cells": executed_cells,
        "cells": cells,
    }
    path = args.output_dir / "sweep_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--probe-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="models/zeroscope_v2_576w")
    parser.add_argument("--seed", type=int, default=15000)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--guidance-scale", type=float, default=9.0)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--width", type=int, default=432)
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit-items", type=int, default=3)
    parser.add_argument("--condition", action="append", choices=DEFAULT_CONDITIONS, default=[])
    parser.add_argument(
        "--alpha-grid",
        type=parse_alpha_grid,
        default=parse_alpha_grid("0.15,0.25,0.35"),
    )
    parser.add_argument(
        "--timestep-window-grid",
        type=parse_window_grid,
        default=parse_window_grid("2:5,3:6,4:7"),
    )
    parser.add_argument("--max-cells", type=int)
    parser.add_argument("--enable-model-cpu-offload", action="store_true")
    parser.add_argument("--enable-sequential-cpu-offload", action="store_true")
    parser.add_argument("--vae-slicing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.condition:
        args.condition = list(DEFAULT_CONDITIONS)
    if args.limit_items <= 0:
        parser.error("--limit-items must be positive")
    if args.max_cells is not None and args.max_cells < 0:
        parser.error("--max-cells must be non-negative")
    cells = build_sweep_cells(args)
    to_execute = cells if args.max_cells is None else cells[: args.max_cells]
    failures = 0
    for cell in to_execute:
        result = run_subprocess(
            list(cell["runner_argv"]),
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
        )
        cell["returncode"] = result.returncode
        cell["stdout"] = result.stdout
        cell["stderr"] = result.stderr
        cell["status"] = "completed" if result.returncode == 0 else "failed"
        if result.returncode != 0:
            failures += 1
    manifest = write_sweep_manifest(args, cells, len(to_execute))
    print(f"Wrote Phase A sweep manifest: {manifest}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
