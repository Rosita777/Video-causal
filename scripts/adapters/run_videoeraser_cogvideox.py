#!/usr/bin/env python3
"""Plan or run the VideoEraser-CogVideoX adapter for project prompt files."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from generate_cogvideox_clean import slugify  # noqa: E402
from run_pilot import parse_prompt_file  # noqa: E402


BASELINE = "videoeraser"
DEFAULT_ROOT = Path("baselines/external/VideoEraser")


def resolve_path(path: Path) -> Path:
    return path.expanduser().resolve()


def resolve_model_arg(model: str) -> str:
    model_path = Path(model).expanduser()
    if model_path.exists():
        return str(model_path.resolve())
    return model


REQUIRED_RELATIVE = [Path("ModelScope") / "inference.py"]


def required_paths(root: Path) -> list[Path]:
    resolved_root = resolve_path(root)
    return [resolved_root / rel for rel in REQUIRED_RELATIVE]


def build_generation_config(args: argparse.Namespace) -> dict[str, object]:
    return {
        "seed": args.seed,
        "num_inference_steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "num_frames": args.num_frames,
        "fps": args.fps,
        "dtype": args.dtype,
        "limit": args.limit,
    }


def build_external_config(args: argparse.Namespace) -> dict[str, object]:
    paths = required_paths(args.external_root)
    return {
        "root": str(resolve_path(args.external_root)),
        "required_files": [str(path) for path in paths],
        "required_files_present": all(path.is_file() for path in paths),
        "adapter_mode": "external_runner_wrapper",
        "notes": "Adapter expects the external VideoEraser ModelScope runner. It records project prompt metadata and delegates real generation to the external runner when available.",
    }


def build_manifest_items(prompts: list[dict[str, str]], output_dir: Path, base_seed: int, limit: int | None) -> list[dict[str, object]]:
    selected = prompts[:limit] if limit is not None else prompts
    items = []
    video_dir = output_dir / "videos"
    for index, item in enumerate(selected):
        seed = base_seed + index
        prompt_slug = slugify(item["prompt"])
        video_path = video_dir / f"{index:03d}_{prompt_slug}_seed{seed}.mp4"
        items.append(
            {
                "index": index,
                "prompt": item["prompt"],
                "target_concept": item["target_concept"],
                "expected_effect": item["expected_effect"],
                "seed": seed,
                "video_path": str(video_path),
            }
        )
    return items


def write_manifest(*, output_dir: Path, model: str, prompts_path: Path, generation: dict[str, object], external: dict[str, object], items: list[dict[str, object]], dry_run: bool) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline": BASELINE,
        "model": model,
        "dry_run": dry_run,
        "prompts": str(prompts_path),
        "generation": generation,
        "external": external,
        "items": items,
    }
    out = output_dir / "generation_manifest.json"
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


def run_external(args: argparse.Namespace, items: list[dict[str, object]]) -> None:
    external_root = resolve_path(args.external_root)
    missing = [str(path) for path in required_paths(external_root) if not path.is_file()]
    if missing:
        raise SystemExit("Missing VideoEraser-CogVideoX external files: " + ", ".join(missing))
    runner = external_root / "ModelScope" / "inference.py"
    command = [
        sys.executable,
        str(runner),
        "--prompts",
        str(resolve_path(args.prompts)),
        "--output-dir",
        str(resolve_path(args.output_dir)),
        "--model",
        resolve_model_arg(args.model),
        "--seed",
        str(args.seed),
        "--steps",
        str(args.steps),
        "--guidance-scale",
        str(args.guidance_scale),
    ]
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    subprocess.run(command, check=True, cwd=external_root)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="models/CogVideoX-2b")
    parser.add_argument("--external-root", "--videoeraser-root", dest="external_root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=6.0)
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--limit", type=int)
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

    prompts = parse_prompt_file(args.prompts)
    generation = build_generation_config(args)
    external = build_external_config(args)
    items = build_manifest_items(prompts, args.output_dir, args.seed, args.limit)

    if not args.dry_run:
        run_external(args, items)

    manifest = write_manifest(
        output_dir=args.output_dir,
        model=args.model,
        prompts_path=args.prompts,
        generation=generation,
        external=external,
        items=items,
        dry_run=args.dry_run,
    )
    if args.dry_run:
        print(f"Dry-run {BASELINE} manifest written: {manifest}")
    else:
        print(f"{BASELINE} generation manifest written: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
