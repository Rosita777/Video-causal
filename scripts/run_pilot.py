#!/usr/bin/env python3
"""Lightweight pilot manifest driver for causal-footprint experiments."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


ALLOWED_BASELINES = {
    "clean",
    "negative_prompt",
    "videoeraser",
    "t2vunlearning",
    "safree_cogvideox",
}


def parse_prompt_file(path: Path) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = [part.strip() for part in line.split("|")]
        if len(parts) != 3 or not all(parts):
            raise ValueError(f"{path}:{line_no}: expected '<prompt> | <target> | <effect>'")
        prompt, target_concept, expected_effect = parts
        items.append(
            {
                "prompt": prompt,
                "target_concept": target_concept,
                "expected_effect": expected_effect,
            }
        )
    return items


def write_dry_run_manifest(baseline: str, prompts: Path, output_dir: Path) -> Path:
    items = parse_prompt_file(prompts)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline": baseline,
        "dry_run": True,
        "prompts": str(prompts),
        "items": items,
    }
    out = output_dir / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, choices=sorted(ALLOWED_BASELINES))
    parser.add_argument("--prompts", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.dry_run:
        raise SystemExit("run_pilot.py only supports --dry-run; use baseline-specific scripts for generation")

    manifest = write_dry_run_manifest(args.baseline, args.prompts, args.output_dir)
    print(f"Dry-run manifest written: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
