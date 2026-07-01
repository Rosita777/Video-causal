#!/usr/bin/env python3
"""Merge clean generation shard manifests from run_parallel_baseline_jobs.py."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def clean_jobs(parallel_manifest: dict[str, Any]) -> list[dict[str, Any]]:
    jobs = [
        job
        for job in parallel_manifest.get("jobs", [])
        if job.get("baseline") == "clean" and job.get("status") in {"finished", "finished_after_retry"}
    ]
    return sorted(jobs, key=lambda job: int(job.get("source_prompt_index", 0)))


def load_shard_item(job: dict[str, Any]) -> dict[str, Any]:
    manifest_path = Path(str(job["output_dir"])) / "generation_manifest.json"
    if not manifest_path.exists():
        raise ValueError(f"missing shard manifest: {manifest_path}")
    manifest = load_json(manifest_path)
    items = manifest.get("items")
    if not isinstance(items, list) or len(items) != 1:
        raise ValueError(f"{manifest_path}: expected exactly one generated item")
    item = dict(items[0])
    item["source_prompt_index"] = int(job["source_prompt_index"])
    item["shard_manifest"] = str(manifest_path)
    return item


def build_merged_manifest(parallel_manifest: dict[str, Any], items: list[dict[str, Any]]) -> dict[str, Any]:
    normalized_items = []
    for index, item in enumerate(items):
        merged_item = dict(item)
        merged_item["index"] = index
        normalized_items.append(merged_item)
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline": "clean",
        "model": parallel_manifest.get("model", ""),
        "dry_run": parallel_manifest.get("dry_run", False),
        "prompts": parallel_manifest.get("prompts", ""),
        "generation": parallel_manifest.get("generation", {}),
        "items": normalized_items,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parallel-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        parallel_manifest = load_json(args.parallel_manifest)
        jobs = clean_jobs(parallel_manifest)
        items = [load_shard_item(job) for job in jobs]
    except (KeyError, ValueError, json.JSONDecodeError) as exc:
        parser.exit(2, f"{exc}\n")
    merged = build_merged_manifest(parallel_manifest, items)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(merged, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Merged {len(items)} clean shard manifests into {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
