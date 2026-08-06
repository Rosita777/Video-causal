#!/usr/bin/env python3
"""Evaluate plain and dual-trajectory adapters on held-out waterdrop eval20."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import imageio.v3 as iio
import numpy as np


DEFAULT_METHOD_DIRS = {
    "plain": "outputs/waterdrop_plain_lora_100_eval20/videos",
    "dual_traj": "outputs/waterdrop_dual_traj_bg1_lora_100_eval20/videos",
    "dual_traj_scale075": "outputs/waterdrop_dual_traj_bg1_lora_100_scale075_eval20/videos",
}


def one_match(root: Path, pattern: str) -> Path:
    matches = sorted(root.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one match for {root / pattern}, found {len(matches)}")
    return matches[0]


def load_video(path: Path) -> np.ndarray:
    frames = iio.imread(path, plugin="pyav").astype(np.float32) / 255.0
    if frames.ndim != 4 or len(frames) < 40:
        raise RuntimeError(f"Unexpected video shape for {path}: {frames.shape}")
    return frames


def post_change_mae(frames: np.ndarray) -> float:
    return float(np.abs(frames[32:49].mean(axis=0) - frames[:8].mean(axis=0)).mean())


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--manifest", type=Path, default=Path("data/waterdrop_dual_traj_eval20.csv"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-dir", type=Path, help="Directory containing the frozen-base videos.")
    parser.add_argument("--adapter-dir", type=Path, help="Directory containing the new adapter videos.")
    parser.add_argument("--legacy-plain-dir", type=Path, help="Optional legacy plain-LoRA directory.")
    parser.add_argument(
        "--index",
        action="append",
        default=[],
        help="Only evaluate this eval index; may be repeated.",
    )
    args = parser.parse_args()

    method_dirs = dict(DEFAULT_METHOD_DIRS)
    if args.base_dir is not None:
        method_dirs["plain"] = args.base_dir
    if args.adapter_dir is not None:
        method_dirs["dual_traj_scale075"] = args.adapter_dir
        method_dirs["dual_traj"] = args.adapter_dir
    if args.legacy_plain_dir is not None:
        method_dirs["plain"] = args.legacy_plain_dir

    root = args.repo_root.resolve()
    manifest = args.manifest if args.manifest.is_absolute() else root / args.manifest
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    with manifest.open(newline="", encoding="utf-8") as handle:
        cases = list(csv.DictReader(handle))
    if args.index:
        selected = set(args.index)
        cases = [case for case in cases if case["eval_index"] in selected]

    raw_rows: list[dict[str, str]] = []
    for case in cases:
        base_path = root / case["video_path"]
        base = load_video(base_path)
        base_change = post_change_mae(base)
        for method, method_dir in method_dirs.items():
            video_path = one_match(root / method_dir, f"{case['eval_index']}_*.mp4")
            candidate = load_video(video_path)
            candidate_change = post_change_mae(candidate)
            suppression = 100.0 * (1.0 - candidate_change / base_change) if base_change else 0.0
            raw_rows.append(
                {
                    "eval_index": case["eval_index"],
                    "scene_id": case["scene_id"],
                    "condition": case["condition"],
                    "family": case["family"],
                    "receiver": case["receiver"],
                    "method": method,
                    "early_base_mae": f"{np.abs(candidate[:17] - base[:17]).mean():.8f}",
                    "base_post_change_mae": f"{base_change:.8f}",
                    "candidate_post_change_mae": f"{candidate_change:.8f}",
                    "post_change_suppression_percent": f"{suppression:.2f}",
                    "video_path": str(video_path.relative_to(root)),
                }
            )

    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in raw_rows:
        grouped[(row["method"], row["condition"], row["family"])].append(row)
    summary_rows: list[dict[str, str]] = []
    for (method, condition, family), rows in sorted(grouped.items()):
        summary_rows.append(
            {
                "method": method,
                "condition": condition,
                "family": family,
                "count": str(len(rows)),
                "mean_early_base_mae": f"{np.mean([float(r['early_base_mae']) for r in rows]):.8f}",
                "mean_post_change_suppression_percent": f"{np.mean([float(r['post_change_suppression_percent']) for r in rows]):.2f}",
            }
        )
    for method in method_dirs:
        rows = [row for row in raw_rows if row["method"] == method]
        summary_rows.append(
            {
                "method": method,
                "condition": "ALL",
                "family": "ALL",
                "count": str(len(rows)),
                "mean_early_base_mae": f"{np.mean([float(r['early_base_mae']) for r in rows]):.8f}",
                "mean_post_change_suppression_percent": f"{np.mean([float(r['post_change_suppression_percent']) for r in rows]):.2f}",
            }
        )

    by_case_method = {(row["eval_index"], row["method"]): row for row in raw_rows}
    improvement_rows: list[dict[str, str]] = []
    for case in cases:
        plain = by_case_method[(case["eval_index"], "plain")]
        dual = by_case_method[(case["eval_index"], "dual_traj")]
        improvement_rows.append(
            {
                "eval_index": case["eval_index"],
                "scene_id": case["scene_id"],
                "condition": case["condition"],
                "family": case["family"],
                "receiver": case["receiver"],
                "plain_suppression_percent": plain["post_change_suppression_percent"],
                "dual_suppression_percent": dual["post_change_suppression_percent"],
                "dual_improvement_points": f"{float(dual['post_change_suppression_percent']) - float(plain['post_change_suppression_percent']):.2f}",
                "plain_early_base_mae": plain["early_base_mae"],
                "dual_early_base_mae": dual["early_base_mae"],
            }
        )

    write_csv(output_dir / "raw_metrics.csv", raw_rows)
    write_csv(output_dir / "summary.csv", summary_rows)
    write_csv(output_dir / "dual_vs_plain.csv", improvement_rows)


if __name__ == "__main__":
    main()
