#!/usr/bin/env python3
"""Compare waterdrop mask-weight adapters on the fixed five-video probe."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import imageio.v3 as iio
import numpy as np


CASES = [
    ("trained_explicit", "train", "explicit_causal", "outputs/waterdrop_train_pilot40_wan/shard_0/videos/000_*seed9600.mp4", 0),
    ("heldout_hard_surface", "test", "explicit_causal", "outputs/waterdrop_five_condition_test100_wan/shard_0/videos/015_*seed9100.mp4", 1),
    ("heldout_liquid_surface", "test", "explicit_causal", "outputs/waterdrop_five_condition_test100_wan/shard_0/videos/000_*seed9100.mp4", 2),
    ("unrelated_footprint", "control", "unrelated_footprint", "outputs/waterdrop_train_pilot40_wan/shard_2/videos/000_*seed9600.mp4", 3),
    ("clean_control", "control", "clean_control", "outputs/waterdrop_five_condition_test100_wan/shard_0/videos/016_*seed9100.mp4", 4),
]

METHODS = {
    "plain": "outputs/waterdrop_plain_lora_100_quick_eval5/videos",
    "mask_w0_bg1": "outputs/waterdrop_mask_w0_bg1_lora_100_quick_eval5/videos",
    "mask_w1_bg1": "outputs/waterdrop_mask_w1_bg1_lora_100_quick_eval5/videos",
    "mask_w2_bg1": "outputs/waterdrop_mask_w2_bg1_lora_100_quick_eval5/videos",
    "mask_w4_bg1": "outputs/waterdrop_mask_bg_lora_100_quick_eval5/videos",
    "paired_sep_bg1": "outputs/waterdrop_paired_sep_bg1_lora_100_quick_eval5/videos",
    "paired_sep_w4_bg1": "outputs/waterdrop_paired_sep_w4_bg1_lora_100_quick_eval5/videos",
    "dual_traj_bg1": "outputs/waterdrop_dual_traj_bg1_lora_100_quick_eval5/videos",
    "dual_traj_rw0025": "outputs/waterdrop_dual_traj_rw0025_lora_100_quick_eval5/videos",
    "dual_traj_rw0100": "outputs/waterdrop_dual_traj_rw0100_lora_100_quick_eval5/videos",
    "dual_traj_scale050": "outputs/waterdrop_dual_traj_bg1_lora_100_scale050_quick_eval5/videos",
    "dual_traj_scale075": "outputs/waterdrop_dual_traj_bg1_lora_100_scale075_quick_eval5/videos",
}


def one_match(root: Path, pattern: str) -> Path:
    matches = sorted(root.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one match for {pattern}, found {len(matches)}")
    return matches[0]


def load_video(path: Path) -> np.ndarray:
    frames = iio.imread(path, plugin="pyav").astype(np.float32) / 255.0
    if frames.ndim != 4 or len(frames) < 40:
        raise RuntimeError(f"Unexpected video shape for {path}: {frames.shape}")
    return frames


def early_base_mae(candidate: np.ndarray, base: np.ndarray) -> float:
    count = min(17, len(candidate), len(base))
    return float(np.abs(candidate[:count] - base[:count]).mean())


def post_change_mae(frames: np.ndarray) -> float:
    early = frames[:8].mean(axis=0)
    late = frames[32:49].mean(axis=0)
    return float(np.abs(late - early).mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.repo_root.resolve()
    rows: list[dict[str, str]] = []
    for case_id, split, condition, base_pattern, index in CASES:
        base_path = one_match(root, base_pattern)
        base = load_video(base_path)
        base_change = post_change_mae(base)
        for method, method_dir in METHODS.items():
            video_path = one_match(root / method_dir, f"{index:03d}_*.mp4")
            candidate = load_video(video_path)
            change = post_change_mae(candidate)
            suppression = 100.0 * (1.0 - change / base_change) if base_change else 0.0
            rows.append(
                {
                    "case_id": case_id,
                    "split": split,
                    "condition": condition,
                    "method": method,
                    "early_base_mae": f"{early_base_mae(candidate, base):.8f}",
                    "base_post_change_mae": f"{base_change:.8f}",
                    "candidate_post_change_mae": f"{change:.8f}",
                    "post_change_suppression_percent": f"{suppression:.2f}",
                    "video_path": str(video_path.relative_to(root)),
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
