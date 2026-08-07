#!/usr/bin/env python3
"""Build aligned counterfactuals and a Wan training manifest for Protocol v1."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


FIELDS = [
    "protocol_version",
    "sample_id",
    "training_role",
    "mechanism",
    "split",
    "generalization_group",
    "source_id",
    "source_object",
    "source_seen",
    "receiver_id",
    "receiver",
    "receiver_seen",
    "prompt_variant",
    "prompt",
    "target_concept",
    "expected_footprint",
    "expected_counterfactual_state",
    "seed",
    "num_frames",
    "fps",
    "reference_start_inclusive",
    "reference_end_exclusive",
    "generated_video",
    "desired_target_video",
    "training_objective",
    "residual_mask_enabled",
    "residual_mask_factual_video",
    "residual_mask_target_video",
]


def one_video(root: Path, prompt_index: int) -> Path:
    matches = sorted((root / "clean_shards" / f"prompt_{prompt_index:03d}" / "videos").glob("*.mp4"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one factual video for prompt {prompt_index}, found {len(matches)}")
    return matches[0]


def load_video(path: Path) -> tuple[np.ndarray, float]:
    reader = imageio.get_reader(path)
    try:
        metadata = reader.get_meta_data()
        frames = np.stack([frame[:, :, :3] for frame in reader])
    finally:
        reader.close()
    return frames, float(metadata.get("fps", 8.0))


def write_static_video(path: Path, frame: np.ndarray, frame_count: int, fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=8, macro_block_size=None)
    try:
        for _ in range(frame_count):
            writer.append_data(frame)
    finally:
        writer.close()


def with_defaults(row: dict[str, str]) -> dict[str, str]:
    return {field: row.get(field, "") for field in FIELDS}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--erase-root", type=Path, required=True)
    parser.add_argument("--preserve-root", type=Path, required=True)
    parser.add_argument("--aligned-root", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    args = parser.parse_args()

    with (args.protocol_dir / "train_erase_manifest.csv").open(newline="", encoding="utf-8") as handle:
        erase_rows = list(csv.DictReader(handle))
    with (args.protocol_dir / "preserve_manifest.csv").open(newline="", encoding="utf-8") as handle:
        preserve_rows = list(csv.DictReader(handle))

    output_rows: list[dict[str, str]] = []
    for prompt_index, source_row in enumerate(erase_rows):
        factual = one_video(args.erase_root, prompt_index)
        frames, fps = load_video(factual)
        reference_end = int(source_row["reference_end_exclusive"])
        if len(frames) < reference_end:
            raise RuntimeError(f"{factual} has {len(frames)} frames, expected {reference_end}")
        reference = np.median(frames[:reference_end], axis=0).astype(np.uint8)
        sample_dir = args.aligned_root / source_row["sample_id"]
        target = sample_dir / "counterfactual_static.mp4"
        write_static_video(target, reference, len(frames), fps)

        row = with_defaults(source_row)
        row.update(
            {
                "generated_video": str(factual),
                "desired_target_video": str(target),
                "training_objective": "counterfactual_noise_prediction",
                "residual_mask_enabled": "yes",
                "residual_mask_factual_video": str(factual),
                "residual_mask_target_video": str(target),
                "fps": f"{fps:g}",
            }
        )
        output_rows.append(row)

    for prompt_index, source_row in enumerate(preserve_rows):
        factual = one_video(args.preserve_root, prompt_index)
        row = with_defaults(source_row)
        row.update(
            {
                "generated_video": str(factual),
                "desired_target_video": str(factual),
                "training_objective": "frozen_base_distillation",
                "residual_mask_enabled": "no",
                "residual_mask_factual_video": "",
                "residual_mask_target_video": "",
            }
        )
        output_rows.append(row)

    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"Wrote {len(output_rows)} rows to {args.output_manifest}")


if __name__ == "__main__":
    main()
