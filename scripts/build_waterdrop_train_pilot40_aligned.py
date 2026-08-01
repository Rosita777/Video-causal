#!/usr/bin/env python3
"""Build preliminary aligned targets and SFT records for train pilot40."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from build_static_counterfactual_pair import (
    build_reference_frame,
    load_video,
    make_contact_sheet,
    write_static_video,
)


CONDITIONS = {
    "explicit_causal",
    "target_only",
    "unrelated_footprint",
    "clean_control",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--auto-screen",
        type=Path,
        default=Path("data/waterdrop_train_pilot40_auto_screen.csv"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/waterdrop_train_pilot40_aligned"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("data/waterdrop_train_pilot40_sft_preliminary.csv"),
    )
    args = parser.parse_args()
    root = args.repo_root.resolve()
    auto_path = args.auto_screen if args.auto_screen.is_absolute() else root / args.auto_screen
    output_root = args.output_root if args.output_root.is_absolute() else root / args.output_root
    output_csv = args.output_csv if args.output_csv.is_absolute() else root / args.output_csv

    prompt_rows = read_csv(root / "data/waterdrop_train_pilot40.csv")
    auto_rows = read_csv(auto_path)
    prompt_by_id = {row["scene_id"]: row for row in prompt_rows}
    auto_by_id = {row["scene_id"]: row for row in auto_rows}
    if set(prompt_by_id) != set(auto_by_id):
        raise ValueError("pilot40 prompt and auto-screen scene IDs do not match")

    groups: dict[str, dict[str, dict[str, str]]] = {}
    for row in prompt_rows:
        groups.setdefault(row["train_group_id"], {})[row["condition"]] = row

    records = []
    skipped = []
    for group_id, group in sorted(groups.items()):
        if set(group) != CONDITIONS:
            raise ValueError(f"{group_id}: incomplete conditions: {sorted(group)}")
        explicit = group["explicit_causal"]
        explicit_media = auto_by_id[explicit["scene_id"]]
        reference_end_text = explicit_media["estimated_clean_end_exclusive"]
        if not reference_end_text or int(reference_end_text) < 2:
            skipped.append(
                {
                    "train_group_id": group_id,
                    "reason": "explicit causal video has fewer than two detected clean-prefix frames",
                }
            )
            continue

        reference_end = int(reference_end_text)
        factual_path = root / explicit_media["video_path"]
        frames, fps = load_video(factual_path)
        reference = build_reference_frame(frames, 0, reference_end)
        pair_dir = output_root / group_id
        target_path = pair_dir / "counterfactual_static.mp4"
        reference_path = pair_dir / "clean_reference.png"
        contact_path = pair_dir / "factual_counterfactual_contact_sheet.jpg"
        write_static_video(target_path, reference, len(frames), fps)
        pair_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(reference).save(reference_path)
        make_contact_sheet(frames, reference, contact_path)
        first_frame_mae = float(
            abs(frames[0].astype("int16") - reference.astype("int16")).mean()
        )
        pair_manifest = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "train_group_id": group_id,
            "method": "median_of_detected_clean_prefix_repeated_as_static_video",
            "factual_video": str(factual_path.relative_to(root)),
            "target_video": str(target_path.relative_to(root)),
            "clean_reference": str(reference_path.relative_to(root)),
            "contact_sheet": str(contact_path.relative_to(root)),
            "reference_end_exclusive": reference_end,
            "first_frame_reference_mae_255": first_frame_mae,
        }
        (pair_dir / "pair_manifest.json").write_text(
            json.dumps(pair_manifest, indent=2) + "\n", encoding="utf-8"
        )

        for condition in sorted(CONDITIONS):
            row = group[condition]
            media = auto_by_id[row["scene_id"]]
            is_erase = condition in {"explicit_causal", "target_only"}
            is_explicit = condition == "explicit_causal"
            desired_video = target_path.relative_to(root) if is_erase else Path(media["video_path"])
            records.append(
                {
                    "scene_id": row["scene_id"],
                    "train_group_id": group_id,
                    "receiver_id": row["receiver_id"],
                    "receiver": row["receiver"],
                    "condition": condition,
                    "prompt": row["prompt"],
                    "generated_video": media["video_path"],
                    "desired_target_video": str(desired_video),
                    "training_role": "erase" if is_erase else "preserve",
                    "training_objective": (
                        "counterfactual_noise_prediction"
                        if is_erase
                        else "frozen_base_distillation"
                    ),
                    "residual_mask_enabled": "yes" if is_explicit else "no",
                    "residual_mask_factual_video": media["video_path"] if is_explicit else "",
                    "residual_mask_target_video": str(target_path.relative_to(root)) if is_explicit else "",
                    "reference_end_exclusive": str(reference_end),
                    "pair_contact_sheet": str(contact_path.relative_to(root)),
                    "semantic_status": "pending_manual_review",
                }
            )

    if not records:
        raise ValueError("no aligned pilot groups were built")
    write_csv(output_csv, records)
    if skipped:
        write_csv(root / "data/waterdrop_train_pilot40_aligned_skipped.csv", skipped)
    print(
        f"Built {len(records) // 4} aligned groups and {len(records)} preliminary SFT records; "
        f"skipped={len(skipped)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
