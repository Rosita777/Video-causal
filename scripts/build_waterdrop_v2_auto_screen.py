#!/usr/bin/env python3
"""Run first-pass technical screening and build contact sheets for waterdrop v2."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw


def portable_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def load_video(path: Path) -> tuple[np.ndarray, float]:
    reader = imageio.get_reader(path)
    try:
        metadata = reader.get_meta_data()
        frames = np.stack([frame[:, :, :3] for frame in reader], axis=0)
    finally:
        reader.close()
    return frames, float(metadata.get("fps", 0.0))


def measure_activity(frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    baseline = frames[0].astype(np.int16)
    maes = []
    changed_fractions = []
    for frame in frames:
        pixel_delta = np.abs(frame.astype(np.int16) - baseline).mean(axis=2)
        maes.append(float(pixel_delta.mean()))
        changed_fractions.append(float((pixel_delta > 12.0).mean()))
    return np.asarray(maes), np.asarray(changed_fractions)


def estimate_onset(maes: np.ndarray, changed: np.ndarray) -> int | None:
    for index in range(1, len(maes)):
        if maes[index] > 0.25 and changed[index] > 0.0015:
            return index
    return None


def make_contact_sheet(
    frames: np.ndarray,
    output: Path,
    *,
    scene_id: str,
    family: str,
    receiver: str,
    onset: int | None,
    auto_status: str,
) -> None:
    indices = np.linspace(0, len(frames) - 1, 12, dtype=int)
    width = 240
    height = round(frames.shape[1] * width / frames.shape[2])
    label_height = 24
    header_height = 52
    sheet = Image.new("RGB", (width * 4, header_height + (height + label_height) * 3), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 6), f"{scene_id} | {family} | {auto_status} | onset={onset}", fill="black")
    draw.text((8, 28), receiver[:115], fill="black")
    for position, frame_index in enumerate(indices):
        x = (position % 4) * width
        y = header_height + (position // 4) * (height + label_height)
        draw.text((x + 5, y + 5), f"frame {frame_index}", fill="black")
        image = Image.fromarray(frames[frame_index]).resize((width, height))
        sheet.paste(image, (x, y + label_height))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=90)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--run-manifest",
        type=Path,
        default=Path("data/waterdrop_prompt_bank_v2_simple_run_manifest.csv"),
    )
    parser.add_argument(
        "--generation-root",
        type=Path,
        default=Path("outputs/waterdrop_prompt_bank_v2_simple_wan"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("data/waterdrop_prompt_bank_v2_auto_screen.csv"),
    )
    parser.add_argument(
        "--contact-sheet-dir",
        type=Path,
        default=Path("outputs/waterdrop_prompt_bank_v2_auto_contact_sheets"),
    )
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    root = args.repo_root.resolve()
    run_manifest = args.run_manifest if args.run_manifest.is_absolute() else root / args.run_manifest
    generation_root = (
        args.generation_root if args.generation_root.is_absolute() else root / args.generation_root
    )
    output_csv = args.output_csv if args.output_csv.is_absolute() else root / args.output_csv
    sheet_dir = (
        args.contact_sheet_dir
        if args.contact_sheet_dir.is_absolute()
        else root / args.contact_sheet_dir
    )
    with run_manifest.open(newline="", encoding="utf-8") as handle:
        run_rows = list(csv.DictReader(handle))

    generation_items: dict[tuple[int, int], dict[str, object]] = {}
    for shard in range(6):
        path = generation_root / f"shard_{shard}" / "generation_manifest.json"
        if path.is_file():
            with path.open(encoding="utf-8") as handle:
                manifest = json.load(handle)
            for item in manifest["items"]:
                generation_items[(shard, int(item["index"]))] = item

    results: list[dict[str, object]] = []
    for row in run_rows:
        shard = int(row["shard"])
        shard_index = int(row["shard_index"])
        item = generation_items.get((shard, shard_index))
        if item is not None:
            video_path = root / str(item["video_path"])
        else:
            matches = list(
                (generation_root / f"shard_{shard}" / "videos").glob(
                    f"{shard_index:03d}_*_seed{row['fixed_seed']}.mp4"
                )
            )
            video_path = matches[0] if len(matches) == 1 else Path()
        if not video_path.is_file():
            if args.allow_partial:
                continue
            raise FileNotFoundError(video_path)

        try:
            frames, fps = load_video(video_path)
            valid_shape = frames.shape == (49, 480, 832, 3)
            maes, changed = measure_activity(frames)
            onset = estimate_onset(maes, changed)
            peak_mae = float(maes.max())
            peak_changed = float(changed.max())
            final_changed = float(changed[-4:].mean())
            clean_mae = float(maes[:onset].max()) if onset and onset > 1 else float("nan")
            if not valid_shape or abs(fps - 8.0) > 0.1:
                auto_status = "reject_invalid_video"
            elif onset is None or peak_changed < 0.002:
                auto_status = "reject_no_detectable_event"
            elif onset < 2:
                auto_status = "reject_no_clean_prefix"
            elif onset < 4:
                auto_status = "review_short_prefix"
            else:
                auto_status = "candidate"
            error = ""
        except Exception as exc:  # Keep the batch report complete when one video is corrupt.
            frames = None
            fps = 0.0
            onset = None
            peak_mae = 0.0
            peak_changed = 0.0
            final_changed = 0.0
            clean_mae = float("nan")
            valid_shape = False
            auto_status = "reject_decode_error"
            error = f"{type(exc).__name__}: {exc}"

        sheet_path = sheet_dir / f"{row['scene_id']}_{row['family']}_shard{shard}_{shard_index:03d}.jpg"
        if frames is not None:
            make_contact_sheet(
                frames,
                sheet_path,
                scene_id=row["scene_id"],
                family=row["family"],
                receiver=row["receiver"],
                onset=onset,
                auto_status=auto_status,
            )
        results.append(
            {
                **row,
                "video_path": portable_path(video_path, root),
                "frame_count": 0 if frames is None else len(frames),
                "fps": fps,
                "estimated_clean_end_exclusive": "" if onset is None else onset,
                "clean_prefix_max_mae_255": clean_mae,
                "peak_frame0_mae_255": peak_mae,
                "peak_changed_fraction": peak_changed,
                "final_changed_fraction": final_changed,
                "auto_status": auto_status,
                "contact_sheet": "" if frames is None else portable_path(sheet_path, root),
                "error": error,
            }
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    counts = Counter(str(result["auto_status"]) for result in results)
    print(f"Screened {len(results)} videos: {dict(counts)}")
    print(f"Contact sheets: {sheet_dir}")
    print(f"CSV: {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
