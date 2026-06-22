from pathlib import Path
import csv
import subprocess
import sys

import av
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def write_tiny_video(path: Path, colors: list[tuple[int, int, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=5)
        stream.width = 64
        stream.height = 48
        stream.pix_fmt = "yuv420p"
        for color in colors:
            image = Image.new("RGB", (64, 48), color)
            frame = av.VideoFrame.from_image(image)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def write_gold(path: Path, video_path: Path, reference_video_path: Path | None = None) -> None:
    fieldnames = [
        "output_id",
        "item_id",
        "source_name",
        "pair_id",
        "mechanism_type",
        "baseline",
        "video_path",
        "reference_video_path",
        "reference_video_exists",
        "reference_video_quality",
        "reference_notes",
        "seed",
        "target_concept",
        "expected_effect",
        "source_prompt",
        "target_visible",
        "causal_effect_visible",
        "causeless_effect",
        "video_quality",
        "usable_for_claim",
        "failure_mode",
        "human_label",
        "notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "output_id": "valid5:p1::videoeraser",
                "item_id": "valid5:p1",
                "source_name": "valid5",
                "pair_id": "p1",
                "mechanism_type": "fluid_impact",
                "baseline": "videoeraser",
                "video_path": str(video_path),
                "reference_video_path": "" if reference_video_path is None else str(reference_video_path),
                "reference_video_exists": "false" if reference_video_path is None else "true",
                "reference_video_quality": "good" if reference_video_path is not None else "",
                "reference_notes": "clean reference" if reference_video_path is not None else "",
                "seed": "7",
                "target_concept": "pebble",
                "expected_effect": "ripples spread outward",
                "source_prompt": "A pebble falls into still water.",
                "target_visible": "no",
                "causal_effect_visible": "yes",
                "causeless_effect": "yes",
                "video_quality": "good",
                "usable_for_claim": "yes",
                "failure_mode": "causal_footprint_leakage",
                "human_label": "strict_leakage",
                "notes": "ripples remain",
            }
        )


def test_build_vlm_eval_inputs_creates_sheet_and_csv_row(tmp_path):
    video = tmp_path / "videos" / "sample.mp4"
    reference_video = tmp_path / "videos" / "reference.mp4"
    gold = tmp_path / "gold.csv"
    sheet_dir = tmp_path / "sheets"
    output = tmp_path / "vlm_inputs.csv"
    write_tiny_video(video, [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)])
    write_tiny_video(reference_video, [(0, 0, 0), (30, 30, 30), (60, 60, 60), (90, 90, 90), (120, 120, 120)])
    write_gold(gold, video, reference_video)

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "build_vlm_eval_inputs.py"),
            "--gold",
            str(gold),
            "--sheet-dir",
            str(sheet_dir),
            "--output",
            str(output),
            "--frames-per-video",
            "5",
            "--thumb-width",
            "32",
            "--thumb-height",
            "24",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "Wrote 1 VLM input rows" in result.stdout
    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert rows[0]["output_id"] == "valid5:p1::videoeraser"
    assert rows[0]["sheet_exists"] == "true"
    assert rows[0]["sheet_error"] == ""
    assert rows[0]["target_concept"] == "pebble"
    assert rows[0]["human_label"] == "strict_leakage"
    assert rows[0]["reference_sheet_exists"] == "true"
    assert rows[0]["reference_sheet_error"] == ""
    sheet = Image.open(rows[0]["sheet_path"])
    reference_sheet = Image.open(rows[0]["reference_sheet_path"])
    assert sheet.size == (160, 24)
    assert reference_sheet.size == (160, 24)


def test_build_vlm_eval_inputs_keeps_missing_video_rows(tmp_path):
    missing_video = tmp_path / "missing.mp4"
    gold = tmp_path / "gold.csv"
    sheet_dir = tmp_path / "sheets"
    output = tmp_path / "vlm_inputs.csv"
    write_gold(gold, missing_video)

    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "build_vlm_eval_inputs.py"),
            "--gold",
            str(gold),
            "--sheet-dir",
            str(sheet_dir),
            "--output",
            str(output),
        ],
        cwd=PROJECT_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert rows[0]["video_exists"] == "false"
    assert rows[0]["sheet_exists"] == "false"
    assert rows[0]["sheet_path"] == ""
    assert "missing video" in rows[0]["sheet_error"]
