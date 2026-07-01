from pathlib import Path
import csv
import json
import subprocess
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_chunk_ranges_cover_49_frames_with_overlap():
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from evaluate_clean_chunks_with_vlm import chunk_ranges

    ranges = chunk_ranges(frame_count=49, chunk_count=5, overlap=3)

    assert ranges == [(0, 12), (9, 21), (18, 30), (27, 39), (36, 48)]
    covered = {frame for start, end in ranges for frame in range(start, end + 1)}
    assert covered == set(range(49))


def test_aggregate_chunk_predictions_uses_frame_ids_across_chunks():
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from evaluate_clean_chunks_with_vlm import aggregate_chunk_predictions

    row = {
        "prompt_id": "case_000",
        "pair_id": "fluid_pebble_000",
        "target_concept": "pebble",
        "expected_effect": "ripples spread outward",
    }
    chunks = [
        {
            "target_visible": "yes",
            "target_frame_ids": ["f03"],
            "effect_visible": "no",
            "effect_frame_ids": [],
            "local_temporal_order": "partial",
            "local_causal_transition": "partial",
            "video_quality": "yes",
            "confidence": "0.8000",
            "reason": "pebble first appears before the water changes",
        },
        {
            "target_visible": "no",
            "target_frame_ids": [],
            "effect_visible": "yes",
            "effect_frame_ids": ["f14", "f15"],
            "local_temporal_order": "partial",
            "local_causal_transition": "partial",
            "video_quality": "yes",
            "confidence": "0.7000",
            "reason": "ripples persist after the impact",
        },
    ]

    pred = aggregate_chunk_predictions(row, chunks)

    assert pred["target_present"] == "yes"
    assert pred["effect_present"] == "yes"
    assert pred["first_target_frame"] == "3"
    assert pred["first_effect_frame"] == "14"
    assert pred["temporal_order_support"] == "partial"
    assert pred["rule_clean_source_candidate"] == "yes"


def test_normalize_chunk_prediction_accepts_single_item_flag_lists():
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from evaluate_clean_chunks_with_vlm import normalize_chunk_prediction

    pred = normalize_chunk_prediction(
        {
            "target_visible": ["partial"],
            "target_frame_ids": ["f003"],
            "effect_visible": ["no"],
            "effect_frame_ids": [],
            "local_temporal_order": ["partial"],
            "local_causal_transition": ["no"],
            "video_quality": ["yes"],
            "confidence": 0.75,
            "reason": "model used schema enum lists as values",
        }
    )

    assert pred["target_visible"] == "partial"
    assert pred["effect_visible"] == "no"
    assert pred["video_quality"] == "yes"
    assert pred["target_frame_ids"] == "f003"


def test_call_api_with_retries_recovers_from_transient_errors():
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    import evaluate_clean_chunks_with_vlm as clean_vlm

    attempts = []

    def flaky_transport(url, key, payload, timeout):
        attempts.append((url, key, payload, timeout))
        if len(attempts) < 3:
            raise TimeoutError("temporary timeout")
        return {"choices": [{"message": {"content": "{}"}}]}

    response = clean_vlm.call_api_with_retries(
        "https://example.test/chat/completions",
        "key",
        {"model": "demo"},
        timeout=12,
        retries=3,
        retry_sleep=0,
        transport=flaky_transport,
    )

    assert response["choices"][0]["message"]["content"] == "{}"
    assert len(attempts) == 3


def test_chunk_prediction_key_identifies_completed_chunk():
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from evaluate_clean_chunks_with_vlm import chunk_prediction_key

    row = {
        "row_index": "12",
        "chunk_index": "3",
        "frame_start": "8",
        "frame_end": "15",
    }

    assert chunk_prediction_key(row) == ("12", "3", "8", "15")


def test_pending_chunks_skips_existing_predictions():
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from evaluate_clean_chunks_with_vlm import pending_chunks

    chunks = [
        {"row_index": "0", "chunk_index": "0", "frame_start": "0", "frame_end": "7"},
        {"row_index": "0", "chunk_index": "1", "frame_start": "4", "frame_end": "11"},
    ]
    existing = [
        {"row_index": "0", "chunk_index": "0", "frame_start": "0", "frame_end": "7"},
    ]

    assert pending_chunks(chunks, existing) == [chunks[1]]


def test_chunked_clean_vlm_dry_run_builds_all_frame_sheets(tmp_path):
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    video = tmp_path / "demo.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 8, (64, 36))
    try:
        for index in range(8):
            frame = np.full((36, 64, 3), 20 + index * 20, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()

    review = tmp_path / "clean_source_screening.csv"
    fields = [
        "prompt_id",
        "pair_id",
        "baseline",
        "mechanism_type",
        "video_path",
        "prompt",
        "target_concept",
        "expected_effect",
    ]
    with review.open("w", newline="", encoding="utf-8") as handle:
        writer_csv = csv.DictWriter(handle, fieldnames=fields)
        writer_csv.writeheader()
        writer_csv.writerow(
            {
                "prompt_id": "case_000",
                "pair_id": "pair_000",
                "baseline": "clean",
                "mechanism_type": "fluid_impact",
                "video_path": str(video),
                "prompt": "A pebble falls into water and ripples spread.",
                "target_concept": "pebble",
                "expected_effect": "ripples spread outward",
            }
        )
    output_dir = tmp_path / "chunked"

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "evaluate_clean_chunks_with_vlm.py"),
            "--review-csv",
            str(review),
            "--output-dir",
            str(output_dir),
            "--dry-run",
            "--chunk-count",
            "3",
            "--chunk-overlap",
            "1",
            "--thumb-width",
            "64",
            "--thumb-height",
            "36",
            "--sheet-columns",
            "4",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    payloads = [json.loads(line) for line in (output_dir / "chunk_payloads.jsonl").read_text(encoding="utf-8").splitlines()]
    manifest_rows = list(csv.DictReader((output_dir / "chunk_manifest.csv").open(newline="", encoding="utf-8")))
    assert len(payloads) == 3
    assert len(manifest_rows) == 3
    assert {row["frame_start"] for row in manifest_rows} == {"0", "2", "4"}
    assert all(Path(row["sheet_path"]).exists() for row in manifest_rows)
