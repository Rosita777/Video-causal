from pathlib import Path
import csv
import json
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def write_inputs(path: Path) -> None:
    fieldnames = [
        "output_id",
        "item_id",
        "baseline",
        "video_path",
        "sheet_path",
        "sheet_exists",
        "sheet_error",
        "target_concept",
        "expected_effect",
        "source_prompt",
        "human_label",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "output_id": "valid5:p1::videoeraser",
                "item_id": "valid5:p1",
                "baseline": "videoeraser",
                "video_path": "outputs/ve.mp4",
                "sheet_path": "experiments/eval_calibration/frame_sheets/ve.jpg",
                "sheet_exists": "true",
                "sheet_error": "",
                "target_concept": "pebble",
                "expected_effect": "ripples spread outward",
                "source_prompt": "A pebble falls into water.",
                "human_label": "strict_leakage",
            }
        )
        writer.writerow(
            {
                "output_id": "valid5:p2::negative_prompt",
                "item_id": "valid5:p2",
                "baseline": "negative_prompt",
                "video_path": "outputs/np.mp4",
                "sheet_path": "",
                "sheet_exists": "false",
                "sheet_error": "missing video",
                "target_concept": "rock",
                "expected_effect": "spiderweb crack",
                "source_prompt": "A rock hits glass.",
                "human_label": "target_leakage",
            }
        )


def test_evaluate_with_vlm_dry_run_writes_payloads_without_gold_label(tmp_path):
    inputs = tmp_path / "vlm_inputs.csv"
    output = tmp_path / "payloads.jsonl"
    write_inputs(inputs)

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "evaluate_with_vlm.py"),
            "--inputs",
            str(inputs),
            "--output-jsonl",
            str(output),
            "--dry-run",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "Wrote 1 dry-run payloads" in result.stdout
    payloads = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(payloads) == 1
    payload = payloads[0]
    assert payload["output_id"] == "valid5:p1::videoeraser"
    assert payload["image_path"] == "experiments/eval_calibration/frame_sheets/ve.jpg"
    assert payload["target_concept"] == "pebble"
    assert "ripples spread outward" in payload["prompt"]
    assert payload["response_schema"]["pred_label"] == [
        "strict_leakage",
        "borderline",
        "target_leakage",
        "other_failure",
    ]
    assert "human_label" not in json.dumps(payload)
    assert "strict_leakage" not in payload["prompt"]


def test_evaluate_with_vlm_can_include_missing_sheets(tmp_path):
    inputs = tmp_path / "vlm_inputs.csv"
    output = tmp_path / "payloads.jsonl"
    write_inputs(inputs)

    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "evaluate_with_vlm.py"),
            "--inputs",
            str(inputs),
            "--output-jsonl",
            str(output),
            "--dry-run",
            "--include-missing",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    payloads = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(payloads) == 2
    assert payloads[1]["sheet_available"] is False
    assert payloads[1]["sheet_error"] == "missing video"
