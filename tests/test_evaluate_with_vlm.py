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
        "reference_sheet_path",
        "reference_sheet_exists",
        "reference_sheet_error",
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
                "reference_sheet_path": "experiments/eval_calibration/frame_sheets/ref.jpg",
                "reference_sheet_exists": "true",
                "reference_sheet_error": "",
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
                "reference_sheet_path": "",
                "reference_sheet_exists": "false",
                "reference_sheet_error": "missing reference",
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
    assert payload["reference_image_path"] == "experiments/eval_calibration/frame_sheets/ref.jpg"
    assert payload["reference_available"] is True
    assert payload["target_concept"] == "pebble"
    assert "first image is a clean reference" in payload["prompt"]
    assert "second image is the erased output" in payload["prompt"]
    assert "ripples spread outward" in payload["prompt"]
    assert payload["response_schema"]["target_visible"] == ["yes", "no", "partial"]
    assert payload["response_schema"]["separation_clear"] == ["yes", "no"]
    assert "pred_label" not in payload["response_schema"]
    assert '"target_visible": "yes|no|partial"' in payload["prompt"]
    assert '"effect_visible": "yes|no|partial"' in payload["prompt"]
    assert "Do not choose the final benchmark label" in payload["prompt"]
    assert "Use target_visible=no only when the target is absent in every frame" in payload["prompt"]
    assert "If any candidate target cue remains" in payload["prompt"]
    assert "Use separation_clear=no when the effect could hide or mimic the target" in payload["prompt"]
    assert "human_label" not in json.dumps(payload)


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
    assert payloads[1]["reference_available"] is False
    assert payloads[1]["sheet_error"] == "missing video"


def test_filter_rows_can_require_reference_sheets():
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from evaluate_with_vlm import filter_rows

    rows = [
        {"sheet_exists": "true", "reference_sheet_exists": "true", "id": "keep"},
        {"sheet_exists": "true", "reference_sheet_exists": "false", "id": "drop_reference"},
        {"sheet_exists": "false", "reference_sheet_exists": "true", "id": "drop_sheet"},
    ]

    filtered = filter_rows(rows, include_missing=False, require_reference=True, limit=None)

    assert [row["id"] for row in filtered] == ["keep"]


def test_parse_model_json_accepts_fenced_json():
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from evaluate_with_vlm import parse_model_json

    parsed = parse_model_json(
        "```json\n"
        '{"target_absent":"yes","effect_visible":"partial","quality_ok":"yes",'
        '"pred_label":"borderline","confidence":0.7,"reason":"ambiguous source cue"}'
        "\n```"
    )

    assert parsed["target_absent"] == "yes"
    assert parsed["effect_visible"] == "partial"
    assert parsed["pred_label"] == "borderline"
    assert parsed["confidence"] == 0.7


def test_normalize_prediction_accepts_boolean_flags():
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from evaluate_with_vlm import normalize_prediction

    normalized = normalize_prediction(
        {
            "target_visible": False,
            "effect_visible": False,
            "separation_clear": True,
            "quality_ok": True,
            "confidence": "0.5",
            "reason": "boolean flags from model",
        }
    )

    assert normalized["target_absent"] == "yes"
    assert normalized["effect_visible"] == "no"
    assert normalized["quality_ok"] == "yes"
    assert normalized["pred_label"] == "other_failure"
    assert normalized["confidence"] == "0.5000"


def test_normalize_prediction_derives_label_from_atomic_fields():
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from evaluate_with_vlm import normalize_prediction

    cases = [
        (
            {
                "target_visible": "yes",
                "effect_visible": "yes",
                "separation_clear": "yes",
                "quality_ok": "yes",
            },
            "no",
            "target_leakage",
        ),
        (
            {
                "target_visible": "no",
                "effect_visible": "yes",
                "separation_clear": "yes",
                "quality_ok": "yes",
            },
            "yes",
            "strict_leakage",
        ),
        (
            {
                "target_visible": "partial",
                "effect_visible": "yes",
                "separation_clear": "yes",
                "quality_ok": "yes",
            },
            "partial",
            "borderline",
        ),
        (
            {
                "target_visible": "no",
                "effect_visible": "partial",
                "separation_clear": "yes",
                "quality_ok": "yes",
            },
            "yes",
            "borderline",
        ),
        (
            {
                "target_visible": "no",
                "effect_visible": "yes",
                "separation_clear": "no",
                "quality_ok": "yes",
            },
            "yes",
            "borderline",
        ),
        (
            {
                "target_visible": "no",
                "effect_visible": "yes",
                "separation_clear": "yes",
                "quality_ok": "no",
            },
            "yes",
            "other_failure",
        ),
    ]

    for parsed, expected_absent, expected_label in cases:
        parsed = {**parsed, "confidence": 1.2, "reason": expected_label}
        normalized = normalize_prediction(parsed)
        assert normalized["target_absent"] == expected_absent
        assert normalized["pred_label"] == expected_label
        assert normalized["confidence"] == "1.0000"


def test_run_api_with_fake_transport_writes_prediction_csv(tmp_path):
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from evaluate_with_vlm import run_api_mode

    inputs = tmp_path / "vlm_inputs.csv"
    image = tmp_path / "sheet.jpg"
    reference = tmp_path / "reference.jpg"
    predictions = tmp_path / "predictions.csv"
    raw = tmp_path / "raw.jsonl"
    image.write_bytes(b"fake-image")
    reference.write_bytes(b"fake-reference")
    write_inputs(inputs)

    def fake_transport(_url, _api_key, payload, _timeout):
        assert payload["model"] == "openai/gpt-4o"
        content = payload["messages"][0]["content"]
        image_parts = [part for part in content if part["type"] == "image_url"]
        assert len(image_parts) == 2
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "target_visible": "no",
                                "effect_visible": "yes",
                                "separation_clear": "yes",
                                "quality_ok": "yes",
                                "confidence": 0.88,
                                "reason": "target absent while ripples remain",
                            }
                        )
                    }
                }
            ]
        }

    # Rewrite the first row's sheet path to an existing image; the second row is skipped.
    rows = list(csv.DictReader(inputs.open(newline="", encoding="utf-8")))
    rows[0]["reference_sheet_path"] = str(reference)
    rows[0]["sheet_path"] = str(image)
    with inputs.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    count = run_api_mode(
        inputs_path=inputs,
        predictions_path=predictions,
        raw_output_path=raw,
        base_url="https://example.test/v1",
        api_key="not-secret",
        model="openai/gpt-4o",
        include_missing=False,
        require_reference=False,
        limit=None,
        temperature=0.0,
        max_tokens=200,
        timeout=10,
        transport=fake_transport,
    )

    assert count == 1
    with predictions.open(newline="", encoding="utf-8") as handle:
        prediction_rows = list(csv.DictReader(handle))
    assert prediction_rows == [
        {
            "item_id": "valid5:p1",
            "baseline": "videoeraser",
            "video_path": "outputs/ve.mp4",
            "target_absent": "yes",
            "effect_visible": "yes",
            "quality_ok": "yes",
            "pred_label": "strict_leakage",
            "confidence": "0.8800",
            "reason": "target absent while ripples remain",
        }
    ]
    raw_payload = json.loads(raw.read_text(encoding="utf-8").splitlines()[0])
    assert raw_payload["output_id"] == "valid5:p1::videoeraser"
    assert "api_key" not in json.dumps(raw_payload)
    assert "data:image" not in json.dumps(raw_payload)
    assert raw_payload["model_content"]
