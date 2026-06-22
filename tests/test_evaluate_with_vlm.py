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
    assert '"effect_visible": "yes|no|partial"' in payload["prompt"]
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
    assert payloads[1]["sheet_error"] == "missing video"


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
            "target_absent": True,
            "effect_visible": False,
            "quality_ok": True,
            "pred_label": "strict_leakage",
            "confidence": "0.5",
            "reason": "boolean flags from model",
        }
    )

    assert normalized["target_absent"] == "yes"
    assert normalized["effect_visible"] == "no"
    assert normalized["quality_ok"] == "yes"
    assert normalized["confidence"] == "0.5000"


def test_normalize_prediction_accepts_common_aliases():
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from evaluate_with_vlm import normalize_prediction

    normalized = normalize_prediction(
        {
            "target_absent": "partial",
            "causal_effect_visible": "yes",
            "quality_sufficient": "true",
            "label": "target_leakage",
            "confidence": 1.2,
            "reason": "target cue remains",
        }
    )

    assert normalized["effect_visible"] == "yes"
    assert normalized["quality_ok"] == "yes"
    assert normalized["pred_label"] == "target_leakage"
    assert normalized["confidence"] == "1.0000"


def test_run_api_with_fake_transport_writes_prediction_csv(tmp_path):
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from evaluate_with_vlm import run_api_mode

    inputs = tmp_path / "vlm_inputs.csv"
    image = tmp_path / "sheet.jpg"
    predictions = tmp_path / "predictions.csv"
    raw = tmp_path / "raw.jsonl"
    image.write_bytes(b"fake-image")
    write_inputs(inputs)

    def fake_transport(_url, _api_key, payload, _timeout):
        assert payload["model"] == "openai/gpt-4o"
        content = payload["messages"][0]["content"]
        assert content[1]["type"] == "image_url"
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "target_absent": "yes",
                                "effect_visible": "yes",
                                "quality_ok": "yes",
                                "pred_label": "strict_leakage",
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
