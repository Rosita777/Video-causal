from pathlib import Path
import csv
import json
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def write_candidate_manifest(path: Path) -> None:
    items = [
        {
            "pair_id": "fluid_pebble",
            "target_concept": "pebble",
            "expected_effect": "ripples spread outward",
            "causal_footprint": "ripples spread outward",
            "mechanism_type": "fluid_impact",
            "temporal_type": "delayed",
            "source_prompt": "A pebble drops into still water, causing ripples to spread outward.",
            "counterfactual_prompt": "Still water with no pebble.",
            "control_prompt": "Ripples without a visible pebble.",
            "scores": {"exclusivity_score": 5},
        },
        {
            "pair_id": "fracture_rock",
            "target_concept": "rock",
            "expected_effect": "spiderweb crack appears",
            "causal_footprint": "spiderweb crack appears",
            "mechanism_type": "fracture_damage",
            "temporal_type": "persistent",
            "source_prompt": "A rock hits glass, causing a spiderweb crack.",
            "counterfactual_prompt": "Intact glass with no rock.",
            "control_prompt": "A cracked glass pane with no rock.",
            "scores": {"exclusivity_score": 4},
        },
        {
            "pair_id": "surface_stamp",
            "target_concept": "stamp",
            "expected_effect": "ink mark remains",
            "causal_footprint": "ink mark remains",
            "mechanism_type": "surface_trace",
            "temporal_type": "persistent",
            "source_prompt": "A stamp presses onto paper, leaving an ink mark.",
            "counterfactual_prompt": "Blank paper with no stamp.",
            "control_prompt": "Paper with an ink mark and no stamp.",
            "scores": {"exclusivity_score": 5},
        },
    ]
    path.write_text(json.dumps({"items": items}, indent=2) + "\n", encoding="utf-8")


def write_clean_manifest(path: Path) -> None:
    items = [
        {"source_prompt_index": 0, "video_path": "outputs/clean/prompt_000.mp4"},
        {"source_prompt_index": 1, "video_path": "outputs/clean/prompt_001.mp4"},
        {"source_prompt_index": 2, "video_path": "outputs/clean/prompt_002.mp4"},
    ]
    path.write_text(json.dumps({"items": items}, indent=2) + "\n", encoding="utf-8")


def write_clean_predictions(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_id",
                "rule_clean_source_candidate",
                "target_present",
                "effect_present",
                "confidence",
                "reason",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "sample_id": "case_000",
                "rule_clean_source_candidate": "yes",
                "target_present": "yes",
                "effect_present": "yes",
                "confidence": "0.90",
                "reason": "valid clean source",
            }
        )
        writer.writerow(
            {
                "sample_id": "case_001",
                "rule_clean_source_candidate": "no",
                "target_present": "yes",
                "effect_present": "no",
                "confidence": "0.80",
                "reason": "effect missing",
            }
        )
        writer.writerow(
            {
                "sample_id": "case_002",
                "rule_clean_source_candidate": "yes",
                "target_present": "yes",
                "effect_present": "yes",
                "confidence": "0.88",
                "reason": "valid clean source",
            }
        )


def test_export_v2_clean_source_slice_preserves_source_and_slice_indices(tmp_path):
    candidates = tmp_path / "candidate_manifest.json"
    clean_manifest = tmp_path / "clean_manifest.json"
    predictions = tmp_path / "clean_predictions.csv"
    prompt_output = tmp_path / "prompts.txt"
    manifest_output = tmp_path / "slice_manifest.json"
    write_candidate_manifest(candidates)
    write_clean_manifest(clean_manifest)
    write_clean_predictions(predictions)

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "export_v2_clean_source_slice.py"),
            "--candidate-manifest",
            str(candidates),
            "--clean-predictions",
            str(predictions),
            "--clean-generation-manifest",
            str(clean_manifest),
            "--output-prompts",
            str(prompt_output),
            "--output-manifest",
            str(manifest_output),
            "--slice-name",
            "zeroscope_test_yes2",
            "--clean-label-source-note",
            "unit-test labels",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Wrote 2 clean-source items" in result.stdout

    prompt_lines = [
        line for line in prompt_output.read_text(encoding="utf-8").splitlines() if line and not line.startswith("#")
    ]
    assert prompt_lines == [
        "A pebble drops into still water, causing ripples to spread outward. | pebble | ripples spread outward",
        "A stamp presses onto paper, leaving an ink mark. | stamp | ink mark remains",
    ]

    manifest = json.loads(manifest_output.read_text(encoding="utf-8"))
    assert manifest["slice_name"] == "zeroscope_test_yes2"
    assert manifest["count"] == 2
    assert manifest["output_prompts"] == str(prompt_output)
    assert manifest["clean_label_source"] == str(predictions)
    assert manifest["clean_source_valid"] == ["yes"]

    assert [(item["slice_index"], item["source_index"]) for item in manifest["items"]] == [
        (0, "0"),
        (1, "2"),
    ]
    assert manifest["items"][0]["clean_video_path"] == "outputs/clean/prompt_000.mp4"
    assert manifest["items"][1]["clean_video_path"] == "outputs/clean/prompt_002.mp4"
    assert manifest["items"][0]["clean_prompt_id"] == "case_000"
    assert manifest["items"][1]["clean_prompt_id"] == "case_002"
    assert manifest["items"][0]["clean_source_notes"] == "unit-test labels"
