from pathlib import Path
import csv
import importlib.util
import json
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCORER_PATH = PROJECT_ROOT / "scripts" / "score_c0_counterfactual_grid.py"


def load_scorer_module():
    spec = importlib.util.spec_from_file_location("score_c0_counterfactual_grid", SCORER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def prediction_row(
    pair_id: str,
    baseline: str,
    *,
    target_visible: str,
    footprint_visible: str,
    video_quality: str = "yes",
    item_index: int = 0,
) -> dict[str, str]:
    return {
        "item_id": f"c0:{pair_id}",
        "item_index": str(item_index),
        "pair_id": pair_id,
        "mechanism_type": "fracture_damage",
        "baseline": baseline,
        "target_concept": "black hockey puck",
        "expected_effect": "a star-shaped crack spreads across the mirror",
        "target_visible": target_visible,
        "footprint_visible": footprint_visible,
        "video_quality": video_quality,
        "confidence": "0.95",
        "final_label": "synthetic",
        "notes": "",
    }


def complete_item_rows(pair_id: str, *, item_index: int = 0) -> list[dict[str, str]]:
    return [
        prediction_row(
            pair_id,
            "original",
            target_visible="yes",
            footprint_visible="yes",
            item_index=item_index,
        ),
        prediction_row(
            pair_id,
            "remove_target",
            target_visible="no",
            footprint_visible="no",
            item_index=item_index,
        ),
        prediction_row(
            pair_id,
            "footprint_only",
            target_visible="no",
            footprint_visible="yes",
            item_index=item_index,
        ),
        prediction_row(
            pair_id,
            "target_only",
            target_visible="yes",
            footprint_visible="no",
            item_index=item_index,
        ),
    ]


def write_predictions_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_normalizes_yes_no_labels_and_scores_expected_variant_states():
    module = load_scorer_module()
    rows = complete_item_rows("good_item")
    scored = module.score_prediction_rows(rows)

    assert module.normalize_bool("YES") is True
    assert module.normalize_bool("false") is False
    assert module.normalize_bool("") is None
    assert [row["variant_pass"] for row in scored] == [True, True, True, True]

    target_only = next(row for row in scored if row["variant_role"] == "target_only")
    assert target_only["expected_target_visible"] is True
    assert target_only["expected_footprint_visible"] is False
    assert target_only["observed_target_visible"] is True
    assert target_only["observed_footprint_visible"] is False


def test_aggregates_original_validity_and_counterfactual_failures():
    module = load_scorer_module()
    rows = []
    rows.extend(complete_item_rows("good_item", item_index=0))

    invalid_original = complete_item_rows("invalid_original", item_index=1)
    invalid_original[0]["footprint_visible"] = "no"
    rows.extend(invalid_original)

    failed_counterfactual = complete_item_rows("failed_counterfactual", item_index=2)
    failed_counterfactual[1]["target_visible"] = "yes"
    rows.extend(failed_counterfactual)

    scored = module.score_prediction_rows(rows)
    items = module.aggregate_item_scores(scored)
    by_pair = {row["pair_id"]: row for row in items}

    assert by_pair["good_item"]["original_valid"] is True
    assert by_pair["good_item"]["counterfactual_pass"] is True
    assert by_pair["good_item"]["c0_grid_pass"] is True
    assert by_pair["good_item"]["failure_mode"] == "pass"

    assert by_pair["invalid_original"]["original_valid"] is False
    assert by_pair["invalid_original"]["counterfactual_pass"] is True
    assert by_pair["invalid_original"]["c0_grid_pass"] is False
    assert by_pair["invalid_original"]["failure_mode"] == "invalid_original"

    assert by_pair["failed_counterfactual"]["original_valid"] is True
    assert by_pair["failed_counterfactual"]["counterfactual_pass"] is False
    assert by_pair["failed_counterfactual"]["failure_mode"] == "failed:remove_target"


def test_cli_writes_variant_item_and_summary_outputs(tmp_path):
    rows = []
    rows.extend(complete_item_rows("good_item", item_index=0))
    failed = complete_item_rows("failed_item", item_index=1)
    failed[3]["footprint_visible"] = "yes"
    rows.extend(failed)
    predictions = tmp_path / "vlm_predictions.csv"
    output_dir = tmp_path / "scores"
    write_predictions_csv(predictions, rows)

    result = subprocess.run(
        [
            sys.executable,
            str(SCORER_PATH),
            "--predictions-csv",
            str(predictions),
            "--output-dir",
            str(output_dir),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert (output_dir / "c0_variant_scores.csv").exists()
    assert (output_dir / "c0_item_scores.csv").exists()
    summary = json.loads((output_dir / "c0_summary.json").read_text(encoding="utf-8"))
    assert summary["total_items"] == 2
    assert summary["original_valid_items"] == 2
    assert summary["counterfactual_pass_items"] == 1
    assert summary["c0_grid_pass_items"] == 1
    assert summary["variant_pass_counts"]["target_only"] == 1
