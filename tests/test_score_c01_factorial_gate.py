import csv
import importlib.util
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "score_c01_factorial_gate.py"


def load_module():
    spec = importlib.util.spec_from_file_location("score_c01_factorial_gate", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_rows(pair_id: str = "pair_a") -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    review_rows = []
    key_rows = []
    expected = {
        "original": ("yes", "yes"),
        "remove_target": ("no", "no"),
        "footprint_only": ("no", "yes"),
        "target_only": ("yes", "no"),
    }
    for seed_index in range(5):
        for variant, (target_expected, footprint_expected) in expected.items():
            review_id = f"c01_000_s{seed_index:02d}_{variant}"
            target_label = "present" if target_expected == "yes" else "absent"
            footprint_label = "present" if footprint_expected == "yes" else "absent"
            review_rows.append(
                {
                    "review_id": review_id,
                    "target_visible": target_label,
                    "footprint_visible": footprint_label,
                    "scene_structure_preserved": "yes",
                    "cells_distinguishable": "yes",
                    "generation_failure": "no",
                    "mode_collapse": "no",
                    "notes": "",
                }
            )
            key_rows.append(
                {
                    "review_id": review_id,
                    "pair_id": pair_id,
                    "item_index": "0",
                    "seed_index": str(seed_index),
                    "variant": variant,
                    "expected_target_visible": target_expected,
                    "expected_footprint_visible": footprint_expected,
                }
            )
    return review_rows, key_rows


def test_score_gate_passes_clean_item(tmp_path):
    module = load_module()
    review_rows, key_rows = make_rows()
    review_csv = tmp_path / "blind_review.csv"
    key_csv = tmp_path / "answer_key.csv"
    write_csv(review_csv, review_rows)
    write_csv(key_csv, key_rows)

    output_dir = tmp_path / "scores"
    result = module.main(["--review-csv", str(review_csv), "--answer-key", str(key_csv), "--output-dir", str(output_dir)])

    assert result == 0
    item_rows = list(csv.DictReader((output_dir / "item_gate_summary.csv").open(encoding="utf-8")))
    assert item_rows[0]["gate_status"] == "pass"
    assert item_rows[0]["original_successes"] == "5"
    assert item_rows[0]["footprint_only_successes"] == "5"


def test_score_gate_fails_uncertain_and_scene_drift(tmp_path):
    review_rows, key_rows = make_rows(pair_id="pair_b")
    review_rows[0]["target_visible"] = "uncertain"
    review_rows[1]["scene_structure_preserved"] = "no"
    review_rows[2]["cells_distinguishable"] = "no"
    review_csv = tmp_path / "blind_review.csv"
    key_csv = tmp_path / "answer_key.csv"
    write_csv(review_csv, review_rows)
    write_csv(key_csv, key_rows)

    module = load_module()
    output_dir = tmp_path / "scores"
    module.main(["--review-csv", str(review_csv), "--answer-key", str(key_csv), "--output-dir", str(output_dir)])

    item_rows = list(csv.DictReader((output_dir / "item_gate_summary.csv").open(encoding="utf-8")))
    assert item_rows[0]["gate_status"] == "fail"
    assert "review_uncertain" in item_rows[0]["rejection_reasons"]
    assert "scene_drift" in item_rows[0]["rejection_reasons"]
    assert "cells_indistinguishable" in item_rows[0]["rejection_reasons"]


def test_score_gate_fails_when_variant_success_count_misses_threshold(tmp_path):
    review_rows, key_rows = make_rows(pair_id="pair_c")
    target_only_rows = [
        row for row in review_rows if row["review_id"].endswith("_target_only")
    ]
    target_only_rows[0]["footprint_visible"] = "present"
    target_only_rows[1]["footprint_visible"] = "present"
    review_csv = tmp_path / "blind_review.csv"
    key_csv = tmp_path / "answer_key.csv"
    write_csv(review_csv, review_rows)
    write_csv(key_csv, key_rows)

    module = load_module()
    output_dir = tmp_path / "scores"
    module.main(["--review-csv", str(review_csv), "--answer-key", str(key_csv), "--output-dir", str(output_dir)])

    cell_rows = list(csv.DictReader((output_dir / "cell_gate_summary.csv").open(encoding="utf-8")))
    item_rows = list(csv.DictReader((output_dir / "item_gate_summary.csv").open(encoding="utf-8")))

    assert item_rows[0]["gate_status"] == "fail"
    assert item_rows[0]["target_only_successes"] == "3"
    assert "target_only_below_threshold" in item_rows[0]["rejection_reasons"]
    assert {
        row["rejection_reasons"]
        for row in cell_rows
        if row["variant"] == "target_only" and row["cell_success"] == "false"
    } == {"target_only_preserves_footprint"}
