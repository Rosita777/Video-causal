from pathlib import Path
import csv


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_cross_round_master_matrix_has_expected_recovered_rows():
    path = PROJECT_ROOT / "experiments/pilot_week1/cross_round_summary/rounds_1_3_master_matrix.csv"

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 59
    strict = {(row["prompt_id"], row["baseline"]) for row in rows if row["outcome"] == "strict_causal_footprint"}
    assert strict == {
        ("pitcher_seed63", "negative_prompt"),
        ("pitcher_seed63", "videoeraser"),
        ("ice_cube_seed66", "negative_prompt"),
        ("ice_cube_seed67", "negative_prompt"),
    }


def test_required_baseline_coverage_records_round2_gaps():
    path = PROJECT_ROOT / "experiments/pilot_week1/cross_round_summary/rounds_1_3_required_baseline_coverage.csv"

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    missing = [row for row in rows if row["coverage_status"] != "annotated"]
    assert len(rows) == 65
    assert len(missing) == 6
    assert {row["baseline"] for row in missing} == {"t2vunlearning", "safree_cogvideox"}
    assert {row["round"] for row in missing} == {"round2_car_barrier"}
