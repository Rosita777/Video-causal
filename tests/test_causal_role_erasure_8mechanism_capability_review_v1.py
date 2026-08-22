from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import causal_role_erasure_8mechanism_capability_review_v1 as review  # noqa: E402


MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "causal_role_erasure_8mechanism_capability_v1_manifest.canonical.json"
)
DATA_DIR = PROJECT_ROOT / "data"
SCRIPT_PATH = SCRIPTS_DIR / "causal_role_erasure_8mechanism_capability_review_v1.py"
EXPECTED_REVIEW_ARTIFACT_SHA256 = {
    "review_template": "286bb5a1dbe3cceb606ec46e0f69202e93af5afd4de1978e83835292b67230ce",
    "review_rubric": "aa88f142891312a404832128ba4688beb1b537d1cfc70f8ccf395a840ef9405f",
    "review_freeze": "d3a331982d1c0a89146edcaf78fcb63fc12476299b942a5344b079f8a6deb3a8",
}


def _manifest_rows():
    return review.load_manifest(MANIFEST_PATH)


def _filled_rows(*, pass_count: int = 15):
    rows = deepcopy(review.build_review_rows(_manifest_rows()))
    for mechanism in review.MECHANISM_ORDER:
        mechanism_rows = [row for row in rows if row["mechanism"] == mechanism]
        direct = [row for row in mechanism_rows if row["prompt_style"] == "direct"]
        natural = [row for row in mechanism_rows if row["prompt_style"] == "natural"]
        direct_count = min(7, pass_count)
        natural_count = max(0, pass_count - direct_count)
        selected = {row["review_id"] for row in direct[:direct_count] + natural[:natural_count]}
        for row in mechanism_rows:
            row["decodable"] = "1"
            for field in review.ORDINAL_SCORE_FIELDS:
                row[field] = "2"
            if row["review_id"] not in selected:
                row["quality"] = "0"
            row["reviewer_notes"] = "SENSITIVE ROW-LEVEL NOTE THAT MUST NOT LEAK"
    return rows


def _write_review_csv(path: Path, rows) -> None:
    path.write_bytes(review.review_csv_bytes(rows))


def test_blank_template_is_exactly_bound_and_all_atomic_fields_are_blank():
    manifest = _manifest_rows()
    rows = review.build_review_rows(manifest)

    assert len(rows) == 192
    assert tuple(rows[0]) == review.REVIEW_FIELDS
    assert [row["review_id"] for row in rows] == [f"caprev{i:03d}" for i in range(192)]
    assert all(row["video_binding_key"] == row["generation_id"] for row in rows)
    assert all(len(row["manifest_row_sha256"]) == 64 for row in rows)
    assert all(
        not row[field]
        for row in rows
        for field in (*review.SCORE_FIELDS, "reviewer_notes")
    )
    assert {row["mechanism"] for row in rows} == set(review.MECHANISM_ORDER)
    assert len({row["generation_id"] for row in rows}) == 192


def test_rubric_freezes_full_frame_atomic_review_and_adjudication_policy():
    rubric = review.build_rubric_payload(_manifest_rows())

    assert rubric["video_contract"] == {
        "required_frame_count": 49,
        "required_fps": 8,
        "review_frame_indices_inclusive": [0, 48],
        "all_49_frames_must_be_inspected": True,
        "clean_prefix_frame_indices_inclusive": [0, 15],
        "post_prefix_frame_indices_inclusive": [16, 48],
        "missing_or_decode_failure_policy": "ineligible",
    }
    assert [field["name"] for field in rubric["atomic_fields"]] == list(
        review.SCORE_FIELDS
    )
    assert rubric["eligibility"]["required_values"] == {
        "decodable": "1",
        **{field: "2" for field in review.ORDINAL_SCORE_FIELDS},
    }
    assert rubric["review_workflow"]["reviewers_must_not_share_scores"] is True
    assert "third_reviewer" in rubric["review_workflow"]
    assert rubric["gate_output"]["granularity"] == "aggregate_only"
    assert rubric["video_binding"]["path_and_video_sha256_source"] == (
        "later_frozen_generation_manifest"
    )
    assert tuple(rubric["mechanisms"]) == review.MECHANISM_ORDER


def test_exact_15_with_style_and_identity_coverage_passes_aggregate_gate():
    blank = review.build_review_rows(_manifest_rows())
    rows = _filled_rows(pass_count=15)
    payload = review.score_adjudicated_rows(
        rows,
        blank,
        adjudicated_sha256="a" * 64,
        review_freeze_sha256="b" * 64,
    )

    assert payload["status"] == "pass"
    assert payload["aggregate"] == {
        "total_rows": 192,
        "eligible_rows": 120,
        "ineligible_rows": 72,
        "missing_atomic_rows": 0,
        "decode_fail_rows": 0,
        "mechanisms_total": 8,
        "mechanisms_passing": 8,
        "equal_mechanism_weight": "1/8",
        "pass": True,
    }
    for mechanism in review.MECHANISM_ORDER:
        result = payload["per_mechanism"][mechanism]
        assert result["eligible_rows"] == 15
        assert result["eligible_by_prompt_style"] == {"direct": 7, "natural": 8}
        assert result["distinct_eligible_source_ids"] >= 2
        assert result["distinct_eligible_receiver_ids"] >= 2
        assert result["pass"] is True
    serialized = json.dumps(payload)
    assert "SENSITIVE ROW-LEVEL NOTE" not in serialized
    assert "review_id" not in serialized
    assert "generation_id" not in serialized


def test_missing_or_decode_failure_is_ineligible_and_can_fail_one_mechanism():
    blank = review.build_review_rows(_manifest_rows())
    rows = _filled_rows(pass_count=15)
    first_water = next(row for row in rows if row["mechanism"] == "water_impact")
    first_water["quality"] = ""
    payload = review.score_adjudicated_rows(
        rows,
        blank,
        adjudicated_sha256="a" * 64,
        review_freeze_sha256="b" * 64,
    )
    assert payload["status"] == "fail"
    assert payload["aggregate"]["mechanisms_passing"] == 7
    assert payload["aggregate"]["missing_atomic_rows"] == 1
    assert payload["per_mechanism"]["water_impact"]["eligible_rows"] == 14

    rows = _filled_rows(pass_count=15)
    next(row for row in rows if row["mechanism"] == "water_impact")["decodable"] = "0"
    payload = review.score_adjudicated_rows(
        rows,
        blank,
        adjudicated_sha256="a" * 64,
        review_freeze_sha256="b" * 64,
    )
    assert payload["status"] == "fail"
    assert payload["aggregate"]["decode_fail_rows"] == 1
    assert payload["per_mechanism"]["water_impact"]["eligible_rows"] == 14


@pytest.mark.parametrize("mutation,match", [
    ("drop", "exactly 192"),
    ("reorder", "IDs/order"),
    ("metadata", "frozen metadata changed"),
    ("score", "invalid ordinal score"),
])
def test_adjudicated_review_fails_closed_on_coverage_or_schema_tamper(mutation, match):
    blank = review.build_review_rows(_manifest_rows())
    rows = _filled_rows()
    if mutation == "drop":
        rows.pop()
    elif mutation == "reorder":
        rows[0], rows[1] = rows[1], rows[0]
    elif mutation == "metadata":
        rows[0]["receiver_id"] = "changed"
    else:
        rows[0]["quality"] = "3"
    with pytest.raises(ValueError, match=match):
        review.validate_adjudicated_rows(rows, blank)


def test_freeze_cli_is_deterministic_hashed_and_refuses_overwrite(tmp_path):
    command = [
        sys.executable,
        str(SCRIPT_PATH),
        "freeze",
        "--canonical-manifest",
        str(MANIFEST_PATH),
        "--data-output-dir",
        str(tmp_path),
    ]
    first = subprocess.run(
        command, cwd=PROJECT_ROOT, text=True, capture_output=True, check=False
    )
    assert first.returncode == 0, first.stderr
    paths = review.artifact_paths(tmp_path)
    assert all(path.is_file() and not path.is_symlink() for path in paths.values())
    freeze = json.loads(paths["review_freeze"].read_text(encoding="utf-8"))
    assert freeze["artifacts"]["canonical_manifest"]["sha256"] == (
        review.EXPECTED_CANONICAL_MANIFEST_SHA256
    )
    assert freeze["artifacts"]["review_template"]["sha256"] == hashlib.sha256(
        paths["review_template"].read_bytes()
    ).hexdigest()
    assert freeze["artifacts"]["review_rubric"]["sha256"] == hashlib.sha256(
        paths["review_rubric"].read_bytes()
    ).hexdigest()
    before = {name: path.read_bytes() for name, path in paths.items()}
    second = subprocess.run(
        command, cwd=PROJECT_ROOT, text=True, capture_output=True, check=False
    )
    assert second.returncode != 0
    assert "refusing to overwrite existing artifact" in second.stderr
    assert {name: path.read_bytes() for name, path in paths.items()} == before


def test_checked_in_blank_artifacts_are_the_exact_deterministic_bytes():
    payloads, freeze = review.build_artifact_payloads(_manifest_rows())
    paths = review.artifact_paths(DATA_DIR)
    assert {name: path.read_bytes() for name, path in paths.items()} == payloads
    assert {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in paths.items()
    } == EXPECTED_REVIEW_ARTIFACT_SHA256
    assert json.loads(paths["review_freeze"].read_text(encoding="utf-8")) == freeze


def test_score_cli_verifies_freeze_outputs_no_row_data_and_refuses_overwrite(tmp_path):
    adjudicated = tmp_path / "canonical_adjudicated.csv"
    output = tmp_path / "aggregate_gate.json"
    _write_review_csv(adjudicated, _filled_rows())
    paths = review.artifact_paths(DATA_DIR)
    command = [
        sys.executable,
        str(SCRIPT_PATH),
        "score",
        "--canonical-manifest",
        str(MANIFEST_PATH),
        "--review-template",
        str(paths["review_template"]),
        "--review-rubric",
        str(paths["review_rubric"]),
        "--review-freeze",
        str(paths["review_freeze"]),
        "--canonical-adjudicated-review",
        str(adjudicated),
        "--output",
        str(output),
    ]
    first = subprocess.run(
        command, cwd=PROJECT_ROOT, text=True, capture_output=True, check=False
    )
    assert first.returncode == 0, first.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "pass"
    raw = output.read_text(encoding="utf-8")
    assert "SENSITIVE ROW-LEVEL NOTE" not in raw
    assert "caprev" not in raw
    before = output.read_bytes()
    second = subprocess.run(
        command, cwd=PROJECT_ROOT, text=True, capture_output=True, check=False
    )
    assert second.returncode != 0
    assert "refusing to overwrite existing artifact" in second.stderr
    assert output.read_bytes() == before


def test_scorer_rejects_rubric_drift_before_writing_output(tmp_path):
    adjudicated = tmp_path / "canonical_adjudicated.csv"
    changed_rubric = tmp_path / "rubric.json"
    output = tmp_path / "gate.json"
    _write_review_csv(adjudicated, _filled_rows())
    paths = review.artifact_paths(DATA_DIR)
    payload = json.loads(paths["review_rubric"].read_text(encoding="utf-8"))
    payload["mechanism_gate"]["minimum_eligible_rows"] = 14
    changed_rubric.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="rubric differs"):
        review.verify_frozen_review_inputs(
            _manifest_rows(),
            review_template_path=paths["review_template"],
            review_rubric_path=changed_rubric,
            review_freeze_path=paths["review_freeze"],
        )
    assert not output.exists()
