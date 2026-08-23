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

import causal_role_erasure_8mechanism_capability_review_v2 as review  # noqa: E402
import build_causal_role_erasure_8mechanism_capability_v2 as capability  # noqa: E402


SCRIPT_PATH = SCRIPTS_DIR / "causal_role_erasure_8mechanism_capability_review_v2.py"
DATA_DIR = PROJECT_ROOT / "data"
MANIFEST_PATH = DATA_DIR / f"{review.ARTIFACT_STEM}_manifest.canonical.json"
EXPECTED_REVIEW_ARTIFACT_SHA256 = {
    "review_template": "4f8f3853a25931145d86e663d70ded54da0d162ce2d17174e5c9df7957b2820b",
    "review_rubric": "052232152cbf97a070f2479cbea2289955923d144fc48cc1420238080b2547ef",
    "review_freeze": "3c735ac020c63c62b15bab324258cd3d7a6a5bc457dd2da32793ea5bb29f92f0",
}


def _manifest_payload() -> list[dict[str, str]]:
    return deepcopy(capability.build_rows())


def _write_manifest(directory: Path) -> tuple[Path, str, list[dict[str, str]]]:
    rows = _manifest_payload()
    raw = review.canonical_json_bytes(rows)
    path = directory / f"{review.ARTIFACT_STEM}_manifest.canonical.json"
    path.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    assert digest == review.EXPECTED_CANONICAL_MANIFEST_SHA256
    return path, digest, rows


def _eligible_scores(*, clean_prefix: str = "0") -> dict[str, str]:
    return {
        "decodable": "1",
        "clean_prefix": clean_prefix,
        "source_after16": "2",
        "trigger_visible": "1",
        "footprint_after_trigger": "2",
        "receiver_recognizable": "1",
        "fixed_camera": "1",
        "quality": "1",
    }


def _filled_rows(
    manifest_rows: list[dict[str, str]],
    *,
    default_direct: int = 6,
    default_natural: int = 6,
    per_mechanism: dict[str, tuple[int, int]] | None = None,
) -> list[dict[str, str]]:
    def take_case_balanced(candidates: list[dict[str, str]], count: int):
        by_case: dict[str, list[dict[str, str]]] = {}
        for candidate in candidates:
            by_case.setdefault(candidate["case_id"], []).append(candidate)
        interleaved = [
            row
            for repetition in range(max(len(group) for group in by_case.values()))
            for group in by_case.values()
            for row in group[repetition : repetition + 1]
        ]
        return interleaved[:count]

    rows = deepcopy(review.build_review_rows(manifest_rows))
    overrides = per_mechanism or {}
    for mechanism in review.MECHANISM_ORDER:
        mechanism_rows = [row for row in rows if row["mechanism"] == mechanism]
        direct = [row for row in mechanism_rows if row["prompt_style"] == "direct"]
        natural = [row for row in mechanism_rows if row["prompt_style"] == "natural"]
        direct_count, natural_count = overrides.get(
            mechanism, (default_direct, default_natural)
        )
        selected = {
            row["review_id"]
            for row in take_case_balanced(direct, direct_count)
            + take_case_balanced(natural, natural_count)
        }
        for row in mechanism_rows:
            row.update(_eligible_scores(clean_prefix="0"))
            if row["review_id"] not in selected:
                row["quality"] = "0"
            row["reviewer_notes"] = "SENSITIVE ROW-LEVEL NOTE THAT MUST NOT LEAK"
    return rows


def _write_review_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.write_bytes(review.review_csv_bytes(rows))


def _freeze_artifacts(
    directory: Path,
) -> tuple[Path, str, list[dict[str, str]], dict[str, Path]]:
    manifest_path, manifest_sha256, manifest_rows = _write_manifest(directory)
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "freeze",
            "--canonical-manifest",
            str(manifest_path),
            "--canonical-manifest-sha256",
            manifest_sha256,
            "--data-output-dir",
            str(directory),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return manifest_path, manifest_sha256, manifest_rows, review.artifact_paths(directory)


def test_blank_template_preserves_exact_full_49_frame_atomic_schema():
    manifest = _manifest_payload()
    rows = review.build_review_rows(manifest)

    assert len(rows) == 192
    assert tuple(rows[0]) == review.REVIEW_FIELDS
    assert [row["review_id"] for row in rows] == [
        f"capv2rev{i:03d}" for i in range(192)
    ]
    assert review.SCORE_FIELDS == (
        "decodable",
        "clean_prefix",
        "source_after16",
        "trigger_visible",
        "footprint_after_trigger",
        "receiver_recognizable",
        "fixed_camera",
        "quality",
    )
    assert all(row["video_binding_key"] == row["generation_id"] for row in rows)
    assert all(
        not row[field]
        for row in rows
        for field in (*review.SCORE_FIELDS, "reviewer_notes")
    )


def test_checked_in_manifest_and_review_artifacts_match_exact_frozen_bytes():
    manifest_raw = MANIFEST_PATH.read_bytes()
    assert hashlib.sha256(manifest_raw).hexdigest() == (
        review.EXPECTED_CANONICAL_MANIFEST_SHA256
    )
    manifest_rows = review.load_manifest(
        MANIFEST_PATH,
        expected_sha256=review.EXPECTED_CANONICAL_MANIFEST_SHA256,
    )
    payloads, freeze = review.build_artifact_payloads(
        manifest_rows,
        canonical_manifest_name=MANIFEST_PATH.name,
        canonical_manifest_sha256=review.EXPECTED_CANONICAL_MANIFEST_SHA256,
    )
    paths = review.artifact_paths(DATA_DIR)

    assert {name: path.read_bytes() for name, path in paths.items()} == payloads
    assert {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in paths.items()
    } == EXPECTED_REVIEW_ARTIFACT_SHA256
    assert json.loads(paths["review_freeze"].read_text(encoding="utf-8")) == freeze


def test_public_rubric_discloses_amendment_partial_rule_and_formal_alignment():
    rubric = review.build_rubric_payload(_manifest_payload())

    assert rubric["protocol_amendment"]["status"] == (
        "post_hoc_pre_treatment_amendment"
    )
    assert rubric["protocol_amendment"]["v1_is_unchanged"] is True
    assert rubric["protocol_amendment"][
        "treatment_outputs_may_not_be_inspected_before_freeze"
    ] is True
    assert rubric["protocol_amendment"]["field_mediated_subtype"] == (
        "airflow_non_contact_motion"
    )
    assert rubric["protocol_amendment"]["elastic_receiver_scope"] == (
        "black_round_and_blue_square_trampolines"
    )
    assert [field["name"] for field in rubric["atomic_fields"]] == list(
        review.SCORE_FIELDS
    )
    assert rubric["eligibility"]["diagnostic_only_fields"] == ["clean_prefix"]
    assert rubric["eligibility"]["partial_effect_accepted"] is True
    assert rubric["eligibility"]["all_fields_equal_2_required"] is False
    assert rubric["eligibility"]["coupled_requirement"] == {
        "fields": ["trigger_visible", "footprint_after_trigger"],
        "operator": "sum_at_least",
        "value": 3,
        "purpose": "exclude rows where both causal trigger and effect are ambiguous",
    }
    assert rubric["formal_metric_alignment"]["formal_E_b_unchanged"] is True
    assert rubric["formal_metric_alignment"]["formal_CES_unchanged"] is True
    assert rubric["formal_metric_alignment"]["continuous_primary_endpoint"] is True
    assert {
        mechanism: rubric["mechanisms"][mechanism]["trigger_definition"]
        for mechanism in review.MECHANISM_ORDER
    } == {
        mechanism: capability.MECHANISM_SPECS[mechanism]["trigger"]
        for mechanism in review.MECHANISM_ORDER
    }
    field = rubric["mechanisms"]["field_mediated_response"]
    assert field["subtype"] == "airflow_non_contact_motion"
    assert field["trigger_definition"] == (
        "visible fan blades begin spinning while a clear gap remains"
    )
    assert field["expected_footprint_by_receiver_id"] == {
        "v2_field_red_pinwheel": (
            "the red pinwheel spins rapidly in the airflow while a clear gap remains"
        ),
        "v2_field_yellow_ribbon": (
            "the yellow ribbon flutters strongly in the airflow while a clear gap remains"
        ),
    }
    elastic = rubric["mechanisms"]["elastic_deformation"]
    assert elastic["receiver_scope"] == "black_round_and_blue_square_trampolines"
    serialized = json.dumps(rubric).lower()
    assert "magnet" not in serialized
    assert rubric["mechanism_gate"] == {
        "total_rows_required": 24,
        "minimum_eligible_rows": 12,
        "minimum_eligible_direct_rows": 6,
        "minimum_eligible_natural_rows": 6,
        "minimum_distinct_eligible_source_ids": 2,
        "minimum_distinct_eligible_receiver_ids": 2,
        "all_eight_mechanisms_must_pass": True,
    }


@pytest.mark.parametrize("clean_prefix", ["0", "1", "2"])
def test_clean_prefix_value_is_diagnostic_only(clean_prefix: str):
    scores = _eligible_scores(clean_prefix=clean_prefix)
    assert review.row_is_eligible(scores) is True


@pytest.mark.parametrize(
    "mutation,eligible",
    [
        ({"trigger_visible": "1", "footprint_after_trigger": "1"}, False),
        ({"trigger_visible": "1", "footprint_after_trigger": "2"}, True),
        ({"trigger_visible": "2", "footprint_after_trigger": "1"}, True),
        ({"source_after16": "1"}, False),
        ({"receiver_recognizable": "0"}, False),
        ({"fixed_camera": "0"}, False),
        ({"quality": "0"}, False),
        ({"decodable": "0"}, False),
        ({"clean_prefix": ""}, False),
    ],
)
def test_row_eligibility_exact_thresholds_and_complete_record(mutation, eligible):
    scores = _eligible_scores()
    scores.update(mutation)
    assert review.row_is_eligible(scores) is eligible


def test_exact_12_with_six_per_style_and_coverage_passes_aggregate_gate():
    manifest = _manifest_payload()
    blank = review.build_review_rows(manifest)
    rows = _filled_rows(manifest)
    payload = review.score_adjudicated_rows(
        rows,
        blank,
        canonical_manifest_sha256="c" * 64,
        adjudicated_sha256="a" * 64,
        review_freeze_sha256="b" * 64,
    )

    assert payload["status"] == "pass"
    assert payload["aggregate"] == {
        "total_rows": 192,
        "eligible_rows": 96,
        "ineligible_rows": 96,
        "missing_atomic_rows": 0,
        "decode_fail_rows": 0,
        "mechanisms_total": 8,
        "mechanisms_passing": 8,
        "equal_mechanism_weight": "1/8",
        "pass": True,
    }
    for mechanism in review.MECHANISM_ORDER:
        result = payload["per_mechanism"][mechanism]
        assert result["eligible_rows"] == 12
        assert result["eligible_by_prompt_style"] == {"direct": 6, "natural": 6}
        assert result["distinct_eligible_source_ids"] >= 2
        assert result["distinct_eligible_receiver_ids"] >= 2
        assert result["diagnostics"] == {
            "clean_prefix_distribution": {
                "0": 24,
                "1": 0,
                "2": 0,
                "missing": 0,
            },
            "clean_prefix_enters_gate": False,
        }
        assert result["pass"] is True
    serialized = json.dumps(payload)
    assert "SENSITIVE ROW-LEVEL NOTE" not in serialized
    assert "capv2rev" not in serialized
    assert "generation_id" not in serialized


def test_total_12_does_not_override_six_per_style_requirement():
    manifest = _manifest_payload()
    blank = review.build_review_rows(manifest)
    rows = _filled_rows(
        manifest,
        per_mechanism={"rigid_collision": (5, 7)},
    )
    payload = review.score_adjudicated_rows(
        rows,
        blank,
        canonical_manifest_sha256="c" * 64,
        adjudicated_sha256="a" * 64,
        review_freeze_sha256="b" * 64,
    )

    assert payload["status"] == "fail"
    assert payload["aggregate"]["mechanisms_passing"] == 7
    rigid = payload["per_mechanism"]["rigid_collision"]
    assert rigid["eligible_rows"] == 12
    assert rigid["eligible_by_prompt_style"] == {"direct": 5, "natural": 7}
    assert rigid["gates"]["eligible_rows_at_least_12"] is True
    assert rigid["gates"]["eligible_direct_rows_at_least_6"] is False


def test_missing_atomic_or_decode_failure_is_ineligible():
    manifest = _manifest_payload()
    blank = review.build_review_rows(manifest)

    rows = _filled_rows(manifest)
    first_water = next(row for row in rows if row["mechanism"] == "water_impact")
    first_water["clean_prefix"] = ""
    payload = review.score_adjudicated_rows(
        rows,
        blank,
        canonical_manifest_sha256="c" * 64,
        adjudicated_sha256="a" * 64,
        review_freeze_sha256="b" * 64,
    )
    assert payload["status"] == "fail"
    assert payload["aggregate"]["missing_atomic_rows"] == 1
    assert payload["per_mechanism"]["water_impact"]["eligible_rows"] == 11

    rows = _filled_rows(manifest)
    next(row for row in rows if row["mechanism"] == "water_impact")[
        "decodable"
    ] = "0"
    payload = review.score_adjudicated_rows(
        rows,
        blank,
        canonical_manifest_sha256="c" * 64,
        adjudicated_sha256="a" * 64,
        review_freeze_sha256="b" * 64,
    )
    assert payload["aggregate"]["decode_fail_rows"] == 1
    assert payload["per_mechanism"]["water_impact"]["eligible_rows"] == 11


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("drop", "exactly 192"),
        ("reorder", "IDs/order"),
        ("metadata", "frozen metadata changed"),
        ("score", "invalid ordinal score"),
    ],
)
def test_adjudicated_review_fails_closed_on_schema_or_metadata_tamper(
    mutation, match
):
    manifest = _manifest_payload()
    blank = review.build_review_rows(manifest)
    rows = _filled_rows(manifest)
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


def test_freeze_binds_manifest_hash_is_deterministic_and_refuses_overwrite(tmp_path):
    manifest_path, manifest_sha256, _, paths = _freeze_artifacts(tmp_path)
    freeze = json.loads(paths["review_freeze"].read_text(encoding="utf-8"))

    assert freeze["artifacts"]["canonical_manifest"] == {
        "name": manifest_path.name,
        "sha256": manifest_sha256,
    }
    assert freeze["checks"]["canonical_manifest_sha256_bound"] is True
    assert freeze["artifacts"]["review_template"]["sha256"] == hashlib.sha256(
        paths["review_template"].read_bytes()
    ).hexdigest()
    assert freeze["artifacts"]["review_rubric"]["sha256"] == hashlib.sha256(
        paths["review_rubric"].read_bytes()
    ).hexdigest()
    before = {name: path.read_bytes() for name, path in paths.items()}

    second = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "freeze",
            "--canonical-manifest",
            str(manifest_path),
            "--canonical-manifest-sha256",
            manifest_sha256,
            "--data-output-dir",
            str(tmp_path),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert second.returncode != 0
    assert "refusing to overwrite existing artifact" in second.stderr
    assert {name: path.read_bytes() for name, path in paths.items()} == before


def test_freeze_rejects_wrong_manifest_hash_before_any_artifact_write(tmp_path):
    manifest_path, _, _ = _write_manifest(tmp_path)
    output_dir = tmp_path / "review"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "freeze",
            "--canonical-manifest",
            str(manifest_path),
            "--canonical-manifest-sha256",
            "0" * 64,
            "--data-output-dir",
            str(output_dir),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "manifest hash mismatch" in result.stderr
    assert not output_dir.exists()


def test_score_cli_verifies_freeze_outputs_only_aggregate_and_refuses_overwrite(
    tmp_path,
):
    manifest_path, manifest_sha256, manifest_rows, paths = _freeze_artifacts(tmp_path)
    adjudicated = tmp_path / "canonical_adjudicated_v2.csv"
    output = tmp_path / "aggregate_gate_v2.json"
    _write_review_csv(adjudicated, _filled_rows(manifest_rows))

    command = [
        sys.executable,
        str(SCRIPT_PATH),
        "score",
        "--canonical-manifest",
        str(manifest_path),
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
    assert payload["input_sha256"]["canonical_manifest"] == manifest_sha256
    raw = output.read_text(encoding="utf-8")
    assert "SENSITIVE ROW-LEVEL NOTE" not in raw
    assert "capv2rev" not in raw
    before = output.read_bytes()

    second = subprocess.run(
        command, cwd=PROJECT_ROOT, text=True, capture_output=True, check=False
    )
    assert second.returncode != 0
    assert "refusing to overwrite existing artifact" in second.stderr
    assert output.read_bytes() == before


def test_scorer_rejects_rubric_drift_against_frozen_bytes(tmp_path):
    manifest_path, manifest_sha256, manifest_rows, paths = _freeze_artifacts(tmp_path)
    changed_rubric = tmp_path / "changed_rubric.json"
    payload = json.loads(paths["review_rubric"].read_text(encoding="utf-8"))
    payload["mechanism_gate"]["minimum_eligible_rows"] = 11
    changed_rubric.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="rubric differs"):
        review.verify_frozen_review_inputs(
            manifest_rows,
            canonical_manifest_name=manifest_path.name,
            canonical_manifest_sha256=manifest_sha256,
            review_template_path=paths["review_template"],
            review_rubric_path=changed_rubric,
            review_freeze_path=paths["review_freeze"],
        )
