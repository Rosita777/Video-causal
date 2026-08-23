from __future__ import annotations

import copy
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import subprocess
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import build_causal_role_erasure_8mechanism_capability_v2 as capability  # noqa: E402
from run_pilot import parse_prompt_file  # noqa: E402


EXPECTED_CANONICAL_MANIFEST_SHA256 = (
    "8e8cc14a6d327193cb7f270d2386490dc00ee783d9e7d0f76a697837f218dbee"
)


def test_builds_fresh_balanced_v2_batch_with_new_ids_and_seeds():
    rows = capability.build_rows()

    assert len(rows) == 192
    assert len({row["case_id"] for row in rows}) == 64
    assert len({row["generation_id"] for row in rows}) == 192
    assert len({row["seed"] for row in rows}) == 192
    assert {row["protocol_version"] for row in rows} == {
        "causal_role_erasure_8mechanism_capability_v2"
    }
    assert {row["ontology_status"] for row in rows} == {"fresh_capability_v2"}
    assert {row["ontology_provenance"] for row in rows} == {
        "causal_role_erasure_8mechanism_capability_v2:fresh_simplified_ontology"
    }
    assert all(row["case_id"].startswith("cap8v2") for row in rows)
    assert all(row["generation_id"].startswith("cap8v2") for row in rows)
    assert all(row["source_id"].startswith("v2_") for row in rows)
    assert all(row["receiver_id"].startswith("v2_") for row in rows)
    assert {row["method_arm"] for row in rows} == {"original"}
    assert min(int(row["seed"]) for row in rows) == 940000

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["mechanism"]].append(row)
    assert tuple(grouped) == capability.MECHANISM_ORDER
    for mechanism_rows in grouped.values():
        assert len(mechanism_rows) == 24
        assert Counter(row["prompt_style"] for row in mechanism_rows) == {
            "direct": 12,
            "natural": 12,
        }
        assert len({row["source_id"] for row in mechanism_rows}) == 2
        assert len({row["receiver_id"] for row in mechanism_rows}) == 2
        assert len(
            {(row["source_id"], row["receiver_id"]) for row in mechanism_rows}
        ) == 4
        case_counts = Counter(row["case_id"] for row in mechanism_rows)
        assert len(case_counts) == 8
        assert set(case_counts.values()) == {3}
        for style in capability.PROMPT_STYLES:
            style_rows = [
                row for row in mechanism_rows if row["prompt_style"] == style
            ]
            assert len(
                {(row["source_id"], row["receiver_id"]) for row in style_rows}
            ) == 4
            assert set(Counter(row["source_id"] for row in style_rows).values()) == {
                6
            }
            assert set(
                Counter(row["receiver_id"] for row in style_rows).values()
            ) == {6}


def test_protocol_amendment_and_prompt_contract_are_explicit_and_enforced():
    rows = capability.build_rows()
    amendment = capability.PROTOCOL_AMENDMENT

    assert amendment["fresh_ontology_for_all_eight_mechanisms"] is True
    assert amendment["protocol_v1_registry_reused"] is False
    assert amendment["prompt_sentence_count_exact"] == 2
    assert amendment["prompt_max_english_words"] == 55
    assert amendment["exact_initial_timing_required"] is False
    assert amendment["source_offscreen_clause_required"] is False
    assert amendment["field_mediated_subtype"] == "airflow_non_contact_motion"
    assert amendment["reference_frames_0_15"] == (
        "diagnostic_only_not_v2_eligibility"
    )

    prompts = {row["prompt"] for row in rows}
    assert len(prompts) == 64
    assert max(capability.english_word_count(prompt) for prompt in prompts) <= 55
    for row in rows:
        sentences = capability.prompt_sentences(row["prompt"])
        assert len(sentences) == 2
        assert row["source_object"].lower() not in sentences[0].lower()
        assert row["source_object"].lower() in sentences[1].lower()
        assert row["receiver"].lower() in row["prompt"].lower()
        assert row["expected_footprint"].lower() in sentences[1].lower()
        assert capability.FORBIDDEN_HUMAN_TERMS.search(row["prompt"]) is None
        assert capability.FORBIDDEN_NEGATION_TERMS.search(row["prompt"]) is None
        assert "first two seconds" not in row["prompt"].lower()
        assert "offscreen" not in row["prompt"].lower()
        assert row["num_frames"] == "49"
        assert row["fps"] == "8"
        assert row["reference_start_inclusive"] == "0"
        assert row["reference_end_exclusive"] == "16"


def test_field_mechanism_is_fresh_airflow_non_contact_only():
    rows = [
        row
        for row in capability.build_rows()
        if row["mechanism"] == "field_mediated_response"
    ]

    assert {row["source_family"] for row in rows} == {"v2_airflow_non_contact"}
    assert {row["receiver_family"] for row in rows} == {"v2_airflow_non_contact"}
    assert {row["source_id"] for row in rows} == {
        "v2_field_blue_desk_fan",
        "v2_field_black_desk_fan",
    }
    assert {row["receiver_id"] for row in rows} == {
        "v2_field_red_pinwheel",
        "v2_field_yellow_ribbon",
    }
    assert all("fan" in row["prompt"].lower() for row in rows)
    assert all("blades" in row["prompt"].lower() for row in rows)
    assert all("airflow" in row["prompt"].lower() for row in rows)
    assert all("clear gap remains" in row["prompt"].lower() for row in rows)
    assert all(capability.english_word_count(row["prompt"]) <= 42 for row in rows)
    assert all("five centimeters" not in row["prompt"].lower() for row in rows)
    assert all(
        row["prompt"].lower().count(row["receiver"].lower()) == 1 for row in rows
    )
    assert all(
        term not in row["prompt"].lower()
        for row in rows
        for term in ("magnet", "electrostatic", "charged", "balloon", "comb")
    )


def test_smoke_feedback_replacements_are_frozen_in_v2_ontology():
    rigid = capability.MECHANISM_SPECS["rigid_collision"]
    assert {source["id"] for source in rigid["sources"]} == {
        "v2_collision_red_ball",
        "v2_collision_blue_ball",
    }
    assert all("rubber ball" in source["name"] for source in rigid["sources"])
    assert all("cube" not in source["name"] for source in rigid["sources"])

    fracture = capability.MECHANISM_SPECS["brittle_fracture"]
    assert {receiver["id"] for receiver in fracture["receivers"]} == {
        "v2_fracture_glass_bowl",
        "v2_fracture_glass_pane",
    }
    assert all("wine glass" not in receiver["name"] for receiver in fracture["receivers"])

    elastic = capability.MECHANISM_SPECS["elastic_deformation"]
    assert {receiver["id"] for receiver in elastic["receivers"]} == {
        "v2_elastic_black_trampoline",
        "v2_elastic_blue_square_trampoline",
    }
    assert all("trampoline" in receiver["name"] for receiver in elastic["receivers"])
    assert all("foam" not in receiver["name"] for receiver in elastic["receivers"])


def test_seed_formula_repetitions_and_canonical_hash_are_deterministic():
    rows = capability.build_rows()
    by_case = defaultdict(list)
    for row in rows:
        by_case[row["case_id"]].append(row)
        assert int(row["seed"]) == capability.seed_for(
            int(row["mechanism_index"]),
            int(row["combination_index"]),
            int(row["repetition_index"]),
        )
        assert row["seed_formula"] == (
            "940000 + 1000*mechanism_index + 10*combination_index + repetition_index"
        )

    assert all(
        {int(row["repetition_index"]) for row in case_rows} == {0, 1, 2}
        for case_rows in by_case.values()
    )
    first = capability.canonical_json_bytes(rows)
    second = capability.canonical_json_bytes(capability.build_rows())
    assert first == second
    assert hashlib.sha256(first).hexdigest() == EXPECTED_CANONICAL_MANIFEST_SHA256


def test_cli_writes_hashed_artifacts_exclusively_and_does_not_overwrite(
    tmp_path: Path,
):
    data_dir = tmp_path / "data"
    prompts_dir = tmp_path / "prompts"
    command = [
        sys.executable,
        str(SCRIPTS_DIR / "build_causal_role_erasure_8mechanism_capability_v2.py"),
        "--data-output-dir",
        str(data_dir),
        "--prompts-output-dir",
        str(prompts_dir),
    ]

    first = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert first.returncode == 0, first.stderr

    paths = capability.artifact_paths(data_dir, prompts_dir)
    assert all(path.is_file() for path in paths.values())
    summary = json.loads(paths["summary_json"].read_text(encoding="utf-8"))
    assert summary["protocol_amendment"] == capability.PROTOCOL_AMENDMENT
    assert summary["counts"]["total_cases"] == 64
    assert summary["counts"]["total_rows"] == 192
    assert summary["video"]["num_frames"] == 49
    assert summary["video"]["fps"] == 8
    assert summary["video"]["diagnostic_reference_role"] == (
        "diagnostic_only_not_v2_eligibility"
    )
    assert summary["recommended_aggregate_gate"] == {
        "rows_per_mechanism": 24,
        "minimum_eligible_rows": 12,
        "minimum_eligible_direct_rows": 6,
        "minimum_eligible_natural_rows": 6,
        "minimum_distinct_eligible_source_ids": 2,
        "minimum_distinct_eligible_receiver_ids": 2,
        "all_eight_mechanisms_must_pass": True,
        "clean_prefix_is_eligibility_requirement": False,
    }
    assert summary["canonical_manifest_sha256"] == (
        EXPECTED_CANONICAL_MANIFEST_SHA256
    )
    for name in ("manifest_csv", "canonical_manifest_json", "prompts"):
        assert summary["artifact_sha256"][name] == hashlib.sha256(
            paths[name].read_bytes()
        ).hexdigest()

    with paths["manifest_csv"].open(newline="", encoding="utf-8") as handle:
        csv_rows = list(csv.DictReader(handle))
    canonical_rows = json.loads(
        paths["canonical_manifest_json"].read_text(encoding="utf-8")
    )
    assert csv_rows == canonical_rows == capability.build_rows()
    parsed_prompts = parse_prompt_file(paths["prompts"])
    assert len(parsed_prompts) == 192
    assert [item["prompt"] for item in parsed_prompts] == [
        row["prompt"] for row in csv_rows
    ]

    before = {name: path.read_bytes() for name, path in paths.items()}
    second = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert second.returncode != 0
    assert "refusing to overwrite existing artifact" in second.stderr
    assert {name: path.read_bytes() for name, path in paths.items()} == before


def test_validation_fails_closed_on_prompt_or_airflow_ontology_drift():
    rows = capability.build_rows()
    bad_sentence = dict(rows[0])
    bad_sentence["prompt"] = "One sentence only."
    with pytest.raises(ValueError, match="exactly two sentences"):
        capability.validate_rows([bad_sentence] + rows[1:])

    source = capability.MECHANISM_SPECS["water_impact"]["sources"][0]
    receiver = capability.MECHANISM_SPECS["water_impact"]["receivers"][0]
    too_long = (
        "Locked-camera close-up: "
        + "stable " * 54
        + ". Then one small red apple falls straight down; one splash and expanding circular ripples appear."
    )
    with pytest.raises(ValueError, match="exceeds 55 English words"):
        capability.validate_prompt(
            too_long,
            source=source,
            receiver=receiver,
            footprint="one splash and expanding circular ripples appear",
        )

    bad_specs = copy.deepcopy(capability.MECHANISM_SPECS)
    bad_specs["field_mediated_response"]["sources"][0]["family"] = (
        "v2_magnetic_non_contact"
    )
    with pytest.raises(ValueError, match="compatibility family mismatch"):
        capability.validate_mechanism_specs(bad_specs)
