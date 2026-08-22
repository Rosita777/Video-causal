from __future__ import annotations

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

import build_causal_role_erasure_8mechanism_capability_v1 as capability  # noqa: E402
from run_pilot import parse_prompt_file  # noqa: E402


REGISTRY_PATH = PROJECT_ROOT / "data" / "protocol_v1" / "registry.json"
EXPECTED_CANONICAL_MANIFEST_SHA256 = (
    "6d425076a7156aabc9695e6ab4fbe7cf85922261d1282c067fd56d176d05031d"
)


def load_rows():
    registry = capability.load_protocol_v1_registry(REGISTRY_PATH)
    return capability.build_rows(registry)


def test_builds_balanced_eight_mechanism_original_capability_batch():
    rows = load_rows()

    assert len(rows) == 192
    assert Counter(row["mechanism"] for row in rows) == {
        mechanism: 24 for mechanism in capability.MECHANISM_ORDER
    }
    assert len({row["generation_id"] for row in rows}) == 192
    assert len({row["seed"] for row in rows}) == 192
    assert len({row["case_id"] for row in rows}) == 64
    assert {row["method_arm"] for row in rows} == {"original"}
    assert {row["intended_use"] for row in rows} == {
        "original_capability_screening_only"
    }
    assert {row["treatment_status"] for row in rows} == {
        "pre_method_original_only"
    }

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["mechanism"]].append(row)
    assert tuple(grouped) == capability.MECHANISM_ORDER
    for mechanism_rows in grouped.values():
        assert Counter(row["prompt_style"] for row in mechanism_rows) == {
            "direct": 12,
            "natural": 12,
        }
        assert len(
            {(row["source_id"], row["receiver_id"]) for row in mechanism_rows}
        ) == 8
        case_counts = Counter(row["case_id"] for row in mechanism_rows)
        assert len(case_counts) == 8
        assert set(case_counts.values()) == {3}


def test_reuses_frozen_protocol_v1_and_marks_new_ontologies_capability_only():
    registry = capability.load_protocol_v1_registry(REGISTRY_PATH)
    rows = capability.build_rows(registry)

    for mechanism in capability.PROTOCOL_V1_MECHANISMS:
        mechanism_rows = [row for row in rows if row["mechanism"] == mechanism]
        source_by_id = {
            item["id"]: item
            for key in ("train_sources", "test_sources")
            for item in registry["mechanisms"][mechanism][key]
        }
        receiver_by_id = {
            item["id"]: item
            for key in ("train_receivers", "test_receivers")
            for item in registry["mechanisms"][mechanism][key]
        }
        assert {row["ontology_status"] for row in mechanism_rows} == {
            capability.ONTOLOGY_STATUS_PROTOCOL_V1
        }
        for row in mechanism_rows:
            assert row["source_object"] == source_by_id[row["source_id"]]["name"]
            assert row["source_motion"] == source_by_id[row["source_id"]]["motion"]
            assert row["receiver"] == receiver_by_id[row["receiver_id"]]["name"]
            assert (
                row["receiver_clean_state"]
                == receiver_by_id[row["receiver_id"]]["clean_state"]
            )

    for mechanism in capability.DRAFT_MECHANISMS:
        mechanism_rows = [row for row in rows if row["mechanism"] == mechanism]
        assert {row["ontology_status"] for row in mechanism_rows} == {
            "draft_capability_only_not_training_ready"
        }
        assert all(
            row["source_family"] == row["receiver_family"]
            for row in mechanism_rows
        )

    field_rows = [
        row for row in rows if row["mechanism"] == "field_mediated_response"
    ]
    assert {row["source_family"] for row in field_rows} == {"electrostatic"}
    assert {row["receiver_family"] for row in field_rows} == {"electrostatic"}


def test_prompts_and_generation_settings_encode_the_capability_contract():
    rows = load_rows()

    for row in rows:
        assert row["num_frames"] == "49"
        assert row["fps"] == "8"
        assert row["reference_start_inclusive"] == "0"
        assert row["reference_end_exclusive"] == "16"
        assert "During the first two seconds" in row["prompt"]
        assert "The source object is not visible" in row["prompt"]
        assert row["source_object"] in row["prompt"]
        assert row["receiver"] in row["prompt"]
        assert row["expected_footprint"] in row["prompt"]
        assert "locked-camera" in row["prompt"]
        assert "No cuts, camera motion" in row["prompt"]

    field_rows = [
        row for row in rows if row["mechanism"] == "field_mediated_response"
    ]
    assert all("clear gap remains" in row["prompt"] for row in field_rows)
    assert all("no direct contact occurs" in row["prompt"] for row in field_rows)


def test_seed_formula_and_repetitions_are_exact_and_deterministic():
    rows = load_rows()
    by_case = defaultdict(list)
    for row in rows:
        by_case[row["case_id"]].append(row)
        assert int(row["seed"]) == capability.seed_for(
            int(row["mechanism_index"]),
            int(row["combination_index"]),
            int(row["repetition_index"]),
        )
        assert row["seed_formula"] == capability.SEED_FORMULA

    for case_rows in by_case.values():
        assert {int(row["repetition_index"]) for row in case_rows} == {0, 1, 2}
        assert len({row["prompt"] for row in case_rows}) == 1
        assert len({row["seed"] for row in case_rows}) == 3

    first = capability.canonical_json_bytes(rows)
    second = capability.canonical_json_bytes(load_rows())
    assert first == second
    assert hashlib.sha256(first).hexdigest() == EXPECTED_CANONICAL_MANIFEST_SHA256


def test_cli_writes_hashed_runner_compatible_artifacts_exclusively(tmp_path):
    data_dir = tmp_path / "data"
    prompts_dir = tmp_path / "prompts"
    command = [
        sys.executable,
        str(SCRIPTS_DIR / "build_causal_role_erasure_8mechanism_capability_v1.py"),
        "--protocol-v1-registry",
        str(REGISTRY_PATH),
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
    assert summary["counts"]["total_rows"] == 192
    assert summary["scope"] == {
        "intended_use": "original_capability_screening_only",
        "method_arm": "original",
        "treatment_status": "pre_method_original_only",
        "training_authorized": False,
        "evaluation_selection_authorized": False,
        "treatment_generation_authorized": False,
    }
    assert summary["canonical_manifest_sha256"] == EXPECTED_CANONICAL_MANIFEST_SHA256
    assert summary["artifact_sha256"]["canonical_manifest_json"] == (
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
    assert csv_rows == canonical_rows == load_rows()
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


def test_fail_closed_on_registry_drift_and_incompatible_draft_pair(tmp_path):
    changed_registry = tmp_path / "registry.json"
    payload = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    payload["video_frames"] = 48
    changed_registry.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="registry hash mismatch"):
        capability.load_protocol_v1_registry(changed_registry)

    registry = capability.load_protocol_v1_registry(REGISTRY_PATH)
    specs = capability.mechanism_specs(registry)
    specs["elastic_deformation"]["pair_indices"] = tuple(
        list(specs["elastic_deformation"]["pair_indices"][:-1]) + [(0, 2)]
    )
    with pytest.raises(ValueError, match="incompatible draft source/receiver pair"):
        capability.validate_mechanism_specs(specs)

    field_specs = capability.mechanism_specs(registry)
    field_specs["field_mediated_response"]["sources"][0]["family"] = "airflow"
    with pytest.raises(ValueError, match="purely electrostatic"):
        capability.validate_mechanism_specs(field_specs)
