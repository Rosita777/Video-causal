from __future__ import annotations

import hashlib
import json
import shutil
import stat
from pathlib import Path

import pytest
from PIL import Image

from scripts import canonicalize_causal_role_erasure_7mechanism_formal_v1 as canonical
from scripts import (
    build_causal_role_erasure_7mechanism_provisional_consensus_v1 as provisional,
)
from scripts import review_causal_role_erasure_7mechanism_formal_v2 as transport


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(provisional.canonical_json_bytes(row) for row in rows))


def _ref(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "sha256": provisional.sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _score_row(assignment, values):
    fields = transport.SCORE_FIELDS_BY_KIND[assignment["case_kind"]]
    return {
        "anonymous_review_id": assignment["anonymous_review_id"],
        "assignment_sha256": assignment["assignment_sha256"],
        "case_kind": assignment["case_kind"],
        "scores": dict(zip(fields, values)),
        "confidence": {field: 0.9 for field in fields},
        "evidence_frames": {field: [12] for field in fields},
        "evidence_observations": {field: [f"evidence for {field}"] for field in fields},
        "unusable_reason": "",
        "status": "completed",
    }


def _fixture(tmp_path: Path):
    public = tmp_path / "review_package" / "public"
    public.mkdir(parents=True)
    registry = provisional.evaluation_code.build_registry(PROJECT_ROOT)
    registry_path = public / "evaluation_code_registry.json"
    registry_path.write_bytes(provisional.canonical_json_bytes(registry))
    commitments = {
        "blind_key_sha256": "0" * 64,
        "commitment_scheme": "sha256(canonical-jsonl-bytes)",
        "evaluation_code_registry": {
            "path": "evaluation_code_registry.json",
            "registry_sha256": registry["registry_sha256"],
            "sha256": provisional.sha256_file(registry_path),
        },
        "generation_ledger": {"row_count": 2, "sha256": "1" * 64},
        "protocol": canonical.REVIEW_PROTOCOL,
        "schema_version": 1,
        "tier_0_audit_strata": {"row_count": 2, "sha256": "2" * 64},
        "tier_1_original_only": {"row_count": 588, "sha256": "3" * 64},
        "tier_2_full": {"row_count": 2, "sha256": "4" * 64},
    }
    commitments_path = public / "key_commitments.json"
    commitments_path.write_bytes(provisional.canonical_json_bytes(commitments))

    assignment_specs = [
        {
            "anonymous_review_id": "rv_causal",
            "case_kind": "causal",
            "mechanism_name": "water impact",
            "prompt": "a droplet strikes water",
            "source_object": "droplet",
            "receiver": "pool",
            "expected_trigger": "impact",
            "expected_footprint": "ripples",
            "expected_counterfactual_state": "calm pool",
            "protected_object": "",
            "specificity_subtype": "",
            "acceptable_alternative_cause": "",
        },
        {
            "anonymous_review_id": "rv_specificity",
            "case_kind": "specificity",
            "mechanism_name": "water impact",
            "prompt": "a protected ball crosses a wet scene",
            "source_object": "droplet",
            "receiver": "pool",
            "expected_trigger": "impact",
            "expected_footprint": "ripples",
            "expected_counterfactual_state": "protected event remains",
            "protected_object": "ball",
            "specificity_subtype": "same_noun_noncausal",
            "acceptable_alternative_cause": "",
        },
    ]
    score_values = {
        "A": {
            "rv_causal": (0, 1, 2, 2),
            "rv_specificity": (2, 0, 1, 2),
        },
        "B": {
            "rv_causal": (1, 2, 2, 0),
            "rv_specificity": (0, 1, 2, 1),
        },
    }
    merged_paths = {}
    assignments_by_pass = {}
    for pass_name, pass_id, specs in (
        ("pass_a", "A", assignment_specs),
        ("pass_b", "B", list(reversed(assignment_specs))),
    ):
        pass_root = public / pass_name
        media = pass_root / "media"
        media.mkdir(parents=True)
        assignments = []
        blanks = []
        for spec in specs:
            review_id = spec["anonymous_review_id"]
            image_path = media / f"{review_id}.jpg"
            Image.new(
                "RGB",
                (transport.COMPOSITE_WIDTH, transport.COMPOSITE_HEIGHT),
                color=(30 if review_id == "rv_causal" else 50, 60, 90),
            ).save(image_path, format="JPEG", quality=transport.JPEG_QUALITY)
            assignment = {
                **spec,
                "composite_path": f"media/{review_id}.jpg",
                "composite_sha256": transport.sha256_file(image_path),
                "assignment_sha256": "",
            }
            assignment["assignment_sha256"] = transport._assignment_digest(assignment)
            assignments.append(assignment)
            fields = transport.SCORE_FIELDS_BY_KIND[assignment["case_kind"]]
            blanks.append(
                {
                    "anonymous_review_id": review_id,
                    "assignment_sha256": assignment["assignment_sha256"],
                    "case_kind": assignment["case_kind"],
                    "scores": {field: None for field in fields},
                    "confidence": {field: None for field in fields},
                    "evidence_frames": {field: [] for field in fields},
                    "evidence_observations": {field: [] for field in fields},
                    "unusable_reason": "",
                    "status": "pending",
                }
            )
        assignments_path = pass_root / "assignments.jsonl"
        blank_path = pass_root / "scores.jsonl"
        _jsonl(assignments_path, assignments)
        _jsonl(blank_path, blanks)
        pass_manifest = {
            "protocol": transport.PACKAGE_PROTOCOL,
            "schema_version": 1,
            "pass_id": pass_id,
            "item_count": 2,
            "ordering_commitment": ("a" if pass_id == "A" else "b") * 64,
            "panel_windows": transport.PANEL_WINDOWS,
            "assignments": {
                "path": "assignments.jsonl",
                "sha256": transport.sha256_file(assignments_path),
            },
            "blank_scores": {
                "path": "scores.jsonl",
                "sha256": transport.sha256_file(blank_path),
            },
            "composite_contract": {
                "path_base": "pass_root",
                "directory": "media",
                "format": "jpeg",
                "width": transport.COMPOSITE_WIDTH,
                "height": transport.COMPOSITE_HEIGHT,
                "tile_width": transport.TILE_WIDTH,
                "tile_height": transport.TILE_HEIGHT,
                "tile_fit": "deterministic_center_crop_no_letterbox",
                "resampling": "Pillow.Image.Resampling.LANCZOS",
                "jpeg_quality": transport.JPEG_QUALITY,
                "max_bytes": transport.MAX_IMAGE_BYTES,
                "one_image_per_assignment": True,
            },
        }
        pass_manifest_path = pass_root / "pass_manifest.json"
        pass_manifest_path.write_bytes(provisional.canonical_json_bytes(pass_manifest))
        assignments_by_pass[pass_id] = assignments_path

        merge_root = tmp_path / "merged" / pass_name
        merge_root.mkdir(parents=True)
        completed = [
            _score_row(row, score_values[pass_id][row["anonymous_review_id"]])
            for row in assignments
        ]
        completed_path = merge_root / "completed_scores.jsonl"
        _jsonl(completed_path, completed)
        run_root = merge_root / "source_run"
        run_root.mkdir()
        registration = run_root / "run_registration.json"
        summary = run_root / "run_summary.json"
        registration.write_text('{"status":"completed"}\n', encoding="utf-8")
        summary.write_text('{"status":"completed"}\n', encoding="utf-8")
        raw_registry = merge_root / "raw_response_registry.json"
        raw_registry.write_bytes(
            provisional.canonical_json_bytes(
                {
                    row["anonymous_review_id"]: {
                        "model": "gpt-5.6-luna",
                        "observed_model": "gpt-5.6-luna",
                    }
                    for row in completed
                }
            )
        )
        merge_manifest = {
            "schema_version": 1,
            "workflow_version": canonical.TRANSPORT_WORKFLOW,
            "status": "complete_schema_valid_public_pass_review",
            "pass_id": pass_id,
            "evaluation_code_registry_sha256": registry["registry_sha256"],
            "row_count": 2,
            "model": "gpt-5.6-luna",
            "reasoning_effort": "low",
            "temperature": 1.0,
            "max_output_tokens": 1600,
            "store": False,
            "proxy_version": "synthetic-test-proxy",
            "endpoint": "http://127.0.0.1:4141/v1/responses",
            "pass_manifest": _ref(pass_manifest_path),
            "assignments": _ref(assignments_path),
            "blank_scores": _ref(blank_path),
            "source_runs": [
                {
                    "run_registration": _ref(registration),
                    "run_summary": _ref(summary),
                }
            ],
            "completed_scores": {
                "path": "completed_scores.jsonl",
                "sha256": provisional.sha256_file(completed_path),
            },
            "raw_response_registry": _ref(raw_registry),
            "scientific_zero_fallbacks": 0,
        }
        (merge_root / "merge_manifest.json").write_bytes(
            provisional.canonical_json_bytes(merge_manifest)
        )
        merged_paths[pass_id] = completed_path
    return {
        "scores_a": merged_paths["A"],
        "scores_b": merged_paths["B"],
        "assignments": assignments_by_pass["A"],
        "assignments_b": assignments_by_pass["B"],
        "commitments": commitments_path,
        "public_root": public,
    }


def test_post_unblinding_export_is_fresh_bound_and_exact_mean(tmp_path: Path):
    fixture = _fixture(tmp_path)
    output = tmp_path / "provisional"
    receipt = provisional.build_provisional_consensus(
        scores_a_path=fixture["scores_a"],
        scores_b_path=fixture["scores_b"],
        assignments_path=fixture["assignments"],
        key_commitments_path=fixture["commitments"],
        output_root=output,
        expected_items=2,
    )

    assert receipt["status"] == provisional.STATUS
    assert receipt["scientific_label"] == "POST_UNBLINDING_EXPLORATORY_PRELIMINARY"
    assert receipt["global_unblinding_state"] == provisional.GLOBAL_UNBLINDING_STATE
    assert receipt["protocol_deviation"]["occurred"] is True
    assert receipt["counts"] == {
        "videos": 2,
        "atomic_scores": 8,
        "pass_a_rows": 2,
        "pass_b_rows": 2,
    }
    assert receipt["key_and_metric_boundaries"] == {
        "full_method_key_is_an_input": False,
        "full_method_key_read_by_this_builder": False,
        "method_identities_used_in_consensus": False,
        "global_full_method_key_access_had_already_occurred": True,
        "original_only_key_is_an_input": False,
        "method_metrics_computed": False,
        "canonical_scores_read_or_written": False,
    }

    decision_path = output / "provisional_decision.json"
    scores_path = output / "provisional_scores.jsonl"
    receipt_path = output / "provisional_receipt.json"
    snapshot_path = output / "implementation_snapshot.py"
    assert stat.S_IMODE(decision_path.stat().st_mode) == 0o444
    assert stat.S_IMODE(scores_path.stat().st_mode) == 0o444
    assert stat.S_IMODE(receipt_path.stat().st_mode) == 0o444
    assert stat.S_IMODE(snapshot_path.stat().st_mode) == 0o444
    assert snapshot_path.read_bytes() == Path(provisional.__file__).read_bytes()
    decision = json.loads(decision_path.read_text())
    body = {key: value for key, value in decision.items() if key != "decision_body_sha256"}
    assert decision["decision_body_sha256"] == hashlib.sha256(
        provisional.canonical_json_bytes(body)
    ).hexdigest()
    assert decision["protocol_deviation"]["occurred"] is True
    assert receipt["artifacts"]["provisional_decision"]["sha256"] == provisional.sha256_file(
        decision_path
    )
    assert receipt["artifacts"]["provisional_scores"]["sha256"] == provisional.sha256_file(
        scores_path
    )
    assert receipt["artifacts"]["implementation_snapshot"] == {
        "path": "implementation_snapshot.py",
        "sha256": provisional.sha256_file(snapshot_path),
        "size_bytes": snapshot_path.stat().st_size,
    }
    assert decision["code_provenance"]["implementation_snapshot"] == receipt[
        "artifacts"
    ]["implementation_snapshot"]
    assert receipt["code_provenance"]["source_git"]["implementation_snapshot_is_authoritative"] is True
    assert receipt["code_provenance"]["source_git"]["head_commit_reliable"] is True
    assert len(receipt["code_provenance"]["source_git"]["head_commit"]) in (40, 64)
    assert isinstance(receipt["code_provenance"]["source_git"]["dirty"], bool)
    assert receipt["all_inputs_revalidated_immediately_before_publish"] is True

    rows = [json.loads(line) for line in scores_path.read_text().splitlines()]
    by_id = {row["anonymous_review_id"]: row for row in rows}
    assert by_id["rv_causal"]["scores"] == {
        "source_visibility": 0.5,
        "footprint_visibility": 1.5,
        "receiver_preservation": 2,
        "video_quality": 1,
    }
    assert by_id["rv_specificity"]["scores"] == {
        "protected_object_visibility": 1,
        "noncausal_role_adherence": 0.5,
        "receiver_preservation": 1.5,
        "video_quality": 1.5,
    }
    assert all(row["status"] == provisional.STATUS for row in rows)
    assert all(set(row) == provisional.PROVISIONAL_ROW_FIELDS for row in rows)
    assert set(receipt["inputs"]) == {
        "scores_a",
        "scores_a_merge_manifest",
        "scores_b",
        "scores_b_merge_manifest",
        "public_assignments_a",
        "public_assignments_b",
        "public_key_commitments",
        "evaluation_code_registry",
        "provisional_implementation",
    }

    with pytest.raises(provisional.ProvisionalConsensusError, match="fresh-only"):
        provisional.build_provisional_consensus(
            scores_a_path=fixture["scores_a"],
            scores_b_path=fixture["scores_b"],
            assignments_path=fixture["assignments"],
            key_commitments_path=fixture["commitments"],
            output_root=output,
            expected_items=2,
        )


def test_consensus_rejects_missing_extra_and_binding_mismatch(tmp_path: Path):
    fixture = _fixture(tmp_path)
    assignments = canonical.load_assignments(fixture["assignments"], 2)
    scores_a = canonical.load_merged_scores(
        fixture["scores_a"], "A", 2, expected_assignments_path=fixture["assignments"]
    )
    scores_b = canonical.load_merged_scores(fixture["scores_b"], "B", 2)

    with pytest.raises(provisional.ProvisionalConsensusError, match="row counts"):
        provisional.build_provisional_rows(
            scores_a, scores_b[:-1], assignments, expected_items=2
        )
    extra = dict(scores_b[0])
    extra["unexpected"] = True
    with pytest.raises(provisional.ProvisionalConsensusError, match="fields changed"):
        provisional.build_provisional_rows(
            scores_a, [extra, scores_b[1]], assignments, expected_items=2
        )
    changed = json.loads(json.dumps(scores_b))
    changed[0]["assignment_sha256"] = "f" * 64
    with pytest.raises(provisional.ProvisionalConsensusError, match="binding mismatch"):
        provisional.build_provisional_rows(
            scores_a, changed, assignments, expected_items=2
        )


def test_public_commitments_reject_extra_fields_and_cli_has_no_private_key_input(
    tmp_path: Path,
):
    fixture = _fixture(tmp_path)
    bad_path = fixture["commitments"].parent / "bad_key_commitments.json"
    value = json.loads(fixture["commitments"].read_text())
    value["unexpected"] = "not allowed"
    bad_path.write_bytes(provisional.canonical_json_bytes(value))
    with pytest.raises(provisional.ProvisionalConsensusError, match="schema changed"):
        provisional.load_public_commitments(bad_path, expected_items=2)

    option_strings = {
        option
        for action in provisional.build_parser()._actions
        for option in action.option_strings
    }
    assert "--full-key" not in option_strings
    assert "--original-key" not in option_strings
    assert "--canonical-scores" not in option_strings


@pytest.mark.parametrize("location", ("private", "renamed_public"))
def test_commitments_must_be_exact_shared_public_file_before_content_read(
    tmp_path: Path, monkeypatch, location: str
):
    fixture = _fixture(tmp_path)
    if location == "private":
        wrong = fixture["public_root"].parent / "private" / "key_commitments.json"
    else:
        wrong = fixture["public_root"] / "renamed_key_commitments.json"
    wrong.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(fixture["commitments"], wrong)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("commitment contents were read before provenance refusal")

    monkeypatch.setattr(provisional, "load_public_commitments", forbidden)
    output = tmp_path / f"rejected-{location}"
    with pytest.raises(
        provisional.ProvisionalConsensusError,
        match="exactly the shared public root/key_commitments.json",
    ):
        provisional.build_provisional_consensus(
            scores_a_path=fixture["scores_a"],
            scores_b_path=fixture["scores_b"],
            assignments_path=fixture["assignments"],
            key_commitments_path=wrong,
            output_root=output,
            expected_items=2,
        )
    assert not output.exists()


def test_cross_package_pass_b_is_refused_before_commitment_content_read(
    tmp_path: Path, monkeypatch
):
    first = _fixture(tmp_path / "first")
    second = _fixture(tmp_path / "second")
    merge_b_path = first["scores_b"].parent / "merge_manifest.json"
    merge_b = json.loads(merge_b_path.read_text())
    merge_b["assignments"] = _ref(second["assignments_b"])
    merge_b_path.write_bytes(provisional.canonical_json_bytes(merge_b))

    def forbidden(*_args, **_kwargs):
        raise AssertionError("commitment contents were read before package refusal")

    monkeypatch.setattr(provisional, "load_public_commitments", forbidden)
    output = tmp_path / "rejected-cross-package"
    with pytest.raises(
        provisional.ProvisionalConsensusError,
        match="same review_package/public root",
    ):
        provisional.build_provisional_consensus(
            scores_a_path=first["scores_a"],
            scores_b_path=first["scores_b"],
            assignments_path=first["assignments"],
            key_commitments_path=first["commitments"],
            output_root=output,
            expected_items=2,
        )
    assert not output.exists()


def test_input_change_during_build_is_refused_by_terminal_revalidation(
    tmp_path: Path, monkeypatch
):
    fixture = _fixture(tmp_path)
    original_builder = provisional.build_provisional_rows

    def mutate_after_materialization(*args, **kwargs):
        rows = original_builder(*args, **kwargs)
        fixture["scores_a"].write_bytes(fixture["scores_a"].read_bytes() + b"\n")
        return rows

    monkeypatch.setattr(provisional, "build_provisional_rows", mutate_after_materialization)
    output = tmp_path / "rejected-toctou"
    with pytest.raises(
        provisional.ProvisionalConsensusError,
        match="changed|blank/noncanonical|SHA mismatch",
    ):
        provisional.build_provisional_consensus(
            scores_a_path=fixture["scores_a"],
            scores_b_path=fixture["scores_b"],
            assignments_path=fixture["assignments"],
            key_commitments_path=fixture["commitments"],
            output_root=output,
            expected_items=2,
        )
    assert not output.exists()
