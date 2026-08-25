from __future__ import annotations

import csv
import importlib
import itertools
import json
import random
import sys
from pathlib import Path

import pytest

from scripts import (
    finalize_causal_role_erasure_7mechanism_targeted_audit_v1 as finalizer,
)
from scripts import (
    screen_causal_role_erasure_7mechanism_targets_v2 as screening,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_CANDIDATES = (
    REPO_ROOT / "data/causal_role_erasure_7mechanism_main_v2/target_candidates.csv"
)


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def _write_csv(path: Path, fields, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _evidence(label: str):
    return [{"frame_index": 0, "observation": label}]


def _fixture(tmp_path: Path):
    project = tmp_path / "project"
    project.mkdir()
    candidate_fields, candidate_rows = _read_csv(SOURCE_CANDIDATES)
    candidate_manifest = project / "target_candidates.csv"
    _write_csv(candidate_manifest, candidate_fields, candidate_rows)
    new_candidates = [
        row for row in candidate_rows if row["mechanism"] in screening.NEW_MECHANISMS
    ]

    assignments = []
    bindings = []
    scores_by_id = {}
    candidate_to_anonymous = {}
    for position, candidate in enumerate(new_candidates):
        anonymous_id = f"a{position:04d}"
        candidate_to_anonymous[candidate["candidate_id"]] = anonymous_id
        assignments.append(
            {
                "assignment_position": str(position),
                "anonymous_review_id": anonymous_id,
                "mechanism": candidate["mechanism"],
                "mechanism_name": candidate["mechanism_name"],
                "composite_path": f"public/reviewer_a/media/{anonymous_id}.jpg",
                "target_prompt": candidate["target_prompt"],
                "receiver": candidate["receiver"],
                "source_object": candidate["source_object"],
                "expected_trigger": candidate["expected_trigger"],
                "expected_footprint": candidate["expected_footprint"],
                "expected_counterfactual_state": candidate["expected_counterfactual_state"],
                "excluded_content_phrase": candidate["excluded_content_phrase"],
            }
        )
        video_path = project / "generated" / f"{candidate['candidate_id']}.mp4"
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_path.write_bytes((candidate["candidate_id"] + "\n").encode())
        bindings.append(
            {
                "anonymous_review_id": anonymous_id,
                "pass_name": "reviewer_a",
                "candidate_id": candidate["candidate_id"],
                "mechanism": candidate["mechanism"],
                "candidate_row_sha256": screening.canonical_row_sha256(candidate),
                "video_path": video_path.relative_to(project).as_posix(),
                "video_sha256": screening.sha256_file(video_path),
                "composite_path": f"public/reviewer_a/media/{anonymous_id}.jpg",
                "composite_sha256": "0" * 64,
            }
        )
        scores_by_id[anonymous_id] = {field: 2 for field in screening.SCORE_FIELDS}

    rigid = [
        row for row in new_candidates if row["mechanism"] == "rigid_collision"
    ]
    rigid.sort(key=lambda row: finalizer.selection_hash(row["candidate_id"]))
    high = rigid[:177]
    boundary = rigid[-1]
    low = rigid[177:-1]
    boundary_id = candidate_to_anonymous[boundary["candidate_id"]]
    stable_audited_id = candidate_to_anonymous[high[0]["candidate_id"]]
    scores_by_id[boundary_id]["natural_motion"] = 1
    for candidate in low:
        scores_by_id[candidate_to_anonymous[candidate["candidate_id"]]]["source_absent"] = 1

    completed_rows = [
        {
            "anonymous_review_id": assignment["anonymous_review_id"],
            **{
                field: str(scores_by_id[assignment["anonymous_review_id"]][field])
                for field in screening.SCORE_FIELDS
            },
            "reviewer_notes": "synthetic complete Reviewer A",
        }
        for assignment in assignments
    ]
    assignment_path = project / "reviewer_a_assignment.csv"
    completed_path = project / "reviewer_a_completed.csv"
    binding_path = project / "reviewer_a_binding.csv"
    _write_csv(assignment_path, screening.ASSIGNMENT_FIELDS, assignments)
    _write_csv(completed_path, screening.SCORE_TEMPLATE_FIELDS, completed_rows)
    _write_csv(binding_path, screening.BINDING_FIELDS, bindings)

    audit_items = []
    for anonymous_id, changed_field, changed_score in (
        (boundary_id, "quality", 0),
        (stable_audited_id, "natural_motion", 1),
    ):
        scores = dict(scores_by_id[anonymous_id])
        scores[changed_field] = changed_score
        audit_items.append(
            {
                "anonymous_review_id": anonymous_id,
                "scores": scores,
                "evidence": _evidence(f"independent evidence for {anonymous_id}"),
            }
        )
    audit_path = project / "targeted_audit.json"
    audit_path.write_text(
        json.dumps(
            {
                "reviewer": "blind_synthetic_agent",
                "blind_to_a": True,
                "items": audit_items,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    water_candidates = [
        row
        for row in candidate_rows
        if row["mechanism"] == screening.WATER_MECHANISM
        and row["historical_screen_status"] == "accept"
    ]
    assert len(water_candidates) == screening.SELECTED_PER_MECHANISM
    water_rows = []
    for candidate in water_candidates:
        video = project / candidate["target_video_path"]
        video.parent.mkdir(parents=True, exist_ok=True)
        video.write_bytes((candidate["candidate_id"] + "\n").encode())
        water_rows.append(
            {
                "pair_id": candidate["historical_pair_id"],
                "desired_target_video": candidate["target_video_path"],
            }
        )
    water_path = project / "water_accepted178.csv"
    _write_csv(water_path, ("pair_id", "desired_target_video"), water_rows)
    return {
        "project": project,
        "candidate_manifest": candidate_manifest,
        "assignment": assignment_path,
        "completed": completed_path,
        "binding": binding_path,
        "audit": audit_path,
        "water": water_path,
        "boundary_id": boundary_id,
        "boundary_candidate": boundary["candidate_id"],
        "stable_audited_id": stable_audited_id,
        "stable_audited_candidate": high[0]["candidate_id"],
    }


def _finalize_args(fixture, output: Path, adjudication: Path | None):
    return {
        "project_root": fixture["project"],
        "reviewer_a_completed": fixture["completed"],
        "reviewer_a_assignment": fixture["assignment"],
        "agent_audit_paths": [fixture["audit"]],
        "adjudication_path": adjudication,
        "reviewer_a_binding": fixture["binding"],
        "candidate_manifest": fixture["candidate_manifest"],
        "water_accepted178": fixture["water"],
        "output_root": output,
    }


def test_natural_motion_is_soft_and_frozen_rank_tuple_is_score_first():
    natural_zero = {field: 2 for field in finalizer.SCORE_FIELDS}
    natural_zero["natural_motion"] = 0
    assert not finalizer.hard_reject(natural_zero)
    quality_zero = dict(natural_zero, quality=0)
    assert finalizer.hard_reject(quality_zero)
    natural_two = dict(natural_zero, natural_motion=2)
    assert finalizer.ranking_sort_key(natural_two, "candidate-a") < finalizer.ranking_sort_key(
        natural_zero, "candidate-b"
    )


def test_membership_influence_helper_matches_brute_force():
    rng = random.Random(20260826)
    possible_states = [None, (0,), (1,), (2,), (3,)]
    for _ in range(500):
        domain_count = rng.randint(0, 5)
        domains = [
            set(rng.sample(possible_states, rng.randint(1, len(possible_states))))
            for _ in range(domain_count)
        ]
        left, right = rng.sample(possible_states, 2)
        selected_count = rng.randint(1, max(1, domain_count + 1))
        expected = False
        for states in itertools.product(*domains):
            left_selected = left is not None and sum(
                state is not None and state > left for state in states
            ) < selected_count
            right_selected = right is not None and sum(
                state is not None and state > right for state in states
            ) < selected_count
            if left_selected != right_selected:
                expected = True
                break
        assert finalizer._priority_pair_can_change_membership(
            left, right, domains, selected_count
        ) is expected


def test_plan_emits_only_selection_affecting_disagreement_and_hides_scores(tmp_path: Path):
    fixture = _fixture(tmp_path)
    output = fixture["project"] / "adjudication_plan"
    result = finalizer.plan_adjudication(
        reviewer_a_completed=fixture["completed"],
        reviewer_a_assignment=fixture["assignment"],
        agent_audit_paths=[fixture["audit"]],
        reviewer_a_binding=None,
        candidate_manifest=fixture["candidate_manifest"],
        output_root=output,
    )
    assert result["atomic_audit_disagreements"] == 2
    assert result["selection_affecting_atomic_disagreements"] == 1
    assert result["selection_analysis_tie_break_basis"] == (
        "candidate_id_from_exact_public_manifest_join"
    )
    public = json.loads((output / "public/adjudication_assignment.json").read_text())
    assert public["blind_to_reviewer_a_and_agent_audit_scores"] is True
    assert public["items"] == [
        {
            **{
                key: value
                for key, value in public["items"][0].items()
                if key != "requested_fields"
            },
            "requested_fields": ["quality"],
        }
    ]
    assert public["items"][0]["anonymous_review_id"] == fixture["boundary_id"]
    assert all("scores" not in item for item in public["items"])
    assert all(
        field not in item
        for item in public["items"]
        for field in finalizer.SCORE_FIELDS
    )
    template = json.loads((output / "public/adjudication_template.json").read_text())
    assert template["items"][0]["scores"] == {"quality": None}


def test_finalize_fails_until_affecting_field_is_adjudicated_then_freezes_178(tmp_path: Path):
    fixture = _fixture(tmp_path)
    with pytest.raises(finalizer.TargetedAuditError, match="selection-affecting"):
        finalizer.finalize_selection(
            **_finalize_args(fixture, fixture["project"] / "must_not_exist", None)
        )

    adjudication = fixture["project"] / "adjudication.json"
    adjudication.write_text(
        json.dumps(
            {
                "reviewer": "blind_adjudicator",
                "blind_to_reviewer_a_and_agent_audit_scores": True,
                "items": [
                    {
                        "anonymous_review_id": fixture["boundary_id"],
                        "scores": {"quality": 2},
                        "evidence": _evidence("receiver is usable throughout"),
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = fixture["project"] / "selection"
    result = finalizer.finalize_selection(
        **_finalize_args(fixture, output, adjudication)
    )
    assert result["selected_total"] == 7 * 178
    assert result["adjudicated_atomic_disagreements"] == 1
    assert result["unresolved_selection_inert_atomic_disagreements"] == 1
    for mechanism_index, mechanism in enumerate(finalizer.MECHANISM_ORDER):
        _, rows = _read_csv(
            output / f"selected_by_mechanism/{mechanism_index:02d}_{mechanism}.csv"
        )
        assert len(rows) == 178
        assert [int(row["selection_rank"]) for row in rows] == list(range(1, 179))
        assert [row["selection_hash"] for row in rows] == sorted(
            row["selection_hash"] for row in rows
        )
        if mechanism == "rigid_collision":
            assert fixture["boundary_candidate"] in {
                row["candidate_id"] for row in rows
            }

    canonical = {
        row["candidate_id"]: row
        for row in _read_csv(output / "canonical_nonwater_scores.csv")[1]
    }
    assert canonical[fixture["boundary_candidate"]]["quality"] == "2"
    stable = canonical[fixture["stable_audited_candidate"]]
    assert stable["natural_motion"] == ""
    assert stable["canonical_status"] == "partially_unresolved_selection_inert"
    assert json.loads(stable["score_domains_json"])["natural_motion"] == [1, 2]
    provenance = [
        row
        for row in _read_csv(output / "audit_provenance.csv")[1]
        if row["candidate_id"] == fixture["stable_audited_candidate"]
        and row["field"] == "natural_motion"
    ]
    assert provenance[0]["resolution"] == "unresolved"
    assert provenance[0]["canonical_score"] == ""

    # The next frozen stage consumes this aggregate without adapting fields,
    # ordering, hashes, prompts, or mechanism blocks.
    scripts_dir = str(REPO_ROOT / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    registries = importlib.import_module(
        "build_causal_role_erasure_7mechanism_training_registries_v2"
    )
    _, ontologies = registries.load_ontology(
        REPO_ROOT
        / "data/causal_role_erasure_7mechanism_main_v2/ontology_registry.json"
    )
    grouped = registries.validate_selected_targets(
        output / "selected_targets.csv",
        project_root=fixture["project"],
        ontologies=ontologies,
    )
    assert {mechanism: len(rows) for mechanism, rows in grouped.items()} == {
        mechanism: 178 for mechanism in finalizer.MECHANISM_ORDER
    }


def test_finalize_fails_if_a_mechanism_has_fewer_than_178_nonhard(tmp_path: Path):
    fixture = _fixture(tmp_path)
    fields, rows = _read_csv(fixture["completed"])
    rigid_rows = [row for row in rows if int(row["anonymous_review_id"][1:]) < 192]
    assert len(rigid_rows) == 192
    for row in rigid_rows[:15]:
        row["quality"] = "0"
    _write_csv(fixture["completed"], fields, rows)

    # Make the targeted audit agree with the now-canonical values so this test
    # reaches the count gate rather than the disagreement gate.
    payload = json.loads(fixture["audit"].read_text())
    score_by_id = {row["anonymous_review_id"]: row for row in rows}
    for item in payload["items"]:
        item["scores"] = {
            field: int(score_by_id[item["anonymous_review_id"]][field])
            for field in finalizer.SCORE_FIELDS
        }
    fixture["audit"].write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(finalizer.TargetedAuditError, match="only 177 non-hard-reject"):
        finalizer.finalize_selection(
            **_finalize_args(fixture, fixture["project"] / "too_few", None)
        )


def test_finalize_rejects_selected_video_drift(tmp_path: Path):
    fixture = _fixture(tmp_path)
    adjudication = fixture["project"] / "adjudication.json"
    adjudication.write_text(
        json.dumps(
            {
                "reviewer": "blind_adjudicator",
                "blind_to_reviewer_a_and_agent_audit_scores": True,
                "items": [
                    {
                        "anonymous_review_id": fixture["boundary_id"],
                        "scores": {"quality": 2},
                        "evidence": _evidence("usable receiver"),
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    binding_rows = _read_csv(fixture["binding"])[1]
    drift = next(
        row
        for row in binding_rows
        if row["candidate_id"] == fixture["stable_audited_candidate"]
    )
    (fixture["project"] / drift["video_path"]).write_bytes(b"drift")
    with pytest.raises(finalizer.TargetedAuditError, match="video bytes differ from binding"):
        finalizer.finalize_selection(
            **_finalize_args(
                fixture,
                fixture["project"] / "drift_must_not_freeze",
                adjudication,
            )
        )
