from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import causal_role_erasure_8mechanism_capability_review_workflow_v2 as workflow  # noqa: E402


CANONICAL = PROJECT_ROOT / workflow.DEFAULT_CANONICAL
TEMPLATE = PROJECT_ROOT / workflow.DEFAULT_REVIEW_TEMPLATE
RUBRIC = PROJECT_ROOT / workflow.DEFAULT_REVIEW_RUBRIC
FREEZE = PROJECT_ROOT / workflow.DEFAULT_REVIEW_FREEZE


def _write_csv(path: Path, rows, fields) -> None:
    workflow.write_bytes_exclusive(path, workflow.csv_bytes(rows, fields))


def _fake_generation(tmp_path: Path) -> tuple[Path, list[dict[str, str]]]:
    canonical = workflow.read_json_list(CANONICAL, "canonical")
    videos = tmp_path / "videos"
    videos.mkdir()
    items = []
    for index, row in enumerate(canonical):
        video = videos / f"{index:03d}.mp4"
        video.write_bytes(f"fake-video-{index}\n".encode())
        items.append(
            {
                "canonical_row_index": index,
                "generation_id": row["generation_id"],
                "mechanism": row["mechanism"],
                "seed": int(row["seed"]),
                "video_path": str(video.resolve()),
                "video_sha256": workflow.sha256_file(video),
                "size_bytes": video.stat().st_size,
                "media": {
                    "decoded_frames": 49,
                    "fps": "8/1",
                    "height": 480,
                    "width": 832,
                },
            }
        )
    mechanism_manifests = []
    for index, mechanism in enumerate(workflow.MECHANISM_ORDER):
        start = index * 24
        end = start + 24
        canonical_slice = canonical[start:end]
        top_slice = items[start:end]
        prompt_shard = tmp_path / f"{index:02d}_{mechanism}.prompts"
        prompt_shard.write_text(
            "\n".join(
                [f"# {mechanism}", ""]
                + [
                    f"{row['prompt']} | {row['target_concept']} | {row['expected_footprint']}"
                    for row in canonical_slice
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        seeds = [int(row["seed"]) for row in canonical_slice]
        mechanism_items = [
            {
                "index": local_index,
                "prompt": row["prompt"],
                "target_concept": row["target_concept"],
                "expected_effect": row["expected_footprint"],
                "seed": int(row["seed"]),
                "video_path": top_item["video_path"],
            }
            for local_index, (row, top_item) in enumerate(
                zip(canonical_slice, top_slice)
            )
        ]
        mechanism_payload = {
            "created_at_utc": "2026-08-23T00:00:00+00:00",
            "baseline": "clean",
            "pipeline": "WanPipeline",
            "model": "test-model",
            "dry_run": False,
            "prompts": str(prompt_shard.resolve()),
            "generation": {
                **workflow.MECHANISM_GENERATION_FIXED_FIELDS,
                "seed": seeds[0],
                "seeds": seeds,
            },
            "items": mechanism_items,
        }
        mechanism_manifest = tmp_path / f"{index:02d}_{mechanism}_generation.json"
        mechanism_manifest.write_text(
            json.dumps(mechanism_payload, indent=2) + "\n",
            encoding="utf-8",
        )
        mechanism_manifests.append(
            {
                "mechanism": mechanism,
                "generation_manifest": str(mechanism_manifest.resolve()),
                "generation_manifest_sha256": workflow.sha256_file(
                    mechanism_manifest
                ),
            }
        )
    payload = {
        "schema_version": 1,
        "protocol_version": workflow.MANIFEST_PROTOCOL,
        "status": "frozen_after_exact_media_validation",
        "stage_registry_sha256": "1" * 64,
        "canonical_manifest_sha256": workflow.EXPECTED_CANONICAL_SHA256,
        "prompts_sha256": workflow.EXPECTED_PROMPTS_SHA256,
        "generator_sha256": workflow.EXPECTED_GENERATOR_SHA256,
        "generation": workflow.GENERATION_FIELDS,
        "mechanism_manifests": mechanism_manifests,
        "video_binding_key": "generation_id",
        "video_count": 192,
        "items": items,
    }
    manifest = tmp_path / "generation.json"
    manifest.write_bytes(workflow.canonical_json_bytes(payload))
    return manifest, canonical


def _fake_render(_frames, path: Path) -> None:
    workflow.write_bytes_exclusive(path, b"synthetic-jpeg\n")


def _filled_assignment(path: Path, *, disagreement: bool = False):
    rows = workflow.read_csv_exact(path, workflow.ASSIGNMENT_FIELDS, "assignment")
    for row in rows:
        row["decodable"] = "1"
        for field in workflow.ORDINAL_FIELDS:
            row[field] = "2"
    if disagreement:
        rows[0]["quality"] = "1"
    return rows


def test_panel_ranges_cover_all_49_and_review_orders_are_isolated():
    assert workflow.panel_ranges(49) == (
        (0, 12),
        (9, 21),
        (18, 30),
        (27, 39),
        (36, 48),
    )
    covered = {
        frame
        for start, end in workflow.PANEL_RANGES
        for frame in range(start, end + 1)
    }
    assert covered == set(range(49))
    ids = [f"g{index:03d}" for index in range(192)]
    left = workflow.deterministic_order(ids, "reviewer_a")
    right = workflow.deterministic_order(ids, "reviewer_b")
    assert left != right
    assert set(left) == set(right) == set(ids)
    with pytest.raises(ValueError, match="exactly 49"):
        workflow.panel_ranges(48)


def test_frozen_v2_inputs_are_exact_hash_bound_and_blank():
    canonical, template, rubric = workflow.validate_frozen_inputs(
        CANONICAL,
        TEMPLATE,
        RUBRIC,
        FREEZE,
    )
    assert len(canonical) == len(template) == 192
    assert workflow.sha256_file(CANONICAL) == workflow.EXPECTED_CANONICAL_SHA256
    assert workflow.sha256_file(TEMPLATE) == workflow.EXPECTED_REVIEW_TEMPLATE_SHA256
    assert workflow.sha256_file(RUBRIC) == workflow.EXPECTED_REVIEW_RUBRIC_SHA256
    assert workflow.sha256_file(FREEZE) == workflow.EXPECTED_REVIEW_FREEZE_SHA256
    assert rubric["review_protocol_version"] == workflow.REVIEW_PROTOCOL
    assert rubric["review_workflow"]["reviewers_must_not_share_scores"] is True
    assert all(
        not row[field]
        for row in template
        for field in (*workflow.SCORE_FIELDS, "reviewer_notes")
    )


def test_build_parser_requires_post_generation_manifest_commitment():
    assert not hasattr(workflow, "EXPECTED_GENERATION_MANIFEST_SHA256")
    with pytest.raises(SystemExit):
        workflow.build_parser().parse_args(
            [
                "build",
                "--generation-manifest",
                "generation.json",
                "--output-root",
                "package",
            ]
        )


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("protocol_version", "wrong", "generation protocol mismatch"),
        ("canonical_manifest_sha256", "0" * 64, "canonical binding mismatch"),
        ("prompts_sha256", "0" * 64, "prompt binding mismatch"),
        ("generator_sha256", "0" * 64, "implementation binding mismatch"),
        ("stage_registry_sha256", "not-a-sha", "stage-registry SHA-256"),
    ],
)
def test_generation_manifest_strictly_rejects_binding_drift(
    tmp_path: Path,
    field: str,
    value: str,
    error: str,
):
    generation, canonical = _fake_generation(tmp_path)
    payload = json.loads(generation.read_text(encoding="utf-8"))
    payload[field] = value
    changed = tmp_path / f"changed-{field}.json"
    changed.write_bytes(workflow.canonical_json_bytes(payload))
    with pytest.raises(ValueError, match=error):
        workflow.validate_generation_manifest(
            changed,
            workflow.sha256_file(changed),
            canonical,
        )


@pytest.mark.parametrize(
    "mutation,error",
    [
        ("generation_id", "order/ID mismatch"),
        ("seed", "seed mismatch"),
        ("media", "media contract mismatch"),
    ],
)
def test_generation_manifest_rejects_id_seed_and_media_drift(
    tmp_path: Path,
    mutation: str,
    error: str,
):
    generation, canonical = _fake_generation(tmp_path)
    payload = json.loads(generation.read_text(encoding="utf-8"))
    if mutation == "generation_id":
        payload["items"][0]["generation_id"] = "wrong"
    elif mutation == "seed":
        payload["items"][0]["seed"] += 1
    else:
        payload["items"][0]["media"]["decoded_frames"] = 48
    changed = tmp_path / f"changed-{mutation}.json"
    changed.write_bytes(workflow.canonical_json_bytes(payload))
    with pytest.raises(ValueError, match=error):
        workflow.validate_generation_manifest(
            changed,
            workflow.sha256_file(changed),
            canonical,
        )


def test_generation_manifest_rejects_wrong_caller_commitment(tmp_path: Path):
    generation, canonical = _fake_generation(tmp_path)
    with pytest.raises(ValueError, match="generation-manifest SHA-256 mismatch"):
        workflow.validate_generation_manifest(generation, "0" * 64, canonical)


def test_mechanism_manifest_is_rehashed_and_items_match_top_slice(tmp_path: Path):
    generation, canonical = _fake_generation(tmp_path)
    top = json.loads(generation.read_text(encoding="utf-8"))
    record = top["mechanism_manifests"][0]
    mechanism_path = Path(record["generation_manifest"])
    payload = json.loads(mechanism_path.read_text(encoding="utf-8"))
    payload["items"][0]["prompt"] = "drifted prompt"
    mechanism_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    record["generation_manifest_sha256"] = workflow.sha256_file(mechanism_path)
    changed = tmp_path / "generation-item-drift.json"
    changed.write_bytes(workflow.canonical_json_bytes(top))
    with pytest.raises(ValueError, match="generation item prompt mismatch"):
        workflow.validate_generation_manifest(
            changed,
            workflow.sha256_file(changed),
            canonical,
        )


def test_mechanism_manifest_symlink_is_rejected_before_read(tmp_path: Path):
    generation, canonical = _fake_generation(tmp_path)
    top = json.loads(generation.read_text(encoding="utf-8"))
    record = top["mechanism_manifests"][0]
    mechanism_path = Path(record["generation_manifest"])
    real_path = mechanism_path.with_name("real-generation.json")
    mechanism_path.rename(real_path)
    mechanism_path.symlink_to(real_path)
    changed = tmp_path / "generation-symlink.json"
    changed.write_bytes(workflow.canonical_json_bytes(top))
    with pytest.raises(ValueError, match="symlink ancestor|changes after resolution"):
        workflow.validate_generation_manifest(
            changed,
            workflow.sha256_file(changed),
            canonical,
        )


def _call_public_api_with_output(name: str, output_root: Path) -> None:
    if name == "build":
        workflow.build_package(
            project_root=PROJECT_ROOT,
            generation_manifest=Path("missing-generation.json"),
            expected_generation_sha256="0" * 64,
            canonical_path=Path("missing-canonical.json"),
            template_path=Path("missing-template.csv"),
            rubric_path=Path("missing-rubric.json"),
            freeze_path=Path("missing-freeze.json"),
            output_root=output_root,
        )
    elif name == "plan":
        workflow.plan_adjudication(
            package_root=Path("missing-package"),
            expected_package_manifest_sha256="0" * 64,
            reviewer_a_completed=Path("missing-a.csv"),
            reviewer_b_completed=Path("missing-b.csv"),
            output_root=output_root,
        )
    elif name == "finalize":
        workflow.finalize_reviews(
            package_root=Path("missing-package"),
            expected_package_manifest_sha256="0" * 64,
            adjudication_root=Path("missing-adjudication"),
            expected_adjudication_manifest_sha256="0" * 64,
            third_completed=Path("missing-third.csv"),
            output_root=output_root,
        )
    else:
        workflow.score_bound_review(
            final_root=Path("missing-final"),
            expected_review_run_registry_sha256="0" * 64,
            output_root=output_root,
        )


@pytest.mark.parametrize("entry", ["build", "plan", "finalize", "score"])
def test_public_api_rejects_output_ancestor_symlink_before_any_input_read(
    tmp_path: Path,
    entry: str,
):
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    alias = tmp_path / "ordinary-alias"
    alias.symlink_to(real_parent, target_is_directory=True)
    output = alias / f"{entry}-output"
    with pytest.raises(ValueError, match="existing symlink ancestor"):
        _call_public_api_with_output(entry, output)
    assert not (real_parent / f"{entry}-output").exists()


def test_build_adjudicate_finalize_and_v2_score_are_hash_bound(tmp_path: Path):
    generation, canonical = _fake_generation(tmp_path)
    generation_sha256 = workflow.sha256_file(generation)
    package = tmp_path / "package"
    result = workflow.build_package(
        project_root=PROJECT_ROOT,
        generation_manifest=generation,
        expected_generation_sha256=generation_sha256,
        canonical_path=CANONICAL,
        template_path=TEMPLATE,
        rubric_path=RUBRIC,
        freeze_path=FREEZE,
        output_root=package,
        frame_loader=lambda _path: [object()] * 49,
        composite_renderer=_fake_render,
    )
    assert result["workflow_version"] == workflow.WORKFLOW_VERSION
    assert result["video_count"] == 192
    assert result["generation_manifest"]["sha256"] == generation_sha256
    verified = workflow.verify_package(package)
    assert verified["status"] == "committed_before_independent_review"
    assert len(verified["composites"]) == 192

    a_assignment = package / "public/reviewer_a/assignment.csv"
    b_assignment = package / "public/reviewer_b/assignment.csv"
    a_blank = workflow.read_csv_exact(a_assignment, workflow.ASSIGNMENT_FIELDS, "A")
    b_blank = workflow.read_csv_exact(b_assignment, workflow.ASSIGNMENT_FIELDS, "B")
    assert [row["anonymous_review_id"] for row in a_blank] == [
        f"a{i:03d}" for i in range(192)
    ]
    assert [row["anonymous_review_id"] for row in b_blank] == [
        f"b{i:03d}" for i in range(192)
    ]
    assert [row["prompt"] for row in a_blank] != [row["prompt"] for row in b_blank]
    assert not ({"generation_id", "mechanism", "seed"} & set(a_blank[0]))
    assert {
        row["composite_path"] for row in a_blank
    }.isdisjoint({row["composite_path"] for row in b_blank})
    assert stat.S_IMODE((package / "private").stat().st_mode) == 0o700
    assert stat.S_IMODE(
        (package / "private/reviewer_a_binding.csv").stat().st_mode
    ) == 0o600
    instructions = json.loads(
        (package / "public/reviewer_a/instructions.json").read_text(encoding="utf-8")
    )
    assert instructions["review_protocol_version"] == workflow.REVIEW_PROTOCOL
    assert instructions["video_contract"]["all_49_frames_must_be_inspected"] is True
    assert instructions["diagnostic_only_fields"] == ["clean_prefix"]
    assert instructions["coupled_requirement"]["value"] == 3

    delivered = package / a_blank[0]["composite_path"]
    original = delivered.read_bytes()
    delivered.write_bytes(original + b"tamper")
    with pytest.raises(ValueError, match="delivered composite hash drift"):
        workflow.verify_package(
            package,
            workflow.sha256_file(package / "review_package_manifest.json"),
        )
    delivered.write_bytes(original)

    a_completed = tmp_path / "reviewer_a_completed.csv"
    b_completed = tmp_path / "reviewer_b_completed.csv"
    _write_csv(
        a_completed,
        _filled_assignment(a_assignment),
        workflow.ASSIGNMENT_FIELDS,
    )
    _write_csv(
        b_completed,
        _filled_assignment(b_assignment, disagreement=True),
        workflow.ASSIGNMENT_FIELDS,
    )

    package_sha256 = workflow.sha256_file(package / "review_package_manifest.json")
    adjudication = tmp_path / "adjudication"
    planned = workflow.plan_adjudication(
        package_root=package,
        expected_package_manifest_sha256=package_sha256,
        reviewer_a_completed=a_completed,
        reviewer_b_completed=b_completed,
        output_root=adjudication,
    )
    assert planned["videos_with_disagreement"] == 1
    assert planned["atomic_disagreement_count"] == 1
    third_assignment = adjudication / "public/third_reviewer/assignment.csv"
    third_rows = workflow.read_csv_exact(
        third_assignment,
        workflow.ADJUDICATION_FIELDS,
        "third",
    )
    assert len(third_rows) == 1
    assert third_rows[0]["requested_fields"] == "quality"
    assert stat.S_IMODE((adjudication / "private").stat().st_mode) == 0o700
    third_rows[0]["quality"] = "2"
    third_completed = tmp_path / "third_completed.csv"
    _write_csv(third_completed, third_rows, workflow.ADJUDICATION_FIELDS)

    final = tmp_path / "final"
    frozen = workflow.finalize_reviews(
        package_root=package,
        expected_package_manifest_sha256=package_sha256,
        adjudication_root=adjudication,
        expected_adjudication_manifest_sha256=workflow.sha256_file(
            adjudication / "adjudication_manifest.json"
        ),
        third_completed=third_completed,
        output_root=final,
    )
    assert frozen["row_count"] == 192
    assert frozen["resolved_atomic_disagreements"] == 1
    assert stat.S_IMODE(final.stat().st_mode) == 0o700
    assert stat.S_IMODE((final / "canonical_adjudicated.csv").stat().st_mode) == 0o600
    canonical_scores = workflow.read_csv_exact(
        final / "canonical_adjudicated.csv",
        workflow.FROZEN_REVIEW_FIELDS,
        "canonical adjudicated",
    )
    assert [row["generation_id"] for row in canonical_scores] == [
        row["generation_id"] for row in canonical
    ]
    assert all(row["decodable"] == "1" for row in canonical_scores)
    assert all(
        row[field] == "2"
        for row in canonical_scores
        for field in workflow.ORDINAL_FIELDS
    )

    gate_root = tmp_path / "gate"
    gate = workflow.score_bound_review(
        final_root=final,
        expected_review_run_registry_sha256=workflow.sha256_file(
            final / "review_run_registry.json"
        ),
        output_root=gate_root,
    )
    assert gate["capability_gate_status"] == "pass"
    aggregate = json.loads((gate_root / "aggregate_gate.json").read_text(encoding="utf-8"))
    assert aggregate["schema_version"] == 2
    assert aggregate["review_protocol_version"] == workflow.REVIEW_PROTOCOL
    assert aggregate["aggregate"]["eligible_rows"] == 192
    assert aggregate["thresholds"]["eligible_rows_per_mechanism"] == 12
    assert aggregate["authorization"]["training_authorized"] is False
    bound = json.loads(
        (gate_root / "bound_gate_registry.json").read_text(encoding="utf-8")
    )
    assert Path(bound["frozen_scorer_implementation"]["path"]).name == (
        "causal_role_erasure_8mechanism_capability_review_v2.py"
    )

    with pytest.raises(ValueError, match="already exists"):
        workflow.build_package(
            project_root=PROJECT_ROOT,
            generation_manifest=generation,
            expected_generation_sha256=generation_sha256,
            canonical_path=CANONICAL,
            template_path=TEMPLATE,
            rubric_path=RUBRIC,
            freeze_path=FREEZE,
            output_root=package,
            frame_loader=lambda _path: [object()] * 49,
            composite_renderer=_fake_render,
        )


def test_boolean_majority_and_ordinal_median_are_exact():
    assert workflow.median_of_three("0", "1", "1") == "1"
    assert workflow.median_of_three("2", "0", "1") == "1"
    assert workflow.median_of_three("2", "1", "2") == "2"


def test_build_failure_rolls_back_owned_partial(tmp_path: Path):
    generation, _ = _fake_generation(tmp_path)
    output = tmp_path / "failed-package"
    calls = 0

    def fail_second(frames, path):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected render failure")
        _fake_render(frames, path)

    with pytest.raises(RuntimeError, match="injected render failure"):
        workflow.build_package(
            project_root=PROJECT_ROOT,
            generation_manifest=generation,
            expected_generation_sha256=workflow.sha256_file(generation),
            canonical_path=CANONICAL,
            template_path=TEMPLATE,
            rubric_path=RUBRIC,
            freeze_path=FREEZE,
            output_root=output,
            frame_loader=lambda _path: [object()] * 49,
            composite_renderer=fail_second,
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".failed-package.partial-*"))


def test_sealed_parent_symlink_is_rejected_before_completed_review_read(
    tmp_path: Path,
):
    safe_assignment = tmp_path / "assignment.csv"
    rows = [
        {
            **{field: "context" for field in workflow.ASSIGNMENT_CONTEXT_FIELDS},
            **{
                field: ("1" if field == "decodable" else "2")
                for field in workflow.SCORE_FIELDS
            },
            "reviewer_notes": "",
        }
        for _ in range(192)
    ]
    for index, row in enumerate(rows):
        row["assignment_position"] = str(index)
        row["anonymous_review_id"] = f"a{index:03d}"
    _write_csv(safe_assignment, rows, workflow.ASSIGNMENT_FIELDS)
    sealed = tmp_path / "synthetic_final36"
    sealed.mkdir()
    hidden = sealed / "completed.csv"
    hidden.write_bytes(workflow.csv_bytes(rows, workflow.ASSIGNMENT_FIELDS))
    alias = tmp_path / "ordinary_alias"
    alias.symlink_to(sealed, target_is_directory=True)
    with pytest.raises(ValueError, match="sealed/final36"):
        workflow.validate_completed_review(
            alias / "completed.csv",
            safe_assignment,
            "completed",
        )
