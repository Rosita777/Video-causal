from __future__ import annotations

import csv
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

import causal_role_erasure_8mechanism_capability_review_workflow_v1 as workflow  # noqa: E402


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
                "video_path": str(video),
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
    payload = {
        "schema_version": 1,
        "protocol_version": workflow.MANIFEST_PROTOCOL,
        "status": "frozen_after_exact_media_validation",
        "canonical_manifest_sha256": workflow.EXPECTED_CANONICAL_SHA256,
        "video_binding_key": "generation_id",
        "video_count": 192,
        "items": items,
    }
    manifest = tmp_path / "generation.json"
    manifest.write_bytes(workflow.canonical_json_bytes(payload))
    return manifest, canonical


def _fake_render(_frames, path: Path) -> None:
    workflow.write_bytes_exclusive(path, b"synthetic-jpeg\n")


def _filled_assignment(path: Path, *, disagreement: bool = False) -> list[dict[str, str]]:
    rows = workflow.read_csv_exact(path, workflow.ASSIGNMENT_FIELDS, "assignment")
    for row in rows:
        row["decodable"] = "1"
        for field in workflow.ORDINAL_FIELDS:
            row[field] = "2"
    if disagreement:
        rows[0]["quality"] = "1"
    return rows


def test_panel_ranges_cover_all_49_and_orders_are_independent():
    assert workflow.panel_ranges(49) == (
        (0, 12),
        (9, 21),
        (18, 30),
        (27, 39),
        (36, 48),
    )
    covered = {frame for start, end in workflow.PANEL_RANGES for frame in range(start, end + 1)}
    assert covered == set(range(49))
    ids = [f"g{index:03d}" for index in range(192)]
    left = workflow.deterministic_order(ids, "reviewer_a")
    right = workflow.deterministic_order(ids, "reviewer_b")
    assert left != right
    assert set(left) == set(right) == set(ids)
    with pytest.raises(ValueError, match="exactly 49"):
        workflow.panel_ranges(48)


def test_frozen_inputs_remain_exact_and_blank():
    canonical, template, rubric = workflow.validate_frozen_inputs(
        CANONICAL, TEMPLATE, RUBRIC, FREEZE
    )
    assert len(canonical) == len(template) == 192
    assert rubric["review_workflow"]["reviewers_must_not_share_scores"] is True
    assert all(not row[field] for row in template for field in workflow.SCORE_FIELDS)


def test_build_adjudicate_finalize_is_hash_bound_and_fail_closed(tmp_path):
    generation, canonical = _fake_generation(tmp_path)
    package = tmp_path / "package"
    result = workflow.build_package(
        project_root=PROJECT_ROOT,
        generation_manifest=generation,
        expected_generation_sha256=workflow.sha256_file(generation),
        canonical_path=CANONICAL,
        template_path=TEMPLATE,
        rubric_path=RUBRIC,
        freeze_path=FREEZE,
        output_root=package,
        frame_loader=lambda _path: [object()] * 49,
        composite_renderer=_fake_render,
    )
    assert result["video_count"] == 192
    verified = workflow.verify_package(package)
    assert verified["status"] == "committed_before_independent_review"
    assert len(verified["composites"]) == 192

    a_assignment = package / "public/reviewer_a/assignment.csv"
    b_assignment = package / "public/reviewer_b/assignment.csv"
    a_blank = workflow.read_csv_exact(a_assignment, workflow.ASSIGNMENT_FIELDS, "A")
    b_blank = workflow.read_csv_exact(b_assignment, workflow.ASSIGNMENT_FIELDS, "B")
    assert len(a_blank) == len(b_blank) == 192
    assert [row["anonymous_review_id"] for row in a_blank] == [f"a{i:03d}" for i in range(192)]
    assert [row["anonymous_review_id"] for row in b_blank] == [f"b{i:03d}" for i in range(192)]
    assert [row["prompt"] for row in a_blank] != [row["prompt"] for row in b_blank]
    assert not ({"generation_id", "mechanism", "seed"} & set(a_blank[0]))
    a_media = {row["composite_path"] for row in a_blank}
    b_media = {row["composite_path"] for row in b_blank}
    assert a_media.isdisjoint(b_media)
    assert all("/reviewer_a/media/a_" in path for path in a_media)
    assert all("/reviewer_b/media/b_" in path for path in b_media)
    assert stat.S_IMODE((package / "private").stat().st_mode) == 0o700
    assert stat.S_IMODE((package / "private/reviewer_a_binding.csv").stat().st_mode) == 0o600
    instructions = json.loads(
        (package / "public/reviewer_a/instructions.json").read_text(encoding="utf-8")
    )
    assert [field["name"] for field in instructions["atomic_fields"]] == list(
        workflow.SCORE_FIELDS
    )
    assert instructions["video_contract"]["all_49_frames_must_be_inspected"] is True

    delivered = package / a_blank[0]["composite_path"]
    delivered_raw = delivered.read_bytes()
    delivered.write_bytes(delivered_raw + b"tamper")
    with pytest.raises(ValueError, match="delivered composite hash drift"):
        workflow.verify_package(package, workflow.sha256_file(package / "review_package_manifest.json"))
    delivered.write_bytes(delivered_raw)

    a_completed = tmp_path / "reviewer_a_completed.csv"
    b_completed = tmp_path / "reviewer_b_completed.csv"
    _write_csv(a_completed, _filled_assignment(a_assignment), workflow.ASSIGNMENT_FIELDS)
    _write_csv(
        b_completed,
        _filled_assignment(b_assignment, disagreement=True),
        workflow.ASSIGNMENT_FIELDS,
    )

    adjudication = tmp_path / "adjudication"
    package_sha256 = workflow.sha256_file(package / "review_package_manifest.json")
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
        third_assignment, workflow.ADJUDICATION_FIELDS, "third"
    )
    assert len(third_rows) == 1
    assert third_rows[0]["requested_fields"] == "quality"
    assert "/third_reviewer/media/t_" in third_rows[0]["composite_path"]
    assert stat.S_IMODE((adjudication / "private").stat().st_mode) == 0o700
    assert stat.S_IMODE((adjudication / "private/adjudication_binding.csv").stat().st_mode) == 0o600
    third_rows[0]["quality"] = "2"
    third_completed = tmp_path / "third_completed.csv"
    _write_csv(third_completed, third_rows, workflow.ADJUDICATION_FIELDS)

    final = tmp_path / "final"
    adjudication_sha256 = workflow.sha256_file(
        adjudication / "adjudication_manifest.json"
    )
    frozen = workflow.finalize_reviews(
        package_root=package,
        expected_package_manifest_sha256=package_sha256,
        adjudication_root=adjudication,
        expected_adjudication_manifest_sha256=adjudication_sha256,
        third_completed=third_completed,
        output_root=final,
    )
    assert frozen["row_count"] == 192
    assert frozen["resolved_atomic_disagreements"] == 1
    assert frozen["training_authorized"] is False
    canonical_scores = workflow.read_csv_exact(
        final / "canonical_adjudicated.csv",
        workflow.FROZEN_REVIEW_FIELDS,
        "canonical adjudicated",
    )
    assert [row["generation_id"] for row in canonical_scores] == [
        row["generation_id"] for row in canonical
    ]
    assert all(row["decodable"] == "1" for row in canonical_scores)
    assert all(row[field] == "2" for row in canonical_scores for field in workflow.ORDINAL_FIELDS)
    assert stat.S_IMODE(final.stat().st_mode) == 0o700
    assert stat.S_IMODE((final / "canonical_adjudicated.csv").stat().st_mode) == 0o600

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
    assert aggregate["aggregate"]["eligible_rows"] == 192
    assert aggregate["authorization"]["training_authorized"] is False

    with pytest.raises(ValueError, match="already exists"):
        workflow.build_package(
            project_root=PROJECT_ROOT,
            generation_manifest=generation,
            expected_generation_sha256=workflow.sha256_file(generation),
            canonical_path=CANONICAL,
            template_path=TEMPLATE,
            rubric_path=RUBRIC,
            freeze_path=FREEZE,
            output_root=package,
            frame_loader=lambda _path: [object()] * 49,
            composite_renderer=_fake_render,
        )


def test_completed_review_rejects_context_drift(tmp_path):
    blank = [
        {
            **{field: "context" for field in workflow.ASSIGNMENT_CONTEXT_FIELDS},
            **{field: "" for field in workflow.SCORE_FIELDS},
            "reviewer_notes": "",
        }
    ] * 192
    for index, row in enumerate(blank):
        row = dict(row)
        row["assignment_position"] = str(index)
        row["anonymous_review_id"] = f"a{index:03d}"
        blank[index] = row
    assignment = tmp_path / "assignment.csv"
    _write_csv(assignment, blank, workflow.ASSIGNMENT_FIELDS)
    completed = [dict(row) for row in blank]
    for row in completed:
        row["decodable"] = "1"
        for field in workflow.ORDINAL_FIELDS:
            row[field] = "2"
    completed[0]["prompt"] = "changed"
    completed_path = tmp_path / "completed.csv"
    _write_csv(completed_path, completed, workflow.ASSIGNMENT_FIELDS)
    with pytest.raises(ValueError, match="changed frozen context prompt"):
        workflow.validate_completed_review(completed_path, assignment, "completed")


def test_median_of_three_matches_boolean_majority_and_ordinal_median():
    assert workflow.median_of_three("0", "1", "1") == "1"
    assert workflow.median_of_three("2", "0", "1") == "1"
    assert workflow.median_of_three("2", "1", "2") == "2"


def test_real_renderer_uses_registered_frame_indices(tmp_path):
    from PIL import Image

    frames = [Image.new("RGB", (832, 480), (index * 5 % 256, index * 3 % 256, index * 7 % 256)) for index in range(49)]
    output = tmp_path / "full49.jpg"
    workflow.render_full49_composite(frames, output)
    rendered = Image.open(output).convert("RGB")
    assert rendered.size == (1344, 1440)

    frame_width = 192
    frame_height = round(480 * frame_width / 832)
    panel_height = 26 + 2 * (frame_height + 20)
    for panel_index, (start, end) in enumerate(workflow.PANEL_RANGES):
        for local_index, source_index in ((0, start), (end - start, end)):
            x = (local_index % 7) * frame_width + frame_width // 2
            y = panel_index * panel_height + 26 + (local_index // 7) * (frame_height + 20) + 20 + frame_height // 2
            observed = rendered.getpixel((x, y))
            expected = (source_index * 5 % 256, source_index * 3 % 256, source_index * 7 % 256)
            assert all(abs(left - right) <= 4 for left, right in zip(observed, expected))


def test_atomic_exposure_never_replaces_existing_destination(tmp_path):
    destination = tmp_path / "destination"
    partial = workflow.prepare_partial_root(destination)
    marker = destination / "marker"
    destination.mkdir()
    marker.write_text("keep\n", encoding="utf-8")
    with pytest.raises(OSError):
        workflow.expose_partial_root(partial, destination)
    assert marker.read_text(encoding="utf-8") == "keep\n"
    assert partial.is_dir()
    workflow._safe_cleanup_partial(partial)
    assert not partial.exists()


def test_build_failure_rolls_back_owned_partial(tmp_path):
    generation, _ = _fake_generation(tmp_path)
    output = tmp_path / "failed-package"
    calls = 0

    def fail_second(_frames, path):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected render failure")
        _fake_render(_frames, path)

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


def test_sealed_parent_symlink_is_rejected_before_completed_review_read(tmp_path):
    safe_assignment = tmp_path / "assignment.csv"
    rows = [
        {
            **{field: "context" for field in workflow.ASSIGNMENT_CONTEXT_FIELDS},
            **{field: ("1" if field == "decodable" else "2") for field in workflow.SCORE_FIELDS},
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
    with pytest.raises(ValueError, match="sealed-final36|symlink component"):
        workflow.validate_completed_review(alias / "completed.csv", safe_assignment, "completed")
