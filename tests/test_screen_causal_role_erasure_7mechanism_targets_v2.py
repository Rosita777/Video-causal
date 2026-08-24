from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts import screen_causal_role_erasure_7mechanism_targets_v2 as screening
from scripts import run_causal_role_erasure_7mechanism_targets_v2 as target_runner


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_CANDIDATES = (
    REPO_ROOT
    / "data/causal_role_erasure_7mechanism_main_v2/target_candidates.csv"
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


def _fixture(tmp_path: Path):
    project = tmp_path / "project"
    project.mkdir()
    fields, rows = _read_csv(SOURCE_CANDIDATES)
    items = []
    for row in rows:
        if row["target_origin"] == "new_generation_v2":
            logical_relative = (
                Path("new_videos") / f"{row['candidate_id']}.mp4"
            )
            actual_relative = (
                Path("new_videos")
                / f"{int(row['global_index']):04d}_generated-slug_seed{row['seed']}.mp4"
            )
            path = project / actual_relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((row["candidate_id"] + "\n").encode())
            row["target_video_path"] = logical_relative.as_posix()
            items.append(
                {
                    "global_index": int(row["global_index"]),
                    "candidate_id": row["candidate_id"],
                    "mechanism": row["mechanism"],
                    "seed": int(row["seed"]),
                    "prompt": row["target_prompt"],
                    "video_path": actual_relative.as_posix(),
                    "video_sha256": screening.sha256_file(path),
                    "size_bytes": path.stat().st_size,
                    "media": screening.expected_media(),
                }
            )
        else:
            relative = Path("water_videos") / f"{row['candidate_id']}.mp4"
            row["target_video_path"] = relative.as_posix()
            if row["historical_screen_status"] == "accept":
                path = project / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes((row["candidate_id"] + "\n").encode())
    candidates = project / "target_candidates.csv"
    _write_csv(candidates, fields, rows)
    aggregate = {
        "schema_version": 1,
        "protocol_version": screening.PROTOCOL_VERSION,
        "status": "completed",
        "target_candidates_sha256": screening.sha256_file(candidates),
        "video_binding_key": "candidate_id",
        "expected_videos": screening.EXPECTED_NEW_VIDEOS,
        "validated_videos": screening.EXPECTED_NEW_VIDEOS,
        "media_contract": screening.expected_media(),
        "items": items,
    }
    aggregate_path = project / "generation_aggregate.json"
    aggregate_path.write_text(json.dumps(aggregate) + "\n", encoding="utf-8")

    water = [row for row in rows if row["mechanism"] == screening.WATER_MECHANISM]
    screen_rows = [
        {
            "pair_id": row["historical_pair_id"],
            "video_path": row["target_video_path"],
            "final_status": row["historical_screen_status"],
        }
        for row in water
    ]
    selected_rows = [
        {
            "pair_id": row["historical_pair_id"],
            "desired_target_video": row["target_video_path"],
        }
        for row in water
        if row["historical_screen_status"] == "accept"
    ]
    screen_path = project / "water_screen.csv"
    selected_path = project / "water_selected.csv"
    _write_csv(screen_path, ("pair_id", "video_path", "final_status"), screen_rows)
    _write_csv(selected_path, ("pair_id", "desired_target_video"), selected_rows)
    return project, candidates, aggregate_path, screen_path, selected_path


def _fake_frame_loader(_path: Path):
    return [object()] * screening.FRAME_COUNT, screening.expected_media()


def _fake_renderer(frames, path: Path):
    assert len(frames) == screening.FRAME_COUNT
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake-full49-composite")


def _complete_scores(template: Path, output: Path, fill: str = "2"):
    fields, rows = _read_csv(template)
    for row in rows:
        for field in screening.SCORE_FIELDS:
            row[field] = fill
    _write_csv(output, fields, rows)


def test_candidate_graph_contract_and_fixed_selection_rule():
    fields, rows = screening.load_and_validate_candidates(SOURCE_CANDIDATES)
    assert tuple(fields[: len(screening.CANDIDATE_REQUIRED_FIELDS)]) == screening.CANDIDATE_REQUIRED_FIELDS
    assert len(rows) == 1344
    assert screening.eligible_from_scores({field: 2 for field in screening.SCORE_FIELDS})
    assert not screening.eligible_from_scores(
        {**{field: 2 for field in screening.SCORE_FIELDS}, "trigger_absent": 1}
    )
    assert screening.selection_hash(rows[0]["candidate_id"]) == screening.selection_hash(
        rows[0]["candidate_id"]
    )


def test_generation_binding_rejects_hash_drift(tmp_path: Path):
    project, candidates, aggregate, _, _ = _fixture(tmp_path)
    payload = json.loads(aggregate.read_text())
    payload["items"][0]["video_sha256"] = "0" * 64
    aggregate.write_text(json.dumps(payload) + "\n")
    with pytest.raises(screening.ScreeningError, match="video SHA-256 mismatch"):
        screening.load_and_validate_generation_aggregate(
            aggregate,
            candidates_path=candidates,
            candidate_rows=screening.load_and_validate_candidates(candidates)[1],
            project_root=project,
        )


def test_logical_candidate_filename_may_differ_from_canonical_slug_filename(
    tmp_path: Path,
):
    project, candidates, aggregate, _, _ = _fixture(tmp_path)
    _, rows = screening.load_and_validate_candidates(candidates)
    payload, normalized = screening.load_and_validate_generation_aggregate(
        aggregate,
        candidates_path=candidates,
        candidate_rows=rows,
        project_root=project,
    )
    first = next(row for row in rows if row["target_origin"] == "new_generation_v2")
    item = normalized[first["candidate_id"]]
    assert Path(first["target_video_path"]).name == f"{first['candidate_id']}.mp4"
    assert Path(item["video_path"]).name != Path(first["target_video_path"]).name
    assert Path(item["video_path"]).parent == Path(first["target_video_path"]).parent
    assert payload["validated_videos"] == 1152


def test_manifest_video_cannot_escape_registered_logical_videos_directory(
    tmp_path: Path,
):
    project, candidates, aggregate, _, _ = _fixture(tmp_path)
    payload = json.loads(aggregate.read_text())
    first = payload["items"][0]
    escaped_relative = Path("escaped_videos") / Path(first["video_path"]).name
    escaped = project / escaped_relative
    escaped.parent.mkdir()
    escaped.write_bytes((project / first["video_path"]).read_bytes())
    first["video_path"] = escaped_relative.as_posix()
    first["video_sha256"] = screening.sha256_file(escaped)
    first["size_bytes"] = escaped.stat().st_size
    aggregate.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(
        screening.ScreeningError,
        match="logical target_video_path escaped registered videos directory",
    ):
        screening.load_and_validate_generation_aggregate(
            aggregate,
            candidates_path=candidates,
            candidate_rows=screening.load_and_validate_candidates(candidates)[1],
            project_root=project,
        )


def test_real_runner_native_generation_schema_is_strictly_normalized(tmp_path: Path):
    assert screening.RUNNER_NATIVE_ID == target_runner.RUNNER_ID
    assert screening.WATER_REUSE_MANIFEST_SHA256 == target_runner.EXPECTED_WATER_REUSE_MANIFEST_SHA256
    assert screening.RUNNER_NATIVE_GENERATION == {
        "baseline": "negative_prompt",
        "num_inference_steps": target_runner.STEPS,
        "guidance_scale": target_runner.GUIDANCE_SCALE,
        "num_frames": target_runner.NUM_FRAMES,
        "fps": target_runner.FPS,
        "height": target_runner.HEIGHT,
        "width": target_runner.WIDTH,
        "dtype": target_runner.DTYPE,
        "skip_existing": False,
        "resume": False,
        "per_prompt_seeds": "explicit_from_target_candidates_csv",
    }
    project, candidates, aggregate, _, _ = _fixture(tmp_path)
    canonical = json.loads(aggregate.read_text())
    native_items = []
    candidate_by_id = {
        row["candidate_id"]: row
        for row in screening.load_and_validate_candidates(candidates)[1]
    }
    for item in canonical["items"]:
        row = candidate_by_id[item["candidate_id"]]
        native_items.append(
            {
                "global_index": item["global_index"],
                "candidate_id": item["candidate_id"],
                "mechanism": item["mechanism"],
                "prompt_shard_index": int(row["prompt_shard_index"]),
                "target_prompt": item["prompt"],
                "seed": item["seed"],
                "video_path": item["video_path"],
                "video_sha256": item["video_sha256"],
                "size_bytes": item["size_bytes"],
                "media": {
                    "decoded_frames": 49,
                    "fps": "8/1",
                    "width": 832,
                    "height": 480,
                },
            }
        )
    native = {
        "schema_version": 1,
        "protocol_id": screening.PROTOCOL_VERSION,
        "runner_id": screening.RUNNER_NATIVE_ID,
        "status": "frozen_after_exact_1152_video_validation",
        "target_candidates_sha256": screening.sha256_file(candidates),
        "water_reuse_manifest_sha256": screening.WATER_REUSE_MANIFEST_SHA256,
        "water_reuse_rows": 192,
        "baseline": "negative_prompt",
        "generation": screening.RUNNER_NATIVE_GENERATION,
        "video_count": 1152,
        "items": native_items,
    }
    aggregate.write_text(json.dumps(native) + "\n", encoding="utf-8")
    _, normalized = screening.load_and_validate_generation_aggregate(
        aggregate,
        candidates_path=candidates,
        candidate_rows=list(candidate_by_id.values()),
        project_root=project,
    )
    first = normalized[native_items[0]["candidate_id"]]
    assert first["prompt"] == native_items[0]["target_prompt"]
    assert first["media"] == screening.expected_media()


def test_fewer_than_178_eligible_fails_closed():
    _, candidates = screening.load_and_validate_candidates(SOURCE_CANDIDATES)
    new_rows = [row for row in candidates if row["target_origin"] == "new_generation_v2"]
    scores = {
        row["candidate_id"]: {field: 2 for field in screening.SCORE_FIELDS}
        for row in new_rows
    }
    rigid = [row for row in new_rows if row["mechanism"] == "rigid_collision"]
    for row in rigid[:15]:
        scores[row["candidate_id"]]["source_absent"] = 1
    with pytest.raises(screening.ScreeningError, match="only 177 eligible targets"):
        screening.select_new_targets(candidates, scores)


def test_full_blind_review_adjudication_and_178_selection(tmp_path: Path):
    project, candidates, aggregate, water_screen, water_selected = _fixture(tmp_path)
    package_root = project / "review_package"
    package = screening.build_review_package(
        project_root=project,
        candidates_path=candidates,
        generation_aggregate_path=aggregate,
        output_root=package_root,
        frame_loader=_fake_frame_loader,
        renderer=_fake_renderer,
    )
    assert package["nonwater_candidate_count"] == 1152
    screening.verify_review_package(package_root, package["sha256"])

    delivered = next((package_root / "public/reviewer_a/media").iterdir())
    original_bytes = delivered.read_bytes()
    delivered.write_bytes(original_bytes + b"drift")
    with pytest.raises(screening.ScreeningError, match="delivered composite drift"):
        screening.verify_review_package(package_root, package["sha256"])
    delivered.write_bytes(original_bytes)

    a_assignment = _read_csv(package_root / "public/reviewer_a/assignment.csv")[1]
    b_assignment = _read_csv(package_root / "public/reviewer_b/assignment.csv")[1]
    for mechanism in screening.NEW_MECHANISMS:
        a_ids = [row["composite_path"] for row in a_assignment if row["mechanism"] == mechanism]
        b_ids = [row["composite_path"] for row in b_assignment if row["mechanism"] == mechanism]
        assert len(a_ids) == len(b_ids) == 192
        assert a_ids != b_ids
    assert not any("candidate_id" in row for row in a_assignment)

    a_completed = project / "reviewer_a_completed.csv"
    b_completed = project / "reviewer_b_completed.csv"
    _complete_scores(package_root / "public/reviewer_a/scoring_template.csv", a_completed)
    _complete_scores(package_root / "public/reviewer_b/scoring_template.csv", b_completed)

    # Introduce one A/B field disagreement on a known common candidate.
    b_fields, b_rows = _read_csv(b_completed)
    b_binding = _read_csv(package_root / "private/reviewer_b_binding.csv")[1]
    target_candidate = b_binding[0]["candidate_id"]
    b_rows[0]["natural_motion"] = "1"
    _write_csv(b_completed, b_fields, b_rows)

    adjudication_root = project / "adjudication"
    adjudication = screening.plan_adjudication(
        package_root=package_root,
        expected_package_sha256=package["sha256"],
        reviewer_a_completed=a_completed,
        reviewer_b_completed=b_completed,
        output_root=adjudication_root,
    )
    assert adjudication["videos_with_disagreement"] == 1
    assert adjudication["atomic_disagreement_count"] == 1
    third_template = adjudication_root / "public/third_reviewer/scoring_template.csv"
    third_fields, third_rows = _read_csv(third_template)
    assert len(third_rows) == 1
    assert third_rows[0]["requested_fields"] == "natural_motion"
    third_rows[0]["natural_motion"] = "2"
    third_completed = project / "third_completed.csv"
    _write_csv(third_completed, third_fields, third_rows)

    final_root = project / "selection"
    result = screening.finalize_screening(
        project_root=project,
        package_root=package_root,
        expected_package_sha256=package["sha256"],
        adjudication_root=adjudication_root,
        expected_adjudication_sha256=adjudication["sha256"],
        third_completed=third_completed,
        water_screen_csv=water_screen,
        water_selected_csv=water_selected,
        water_media_root=project,
        output_root=final_root,
        water_frame_loader=_fake_frame_loader,
    )
    assert result["selected_total"] == 7 * 178
    assert result["eligible_counts"] == {
        mechanism: (178 if mechanism == screening.WATER_MECHANISM else 192)
        for mechanism in screening.MECHANISM_ORDER
    }
    selected_fields, selected = _read_csv(final_root / "selected_targets.csv")
    assert len(selected) == 1246
    assert set(row["mechanism"] for row in selected) == set(screening.MECHANISM_ORDER)
    assert len({row["candidate_id"] for row in selected}) == 1246
    canonical = {
        row["candidate_id"]: row
        for row in _read_csv(final_root / "canonical_nonwater_scores.csv")[1]
    }
    assert canonical[target_candidate]["natural_motion"] == "2"
    assert canonical[target_candidate]["resolution"] == "median_of_three:natural_motion"
    assert "selected_video_sha256" in selected_fields
