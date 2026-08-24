from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_causal_role_erasure_7mechanism_targets_v2 as runner  # noqa: E402


def _prompt(mechanism: str, index: int) -> str:
    return f"A clean target-only scene for {mechanism} candidate {index}."


def _build_inputs(tmp_path: Path, monkeypatch) -> tuple[Path, Path, Path]:
    project = tmp_path / "project"
    (project / "scripts").mkdir(parents=True)
    (project / "prompts/causal_role_erasure_7mechanism_main_v2").mkdir(parents=True)
    (project / "data/causal_role_erasure_7mechanism_main_v2").mkdir(parents=True)
    (project / "scripts/generate_wan_clean.py").write_text("# frozen fake generator\n", encoding="utf-8")

    rows: list[dict[str, str]] = []
    water_items: list[dict[str, object]] = []
    for mechanism_index, mechanism in enumerate(runner.MECHANISM_ORDER):
        shard_relative = (
            f"prompts/causal_role_erasure_7mechanism_main_v2/"
            f"target_candidates_{mechanism}.prompts"
        )
        if mechanism != "water_impact":
            shard = project / shard_relative
            shard.write_text(
                "".join(
                    f"{_prompt(mechanism, local)} | exclude {mechanism} | receiver remains intact {local}\n"
                    for local in range(runner.ROWS_PER_MECHANISM)
                ),
                encoding="utf-8",
            )
        for local in range(runner.ROWS_PER_MECHANISM):
            global_index = mechanism_index * runner.ROWS_PER_MECHANISM + local
            prompt = _prompt(mechanism, local)
            seed = 26000 + local if mechanism == "water_impact" else 1_100_000 + global_index
            rows.append(
                {
                    "protocol_version": runner.PROTOCOL_ID,
                    "candidate_id": f"target7m_{global_index:04d}",
                    "global_index": str(global_index),
                    "mechanism": mechanism,
                    "seed": str(seed),
                    "target_prompt": prompt,
                    "target_origin": (
                        runner.WATER_ORIGIN if mechanism == "water_impact" else runner.NEW_ORIGIN
                    ),
                    "prompt_shard": "" if mechanism == "water_impact" else shard_relative,
                    "prompt_shard_index": "" if mechanism == "water_impact" else str(local),
                    "expected_counterfactual_state": f"receiver remains intact {local}",
                    "target_video_path": (
                        f"outputs/water_impact_dynamic_v1/train_targets_v1/videos/"
                        f"water_{local:03d}.mp4"
                        if mechanism == "water_impact"
                        else (
                            "outputs/causal_role_erasure_7mechanism_main_v2/"
                            f"target_generation/{mechanism}/videos/"
                            f"target7m_{global_index:04d}.mp4"
                        )
                    ),
                }
            )
            if mechanism == "water_impact":
                water_items.append(
                    {
                        "index": local,
                        "prompt": prompt,
                        "target_concept": "falling object, water impact, splash, ripple",
                        "expected_effect": f"receiver remains intact {local}",
                        "seed": seed,
                        "video_path": str(
                            project
                            / "outputs/water_impact_dynamic_v1/train_targets_v1/videos"
                            / f"water_{local:03d}.mp4"
                        ),
                        "negative_prompt": "falling object, water impact, splash, ripple",
                    }
                )

    candidates = project / runner.DEFAULT_TARGET_CANDIDATES
    with candidates.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    water = project / "historical/water_generation_manifest.json"
    water.parent.mkdir()
    water.write_text(
        json.dumps(
            {
                "baseline": "negative_prompt",
                "pipeline": "WanPipeline",
                "model": "models/Wan2.1-T2V-1.3B-Diffusers",
                "dry_run": False,
                "generation": {"baseline": "negative_prompt"},
                "items": water_items,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(runner, "EXPECTED_WATER_REUSE_MANIFEST_SHA256", runner.sha256_file(water))
    return project, candidates, water


def _load_and_plan(tmp_path: Path, monkeypatch, *, dry_run: bool = True):
    project, candidates, water = _build_inputs(tmp_path, monkeypatch)
    rows, shards, schema = runner.load_target_candidates(project, candidates, water)
    output = project / runner.DEFAULT_OUTPUT_ROOT
    generator = project / runner.DEFAULT_GENERATOR
    plan = runner.build_plan(
        project_root=project,
        rows=rows,
        schema=schema,
        target_candidates=candidates,
        water_reuse_manifest=water,
        output_root=output,
        generator=generator,
        python_executable=project / runner.DEFAULT_PYTHON,
        model=project / runner.DEFAULT_MODEL,
        gpus=[4, 5, 6, 7],
        dry_run=dry_run,
    )
    return project, candidates, water, output, rows, shards, plan


def test_loader_binds_1344_rows_skips_water_and_six_shards(tmp_path: Path, monkeypatch):
    _, _, _, _, rows, shards, _ = _load_and_plan(tmp_path, monkeypatch)
    assert len(rows) == 1344
    assert len(shards) == 6
    assert [row["global_index"] for row in rows] == list(range(1344))
    assert {row["target_origin"] for row in rows[:192]} == {runner.WATER_ORIGIN}
    assert {row["target_origin"] for row in rows[192:]} == {runner.NEW_ORIGIN}
    assert all(row["prompt_shard_index"] is None for row in rows[:192])
    assert all(len(items) == 192 for items in shards.values())


def test_dry_run_plan_is_negative_prompt_four_plus_two_and_fresh_only(
    tmp_path: Path, monkeypatch
):
    project, candidates, water, output, _, _, _ = _load_and_plan(tmp_path, monkeypatch)
    result = runner.main(
        [
            "--project-root", str(project),
            "--target-candidates", str(candidates),
            "--water-reuse-manifest", str(water),
            "--output-root", str(output),
            "--gpus", "4,5,6,7",
            "--dry-run",
        ]
    )
    assert result == 0
    plan = json.loads((output / "target_generation_run_manifest.json").read_text())
    aggregate = json.loads((output / "target_generation_aggregate.json").read_text())
    assert plan["inputs"]["candidate_rows"] == 1344
    assert plan["inputs"]["water_reuse_rows"] == 192
    assert plan["inputs"]["fresh_generation_rows"] == 1152
    assert plan["generation"]["baseline"] == "negative_prompt"
    assert plan["scheduler"]["jobs_per_wave"] == [4, 2]
    assert [(job["wave_index"], job["gpu"]) for job in plan["jobs"]] == [
        (0, 4), (0, 5), (0, 6), (0, 7), (1, 4), (1, 5)
    ]
    assert [job["mechanism"] for job in plan["jobs"]] == list(runner.GENERATED_MECHANISMS)
    for job in plan["jobs"]:
        command = job["command"]
        assert Path(command[1]).resolve() == (project / "scripts/generate_wan_clean.py").resolve()
        assert command[command.index("--baseline") + 1] == "negative_prompt"
        assert len(command[command.index("--seeds") + 1].split(",")) == 192
        assert "--skip-existing" not in command
        assert "--dry-run" not in command
        assert not Path(job["output_dir"]).exists()
    assert aggregate["status"] == "planned"
    assert aggregate["baseline"] == "negative_prompt"
    assert aggregate["expected_generated_rows"] == 1152
    assert runner.main(
        [
            "--project-root", str(project),
            "--target-candidates", str(candidates),
            "--water-reuse-manifest", str(water),
            "--output-root", str(output),
            "--gpus", "4,5,6,7",
            "--dry-run",
        ]
    ) == 1


def test_loader_rejects_wrong_nonwater_origin(tmp_path: Path, monkeypatch):
    project, candidates, water = _build_inputs(tmp_path, monkeypatch)
    rows = list(csv.DictReader(candidates.open(newline="", encoding="utf-8")))
    rows[192]["target_origin"] = runner.WATER_ORIGIN
    with candidates.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="target_origin mismatch"):
        runner.load_target_candidates(project, candidates, water)


def test_supported_aliases_are_canonicalized(tmp_path: Path, monkeypatch):
    project, candidates, water = _build_inputs(tmp_path, monkeypatch)
    rows = list(csv.DictReader(candidates.open(newline="", encoding="utf-8")))
    for row in rows:
        row["candidate_index"] = row.pop("global_index")
        row["target_asset_mode"] = (
            "historical_water_v1_import_candidate"
            if row.pop("target_origin") == runner.WATER_ORIGIN
            else "generate_new"
        )
    with candidates.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    loaded, _, schema = runner.load_target_candidates(project, candidates, water)
    assert loaded[0]["target_origin"] == runner.WATER_ORIGIN
    assert loaded[192]["target_origin"] == runner.NEW_ORIGIN
    assert schema["aliases_used"] == ["candidate_index", "target_asset_mode"]


def test_formal_run_rejects_alias_only_schema_before_output_or_gpu(
    tmp_path: Path, monkeypatch
):
    project, candidates, water = _build_inputs(tmp_path, monkeypatch)
    rows = list(csv.DictReader(candidates.open(newline="", encoding="utf-8")))
    for row in rows:
        row["candidate_index"] = row.pop("global_index")
        row["target_asset_mode"] = (
            "historical_water_v1_import_candidate"
            if row.pop("target_origin") == runner.WATER_ORIGIN
            else "generate_new"
        )
    with candidates.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    runtime = project / runner.DEFAULT_PYTHON
    runtime.parent.mkdir(parents=True)
    runtime.write_text("#!/bin/sh\nexit 99\n", encoding="utf-8")
    runtime.chmod(0o755)
    (project / runner.DEFAULT_MODEL).mkdir(parents=True)
    output = project / runner.DEFAULT_OUTPUT_ROOT
    assert runner.main(
        [
            "--project-root", str(project),
            "--target-candidates", str(candidates),
            "--water-reuse-manifest", str(water),
            "--output-root", str(output),
            "--gpus", "0,1,2,3",
            "--run",
        ]
    ) == 1
    assert not output.exists()


def _fake_generation(job: dict, plan: dict) -> None:
    output = Path(job["output_dir"])
    videos = output / "videos"
    videos.mkdir(parents=True)
    items = []
    for index in range(192):
        video = videos / f"{index:03d}_seed{job['seeds'][index]}.mp4"
        video.write_bytes(f"video-{job['candidate_ids'][index]}".encode())
        items.append(
            {
                "index": index,
                "prompt": job["target_prompts"][index],
                "target_concept": job["target_concepts"][index],
                "expected_effect": job["expected_effects"][index],
                "seed": job["seeds"][index],
                "negative_prompt": job["target_concepts"][index],
                "video_path": str(video),
            }
        )
    generation = {
        **runner._expected_generation_fields(),
        "seed": job["seeds"][0],
        "seeds": job["seeds"],
    }
    (output / "generation_manifest.json").write_text(
        json.dumps(
            {
                "baseline": "negative_prompt",
                "pipeline": "WanPipeline",
                "model": plan["implementation"]["model"],
                "dry_run": False,
                "prompts": job["prompt_shard"],
                "generation": generation,
                "items": items,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _media(_python: Path, _video: Path) -> dict[str, object]:
    return {"decoded_frames": 49, "fps": "8/1", "width": 832, "height": 480}


def test_exact_1152_validation_binds_ids_seeds_prompts_media_and_hashes(
    tmp_path: Path, monkeypatch
):
    _, _, _, output, _, _, plan = _load_and_plan(tmp_path, monkeypatch, dry_run=False)
    runner.prepare_output_root(output, plan)
    for job in plan["jobs"]:
        _fake_generation(job, plan)
        validation = runner.validate_job_outputs(job, plan, media_probe=_media)
        assert validation["validated_video_count"] == 192
        assert validation["outputs"][0]["candidate_id"] == job["candidate_ids"][0]
        assert validation["outputs"][0]["seed"] == job["seeds"][0]
        assert validation["outputs"][0]["logical_expected_video_path"] == job["expected_video_paths"][0]
        assert validation["outputs"][0]["video_path"] != job["expected_video_paths"][0]
        assert len(validation["outputs"][0]["video_sha256"]) == 64
        runner.write_json_atomic(Path(job["status_path"]), runner._status(job, "completed", **validation))
    manifest = runner.write_generation_manifest(output, plan)
    payload = json.loads(manifest.read_text())
    assert payload["baseline"] == "negative_prompt"
    assert payload["video_count"] == 1152
    assert len(payload["items"]) == 1152
    assert all(set(item) == set(runner.FINAL_ITEM_FIELDS) for item in payload["items"])
    assert all("logical_expected_video_path" not in item for item in payload["items"])
    assert [item["global_index"] for item in payload["items"]] == list(range(192, 1344))
    assert len({item["candidate_id"] for item in payload["items"]}) == 1152
    assert all(item["media"] == _media(Path(), Path()) for item in payload["items"])


def test_output_validation_rejects_prompt_or_seed_drift(tmp_path: Path, monkeypatch):
    _, _, _, output, _, _, plan = _load_and_plan(tmp_path, monkeypatch, dry_run=False)
    runner.prepare_output_root(output, plan)
    job = plan["jobs"][0]
    _fake_generation(job, plan)
    manifest = Path(job["output_dir"]) / "generation_manifest.json"
    payload = json.loads(manifest.read_text())
    payload["items"][0]["prompt"] += " drift"
    manifest.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="prompt mismatch"):
        runner.validate_job_outputs(job, plan, media_probe=_media)


def _prepare_exact_path_binding_failure(output: Path, plan: dict) -> dict[str, str]:
    runner.prepare_output_root(output, plan)
    original_hashes = {}
    failures = []
    for job in plan["jobs"]:
        if job["wave_index"] != 0:
            continue
        _fake_generation(job, plan)
        manifest = Path(job["output_dir"]) / "generation_manifest.json"
        original_hashes[job["mechanism"]] = runner.sha256_file(manifest)
        error = f"{job['mechanism']} item 0: target_video_path mismatch"
        runner.write_json_atomic(
            Path(job["status_path"]),
            runner._status(job, "failed", return_code=0, error=error),
        )
        failures.append(f"{job['mechanism']}: ValueError: {error}")
    runner.write_aggregate(output, plan, "failed", "; ".join(failures))
    return original_hashes


def test_recover_path_binding_revalidates_wave0_without_regeneration_then_runs_only_wave1(
    tmp_path: Path, monkeypatch
):
    project, _, _, output, _, _, plan = _load_and_plan(
        tmp_path, monkeypatch, dry_run=False
    )
    runtime = project / runner.DEFAULT_PYTHON
    runtime.parent.mkdir(parents=True)
    runtime.write_text("#!/bin/sh\n", encoding="utf-8")
    runtime.chmod(0o755)
    (project / runner.DEFAULT_MODEL).mkdir(parents=True)
    original_hashes = _prepare_exact_path_binding_failure(output, plan)
    launched = []

    class Done:
        def poll(self):
            return 0

        def terminate(self):
            raise AssertionError("completed fake process must not be terminated")

        def kill(self):
            raise AssertionError("completed fake process must not be killed")

    def fake_popen(command, **_kwargs):
        job = next(job for job in plan["jobs"] if job["command"] == command)
        launched.append(job["mechanism"])
        assert job["wave_index"] == 1
        _fake_generation(job, plan)
        return Done()

    runner.execute_path_binding_recovery(
        plan,
        output,
        poll_interval=0.001,
        popen_factory=fake_popen,
        sleep_fn=lambda _seconds: None,
        media_probe=_media,
    )
    assert launched == ["material_release", "surface_trace"]
    assert all(runner.read_status(job)["status"] == "completed" for job in plan["jobs"])
    for job in plan["jobs"][:4]:
        assert runner.read_status(job)["recovered_without_regeneration"] is True
        assert runner.sha256_file(Path(job["output_dir"]) / "generation_manifest.json") == original_hashes[job["mechanism"]]
    receipt = json.loads((output / "path_binding_recovery_receipt.json").read_text())
    assert receipt["wave0_regenerated"] is False
    assert receipt["wave0_validated_video_count"] == 768
    assert len(receipt["wave0"]) == 4
    final = json.loads((output / "target_generation_manifest.json").read_text())
    aggregate = json.loads((output / "target_generation_aggregate.json").read_text())
    assert final["video_count"] == 1152
    assert "path_binding_recovery_receipt" not in final
    assert set(final) == {
        "schema_version", "protocol_id", "runner_id", "status",
        "target_candidates_sha256", "water_reuse_manifest_sha256",
        "water_reuse_rows", "baseline", "generation", "video_count", "items",
    }
    assert aggregate["status"] == "completed"
    assert aggregate["validated_generated_rows"] == 1152
    assert aggregate["path_binding_recovery_receipt"]["sha256"] == runner.sha256_file(
        output / "path_binding_recovery_receipt.json"
    )
    assert all(set(item) == set(runner.FINAL_ITEM_FIELDS) for item in final["items"])
    assert all("logical_expected_video_path" not in item for item in final["items"])
    assert all(
        runner.read_status(job)["outputs"][0]["logical_expected_video_path"]
        for job in plan["jobs"]
    )


def test_recovery_rejects_nonzero_wave0_or_touched_wave1_before_launch(
    tmp_path: Path, monkeypatch
):
    project, _, _, output, _, _, plan = _load_and_plan(
        tmp_path, monkeypatch, dry_run=False
    )
    _prepare_exact_path_binding_failure(output, plan)
    first = plan["jobs"][0]
    status = runner.read_status(first)
    status["return_code"] = 1
    runner.write_json_atomic(Path(first["status_path"]), status)
    with pytest.raises(ValueError, match="return_code=0"):
        runner.validate_path_binding_recovery_state(plan, output, media_probe=_media)
    assert not (output / "path_binding_recovery_receipt.json").exists()


def test_sealed_path_is_rejected_before_input_read(tmp_path: Path, monkeypatch):
    called = False

    def forbidden(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("loader must not see sealed path")

    monkeypatch.setattr(runner, "load_target_candidates", forbidden)
    result = runner.main(
        [
            "--project-root", str(tmp_path),
            "--target-candidates", "data/sealed-final36.csv",
            "--water-reuse-manifest", "water.json",
            "--output-root", "outputs/plan",
            "--gpus", "0,1,2,3",
            "--dry-run",
        ]
    )
    assert result == 1
    assert called is False
    assert not (tmp_path / "outputs/plan").exists()
