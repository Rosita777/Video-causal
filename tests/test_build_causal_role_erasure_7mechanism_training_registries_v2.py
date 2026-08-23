from __future__ import annotations

import csv
import hashlib
import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_causal_role_erasure_7mechanism_main_v2 as main_data  # noqa: E402
import build_causal_role_erasure_7mechanism_training_registries_v2 as registries  # noqa: E402
import prepare_causal_role_erasure_7mechanism_training_cache_v2 as cache  # noqa: E402
import train_wan_causal_role_lora_v2 as trainer  # noqa: E402


def _write_csv(path: Path, rows: list[dict], fields=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields or rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _sha(path: Path) -> str:
    return registries.sha256_file(path)


def _fixture(tmp_path: Path) -> dict[str, Path]:
    project = tmp_path / "project"
    project.mkdir()
    data = project / "data"
    data.mkdir()
    prompts = project / "prompts"
    prompts.mkdir()

    ontology = main_data.build_ontologies(
        ROOT / "data/water_impact_dynamic_v4/source_bank_public64_registry_v2.json"
    )
    ontology_path = data / "ontology_registry.json"
    ontology_path.write_bytes(main_data.canonical_json_bytes(ontology))
    targets, _prompt_payloads = main_data.build_target_candidates(
        ontology,
        water_train_pairs_path=ROOT / "data/water_impact_dynamic_v1/train_pairs.csv",
        water_screen_path=ROOT / "data/water_impact_dynamic_v1/train_targets_v1_screen_final.csv",
        prompt_dir=prompts,
    )
    videos = project / "selected-videos"
    videos.mkdir()
    selected: list[dict] = []
    selected_fields = (
        *main_data.TARGET_FIELDS,
        "eligible", "selection_hash", "selection_rank",
        "selected_video_path", "selected_video_sha256", "selected_media",
    )
    for mechanism in main_data.MECHANISM_ORDER:
        candidates = [row for row in targets if row["mechanism"] == mechanism]
        candidates.sort(
            key=lambda row: (
                hashlib.sha256(f"target-select-v1|{row['candidate_id']}".encode()).hexdigest(),
                row["candidate_id"],
            )
        )
        for rank, row in enumerate(candidates[: registries.ERASE_ROWS], 1):
            video = videos / f"{row['candidate_id']}.mp4"
            video.write_bytes(f"selected::{row['candidate_id']}".encode())
            selection_hash = hashlib.sha256(
                f"target-select-v1|{row['candidate_id']}".encode()
            ).hexdigest()
            selected.append(
                {
                    **row,
                    "eligible": "yes",
                    "selection_hash": selection_hash,
                    "selection_rank": rank,
                    "selected_video_path": video.relative_to(project).as_posix(),
                    "selected_video_sha256": _sha(video),
                    "selected_media": json.dumps(
                        registries.MEDIA_CONTRACT,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                }
            )
    selected_path = data / "selected_targets.csv"
    _write_csv(selected_path, selected, selected_fields)

    run_matrix = main_data.build_run_matrix(36)
    run_matrix_path = data / "run_matrix.csv"
    _write_csv(run_matrix_path, run_matrix, main_data.RUN_FIELDS)

    preserve_manifest = data / "preserve_manifest.csv"
    preserve_manifest.write_bytes(
        (ROOT / "data/protocol_v1/preserve_manifest.csv").read_bytes()
    )
    preserve_media = project / "preserve-media"
    for index, row in enumerate(_read_csv(preserve_manifest)):
        folder = preserve_media / f"prompt_{index:03d}" / "videos"
        folder.mkdir(parents=True)
        (folder / f"{row['sample_id']}.mp4").write_bytes(
            f"preserve::{row['sample_id']}".encode()
        )

    model_root = project / "models/Wan"
    model_root.mkdir(parents=True)
    model_file = model_root / "weights.bin"
    model_file.write_bytes(b"model")
    model_inventory = data / "model_inventory.json"
    model_inventory.write_text(
        json.dumps(
            {
                "file_count": 1,
                "files": [
                    {
                        "path": "weights.bin",
                        "sha256": _sha(model_file),
                        "size_bytes": model_file.stat().st_size,
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    runtime_root = project / "models/runtime"
    python = runtime_root / "bin/python"
    python.parent.mkdir(parents=True)
    python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    python.chmod(0o755)
    runtime_registry = data / "runtime_registry.json"
    runtime_registry.write_text(json.dumps({"status": "frozen"}) + "\n", encoding="utf-8")
    return {
        "project": project,
        "selected": selected_path,
        "ontology": ontology_path,
        "run_matrix": run_matrix_path,
        "preserve_manifest": preserve_manifest,
        "preserve_media": preserve_media,
        "model_root": model_root,
        "model_inventory": model_inventory,
        "runtime_root": runtime_root,
        "runtime_registry": runtime_registry,
        "python": python,
        "output": project / "data/training-inputs",
        "cache_root": project / "outputs/training-cache",
        "run_spec_root": project / "data/run-specs",
    }


def _fake_probe(_python: Path, _video: Path):
    return dict(registries.MEDIA_CONTRACT)


def _build(f: dict[str, Path]) -> dict:
    return registries.build_inputs(
        project_root=f["project"],
        selected_targets=f["selected"],
        ontology_registry=f["ontology"],
        run_matrix=f["run_matrix"],
        preserve_manifest=f["preserve_manifest"],
        preserve_media_root=f["preserve_media"],
        model_root=f["model_root"],
        model_inventory=f["model_inventory"],
        runtime_root=f["runtime_root"],
        runtime_registry=f["runtime_registry"],
        python_executable=f["python"],
        output_root=f["output"],
        cache_output_root=f["cache_root"],
        run_spec_output_root=f["run_spec_root"],
        media_probe=_fake_probe,
    )


def test_build_inputs_emits_exact_counts_and_cache_contracts(tmp_path: Path):
    f = _fixture(tmp_path)
    result = _build(f)
    assert result["status"] == registries.BUILD_STATUS
    preserve = _read_csv(f["output"] / "preserve36.csv")
    assert len(preserve) == 36
    assert tuple(preserve[0]) == registries.PRESERVE_FIELDS
    assert len({row["target_video_sha256"] for row in preserve}) == 36
    for index, mechanism in enumerate(main_data.MECHANISM_ORDER):
        selected_path = (
            f["output"] / "selected_by_mechanism"
            / f"{index:02d}_{mechanism}_selected178.csv"
        )
        selected = _read_csv(selected_path)
        assert len(selected) == 178
        assert [int(row["selected_index"]) for row in selected] == list(range(178))
        registry_path = f["output"] / "cache_registries" / f"{index:02d}_{mechanism}.json"
        loaded = cache.load_registry(f["project"], registry_path, _sha(registry_path))
        assert loaded["mechanism"] == mechanism
        assert loaded["selected178"]["row_count"] == 178
        assert loaded["preserve36"]["row_count"] == 36
        assert loaded["arms"]["matched"] is None
        assert loaded["arms"]["v4"]["row_count"] == 178
        if mechanism in {"water_impact", "brittle_fracture"}:
            assert set(loaded["arms"]) == {
                "matched", "v4", "generic_paraphrase", "bystander"
            }
        else:
            assert set(loaded["arms"]) == {"matched", "v4"}


def test_v4_active100_full178_balanced_deranged_and_bystander_matched(tmp_path: Path):
    f = _fixture(tmp_path)
    _build(f)
    for index, mechanism in enumerate(main_data.MECHANISM_ORDER):
        mapping = _read_csv(
            f["output"] / "mappings" / f"{index:02d}_{mechanism}_v4.csv"
        )
        assert len(mapping) == 178
        assert all(
            row["assigned_source_id"] != row["original_source_id"]
            and row["assigned_source_object"] != row["original_source_object"]
            for row in mapping
        )
        active = sorted(
            (row for row in mapping if row["active_erase_ordinal"] != ""),
            key=lambda row: int(row["active_erase_ordinal"]),
        )
        assert len(active) == 100
        assert set(Counter(row["assigned_source_id"] for row in active).values()) == {1, 2}
        assert set(Counter(row["assigned_source_id"] for row in mapping).values()) == {2, 3}
        registry = json.loads(
            (
                f["output"] / "mapping_registries"
                / f"{index:02d}_{mechanism}_v4.json"
            ).read_text()
        )
        assert registry["no_original_source_assignments"] is True
        assert registry["active_source_count_min"] == 1
        assert registry["active_source_count_max"] == 2
    for index, mechanism in ((0, "water_impact"), (2, "brittle_fracture")):
        v4 = _read_csv(f["output"] / "mappings" / f"{index:02d}_{mechanism}_v4.csv")
        bystander = _read_csv(
            f["output"] / "mappings" / f"{index:02d}_{mechanism}_bystander.csv"
        )
        generic = _read_csv(
            f["output"] / "mappings" / f"{index:02d}_{mechanism}_generic_paraphrase.csv"
        )
        assert Counter(row["assigned_source_id"] for row in bystander) == Counter(
            row["assigned_source_id"] for row in v4
        )
        assert all(
            main_data.normalize(row["original_source_object"])
            in main_data.normalize(row["student_prompt"])
            for row in bystander
        )
        assert all(row["student_prompt"] != row["original_source_object"] for row in generic)


def test_preserve_probe_failure_is_fail_closed(tmp_path: Path):
    f = _fixture(tmp_path)
    calls = 0

    def bad_probe(_python: Path, _video: Path):
        nonlocal calls
        calls += 1
        payload = dict(registries.MEDIA_CONTRACT)
        if calls == 7:
            payload["decoded_frames"] = 48
        return payload

    with unittest.TestCase().assertRaisesRegex(
        registries.RegistryError, "decoded contract"
    ):
        registries.build_inputs(
            project_root=f["project"],
            selected_targets=f["selected"],
            ontology_registry=f["ontology"],
            run_matrix=f["run_matrix"],
            preserve_manifest=f["preserve_manifest"],
            preserve_media_root=f["preserve_media"],
            model_root=f["model_root"],
            model_inventory=f["model_inventory"],
            runtime_root=f["runtime_root"],
            runtime_registry=f["runtime_registry"],
            python_executable=f["python"],
            output_root=f["output"],
            cache_output_root=f["cache_root"],
            run_spec_output_root=f["run_spec_root"],
            media_probe=bad_probe,
        )
    assert not f["output"].exists()


def _fake_cache_inspector_factory(f: dict[str, Path]):
    selected_by_mechanism = {
        mechanism: _read_csv(
            f["output"] / "selected_by_mechanism"
            / f"{index:02d}_{mechanism}_selected178.csv"
        )
        for index, mechanism in enumerate(main_data.MECHANISM_ORDER)
    }

    def inspect(
        project_root: Path,
        cache_dir: Path,
        mode: str,
        mechanism: str,
        arm: str | None,
        expected_rows: int,
    ):
        cache_dir.mkdir(parents=True, exist_ok=True)
        selected = selected_by_mechanism[mechanism]
        if mode == "prepare-base":
            files = [
                {
                    "index": index,
                    "row_id": row["candidate_id"],
                    "prompt_embeds_sha256": hashlib.sha256(
                        f"prompt::{mechanism}::{row['candidate_id']}".encode()
                    ).hexdigest(),
                }
                for index, row in enumerate(selected)
            ]
            files.extend(
                {
                    "index": 178 + index,
                    "row_id": f"preserve_{index:03d}",
                    "prompt_embeds_sha256": hashlib.sha256(
                        f"preserve::{index}".encode()
                    ).hexdigest(),
                }
                for index in range(36)
            )
        else:
            files = [
                {
                    "index": index,
                    "row_id": row["candidate_id"],
                    "prompt_embeds_sha256": hashlib.sha256(
                        f"prompt::{mechanism}::{row['candidate_id']}".encode()
                    ).hexdigest(),
                }
                for index, row in enumerate(selected)
            ]
        record = {
            "path": registries.canonical_path(project_root, cache_dir),
            "manifest_sha256": hashlib.sha256(
                f"manifest::{mechanism}::{mode}::{arm}".encode()
            ).hexdigest(),
            "ordered_inventory_sha256": hashlib.sha256(
                f"inventory::{mechanism}::{mode}::{arm}".encode()
            ).hexdigest(),
            "row_count": expected_rows,
        }
        return record, {"manifest": {"files": files}}

    return inspect


def test_finalize_runs_emits_only_after_cache_validation(tmp_path: Path):
    f = _fixture(tmp_path)
    result = _build(f)
    input_registry = f["output"] / "training_input_registry.json"
    final = registries.finalize_runs(
        project_root=f["project"],
        input_registry_path=input_registry,
        input_registry_sha256=_sha(input_registry),
        output_root=f["run_spec_root"],
        cache_inspector=_fake_cache_inspector_factory(f),
    )
    assert final["run_spec_count"] == 18
    assert final["matched_receipt_count"] == 7
    specs = sorted((f["run_spec_root"] / "run_specs").glob("*.json"))
    receipts = sorted((f["run_spec_root"] / "matched_receipts").glob("*.json"))
    assert len(specs) == 18
    assert len(receipts) == 7
    for path in specs:
        payload = json.loads(path.read_text())
        assert set(payload) == {
            "schema_version", "protocol", "protocol_version", "status",
            "run_id", "mechanism", "arm", "base_cache", "teacher_cache",
            "arm_cache", "model_root", "model_inventory", "runtime_root",
            "runtime_registry", "python_executable", "training_config",
            "output_dir", "matched_tensor_equality_receipt",
        }
        assert payload["training_config"] == trainer.EXPECTED_CONFIG
        if payload["arm"] == "matched_control":
            assert payload["matched_tensor_equality_receipt"]["row_count"] == 178
        else:
            assert payload["matched_tensor_equality_receipt"] is None
    assert result["run_specs_emitted"] is False  # immutable input registry was not rewritten


def test_fresh_only_rejects_rebuild(tmp_path: Path):
    f = _fixture(tmp_path)
    _build(f)
    with unittest.TestCase().assertRaisesRegex(
        registries.RegistryError, "output root must be fresh"
    ):
        _build(f)


def load_tests(_loader, _tests, _pattern):
    suite = unittest.TestSuite()
    functions = (
        test_build_inputs_emits_exact_counts_and_cache_contracts,
        test_v4_active100_full178_balanced_deranged_and_bystander_matched,
        test_preserve_probe_failure_is_fail_closed,
        test_finalize_runs_emits_only_after_cache_validation,
        test_fresh_only_rejects_rebuild,
    )
    for function in functions:
        def invoke(fn=function):
            with tempfile.TemporaryDirectory() as directory:
                fn(Path(directory))

        suite.addTest(unittest.FunctionTestCase(invoke, description=function.__name__))
    return suite


if __name__ == "__main__":
    unittest.main()
