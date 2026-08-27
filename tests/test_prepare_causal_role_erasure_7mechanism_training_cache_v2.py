from __future__ import annotations

import csv
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import prepare_causal_role_erasure_7mechanism_training_cache_v2 as cache  # noqa: E402


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _artifact(project: Path, path: Path, rows: int) -> dict[str, object]:
    return {
        "path": path.relative_to(project).as_posix(),
        "sha256": cache.sha256_file(path),
        "row_count": rows,
    }


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    project = tmp_path / "project"
    project.mkdir()
    videos = project / "videos"
    videos.mkdir()
    selected = []
    for index in range(cache.ERASE_ROWS):
        video = videos / f"erase_{index:03d}.mp4"
        video.write_bytes(f"erase-video-{index}".encode())
        selected.append(
            {
                "protocol_version": cache.PROTOCOL_VERSION,
                "mechanism": "rigid_collision",
                "selected_index": str(index),
                "candidate_id": f"rigid_target_{index:03d}",
                "global_index": str(192 + index),
                "source_id": f"source_{index % 8}",
                "source_object": f"source object {index % 8}",
                "receiver_id": f"receiver_{index % 12}",
                "receiver": f"receiver object {index % 12}",
                "prompt_style": "direct" if index % 2 == 0 else "natural",
                "factual_prompt": f"Factual prompt {index}",
                "target_prompt": f"Target prompt {index}",
                "seed": str(1_110_000 + index),
                "target_video_path": video.relative_to(project).as_posix(),
                "target_video_sha256": cache.sha256_file(video),
            }
        )
    selected_path = project / "data/selected178.csv"
    _write_csv(selected_path, selected)

    preserve = []
    for index in range(cache.PRESERVE_ROWS):
        video = videos / f"preserve_{index:03d}.mp4"
        video.write_bytes(f"preserve-video-{index}".encode())
        preserve.append(
            {
                "protocol_version": cache.PROTOCOL_VERSION,
                "preserve_index": str(index),
                "preserve_id": f"preserve_{index:03d}",
                "prompt": f"Preservation prompt {index}",
                "target_video_path": video.relative_to(project).as_posix(),
                "target_video_sha256": cache.sha256_file(video),
            }
        )
    preserve_path = project / "data/preserve36.csv"
    _write_csv(preserve_path, preserve)

    arm_records = {}
    for arm in ("v4", "generic_paraphrase", "bystander"):
        mappings = []
        for index, row in enumerate(selected):
            mappings.append(
                {
                    "protocol_version": cache.PROTOCOL_VERSION,
                    "mechanism": "rigid_collision",
                    "arm": arm,
                    "selected_index": str(index),
                    "candidate_id": row["candidate_id"],
                    "student_prompt": f"{arm} student prompt {index}",
                    "assigned_source_id": f"assigned_{index % 64}" if arm == "v4" else "",
                    "assigned_source_object": f"assigned object {index % 64}" if arm == "v4" else "",
                }
            )
        mapping_path = project / f"data/{arm}.csv"
        _write_csv(mapping_path, mappings)
        arm_records[arm] = _artifact(project, mapping_path, cache.ERASE_ROWS)
    arm_records["matched"] = None

    model = project / "models/Wan"
    model.mkdir(parents=True)
    model_file = model / "weights.bin"
    model_file.write_bytes(b"model")
    model_inventory = project / "data/model_inventory.json"
    model_inventory.write_text(
        json.dumps(
            {
                "file_count": 1,
                "files": [
                    {
                        "path": "weights.bin",
                        "sha256": cache.sha256_file(model_file),
                        "size_bytes": model_file.stat().st_size,
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    runtime = project / "models/runtime"
    python = runtime / "bin/python"
    python.parent.mkdir(parents=True)
    python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    python.chmod(0o755)
    runtime_registry = project / "data/runtime.json"
    runtime_registry.write_text(json.dumps({"status": "frozen"}) + "\n", encoding="utf-8")

    registry = project / "data/cache_registry.json"
    registry.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "protocol": cache.REGISTRY_PROTOCOL,
                "protocol_version": cache.PROTOCOL_VERSION,
                "status": "frozen",
                "mechanism": "rigid_collision",
                "selected178": _artifact(project, selected_path, cache.ERASE_ROWS),
                "preserve36": _artifact(project, preserve_path, cache.PRESERVE_ROWS),
                "model_root": model.relative_to(project).as_posix(),
                "model_inventory": _artifact(project, model_inventory, 1),
                "runtime_root": runtime.relative_to(project).as_posix(),
                "runtime_registry": _artifact(project, runtime_registry, 1),
                "python_executable": python.relative_to(project).as_posix(),
                "arms": arm_records,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return project, registry


@dataclass(frozen=True)
class FakeTensor:
    shape: tuple[int, ...]
    dtype: str
    data: str


class FakeBackend:
    dtype_name = cache.TENSOR_DTYPE

    def encode_prompts(self, prompts):
        return {
            prompt: FakeTensor(cache.PROMPT_SHAPE, cache.TENSOR_DTYPE, f"prompt::{prompt}")
            for prompt in dict.fromkeys(prompts)
        }

    def encode_latent(self, video_path):
        return FakeTensor(cache.LATENT_SHAPE, cache.TENSOR_DTYPE, f"latent::{video_path.name}")

    def tensor_shape(self, tensor):
        return tensor.shape

    def tensor_dtype(self, tensor):
        return tensor.dtype

    def tensor_sha256(self, tensor):
        return hashlib.sha256(repr(tensor).encode()).hexdigest()

    def tensor_equal(self, left, right):
        return left == right

    def save(self, payload, path):
        def convert(value):
            if isinstance(value, FakeTensor):
                return {"__tensor__": True, "shape": list(value.shape), "dtype": value.dtype, "data": value.data}
            return value
        path.write_text(json.dumps({key: convert(value) for key, value in payload.items()}) + "\n", encoding="utf-8")

    def load(self, path):
        payload = json.loads(path.read_text())
        for key, value in list(payload.items()):
            if isinstance(value, dict) and value.get("__tensor__"):
                payload[key] = FakeTensor(tuple(value["shape"]), value["dtype"], value["data"])
        return payload

    def close(self):
        pass


def _plan(project: Path, registry_path: Path, mode: str, output: Path, *, arm=None, base=None, dry=False):
    registry = cache.load_registry(project, registry_path, cache.sha256_file(registry_path))
    return cache.build_plan(registry, mode, output, arm=arm, base_cache_dir=base, dry_run=dry)


def test_dry_run_writes_plan_without_constructing_backend(tmp_path: Path, monkeypatch):
    project, registry = _fixture(tmp_path)
    output = project / "outputs/plan"

    class Forbidden:
        def __init__(self, *_args, **_kwargs):
            raise AssertionError("dry-run loaded the ML backend")

    monkeypatch.setattr(cache, "RealBackend", Forbidden)
    assert cache.main(
        [
            "prepare-base", "--project-root", str(project), "--registry", str(registry),
            "--registry-sha256", cache.sha256_file(registry),
            "--output-dir", str(output), "--dry-run",
        ]
    ) == 0
    plan = json.loads((output / "cache_plan.json").read_text())
    assert plan["dry_run"] is True
    assert plan["mode"] == "prepare-base"
    assert plan["row_count"] == 214
    assert plan["tensor_contract"]["prompt_shape"] == [1, 226, 4096]
    assert plan["tensor_contract"]["latent_shape"] == [1, 16, 13, 60, 104]
    assert plan["tensor_contract"]["prompt_batch_size"] == 16
    assert not list(output.glob("*.pt"))
    assert cache.main(
        [
            "prepare-base", "--project-root", str(project), "--registry", str(registry),
            "--registry-sha256", cache.sha256_file(registry),
            "--output-dir", str(output), "--dry-run",
        ]
    ) == 1


def test_prepare_base_writes_exact_214_ordered_inventory(tmp_path: Path):
    project, registry_path = _fixture(tmp_path)
    output = project / "outputs/base"
    plan, selected, preserve, mapping = _plan(project, registry_path, "prepare-base", output)
    cache.reserve_output(output, plan)
    manifest = cache.prepare_cache(plan, selected, preserve, mapping, FakeBackend())
    assert manifest["row_count"] == 214
    assert len(manifest["files"]) == 214
    assert len(list(output.glob("*.pt"))) == 214
    assert manifest["files"][0]["row_id"] == selected[0]["candidate_id"]
    assert manifest["files"][178]["row_id"] == preserve[0]["preserve_id"]
    assert manifest["ordered_inventory_sha256"] == cache.inventory_sha256(
        [output / record["file"] for record in manifest["files"]]
    )
    payload = FakeBackend().load(output / manifest["files"][0]["file"])
    assert payload["latents"].shape == cache.LATENT_SHAPE
    assert payload["prompt_embeds"].shape == cache.PROMPT_SHAPE
    assert not (output / ".run_reservation").exists()


def test_prepare_teacher_and_v4_arm_each_write_178(tmp_path: Path):
    project, registry_path = _fixture(tmp_path)
    for mode, arm in (("prepare-teacher", None), ("prepare-arm", "v4")):
        output = project / f"outputs/{arm or 'teacher'}"
        plan, selected, preserve, mapping = _plan(project, registry_path, mode, output, arm=arm)
        cache.reserve_output(output, plan)
        manifest = cache.prepare_cache(plan, selected, preserve, mapping, FakeBackend())
        assert manifest["row_count"] == 178
        assert len(manifest["files"]) == 178
        payload = FakeBackend().load(output / manifest["files"][0]["file"])
        key = "teacher_prompt_embeds" if mode == "prepare-teacher" else "student_prompt_embeds"
        assert payload[key].shape == cache.PROMPT_SHAPE
        if arm == "v4":
            assert payload["assigned_source_id"].startswith("assigned_")


def test_matched_fresh_embeddings_must_equal_base_tensor(tmp_path: Path):
    project, registry_path = _fixture(tmp_path)
    base = project / "outputs/base"
    plan, selected, preserve, mapping = _plan(project, registry_path, "prepare-base", base)
    cache.reserve_output(base, plan)
    base_manifest = cache.prepare_cache(plan, selected, preserve, mapping, FakeBackend())

    matched = project / "outputs/matched"
    plan, selected, preserve, mapping = _plan(
        project, registry_path, "prepare-arm", matched, arm="matched", base=base
    )
    cache.reserve_output(matched, plan)
    manifest = cache.prepare_cache(plan, selected, preserve, mapping, FakeBackend())
    assert all(FakeBackend().load(matched / item["file"])["matched_base_tensor_equal"] for item in manifest["files"])

    first = base / base_manifest["files"][0]["file"]
    payload = FakeBackend().load(first)
    payload["prompt_embeds"] = FakeTensor(cache.PROMPT_SHAPE, cache.TENSOR_DTYPE, "drift")
    FakeBackend().save(payload, first)
    base_manifest_path = base / "cache_manifest.json"
    changed_manifest = json.loads(base_manifest_path.read_text())
    changed_manifest["files"][0]["sha256"] = cache.sha256_file(first)
    changed_manifest["ordered_inventory_sha256"] = cache.inventory_sha256(
        [base / item["file"] for item in changed_manifest["files"]]
    )
    base_manifest_path.write_bytes(cache.canonical_json_bytes(changed_manifest))
    broken = project / "outputs/matched-broken"
    plan, selected, preserve, mapping = _plan(
        project, registry_path, "prepare-arm", broken, arm="matched", base=base
    )
    cache.reserve_output(broken, plan)
    with pytest.raises(ValueError, match="differs from base canonical tensor"):
        cache.prepare_cache(plan, selected, preserve, mapping, FakeBackend())


def test_registry_or_selected_order_drift_fails_before_output(tmp_path: Path):
    project, registry_path = _fixture(tmp_path)
    registry = json.loads(registry_path.read_text())
    selected_path = project / registry["selected178"]["path"]
    rows = list(csv.DictReader(selected_path.open(newline="", encoding="utf-8")))
    rows[0]["selected_index"] = "1"
    _write_csv(selected_path, rows)
    registry["selected178"]["sha256"] = cache.sha256_file(selected_path)
    registry_path.write_text(json.dumps(registry) + "\n", encoding="utf-8")
    loaded = cache.load_registry(project, registry_path, cache.sha256_file(registry_path))
    with pytest.raises(ValueError, match="selected_index"):
        cache.build_plan(loaded, "prepare-base", project / "outputs/no", arm=None, base_cache_dir=None, dry_run=True)


def test_external_registry_sha_is_required_and_wrong_hash_fails_before_output(tmp_path: Path):
    project, registry = _fixture(tmp_path)
    output = project / "outputs/no-registry-trust"
    with pytest.raises(SystemExit):
        cache.main(
            [
                "prepare-base", "--project-root", str(project), "--registry", str(registry),
                "--output-dir", str(output), "--dry-run",
            ]
        )
    assert not output.exists()
    assert cache.main(
        [
            "prepare-base", "--project-root", str(project), "--registry", str(registry),
            "--registry-sha256", "0" * 64, "--output-dir", str(output), "--dry-run",
        ]
    ) == 1
    assert not output.exists()


def test_sealed_registry_path_rejected_before_read(tmp_path: Path, monkeypatch):
    called = False

    def forbidden(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError

    monkeypatch.setattr(cache, "load_registry", forbidden)
    assert cache.main(
        [
            "prepare-base", "--project-root", str(tmp_path),
            "--registry", "data/sealed-final36.json", "--registry-sha256", "0" * 64,
            "--output-dir", "outputs/cache",
            "--dry-run",
        ]
    ) == 1
    assert called is False


def test_live_model_inventory_accepts_frozen_project_relative_paths_and_cache_files(
    tmp_path: Path,
):
    project, registry_path = _fixture(tmp_path)
    registry_payload = json.loads(registry_path.read_text())
    model_root = project / registry_payload["model_root"]
    cache_file = model_root / ".cache/huggingface/download/config.json.lock"
    cache_file.parent.mkdir(parents=True)
    cache_file.write_bytes(b"frozen-cache-record")
    files = []
    for path in sorted(value for value in model_root.rglob("*") if value.is_file()):
        files.append(
            {
                "path": path.relative_to(project).as_posix(),
                "sha256": cache.sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    inventory_path = project / registry_payload["model_inventory"]["path"]
    inventory_path.write_text(
        json.dumps(
            {
                "model_root": registry_payload["model_root"],
                "file_count": len(files),
                "files": files,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    registry_payload["model_inventory"] = _artifact(
        project, inventory_path, len(files)
    )
    registry_path.write_text(json.dumps(registry_payload) + "\n", encoding="utf-8")
    loaded = cache.load_registry(
        project, registry_path, cache.sha256_file(registry_path)
    )
    validated = cache.validate_model_inventory(loaded, live=True)
    assert validated["files"] == files


def test_live_model_inventory_rejects_unregistered_cache_file(tmp_path: Path):
    project, registry_path = _fixture(tmp_path)
    loaded = cache.load_registry(
        project, registry_path, cache.sha256_file(registry_path)
    )
    model_root = Path(loaded["model_root"])
    extra = model_root / ".cache/download/new.lock"
    extra.parent.mkdir(parents=True)
    extra.write_bytes(b"not frozen")
    with pytest.raises(ValueError, match="missing or extra"):
        cache.validate_model_inventory(loaded, live=True)
