#!/usr/bin/env python3
"""Generate the frozen formal SAFREE-CogVideoX baseline.

The runner consumes the authorized seven-mechanism baseline registry, imports
the exact registered official SAFREE pipeline, and injects each mechanism name
as a single ``CONCEPT_DICT`` entry.  Every formal case keeps its frozen seed.
Outputs are fresh-only unless ``--resume`` is used with the byte-identical
generation plan, and every accepted video receives a hash-bound media receipt.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from fractions import Fraction
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Mapping

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_causal_role_erasure_7mechanism_baseline_registry_v2 as baseline_registry


PROTOCOL = "causal_role_erasure_7mechanism_safree_cogvideox_v2"
BASELINE = "safree_cogvideox"
BUILDER_RELATIVE = "scripts/build_causal_role_erasure_7mechanism_baseline_registry_v2.py"
RUNNER_RELATIVE = "scripts/run_causal_role_erasure_7mechanism_safree_cogvideox_v2.py"
MEDIA = {"decoded_frames": 49, "fps": "8/1", "height": 480, "width": 720}
SAFREE_STREAM = {
    "dtype": "fp32",
    "method": "official_pipeline_with_mechanism_concept_injection",
}
SAFREE_CONCEPT_INJECTION = "mechanism_name_as_single_CONCEPT_DICT_entry"
PYAV_PROBE_CODE = r"""
import av, json, sys
from fractions import Fraction
path = sys.argv[1]
with av.open(path) as container:
    videos = [stream for stream in container.streams if stream.type == 'video']
    audios = [stream for stream in container.streams if stream.type == 'audio']
    if len(videos) != 1 or audios:
        raise SystemExit('expected exactly one video stream and no audio')
    stream = videos[0]
    rate = stream.average_rate or stream.guessed_rate
    count = sum(1 for _ in container.decode(video=stream.index))
    print(json.dumps({
        'decoded_frames': count,
        'fps': '' if rate is None else f'{Fraction(rate).numerator}/{Fraction(rate).denominator}',
        'height': int(stream.height),
        'width': int(stream.width),
    }, sort_keys=True))
"""
RUNTIME_PROBE_CODE = r"""
import importlib.metadata as metadata
import json, sys
import diffusers, torch, transformers
print(json.dumps({
    'executable': sys.executable,
    'packages': {
        name: metadata.version(name)
        for name in ('torch', 'diffusers', 'transformers')
    },
}, sort_keys=True))
"""


class SAFREEError(ValueError):
    """A fail-closed formal SAFREE generation error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SAFREEError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_json_regular(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"{label} missing or symlinked: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SAFREEError(f"invalid {label}: {path}") from exc
    require(isinstance(value, dict), f"{label} must contain a JSON object: {path}")
    return value


def resolve_registered(project_root: Path, value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else project_root / path).resolve()


def probe_registered_runtime(runtime_python: Path) -> dict[str, Any]:
    try:
        result = subprocess.run(
            [str(runtime_python), "-I", "-c", RUNTIME_PROBE_CODE],
            check=True,
            text=True,
            capture_output=True,
        )
        payload = json.loads(result.stdout)
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        raise SAFREEError(
            f"failed to probe registered runtime with isolated imports: {runtime_python}"
        ) from exc
    require(isinstance(payload, dict), "registered runtime probe did not return an object")
    packages = payload.get("packages")
    require(isinstance(packages, dict), "registered runtime probe has no package versions")
    require(
        set(packages) == set(baseline_registry.REQUIRED_RUNTIME_PACKAGES),
        "registered runtime probe package inventory is incomplete",
    )
    return payload


def _validate_model_inventory(project_root: Path, registry: Mapping[str, Any]) -> None:
    model = registry.get("model")
    require(isinstance(model, Mapping), "baseline registry has no model binding")
    model_root = resolve_registered(project_root, str(model.get("root", "")))
    require(model_root.is_dir() and not model_root.is_symlink(), f"registered model root missing or symlinked: {model_root}")
    symlinks = [path for path in model_root.rglob("*") if path.is_symlink()]
    require(not symlinks, f"registered model tree now contains symlinks: {symlinks[:3]}")
    inventory_path = resolve_registered(project_root, str(model.get("inventory_path", "")))
    inventory = load_json_regular(inventory_path, "model inventory")
    require(sha256_file(inventory_path) == model.get("inventory_sha256"), "model inventory changed")
    files = inventory.get("files")
    require(isinstance(files, list) and files, "model inventory has no files")
    require(inventory.get("file_count") == len(files), "model inventory count is internally inconsistent")
    require(model.get("file_count") == len(files), "registered model file count changed")
    total_size = 0
    seen: set[str] = set()
    for item in files:
        require(isinstance(item, Mapping), "invalid model inventory entry")
        relative_text = str(item.get("path", ""))
        relative = Path(relative_text)
        require(relative_text not in seen, f"model inventory path repeats: {relative_text}")
        seen.add(relative_text)
        require(relative_text and not relative.is_absolute() and ".." not in relative.parts, f"unsafe model inventory path: {relative_text}")
        candidate = model_root / relative
        require(candidate.is_file() and not candidate.is_symlink(), f"registered model file missing or symlinked: {candidate}")
        require(candidate.resolve().is_relative_to(model_root), f"model inventory path escapes model root: {relative_text}")
        size = candidate.stat().st_size
        require(size == item.get("size_bytes"), f"registered model file size changed: {candidate}")
        require(sha256_file(candidate) == item.get("sha256"), f"registered model file changed: {candidate}")
        total_size += size
    require(inventory.get("total_size_bytes") == total_size, "model inventory total size is inconsistent")
    require(model.get("total_size_bytes") == total_size, "registered model total size changed")


def load_registry(
    project_root: Path,
    registry_path: Path,
    receipt_path: Path,
    runtime_probe: Callable[[Path], Mapping[str, Any]] = probe_registered_runtime,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    registry = load_json_regular(registry_path, "baseline registry")
    registry_raw = registry_path.read_bytes()
    receipt = load_json_regular(receipt_path, "baseline registry receipt")
    require(registry.get("protocol") == baseline_registry.REGISTRY_PROTOCOL, "unexpected baseline registry protocol")
    require(registry.get("protocol_version") == baseline_registry.PROTOCOL_VERSION, "baseline protocol version changed")
    require(registry.get("status") == "baseline_stack_frozen_ready", "baseline registry is not frozen ready")
    require(registry.get("formal_generation_authorized") is True, "baseline registry does not authorize formal generation")
    require(receipt.get("protocol") == baseline_registry.REGISTRY_PROTOCOL, "unexpected baseline receipt protocol")
    require(receipt.get("status") == registry.get("status"), "baseline registry receipt status mismatch")
    require(receipt.get("registry_sha256") == hashlib.sha256(registry_raw).hexdigest(), "baseline registry receipt mismatch")

    code = registry.get("code_sha256")
    require(isinstance(code, Mapping), "baseline registry has no code bindings")
    for relative, label in (
        (BUILDER_RELATIVE, "baseline registry builder"),
        (RUNNER_RELATIVE, "SAFREE formal runner"),
    ):
        require(relative in code, f"{label} is not bound by baseline registry")
        path = project_root / relative
        require(path.is_file() and not path.is_symlink(), f"registered {label} missing or symlinked: {path}")
        require(sha256_file(path) == code[relative], f"{label} changed after baseline freeze")

    formal = registry.get("formal_cases")
    require(isinstance(formal, Mapping), "baseline registry has no formal-case binding")
    require(formal.get("row_count") == baseline_registry.FORMAL_CASES, "registered formal case count changed")
    require(formal.get("same_case_same_seed_across_streams") is True, "formal seed-sharing contract changed")
    formal_path = resolve_registered(project_root, str(formal.get("path", "")))
    require(formal_path.is_file() and not formal_path.is_symlink(), f"formal cases missing or symlinked: {formal_path}")
    require(sha256_file(formal_path) == formal.get("sha256"), "formal cases changed after registration")
    baseline_registry.validate_formal_cases(formal_path)

    require(registry.get("mechanism_concepts") == baseline_registry.MECHANISM_CONCEPTS, "registered mechanism concepts changed")
    require(BASELINE in registry.get("streams", []), "SAFREE stream is absent from baseline registry")
    generation = registry.get("generation")
    require(isinstance(generation, Mapping), "baseline registry has no generation contract")
    require(generation.get("shared") == baseline_registry.COG_GENERATION, "shared CogVideoX generation contract changed")
    streams = generation.get("streams")
    require(isinstance(streams, Mapping) and streams.get(BASELINE) == SAFREE_STREAM, "SAFREE generation stream changed")

    implementations = registry.get("implementations")
    require(isinstance(implementations, Mapping), "baseline registry has no implementation bindings")
    implementation = implementations.get(BASELINE)
    require(isinstance(implementation, Mapping), "baseline registry has no SAFREE implementation")
    require(implementation.get("status") == "ready", "registered SAFREE implementation is not ready")
    require(implementation.get("dtype") == "fp32", "registered SAFREE dtype changed")
    require(implementation.get("concept_injection") == SAFREE_CONCEPT_INJECTION, "registered SAFREE concept injection changed")
    commit = str(implementation.get("commit", "")).casefold()
    require(baseline_registry.HEX40.fullmatch(commit) is not None, "registered SAFREE commit is invalid")
    pipeline_path = resolve_registered(project_root, str(implementation.get("pipeline_path", "")))
    require(pipeline_path.is_file() and not pipeline_path.is_symlink(), f"registered SAFREE pipeline missing or symlinked: {pipeline_path}")
    require(sha256_file(pipeline_path) == implementation.get("pipeline_sha256"), "registered SAFREE pipeline changed")
    observed_commit = baseline_registry.observed_git_commit(pipeline_path.parent.parent)
    require(observed_commit == commit, "SAFREE checkout commit changed after baseline freeze")

    _validate_model_inventory(project_root, registry)
    runtime = registry.get("runtime")
    require(isinstance(runtime, Mapping), "baseline registry has no runtime binding")
    runtime_python = resolve_registered(project_root, str(runtime.get("python_executable", "")))
    require(runtime_python.is_file() and not runtime_python.is_symlink(), f"registered runtime Python missing or symlinked: {runtime_python}")
    require(sha256_file(runtime_python) == runtime.get("python_executable_sha256"), "registered runtime Python changed")
    require(Path(sys.executable).resolve() == runtime_python, f"wrong runtime Python: {Path(sys.executable).resolve()} != {runtime_python}")
    registered_packages = runtime.get("packages")
    require(isinstance(registered_packages, Mapping), "baseline registry has no runtime package binding")
    observed_runtime = dict(runtime_probe(runtime_python))
    observed_executable = Path(str(observed_runtime.get("executable", ""))).resolve()
    require(
        observed_executable == runtime_python,
        f"registered runtime probe used the wrong Python: {observed_executable} != {runtime_python}",
    )
    require(
        dict(observed_runtime.get("packages", {})) == dict(registered_packages),
        "registered runtime package versions changed",
    )
    return registry


def read_formal_rows(project_root: Path, registry: Mapping[str, Any], mechanism: str | None) -> list[dict[str, str]]:
    path = resolve_registered(project_root, str(registry["formal_cases"]["path"]))
    rows = baseline_registry.validate_formal_cases(path)
    require([int(row["global_case_index"]) for row in rows] == list(range(baseline_registry.FORMAL_CASES)), "formal global-case order changed")
    if mechanism is not None:
        rows = [row for row in rows if row["mechanism"] == mechanism]
        require(len(rows) == baseline_registry.CASES_PER_MECHANISM, f"{mechanism} formal row count must be 42")
    return rows


def build_plan(
    *,
    project_root: Path,
    registry_path: Path,
    registry: Mapping[str, Any],
    mechanism: str | None,
    output_dir: Path,
) -> dict[str, Any]:
    rows = read_formal_rows(project_root, registry, mechanism)
    concepts = registry["mechanism_concepts"]
    items = []
    for row in rows:
        case_id = row["case_id"]
        require(Path(case_id).name == case_id, f"unsafe formal case ID: {case_id}")
        concept = str(concepts[row["mechanism"]])
        require(row["mechanism_name"] == concept, f"formal mechanism name changed for {case_id}")
        items.append(
            {
                "case_id": case_id,
                "global_case_index": int(row["global_case_index"]),
                "mechanism": row["mechanism"],
                "case_kind": row["case_kind"],
                "prompt": row["prompt"],
                "target_concept": concept,
                "safree_concept_key": concept,
                "safree_concept_terms": [concept],
                "seed": int(row["seed"]),
                "video_path": (output_dir / "videos" / f"{case_id}.mp4").as_posix(),
                "receipt_path": (output_dir / "receipts" / f"{case_id}.json").as_posix(),
            }
        )
    implementation = registry["implementations"][BASELINE]
    try:
        registry_reference = registry_path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        registry_reference = str(registry_path.resolve())
    return {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "protocol_version": baseline_registry.PROTOCOL_VERSION,
        "baseline": BASELINE,
        "mechanism": mechanism or "ALL",
        "baseline_registry": registry_reference,
        "baseline_registry_sha256": sha256_file(registry_path),
        "formal_cases_sha256": registry["formal_cases"]["sha256"],
        "model_root": registry["model"]["root"],
        "generation": {**registry["generation"]["shared"], **SAFREE_STREAM},
        "safree": {
            "commit": implementation["commit"],
            "pipeline_path": implementation["pipeline_path"],
            "pipeline_sha256": implementation["pipeline_sha256"],
            "concept_injection": implementation["concept_injection"],
        },
        "items": items,
    }


def probe_video(runtime_python: Path, video: Path) -> dict[str, Any]:
    try:
        result = subprocess.run(
            [str(runtime_python), "-I", "-c", PYAV_PROBE_CODE, str(video)],
            check=True,
            text=True,
            capture_output=True,
        )
        payload = json.loads(result.stdout)
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        raise SAFREEError(f"failed to decode formal SAFREE video: {video}") from exc
    if payload.get("fps"):
        value = Fraction(str(payload["fps"]))
        payload["fps"] = f"{value.numerator}/{value.denominator}"
    return payload


def validate_video(
    runtime_python: Path,
    video: Path,
    media_probe: Callable[[Path, Path], Mapping[str, Any]] = probe_video,
) -> dict[str, Any]:
    require(video.is_file() and not video.is_symlink(), f"formal SAFREE video missing or symlinked: {video}")
    payload = dict(media_probe(runtime_python, video))
    require(payload == MEDIA, f"formal SAFREE video media contract mismatch for {video}: {payload}")
    return payload


def load_safree_module(pipeline_path: Path, pipeline_sha256: str) -> ModuleType:
    require(pipeline_path.is_file() and not pipeline_path.is_symlink(), f"registered SAFREE pipeline missing or symlinked: {pipeline_path}")
    require(sha256_file(pipeline_path) == pipeline_sha256, "registered SAFREE pipeline changed before import")
    module_dir = str(pipeline_path.parent)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    module_name = f"frozen_safree_cogvideox_{pipeline_sha256[:16]}"
    spec = importlib.util.spec_from_file_location(module_name, pipeline_path)
    require(spec is not None and spec.loader is not None, f"cannot import registered SAFREE pipeline: {pipeline_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(hasattr(module, "CogVideoXPipeline"), "registered SAFREE pipeline has no CogVideoXPipeline")
    require(isinstance(getattr(module, "CONCEPT_DICT", None), dict), "registered SAFREE pipeline has no mutable CONCEPT_DICT")
    return module


def load_pipeline(registry: Mapping[str, Any], project_root: Path):
    import torch
    from diffusers import CogVideoXDPMScheduler
    from diffusers.utils import export_to_video

    require(torch.cuda.is_available(), "formal SAFREE generation requires CUDA")
    implementation = registry["implementations"][BASELINE]
    pipeline_path = resolve_registered(project_root, str(implementation["pipeline_path"]))
    module = load_safree_module(pipeline_path, str(implementation["pipeline_sha256"]))
    model_root = resolve_registered(project_root, str(registry["model"]["root"]))
    pipe = module.CogVideoXPipeline.from_pretrained(model_root, torch_dtype=torch.float32)
    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    pipe.to("cuda")
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    return module, torch, export_to_video, pipe


def render_item(
    module: ModuleType,
    torch: Any,
    export_to_video: Callable[..., Any],
    pipe: Any,
    item: Mapping[str, Any],
    generation: Mapping[str, Any],
) -> None:
    concept_key = str(item["safree_concept_key"])
    concept_terms = list(item["safree_concept_terms"])
    require(concept_terms == [concept_key], f"SAFREE concept entry is not a single mechanism name: {concept_key}")
    concept_dict = module.CONCEPT_DICT
    require(isinstance(concept_dict, dict), "registered SAFREE CONCEPT_DICT is no longer mutable")
    sentinel = object()
    previous = concept_dict.get(concept_key, sentinel)
    concept_dict[concept_key] = concept_terms
    video = Path(str(item["video_path"]))
    video.parent.mkdir(parents=True, exist_ok=True)
    require(not video.exists(), f"refusing to overwrite formal SAFREE video: {video}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{video.stem}.", suffix=".mp4", dir=video.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    temporary.unlink()
    try:
        frames = pipe(
            prompt=str(item["prompt"]),
            num_videos_per_prompt=1,
            num_inference_steps=int(generation["num_inference_steps"]),
            num_frames=int(generation["num_frames"]),
            height=int(generation["height"]),
            width=int(generation["width"]),
            use_dynamic_cfg=bool(generation["use_dynamic_cfg"]),
            guidance_scale=float(generation["guidance_scale"]),
            generator=torch.Generator(device="cuda").manual_seed(int(item["seed"])),
            concept=concept_key,
        ).frames[0]
        export_to_video(frames, str(temporary), fps=int(generation["fps"]))
        require(temporary.is_file() and not temporary.is_symlink(), "SAFREE exporter did not create a regular video")
        os.replace(temporary, video)
    finally:
        if temporary.exists():
            temporary.unlink()
        if previous is sentinel:
            concept_dict.pop(concept_key, None)
        else:
            concept_dict[concept_key] = previous


def _receipt_for(
    item: Mapping[str, Any],
    plan: Mapping[str, Any],
    *,
    video_sha256: str,
    media: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "protocol_version": plan["protocol_version"],
        "status": "validated",
        "baseline": BASELINE,
        "case_id": item["case_id"],
        "global_case_index": item["global_case_index"],
        "mechanism": item["mechanism"],
        "case_kind": item["case_kind"],
        "seed": item["seed"],
        "target_concept": item["target_concept"],
        "safree_concept_key": item["safree_concept_key"],
        "safree_concept_terms": item["safree_concept_terms"],
        "video_path": item["video_path"],
        "video_sha256": video_sha256,
        "media": dict(media),
        "baseline_registry_sha256": plan["baseline_registry_sha256"],
        "safree_pipeline_sha256": plan["safree"]["pipeline_sha256"],
    }


def _validated_receipt(
    runtime_python: Path,
    item: Mapping[str, Any],
    plan: Mapping[str, Any],
    receipt_path: Path,
    media_probe: Callable[[Path, Path], Mapping[str, Any]],
) -> dict[str, Any]:
    receipt = load_json_regular(receipt_path, "formal SAFREE video receipt")
    video = Path(str(item["video_path"]))
    media = validate_video(runtime_python, video, media_probe)
    expected = _receipt_for(item, plan, video_sha256=sha256_file(video), media=media)
    require(receipt == expected, f"formal SAFREE receipt changed or does not match its case: {receipt_path}")
    return receipt


def run_plan(
    *,
    project_root: Path,
    registry: Mapping[str, Any],
    plan: Mapping[str, Any],
    output_dir: Path,
    resume: bool,
    dry_run: bool,
    media_probe: Callable[[Path, Path], Mapping[str, Any]] = probe_video,
    pipeline_loader: Callable[[Mapping[str, Any], Path], Any] = load_pipeline,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    plan_path = output_dir / "generation_plan.json"
    manifest_path = output_dir / "generation_manifest.json"
    complete_path = output_dir / ".complete"
    plan_bytes = canonical_json_bytes(plan)
    require(not complete_path.is_symlink(), "completion marker is symlinked")
    require(not manifest_path.is_symlink(), "generation manifest is symlinked")
    had_complete = complete_path.exists()
    if output_dir.exists():
        require(resume, f"output directory already exists; use --resume: {output_dir}")
        require(output_dir.is_dir() and not output_dir.is_symlink(), f"resume output is not a regular directory: {output_dir}")
        require(plan_path.is_file() and not plan_path.is_symlink(), "resume output has no regular generation plan")
        require(plan_path.read_bytes() == plan_bytes, "resume plan differs from frozen plan")
    else:
        output_dir.mkdir(parents=True)
        atomic_write(plan_path, plan_bytes)
    if had_complete:
        require(complete_path.is_file() and not complete_path.is_symlink(), "completion marker is not a regular file")
        require(manifest_path.is_file() and not manifest_path.is_symlink(), "completed output has no regular manifest")
        require(complete_path.read_text(encoding="utf-8") == sha256_file(manifest_path) + "\n", "completion marker does not bind the manifest")
    if dry_run:
        return {"status": "dry_run_plan_frozen", "completed": 0, "expected": len(plan["items"])}

    runtime_python = resolve_registered(project_root, str(registry["runtime"]["python_executable"]))
    loaded_pipeline: Any | None = None
    completed: list[dict[str, Any]] = []
    for item in plan["items"]:
        receipt_path = Path(str(item["receipt_path"]))
        video_path = Path(str(item["video_path"]))
        require(not receipt_path.is_symlink(), f"formal SAFREE receipt is symlinked: {receipt_path}")
        require(not video_path.is_symlink(), f"formal SAFREE video is symlinked: {video_path}")
        if receipt_path.exists():
            completed.append(_validated_receipt(runtime_python, item, plan, receipt_path, media_probe))
            continue
        require(
            not video_path.exists(),
            f"unreceipted formal SAFREE video exists; refusing receipt backfill: {video_path}",
        )
        if loaded_pipeline is None:
            loaded_pipeline = pipeline_loader(registry, project_root)
            require(isinstance(loaded_pipeline, tuple) and len(loaded_pipeline) == 4, "SAFREE pipeline loader returned an invalid runtime")
        render_item(*loaded_pipeline, item, plan["generation"])
        media = validate_video(runtime_python, video_path, media_probe)
        receipt = _receipt_for(item, plan, video_sha256=sha256_file(video_path), media=media)
        require(not receipt_path.exists(), f"refusing to overwrite formal SAFREE receipt: {receipt_path}")
        atomic_write(receipt_path, canonical_json_bytes(receipt))
        completed.append(receipt)

    require(len(completed) == len(plan["items"]), "formal SAFREE run is incomplete")
    manifest = {**plan, "status": "complete", "outputs": completed}
    manifest_bytes = canonical_json_bytes(manifest)
    if had_complete:
        require(manifest_path.read_bytes() == manifest_bytes, "completed SAFREE manifest differs from validated receipts")
    else:
        require(not manifest_path.is_symlink(), "refusing to replace a symlinked SAFREE manifest")
        atomic_write(manifest_path, manifest_bytes)
        atomic_write(complete_path, (sha256_file(manifest_path) + "\n").encode("utf-8"))
    return {"status": "complete", "completed": len(completed), "expected": len(plan["items"])}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--baseline-registry", type=Path, required=True)
    parser.add_argument("--baseline-receipt", type=Path)
    parser.add_argument("--mechanism", choices=baseline_registry.MECHANISMS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    project_root = args.project_root.resolve()
    registry_path = (args.baseline_registry if args.baseline_registry.is_absolute() else project_root / args.baseline_registry).resolve()
    receipt_path = args.baseline_receipt or registry_path.with_name("build_receipt.json")
    receipt_path = (receipt_path if receipt_path.is_absolute() else project_root / receipt_path).resolve()
    output_dir = (args.output_dir if args.output_dir.is_absolute() else project_root / args.output_dir).resolve()
    registry = load_registry(project_root, registry_path, receipt_path)
    plan = build_plan(
        project_root=project_root,
        registry_path=registry_path,
        registry=registry,
        mechanism=args.mechanism,
        output_dir=output_dir,
    )
    result = run_plan(
        project_root=project_root,
        registry=registry,
        plan=plan,
        output_dir=output_dir,
        resume=args.resume,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
