#!/usr/bin/env python3
"""Generate frozen official VideoEraser outputs for the formal v2 cases.

The runner consumes a complete fail-closed baseline registry, imports the
hash-bound official VideoEraser snapshot, preserves every formal case's frozen
seed, and emits one validated receipt per video.  Resume is infrastructure
only: the generation plan must be byte-identical and every existing artifact
is revalidated before it is trusted.
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
from typing import Any, Callable, Mapping

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_causal_role_erasure_7mechanism_baseline_registry_v2 as baseline_registry


PROTOCOL = "causal_role_erasure_7mechanism_videoeraser_official_v2"
BASELINE = "videoeraser_official"
BUILDER_RELATIVE = "scripts/build_causal_role_erasure_7mechanism_baseline_registry_v2.py"
RUNNER_RELATIVE = "scripts/run_causal_role_erasure_7mechanism_videoeraser_official_v2.py"
MEDIA = {"decoded_frames": 49, "fps": "8/1", "height": 480, "width": 720}
EXPECTED_STREAM = {"dtype": "bf16", "method": "official_SPEA_ARNG_pipeline"}
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


class VideoEraserError(ValueError):
    """A fail-closed formal VideoEraser generation error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VideoEraserError(message)


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


def resolve_registered(project_root: Path, value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else project_root / path).resolve()


def registry_relative(project_root: Path, path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return str(resolved)


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
        raise VideoEraserError(
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


def load_registry(
    project_root: Path,
    registry_path: Path,
    receipt_path: Path,
    runtime_probe: Callable[[Path], Mapping[str, Any]] = probe_registered_runtime,
) -> dict[str, Any]:
    require(
        registry_path.is_file() and not registry_path.is_symlink(),
        f"baseline registry missing: {registry_path}",
    )
    require(
        receipt_path.is_file() and not receipt_path.is_symlink(),
        f"baseline receipt missing: {receipt_path}",
    )
    registry_raw = registry_path.read_bytes()
    registry = json.loads(registry_raw)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    require(
        registry.get("protocol") == baseline_registry.REGISTRY_PROTOCOL,
        "unexpected baseline registry protocol",
    )
    require(
        registry.get("protocol_version") == baseline_registry.PROTOCOL_VERSION,
        "baseline protocol version changed",
    )
    require(
        registry.get("formal_generation_authorized") is True,
        "baseline registry does not authorize formal generation",
    )
    require(
        registry.get("status") == "baseline_stack_frozen_ready",
        "baseline registry is not the frozen ready stack",
    )
    require(
        receipt.get("protocol") == baseline_registry.REGISTRY_PROTOCOL,
        "unexpected baseline receipt protocol",
    )
    require(
        receipt.get("status") == "baseline_stack_frozen_ready",
        "baseline receipt is not the frozen ready stack",
    )
    require(
        receipt.get("registry_sha256") == hashlib.sha256(registry_raw).hexdigest(),
        "baseline registry receipt mismatch",
    )

    code = registry.get("code_sha256")
    require(isinstance(code, Mapping), "baseline registry has no code bindings")
    for relative, label in (
        (BUILDER_RELATIVE, "baseline registry builder"),
        (RUNNER_RELATIVE, "VideoEraser runner"),
    ):
        require(relative in code, f"{label} is not bound by baseline registry")
        path = project_root / relative
        require(path.is_file() and not path.is_symlink(), f"registered {label} missing: {path}")
        require(sha256_file(path) == code[relative], f"{label} changed after baseline freeze")

    formal = resolve_registered(project_root, str(registry["formal_cases"]["path"]))
    require(formal.is_file() and not formal.is_symlink(), f"formal cases missing: {formal}")
    require(
        sha256_file(formal) == registry["formal_cases"]["sha256"],
        "formal cases changed after registration",
    )
    baseline_registry.validate_formal_cases(formal)

    inventory_path = resolve_registered(project_root, str(registry["model"]["inventory_path"]))
    require(
        inventory_path.is_file() and not inventory_path.is_symlink(),
        f"model inventory missing: {inventory_path}",
    )
    require(
        sha256_file(inventory_path) == registry["model"]["inventory_sha256"],
        "model inventory changed",
    )
    model_root = resolve_registered(project_root, str(registry["model"]["root"]))
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    require(
        inventory.get("file_count") == registry["model"]["file_count"],
        "model inventory count mismatch",
    )
    for item in inventory.get("files", []):
        path = model_root / item["path"]
        require(path.is_file() and not path.is_symlink(), f"registered model file missing: {path}")
        require(path.stat().st_size == item["size_bytes"], f"registered model file size changed: {path}")
        require(sha256_file(path) == item["sha256"], f"registered model file hash changed: {path}")

    runtime_python = resolve_registered(project_root, str(registry["runtime"]["python_executable"]))
    require(
        runtime_python.is_file() and not runtime_python.is_symlink(),
        f"registered runtime Python missing: {runtime_python}",
    )
    require(
        sha256_file(runtime_python) == registry["runtime"]["python_executable_sha256"],
        "runtime Python changed",
    )
    require(
        Path(sys.executable).resolve() == runtime_python,
        f"wrong runtime Python: {Path(sys.executable).resolve()} != {runtime_python}",
    )
    registered_packages = registry.get("runtime", {}).get("packages")
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

    implementation = registry.get("implementations", {}).get(BASELINE, {})
    require(implementation.get("status") == "ready", "official VideoEraser implementation is not ready")
    require(
        implementation.get("official_pipeline_unmodified") is True,
        "VideoEraser registry does not bind the unmodified official pipeline",
    )
    require(
        implementation.get("commit") == baseline_registry.EXPECTED_VIDEOERASER_COMMIT,
        "official VideoEraser commit changed",
    )
    require(
        implementation.get("pipeline_sha256") == baseline_registry.EXPECTED_VIDEOERASER_PIPELINE_SHA256,
        "official VideoEraser pipeline digest changed",
    )
    pipeline = resolve_registered(project_root, str(implementation["pipeline_path"]))
    expected_pipeline = (
        registry_path.parent / "external_snapshot/VideoEraser/CogVideoX/cogvideox_pipeline.py"
    ).resolve()
    require(pipeline == expected_pipeline, "VideoEraser pipeline is not the registry snapshot")
    require(pipeline.is_file() and not pipeline.is_symlink(), f"official VideoEraser snapshot missing: {pipeline}")
    require(
        sha256_file(pipeline) == implementation["pipeline_sha256"],
        "official VideoEraser snapshot changed after registration",
    )
    commit_marker = pipeline.parents[1] / "OFFICIAL_COMMIT"
    require(
        commit_marker.is_file() and not commit_marker.is_symlink(),
        f"official VideoEraser commit marker missing: {commit_marker}",
    )
    require(
        commit_marker.read_text(encoding="utf-8").strip().casefold() == implementation["commit"],
        "official VideoEraser snapshot commit marker changed",
    )
    require(
        registry.get("generation", {}).get("shared") == baseline_registry.COG_GENERATION,
        "registered CogVideoX generation contract changed",
    )
    require(
        registry.get("generation", {}).get("streams", {}).get(BASELINE) == EXPECTED_STREAM,
        "registered VideoEraser stream contract changed",
    )
    require(
        BASELINE in registry.get("streams", []),
        "VideoEraser stream is absent from baseline registry",
    )
    return registry


def read_formal_rows(
    project_root: Path,
    registry: Mapping[str, Any],
    mechanism: str | None,
) -> list[dict[str, str]]:
    path = resolve_registered(project_root, str(registry["formal_cases"]["path"]))
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    baseline_registry.validate_formal_cases(path)
    if mechanism is None:
        require(len(rows) == 294, "formal row count must be 294")
        return rows
    selected = [row for row in rows if row["mechanism"] == mechanism]
    require(len(selected) == 42, f"{mechanism} formal row count must be 42")
    return selected


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
    implementation = registry["implementations"][BASELINE]
    items = []
    for row in rows:
        case_id = row["case_id"]
        items.append(
            {
                "case_id": case_id,
                "global_case_index": int(row["global_case_index"]),
                "mechanism": row["mechanism"],
                "case_kind": row["case_kind"],
                "prompt": row["prompt"],
                "target_concept": str(concepts[row["mechanism"]]),
                "seed": int(row["seed"]),
                "video_path": (output_dir / "videos" / f"{case_id}.mp4").as_posix(),
                "receipt_path": (output_dir / "receipts" / f"{case_id}.json").as_posix(),
            }
        )
    return {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "protocol_version": baseline_registry.PROTOCOL_VERSION,
        "baseline": BASELINE,
        "mechanism": mechanism or "ALL",
        "baseline_registry": registry_relative(project_root, registry_path),
        "baseline_registry_sha256": sha256_file(registry_path),
        "formal_cases_sha256": registry["formal_cases"]["sha256"],
        "model_root": registry["model"]["root"],
        "implementation": {
            "repository": implementation["repository"],
            "commit": implementation["commit"],
            "pipeline_path": implementation["pipeline_path"],
            "pipeline_sha256": implementation["pipeline_sha256"],
            "official_pipeline_unmodified": True,
        },
        "generation": {**registry["generation"]["shared"], **EXPECTED_STREAM},
        "items": items,
    }


def probe_video(runtime_python: Path, video: Path) -> dict[str, Any]:
    result = subprocess.run(
        [str(runtime_python), "-I", "-c", PYAV_PROBE_CODE, str(video)],
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(result.stdout)
    if payload.get("fps"):
        value = Fraction(str(payload["fps"]))
        payload["fps"] = f"{value.numerator}/{value.denominator}"
    return payload


def validate_video(
    runtime_python: Path,
    video: Path,
    media_probe: Callable[[Path, Path], Mapping[str, Any]] = probe_video,
) -> dict[str, Any]:
    require(video.is_file() and not video.is_symlink(), f"formal video missing: {video}")
    payload = dict(media_probe(runtime_python, video))
    require(payload == MEDIA, f"formal video media contract mismatch for {video}: {payload}")
    return payload


def load_official_module(pipeline_path: Path):
    module_spec = importlib.util.spec_from_file_location(
        "frozen_videoeraser_cogvideox_pipeline_v2",
        pipeline_path,
    )
    require(
        module_spec is not None and module_spec.loader is not None,
        f"cannot import official VideoEraser pipeline: {pipeline_path}",
    )
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    require(hasattr(module, "CogVideoXPipeline"), "official VideoEraser snapshot has no CogVideoXPipeline")
    return module


def load_pipeline(registry: Mapping[str, Any], project_root: Path):
    import torch
    from diffusers import CogVideoXDPMScheduler
    from diffusers.utils import export_to_video

    implementation = registry["implementations"][BASELINE]
    pipeline_path = resolve_registered(project_root, str(implementation["pipeline_path"]))
    module = load_official_module(pipeline_path)
    model_root = resolve_registered(project_root, str(registry["model"]["root"]))
    pipe = module.CogVideoXPipeline.from_pretrained(model_root, torch_dtype=torch.bfloat16)
    pipe.scheduler = CogVideoXDPMScheduler.from_config(
        pipe.scheduler.config,
        timestep_spacing="trailing",
    )
    pipe.to("cuda")
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    return torch, export_to_video, pipe


def render_item(
    torch,
    export_to_video,
    pipe,
    item: Mapping[str, Any],
    generation: Mapping[str, Any],
) -> None:
    video = Path(str(item["video_path"]))
    video.parent.mkdir(parents=True, exist_ok=True)
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
        concept=str(item["target_concept"]),
    ).frames[0]
    export_to_video(frames, str(video), fps=int(generation["fps"]))


def receipt_for(
    *,
    plan: Mapping[str, Any],
    plan_sha256: str,
    item: Mapping[str, Any],
    video_path: Path,
    media: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "protocol_version": plan["protocol_version"],
        "baseline": BASELINE,
        "case_id": item["case_id"],
        "global_case_index": item["global_case_index"],
        "mechanism": item["mechanism"],
        "seed": item["seed"],
        "video_path": item["video_path"],
        "video_sha256": sha256_file(video_path),
        "media": dict(media),
        "generation_plan_sha256": plan_sha256,
        "baseline_registry_sha256": plan["baseline_registry_sha256"],
        "official_commit": plan["implementation"]["commit"],
        "official_pipeline_sha256": plan["implementation"]["pipeline_sha256"],
    }


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
    require(not output_dir.is_symlink(), f"output directory must not be a symlink: {output_dir}")
    plan_path = output_dir / "generation_plan.json"
    manifest_path = output_dir / "generation_manifest.json"
    complete_path = output_dir / ".complete"
    plan_bytes = canonical_json_bytes(plan)
    plan_sha256 = hashlib.sha256(plan_bytes).hexdigest()
    require(not manifest_path.is_symlink(), "generation manifest is symlinked")
    require(not complete_path.is_symlink(), "completion marker is symlinked")
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
        require(
            complete_path.read_text(encoding="utf-8") == sha256_file(manifest_path) + "\n",
            "completion marker does not bind the manifest",
        )
    if dry_run:
        return {
            "status": "dry_run_plan_frozen",
            "completed": 0,
            "expected": len(plan["items"]),
        }

    runtime_python = resolve_registered(project_root, str(registry["runtime"]["python_executable"]))
    completed: list[dict[str, Any]] = []
    pending: list[Mapping[str, Any]] = []
    for item in plan["items"]:
        receipt_path = Path(str(item["receipt_path"]))
        video_path = Path(str(item["video_path"]))
        require(not receipt_path.is_symlink(), f"receipt path must not be a symlink: {receipt_path}")
        require(not video_path.is_symlink(), f"video path must not be a symlink: {video_path}")
        if receipt_path.exists():
            require(receipt_path.is_file(), f"invalid receipt path: {receipt_path}")
            media = validate_video(runtime_python, video_path, media_probe)
            expected_receipt = receipt_for(
                plan=plan,
                plan_sha256=plan_sha256,
                item=item,
                video_path=video_path,
                media=media,
            )
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            require(receipt == expected_receipt, f"formal receipt mismatch: {receipt_path}")
            completed.append(receipt)
        else:
            require(
                not video_path.exists(),
                f"unreceipted formal video exists; refusing receipt backfill: {video_path}",
            )
            pending.append(item)

    loaded_pipeline = None
    if pending:
        for item in pending:
            video_path = Path(str(item["video_path"]))
            receipt_path = Path(str(item["receipt_path"]))
            require(
                not video_path.exists(),
                f"unreceipted formal video exists; refusing receipt backfill: {video_path}",
            )
            if loaded_pipeline is None:
                loaded_pipeline = pipeline_loader(registry, project_root)
            torch, export_to_video, pipe = loaded_pipeline
            render_item(torch, export_to_video, pipe, item, plan["generation"])
            media = validate_video(runtime_python, video_path, media_probe)
            receipt = receipt_for(
                plan=plan,
                plan_sha256=plan_sha256,
                item=item,
                video_path=video_path,
                media=media,
            )
            atomic_write(receipt_path, canonical_json_bytes(receipt))
            completed.append(receipt)
            print(f"Finished {item['case_id']}: {video_path}", flush=True)

    order = {item["case_id"]: index for index, item in enumerate(plan["items"])}
    completed.sort(key=lambda item: order[item["case_id"]])
    require(len(completed) == len(plan["items"]), "formal VideoEraser run is incomplete")
    manifest = {**plan, "status": "complete", "outputs": completed}
    manifest_bytes = canonical_json_bytes(manifest)
    if had_complete:
        require(
            manifest_path.read_bytes() == manifest_bytes,
            "completed VideoEraser manifest differs from validated receipts",
        )
    else:
        atomic_write(manifest_path, manifest_bytes)
        complete_payload = sha256_file(manifest_path) + "\n"
        atomic_write(complete_path, complete_payload.encode("utf-8"))
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
    registry_path = (
        args.baseline_registry
        if args.baseline_registry.is_absolute()
        else project_root / args.baseline_registry
    ).resolve()
    receipt_path = args.baseline_receipt or registry_path.with_name("build_receipt.json")
    receipt_path = (
        receipt_path if receipt_path.is_absolute() else project_root / receipt_path
    ).resolve()
    output_dir = (
        args.output_dir if args.output_dir.is_absolute() else project_root / args.output_dir
    ).resolve()
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
