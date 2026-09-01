#!/usr/bin/env python3
"""Generate frozen CogVideoX Original/Negative controls for all formal cases.

The runner consumes the fail-closed baseline registry, preserves each case's
arbitrary frozen seed, checkpoints one validated receipt per video, and never
silently trusts an existing file.  It supports infrastructure-only resume from
the exact same deterministic plan.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import build_causal_role_erasure_7mechanism_baseline_registry_v2 as baseline_registry


PROTOCOL = "causal_role_erasure_7mechanism_cogvideox_controls_v2"
BASELINES = ("cogvideox_original", "negative_prompt")
MEDIA = {"decoded_frames": 49, "fps": "8/1", "height": 480, "width": 720}
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


class ControlError(ValueError):
    """A fail-closed formal control generation error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ControlError(message)


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


def resolve_registered(project_root: Path, value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else project_root / path).resolve()


def load_registry(project_root: Path, registry_path: Path, receipt_path: Path) -> dict[str, Any]:
    require(registry_path.is_file() and not registry_path.is_symlink(), f"baseline registry missing: {registry_path}")
    require(receipt_path.is_file() and not receipt_path.is_symlink(), f"baseline receipt missing: {receipt_path}")
    registry_raw = registry_path.read_bytes()
    registry = json.loads(registry_raw)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    require(registry.get("protocol") == baseline_registry.REGISTRY_PROTOCOL, "unexpected baseline registry protocol")
    require(registry.get("protocol_version") == baseline_registry.PROTOCOL_VERSION, "baseline protocol version changed")
    require(registry.get("formal_generation_authorized") is True, "baseline registry does not authorize formal generation")
    require(receipt.get("registry_sha256") == hashlib.sha256(registry_raw).hexdigest(), "baseline registry receipt mismatch")
    code = registry.get("code_sha256", {})
    for relative, label in (
        ("scripts/build_causal_role_erasure_7mechanism_baseline_registry_v2.py", "baseline registry builder"),
        ("scripts/run_causal_role_erasure_7mechanism_cogvideox_controls_v2.py", "control runner"),
    ):
        path = project_root / relative
        require(relative in code, f"{label} is not bound by baseline registry")
        require(path.is_file() and not path.is_symlink(), f"registered {label} missing: {path}")
        require(sha256_file(path) == code[relative], f"{label} changed after baseline freeze")
    formal = resolve_registered(project_root, registry["formal_cases"]["path"])
    require(sha256_file(formal) == registry["formal_cases"]["sha256"], "formal cases changed after registration")
    inventory_path = resolve_registered(project_root, registry["model"]["inventory_path"])
    require(sha256_file(inventory_path) == registry["model"]["inventory_sha256"], "model inventory changed")
    model_root = resolve_registered(project_root, registry["model"]["root"])
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    require(inventory.get("file_count") == registry["model"]["file_count"], "model inventory count mismatch")
    for item in inventory.get("files", []):
        path = model_root / item["path"]
        require(path.is_file() and not path.is_symlink(), f"registered model file missing: {path}")
        require(path.stat().st_size == item["size_bytes"], f"registered model file size changed: {path}")
        require(sha256_file(path) == item["sha256"], f"registered model file hash changed: {path}")
    runtime_python = resolve_registered(project_root, registry["runtime"]["python_executable"])
    require(runtime_python.is_file() and not runtime_python.is_symlink(), f"registered runtime Python missing: {runtime_python}")
    require(sha256_file(runtime_python) == registry["runtime"]["python_executable_sha256"], "runtime Python changed")
    require(Path(sys.executable).resolve() == runtime_python, f"wrong runtime Python: {Path(sys.executable).resolve()} != {runtime_python}")
    observed_runtime = baseline_registry.probe_runtime(runtime_python)
    for field in ("python", "implementation", "packages"):
        require(observed_runtime.get(field) == registry["runtime"].get(field), f"runtime {field} changed after baseline freeze")
    return registry


def read_formal_rows(project_root: Path, registry: Mapping[str, Any], mechanism: str | None) -> list[dict[str, str]]:
    path = resolve_registered(project_root, str(registry["formal_cases"]["path"]))
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    baseline_registry.validate_formal_cases(path)
    if mechanism is not None:
        rows = [row for row in rows if row["mechanism"] == mechanism]
        require(len(rows) == 42, f"{mechanism} formal row count must be 42")
    else:
        require(len(rows) == 294, "formal row count must be 294")
    return rows


def build_plan(
    *,
    project_root: Path,
    registry_path: Path,
    registry: Mapping[str, Any],
    baseline: str,
    mechanism: str | None,
    output_dir: Path,
) -> dict[str, Any]:
    require(baseline in BASELINES, f"unsupported control baseline: {baseline}")
    rows = read_formal_rows(project_root, registry, mechanism)
    concepts = registry["mechanism_concepts"]
    items = []
    for row in rows:
        case_id = row["case_id"]
        concept = str(concepts[row["mechanism"]])
        items.append(
            {
                "case_id": case_id,
                "global_case_index": int(row["global_case_index"]),
                "mechanism": row["mechanism"],
                "case_kind": row["case_kind"],
                "prompt": row["prompt"],
                "target_concept": concept,
                "negative_prompt": concept if baseline == "negative_prompt" else None,
                "seed": int(row["seed"]),
                "video_path": (output_dir / "videos" / f"{case_id}.mp4").as_posix(),
                "receipt_path": (output_dir / "receipts" / f"{case_id}.json").as_posix(),
            }
        )
    return {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "protocol_version": baseline_registry.PROTOCOL_VERSION,
        "baseline": baseline,
        "mechanism": mechanism or "ALL",
        "baseline_registry": (
            registry_path.resolve().relative_to(project_root.resolve()).as_posix()
            if registry_path.resolve().is_relative_to(project_root.resolve())
            else str(registry_path.resolve())
        ),
        "baseline_registry_sha256": sha256_file(registry_path),
        "formal_cases_sha256": registry["formal_cases"]["sha256"],
        "model_root": registry["model"]["root"],
        "generation": {**registry["generation"]["shared"], "dtype": "bf16"},
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


def load_pipeline(registry: Mapping[str, Any], project_root: Path):
    import torch
    from diffusers import CogVideoXDPMScheduler, CogVideoXPipeline
    from diffusers.utils import export_to_video

    model_root = resolve_registered(project_root, str(registry["model"]["root"]))
    pipe = CogVideoXPipeline.from_pretrained(model_root, torch_dtype=torch.bfloat16)
    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    pipe.to("cuda")
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    return torch, export_to_video, pipe


def render_item(torch, export_to_video, pipe, item: Mapping[str, Any], generation: Mapping[str, Any]) -> None:
    video = Path(str(item["video_path"]))
    video.parent.mkdir(parents=True, exist_ok=True)
    frames = pipe(
        prompt=str(item["prompt"]),
        negative_prompt=item.get("negative_prompt"),
        num_videos_per_prompt=1,
        num_inference_steps=int(generation["num_inference_steps"]),
        num_frames=int(generation["num_frames"]),
        height=int(generation["height"]),
        width=int(generation["width"]),
        use_dynamic_cfg=bool(generation["use_dynamic_cfg"]),
        guidance_scale=float(generation["guidance_scale"]),
        generator=torch.Generator(device="cuda").manual_seed(int(item["seed"])),
    ).frames[0]
    export_to_video(frames, str(video), fps=int(generation["fps"]))


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
    plan_path = output_dir / "generation_plan.json"
    plan_bytes = canonical_json_bytes(plan)
    if output_dir.exists():
        require(resume, f"output directory already exists; use --resume: {output_dir}")
        require(plan_path.is_file(), "resume output has no generation plan")
        require(plan_path.read_bytes() == plan_bytes, "resume plan differs from frozen plan")
    else:
        output_dir.mkdir(parents=True)
        plan_path.write_bytes(plan_bytes)
    if dry_run:
        return {"status": "dry_run_plan_frozen", "completed": 0, "expected": len(plan["items"])}

    runtime_python = resolve_registered(project_root, str(registry["runtime"]["python_executable"]))
    pending = []
    completed: list[dict[str, Any]] = []
    for item in plan["items"]:
        receipt_path = Path(str(item["receipt_path"]))
        video_path = Path(str(item["video_path"]))
        if receipt_path.is_file():
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            media = validate_video(runtime_python, video_path, media_probe)
            require(receipt.get("video_sha256") == sha256_file(video_path), f"video changed after receipt: {video_path}")
            require(receipt.get("media") == media, f"media changed after receipt: {video_path}")
            completed.append(receipt)
        else:
            pending.append(item)
    if pending:
        torch, export_to_video, pipe = pipeline_loader(registry, project_root)
        for item in pending:
            video_path = Path(str(item["video_path"]))
            receipt_path = Path(str(item["receipt_path"]))
            if video_path.exists():
                require(resume, f"unreceipted video exists outside resume: {video_path}")
            else:
                render_item(torch, export_to_video, pipe, item, plan["generation"])
            media = validate_video(runtime_python, video_path, media_probe)
            receipt = {
                "case_id": item["case_id"],
                "baseline": plan["baseline"],
                "seed": item["seed"],
                "video_path": item["video_path"],
                "video_sha256": sha256_file(video_path),
                "media": media,
            }
            receipt_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = Path(tempfile.mkstemp(prefix=f".{receipt_path.name}.", dir=receipt_path.parent)[1])
            try:
                temporary.write_bytes(canonical_json_bytes(receipt))
                os.replace(temporary, receipt_path)
            finally:
                if temporary.exists():
                    temporary.unlink()
            completed.append(receipt)
    completed.sort(key=lambda item: next(i for i, plan_item in enumerate(plan["items"]) if plan_item["case_id"] == item["case_id"]))
    require(len(completed) == len(plan["items"]), "formal control run is incomplete")
    manifest = {**plan, "status": "complete", "outputs": completed}
    manifest_path = output_dir / "generation_manifest.json"
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    (output_dir / ".complete").write_text(sha256_file(manifest_path) + "\n", encoding="utf-8")
    return {"status": "complete", "completed": len(completed), "expected": len(plan["items"])}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--baseline-registry", type=Path, required=True)
    parser.add_argument("--baseline-receipt", type=Path)
    parser.add_argument("--baseline", choices=BASELINES, required=True)
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
        baseline=args.baseline,
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
