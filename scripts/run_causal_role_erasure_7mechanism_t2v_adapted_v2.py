#!/usr/bin/env python3
"""Generate one formal mechanism shard with T2VUnlearning-adapted CogVideoX.

The runner is deliberately fail closed.  It consumes the frozen baseline
registry, resolves exactly one of its seven eligible mechanism checkpoints,
keeps every formal case's precommitted seed, and checkpoints one fully
validated receipt per video.  Resume is an infrastructure operation only: a
video is skipped only when its receipt, bytes, media contract, and generation
plan all still agree.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Mapping

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import build_causal_role_erasure_7mechanism_baseline_registry_v2 as baseline_registry


PROTOCOL = "causal_role_erasure_7mechanism_t2v_adapted_generation_v2"
BASELINE = "t2vunlearning_adapted"
RUNNER_RELATIVE = "scripts/run_causal_role_erasure_7mechanism_t2v_adapted_v2.py"
TRAINER_RELATIVE = "scripts/train_causal_role_erasure_7mechanism_t2v_adapted_v2.py"
BASELINE_BUILDER_RELATIVE = "scripts/build_causal_role_erasure_7mechanism_baseline_registry_v2.py"
MEDIA = {"decoded_frames": 49, "fps": "8/1", "height": 480, "width": 720}
ADAPTER_SCALE = 1.0
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


class T2VGenerationError(ValueError):
    """A frozen T2V formal-generation contract was violated."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise T2VGenerationError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def resolve_registered(project_root: Path, value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else project_root / path).resolve()


def _require_regular(path: Path, label: str) -> None:
    require(path.is_file() and not path.is_symlink(), f"{label} missing or symlinked: {path}")


def _validate_model(project_root: Path, registry: Mapping[str, Any]) -> None:
    inventory_path = resolve_registered(project_root, str(registry["model"]["inventory_path"]))
    _require_regular(inventory_path, "model inventory")
    require(
        sha256_file(inventory_path) == registry["model"]["inventory_sha256"],
        "model inventory changed after baseline freeze",
    )
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    files = inventory.get("files")
    require(isinstance(files, list) and files, "model inventory is empty")
    require(len(files) == registry["model"]["file_count"], "model inventory count mismatch")
    model_root = resolve_registered(project_root, str(registry["model"]["root"]))
    require(model_root.is_dir() and not model_root.is_symlink(), f"model root missing or symlinked: {model_root}")
    for item in files:
        path = model_root / str(item["path"])
        _require_regular(path, "registered model file")
        require(path.stat().st_size == int(item["size_bytes"]), f"registered model size changed: {path}")
        require(sha256_file(path) == item["sha256"], f"registered model hash changed: {path}")


def _validate_t2v_checkpoints(project_root: Path, registry: Mapping[str, Any]) -> None:
    implementation = registry.get("implementations", {}).get(BASELINE, {})
    require(implementation.get("status") == "ready", "T2VUnlearning-adapted checkpoints are not ready")
    training_registry_path = resolve_registered(project_root, str(implementation.get("training_registry", "")))
    _require_regular(training_registry_path, "T2V training registry")
    require(
        sha256_file(training_registry_path) == implementation.get("training_registry_sha256"),
        "T2V training registry changed after baseline freeze",
    )
    training_registry = json.loads(training_registry_path.read_text(encoding="utf-8"))
    require(
        training_registry.get("protocol") == "causal_role_erasure_7mechanism_t2v_training_registry_v2",
        "unexpected T2V training registry protocol",
    )
    require(training_registry.get("status") == "t2v_training_specs_frozen_pre_training", "T2V training registry status changed")
    require(training_registry.get("formal_output_inspected") is False, "formal outputs contaminated the T2V training registry")
    baseline_model_root = resolve_registered(project_root, str(registry["model"]["root"]))
    training_model_root = resolve_registered(project_root, str(training_registry.get("model_root", "")))
    require(training_model_root == baseline_model_root, "T2V training base model differs from formal base model")
    training_inventory = resolve_registered(project_root, str(training_registry.get("model_inventory", "")))
    _require_regular(training_inventory, "T2V training model inventory")
    require(
        sha256_file(training_inventory) == training_registry.get("model_inventory_sha256"),
        "T2V training model inventory changed",
    )
    require(
        training_registry.get("model_inventory_sha256") == registry["model"]["inventory_sha256"],
        "T2V training model inventory differs from formal model inventory",
    )
    baseline_runtime = resolve_registered(project_root, str(registry["runtime"]["python_executable"]))
    training_runtime = resolve_registered(project_root, str(training_registry.get("runtime_python", "")))
    require(training_runtime == baseline_runtime, "T2V training runtime differs from formal runtime")
    require(
        training_registry.get("runtime_python_sha256") == registry["runtime"]["python_executable_sha256"],
        "T2V training runtime hash differs from formal runtime",
    )
    for relative, digest in training_registry.get("code_sha256", {}).items():
        path = project_root / str(relative)
        _require_regular(path, "T2V training code artifact")
        require(sha256_file(path) == digest, f"T2V training code changed: {relative}")
    training_runs = training_registry.get("runs")
    require(isinstance(training_runs, list) and len(training_runs) == 7, "T2V training registry must contain seven runs")
    require(
        [item.get("mechanism") for item in training_runs] == list(baseline_registry.MECHANISMS),
        "T2V training-registry mechanism order changed",
    )
    registered_runs = {str(item["mechanism"]): item for item in training_runs}

    checkpoints = implementation.get("checkpoints")
    require(isinstance(checkpoints, list) and len(checkpoints) == 7, "baseline registry must bind seven T2V checkpoints")
    require(
        [item.get("mechanism") for item in checkpoints] == list(baseline_registry.MECHANISMS),
        "T2V checkpoint mechanism order changed",
    )
    for item in checkpoints:
        mechanism = str(item["mechanism"])
        require(item.get("status") == "ready", f"T2V checkpoint is not ready for {mechanism}")
        checkpoint = resolve_registered(project_root, str(item["eligible_checkpoint"]))
        require(checkpoint.is_dir() and not checkpoint.is_symlink(), f"eligible checkpoint missing or symlinked: {checkpoint}")
        training_checkpoint = resolve_registered(
            project_root, str(registered_runs[mechanism]["eligible_checkpoint"])
        )
        require(checkpoint == training_checkpoint, f"eligible checkpoint reference changed for {mechanism}")
        weights = checkpoint / "eraser_weights.pt"
        config = checkpoint / "eraser_config.json"
        state = checkpoint / "training_state.json"
        receipt_path = checkpoint / "training_receipt.json"
        for path, label in (
            (weights, "eraser weights"),
            (config, "eraser config"),
            (state, "training state"),
            (receipt_path, "training receipt"),
        ):
            _require_regular(path, label)
        require(sha256_file(weights) == item.get("weights_sha256"), f"T2V weights changed for {mechanism}")
        require(sha256_file(config) == item.get("config_sha256"), f"T2V config changed for {mechanism}")
        require(sha256_file(state) == item.get("training_state_sha256"), f"T2V state changed for {mechanism}")
        require(sha256_file(receipt_path) == item.get("receipt_sha256"), f"T2V training receipt changed for {mechanism}")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        require(receipt.get("protocol") == training_registry["protocol"], f"T2V receipt protocol mismatch for {mechanism}")
        require(receipt.get("protocol_version") == baseline_registry.PROTOCOL_VERSION, f"T2V receipt version mismatch for {mechanism}")
        require(receipt.get("status") == "eligible", f"T2V checkpoint is ineligible for {mechanism}")
        require(receipt.get("run_id") == registered_runs[mechanism].get("run_id"), f"T2V receipt run ID mismatch for {mechanism}")
        require(receipt.get("mechanism") == mechanism, f"T2V receipt mechanism mismatch for {mechanism}")
        state_payload = json.loads(state.read_text(encoding="utf-8"))
        require(receipt.get("step") == state_payload.get("step"), f"T2V receipt step mismatch for {mechanism}")
        require(
            receipt.get("run_spec_sha256") == registered_runs[mechanism]["run_spec_sha256"],
            f"T2V run-spec receipt mismatch for {mechanism}",
        )
        require(receipt.get("weights_sha256") == item["weights_sha256"], f"T2V weights receipt mismatch for {mechanism}")
        require(receipt.get("config_sha256") == item["config_sha256"], f"T2V config receipt mismatch for {mechanism}")
        require(receipt.get("training_state_sha256") == sha256_file(state), f"T2V state receipt mismatch for {mechanism}")
        require(receipt.get("formal_output_inspected") is False, f"formal outputs contaminated training for {mechanism}")


def load_registry(
    project_root: Path,
    registry_path: Path,
    receipt_path: Path,
    runtime_probe: Callable[[Path], Mapping[str, Any]] = baseline_registry.probe_runtime,
) -> dict[str, Any]:
    _require_regular(registry_path, "baseline registry")
    _require_regular(receipt_path, "baseline registry receipt")
    raw = registry_path.read_bytes()
    registry = json.loads(raw)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    require(registry.get("protocol") == baseline_registry.REGISTRY_PROTOCOL, "unexpected baseline registry protocol")
    require(registry.get("protocol_version") == baseline_registry.PROTOCOL_VERSION, "baseline protocol changed")
    require(registry.get("formal_generation_authorized") is True, "baseline registry does not authorize formal generation")
    require(receipt.get("registry_sha256") == hashlib.sha256(raw).hexdigest(), "baseline registry receipt mismatch")

    formal = resolve_registered(project_root, str(registry["formal_cases"]["path"]))
    _require_regular(formal, "formal case registry")
    require(sha256_file(formal) == registry["formal_cases"]["sha256"], "formal cases changed after baseline freeze")
    baseline_registry.validate_formal_cases(formal)
    require(registry["formal_cases"].get("row_count") == 294, "formal row-count contract changed")
    require(registry["formal_cases"].get("same_case_same_seed_across_streams") is True, "same-seed contract is absent")

    code = registry.get("code_sha256", {})
    for relative, label in (
        (BASELINE_BUILDER_RELATIVE, "baseline registry builder"),
        (TRAINER_RELATIVE, "T2V trainer"),
        (RUNNER_RELATIVE, "T2V formal runner"),
    ):
        path = project_root / relative
        _require_regular(path, f"registered {label}")
        require(relative in code, f"{label} is not baseline-registry bound")
        require(sha256_file(path) == code[relative], f"{label} changed after baseline freeze")

    runtime_python = resolve_registered(project_root, str(registry["runtime"]["python_executable"]))
    _require_regular(runtime_python, "registered runtime Python")
    require(
        sha256_file(runtime_python) == registry["runtime"]["python_executable_sha256"],
        "runtime Python changed after baseline freeze",
    )
    require(Path(sys.executable).resolve() == runtime_python, f"wrong runtime Python: {Path(sys.executable).resolve()} != {runtime_python}")
    observed_runtime = dict(runtime_probe(runtime_python))
    for field in ("python", "implementation", "packages"):
        require(observed_runtime.get(field) == registry["runtime"].get(field), f"runtime {field} changed after baseline freeze")
    _validate_model(project_root, registry)
    _validate_t2v_checkpoints(project_root, registry)
    return registry


def read_mechanism_rows(
    project_root: Path,
    registry: Mapping[str, Any],
    mechanism: str,
) -> list[dict[str, str]]:
    formal = resolve_registered(project_root, str(registry["formal_cases"]["path"]))
    with formal.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["mechanism"] == mechanism]
    require(len(rows) == 42, f"{mechanism} formal shard must contain 42 cases")
    require(len({row["case_id"] for row in rows}) == 42, f"{mechanism} case IDs repeat")
    require(sum(row["case_kind"] == "causal" for row in rows) == 24, f"{mechanism} causal count changed")
    require(sum(row["case_kind"] == "specificity" for row in rows) == 18, f"{mechanism} specificity count changed")
    return rows


def checkpoint_for_mechanism(
    project_root: Path,
    registry: Mapping[str, Any],
    mechanism: str,
) -> tuple[Path, dict[str, Any]]:
    matches = [
        dict(item)
        for item in registry["implementations"][BASELINE]["checkpoints"]
        if item["mechanism"] == mechanism
    ]
    require(len(matches) == 1, f"expected exactly one eligible checkpoint for {mechanism}")
    item = matches[0]
    return resolve_registered(project_root, str(item["eligible_checkpoint"])), item


def build_plan(
    *,
    project_root: Path,
    registry_path: Path,
    registry: Mapping[str, Any],
    mechanism: str,
    output_dir: Path,
) -> dict[str, Any]:
    rows = read_mechanism_rows(project_root, registry, mechanism)
    checkpoint, checkpoint_ref = checkpoint_for_mechanism(project_root, registry, mechanism)
    shared = dict(registry["generation"]["shared"])
    required_generation = {
        "num_frames": 49,
        "fps": 8,
        "height": 480,
        "width": 720,
        "num_inference_steps": 50,
        "guidance_scale": 6.0,
        "scheduler": "CogVideoXDPMScheduler",
        "scheduler_timestep_spacing": "trailing",
        "use_dynamic_cfg": True,
    }
    for key, expected in required_generation.items():
        require(shared.get(key) == expected, f"frozen CogVideoX generation field changed: {key}")
    generation = {**shared, "dtype": "bf16", "adapter_scale": ADAPTER_SCALE}
    items = []
    for row in rows:
        case_id = row["case_id"]
        items.append(
            {
                "case_id": case_id,
                "global_case_index": int(row["global_case_index"]),
                "mechanism": mechanism,
                "case_kind": row["case_kind"],
                "prompt": row["prompt"],
                "prompt_sha256": sha256_text(row["prompt"]),
                "seed": int(row["seed"]),
                "video_path": (output_dir / "videos" / f"{case_id}.mp4").as_posix(),
                "receipt_path": (output_dir / "receipts" / f"{case_id}.json").as_posix(),
            }
        )
    runner = project_root / RUNNER_RELATIVE
    return {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "protocol_version": baseline_registry.PROTOCOL_VERSION,
        "baseline": BASELINE,
        "mechanism": mechanism,
        "baseline_registry": (
            registry_path.resolve().relative_to(project_root.resolve()).as_posix()
            if registry_path.resolve().is_relative_to(project_root.resolve())
            else str(registry_path.resolve())
        ),
        "baseline_registry_sha256": sha256_file(registry_path),
        "runner_sha256": sha256_file(runner),
        "formal_cases_sha256": registry["formal_cases"]["sha256"],
        "model_root": registry["model"]["root"],
        "checkpoint": {
            "path": (
                checkpoint.relative_to(project_root).as_posix()
                if checkpoint.is_relative_to(project_root)
                else str(checkpoint)
            ),
            "weights_sha256": checkpoint_ref["weights_sha256"],
            "config_sha256": checkpoint_ref["config_sha256"],
            "training_receipt_sha256": checkpoint_ref["receipt_sha256"],
        },
        "generation": generation,
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
        rate = Fraction(str(payload["fps"]))
        payload["fps"] = f"{rate.numerator}/{rate.denominator}"
    return payload


def validate_video(
    runtime_python: Path,
    video: Path,
    media_probe: Callable[[Path, Path], Mapping[str, Any]] = probe_video,
) -> dict[str, Any]:
    _require_regular(video, "formal video")
    media = dict(media_probe(runtime_python, video))
    require(media == MEDIA, f"formal video media contract mismatch for {video}: {media}")
    return media


def load_pipeline(registry: Mapping[str, Any], project_root: Path, checkpoint: Path):
    import torch
    import torch.nn as nn
    from diffusers import CogVideoXDPMScheduler, CogVideoXPipeline
    from diffusers.models.attention import Attention
    from diffusers.utils import export_to_video

    config_path = checkpoint / "eraser_config.json"
    weights_path = checkpoint / "eraser_weights.pt"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    require(config.get("eraser_type") == "adapter", "eligible checkpoint eraser type changed")
    rank = int(config["eraser_rank"])
    require(rank > 0, "eraser rank must be positive")

    class AdapterEraser(nn.Module):
        def __init__(self, dim: int, adapter_rank: int):
            super().__init__()
            self.down = nn.Linear(dim, adapter_rank)
            self.act = nn.GELU()
            self.up = nn.Linear(adapter_rank, dim)

        def forward(self, hidden_states):
            dtype = hidden_states.dtype
            return self.up(self.act(self.down(hidden_states.float()))).to(dtype)

    class CogVideoXWithEraser(nn.Module):
        def __init__(self, attention: Attention, adapter_rank: int):
            super().__init__()
            self.attn = attention
            self.adapter = AdapterEraser(attention.to_v.weight.shape[-1], adapter_rank)

        def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
            hidden_states, encoder_hidden_states = self.attn(
                hidden_states, encoder_hidden_states, attention_mask, **kwargs
            )
            hidden_states = hidden_states + ADAPTER_SCALE * self.adapter(hidden_states)
            return hidden_states, encoder_hidden_states

    model_root = resolve_registered(project_root, str(registry["model"]["root"]))
    pipe = CogVideoXPipeline.from_pretrained(model_root, torch_dtype=torch.bfloat16)
    pipe.scheduler = CogVideoXDPMScheduler.from_config(
        pipe.scheduler.config, timestep_spacing="trailing"
    )
    wrappers = []
    for block in pipe.transformer.transformer_blocks:
        block.attn1 = CogVideoXWithEraser(block.attn1, rank)
        wrappers.append(block.attn1)
    saved = torch.load(weights_path, map_location="cpu", weights_only=True)
    expected_keys = {
        f"transformer_blocks.{index}.attn1.adapter" for index in range(len(wrappers))
    }
    require(set(saved) == expected_keys, "adapter checkpoint keys do not match CogVideoX blocks")
    for index, wrapper in enumerate(wrappers):
        wrapper.adapter.load_state_dict(saved[f"transformer_blocks.{index}.attn1.adapter"], strict=True)
    pipe.to("cuda")
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    pipe.set_progress_bar_config(disable=True)
    return torch, export_to_video, pipe


def render_item(
    torch: Any,
    export_to_video: Callable[..., Any],
    pipe: Any,
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
    ).frames[0]
    export_to_video(frames, str(video), fps=int(generation["fps"]))


def _receipt(
    plan: Mapping[str, Any],
    plan_sha256: str,
    item: Mapping[str, Any],
    video_sha256: str,
    media: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "protocol_version": plan["protocol_version"],
        "baseline": BASELINE,
        "mechanism": plan["mechanism"],
        "case_id": item["case_id"],
        "global_case_index": item["global_case_index"],
        "case_kind": item["case_kind"],
        "seed": item["seed"],
        "prompt_sha256": item["prompt_sha256"],
        "generation_plan_sha256": plan_sha256,
        "baseline_registry_sha256": plan["baseline_registry_sha256"],
        "runner_sha256": plan["runner_sha256"],
        "checkpoint": plan["checkpoint"],
        "video_path": item["video_path"],
        "video_sha256": video_sha256,
        "media": dict(media),
    }


def _atomic_write(path: Path, payload: bytes) -> None:
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


def _reject_unplanned_files(output_dir: Path, plan: Mapping[str, Any]) -> None:
    expected_videos = {Path(str(item["video_path"])).resolve() for item in plan["items"]}
    expected_receipts = {Path(str(item["receipt_path"])).resolve() for item in plan["items"]}
    for root_name, expected in (("videos", expected_videos), ("receipts", expected_receipts)):
        root = output_dir / root_name
        if not root.exists():
            continue
        require(root.is_dir() and not root.is_symlink(), f"unexpected {root_name} path: {root}")
        observed = {path.resolve() for path in root.iterdir()}
        extras = sorted(str(path) for path in observed - expected)
        require(not extras, f"unplanned files in formal output: {extras[:3]}")


def run_plan(
    *,
    project_root: Path,
    registry: Mapping[str, Any],
    plan: Mapping[str, Any],
    output_dir: Path,
    resume: bool,
    dry_run: bool,
    media_probe: Callable[[Path, Path], Mapping[str, Any]] = probe_video,
    pipeline_loader: Callable[[Mapping[str, Any], Path, Path], Any] = load_pipeline,
) -> dict[str, Any]:
    plan_path = output_dir / "generation_plan.json"
    manifest_path = output_dir / "generation_manifest.json"
    complete_path = output_dir / ".complete"
    plan_bytes = canonical_json_bytes(plan)
    plan_sha256 = hashlib.sha256(plan_bytes).hexdigest()
    if output_dir.exists():
        require(output_dir.is_dir() and not output_dir.is_symlink(), f"invalid output directory: {output_dir}")
        require(resume, f"fresh-only output directory already exists; use --resume: {output_dir}")
        _require_regular(plan_path, "resume generation plan")
        require(plan_path.read_bytes() == plan_bytes, "resume plan differs from frozen generation plan")
    else:
        output_dir.mkdir(parents=True)
        plan_path.write_bytes(plan_bytes)
    require(
        manifest_path.exists() == complete_path.exists(),
        "generation manifest and completion marker must either both exist or both be absent",
    )
    if manifest_path.exists():
        _require_regular(manifest_path, "generation manifest")
        _require_regular(complete_path, "completion marker")
        require(
            complete_path.read_text(encoding="utf-8") == sha256_file(manifest_path) + "\n",
            "generation manifest completion marker mismatch",
        )
    _reject_unplanned_files(output_dir, plan)
    if dry_run:
        return {"status": "dry_run_plan_frozen", "completed": 0, "expected": len(plan["items"])}

    runtime_python = resolve_registered(project_root, str(registry["runtime"]["python_executable"]))
    completed: list[dict[str, Any]] = []
    pending: list[Mapping[str, Any]] = []
    for item in plan["items"]:
        receipt_path = Path(str(item["receipt_path"]))
        video_path = Path(str(item["video_path"]))
        if receipt_path.exists():
            _require_regular(receipt_path, "formal video receipt")
            media = validate_video(runtime_python, video_path, media_probe)
            expected = _receipt(plan, plan_sha256, item, sha256_file(video_path), media)
            require(
                receipt_path.read_bytes() == canonical_json_bytes(expected),
                f"formal video receipt mismatch: {receipt_path}",
            )
            observed = expected
            completed.append(observed)
        else:
            require(not video_path.exists(), f"unreceipted formal video blocks strict resume: {video_path}")
            pending.append(item)

    if pending:
        checkpoint = resolve_registered(project_root, str(plan["checkpoint"]["path"]))
        torch, export_to_video, pipe = pipeline_loader(registry, project_root, checkpoint)
        for item in pending:
            final_video = Path(str(item["video_path"]))
            final_video.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{final_video.stem}.", suffix=".mp4", dir=final_video.parent
            )
            os.close(descriptor)
            temporary = Path(temporary_name)
            temporary.unlink()
            try:
                temporary_item = {**item, "video_path": temporary.as_posix()}
                render_item(torch, export_to_video, pipe, temporary_item, plan["generation"])
                media = validate_video(runtime_python, temporary, media_probe)
                os.replace(temporary, final_video)
                receipt = _receipt(plan, plan_sha256, item, sha256_file(final_video), media)
                receipt_path = Path(str(item["receipt_path"]))
                _atomic_write(receipt_path, canonical_json_bytes(receipt))
                completed.append(receipt)
                print(f"Finished {item['case_id']}: {final_video}", flush=True)
            finally:
                if temporary.exists():
                    temporary.unlink()

    order = {item["case_id"]: index for index, item in enumerate(plan["items"])}
    completed.sort(key=lambda item: order[item["case_id"]])
    require(len(completed) == len(plan["items"]), "formal T2V shard is incomplete")
    manifest = {**plan, "status": "complete", "outputs": completed}
    manifest_bytes = canonical_json_bytes(manifest)
    if manifest_path.exists():
        require(manifest_path.read_bytes() == manifest_bytes, "completed generation manifest changed on resume")
    else:
        _atomic_write(manifest_path, manifest_bytes)
        _atomic_write(complete_path, (sha256_file(manifest_path) + "\n").encode("utf-8"))
    return {"status": "complete", "completed": len(completed), "expected": len(plan["items"])}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--baseline-registry", type=Path, required=True)
    parser.add_argument("--baseline-receipt", type=Path)
    parser.add_argument("--mechanism", choices=baseline_registry.MECHANISMS, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _under(project_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else project_root / path


def main() -> int:
    args = build_parser().parse_args()
    project_root = args.project_root.resolve()
    registry_path = _under(project_root, args.baseline_registry).resolve()
    receipt_path = args.baseline_receipt or registry_path.with_name("build_receipt.json")
    receipt_path = _under(project_root, receipt_path).resolve()
    output_dir = _under(project_root, args.output_dir).resolve()
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
