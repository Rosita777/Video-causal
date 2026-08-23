#!/usr/bin/env python3
"""Plan or generate the six fresh target-candidate shards for the 7m protocol.

Water's 192 historical targets are hash-bound and never regenerated.  The six
new mechanisms run as one Wan process per mechanism in a four-job wave followed
by a two-job wave.  There is deliberately no resume or skip-existing mode.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import stat
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


PROTOCOL_ID = "causal_role_erasure_7m_single_seed_v2"
RUNNER_ID = "causal_role_erasure_7mechanism_targets_runner_v2"
MECHANISM_ORDER = (
    "water_impact",
    "rigid_collision",
    "brittle_fracture",
    "powder_impact",
    "elastic_deformation",
    "material_release",
    "surface_trace",
)
GENERATED_MECHANISMS = MECHANISM_ORDER[1:]
ROWS_PER_MECHANISM = 192
EXPECTED_ROWS = len(MECHANISM_ORDER) * ROWS_PER_MECHANISM
EXPECTED_REUSED_ROWS = ROWS_PER_MECHANISM
EXPECTED_GENERATED_ROWS = len(GENERATED_MECHANISMS) * ROWS_PER_MECHANISM
WATER_ORIGIN = "water_v1_reuse"
NEW_ORIGIN = "new_generation_v2"
EXPECTED_WATER_REUSE_MANIFEST_SHA256 = (
    "406e4b2d06c80e415dca05bda2262924e07c3aea67398b5caca3687fca7c140d"
)

STEPS = 25
GUIDANCE_SCALE = 5.0
NUM_FRAMES = 49
FPS = 8
HEIGHT = 480
WIDTH = 832
DTYPE = "bf16"

DEFAULT_TARGET_CANDIDATES = Path(
    "data/causal_role_erasure_7mechanism_main_v2/target_candidates.csv"
)
DEFAULT_OUTPUT_ROOT = Path(
    "outputs/causal_role_erasure_7mechanism_main_v2/target_generation"
)
DEFAULT_WATER_REUSE_MANIFEST = Path(
    "outputs/water_impact_dynamic_v1/train_targets_v1/generation_manifest.json"
)
DEFAULT_GENERATOR = Path("scripts/generate_wan_clean.py")
DEFAULT_PYTHON = Path("models/.wan-runtime/bin/python")
DEFAULT_MODEL = Path("models/Wan2.1-T2V-1.3B-Diffusers")

CANONICAL_FIELDS = (
    "candidate_id",
    "global_index",
    "mechanism",
    "seed",
    "target_prompt",
    "target_origin",
    "prompt_shard",
    "prompt_shard_index",
)
CANONICAL_PROTOCOL_FIELD = "protocol_version"
FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "candidate_id": ("target_id",),
    "global_index": ("candidate_index",),
    "target_prompt": ("target_generation_prompt", "prompt"),
    "target_origin": ("target_asset_mode", "target_disposition"),
    "prompt_shard": ("prompt_shard_path", "prompt_file"),
    "prompt_shard_index": ("shard_index", "local_index"),
    "seed": ("fixed_seed",),
}
ORIGIN_ALIASES = {
    "historical_water_v1_import_candidate": WATER_ORIGIN,
    "generate_new": NEW_ORIGIN,
}
SEALED_TOKENS = ("final36", "sealed-final", "sealed_final")

PYAV_PROBE_CODE = r"""
import av, json, sys
container = av.open(sys.argv[1])
streams = [stream for stream in container.streams if stream.type == "video"]
if len(streams) != 1:
    raise SystemExit("expected exactly one video stream")
stream = streams[0]
frames = list(container.decode(stream))
rate = stream.average_rate
print(json.dumps({
    "streams": 1,
    "decoded_frames": len(frames),
    "fps": None if rate is None else str(rate),
    "width": stream.width,
    "height": stream.height,
}))
""".strip()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def regular_file(path: Path, label: str) -> None:
    require(path.is_file() and not path.is_symlink(), f"{label} is missing or symlinked: {path}")


def reject_sealed_paths(*paths: Path | str) -> None:
    for path in paths:
        lowered = str(path).casefold()
        require(
            not any(token in lowered for token in SEALED_TOKENS),
            f"sealed/final36 path is forbidden: {path}",
        )


def parse_gpus(value: str) -> list[int]:
    try:
        result = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--gpus must be four comma-separated integers") from exc
    if len(result) != 4 or len(set(result)) != 4 or any(item < 0 for item in result):
        raise argparse.ArgumentTypeError("--gpus must be exactly four distinct non-negative indices")
    return result


def parse_prompt_shard(path: Path) -> list[dict[str, str]]:
    regular_file(path, "target prompt shard")
    items: list[dict[str, str]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = [part.strip() for part in line.split("|")]
        require(
            len(parts) == 3 and all(parts),
            f"{path}:{line_number}: expected '<target prompt> | <exclusion phrase> | <expected state>'",
        )
        items.append(
            {"prompt": parts[0], "target_concept": parts[1], "expected_effect": parts[2]}
        )
    require(len(items) == ROWS_PER_MECHANISM, f"{path}: expected exactly 192 prompts")
    return items


def _canonical_value(raw: Mapping[str, str], field: str, row_index: int) -> str:
    candidates = (field, *FIELD_ALIASES.get(field, ()))
    present = [(name, raw.get(name, "")) for name in candidates if name in raw]
    require(present, f"row {row_index}: missing canonical field {field} or a supported alias")
    nonempty = [(name, value.strip()) for name, value in present if value is not None and value.strip()]
    if field in {"prompt_shard", "prompt_shard_index"} and not nonempty:
        return ""
    require(nonempty, f"row {row_index}: empty {field}")
    values = {value for _, value in nonempty}
    require(len(values) == 1, f"row {row_index}: conflicting aliases for {field}")
    return nonempty[0][1]


def _resolve_existing(project_root: Path, value: str, label: str) -> Path:
    lexical = Path(value)
    reject_sealed_paths(lexical)
    path = lexical if lexical.is_absolute() else project_root / lexical
    resolved = path.resolve(strict=True)
    reject_sealed_paths(resolved)
    return resolved


def _resolve_lexical(project_root: Path, value: str) -> Path:
    path = Path(value)
    reject_sealed_paths(path)
    return (path if path.is_absolute() else project_root / path).resolve()


def load_target_candidates(
    project_root: Path,
    csv_path: Path,
    water_reuse_manifest: Path,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, str]]], dict[str, Any]]:
    regular_file(csv_path, "target_candidates.csv")
    regular_file(water_reuse_manifest, "Water reuse generation manifest")
    require(
        sha256_file(water_reuse_manifest) == EXPECTED_WATER_REUSE_MANIFEST_SHA256,
        "Water reuse generation manifest SHA-256 mismatch",
    )
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames is not None, "target_candidates.csv has no header")
        raw_rows = list(reader)
        header = set(reader.fieldnames)
    require(len(raw_rows) == EXPECTED_ROWS, f"target_candidates.csv must contain {EXPECTED_ROWS} rows")
    protocol_columns = [
        field for field in (CANONICAL_PROTOCOL_FIELD, "protocol_id") if field in header
    ]
    require(protocol_columns, "target candidates require protocol_version (protocol_id is a compatibility alias)")
    for field in protocol_columns:
        require(
            {str(row.get(field, "")).strip() for row in raw_rows} == {PROTOCOL_ID},
            f"target candidate {field} mismatch",
        )

    rows: list[dict[str, Any]] = []
    for row_index, raw in enumerate(raw_rows):
        canonical = {field: _canonical_value(raw, field, row_index) for field in CANONICAL_FIELDS}
        origin = ORIGIN_ALIASES.get(canonical["target_origin"], canonical["target_origin"])
        try:
            global_index = int(canonical["global_index"])
            seed = int(canonical["seed"])
        except ValueError as exc:
            raise ValueError(f"row {row_index}: global_index and seed must be integers") from exc
        shard_index: int | None
        if canonical["prompt_shard_index"]:
            try:
                shard_index = int(canonical["prompt_shard_index"])
            except ValueError as exc:
                raise ValueError(f"row {row_index}: prompt_shard_index must be an integer") from exc
        else:
            shard_index = None
        row: dict[str, Any] = dict(raw)
        row.update(canonical)
        row.update(
            global_index=global_index,
            seed=seed,
            target_origin=origin,
            prompt_shard_index=shard_index,
        )
        rows.append(row)

    require([row["global_index"] for row in rows] == list(range(EXPECTED_ROWS)), "global_index must be canonical file order 0..1343")
    require(len({row["candidate_id"] for row in rows}) == EXPECTED_ROWS, "candidate_id values must be unique")
    require(len({row["seed"] for row in rows}) == EXPECTED_ROWS, "target candidate seeds must be unique")
    require(
        Counter(row["mechanism"] for row in rows)
        == Counter({mechanism: ROWS_PER_MECHANISM for mechanism in MECHANISM_ORDER}),
        "target candidates must contain exactly 192 rows per mechanism",
    )
    for mechanism_index, mechanism in enumerate(MECHANISM_ORDER):
        block = rows[mechanism_index * ROWS_PER_MECHANISM : (mechanism_index + 1) * ROWS_PER_MECHANISM]
        require({row["mechanism"] for row in block} == {mechanism}, "mechanism blocks/order are not canonical")
        expected_origin = WATER_ORIGIN if mechanism == "water_impact" else NEW_ORIGIN
        require({row["target_origin"] for row in block} == {expected_origin}, f"{mechanism}: target_origin mismatch")
        if mechanism == "water_impact":
            require(all(row["prompt_shard_index"] is None for row in block), "Water reuse rows must not enter a prompt shard")
            require(all(not row["prompt_shard"] for row in block), "Water reuse rows must not name a prompt shard")
        if "mechanism_index" in header:
            require(
                {str(row.get("mechanism_index", "")).strip() for row in block}
                == {str(mechanism_index)},
                f"{mechanism}: mechanism_index mismatch",
            )

    try:
        water_payload = json.loads(water_reuse_manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid Water reuse generation manifest") from exc
    require(isinstance(water_payload, dict), "Water reuse manifest must be an object")
    require(water_payload.get("baseline") == "negative_prompt", "Water reuse baseline must be negative_prompt")
    require(water_payload.get("dry_run") is False, "Water reuse manifest cannot be a dry-run")
    water_items = water_payload.get("items")
    require(isinstance(water_items, list) and len(water_items) == ROWS_PER_MECHANISM, "Water reuse manifest must contain 192 items")
    for index, (row, item) in enumerate(zip(rows[:ROWS_PER_MECHANISM], water_items)):
        require(isinstance(item, dict) and item.get("index") == index, "Water reuse item order mismatch")
        require(item.get("prompt") == row["target_prompt"], f"Water reuse row {index}: prompt mismatch")
        require(item.get("seed") == row["seed"], f"Water reuse row {index}: seed mismatch")
        if row.get("target_video_path", "").strip():
            require(
                Path(str(item.get("video_path", ""))).resolve()
                == _resolve_lexical(project_root, str(row["target_video_path"])),
                f"Water reuse row {index}: video path mismatch",
            )

    shard_items: dict[str, list[dict[str, str]]] = {}
    shard_paths: set[Path] = set()
    for mechanism in GENERATED_MECHANISMS:
        block = [row for row in rows if row["mechanism"] == mechanism]
        paths = {row["prompt_shard"] for row in block}
        require(len(paths) == 1 and next(iter(paths)), f"{mechanism}: exactly one prompt shard is required")
        shard_path = _resolve_existing(project_root, next(iter(paths)), f"{mechanism} prompt shard")
        regular_file(shard_path, f"{mechanism} prompt shard")
        require(shard_path not in shard_paths, "prompt shards must be unique by mechanism")
        shard_paths.add(shard_path)
        parsed = parse_prompt_shard(shard_path)
        require([row["prompt_shard_index"] for row in block] == list(range(ROWS_PER_MECHANISM)), f"{mechanism}: prompt_shard_index must be 0..191")
        for local_index, (row, item) in enumerate(zip(block, parsed)):
            require(item["prompt"] == row["target_prompt"], f"{mechanism} row {local_index}: shard prompt mismatch")
            expected_state = str(row.get("expected_counterfactual_state", "")).strip()
            if expected_state:
                require(item["expected_effect"] == expected_state, f"{mechanism} row {local_index}: expected state mismatch")
            row["resolved_prompt_shard"] = str(shard_path)
            row["target_concept"] = item["target_concept"]
            row["expected_effect"] = item["expected_effect"]
        shard_items[mechanism] = parsed

    require(len(shard_paths) == 6, "exactly six non-Water prompt shards are required")
    schema = {
        "protocol_columns_present": protocol_columns,
        "canonical_columns_present": sorted(set(CANONICAL_FIELDS) & header),
        "aliases_used": sorted(
            alias
            for canonical, aliases in FIELD_ALIASES.items()
            if canonical not in header
            for alias in aliases
            if alias in header
        ),
    }
    return rows, shard_items, schema


def build_generator_command(
    python_executable: Path,
    generator: Path,
    prompt_shard: Path,
    output_dir: Path,
    model: Path,
    seeds: Sequence[int],
) -> list[str]:
    require(len(seeds) == ROWS_PER_MECHANISM, "each generator job requires 192 explicit seeds")
    command = [
        str(python_executable), str(generator),
        "--baseline", "negative_prompt",
        "--prompts", str(prompt_shard),
        "--output-dir", str(output_dir),
        "--model", str(model),
        "--seed", str(seeds[0]),
        "--seeds", ",".join(str(seed) for seed in seeds),
        "--steps", str(STEPS),
        "--guidance-scale", str(GUIDANCE_SCALE),
        "--num-frames", str(NUM_FRAMES),
        "--fps", str(FPS),
        "--height", str(HEIGHT),
        "--width", str(WIDTH),
        "--dtype", DTYPE,
        "--device", "cuda",
        "--vae-slicing", "--vae-tiling",
    ]
    require("--skip-existing" not in command and "--dry-run" not in command, "skip/resume/dry child generation is forbidden")
    return command


def build_plan(
    *,
    project_root: Path,
    rows: Sequence[Mapping[str, Any]],
    schema: Mapping[str, Any],
    target_candidates: Path,
    water_reuse_manifest: Path,
    output_root: Path,
    generator: Path,
    python_executable: Path,
    model: Path,
    gpus: Sequence[int],
    dry_run: bool,
) -> dict[str, Any]:
    require(len(gpus) == 4 and len(set(gpus)) == 4, "exactly four unique GPUs are required")
    jobs: list[dict[str, Any]] = []
    for generated_index, mechanism in enumerate(GENERATED_MECHANISMS):
        block = [row for row in rows if row["mechanism"] == mechanism]
        prompt_shard = Path(str(block[0]["resolved_prompt_shard"]))
        output_dir = output_root / mechanism
        wave_index = 0 if generated_index < 4 else 1
        gpu_slot = generated_index if wave_index == 0 else generated_index - 4
        gpu = int(gpus[gpu_slot])
        seeds = [int(row["seed"]) for row in block]
        jobs.append(
            {
                "job_id": f"wave{wave_index}_{mechanism}",
                "wave_index": wave_index,
                "gpu": gpu,
                "mechanism": mechanism,
                "row_count": ROWS_PER_MECHANISM,
                "global_indices": [int(row["global_index"]) for row in block],
                "candidate_ids": [str(row["candidate_id"]) for row in block],
                "target_prompts": [str(row["target_prompt"]) for row in block],
                "target_concepts": [str(row["target_concept"]) for row in block],
                "expected_effects": [str(row["expected_effect"]) for row in block],
                "expected_video_paths": [
                    (
                        str(_resolve_lexical(project_root, str(row["target_video_path"])))
                        if str(row.get("target_video_path", "")).strip()
                        else ""
                    )
                    for row in block
                ],
                "seeds": seeds,
                "prompt_shard": str(prompt_shard),
                "prompt_shard_sha256": sha256_file(prompt_shard),
                "output_dir": str(output_dir),
                "log_path": str(output_root / "logs" / f"{mechanism}.log"),
                "status_path": str(output_root / "statuses" / f"{mechanism}.json"),
                "environment": {
                    "CUDA_VISIBLE_DEVICES": str(gpu),
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTHONNOUSERSITE": "1",
                    "PYTHONSAFEPATH": "1",
                    "TOKENIZERS_PARALLELISM": "false",
                },
                "unset_environment": ["PYTHONHOME", "PYTHONPATH"],
                "command": build_generator_command(
                    python_executable, generator, prompt_shard, output_dir, model, seeds
                ),
            }
        )
    require(Counter(job["wave_index"] for job in jobs) == Counter({0: 4, 1: 2}), "scheduler must be a 4+2 two-wave layout")
    return {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "runner_id": RUNNER_ID,
        "created_at_utc": utc_now(),
        "dry_run": dry_run,
        "project_root": str(project_root),
        "output_root": str(output_root),
        "inputs": {
            "target_candidates": str(target_candidates),
            "target_candidates_sha256": sha256_file(target_candidates),
            "candidate_rows": EXPECTED_ROWS,
            "canonical_schema": list(CANONICAL_FIELDS),
            "schema_observation": dict(schema),
            "water_reuse_manifest": str(water_reuse_manifest),
            "water_reuse_manifest_sha256": EXPECTED_WATER_REUSE_MANIFEST_SHA256,
            "water_reuse_location_policy": (
                "legacy manifest may be supplied as an explicit absolute path to the old "
                "checkout because the default v4-relative path is not materialized"
            ),
            "water_reuse_rows": EXPECTED_REUSED_ROWS,
            "fresh_generation_rows": EXPECTED_GENERATED_ROWS,
        },
        "implementation": {
            "generator": str(generator),
            "generator_sha256": sha256_file(generator),
            "python_executable": str(python_executable),
            "media_probe_python": str(python_executable),
            "model": str(model),
        },
        "generation": {
            "baseline": "negative_prompt",
            "num_inference_steps": STEPS,
            "guidance_scale": GUIDANCE_SCALE,
            "num_frames": NUM_FRAMES,
            "fps": FPS,
            "height": HEIGHT,
            "width": WIDTH,
            "dtype": DTYPE,
            "skip_existing": False,
            "resume": False,
            "per_prompt_seeds": "explicit_from_target_candidates_csv",
        },
        "scheduler": {
            "gpus": list(gpus),
            "waves": 2,
            "jobs_per_wave": [4, 2],
            "barrier": "wave_1_starts_only_after_all_wave_0_outputs_validate",
            "failure_policy": "fail_closed_no_resume_no_skip",
        },
        "jobs": jobs,
    }


def write_bytes_exclusive(path: Path, raw: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    write_bytes_exclusive(
        temporary,
        (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
    )
    os.replace(temporary, path)


def prepare_output_root(output_root: Path, plan: Mapping[str, Any]) -> None:
    require(not output_root.exists() and not output_root.is_symlink(), f"output root must be fresh/nonexistent: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    require(output_root.parent.is_dir() and not output_root.parent.is_symlink(), "output-root parent is invalid")
    os.mkdir(output_root, 0o755)
    (output_root / "logs").mkdir()
    (output_root / "statuses").mkdir()
    for job in plan["jobs"]:
        write_json_atomic(
            Path(str(job["status_path"])),
            _status(job, "planned"),
        )
    write_bytes_exclusive(
        output_root / "target_generation_run_manifest.json",
        (json.dumps(plan, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
    )
    write_aggregate(output_root, plan, "planned")


def _expected_generation_fields() -> dict[str, Any]:
    return {
        "baseline": "negative_prompt",
        "num_inference_steps": STEPS,
        "guidance_scale": GUIDANCE_SCALE,
        "num_frames": NUM_FRAMES,
        "fps": FPS,
        "height": HEIGHT,
        "width": WIDTH,
        "dtype": DTYPE,
        "device": "cuda",
        "enable_model_cpu_offload": False,
        "enable_sequential_cpu_offload": False,
        "vae_slicing": True,
        "vae_tiling": True,
        "lora_path": None,
        "activation_gate_dir": None,
        "attention_gate_dir": None,
    }


def probe_video_media(runtime_python: Path, video_path: Path) -> dict[str, Any]:
    environment = os.environ.copy()
    environment.pop("PYTHONHOME", None)
    environment.pop("PYTHONPATH", None)
    environment.update(PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1", PYTHONSAFEPATH="1")
    result = subprocess.run(
        [str(runtime_python), "-c", PYAV_PROBE_CODE, str(video_path)],
        text=True,
        capture_output=True,
        check=False,
        env=environment,
    )
    require(result.returncode == 0, f"PyAV could not decode {video_path}: {result.stderr.strip()}")
    payload = json.loads(result.stdout)
    require(payload.get("streams") == 1, f"{video_path.name}: expected one video stream")
    require(payload.get("decoded_frames") == NUM_FRAMES, f"{video_path.name}: expected 49 decoded frames")
    require(payload.get("width") == WIDTH and payload.get("height") == HEIGHT, f"{video_path.name}: resolution mismatch")
    rate = Fraction(str(payload.get("fps")))
    require(rate == Fraction(FPS, 1), f"{video_path.name}: fps mismatch")
    return {"decoded_frames": NUM_FRAMES, "fps": "8/1", "width": WIDTH, "height": HEIGHT}


def validate_bound_inputs(plan: Mapping[str, Any], *, require_runtime: bool) -> None:
    inputs = plan["inputs"]
    implementation = plan["implementation"]
    candidates = Path(str(inputs["target_candidates"]))
    water = Path(str(inputs["water_reuse_manifest"]))
    generator = Path(str(implementation["generator"]))
    regular_file(candidates, "target candidates")
    regular_file(water, "Water reuse manifest")
    regular_file(generator, "Wan generator")
    require(sha256_file(candidates) == inputs["target_candidates_sha256"], "target candidates changed after planning")
    require(sha256_file(water) == inputs["water_reuse_manifest_sha256"], "Water reuse manifest changed after planning")
    require(sha256_file(generator) == implementation["generator_sha256"], "Wan generator changed after planning")
    for job in plan["jobs"]:
        shard = Path(str(job["prompt_shard"]))
        regular_file(shard, f"{job['mechanism']} prompt shard")
        require(sha256_file(shard) == job["prompt_shard_sha256"], f"{job['mechanism']}: prompt shard changed")
    if require_runtime:
        python = Path(str(implementation["python_executable"]))
        model = Path(str(implementation["model"]))
        regular_file(python, "Wan Python")
        require(os.access(python, os.X_OK), "Wan Python is not executable")
        require(model.is_dir() and not model.is_symlink(), "Wan model is missing or symlinked")


def validate_job_outputs(
    job: Mapping[str, Any],
    plan: Mapping[str, Any],
    *,
    media_probe: Callable[[Path, Path], dict[str, Any]] = probe_video_media,
) -> dict[str, Any]:
    output_dir = Path(str(job["output_dir"]))
    manifest = output_dir / "generation_manifest.json"
    regular_file(manifest, f"{job['mechanism']} generation manifest")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    require(payload.get("baseline") == "negative_prompt", f"{job['mechanism']}: baseline mismatch")
    require(payload.get("pipeline") == "WanPipeline", f"{job['mechanism']}: pipeline mismatch")
    require(payload.get("model") == plan["implementation"]["model"], f"{job['mechanism']}: model mismatch")
    require(payload.get("dry_run") is False, f"{job['mechanism']}: dry-run child output is forbidden")
    require(payload.get("prompts") == job["prompt_shard"], f"{job['mechanism']}: prompt shard binding mismatch")
    generation = payload.get("generation")
    require(isinstance(generation, dict), f"{job['mechanism']}: generation config missing")
    for field, expected in _expected_generation_fields().items():
        require(generation.get(field) == expected, f"{job['mechanism']}: generation.{field} mismatch")
    require(generation.get("seeds") == job["seeds"], f"{job['mechanism']}: seed-list mismatch")
    items = payload.get("items")
    require(isinstance(items, list) and len(items) == ROWS_PER_MECHANISM, f"{job['mechanism']}: expected 192 items")
    videos_dir = output_dir / "videos"
    require(videos_dir.is_dir() and not videos_dir.is_symlink(), f"{job['mechanism']}: videos directory invalid")
    outputs: list[dict[str, Any]] = []
    listed: set[Path] = set()
    for index, item in enumerate(items):
        require(isinstance(item, dict) and item.get("index") == index, f"{job['mechanism']}: item order mismatch")
        require(item.get("prompt") == job["target_prompts"][index], f"{job['mechanism']} item {index}: prompt mismatch")
        require(item.get("target_concept") == job["target_concepts"][index], f"{job['mechanism']} item {index}: exclusion phrase mismatch")
        require(item.get("negative_prompt") == job["target_concepts"][index], f"{job['mechanism']} item {index}: negative prompt mismatch")
        require(item.get("expected_effect") == job["expected_effects"][index], f"{job['mechanism']} item {index}: expected state mismatch")
        require(item.get("seed") == job["seeds"][index], f"{job['mechanism']} item {index}: seed mismatch")
        video = Path(str(item.get("video_path", "")))
        regular_file(video, f"{job['mechanism']} video {index}")
        require(video.parent == videos_dir, f"{job['mechanism']} item {index}: video escaped output directory")
        require(video.suffix.lower() == ".mp4" and video.stat().st_size > 0, f"{job['mechanism']} item {index}: invalid MP4")
        expected_path = str(job["expected_video_paths"][index]).strip()
        if expected_path:
            require(video.resolve() == Path(expected_path).resolve(), f"{job['mechanism']} item {index}: target_video_path mismatch")
        require(video not in listed, f"{job['mechanism']}: duplicate video path")
        listed.add(video)
        media = media_probe(Path(str(plan["implementation"]["media_probe_python"])), video)
        require(media == {"decoded_frames": 49, "fps": "8/1", "width": 832, "height": 480}, f"{job['mechanism']} item {index}: media contract mismatch")
        outputs.append(
            {
                "global_index": job["global_indices"][index],
                "candidate_id": job["candidate_ids"][index],
                "mechanism": job["mechanism"],
                "prompt_shard_index": index,
                "target_prompt": item["prompt"],
                "seed": item["seed"],
                "video_path": str(video),
                "video_sha256": sha256_file(video),
                "size_bytes": video.stat().st_size,
                "media": media,
            }
        )
    require(set(videos_dir.iterdir()) == listed, f"{job['mechanism']}: videos directory has extra/missing files")
    require(set(output_dir.iterdir()) == {manifest, videos_dir}, f"{job['mechanism']}: output directory has unexpected files")
    return {
        "generation_manifest_sha256": sha256_file(manifest),
        "validated_video_count": len(outputs),
        "validated_video_bytes": sum(item["size_bytes"] for item in outputs),
        "outputs": outputs,
    }


def _status(job: Mapping[str, Any], status: str, **extra: Any) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "job_id": job["job_id"],
        "mechanism": job["mechanism"],
        "wave_index": job["wave_index"],
        "gpu": job["gpu"],
        "status": status,
        "expected_videos": ROWS_PER_MECHANISM,
        "updated_at_utc": utc_now(),
    }
    payload.update(extra)
    return payload


def read_status(job: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(Path(str(job["status_path"])).read_text(encoding="utf-8"))


def write_aggregate(output_root: Path, plan: Mapping[str, Any], status: str, error: str | None = None) -> None:
    statuses = [read_status(job) for job in plan["jobs"]]
    payload: dict[str, Any] = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "runner_id": RUNNER_ID,
        "status": status,
        "updated_at_utc": utc_now(),
        "baseline": "negative_prompt",
        "water_reuse_rows": EXPECTED_REUSED_ROWS,
        "expected_generated_rows": EXPECTED_GENERATED_ROWS,
        "validated_generated_rows": sum(int(item.get("validated_video_count", 0)) for item in statuses),
        "status_counts": dict(sorted(Counter(item["status"] for item in statuses).items())),
        "run_manifest": {
            "path": str(output_root / "target_generation_run_manifest.json"),
            "sha256": sha256_file(output_root / "target_generation_run_manifest.json"),
        },
    }
    frozen = output_root / "target_generation_manifest.json"
    if frozen.is_file():
        payload["generation_manifest"] = {"path": str(frozen), "sha256": sha256_file(frozen)}
    if error is not None:
        payload["error"] = error
    write_json_atomic(output_root / "target_generation_aggregate.json", payload)


def write_generation_manifest(output_root: Path, plan: Mapping[str, Any]) -> Path:
    statuses = [read_status(job) for job in plan["jobs"]]
    require(all(status["status"] == "completed" for status in statuses), "all six jobs must complete before freeze")
    items = [item for status in statuses for item in status["outputs"]]
    items.sort(key=lambda item: int(item["global_index"]))
    expected_indices = list(range(ROWS_PER_MECHANISM, EXPECTED_ROWS))
    require([item["global_index"] for item in items] == expected_indices, "generated global_index coverage is not exact 192..1343")
    require(len(items) == EXPECTED_GENERATED_ROWS, "final generation manifest must contain 1152 items")
    require(len({item["candidate_id"] for item in items}) == EXPECTED_GENERATED_ROWS, "generated candidate IDs are not unique")
    require(len({item["video_path"] for item in items}) == EXPECTED_GENERATED_ROWS, "generated video paths are not unique")
    payload = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "runner_id": RUNNER_ID,
        "status": "frozen_after_exact_1152_video_validation",
        "target_candidates_sha256": plan["inputs"]["target_candidates_sha256"],
        "water_reuse_manifest_sha256": EXPECTED_WATER_REUSE_MANIFEST_SHA256,
        "water_reuse_rows": EXPECTED_REUSED_ROWS,
        "baseline": "negative_prompt",
        "generation": plan["generation"],
        "video_count": EXPECTED_GENERATED_ROWS,
        "items": items,
    }
    path = output_root / "target_generation_manifest.json"
    write_bytes_exclusive(path, (json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8"))
    return path


def _terminate(running: Mapping[str, tuple[Mapping[str, Any], Any, Any]]) -> None:
    for _, process, _ in running.values():
        if process.poll() is None:
            process.terminate()
    deadline = time.monotonic() + 10
    for _, process, _ in running.values():
        while process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.1)
        if process.poll() is None:
            process.kill()


def execute_plan(
    plan: Mapping[str, Any],
    output_root: Path,
    *,
    poll_interval: float,
    popen_factory: Callable[..., Any] = subprocess.Popen,
    sleep_fn: Callable[[float], None] = time.sleep,
    media_probe: Callable[[Path, Path], dict[str, Any]] = probe_video_media,
) -> None:
    require(plan["dry_run"] is False, "cannot execute a dry-run plan")
    validate_bound_inputs(plan, require_runtime=True)
    write_aggregate(output_root, plan, "running")
    for wave_index, expected_jobs in ((0, 4), (1, 2)):
        validate_bound_inputs(plan, require_runtime=True)
        jobs = [job for job in plan["jobs"] if job["wave_index"] == wave_index]
        require(len(jobs) == expected_jobs, f"wave {wave_index}: job count mismatch")
        running: dict[str, tuple[Mapping[str, Any], Any, Any]] = {}
        failures: list[str] = []
        try:
            for job in jobs:
                require(not Path(str(job["output_dir"])).exists(), f"{job['mechanism']}: output already exists")
                write_json_atomic(Path(str(job["status_path"])), _status(job, "running", started_at_utc=utc_now()))
                log = Path(str(job["log_path"])).open("xb")
                environment = os.environ.copy()
                for key in job["unset_environment"]:
                    environment.pop(str(key), None)
                environment.update({str(key): str(value) for key, value in job["environment"].items()})
                try:
                    process = popen_factory(job["command"], cwd=plan["project_root"], stdout=log, stderr=subprocess.STDOUT, env=environment)
                except BaseException:
                    log.close()
                    raise
                running[str(job["job_id"])] = (job, process, log)
            while running:
                progressed = False
                for job_id, (job, process, log) in list(running.items()):
                    return_code = process.poll()
                    if return_code is None:
                        continue
                    progressed = True
                    log.close()
                    running.pop(job_id)
                    if return_code != 0:
                        failures.append(f"{job['mechanism']}: generator exited {return_code}")
                        write_json_atomic(Path(str(job["status_path"])), _status(job, "failed", return_code=return_code))
                        continue
                    try:
                        validation = validate_job_outputs(job, plan, media_probe=media_probe)
                    except BaseException as exc:
                        failures.append(f"{job['mechanism']}: {type(exc).__name__}: {exc}")
                        write_json_atomic(Path(str(job["status_path"])), _status(job, "failed", return_code=return_code, error=str(exc)))
                        continue
                    write_json_atomic(Path(str(job["status_path"])), _status(job, "completed", return_code=0, **validation))
                write_aggregate(output_root, plan, "running")
                if running and not progressed:
                    sleep_fn(poll_interval)
        except BaseException:
            _terminate(running)
            for job, _, log in running.values():
                log.close()
                write_json_atomic(Path(str(job["status_path"])), _status(job, "failed", error="launcher interrupted"))
            write_aggregate(output_root, plan, "failed", "launcher interrupted")
            raise
        validate_bound_inputs(plan, require_runtime=True)
        if failures:
            message = "; ".join(failures)
            write_aggregate(output_root, plan, "failed", message)
            raise RuntimeError(message)
    validate_bound_inputs(plan, require_runtime=True)
    for job in plan["jobs"]:
        observed = validate_job_outputs(job, plan, media_probe=media_probe)
        status = read_status(job)
        require(observed == {key: status[key] for key in observed}, f"{job['mechanism']}: output drift after validation")
    write_generation_manifest(output_root, plan)
    write_aggregate(output_root, plan, "completed")


def resolve_path(project_root: Path, value: Path, *, must_exist: bool = True) -> Path:
    path = value if value.is_absolute() else project_root / value
    return path.resolve(strict=must_exist)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--target-candidates", type=Path, default=DEFAULT_TARGET_CANDIDATES)
    parser.add_argument("--water-reuse-manifest", type=Path, default=DEFAULT_WATER_REUSE_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--python-executable", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--gpus", type=parse_gpus, required=True)
    parser.add_argument("--poll-interval", type=float, default=5.0)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.poll_interval <= 0:
        parser.error("--poll-interval must be positive")
    raw_paths = (args.project_root, args.target_candidates, args.water_reuse_manifest, args.output_root, args.python_executable, args.model, DEFAULT_GENERATOR)
    try:
        reject_sealed_paths(*raw_paths)
        project_root = args.project_root.resolve(strict=True)
        candidates = resolve_path(project_root, args.target_candidates)
        water = resolve_path(project_root, args.water_reuse_manifest)
        output_root = resolve_path(project_root, args.output_root, must_exist=False)
        generator = resolve_path(project_root, DEFAULT_GENERATOR)
        python = resolve_path(project_root, args.python_executable, must_exist=args.run)
        model = resolve_path(project_root, args.model, must_exist=args.run)
        reject_sealed_paths(project_root, candidates, water, output_root, generator, python, model)
        require(generator == (project_root / DEFAULT_GENERATOR).resolve(strict=True), "only scripts/generate_wan_clean.py is allowed")
        rows, _, schema = load_target_candidates(project_root, candidates, water)
        if args.run:
            require(
                set(schema["canonical_columns_present"]) == set(CANONICAL_FIELDS),
                "formal --run requires every canonical target-candidate column; aliases are compatibility-only",
            )
            require(
                schema["protocol_columns_present"] == [CANONICAL_PROTOCOL_FIELD],
                "formal --run requires canonical protocol_version (protocol_id is compatibility-only)",
            )
        plan = build_plan(
            project_root=project_root,
            rows=rows,
            schema=schema,
            target_candidates=candidates,
            water_reuse_manifest=water,
            output_root=output_root,
            generator=generator,
            python_executable=python,
            model=model,
            gpus=args.gpus,
            dry_run=args.dry_run,
        )
        validate_bound_inputs(plan, require_runtime=args.run)
        prepare_output_root(output_root, plan)
        if args.run:
            execute_plan(plan, output_root, poll_interval=args.poll_interval)
        print(f"{'Planned' if args.dry_run else 'Completed'} target generation at {output_root}")
        return 0
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
