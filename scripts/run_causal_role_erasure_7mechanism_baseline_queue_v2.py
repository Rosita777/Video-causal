#!/usr/bin/env python3
"""Plan and execute the 35-job seven-mechanism CogVideoX baseline queue.

The queue launches one registered formal runner per stream/mechanism pair,
exposes only one physical GPU to each child, and advances in deterministic
waves.  Planning is fresh-only.  Resume skips only jobs whose complete child
plans, manifests, receipts, video hashes, and media have been revalidated;
partial, running, or failed jobs are never retried in place.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_causal_role_erasure_7mechanism_baseline_registry_v2 as baseline_registry


PROTOCOL = "causal_role_erasure_7mechanism_baseline_queue_v2"
RUNNER_RELATIVE = "scripts/run_causal_role_erasure_7mechanism_baseline_queue_v2.py"
BUILDER_RELATIVE = "scripts/build_causal_role_erasure_7mechanism_baseline_registry_v2.py"
STREAMS = tuple(baseline_registry.STREAMS)
MECHANISMS = tuple(baseline_registry.MECHANISMS)
CASES_PER_JOB = 42
MEDIA = {"decoded_frames": 49, "fps": "8/1", "height": 480, "width": 720}
CONTROL_STREAMS = ("cogvideox_original", "negative_prompt")
RUNNER_BY_STREAM = {
    "cogvideox_original": "scripts/run_causal_role_erasure_7mechanism_cogvideox_controls_v2.py",
    "negative_prompt": "scripts/run_causal_role_erasure_7mechanism_cogvideox_controls_v2.py",
    "videoeraser_official": "scripts/run_causal_role_erasure_7mechanism_videoeraser_official_v2.py",
    "t2vunlearning_adapted": "scripts/run_causal_role_erasure_7mechanism_t2v_adapted_v2.py",
    "safree_cogvideox": "scripts/run_causal_role_erasure_7mechanism_safree_cogvideox_v2.py",
}
DEFAULT_REGISTRY = Path(
    "outputs/causal_role_erasure_7mechanism_main_v2/baseline_registry_v2/baseline_registry.json"
)
DEFAULT_OUTPUT_ROOT = Path(
    "outputs/causal_role_erasure_7mechanism_main_v2/formal_baselines_v2"
)
PYAV_PROBE_CODE = r"""
import av, json, sys
from fractions import Fraction
with av.open(sys.argv[1]) as container:
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


class QueueError(ValueError):
    """The baseline queue or one of its frozen artifacts is invalid."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise QueueError(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def regular_file(path: Path, label: str) -> None:
    require(path.is_file() and not path.is_symlink(), f"{label} missing or symlinked: {path}")


def real_directory(path: Path, label: str) -> None:
    require(path.is_dir() and not path.is_symlink(), f"{label} missing or symlinked: {path}")


def resolve_registered(project_root: Path, value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else project_root / path).resolve()


def parse_gpus(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--gpus must be comma-separated non-negative integers") from exc
    if not values or len(set(values)) != len(values) or any(item < 0 for item in values):
        raise argparse.ArgumentTypeError("--gpus must contain distinct non-negative GPU indices")
    return values


def parse_streams(value: str) -> tuple[str, ...]:
    if value.strip().casefold() == "all":
        return STREAMS
    requested = tuple(part.strip() for part in value.split(",") if part.strip())
    invalid = sorted(set(requested) - set(STREAMS))
    if not requested or invalid or len(set(requested)) != len(requested):
        suffix = f"; invalid={invalid}" if invalid else ""
        raise argparse.ArgumentTypeError(f"--streams must contain distinct registered streams{suffix}")
    return tuple(stream for stream in STREAMS if stream in requested)


def load_registry(
    project_root: Path,
    registry_path: Path,
    receipt_path: Path,
    selected_streams: Sequence[str],
    runtime_probe: Callable[[Path], Mapping[str, Any]] = baseline_registry.probe_runtime,
) -> dict[str, Any]:
    regular_file(registry_path, "baseline registry")
    regular_file(receipt_path, "baseline registry receipt")
    raw = registry_path.read_bytes()
    registry = json.loads(raw)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    require(registry.get("protocol") == baseline_registry.REGISTRY_PROTOCOL, "unexpected baseline registry protocol")
    require(registry.get("protocol_version") == baseline_registry.PROTOCOL_VERSION, "baseline protocol changed")
    require(registry.get("formal_generation_authorized") is True, "baseline registry does not authorize formal generation")
    require(registry.get("status") == "baseline_stack_frozen_ready", "baseline stack is not frozen ready")
    require(receipt.get("protocol") == baseline_registry.REGISTRY_PROTOCOL, "unexpected baseline receipt protocol")
    require(receipt.get("registry_sha256") == hashlib.sha256(raw).hexdigest(), "baseline registry receipt mismatch")
    require(tuple(registry.get("streams", ())) == STREAMS, "baseline stream inventory changed")
    require(tuple(selected_streams) == tuple(stream for stream in STREAMS if stream in selected_streams), "selected stream order changed")
    require(registry.get("formal_cases", {}).get("row_count") == 294, "formal case count changed")
    formal = resolve_registered(project_root, str(registry["formal_cases"]["path"]))
    regular_file(formal, "formal cases")
    require(sha256_file(formal) == registry["formal_cases"]["sha256"], "formal cases changed after baseline freeze")

    implementations = registry.get("implementations", {})
    code = registry.get("code_sha256", {})
    builder = project_root / BUILDER_RELATIVE
    regular_file(builder, "baseline registry builder")
    require(BUILDER_RELATIVE in code, "baseline registry builder is not self-bound")
    require(sha256_file(builder) == code[BUILDER_RELATIVE], "baseline registry builder changed after freeze")

    runtime = resolve_registered(project_root, str(registry["runtime"]["python_executable"]))
    regular_file(runtime, "registered runtime Python")
    require(sha256_file(runtime) == registry["runtime"]["python_executable_sha256"], "runtime Python changed")
    require(os.access(runtime, os.X_OK), "registered runtime Python is not executable")
    observed_runtime = dict(runtime_probe(runtime))
    for field in ("python", "implementation", "packages"):
        require(
            observed_runtime.get(field) == registry["runtime"].get(field),
            f"registered runtime {field} changed",
        )
    observed_executable = Path(str(observed_runtime.get("executable", ""))).resolve()
    require(observed_executable == runtime, "runtime probe executed a different Python")

    for stream in selected_streams:
        require(implementations.get(stream, {}).get("status") == "ready", f"selected stream is not ready: {stream}")
        runner_relative = RUNNER_BY_STREAM[stream]
        runner = project_root / runner_relative
        regular_file(runner, f"{stream} runner")
        require(runner_relative in code, f"{stream} runner is not baseline-registry bound")
        require(sha256_file(runner) == code[runner_relative], f"{stream} runner changed after baseline freeze")
    queue_runner = project_root / RUNNER_RELATIVE
    regular_file(queue_runner, "baseline queue runner")
    require(RUNNER_RELATIVE in code, "baseline queue runner is not baseline-registry bound")
    require(sha256_file(queue_runner) == code[RUNNER_RELATIVE], "baseline queue runner changed after baseline freeze")
    return registry


def read_master_formal_cases(path: Path) -> dict[str, dict[str, Any]]:
    regular_file(path, "master formal cases")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == 294, "master formal case count must be 294")
    require(all(field in (rows[0] if rows else {}) for field in ("case_id", "mechanism", "seed")), "master formal case fields are incomplete")
    master: dict[str, dict[str, Any]] = {}
    seeds: set[int] = set()
    for row in rows:
        case_id = str(row["case_id"])
        mechanism = str(row["mechanism"])
        require(case_id and case_id not in master, f"master formal case ID repeats: {case_id}")
        require(mechanism in MECHANISMS, f"master formal mechanism is invalid: {mechanism}")
        try:
            seed = int(row["seed"])
        except (TypeError, ValueError) as exc:
            raise QueueError(f"master formal seed is invalid for {case_id}") from exc
        require(seed > 0 and seed not in seeds, f"master formal seed repeats or is nonpositive: {case_id}")
        seeds.add(seed)
        master[case_id] = {"mechanism": mechanism, "seed": seed}
    require(
        Counter(item["mechanism"] for item in master.values())
        == {mechanism: CASES_PER_JOB for mechanism in MECHANISMS},
        "master formal mechanism balance changed",
    )
    return master


def _child_command(
    runtime: Path,
    runner: Path,
    project_root: Path,
    registry_path: Path,
    receipt_path: Path,
    stream: str,
    mechanism: str,
    output_dir: Path,
) -> list[str]:
    command = [
        str(runtime),
        str(runner),
        "--project-root",
        str(project_root),
        "--baseline-registry",
        str(registry_path),
        "--baseline-receipt",
        str(receipt_path),
    ]
    if stream in CONTROL_STREAMS:
        command.extend(("--baseline", stream))
    command.extend(("--mechanism", mechanism, "--output-dir", str(output_dir)))
    return command


def _assign_waves(
    jobs: list[dict[str, Any]],
    gpus: Sequence[int],
    safree_max_concurrency: int,
) -> list[list[dict[str, Any]]]:
    remaining = list(jobs)
    waves: list[list[dict[str, Any]]] = []
    while remaining:
        chosen: list[dict[str, Any]] = []
        safree_count = 0
        for job in remaining:
            if len(chosen) == len(gpus):
                break
            if job["stream"] == "safree_cogvideox" and safree_count >= safree_max_concurrency:
                continue
            chosen.append(job)
            safree_count += int(job["stream"] == "safree_cogvideox")
        require(bool(chosen), "SAFREE concurrency limit made queue scheduling impossible")
        selected_ids = {id(job) for job in chosen}
        remaining = [job for job in remaining if id(job) not in selected_ids]
        wave_index = len(waves)
        for slot, job in enumerate(chosen):
            job["wave_index"] = wave_index
            job["gpu"] = int(gpus[slot])
        waves.append(chosen)
    return waves


def build_plan(
    *,
    project_root: Path,
    registry_path: Path,
    receipt_path: Path,
    registry: Mapping[str, Any],
    output_root: Path,
    selected_streams: Sequence[str],
    gpus: Sequence[int],
    safree_max_concurrency: int,
) -> dict[str, Any]:
    require(bool(gpus) and len(set(gpus)) == len(gpus), "GPU inventory must be nonempty and unique")
    require(1 <= safree_max_concurrency <= len(gpus), "SAFREE concurrency must be within 1..number of GPUs")
    selected_streams = tuple(stream for stream in STREAMS if stream in selected_streams)
    require(bool(selected_streams), "at least one stream must be selected")
    runtime = resolve_registered(project_root, str(registry["runtime"]["python_executable"]))
    formal_path = resolve_registered(project_root, str(registry["formal_cases"]["path"]))
    master_cases = read_master_formal_cases(formal_path)
    require(len(master_cases) == 294, "master formal inventory changed during planning")
    draft: list[dict[str, Any]] = []
    for mechanism in MECHANISMS:
        for stream in selected_streams:
            draft.append({"stream": stream, "mechanism": mechanism})
    waves = _assign_waves(draft, gpus, safree_max_concurrency)
    jobs: list[dict[str, Any]] = []
    for job_index, draft_job in enumerate(draft):
        stream = str(draft_job["stream"])
        mechanism = str(draft_job["mechanism"])
        runner_relative = RUNNER_BY_STREAM[stream]
        runner = project_root / runner_relative
        job_id = f"{stream}__{mechanism}"
        output_dir = output_root / "streams" / stream / mechanism
        job = {
            "schema_version": 1,
            "protocol": PROTOCOL,
            "protocol_version": baseline_registry.PROTOCOL_VERSION,
            "job_index": job_index,
            "job_id": job_id,
            "wave_index": int(draft_job["wave_index"]),
            "gpu": int(draft_job["gpu"]),
            "stream": stream,
            "mechanism": mechanism,
            "expected_videos": CASES_PER_JOB,
            "runner": str(runner),
            "runner_sha256": sha256_file(runner),
            "output_dir": str(output_dir),
            "log_path": str(output_root / "logs" / f"{job_index:02d}_{job_id}.log"),
            "status_path": str(output_root / "statuses" / f"{job_index:02d}_{job_id}.json"),
            "job_manifest_path": str(output_root / "jobs" / f"{job_index:02d}_{job_id}.json"),
            "environment": {
                "CUDA_VISIBLE_DEVICES": str(draft_job["gpu"]),
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONNOUSERSITE": "1",
                "PYTHONSAFEPATH": "1",
                "TOKENIZERS_PARALLELISM": "false",
            },
            "unset_environment": ["PYTHONHOME", "PYTHONPATH"],
        }
        job["command"] = _child_command(
            runtime,
            runner,
            project_root,
            registry_path,
            receipt_path,
            stream,
            mechanism,
            output_dir,
        )
        jobs.append(job)
    require(len({job["job_id"] for job in jobs}) == len(jobs), "baseline job IDs repeat")
    require(len(jobs) == 7 * len(selected_streams), "baseline job count changed")
    for wave_index, wave in enumerate(waves):
        realized = [job for job in jobs if job["wave_index"] == wave_index]
        require(len(realized) == len(wave) <= len(gpus), "wave size exceeds GPU inventory")
        require(len({job["gpu"] for job in realized}) == len(realized), "a wave reuses one GPU")
        require(
            sum(job["stream"] == "safree_cogvideox" for job in realized) <= safree_max_concurrency,
            "wave exceeds SAFREE concurrency limit",
        )
    queue_runner = project_root / RUNNER_RELATIVE
    return {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "protocol_version": baseline_registry.PROTOCOL_VERSION,
        "created_at_utc": utc_now(),
        "project_root": str(project_root),
        "output_root": str(output_root),
        "inputs": {
            "baseline_registry": str(registry_path),
            "baseline_registry_sha256": sha256_file(registry_path),
            "baseline_receipt": str(receipt_path),
            "baseline_receipt_sha256": sha256_file(receipt_path),
            "formal_cases": str(formal_path),
            "formal_cases_sha256": registry["formal_cases"]["sha256"],
        },
        "implementation": {
            "queue_runner": str(queue_runner),
            "queue_runner_sha256": sha256_file(queue_runner),
            "runtime_python": str(runtime),
            "runtime_python_sha256": registry["runtime"]["python_executable_sha256"],
        },
        "selected_streams": list(selected_streams),
        "mechanisms": list(MECHANISMS),
        "scheduler": {
            "gpus": list(gpus),
            "wave_count": len(waves),
            "jobs_per_wave": [len(wave) for wave in waves],
            "safree_max_concurrency": safree_max_concurrency,
            "resume_policy": "skip_only_fully_revalidated_completed_jobs_refuse_all_partials",
        },
        "job_count": len(jobs),
        "expected_videos": len(jobs) * CASES_PER_JOB,
        "jobs": jobs,
    }


def write_exclusive(path: Path, raw: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    write_exclusive(temporary, canonical_json_bytes(payload))
    os.replace(temporary, path)


def status_payload(job: Mapping[str, Any], status: str, **extra: Any) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "job_id": job["job_id"],
        "job_index": job["job_index"],
        "wave_index": job["wave_index"],
        "gpu": job["gpu"],
        "stream": job["stream"],
        "mechanism": job["mechanism"],
        "status": status,
        "expected_videos": CASES_PER_JOB,
        "updated_at_utc": utc_now(),
    }
    payload.update(extra)
    return payload


def prepare_output_root(output_root: Path, plan: Mapping[str, Any]) -> None:
    require(not output_root.exists() and not output_root.is_symlink(), f"output root must be fresh: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    require(not output_root.parent.is_symlink(), "output-root parent is symlinked")
    os.mkdir(output_root)
    for name in ("jobs", "statuses", "logs", "streams"):
        (output_root / name).mkdir()
    for job in plan["jobs"]:
        write_exclusive(Path(job["job_manifest_path"]), canonical_json_bytes(job))
        write_json_atomic(Path(job["status_path"]), status_payload(job, "planned"))
    write_exclusive(output_root / "queue_plan.json", canonical_json_bytes(plan))
    write_aggregate(output_root, plan, "planned")


def validate_bound_inputs(plan: Mapping[str, Any]) -> None:
    queue_plan_path = Path(str(plan["output_root"])) / "queue_plan.json"
    regular_file(queue_plan_path, "queue plan")
    require(queue_plan_path.read_bytes() == canonical_json_bytes(plan), "queue plan bytes changed after planning")
    for name in ("baseline_registry", "baseline_receipt", "formal_cases"):
        path = Path(plan["inputs"][name])
        regular_file(path, name)
        require(sha256_file(path) == plan["inputs"][f"{name}_sha256"], f"{name} changed after planning")
    queue_runner = Path(plan["implementation"]["queue_runner"])
    regular_file(queue_runner, "queue runner")
    require(sha256_file(queue_runner) == plan["implementation"]["queue_runner_sha256"], "queue runner changed after planning")
    runtime = Path(plan["implementation"]["runtime_python"])
    regular_file(runtime, "runtime Python")
    require(sha256_file(runtime) == plan["implementation"]["runtime_python_sha256"], "runtime Python changed after planning")
    require(os.access(runtime, os.X_OK), "runtime Python is not executable")
    for job in plan["jobs"]:
        runner = Path(job["runner"])
        manifest = Path(job["job_manifest_path"])
        regular_file(runner, f"{job['job_id']} runner")
        regular_file(manifest, f"{job['job_id']} job manifest")
        require(sha256_file(runner) == job["runner_sha256"], f"{job['job_id']}: runner changed")
        require(manifest.read_bytes() == canonical_json_bytes(job), f"{job['job_id']}: job manifest changed")


def probe_video(runtime_python: Path, video: Path) -> dict[str, Any]:
    environment = os.environ.copy()
    environment.pop("PYTHONHOME", None)
    environment.pop("PYTHONPATH", None)
    environment.update(PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1", PYTHONSAFEPATH="1")
    result = subprocess.run(
        [str(runtime_python), "-I", "-c", PYAV_PROBE_CODE, str(video)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    require(result.returncode == 0, f"PyAV could not validate {video}: {result.stderr.strip()}")
    media = json.loads(result.stdout)
    if media.get("fps"):
        rate = Fraction(str(media["fps"]))
        media["fps"] = f"{rate.numerator}/{rate.denominator}"
    return media


def validate_job_outputs(
    job: Mapping[str, Any],
    plan: Mapping[str, Any],
    *,
    media_probe: Callable[[Path, Path], Mapping[str, Any]] = probe_video,
) -> dict[str, Any]:
    output = Path(job["output_dir"])
    real_directory(output, f"{job['job_id']} output")
    child_plan_path = output / "generation_plan.json"
    manifest_path = output / "generation_manifest.json"
    complete_path = output / ".complete"
    videos_dir = output / "videos"
    receipts_dir = output / "receipts"
    for path, label in (
        (child_plan_path, "child generation plan"),
        (manifest_path, "child generation manifest"),
        (complete_path, "child completion marker"),
    ):
        regular_file(path, f"{job['job_id']} {label}")
    real_directory(videos_dir, f"{job['job_id']} videos")
    real_directory(receipts_dir, f"{job['job_id']} receipts")
    require(
        set(output.iterdir()) == {child_plan_path, manifest_path, complete_path, videos_dir, receipts_dir},
        f"{job['job_id']}: output-root inventory mismatch",
    )
    require(complete_path.read_text(encoding="utf-8") == sha256_file(manifest_path) + "\n", f"{job['job_id']}: completion marker mismatch")
    child_plan = json.loads(child_plan_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(child_plan.get("baseline") == job["stream"], f"{job['job_id']}: child baseline mismatch")
    require(child_plan.get("mechanism") == job["mechanism"], f"{job['job_id']}: child mechanism mismatch")
    require(child_plan.get("baseline_registry_sha256") == plan["inputs"]["baseline_registry_sha256"], f"{job['job_id']}: child registry binding mismatch")
    child_items = child_plan.get("items")
    outputs = manifest.get("outputs")
    require(isinstance(child_items, list) and len(child_items) == CASES_PER_JOB, f"{job['job_id']}: child plan must contain 42 cases")
    require(isinstance(outputs, list) and len(outputs) == CASES_PER_JOB, f"{job['job_id']}: child manifest must contain 42 outputs")
    require(manifest.get("status") == "complete", f"{job['job_id']}: child manifest is incomplete")
    require(set(manifest) == set(child_plan) | {"status", "outputs"}, f"{job['job_id']}: child manifest fields changed")
    for key, value in child_plan.items():
        require(manifest.get(key) == value, f"{job['job_id']}: child plan/manifest mismatch for {key}")

    master = read_master_formal_cases(Path(plan["inputs"]["formal_cases"]))
    expected_cases = {
        case_id
        for case_id, binding in master.items()
        if binding["mechanism"] == job["mechanism"]
    }
    require(
        {str(item.get("case_id", "")) for item in child_items} == expected_cases,
        f"{job['job_id']}: child cases do not exactly cover the 42 master mechanism cases",
    )
    expected_videos = {Path(str(item["video_path"])) for item in child_items}
    expected_receipts = {Path(str(item["receipt_path"])) for item in child_items}
    require(len(expected_videos) == CASES_PER_JOB and len(expected_receipts) == CASES_PER_JOB, f"{job['job_id']}: duplicate child output paths")
    require(set(videos_dir.iterdir()) == expected_videos, f"{job['job_id']}: video inventory mismatch")
    require(set(receipts_dir.iterdir()) == expected_receipts, f"{job['job_id']}: receipt inventory mismatch")
    runtime = Path(plan["implementation"]["runtime_python"])
    summaries: list[dict[str, Any]] = []
    seen_cases: set[str] = set()
    for index, (item, output_receipt) in enumerate(zip(child_items, outputs)):
        case_id = str(item.get("case_id", ""))
        require(case_id and case_id not in seen_cases, f"{job['job_id']}: duplicate/empty case ID")
        seen_cases.add(case_id)
        require(item.get("mechanism") == job["mechanism"], f"{job['job_id']} item {index}: mechanism mismatch")
        require(item.get("seed") == master[case_id]["seed"], f"{job['job_id']} item {index}: seed differs from master formal case")
        video = Path(str(item["video_path"]))
        receipt_path = Path(str(item["receipt_path"]))
        regular_file(video, f"{job['job_id']} video {index}")
        regular_file(receipt_path, f"{job['job_id']} receipt {index}")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        require(receipt == output_receipt, f"{job['job_id']} item {index}: receipt/manifest mismatch")
        require(receipt.get("case_id") == case_id, f"{job['job_id']} item {index}: receipt case mismatch")
        require(receipt.get("baseline") == job["stream"], f"{job['job_id']} item {index}: receipt baseline mismatch")
        require(receipt.get("seed") == master[case_id]["seed"], f"{job['job_id']} item {index}: receipt seed differs from master formal case")
        require(receipt.get("video_path") == str(video), f"{job['job_id']} item {index}: receipt path mismatch")
        video_sha = sha256_file(video)
        require(receipt.get("video_sha256") == video_sha, f"{job['job_id']} item {index}: video hash mismatch")
        media = dict(media_probe(runtime, video))
        require(media == MEDIA, f"{job['job_id']} item {index}: media contract mismatch")
        require(receipt.get("media") == media, f"{job['job_id']} item {index}: media receipt mismatch")
        summaries.append(
            {
                "job_id": job["job_id"],
                "stream": job["stream"],
                "mechanism": job["mechanism"],
                "case_id": case_id,
                "seed": item["seed"],
                "video_path": str(video),
                "video_sha256": video_sha,
                "size_bytes": video.stat().st_size,
                "media": media,
            }
        )
    return {
        "generation_plan_sha256": sha256_file(child_plan_path),
        "generation_manifest_sha256": sha256_file(manifest_path),
        "validated_video_count": len(summaries),
        "validated_video_bytes": sum(item["size_bytes"] for item in summaries),
        "output_inventory_sha256": hashlib.sha256(canonical_json_bytes(summaries)).hexdigest(),
        "outputs": summaries,
    }


def read_status(job: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(job["status_path"])
    regular_file(path, f"{job['job_id']} status")
    return json.loads(path.read_text(encoding="utf-8"))


def write_aggregate(
    output_root: Path,
    plan: Mapping[str, Any],
    state: str,
    error: str | None = None,
) -> None:
    statuses = [read_status(job) for job in plan["jobs"]]
    payload: dict[str, Any] = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "status": state,
        "updated_at_utc": utc_now(),
        "selected_streams": plan["selected_streams"],
        "expected_jobs": plan["job_count"],
        "expected_videos": plan["expected_videos"],
        "validated_videos": sum(int(status.get("validated_video_count", 0)) for status in statuses),
        "status_counts": dict(sorted(Counter(status["status"] for status in statuses).items())),
        "queue_plan": {
            "path": str(output_root / "queue_plan.json"),
            "sha256": sha256_file(output_root / "queue_plan.json"),
        },
    }
    final = output_root / "baseline_generation_manifest.json"
    if final.is_file():
        payload["generation_manifest"] = {"path": str(final), "sha256": sha256_file(final)}
    if error:
        payload["error"] = error
    write_json_atomic(output_root / "aggregate.json", payload)


def load_existing_plan(output_root: Path, expected: Mapping[str, Any]) -> dict[str, Any]:
    path = output_root / "queue_plan.json"
    regular_file(path, "existing baseline queue plan")
    actual = json.loads(path.read_text(encoding="utf-8"))
    comparison = dict(expected)
    comparison["created_at_utc"] = actual.get("created_at_utc")
    require(actual == comparison, "existing baseline queue plan differs from current frozen plan")
    require(path.read_bytes() == canonical_json_bytes(actual), "existing baseline queue plan is not canonical")
    return actual


def validate_queue_inventory(output_root: Path, plan: Mapping[str, Any]) -> None:
    jobs_dir = output_root / "jobs"
    statuses_dir = output_root / "statuses"
    logs_dir = output_root / "logs"
    streams_dir = output_root / "streams"
    for path, label in (
        (jobs_dir, "queue jobs directory"),
        (statuses_dir, "queue statuses directory"),
        (logs_dir, "queue logs directory"),
        (streams_dir, "queue streams directory"),
    ):
        real_directory(path, label)
    expected_jobs = {Path(job["job_manifest_path"]) for job in plan["jobs"]}
    expected_statuses = {Path(job["status_path"]) for job in plan["jobs"]}
    expected_logs = {Path(job["log_path"]) for job in plan["jobs"]}
    require(set(jobs_dir.iterdir()) == expected_jobs, "queue job-manifest inventory mismatch")
    require(set(statuses_dir.iterdir()) == expected_statuses, "queue status inventory mismatch")
    require(set(logs_dir.iterdir()) <= expected_logs, "queue contains an unplanned log")
    stream_entries = set(streams_dir.iterdir())
    allowed_stream_entries = {streams_dir / stream for stream in plan["selected_streams"]}
    require(stream_entries <= allowed_stream_entries, "queue contains an unplanned stream output")
    require(all(path.is_dir() and not path.is_symlink() for path in stream_entries), "queue stream output is not a real directory")

    queue_plan = output_root / "queue_plan.json"
    aggregate = output_root / "aggregate.json"
    final = output_root / "baseline_generation_manifest.json"
    complete = output_root / ".complete"
    regular_file(queue_plan, "queue plan")
    regular_file(aggregate, "queue aggregate")
    require(final.exists() == complete.exists(), "final baseline manifest and marker must both exist or both be absent")
    expected_root = {jobs_dir, statuses_dir, logs_dir, streams_dir, queue_plan, aggregate}
    if final.exists():
        regular_file(final, "final baseline manifest")
        regular_file(complete, "baseline queue completion marker")
        require(complete.read_text(encoding="utf-8") == sha256_file(final) + "\n", "baseline queue completion marker mismatch")
        expected_root.update((final, complete))
    require(set(output_root.iterdir()) == expected_root, "baseline queue root inventory mismatch")


def preflight_resume(
    plan: Mapping[str, Any],
    *,
    media_probe: Callable[[Path, Path], Mapping[str, Any]] = probe_video,
) -> set[str]:
    output_root = Path(str(plan["output_root"]))
    validate_queue_inventory(output_root, plan)
    completed: set[str] = set()
    for job in plan["jobs"]:
        status = read_status(job)
        require(status.get("job_id") == job["job_id"] and status.get("job_index") == job["job_index"], f"{job['job_id']}: status identity mismatch")
        expected_static = status_payload(job, str(status.get("status")))
        expected_static.pop("updated_at_utc")
        require(
            all(status.get(key) == value for key, value in expected_static.items()),
            f"{job['job_id']}: status contract changed",
        )
        output = Path(job["output_dir"])
        log = Path(job["log_path"])
        if status.get("status") == "completed":
            regular_file(log, f"{job['job_id']} completed log")
            observed = validate_job_outputs(job, plan, media_probe=media_probe)
            require(observed == {key: status.get(key) for key in observed}, f"{job['job_id']}: completed status/output drift")
            completed.add(str(job["job_id"]))
        else:
            require(status.get("status") == "planned", f"{job['job_id']}: refusing non-completed status {status.get('status')}")
            require(not output.exists() and not output.is_symlink(), f"{job['job_id']}: refusing partial output")
            require(not log.exists() and not log.is_symlink(), f"{job['job_id']}: refusing previously launched log")
    final_exists = (output_root / "baseline_generation_manifest.json").exists()
    require(not final_exists or len(completed) == len(plan["jobs"]), "final baseline manifest exists before all jobs completed")
    return completed


def terminate(running: Mapping[str, tuple[Mapping[str, Any], Any, Any]]) -> None:
    for _job, process, _log in running.values():
        if process.poll() is None:
            process.terminate()
    deadline = time.monotonic() + 10
    for _job, process, _log in running.values():
        while process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.1)
        if process.poll() is None:
            process.kill()


def freeze_final(
    output_root: Path,
    plan: Mapping[str, Any],
    *,
    media_probe: Callable[[Path, Path], Mapping[str, Any]] = probe_video,
) -> None:
    items: list[dict[str, Any]] = []
    for job in plan["jobs"]:
        status = read_status(job)
        require(status.get("status") == "completed", f"{job['job_id']}: not completed")
        observed = validate_job_outputs(job, plan, media_probe=media_probe)
        require(observed == {key: status.get(key) for key in observed}, f"{job['job_id']}: final validation drift")
        items.extend(observed["outputs"])
    require(len(items) == plan["expected_videos"], "final baseline video count changed")
    require(len({(item["stream"], item["case_id"]) for item in items}) == len(items), "final stream/case inventory repeats")
    master = read_master_formal_cases(Path(plan["inputs"]["formal_cases"]))
    master_case_ids = set(master)
    by_stream: dict[str, list[dict[str, Any]]] = {
        stream: [item for item in items if item["stream"] == stream]
        for stream in plan["selected_streams"]
    }
    require(set(item["stream"] for item in items) == set(plan["selected_streams"]), "final stream inventory changed")
    for stream, stream_items in by_stream.items():
        require(len(stream_items) == 294, f"{stream}: final stream must contain exactly 294 videos")
        require({item["case_id"] for item in stream_items} == master_case_ids, f"{stream}: final stream does not exactly cover master formal cases")
        for item in stream_items:
            binding = master[item["case_id"]]
            require(item["seed"] == binding["seed"], f"{stream}/{item['case_id']}: final seed differs from master")
            require(item["mechanism"] == binding["mechanism"], f"{stream}/{item['case_id']}: final mechanism differs from master")
    payload = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "protocol_version": plan["protocol_version"],
        "status": "frozen_after_full_media_validation",
        "selected_streams": plan["selected_streams"],
        "video_count": len(items),
        "inputs": plan["inputs"],
        "scheduler": plan["scheduler"],
        "items": items,
    }
    final = output_root / "baseline_generation_manifest.json"
    raw = canonical_json_bytes(payload)
    if final.exists():
        regular_file(final, "existing final baseline manifest")
        require(final.read_bytes() == raw, "existing final baseline manifest differs from revalidation")
    else:
        write_exclusive(final, raw)
    complete = output_root / ".complete"
    marker = sha256_file(final) + "\n"
    if complete.exists():
        regular_file(complete, "baseline queue completion marker")
        require(complete.read_text(encoding="utf-8") == marker, "baseline queue completion marker changed")
    else:
        write_exclusive(complete, marker.encode("utf-8"))


def execute_plan(
    plan: Mapping[str, Any],
    output_root: Path,
    *,
    poll_interval: float,
    popen_factory: Callable[..., Any] = subprocess.Popen,
    sleep_fn: Callable[[float], None] = time.sleep,
    media_probe: Callable[[Path, Path], Mapping[str, Any]] = probe_video,
) -> None:
    validate_bound_inputs(plan)
    completed = preflight_resume(plan, media_probe=media_probe)
    write_aggregate(output_root, plan, "running")
    for wave_index in range(int(plan["scheduler"]["wave_count"])):
        jobs = [
            job
            for job in plan["jobs"]
            if job["wave_index"] == wave_index and job["job_id"] not in completed
        ]
        if not jobs:
            continue
        running: dict[str, tuple[Mapping[str, Any], Any, Any]] = {}
        failures: list[str] = []
        try:
            for job in jobs:
                validate_bound_inputs(plan)
                write_json_atomic(Path(job["status_path"]), status_payload(job, "running", started_at_utc=utc_now()))
                log = Path(job["log_path"]).open("xb")
                environment = os.environ.copy()
                for key in job["unset_environment"]:
                    environment.pop(key, None)
                environment.update(job["environment"])
                process = popen_factory(
                    job["command"],
                    cwd=plan["project_root"],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=environment,
                )
                running[str(job["job_id"])] = (job, process, log)
            while running:
                progressed = False
                for job_id, (job, process, log) in list(running.items()):
                    code = process.poll()
                    if code is None:
                        continue
                    progressed = True
                    log.close()
                    running.pop(job_id)
                    if code != 0:
                        failures.append(f"{job_id}: formal runner exited {code}")
                        write_json_atomic(Path(job["status_path"]), status_payload(job, "failed", return_code=code))
                        continue
                    try:
                        observed = validate_job_outputs(job, plan, media_probe=media_probe)
                    except BaseException as exc:
                        failures.append(f"{job_id}: {type(exc).__name__}: {exc}")
                        write_json_atomic(Path(job["status_path"]), status_payload(job, "failed", return_code=0, error=str(exc)))
                        continue
                    write_json_atomic(Path(job["status_path"]), status_payload(job, "completed", return_code=0, **observed))
                    completed.add(job_id)
                write_aggregate(output_root, plan, "running")
                if running and not progressed:
                    sleep_fn(poll_interval)
        except BaseException:
            terminate(running)
            for job, _process, log in running.values():
                log.close()
                write_json_atomic(Path(job["status_path"]), status_payload(job, "failed", error="launcher interrupted"))
            write_aggregate(output_root, plan, "failed", "launcher interrupted")
            raise
        if failures:
            message = "; ".join(failures)
            write_aggregate(output_root, plan, "failed", message)
            raise RuntimeError(message)
    validate_bound_inputs(plan)
    freeze_final(output_root, plan, media_probe=media_probe)
    write_aggregate(output_root, plan, "completed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--baseline-registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--baseline-receipt", type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--streams", type=parse_streams, default=STREAMS)
    parser.add_argument("--gpus", type=parse_gpus, default=(0, 1, 2, 3))
    parser.add_argument("--safree-max-concurrency", type=int)
    parser.add_argument("--poll-interval", type=float, default=5.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _under(project_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else project_root / path


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        require(args.poll_interval > 0, "poll interval must be positive")
        project_root = args.project_root.resolve(strict=True)
        registry_path = _under(project_root, args.baseline_registry).resolve(strict=True)
        receipt_path = args.baseline_receipt or registry_path.with_name("build_receipt.json")
        receipt_path = _under(project_root, receipt_path).resolve(strict=True)
        output_root = _under(project_root, args.output_root).resolve(strict=False)
        gpus = tuple(args.gpus)
        selected_streams = tuple(args.streams)
        safree_limit = (
            len(gpus)
            if args.safree_max_concurrency is None
            else args.safree_max_concurrency
        )
        registry = load_registry(project_root, registry_path, receipt_path, selected_streams)
        plan = build_plan(
            project_root=project_root,
            registry_path=registry_path,
            receipt_path=receipt_path,
            registry=registry,
            output_root=output_root,
            selected_streams=selected_streams,
            gpus=gpus,
            safree_max_concurrency=safree_limit,
        )
        if output_root.exists():
            require(args.resume, f"fresh-only output root already exists; use --resume: {output_root}")
            plan = load_existing_plan(output_root, plan)
        else:
            require(not args.resume, f"--resume requires an existing output root: {output_root}")
            prepare_output_root(output_root, plan)
        if args.dry_run:
            validate_bound_inputs(plan)
            preflight_resume(plan)
            print(
                f"Planned {plan['job_count']} jobs / {plan['expected_videos']} videos "
                f"in {plan['scheduler']['wave_count']} waves at {output_root}"
            )
            return 0
        execute_plan(plan, output_root, poll_interval=args.poll_interval)
        print(f"Completed and froze {plan['expected_videos']} baseline videos at {output_root}")
        return 0
    except (OSError, QueueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
