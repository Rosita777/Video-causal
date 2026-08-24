#!/usr/bin/env python3
"""Fail-closed VLM transport for one public target-screening review pass.

The transport consumes only a pass's public ``assignment.csv`` and blank
``scoring_template.csv``.  It never opens a private binding or the other pass.
API/transport/parse failures are infrastructure records, never scientific
scores.  Successful calls are immutable per-review-ID checkpoints that can be
merged with checkpoints from fresh retry/shard roots.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

try:
    from evaluate_v2_baseline_with_vlm import (
        load_api_config,
        message_content,
        parse_model_json,
        urllib_transport,
    )
    import screen_causal_role_erasure_7mechanism_targets_v2 as screening
except ModuleNotFoundError:  # imported as ``scripts.<module>`` in tests
    from scripts.evaluate_v2_baseline_with_vlm import (
        load_api_config,
        message_content,
        parse_model_json,
        urllib_transport,
    )
    from scripts import screen_causal_role_erasure_7mechanism_targets_v2 as screening


WORKFLOW_VERSION = "causal_role_erasure_7mechanism_target_vlm_transport_v2"
CHECKPOINT_STATUS = "schema_valid_scientific_checkpoint"
INFRASTRUCTURE_ERROR_STATUS = "infrastructure_error_no_scientific_score"
PASS_NAMES = ("reviewer_a", "reviewer_b")
SCORE_FIELDS = screening.SCORE_FIELDS
FRAME_RANGES = screening.PANEL_RANGES

RESPONSE_FIELDS = (*SCORE_FIELDS, "frame_evidence", "confidence")
EVIDENCE_ITEM_FIELDS = ("frame_index", "observation")
RUN_REGISTRATION_FIELDS = (
    "schema_version",
    "workflow_version",
    "status",
    "pass_name",
    "assignment",
    "scoring_template",
    "selected_start",
    "selected_limit",
    "selected_count",
    "selected_review_ids",
    "composite_inventory_sha256",
    "model",
    "temperature",
    "max_tokens",
    "token_parameter_name",
    "timeout",
    "workers",
    "response_schema_sha256",
    "dry_run",
)
CHECKPOINT_FIELDS = (
    "schema_version",
    "workflow_version",
    "status",
    "pass_name",
    "anonymous_review_id",
    "assignment_row_sha256",
    "composite_sha256",
    "request_sha256",
    "model",
    "raw_response",
    "model_content_sha256",
    "normalized",
)
ERROR_FIELDS = (
    "schema_version",
    "workflow_version",
    "status",
    "pass_name",
    "anonymous_review_id",
    "assignment_row_sha256",
    "composite_sha256",
    "request_descriptor_sha256",
    "error_type",
    "error_message",
    "raw_response",
    "model_content_sha256",
    "replay_policy",
)
REAL_SUMMARY_FIELDS = (
    "schema_version",
    "workflow_version",
    "status",
    "pass_name",
    "selected_count",
    "successful_checkpoints",
    "infrastructure_errors",
    "run_registration",
    "checkpoints",
    "errors",
    "scientific_zero_fallbacks",
)

RESPONSE_SCHEMA: dict[str, Any] = {
    **{field: "integer 0, 1, or 2" for field in SCORE_FIELDS},
    "frame_evidence": {
        field: [
            {
                "frame_index": "integer 0..48",
                "observation": "short visible observation",
            }
        ]
        for field in SCORE_FIELDS
    },
    "confidence": {field: "number in [0,1]" for field in SCORE_FIELDS},
}

Transport = Callable[[str, str, dict[str, Any], int], dict[str, Any]]
ISOLATED_URLLIB_CHILD_ARG = "--isolated-urllib-transport-child"


class ReviewTransportError(RuntimeError):
    """Fail-closed public-input, API, parsing, or checkpoint error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReviewTransportError(message)


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


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def regular_file(path: Path, label: str) -> None:
    require(path.is_file() and not path.is_symlink(), f"{label} missing or symlinked: {path}")


def file_ref(path: Path) -> dict[str, Any]:
    regular_file(path, "referenced file")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def relative_file_ref(root: Path, path: Path) -> dict[str, Any]:
    regular_file(path, "run artifact")
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def write_bytes_exclusive(path: Path, raw: bytes, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, mode)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def write_json_exclusive(path: Path, payload: Mapping[str, Any], mode: int = 0o644) -> None:
    write_bytes_exclusive(
        path,
        json.dumps(dict(payload), ensure_ascii=False, indent=2).encode("utf-8") + b"\n",
        mode=mode,
    )


def _isolated_urllib_transport_child() -> int:
    """Run one urllib request from a stdin-only private request envelope."""

    api_key = ""
    try:
        envelope = json.loads(sys.stdin.buffer.read().decode("utf-8"))
        require(
            isinstance(envelope, dict)
            and set(envelope) == {"url", "api_key", "payload", "timeout"},
            "isolated transport request schema mismatch",
        )
        api_key = str(envelope["api_key"])
        response = urllib_transport(
            str(envelope["url"]),
            api_key,
            envelope["payload"],
            int(envelope["timeout"]),
        )
        result = {"ok": True, "response": response}
    except BaseException as exc:
        message = str(exc)
        if api_key:
            message = message.replace(api_key, "[REDACTED]")
        result = {
            "ok": False,
            "error_type": type(exc).__name__,
            "error_message": message,
        }
    sys.stdout.buffer.write(canonical_json_bytes(result))
    sys.stdout.buffer.flush()
    return 0


def isolated_urllib_transport(
    url: str,
    api_key: str,
    payload: dict[str, Any],
    timeout: int,
) -> dict[str, Any]:
    """Call urllib in a killable child whose total lifetime is hard-capped.

    The secret and request body are sent over stdin, never argv or the child
    environment.  This protects long review shards from TLS/proxy handshakes
    that occasionally ignore urllib's socket timeout.
    """

    private_request = canonical_json_bytes(
        {
            "url": url,
            "api_key": api_key,
            "payload": payload,
            "timeout": timeout,
        }
    )
    command = [sys.executable, str(Path(__file__).resolve()), ISOLATED_URLLIB_CHILD_ARG]
    try:
        completed = subprocess.run(
            command,
            input=private_request,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(
            f"VLM transport exceeded hard wall-clock timeout of {timeout} seconds"
        ) from exc
    if completed.returncode != 0:
        raise ReviewTransportError(
            f"isolated VLM transport child exited with code {completed.returncode}"
        )
    try:
        result = json.loads(completed.stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReviewTransportError(
            "isolated VLM transport child returned an invalid envelope"
        ) from exc
    require(isinstance(result, dict), "isolated transport result is not an object")
    if result.get("ok") is True:
        require(
            set(result) == {"ok", "response"}
            and isinstance(result["response"], dict),
            "isolated transport success envelope mismatch",
        )
        return result["response"]
    require(
        set(result) == {"ok", "error_type", "error_message"}
        and result.get("ok") is False,
        "isolated transport error envelope mismatch",
    )
    child_type = str(result["error_type"])
    child_message = str(result["error_message"]).replace(api_key, "[REDACTED]")
    if child_type in {"TimeoutError", "socket.timeout"}:
        raise TimeoutError(child_message)
    raise ReviewTransportError(
        f"isolated VLM transport failed ({child_type}): {child_message}"
    )


def _resolve_public_composite(
    assignment_path: Path,
    composite_value: str,
) -> Path:
    pass_root = assignment_path.parent.resolve(strict=True)
    require(pass_root.name in PASS_NAMES, "assignment parent is not reviewer_a or reviewer_b")
    require(pass_root.parent.name == "public", "assignment is not inside a public pass root")
    package_root = pass_root.parent.parent
    raw = Path(composite_value)
    composite = raw if raw.is_absolute() else package_root / raw
    regular_file(composite, "public composite")
    resolved = composite.resolve(strict=True)
    expected_media_root = (pass_root / "media").resolve(strict=True)
    require(resolved.parent == expected_media_root, "composite escaped this pass's public media root")
    require(resolved.suffix.lower() in {".jpg", ".jpeg", ".png"}, "composite is not a supported image")
    return resolved


def load_public_pass(
    assignment_path: Path,
    template_path: Path,
) -> tuple[str, list[dict[str, Any]]]:
    regular_file(assignment_path, "public assignment")
    regular_file(template_path, "public scoring template")
    require(assignment_path.parent.resolve() == template_path.parent.resolve(), "assignment/template must be from the same public pass")
    pass_name = assignment_path.parent.name
    require(pass_name in PASS_NAMES, "unknown public review pass")
    assignment_fields, assignments = screening.read_csv(assignment_path)
    template_fields, templates = screening.read_csv(template_path)
    require(tuple(assignment_fields) == screening.ASSIGNMENT_FIELDS, "assignment columns differ from screening workflow")
    require(tuple(template_fields) == screening.SCORE_TEMPLATE_FIELDS, "scoring-template columns differ from screening workflow")
    require(
        len(assignments) == len(templates) == screening.EXPECTED_NEW_VIDEOS,
        "public pass must contain exactly 1,152 rows",
    )
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    composite_paths: set[Path] = set()
    for position, (assignment, template) in enumerate(zip(assignments, templates)):
        review_id = assignment["anonymous_review_id"]
        require(assignment["assignment_position"] == str(position), f"assignment row {position}: position mismatch")
        require(review_id == template["anonymous_review_id"], f"assignment row {position}: template ID mismatch")
        require(review_id and review_id not in seen_ids, f"assignment row {position}: duplicate/blank review ID")
        require(
            all(template[field] == "" for field in (*SCORE_FIELDS, "reviewer_notes")),
            "public scoring template is not blank",
        )
        composite = _resolve_public_composite(assignment_path, assignment["composite_path"])
        require(composite not in composite_paths, "duplicate public composite path")
        seen_ids.add(review_id)
        composite_paths.add(composite)
        rows.append(
            {
                "assignment": dict(assignment),
                "template": dict(template),
                "composite": composite,
                "composite_sha256": sha256_file(composite),
                "assignment_row_sha256": object_sha256(assignment),
            }
        )
    media_directory = assignment_path.parent / "media"
    require(
        media_directory.is_dir() and not media_directory.is_symlink(),
        "public media directory is missing or symlinked",
    )
    raw_media_entries = list(media_directory.iterdir())
    require(
        all(path.is_file() and not path.is_symlink() for path in raw_media_entries),
        "public media directory contains a non-regular entry",
    )
    media_entries = {path.resolve(strict=True) for path in raw_media_entries}
    require(media_entries == composite_paths, "public media directory differs from assignment inventory")
    return pass_name, rows


def response_schema_sha256() -> str:
    return object_sha256(RESPONSE_SCHEMA)


def token_parameter_name_for(model: str) -> str:
    require(
        isinstance(model, str) and model.strip() == model and bool(model),
        "model name must be nonempty and whitespace-normalized",
    )
    return (
        "max_completion_tokens"
        if model.casefold().startswith("gpt-5.6")
        else "max_tokens"
    )


def validate_model_request_config(model: str, temperature: float) -> str:
    parameter_name = token_parameter_name_for(model)
    require(0.0 <= temperature <= 2.0, "temperature is outside [0,2]")
    if parameter_name == "max_completion_tokens":
        require(
            temperature == 1.0,
            "GPT-5.6 chat completions require temperature=1.0",
        )
    return parameter_name


def prompt_for(row: Mapping[str, Any]) -> str:
    assignment = row["assignment"]
    field_rules = screening.reviewer_instructions()["fields"]
    return "\n".join(
        [
            "You are independently screening one generated training-target video.",
            "The supplied image contains all 49 indexed frames in five overlapping temporal panels:",
            ", ".join(f"{start}-{end}" for start, end in FRAME_RANGES) + ".",
            "Judge only visible image evidence. Text describes what to look for; it is not proof that content is present or absent.",
            "This is target-data screening, not formal method evaluation and not a winner comparison.",
            "",
            f"Mechanism: {assignment['mechanism_name']}",
            f"Requested source-free target scene: {assignment['target_prompt']}",
            f"Receiver that should remain recognizable: {assignment['receiver']}",
            f"Source that should be absent: {assignment['source_object']}",
            f"Trigger that should be absent: {assignment['expected_trigger']}",
            f"Footprint that should be absent: {assignment['expected_footprint']}",
            f"Expected counterfactual state: {assignment['expected_counterfactual_state']}",
            f"Excluded content summary: {assignment['excluded_content_phrase']}",
            "",
            "Score every atomic field with exactly one integer 0, 1, or 2 using these frozen meanings:",
            json.dumps(field_rules, ensure_ascii=False, sort_keys=True),
            "For every field, provide at least one frame-indexed visual observation. Frame indices must be 0..48.",
            "Also provide a separate confidence in [0,1] for every field.",
            "Return only one JSON object with exactly this schema and no extra keys:",
            json.dumps(RESPONSE_SCHEMA, ensure_ascii=False, sort_keys=True),
        ]
    )


def image_data_url(path: Path) -> str:
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def request_payload_for(
    row: Mapping[str, Any],
    *,
    model: str,
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    token_parameter_name = validate_model_request_config(model, temperature)
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_for(row)},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_data_url(row["composite"])},
                    },
                ],
            }
        ],
        "temperature": temperature,
    }
    payload[token_parameter_name] = max_tokens
    return payload


def dry_payload_for(row: Mapping[str, Any], *, model: str, temperature: float, max_tokens: int) -> dict[str, Any]:
    token_parameter_name = validate_model_request_config(model, temperature)
    return {
        "anonymous_review_id": row["assignment"]["anonymous_review_id"],
        "composite_path": str(row["composite"]),
        "composite_sha256": row["composite_sha256"],
        "prompt": prompt_for(row),
        "model": model,
        "temperature": temperature,
        "token_parameter_name": token_parameter_name,
        "token_limit": max_tokens,
        "response_schema": RESPONSE_SCHEMA,
        "actual_request_image_policy": "single full composite as base64 data URL",
    }


def normalize_model_object(parsed: Mapping[str, Any]) -> dict[str, Any]:
    require(isinstance(parsed, Mapping), "model response is not a JSON object")
    require(set(parsed) == set(RESPONSE_FIELDS), "model response keys differ from fixed schema")
    scores: dict[str, int] = {}
    for field in SCORE_FIELDS:
        value = parsed[field]
        require(type(value) is int and value in {0, 1, 2}, f"invalid score for {field}")
        scores[field] = value
    evidence_raw = parsed["frame_evidence"]
    confidence_raw = parsed["confidence"]
    require(isinstance(evidence_raw, Mapping) and set(evidence_raw) == set(SCORE_FIELDS), "frame_evidence keys differ")
    require(isinstance(confidence_raw, Mapping) and set(confidence_raw) == set(SCORE_FIELDS), "confidence keys differ")
    evidence: dict[str, list[dict[str, Any]]] = {}
    confidence: dict[str, float] = {}
    for field in SCORE_FIELDS:
        items = evidence_raw[field]
        require(isinstance(items, list) and 1 <= len(items) <= 8, f"{field}: frame evidence must contain 1..8 entries")
        normalized_items: list[dict[str, Any]] = []
        for item in items:
            require(
                isinstance(item, Mapping)
                and set(item) == set(EVIDENCE_ITEM_FIELDS),
                f"{field}: evidence item schema mismatch",
            )
            frame_index = item["frame_index"]
            observation = item["observation"]
            require(type(frame_index) is int and 0 <= frame_index < screening.FRAME_COUNT, f"{field}: invalid evidence frame")
            require(isinstance(observation, str) and 1 <= len(observation.strip()) <= 1000, f"{field}: invalid evidence observation")
            normalized_items.append(
                {"frame_index": frame_index, "observation": observation.strip()}
            )
        raw_confidence = confidence_raw[field]
        require(
            type(raw_confidence) in {int, float}
            and 0.0 <= float(raw_confidence) <= 1.0,
            f"{field}: invalid confidence",
        )
        evidence[field] = normalized_items
        confidence[field] = float(raw_confidence)
    return {**scores, "frame_evidence": evidence, "confidence": confidence}


def _request_descriptor(
    row: Mapping[str, Any], *, model: str, temperature: float, max_tokens: int
) -> dict[str, Any]:
    token_parameter_name = validate_model_request_config(model, temperature)
    return {
        "anonymous_review_id": row["assignment"]["anonymous_review_id"],
        "assignment_row_sha256": row["assignment_row_sha256"],
        "composite_sha256": row["composite_sha256"],
        "prompt_sha256": hashlib.sha256(prompt_for(row).encode("utf-8")).hexdigest(),
        "response_schema_sha256": response_schema_sha256(),
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "token_parameter_name": token_parameter_name,
    }


def evaluate_one(
    row: Mapping[str, Any],
    *,
    url: str,
    api_key: str,
    model: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
    transport: Transport = isolated_urllib_transport,
) -> dict[str, Any]:
    descriptor = _request_descriptor(
        row, model=model, temperature=temperature, max_tokens=max_tokens
    )
    response: dict[str, Any] | None = None
    content = ""
    try:
        request = request_payload_for(
            row, model=model, temperature=temperature, max_tokens=max_tokens
        )
        response = transport(url, api_key, request, timeout)
        require(isinstance(response, dict), "API response is not an object")
        content = message_content(response)
        require(content.strip(), "API response contains no model content")
        parsed = parse_model_json(content)
        normalized = normalize_model_object(parsed)
        return {
            "ok": True,
            "descriptor": descriptor,
            "response": response,
            "model_content": content,
            "normalized": normalized,
        }
    except Exception as exc:
        return {
            "ok": False,
            "descriptor": descriptor,
            "error_type": type(exc).__name__,
            "error_message": str(exc).replace(api_key, "[REDACTED]"),
            "response": response,
            "model_content": content,
        }


def _select_rows(
    rows: Sequence[Mapping[str, Any]], start: int, limit: int | None
) -> list[Mapping[str, Any]]:
    require(start >= 0 and start <= len(rows), "--start is outside assignment")
    require(limit is None or limit > 0, "--limit must be positive")
    selected = list(rows[start:] if limit is None else rows[start : start + limit])
    require(selected, "selected review shard is empty")
    return selected


def _prepare_fresh_run_root(output_root: Path) -> None:
    require(not output_root.exists() and not output_root.is_symlink(), f"fresh-only output root already exists: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    output_root.mkdir()
    (output_root / "checkpoints").mkdir()
    (output_root / "raw_responses").mkdir()
    (output_root / "errors").mkdir()
    write_bytes_exclusive(output_root / ".incomplete", b"review run incomplete\n", mode=0o600)


def run_review(
    *,
    assignment_path: Path,
    template_path: Path,
    output_root: Path,
    dry_run: bool,
    start: int,
    limit: int | None,
    workers: int,
    model: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
    base_url: str | None = None,
    api_key: str | None = None,
    transport: Transport = isolated_urllib_transport,
) -> dict[str, Any]:
    require(workers > 0, "workers must be positive")
    require(timeout > 0 and max_tokens > 0, "timeout/token limit must be positive")
    token_parameter_name = validate_model_request_config(model, temperature)
    pass_name, all_rows = load_public_pass(assignment_path, template_path)
    selected = _select_rows(all_rows, start, limit)
    _prepare_fresh_run_root(output_root)
    registration = {
        "schema_version": 1,
        "workflow_version": WORKFLOW_VERSION,
        "status": "registered_fresh_public_pass_shard",
        "pass_name": pass_name,
        "assignment": file_ref(assignment_path),
        "scoring_template": file_ref(template_path),
        "selected_start": start,
        "selected_limit": limit,
        "selected_count": len(selected),
        "selected_review_ids": [row["assignment"]["anonymous_review_id"] for row in selected],
        "composite_inventory_sha256": object_sha256(
            [
                {
                    "anonymous_review_id": row["assignment"]["anonymous_review_id"],
                    "sha256": row["composite_sha256"],
                }
                for row in selected
            ]
        ),
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "token_parameter_name": token_parameter_name,
        "timeout": timeout,
        "workers": workers,
        "response_schema_sha256": response_schema_sha256(),
        "dry_run": dry_run,
    }
    require(tuple(registration) == RUN_REGISTRATION_FIELDS, "internal run-registration schema drift")
    registration_path = output_root / "run_registration.json"
    write_json_exclusive(registration_path, registration, mode=0o600)

    if dry_run:
        payload_path = output_root / "payloads.jsonl"
        write_bytes_exclusive(
            payload_path,
            b"".join(
                canonical_json_bytes(
                    dry_payload_for(
                        row,
                        model=model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                )
                for row in selected
            ),
        )
        summary = {
            "schema_version": 1,
            "workflow_version": WORKFLOW_VERSION,
            "status": "dry_run_payloads_only_no_api_calls",
            "pass_name": pass_name,
            "selected_count": len(selected),
            "successful_checkpoints": 0,
            "infrastructure_errors": 0,
            "run_registration": relative_file_ref(output_root, registration_path),
            "payloads": relative_file_ref(output_root, payload_path),
        }
        write_json_exclusive(output_root / "run_summary.json", summary, mode=0o600)
        (output_root / ".incomplete").unlink()
        return summary

    require(bool(base_url) and bool(api_key), "real review requires nonempty API URL and key")
    url = base_url.rstrip("/") + "/chat/completions"
    successful_by_index: dict[int, dict[str, Any]] = {}
    errors_by_index: dict[int, dict[str, Any]] = {}

    def persist_result(
        index: int, row: Mapping[str, Any], result: Mapping[str, Any]
    ) -> None:
        review_id = row["assignment"]["anonymous_review_id"]
        descriptor = result["descriptor"]
        if not result["ok"]:
            error_raw_ref: dict[str, Any] | None = None
            error_content_sha256: str | None = None
            if result.get("response") is not None:
                error_raw_path = (
                    output_root / "raw_responses" / f"{review_id}.json"
                )
                write_json_exclusive(
                    error_raw_path, result["response"], mode=0o600
                )
                error_raw_ref = relative_file_ref(output_root, error_raw_path)
                error_content_sha256 = hashlib.sha256(
                    str(result.get("model_content", "")).encode("utf-8")
                ).hexdigest()
            error = {
                "schema_version": 1,
                "workflow_version": WORKFLOW_VERSION,
                "status": INFRASTRUCTURE_ERROR_STATUS,
                "pass_name": pass_name,
                "anonymous_review_id": review_id,
                "assignment_row_sha256": row["assignment_row_sha256"],
                "composite_sha256": row["composite_sha256"],
                "request_descriptor_sha256": object_sha256(descriptor),
                "error_type": result["error_type"],
                "error_message": result["error_message"],
                "raw_response": error_raw_ref,
                "model_content_sha256": error_content_sha256,
                "replay_policy": "same anonymous_review_id in a new fresh-only run root",
            }
            require(tuple(error) == ERROR_FIELDS, "internal error schema drift")
            error_path = output_root / "errors" / f"{review_id}.json"
            write_json_exclusive(error_path, error, mode=0o600)
            errors_by_index[index] = relative_file_ref(output_root, error_path)
            return
        raw_path = output_root / "raw_responses" / f"{review_id}.json"
        write_json_exclusive(raw_path, result["response"], mode=0o600)
        checkpoint = {
            "schema_version": 1,
            "workflow_version": WORKFLOW_VERSION,
            "status": CHECKPOINT_STATUS,
            "pass_name": pass_name,
            "anonymous_review_id": review_id,
            "assignment_row_sha256": row["assignment_row_sha256"],
            "composite_sha256": row["composite_sha256"],
            "request_sha256": object_sha256(descriptor),
            "model": model,
            "raw_response": relative_file_ref(output_root, raw_path),
            "model_content_sha256": hashlib.sha256(
                result["model_content"].encode("utf-8")
            ).hexdigest(),
            "normalized": result["normalized"],
        }
        require(tuple(checkpoint) == CHECKPOINT_FIELDS, "internal checkpoint schema drift")
        checkpoint_path = output_root / "checkpoints" / f"{review_id}.json"
        write_json_exclusive(checkpoint_path, checkpoint, mode=0o600)
        successful_by_index[index] = relative_file_ref(output_root, checkpoint_path)

    def evaluate_selected(row: Mapping[str, Any]) -> dict[str, Any]:
        return evaluate_one(
            row,
            url=url,
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            transport=transport,
        )

    if workers == 1:
        for index, row in enumerate(selected):
            persist_result(index, row, evaluate_selected(row))
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            pending = {
                executor.submit(evaluate_selected, row): (index, row)
                for index, row in enumerate(selected)
            }
            for future in as_completed(pending):
                index, row = pending[future]
                persist_result(index, row, future.result())

    successful = [successful_by_index[index] for index in sorted(successful_by_index)]
    errors = [errors_by_index[index] for index in sorted(errors_by_index)]

    summary = {
        "schema_version": 1,
        "workflow_version": WORKFLOW_VERSION,
        "status": (
            "completed_checkpoint_shard"
            if not errors
            else "completed_with_infrastructure_errors_requires_fresh_replay"
        ),
        "pass_name": pass_name,
        "selected_count": len(selected),
        "successful_checkpoints": len(successful),
        "infrastructure_errors": len(errors),
        "run_registration": relative_file_ref(output_root, registration_path),
        "checkpoints": successful,
        "errors": errors,
        "scientific_zero_fallbacks": 0,
    }
    write_json_exclusive(output_root / "run_summary.json", summary, mode=0o600)
    (output_root / ".incomplete").unlink()
    return summary


def _resolve_ref(root: Path, ref: Mapping[str, Any], label: str) -> Path:
    require(set(ref) == {"path", "sha256", "size_bytes"}, f"{label} ref schema mismatch")
    path = Path(str(ref["path"]))
    if not path.is_absolute():
        path = root / path
    regular_file(path, label)
    require(path.stat().st_size == int(ref["size_bytes"]), f"{label} size changed")
    require(sha256_file(path) == ref["sha256"], f"{label} SHA-256 changed")
    return path


def _load_json(path: Path, label: str) -> dict[str, Any]:
    regular_file(path, label)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReviewTransportError(f"cannot parse {label}") from exc
    require(isinstance(payload, dict), f"{label} is not an object")
    return payload


def _validate_checkpoint(
    run_root: Path,
    checkpoint_path: Path,
    public_by_id: Mapping[str, Mapping[str, Any]],
    pass_name: str,
    registration: Mapping[str, Any],
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    checkpoint = _load_json(checkpoint_path, "review checkpoint")
    require(tuple(checkpoint) == CHECKPOINT_FIELDS, "review checkpoint fields differ")
    review_id = str(checkpoint["anonymous_review_id"])
    row = public_by_id.get(review_id)
    require(row is not None, "checkpoint review ID is not in public assignment")
    require(
        checkpoint["workflow_version"] == WORKFLOW_VERSION
        and checkpoint["status"] == CHECKPOINT_STATUS
        and checkpoint["pass_name"] == pass_name,
        "checkpoint identity/status mismatch",
    )
    require(checkpoint["assignment_row_sha256"] == row["assignment_row_sha256"], "checkpoint assignment binding changed")
    require(checkpoint["composite_sha256"] == row["composite_sha256"], "checkpoint composite binding changed")
    require(checkpoint["model"] == registration["model"], "checkpoint model differs from run registration")
    descriptor = _request_descriptor(
        row,
        model=str(registration["model"]),
        temperature=float(registration["temperature"]),
        max_tokens=int(registration["max_tokens"]),
    )
    require(checkpoint["request_sha256"] == object_sha256(descriptor), "checkpoint request binding changed")
    normalized = normalize_model_object(checkpoint["normalized"])
    raw_path = _resolve_ref(run_root, checkpoint["raw_response"], "raw API response")
    require(
        raw_path.parent.resolve(strict=True)
        == (run_root / "raw_responses").resolve(strict=True),
        "raw API response escaped run raw_responses directory",
    )
    raw = _load_json(raw_path, "raw API response")
    content = message_content(raw)
    require(
        hashlib.sha256(content.encode("utf-8")).hexdigest()
        == checkpoint["model_content_sha256"],
        "raw model content changed",
    )
    reparsed = normalize_model_object(parse_model_json(content))
    require(reparsed == normalized, "checkpoint normalization differs from raw response")
    return review_id, checkpoint, normalized


def merge_checkpoints(
    *,
    assignment_path: Path,
    template_path: Path,
    run_roots: Sequence[Path],
    output_root: Path,
) -> dict[str, Any]:
    require(run_roots, "merge requires at least one run root")
    pass_name, public_rows = load_public_pass(assignment_path, template_path)
    public_by_id = {
        row["assignment"]["anonymous_review_id"]: row for row in public_rows
    }
    require(not output_root.exists() and not output_root.is_symlink(), f"fresh-only merge output exists: {output_root}")
    checkpoints: dict[str, tuple[Path, dict[str, Any], dict[str, Any]]] = {}
    models: set[str] = set()
    review_configs: set[tuple[Any, ...]] = set()
    run_refs: list[dict[str, Any]] = []
    for run_root in run_roots:
        require(run_root.is_dir() and not run_root.is_symlink(), f"run root missing or symlinked: {run_root}")
        require(not (run_root / ".incomplete").exists(), f"run root is incomplete: {run_root}")
        require(
            {path.name for path in run_root.iterdir()}
            == {
                "run_registration.json",
                "run_summary.json",
                "checkpoints",
                "raw_responses",
                "errors",
            },
            f"run root contains unexpected entries: {run_root}",
        )
        for directory_name in ("checkpoints", "raw_responses", "errors"):
            directory = run_root / directory_name
            require(
                directory.is_dir() and not directory.is_symlink(),
                f"unsafe run directory: {directory}",
            )
        registration_path = run_root / "run_registration.json"
        summary_path = run_root / "run_summary.json"
        registration = _load_json(registration_path, "run registration")
        summary = _load_json(summary_path, "run summary")
        require(tuple(registration) == RUN_REGISTRATION_FIELDS, "run registration fields differ")
        require(tuple(summary) == REAL_SUMMARY_FIELDS, "run summary fields differ")
        require(
            registration["workflow_version"] == WORKFLOW_VERSION
            and registration["pass_name"] == pass_name
            and registration["dry_run"] is False,
            "run registration identity mismatch",
        )
        require(
            registration["response_schema_sha256"] == response_schema_sha256(),
            "run response schema differs from current fixed schema",
        )
        require(
            registration["token_parameter_name"]
            == validate_model_request_config(
                str(registration["model"]), float(registration["temperature"])
            ),
            "run token-parameter policy differs from model contract",
        )
        bound_assignment = _resolve_ref(
            run_root, registration["assignment"], "registered public assignment"
        )
        bound_template = _resolve_ref(
            run_root,
            registration["scoring_template"],
            "registered public scoring template",
        )
        require(
            bound_assignment.resolve() == assignment_path.resolve()
            and bound_template.resolve() == template_path.resolve(),
            "run root binds different public inputs",
        )
        require(
            summary.get("status")
            in {
                "completed_checkpoint_shard",
                "completed_with_infrastructure_errors_requires_fresh_replay",
            },
            "run summary is not mergeable",
        )
        require(
            summary["workflow_version"] == WORKFLOW_VERSION
            and summary["pass_name"] == pass_name
            and summary["selected_count"] == registration["selected_count"]
            and summary["scientific_zero_fallbacks"] == 0,
            "run summary identity/count mismatch",
        )
        bound_registration = _resolve_ref(
            run_root, summary["run_registration"], "summary run registration"
        )
        require(
            bound_registration.resolve() == registration_path.resolve(),
            "run summary binds a different registration",
        )
        require(
            summary["successful_checkpoints"] == len(summary["checkpoints"])
            and summary["infrastructure_errors"] == len(summary["errors"])
            and summary["successful_checkpoints"]
            + summary["infrastructure_errors"]
            == summary["selected_count"],
            "run summary artifact counts mismatch",
        )
        models.add(str(registration["model"]))
        review_configs.add(
            (
                registration["model"],
                registration["temperature"],
                registration["max_tokens"],
                registration["token_parameter_name"],
                registration["response_schema_sha256"],
            )
        )
        start = int(registration["selected_start"])
        limit = registration["selected_limit"]
        expected_selected_rows = _select_rows(
            public_rows, start, None if limit is None else int(limit)
        )
        expected_selected_ids = [
            row["assignment"]["anonymous_review_id"]
            for row in expected_selected_rows
        ]
        require(
            registration["selected_review_ids"] == expected_selected_ids,
            "run registered shard differs from public assignment order",
        )
        require(
            registration["composite_inventory_sha256"]
            == object_sha256(
                [
                    {
                        "anonymous_review_id": row["assignment"][
                            "anonymous_review_id"
                        ],
                        "sha256": row["composite_sha256"],
                    }
                    for row in expected_selected_rows
                ]
            ),
            "run composite inventory binding changed",
        )
        selected_ids = set(registration["selected_review_ids"])
        require(
            len(selected_ids) == registration["selected_count"]
            and selected_ids <= set(public_by_id),
            "run selected IDs escape public assignment or contain duplicates",
        )
        checkpoint_refs = {
            _resolve_ref(run_root, ref, "summary checkpoint").resolve(): ref
            for ref in summary["checkpoints"]
        }
        actual_checkpoints = {
            path.resolve()
            for path in (run_root / "checkpoints").iterdir()
            if path.is_file() and not path.is_symlink()
        }
        require(set(checkpoint_refs) == actual_checkpoints, "run checkpoint directory differs from summary")
        error_refs = {
            _resolve_ref(run_root, ref, "summary infrastructure error").resolve(): ref
            for ref in summary["errors"]
        }
        actual_errors = {
            path.resolve()
            for path in (run_root / "errors").iterdir()
            if path.is_file() and not path.is_symlink()
        }
        require(set(error_refs) == actual_errors, "run error directory differs from summary")
        successful_ids_in_run: set[str] = set()
        for path in sorted(actual_checkpoints):
            review_id, checkpoint, normalized = _validate_checkpoint(
                run_root, path, public_by_id, pass_name, registration
            )
            require(review_id in selected_ids, "checkpoint ID is outside registered shard")
            require(review_id not in checkpoints, f"duplicate successful checkpoint for {review_id}")
            successful_ids_in_run.add(review_id)
            checkpoints[review_id] = (path, checkpoint, normalized)
        error_ids_in_run: set[str] = set()
        error_raw_paths: set[Path] = set()
        for path in sorted(actual_errors):
            error = _load_json(path, "infrastructure error checkpoint")
            require(tuple(error) == ERROR_FIELDS, "infrastructure error fields differ")
            review_id = str(error["anonymous_review_id"])
            row = public_by_id.get(review_id)
            require(
                row is not None
                and review_id in selected_ids
                and review_id not in error_ids_in_run
                and review_id not in successful_ids_in_run,
                "infrastructure error ID is invalid, duplicate, or also successful",
            )
            require(
                error["workflow_version"] == WORKFLOW_VERSION
                and error["status"] == INFRASTRUCTURE_ERROR_STATUS
                and error["pass_name"] == pass_name
                and error["assignment_row_sha256"]
                == row["assignment_row_sha256"]
                and error["composite_sha256"] == row["composite_sha256"],
                "infrastructure error binding mismatch",
            )
            descriptor = _request_descriptor(
                row,
                model=str(registration["model"]),
                temperature=float(registration["temperature"]),
                max_tokens=int(registration["max_tokens"]),
            )
            require(
                error["request_descriptor_sha256"] == object_sha256(descriptor),
                "infrastructure error request binding changed",
            )
            require(
                not set(SCORE_FIELDS) & set(error),
                "infrastructure error illegally contains scientific scores",
            )
            raw_ref = error["raw_response"]
            if raw_ref is None:
                require(
                    error["model_content_sha256"] is None,
                    "transport error has content hash without raw response",
                )
            else:
                raw_path = _resolve_ref(
                    run_root, raw_ref, "infrastructure-error raw API response"
                )
                require(
                    raw_path.parent.resolve(strict=True)
                    == (run_root / "raw_responses").resolve(strict=True),
                    "infrastructure-error raw response escaped run directory",
                )
                raw = _load_json(raw_path, "infrastructure-error raw API response")
                content = message_content(raw)
                require(
                    hashlib.sha256(content.encode("utf-8")).hexdigest()
                    == error["model_content_sha256"],
                    "infrastructure-error raw model content changed",
                )
                error_raw_paths.add(raw_path.resolve())
            error_ids_in_run.add(review_id)
        actual_raw = {
            path.resolve()
            for path in (run_root / "raw_responses").iterdir()
            if path.is_file() and not path.is_symlink()
        }
        # Only checkpoints from this run, not earlier run roots, define its raw set.
        this_run_raw = {
            _resolve_ref(run_root, checkpoint["raw_response"], "raw API response").resolve()
            for review_id, (_, checkpoint, _) in checkpoints.items()
            if review_id in successful_ids_in_run
        }
        require(
            this_run_raw | error_raw_paths == actual_raw,
            "run raw-response directory differs from checkpoints/errors",
        )
        run_refs.append(
            {
                "run_registration": file_ref(registration_path),
                "run_summary": file_ref(summary_path),
            }
        )
    require(len(models) == 1, "merged checkpoints use different VLM models")
    require(len(review_configs) == 1, "merged checkpoints use different review configurations")
    missing = [review_id for review_id in public_by_id if review_id not in checkpoints]
    require(not missing, f"merge is missing {len(missing)} schema-valid checkpoints")

    output_root.parent.mkdir(parents=True, exist_ok=True)
    output_root.mkdir()
    completed_rows: list[dict[str, str]] = []
    raw_registry: dict[str, Any] = {}
    for row in public_rows:
        review_id = row["assignment"]["anonymous_review_id"]
        checkpoint_path, checkpoint, normalized = checkpoints[review_id]
        notes = json.dumps(
            {
                "frame_evidence": normalized["frame_evidence"],
                "confidence": normalized["confidence"],
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        completed_rows.append(
            {
                "anonymous_review_id": review_id,
                **{field: str(normalized[field]) for field in SCORE_FIELDS},
                "reviewer_notes": notes,
            }
        )
        raw_registry[review_id] = {
            "checkpoint": file_ref(checkpoint_path),
            "raw_response": file_ref(
                _resolve_ref(
                    checkpoint_path.parent.parent,
                    checkpoint["raw_response"],
                    "merged raw API response",
                )
            ),
            "model": checkpoint["model"],
        }
    completed_path = output_root / "completed_review.csv"
    screening.write_bytes_exclusive(
        completed_path,
        screening.csv_bytes(completed_rows, screening.SCORE_TEMPLATE_FIELDS),
        mode=0o600,
    )
    raw_registry_path = output_root / "raw_response_registry.json"
    write_json_exclusive(raw_registry_path, raw_registry, mode=0o600)
    manifest = {
        "schema_version": 1,
        "workflow_version": WORKFLOW_VERSION,
        "status": "complete_schema_valid_public_pass_review",
        "pass_name": pass_name,
        "model": next(iter(models)),
        "row_count": len(completed_rows),
        "assignment": file_ref(assignment_path),
        "scoring_template": file_ref(template_path),
        "source_runs": run_refs,
        "completed_review": relative_file_ref(output_root, completed_path),
        "raw_response_registry": relative_file_ref(output_root, raw_registry_path),
        "scientific_zero_fallbacks": 0,
    }
    write_json_exclusive(output_root / "merge_manifest.json", manifest, mode=0o600)
    return manifest


def resolve_path(project_root: Path, value: Path) -> Path:
    return value.resolve() if value.is_absolute() else (project_root / value).resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    review = subparsers.add_parser("review", help="create a fresh dry-run or API checkpoint shard")
    review.add_argument("--assignment", type=Path, required=True)
    review.add_argument("--template", type=Path, required=True)
    review.add_argument("--output-root", type=Path, required=True)
    mode = review.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--run-api", action="store_true")
    review.add_argument("--start", type=int, default=0)
    review.add_argument("--limit", type=int)
    review.add_argument("--workers", type=int, default=1)
    review.add_argument("--model", default="gpt-5.4")
    review.add_argument("--temperature", type=float, default=0.0)
    review.add_argument("--max-tokens", type=int, default=1600)
    review.add_argument("--timeout", type=int, default=180)
    review.add_argument("--base-url")
    review.add_argument("--api-key")
    review.add_argument("--api-config-file", type=Path)

    merge = subparsers.add_parser("merge", help="merge immutable success checkpoints into one completed public pass")
    merge.add_argument("--assignment", type=Path, required=True)
    merge.add_argument("--template", type=Path, required=True)
    merge.add_argument("--run-root", type=Path, action="append", required=True)
    merge.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    project_root = args.project_root.resolve()
    try:
        require(project_root.is_dir(), f"project root missing: {project_root}")
        if args.command == "review":
            assignment = resolve_path(project_root, args.assignment)
            template = resolve_path(project_root, args.template)
            output_root = resolve_path(project_root, args.output_root)
            base_url = api_key = None
            if args.run_api:
                config = (
                    None
                    if args.api_config_file is None
                    else resolve_path(project_root, args.api_config_file)
                )
                base_url, api_key = load_api_config(
                    config, args.base_url, args.api_key
                )
            result = run_review(
                assignment_path=assignment,
                template_path=template,
                output_root=output_root,
                dry_run=args.dry_run,
                start=args.start,
                limit=args.limit,
                workers=args.workers,
                model=args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
                base_url=base_url,
                api_key=api_key,
            )
            exit_code = 0 if result["infrastructure_errors"] == 0 else 2
        else:
            result = merge_checkpoints(
                assignment_path=resolve_path(project_root, args.assignment),
                template_path=resolve_path(project_root, args.template),
                run_roots=[resolve_path(project_root, value) for value in args.run_root],
                output_root=resolve_path(project_root, args.output_root),
            )
            exit_code = 0
    except (OSError, ReviewTransportError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return exit_code


if __name__ == "__main__":
    if sys.argv[1:] == [ISOLATED_URLLIB_CHILD_ARG]:
        raise SystemExit(_isolated_urllib_transport_child())
    raise SystemExit(main())
