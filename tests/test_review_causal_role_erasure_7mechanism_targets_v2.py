from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

import pytest

from scripts import review_causal_role_erasure_7mechanism_targets_v2 as review
from scripts import screen_causal_role_erasure_7mechanism_targets_v2 as screening


def _write_csv(path: Path, fields, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def _public_pass(tmp_path: Path):
    package = tmp_path / "package"
    pass_root = package / "public/reviewer_a"
    media = pass_root / "media"
    media.mkdir(parents=True)
    assignments = []
    templates = []
    for index in range(screening.EXPECTED_NEW_VIDEOS):
        review_id = f"a{index:04d}"
        image = media / f"anonymous_{index:04d}.jpg"
        image.write_bytes(f"full49-{index}".encode())
        assignments.append(
            {
                "assignment_position": str(index),
                "anonymous_review_id": review_id,
                "mechanism": screening.NEW_MECHANISMS[
                    index // screening.ROWS_PER_MECHANISM
                ],
                "mechanism_name": "Anonymous mechanism context",
                "composite_path": image.relative_to(package).as_posix(),
                "target_prompt": f"source-free target scene {index}",
                "receiver": f"receiver {index}",
                "source_object": f"source {index}",
                "expected_trigger": f"trigger {index}",
                "expected_footprint": f"footprint {index}",
                "expected_counterfactual_state": f"counterfactual {index}",
                "excluded_content_phrase": f"excluded {index}",
            }
        )
        templates.append(
            {
                "anonymous_review_id": review_id,
                **{field: "" for field in screening.SCORE_FIELDS},
                "reviewer_notes": "",
            }
        )
    assignment = pass_root / "assignment.csv"
    template = pass_root / "scoring_template.csv"
    _write_csv(assignment, screening.ASSIGNMENT_FIELDS, assignments)
    _write_csv(template, screening.SCORE_TEMPLATE_FIELDS, templates)
    return assignment, template


def _valid_model_object(score: int = 2):
    return {
        **{field: score for field in screening.SCORE_FIELDS},
        "frame_evidence": {
            field: [{"observation": f"visible evidence for {field}", "frame_index": 24}]
            for field in screening.SCORE_FIELDS
        },
        "confidence": {field: 0.9 for field in screening.SCORE_FIELDS},
    }


def _response(value=None):
    return {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        _valid_model_object() if value is None else value
                    )
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }


def test_fixed_schema_requires_scores_evidence_and_per_field_confidence():
    normalized = review.normalize_model_object(_valid_model_object())
    assert all(normalized[field] == 2 for field in screening.SCORE_FIELDS)
    assert normalized["frame_evidence"]["source_absent"][0]["frame_index"] == 24
    broken = _valid_model_object()
    broken["trigger_absent"] = 3
    with pytest.raises(review.ReviewTransportError, match="invalid score"):
        review.normalize_model_object(broken)
    broken = _valid_model_object()
    broken["frame_evidence"]["quality"] = []
    with pytest.raises(review.ReviewTransportError, match="1..8 entries"):
        review.normalize_model_object(broken)


def test_public_pass_rejects_nonblank_template_and_cross_pass_media(tmp_path: Path):
    assignment, template = _public_pass(tmp_path)
    fields, rows = _read_csv(template)
    rows[0]["quality"] = "0"
    _write_csv(template, fields, rows)
    with pytest.raises(review.ReviewTransportError, match="not blank"):
        review.load_public_pass(assignment, template)

    rows[0]["quality"] = ""
    _write_csv(template, fields, rows)
    a_fields, a_rows = _read_csv(assignment)
    other = assignment.parent.parent / "reviewer_b/media/other.jpg"
    other.parent.mkdir(parents=True)
    other.write_bytes(b"other")
    a_rows[0]["composite_path"] = other.relative_to(
        assignment.parent.parent.parent
    ).as_posix()
    _write_csv(assignment, a_fields, a_rows)
    with pytest.raises(review.ReviewTransportError, match="escaped this pass"):
        review.load_public_pass(assignment, template)


def test_dry_run_exports_payloads_without_transport_call(tmp_path: Path):
    assignment, template = _public_pass(tmp_path)
    output = tmp_path / "dry"

    def forbidden_transport(*_args, **_kwargs):
        raise AssertionError("dry run called transport")

    result = review.run_review(
        assignment_path=assignment,
        template_path=template,
        output_root=output,
        dry_run=True,
        start=10,
        limit=3,
        workers=2,
        model="test-vlm",
        temperature=0.0,
        max_tokens=500,
        timeout=30,
        transport=forbidden_transport,
    )
    assert result["status"] == "dry_run_payloads_only_no_api_calls"
    assert result["selected_count"] == 3
    payloads = [json.loads(line) for line in (output / "payloads.jsonl").read_text().splitlines()]
    assert [row["anonymous_review_id"] for row in payloads] == [
        "a0010",
        "a0011",
        "a0012",
    ]
    assert not list((output / "checkpoints").iterdir())
    assert not list((output / "errors").iterdir())
    with pytest.raises(review.ReviewTransportError, match="fresh-only"):
        review.run_review(
            assignment_path=assignment,
            template_path=template,
            output_root=output,
            dry_run=True,
            start=10,
            limit=3,
            workers=1,
            model="test-vlm",
            temperature=0.0,
            max_tokens=500,
            timeout=30,
        )


def test_gpt56_luna_uses_completion_tokens_and_requires_temperature_one(
    tmp_path: Path,
):
    assignment, template = _public_pass(tmp_path)
    _, public_rows = review.load_public_pass(assignment, template)
    row = public_rows[0]
    payload = review.request_payload_for(
        row,
        model="gpt-5.6-luna",
        temperature=1.0,
        max_tokens=777,
    )
    assert payload["max_completion_tokens"] == 777
    assert "max_tokens" not in payload
    assert payload["temperature"] == 1.0
    descriptor = review._request_descriptor(
        row,
        model="gpt-5.6-luna",
        temperature=1.0,
        max_tokens=777,
    )
    assert descriptor["token_parameter_name"] == "max_completion_tokens"

    dry_root = tmp_path / "luna_dry"
    review.run_review(
        assignment_path=assignment,
        template_path=template,
        output_root=dry_root,
        dry_run=True,
        start=0,
        limit=1,
        workers=1,
        model="gpt-5.6-luna",
        temperature=1.0,
        max_tokens=777,
        timeout=30,
    )
    registration = json.loads((dry_root / "run_registration.json").read_text())
    assert registration["token_parameter_name"] == "max_completion_tokens"
    dry_payload = json.loads((dry_root / "payloads.jsonl").read_text())
    assert dry_payload["token_parameter_name"] == "max_completion_tokens"
    assert dry_payload["token_limit"] == 777

    rejected_root = tmp_path / "luna_temp0_must_not_exist"
    with pytest.raises(
        review.ReviewTransportError,
        match="require temperature=1.0",
    ):
        review.run_review(
            assignment_path=assignment,
            template_path=template,
            output_root=rejected_root,
            dry_run=True,
            start=0,
            limit=1,
            workers=1,
            model="gpt-5.6-luna",
            temperature=0.0,
            max_tokens=777,
            timeout=30,
        )
    assert not rejected_root.exists()

    legacy_payload = review.request_payload_for(
        row,
        model="test-vlm",
        temperature=0.0,
        max_tokens=333,
    )
    assert legacy_payload["max_tokens"] == 333
    assert "max_completion_tokens" not in legacy_payload


def test_isolated_transport_uses_stdin_only_and_enforces_hard_timeout(monkeypatch):
    captured = {}
    secret = "secret-must-not-enter-argv"

    def simulated_stuck_child(command, **kwargs):
        captured["command"] = command
        captured.update(kwargs)
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(review.subprocess, "run", simulated_stuck_child)
    with pytest.raises(TimeoutError, match="hard wall-clock timeout of 7 seconds"):
        review.isolated_urllib_transport(
            "https://unit.test/v1/chat/completions",
            secret,
            {"model": "gpt-5.6-luna"},
            7,
        )

    assert secret not in " ".join(captured["command"])
    private_request = json.loads(captured["input"].decode("utf-8"))
    assert private_request["api_key"] == secret
    assert captured["timeout"] == 7
    assert captured["stdout"] is subprocess.PIPE
    assert captured["stderr"] is subprocess.PIPE


def test_serial_review_persists_success_before_later_process_interruption(
    tmp_path: Path,
):
    assignment, template = _public_pass(tmp_path)
    calls = 0

    def interrupted_transport(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return _response()
        raise KeyboardInterrupt("simulated process interruption")

    output = tmp_path / "interrupted"
    with pytest.raises(KeyboardInterrupt, match="simulated process interruption"):
        review.run_review(
            assignment_path=assignment,
            template_path=template,
            output_root=output,
            dry_run=False,
            start=0,
            limit=2,
            workers=1,
            model="test-vlm",
            temperature=0.0,
            max_tokens=500,
            timeout=30,
            base_url="https://unit.test/v1",
            api_key="secret-not-persisted",
            transport=interrupted_transport,
        )
    assert (output / "checkpoints/a0000.json").is_file()
    assert (output / "raw_responses/a0000.json").is_file()
    assert (output / ".incomplete").is_file()
    assert not (output / "run_summary.json").exists()


def test_infrastructure_error_replay_and_checkpoint_merge(tmp_path: Path):
    assignment, template = _public_pass(tmp_path)

    def first_transport(_url, _key, payload, _timeout):
        prompt = payload["messages"][0]["content"][0]["text"]
        if "Requested source-free target scene: source-free target scene 17\n" in prompt:
            return _response({"bad": "schema"})
        return _response()

    first_root = tmp_path / "first"
    first = review.run_review(
        assignment_path=assignment,
        template_path=template,
        output_root=first_root,
        dry_run=False,
        start=0,
        limit=None,
        workers=4,
        model="test-vlm",
        temperature=0.0,
        max_tokens=500,
        timeout=30,
        base_url="https://unit.test/v1",
        api_key="secret-not-persisted",
        transport=first_transport,
    )
    assert first["successful_checkpoints"] == 1151
    assert first["infrastructure_errors"] == 1
    assert not (first_root / "checkpoints/a0017.json").exists()
    error = json.loads((first_root / "errors/a0017.json").read_text())
    assert error["status"] == review.INFRASTRUCTURE_ERROR_STATUS
    assert not set(screening.SCORE_FIELDS) & set(error)
    assert "secret-not-persisted" not in json.dumps(error)
    assert error["raw_response"] is not None
    error_raw = first_root / error["raw_response"]["path"]
    assert json.loads(error_raw.read_text())["choices"][0]["message"]["content"]

    with pytest.raises(review.ReviewTransportError, match="missing 1"):
        review.merge_checkpoints(
            assignment_path=assignment,
            template_path=template,
            run_roots=[first_root],
            output_root=tmp_path / "must_not_publish",
        )
    assert not (tmp_path / "must_not_publish").exists()

    transport_error_root = tmp_path / "transport_error"

    def transport_failure(*_args, **_kwargs):
        raise TimeoutError("simulated timeout transport-secret")

    transport_error = review.run_review(
        assignment_path=assignment,
        template_path=template,
        output_root=transport_error_root,
        dry_run=False,
        start=18,
        limit=1,
        workers=1,
        model="test-vlm",
        temperature=0.0,
        max_tokens=500,
        timeout=30,
        base_url="https://unit.test/v1",
        api_key="transport-secret",
        transport=transport_failure,
    )
    assert transport_error["successful_checkpoints"] == 0
    timeout_error = json.loads(
        (transport_error_root / "errors/a0018.json").read_text()
    )
    assert timeout_error["raw_response"] is None
    assert timeout_error["model_content_sha256"] is None
    assert "transport-secret" not in json.dumps(timeout_error)
    assert not list((transport_error_root / "raw_responses").iterdir())

    retry_root = tmp_path / "retry"
    retry = review.run_review(
        assignment_path=assignment,
        template_path=template,
        output_root=retry_root,
        dry_run=False,
        start=17,
        limit=1,
        workers=1,
        model="test-vlm",
        temperature=0.0,
        max_tokens=500,
        timeout=30,
        base_url="https://unit.test/v1",
        api_key="retry-secret",
        transport=lambda *_args, **_kwargs: _response(),
    )
    assert retry["successful_checkpoints"] == 1
    assert retry["infrastructure_errors"] == 0

    merged_root = tmp_path / "merged"
    merged = review.merge_checkpoints(
        assignment_path=assignment,
        template_path=template,
        run_roots=[first_root, retry_root],
        output_root=merged_root,
    )
    assert merged["status"] == "complete_schema_valid_public_pass_review"
    assert merged["row_count"] == 1152
    fields, rows = _read_csv(merged_root / "completed_review.csv")
    assert tuple(fields) == screening.SCORE_TEMPLATE_FIELDS
    assert len(rows) == 1152
    assert all(row[field] == "2" for row in rows for field in screening.SCORE_FIELDS)
    notes = json.loads(rows[17]["reviewer_notes"])
    assert notes["confidence"]["quality"] == 0.9
    assert notes["frame_evidence"]["quality"][0]["frame_index"] == 24
