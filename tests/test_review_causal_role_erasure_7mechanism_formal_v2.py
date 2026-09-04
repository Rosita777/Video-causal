from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest
from PIL import Image

from scripts import review_causal_role_erasure_7mechanism_formal_v2 as formal


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(formal.canonical_json_bytes(row) for row in rows))


def _package(tmp_path: Path, count: int = 4, pass_name: str = "pass_a"):
    root = tmp_path / "review_package" / "public" / pass_name
    media = root / "media"
    media.mkdir(parents=True)
    assignments = []
    blanks = []
    for index in range(count):
        kind = "causal" if index % 2 == 0 else "specificity"
        review_id = f"anon_{index:04d}"
        image = media / f"{review_id}.jpg"
        Image.new(
            "RGB",
            (formal.COMPOSITE_WIDTH, formal.COMPOSITE_HEIGHT),
            color=(index * 20, 10, 30),
        ).save(image, format="JPEG", quality=formal.JPEG_QUALITY)
        assignment = {
            "anonymous_review_id": review_id,
            "case_kind": kind,
            "mechanism_name": f"mechanism {index}",
            "prompt": f"requested scene {index}",
            "source_object": f"source {index}",
            "receiver": f"receiver {index}",
            "expected_footprint": f"footprint {index}",
            "expected_counterfactual_state": f"counterfactual {index}",
            "composite_path": f"media/{review_id}.jpg",
            "composite_sha256": formal.sha256_file(image),
            "assignment_sha256": "",
        }
        assignment["assignment_sha256"] = formal._assignment_digest(assignment)
        fields = formal.SCORE_FIELDS_BY_KIND[kind]
        blank = {
            "anonymous_review_id": review_id,
            "assignment_sha256": assignment["assignment_sha256"],
            "case_kind": kind,
            "scores": {field: None for field in fields},
            "confidence": {field: None for field in fields},
            "evidence_frames": {field: [] for field in fields},
            "evidence_observations": {field: [] for field in fields},
            "unusable_reason": "",
            "status": "pending",
        }
        assignments.append(assignment)
        blanks.append(blank)
    assignments_path = root / "assignments.jsonl"
    scores_path = root / "scores.jsonl"
    _write_jsonl(assignments_path, assignments)
    _write_jsonl(scores_path, blanks)
    manifest = {
        "protocol": formal.PACKAGE_PROTOCOL,
        "schema_version": 1,
        "pass_id": formal.PASS_IDS[pass_name],
        "item_count": count,
        "assignments": {
            "path": "assignments.jsonl",
            "sha256": formal.sha256_file(assignments_path),
        },
        "blank_scores": {
            "path": "scores.jsonl",
            "sha256": formal.sha256_file(scores_path),
        },
        "panel_windows": formal.PANEL_WINDOWS,
        "composite_contract": {
            "path_base": "pass_root",
            "directory": "media",
            "format": "jpeg",
            "width": formal.COMPOSITE_WIDTH,
            "height": formal.COMPOSITE_HEIGHT,
            "jpeg_quality": formal.JPEG_QUALITY,
            "max_bytes": formal.MAX_IMAGE_BYTES,
            "one_image_per_assignment": True,
        },
    }
    (root / "pass_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
    )
    return assignments_path, scores_path, assignments, blanks


def _model_object(kind: str, score: int = 1):
    fields = formal.SCORE_FIELDS_BY_KIND[kind]
    return {
        **{field: score for field in fields},
        "confidence": {field: 0.9 for field in fields},
        "frame_evidence": {
            field: [
                {
                    "frame_index": offset + 10,
                    "observation": f"visible evidence for {field}",
                }
            ]
            for offset, field in enumerate(fields)
        },
    }


def _response(kind: str, score: int = 1):
    return {
        "id": "resp_unit",
        "status": "completed",
        "output": [
            {"type": "reasoning", "id": "reasoning_unit", "summary": []},
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": json.dumps(_model_object(kind, score)),
                        "annotations": [],
                    }
                ],
            },
        ],
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }


def _rows(assignments: Path, scores: Path, count: int = 4):
    return formal.load_public_pass(assignments, scores, expected_items=count)[1]


def test_responses_payload_is_single_image_stateless_strict_and_dynamic(tmp_path: Path):
    assignments, scores, *_ = _package(tmp_path)
    rows = _rows(assignments, scores)
    causal = formal.request_payload_for(rows[0])
    specificity = formal.request_payload_for(rows[1])

    assert causal["model"] == "gpt-5.6-luna"
    assert causal["reasoning"] == {"effort": "low"}
    assert causal["temperature"] == 1.0
    assert causal["max_output_tokens"] == 1600
    assert causal["store"] is False
    assert [part["type"] for part in causal["input"][0]["content"]] == [
        "input_text",
        "input_image",
    ]
    assert causal["input"][0]["content"][1]["image_url"].startswith(
        "data:image/jpeg;base64,"
    )
    causal_format = causal["text"]["format"]
    assert causal_format["type"] == "json_schema"
    assert causal_format["strict"] is True
    assert causal_format["schema"] == formal.response_json_schema("causal")
    assert "source_visibility" in causal_format["schema"]["properties"]
    assert "protected_object_visibility" not in causal_format["schema"]["properties"]
    assert specificity["text"]["format"]["schema"] == formal.response_json_schema(
        "specificity"
    )
    assert formal.responses_endpoint("http://127.0.0.1:4141") == (
        "http://127.0.0.1:4141/v1/responses"
    )


def test_public_loader_rejects_cross_pass_tamper_and_nonblank_scores(tmp_path: Path):
    assignments, scores, assignment_rows, blanks = _package(tmp_path)
    formal.load_public_pass(assignments, scores, expected_items=4)

    blanks[0]["scores"]["source_visibility"] = 0
    _write_jsonl(scores, blanks)
    manifest_path = assignments.parent / "pass_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["blank_scores"]["sha256"] = formal.sha256_file(scores)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(formal.FormalReviewTransportError, match="contains a value"):
        formal.load_public_pass(assignments, scores, expected_items=4)

    assignments2, scores2, assignment_rows2, blanks2 = _package(
        tmp_path / "second", count=4
    )
    outside = assignments2.parent.parent / "pass_b" / "media" / "other.jpg"
    outside.parent.mkdir(parents=True)
    outside.write_bytes(b"other")
    assignment_rows2[0]["composite_path"] = "../pass_b/media/other.jpg"
    assignment_rows2[0]["composite_sha256"] = formal.sha256_file(outside)
    assignment_rows2[0]["assignment_sha256"] = formal._assignment_digest(
        assignment_rows2[0]
    )
    blanks2[0]["assignment_sha256"] = assignment_rows2[0]["assignment_sha256"]
    _write_jsonl(assignments2, assignment_rows2)
    _write_jsonl(scores2, blanks2)
    manifest2_path = assignments2.parent / "pass_manifest.json"
    manifest2 = json.loads(manifest2_path.read_text())
    manifest2["assignments"]["sha256"] = formal.sha256_file(assignments2)
    manifest2["blank_scores"]["sha256"] = formal.sha256_file(scores2)
    manifest2_path.write_text(json.dumps(manifest2))
    with pytest.raises(formal.FormalReviewTransportError, match="escaped this pass"):
        formal.load_public_pass(assignments2, scores2, expected_items=4)


def test_strict_normalizer_enforces_case_fields_and_per_field_evidence():
    normalized = formal.normalize_model_object(_model_object("causal"), "causal")
    assert normalized["scores"]["source_visibility"] == 1
    assert normalized["confidence"]["video_quality"] == 0.9
    assert normalized["evidence_frames"]["footprint_visibility"] == [11]

    wrong = _model_object("specificity")
    with pytest.raises(formal.FormalReviewTransportError, match="keys differ"):
        formal.normalize_model_object(wrong, "causal")
    duplicate = _model_object("causal")
    duplicate["frame_evidence"]["source_visibility"].append(
        {"frame_index": 10, "observation": "duplicate frame"}
    )
    with pytest.raises(formal.FormalReviewTransportError, match="duplicate evidence"):
        formal.normalize_model_object(duplicate, "causal")
    bad_confidence = _model_object("causal")
    bad_confidence["confidence"]["video_quality"] = 1.1
    with pytest.raises(formal.FormalReviewTransportError, match="invalid confidence"):
        formal.normalize_model_object(bad_confidence, "causal")


def test_dry_shard_registers_contract_without_calling_transport(tmp_path: Path):
    assignments, scores, *_ = _package(tmp_path)
    output = tmp_path / "dry"

    def forbidden(*_args, **_kwargs):
        raise AssertionError("dry-run called transport")

    summary = formal.run_review(
        assignments_path=assignments,
        blank_scores_path=scores,
        output_root=output,
        dry_run=True,
        start=1,
        limit=2,
        workers=16,
        expected_items=4,
        transport=forbidden,
    )
    assert summary["status"] == "dry_run_payloads_only_no_api_calls"
    assert summary["selected_count"] == 2
    registration = json.loads((output / "run_registration.json").read_text())
    assert registration["endpoint"] == "http://127.0.0.1:4141/v1/responses"
    assert registration["selected_review_ids"] == ["anon_0001", "anon_0002"]
    assert registration["workers"] == 16
    assert registration["store"] is False
    assert "api_key" not in registration
    payloads = [json.loads(line) for line in (output / "payloads.jsonl").read_text().splitlines()]
    assert len(payloads) == 2
    assert not list((output / "checkpoints").iterdir())
    with pytest.raises(formal.FormalReviewTransportError, match="fresh-only"):
        formal.run_review(
            assignments_path=assignments,
            blank_scores_path=scores,
            output_root=output,
            dry_run=True,
            limit=1,
            expected_items=4,
        )


def test_api_failures_are_infrastructure_only_and_concurrency_is_used(tmp_path: Path):
    assignments, scores, *_ = _package(tmp_path)
    output = tmp_path / "run"
    lock = threading.Lock()
    seen = []

    def transport(url, _key, payload, _timeout):
        kind = (
            "causal"
            if "source_visibility" in payload["text"]["format"]["schema"]["properties"]
            else "specificity"
        )
        with lock:
            seen.append((url, kind, payload["store"]))
            call = len(seen)
        if call == 2:
            raise TimeoutError("simulated timeout")
        return _response(kind)

    summary = formal.run_review(
        assignments_path=assignments,
        blank_scores_path=scores,
        output_root=output,
        dry_run=False,
        workers=4,
        expected_items=4,
        transport=transport,
    )
    assert summary["successful_checkpoints"] == 3
    assert summary["infrastructure_errors"] == 1
    assert summary["scientific_zero_fallbacks"] == 0
    assert len(seen) == 4
    error = json.loads(next((output / "errors").iterdir()).read_text())
    assert error["status"] == formal.INFRASTRUCTURE_ERROR_STATUS
    assert not ({*formal.CAUSAL_FIELDS, *formal.SPECIFICITY_FIELDS} & set(error))
    assert "simulated timeout" in error["error_message"]


def test_fresh_retry_and_merge_complete_the_pass(tmp_path: Path):
    assignments, scores, *_ = _package(tmp_path)
    first = tmp_path / "first"
    counter = {"value": 0}

    def initially_flaky(_url, _key, payload, _timeout):
        counter["value"] += 1
        kind = (
            "causal"
            if "source_visibility" in payload["text"]["format"]["schema"]["properties"]
            else "specificity"
        )
        if counter["value"] == 3:
            raise RuntimeError("429 rate limited")
        return _response(kind, score=2)

    first_summary = formal.run_review(
        assignments_path=assignments,
        blank_scores_path=scores,
        output_root=first,
        dry_run=False,
        workers=1,
        expected_items=4,
        transport=initially_flaky,
    )
    assert first_summary["infrastructure_errors"] == 1
    failed_id = json.loads(next((first / "errors").iterdir()).read_text())[
        "anonymous_review_id"
    ]

    retried_ids = []

    def succeeds(_url, _key, payload, _timeout):
        prompt = payload["input"][0]["content"][0]["text"]
        retried_ids.append(prompt)
        kind = (
            "causal"
            if "source_visibility" in payload["text"]["format"]["schema"]["properties"]
            else "specificity"
        )
        return _response(kind, score=1)

    retry = tmp_path / "retry"
    retry_summary = formal.run_review(
        assignments_path=assignments,
        blank_scores_path=scores,
        output_root=retry,
        dry_run=False,
        workers=2,
        retry_run_roots=[first],
        expected_items=4,
        transport=succeeds,
    )
    assert retry_summary["selected_count"] == 1
    assert retry_summary["successful_checkpoints"] == 1
    assert len(retried_ids) == 1
    retry_registration = json.loads((retry / "run_registration.json").read_text())
    assert retry_registration["selected_review_ids"] == [failed_id]
    assert retry_registration["selection_mode"] == (
        "fresh_retry_of_infrastructure_errors"
    )

    merged = tmp_path / "merged"
    manifest = formal.merge_checkpoints(
        assignments_path=assignments,
        blank_scores_path=scores,
        run_roots=[first, retry],
        output_root=merged,
        expected_items=4,
    )
    assert manifest["status"] == "complete_schema_valid_public_pass_review"
    assert manifest["row_count"] == 4
    completed = [
        json.loads(line)
        for line in (merged / "completed_scores.jsonl").read_text().splitlines()
    ]
    assert [row["anonymous_review_id"] for row in completed] == [
        "anon_0000",
        "anon_0001",
        "anon_0002",
        "anon_0003",
    ]
    assert all(row["status"] == "completed" for row in completed)
    assert completed[2]["scores"]["source_visibility"] == 1


def test_incomplete_refusal_and_parse_error_are_not_scores(tmp_path: Path):
    assignments, scores, *_ = _package(tmp_path)
    row = _rows(assignments, scores)[0]

    for response in (
        {"status": "incomplete", "output": []},
        {
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "refusal", "refusal": "no"}],
                }
            ],
        },
        {
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "not json"}],
                }
            ],
        },
    ):
        result = formal.evaluate_one(
            row,
            endpoint="http://127.0.0.1:4141/v1/responses",
            api_key="secret",
            timeout=10,
            transport=lambda *_args, value=response: value,
        )
        assert result["ok"] is False
        assert "normalized" not in result
