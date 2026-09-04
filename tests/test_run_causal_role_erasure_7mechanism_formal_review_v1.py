from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path

import pytest
from PIL import Image

from scripts import review_causal_role_erasure_7mechanism_formal_v2 as transport
from scripts import run_causal_role_erasure_7mechanism_formal_review_v1 as launch


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(transport.canonical_json_bytes(row) for row in rows))


def _public_copy(tmp_path: Path, count: int = 2) -> Path:
    public = tmp_path / "copied_package" / "public"
    assignments = []
    for index in range(count):
        kind = "causal" if index % 2 == 0 else "specificity"
        review_id = f"rv_{index:032x}"
        subtype = "" if kind == "causal" else "same_footprint_alternative_cause"
        assignment = {
            "anonymous_review_id": review_id,
            "case_kind": kind,
            "mechanism_name": f"mechanism {index}",
            "prompt": f"scene {index}",
            "source_object": f"source {index}",
            "receiver": f"receiver {index}",
            "expected_trigger": f"trigger {index}",
            "expected_footprint": f"footprint {index}",
            "expected_counterfactual_state": f"counterfactual {index}",
            "protected_object": "" if kind == "causal" else f"protected {index}",
            "specificity_subtype": subtype,
            "acceptable_alternative_cause": "" if kind == "causal" else f"alternative {index}",
            "composite_path": f"media/{review_id}.jpg",
            "composite_sha256": "",
            "assignment_sha256": "",
        }
        assignments.append(assignment)
    orders = {"pass_a": assignments, "pass_b": list(reversed(assignments))}
    for pass_name, ordered in orders.items():
        root = public / pass_name
        (root / "media").mkdir(parents=True)
        finalized = []
        scores = []
        for index, source in enumerate(ordered):
            row = dict(source)
            image = root / row["composite_path"]
            identity_color = int(row["anonymous_review_id"].split("_")[-1], 16) * 30
            Image.new(
                "RGB",
                (transport.COMPOSITE_WIDTH, transport.COMPOSITE_HEIGHT),
                color=(identity_color, 20, 10),
            ).save(image, format="JPEG", quality=transport.JPEG_QUALITY)
            row["composite_sha256"] = transport.sha256_file(image)
            row["assignment_sha256"] = transport._assignment_digest(row)
            fields = transport.SCORE_FIELDS_BY_KIND[row["case_kind"]]
            finalized.append(row)
            scores.append(
                {
                    "anonymous_review_id": row["anonymous_review_id"],
                    "assignment_sha256": row["assignment_sha256"],
                    "case_kind": row["case_kind"],
                    "scores": {field: None for field in fields},
                    "confidence": {field: None for field in fields},
                    "evidence_frames": {field: [] for field in fields},
                    "evidence_observations": {field: [] for field in fields},
                    "unusable_reason": "",
                    "status": "pending",
                }
            )
        assignments_path = root / "assignments.jsonl"
        scores_path = root / "scores.jsonl"
        _jsonl(assignments_path, finalized)
        _jsonl(scores_path, scores)
        manifest = {
            "protocol": transport.PACKAGE_PROTOCOL,
            "schema_version": 1,
            "pass_id": transport.PASS_IDS[pass_name],
            "item_count": count,
            "panel_windows": transport.PANEL_WINDOWS,
            "assignments": {
                "path": "assignments.jsonl",
                "sha256": transport.sha256_file(assignments_path),
            },
            "blank_scores": {
                "path": "scores.jsonl",
                "sha256": transport.sha256_file(scores_path),
            },
            "composite_contract": {
                "path_base": "pass_root",
                "directory": "media",
                "format": "jpeg",
                "width": transport.COMPOSITE_WIDTH,
                "height": transport.COMPOSITE_HEIGHT,
                "tile_width": transport.TILE_WIDTH,
                "tile_height": transport.TILE_HEIGHT,
                "tile_fit": "deterministic_center_crop_no_letterbox",
                "resampling": "Pillow.Image.Resampling.LANCZOS",
                "jpeg_quality": transport.JPEG_QUALITY,
                "max_bytes": transport.MAX_IMAGE_BYTES,
                "one_image_per_assignment": True,
            },
            "ordering_commitment": str(index) * 64,
        }
        (root / "pass_manifest.json").write_bytes(
            transport.canonical_json_bytes(manifest)
        )
    commitments = {
        "protocol": transport.PACKAGE_PROTOCOL,
        "schema_version": 1,
        "commitment_scheme": "sha256(canonical-jsonl-bytes)",
        "tier_0_audit_strata": {"row_count": count, "sha256": "0" * 64},
        "tier_1_original_only": {"row_count": 588, "sha256": "1" * 64},
        "tier_2_full": {"row_count": count, "sha256": "2" * 64},
        "generation_ledger": {"row_count": count, "sha256": "3" * 64},
        "blind_key_sha256": "4" * 64,
    }
    code_registry = launch.evaluation_code.build_registry(PROJECT_ROOT)
    code_registry_path = public / "evaluation_code_registry.json"
    code_registry_path.write_bytes(transport.canonical_json_bytes(code_registry))
    commitments["evaluation_code_registry"] = {
        "path": "evaluation_code_registry.json",
        "sha256": transport.sha256_file(code_registry_path),
        "registry_sha256": code_registry["registry_sha256"],
    }
    (public / "key_commitments.json").write_bytes(
        transport.canonical_json_bytes(commitments)
    )
    return public


def _minimal_snapshot(root: Path) -> tuple[Path, Path, Path]:
    authority = root / "authority"
    wan = root / "wan"
    formal = authority / launch.FORMAL_CASES_REL
    identification = authority / launch.IDENTIFICATION_REL
    formal.parent.mkdir(parents=True)
    formal.write_text("formal\n")
    identification.write_text("identification\n")
    remote_video = launch.REMOTE_PROJECT_ROOT / "outputs/generated/example.mp4"
    local_video = authority / remote_video.relative_to(launch.REMOTE_PROJECT_ROOT)
    local_video.parent.mkdir(parents=True)
    local_video.write_bytes(b"video-bytes")
    manifest = {
        "inputs": {
            "formal_cases": str(launch.REMOTE_PROJECT_ROOT / launch.FORMAL_CASES_REL),
            "identification_subset": str(
                launch.REMOTE_PROJECT_ROOT / launch.IDENTIFICATION_REL
            ),
        },
        "items": [
            {
                "video_path": str(remote_video),
                "size_bytes": local_video.stat().st_size,
                "video_sha256": transport.sha256_file(local_video),
            }
        ],
    }
    manifest_path = authority / "source.json"
    manifest_path.write_text(json.dumps(manifest))
    return authority, wan, manifest_path


def test_dry_plan_is_32_total_concurrency_and_never_reads_key(tmp_path: Path, monkeypatch):
    public = _public_copy(tmp_path)
    launch_root = tmp_path / "must_not_be_created"

    def forbidden(_path):
        raise AssertionError("dry plan read Copilot key")

    monkeypatch.setattr(launch, "read_copilot_key", forbidden)
    plan = launch.build_dry_run_plan(
        authority_snapshot=None,
        wan_original_snapshot=None,
        review_package_public=public,
        launch_root=launch_root,
        shard_size=256,
        proxy_version="copilot-test-v1",
    )
    assert plan["status"] == "dry_run_ready_no_api_calls"
    assert plan["source_mode"] == "copied_public_package"
    assert plan["transport"]["workers_per_pass"] == 16
    assert plan["transport"]["maximum_total_concurrency"] == 32
    assert plan["transport"]["endpoint"] == launch.LOCAL_RESPONSES_ENDPOINT
    assert plan["smoke"] == {"start": 0, "limit": 10, "both_passes": True}
    assert sum(int(row["limit"]) for row in plan["full_shards"]) == 2438
    assert plan["primary_requests_total"] == 4896
    assert plan["api_calls"] == 0
    assert not launch_root.exists()


def test_prepare_package_cli_breaks_snapshot_preflight_bootstrap_cycle(
    tmp_path: Path, monkeypatch, capsys
):
    workspace = tmp_path / "workspace"
    authority = workspace / "authority"
    wan = workspace / "wan"
    launch_root = workspace / "launch"
    for path in (workspace, authority, wan):
        path.mkdir(exist_ok=True)
    package_root = workspace / "prepared" / "review_package"
    public_root = package_root / "public"
    public_root.mkdir(parents=True)
    calls = []

    def fake_prepare(**kwargs):
        calls.append(kwargs)
        return package_root

    def fake_validate(path, expected_items=launch.TOTAL_ITEMS_PER_PASS):
        assert path == public_root
        return public_root, {"status": "validated", "item_count_per_pass": expected_items}

    monkeypatch.setattr(launch, "prepare_review_package", fake_prepare)
    monkeypatch.setattr(launch, "validate_public_package_copy", fake_validate)
    assert launch.main(
        [
            "--project-root",
            str(workspace),
            "--workspace-root",
            str(workspace),
            "--authority-snapshot",
            str(authority),
            "--wan-original-snapshot",
            str(wan),
            "--launch-root",
            str(launch_root),
            "--prepare-package",
        ]
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "review_package_prepared_no_api_calls"
    assert payload["public_root"] == str(public_root)
    assert payload["api_calls"] == 0
    assert len(calls) == 1
    assert calls[0]["workspace_root"] == workspace.resolve()


def test_copied_public_package_has_independent_equal_blind_inventories(tmp_path: Path):
    public = _public_copy(tmp_path)
    resolved, inventory = launch.validate_public_package_copy(public, expected_items=2)
    assert resolved == public.resolve()
    assert inventory["status"] == "validated_copied_public_only_package"
    assert inventory["item_count_per_pass"] == 2
    assert inventory["orders_independent"] is True
    assert inventory["private_files_opened"] == 0

    extra = public / "method_map.json"
    extra.write_text("{}")
    with pytest.raises(launch.FormalLaunchError, match="inventory differs"):
        launch.validate_public_package_copy(public, expected_items=2)


def test_rebound_manifest_changes_only_authority_paths_and_preserves_source(tmp_path: Path):
    authority, _wan, manifest_path = _minimal_snapshot(tmp_path)
    source_bytes = manifest_path.read_bytes()
    rebound, receipt = launch.rebound_manifest(
        source_manifest=manifest_path,
        source_snapshot=authority,
        authority_snapshot=authority,
        expected_items=1,
    )
    assert manifest_path.read_bytes() == source_bytes
    assert receipt["status"] == "path_rebound_view_only_original_unchanged"
    assert receipt["rewrite_count"] == 3
    assert rebound["items"][0]["video_path"].startswith(str(authority.resolve()))
    assert rebound["inputs"]["formal_cases"] == str(
        (authority / launch.FORMAL_CASES_REL).resolve()
    )
    assert rebound["inputs"]["identification_subset"] == str(
        (authority / launch.IDENTIFICATION_REL).resolve()
    )


def test_fresh_retry_uses_new_root_and_only_latest_error_source(tmp_path: Path):
    assignments = tmp_path / "pass_a" / "assignments.jsonl"
    scores = tmp_path / "pass_a" / "scores.jsonl"
    assignments.parent.mkdir()
    assignments.write_text("placeholder")
    scores.write_text("placeholder")
    calls = []

    def fake_runner(**kwargs):
        calls.append(kwargs)
        output = kwargs["output_root"]
        output.mkdir(parents=True)
        failed = len(calls) == 1
        summary = {
            "workflow_version": transport.WORKFLOW_VERSION,
            "status": (
                "completed_with_infrastructure_errors_requires_fresh_retry"
                if failed
                else "completed_checkpoint_shard"
            ),
            "infrastructure_errors": 1 if failed else 0,
            "scientific_zero_fallbacks": 0,
        }
        (output / "run_summary.json").write_text(json.dumps(summary))
        return summary

    roots = launch._complete_stage_with_retries(
        stage_root=tmp_path / "runs" / "pass_a" / "smoke",
        assignments_path=assignments,
        blank_scores_path=scores,
        api_key="secret-in-memory",
        proxy_version="proxy-test",
        timeout=20,
        start=0,
        limit=10,
        max_retry_rounds=2,
        review_runner=fake_runner,
    )
    assert len(roots) == 2
    assert calls[0]["start"] == 0 and calls[0]["limit"] == 10
    assert calls[0]["retry_run_roots"] == ()
    assert calls[1]["retry_run_roots"] == [roots[0]]
    assert calls[1]["output_root"] != calls[0]["output_root"]


def test_formal_execution_keeps_key_out_of_argv_env_and_artifacts(
    tmp_path: Path, monkeypatch
):
    public = tmp_path / "remote_built_package" / "public"
    for pass_name in transport.PASS_NAMES:
        root = public / pass_name
        root.mkdir(parents=True)
        (root / "assignments.jsonl").write_text("public")
        (root / "scores.jsonl").write_text("public")
    secret = "copilot-secret-never-persist"
    key_file = tmp_path / "copilot.key"
    key_file.write_text("key: " + secret)
    launch_root = tmp_path / "formal_launch"
    preflight_receipt = tmp_path / "preflight_receipt.json"
    preflight_receipt.write_text("{}")
    calls = []
    barrier = threading.Barrier(2)

    monkeypatch.setattr(
        launch,
        "validate_public_package_copy",
        lambda _path: (
            public,
                {
                    "status": "validated_copied_public_only_package",
                    "private_files_opened": 0,
                    "evaluation_code_registry_sha256": "4" * 64,
                },
        ),
    )
    monkeypatch.setattr(
        launch,
        "validate_preflight_receipt",
        lambda **_kwargs: {
            "proxy": {"luna": {"metadata_sha256": "5" * 64}}
        },
    )

    def fake_runner(**kwargs):
        assert kwargs["api_key"] == secret
        assert kwargs["base_url"] == launch.LOCAL_RESPONSES_BASE_URL
        assert kwargs["workers"] == 16
        assert kwargs["dry_run"] is False
        barrier.wait(timeout=5)
        output = kwargs["output_root"]
        output.mkdir(parents=True)
        summary = {
            "workflow_version": transport.WORKFLOW_VERSION,
            "status": "completed_checkpoint_shard",
            "infrastructure_errors": 0,
            "scientific_zero_fallbacks": 0,
        }
        (output / "run_summary.json").write_text(json.dumps(summary))
        calls.append(kwargs)
        return summary

    def fake_merger(**kwargs):
        output = kwargs["output_root"]
        output.mkdir(parents=True)
        result = {
            "workflow_version": transport.WORKFLOW_VERSION,
            "status": "complete_schema_valid_public_pass_review",
            "row_count": 2448,
        }
        (output / "merge_manifest.json").write_text(json.dumps(result))
        return result

    argv_before = list(sys.argv)
    env_before = dict(os.environ)
    result = launch.execute_formal_review(
        workspace_root=None,
        authority_snapshot=None,
        wan_original_snapshot=None,
        review_package_public=public,
        launch_root=launch_root,
        copilot_key_file=key_file,
        preflight_receipt=preflight_receipt,
        proxy_version="proxy-test-v1",
        shard_size=1000,
        timeout=20,
        max_retry_rounds=1,
        review_runner=fake_runner,
        merger=fake_merger,
    )
    assert result["status"] == "completed_two_independent_schema_valid_passes"
    assert result["maximum_total_concurrency"] == 32
    assert len(calls) == 8  # smoke plus three full shards, independently for A/B
    assert sys.argv == argv_before
    assert dict(os.environ) == env_before
    for path in launch_root.rglob("*"):
        if path.is_file():
            assert secret not in path.read_text(encoding="utf-8")
    assert secret not in json.dumps(result)


def test_key_reader_rejects_symlink_and_accepts_raw_json_or_colon(tmp_path: Path):
    raw = tmp_path / "raw.key"
    raw.write_text("raw-secret")
    assert launch.read_copilot_key(raw) == "raw-secret"
    structured = tmp_path / "structured.key"
    structured.write_text('{"api_key":"json-secret"}')
    assert launch.read_copilot_key(structured) == "json-secret"
    colon = tmp_path / "colon.key"
    colon.write_text("url: ignored\nkey: colon-secret\n")
    assert launch.read_copilot_key(colon) == "colon-secret"
    link = tmp_path / "link.key"
    link.symlink_to(raw)
    with pytest.raises(launch.FormalLaunchError, match="symlinked"):
        launch.read_copilot_key(link)


def test_luna_metadata_requires_responses_vision_structured_and_low_reasoning():
    payload = {
        "data": [
            {
                "id": transport.MODEL,
                "supported_endpoints": ["/responses"],
                "capabilities": {
                    "vision": {
                        "max_images": 1,
                        "max_image_size": transport.MAX_IMAGE_BYTES,
                    },
                    "structured_outputs": True,
                    "reasoning_efforts": ["low", "medium"],
                },
            }
        ]
    }
    validated = launch.validate_luna_metadata(payload)
    assert validated["responses_supported"] is True
    assert validated["vision_supported"] is True
    assert validated["structured_outputs"] is True
    assert len(validated["metadata_sha256"]) == 64
    payload["data"][0]["capabilities"]["structured_outputs"] = False
    with pytest.raises(launch.FormalLaunchError, match="structured outputs"):
        launch.validate_luna_metadata(payload)


def test_preflight_records_health_metadata_success_rate_and_latency_without_key(
    tmp_path: Path, monkeypatch
):
    public = tmp_path / "public"
    pass_root = public / "pass_a"
    pass_root.mkdir(parents=True)
    (pass_root / "assignments.jsonl").write_text("public")
    (pass_root / "scores.jsonl").write_text("public")
    secret = "preflight-secret-never-persist"
    key_file = tmp_path / "copilot.key"
    key_file.write_text(secret)
    inventory = {
        "semantic_inventory_sha256": "6" * 64,
        "pass_manifests": {"pass_a": {"sha256": "7" * 64}},
        "key_commitments": {"sha256": "8" * 64},
        "evaluation_code_registry_sha256": "5" * 64,
    }
    monkeypatch.setattr(
        launch,
        "validate_public_package_copy",
        lambda _path, expected_items: (public, inventory),
    )

    def fake_probe(base_url, api_key, timeout):
        assert base_url == launch.LOCAL_RESPONSES_BASE_URL
        assert api_key == secret and timeout == 30
        return {
            "health": {
                "status": "healthy_via_health_endpoint",
                "http_status": 200,
                "observed_version": "proxy-test-v1",
            },
            "models_http_status": 200,
            "models_latency_ms": 1.0,
            "models_body_sha256": "9" * 64,
            "luna": {
                "model": transport.MODEL,
                "metadata_sha256": "a" * 64,
                "responses_supported": True,
                "vision_supported": True,
                "max_images": 1,
                "max_image_size": transport.MAX_IMAGE_BYTES,
                "structured_outputs": True,
                "low_reasoning_supported": True,
            },
        }

    def fake_api(url, api_key, payload, timeout):
        assert url == launch.LOCAL_RESPONSES_ENDPOINT
        assert api_key == secret and timeout == 30
        assert payload == {"synthetic": True}
        return {"model": transport.MODEL, "status": "completed", "output": []}

    def fake_runner(**kwargs):
        assert kwargs["workers"] == 4
        assert kwargs["start"] == 1 and kwargs["limit"] == 3
        for _ in range(kwargs["limit"]):
            kwargs["transport"](
                launch.LOCAL_RESPONSES_ENDPOINT,
                kwargs["api_key"],
                {"synthetic": True},
                kwargs["timeout"],
            )
        output = kwargs["output_root"]
        output.mkdir(parents=True)
        summary = {
            "successful_checkpoints": kwargs["limit"],
            "infrastructure_errors": 0,
            "scientific_zero_fallbacks": 0,
        }
        (output / "run_summary.json").write_text(json.dumps(summary))
        return summary

    output = tmp_path / "preflight"
    receipt = launch.run_preflight(
        review_package_public=public,
        output_root=output,
        copilot_key_file=key_file,
        proxy_version="proxy-test-v1",
        pass_name="pass_a",
        start=1,
        limit=3,
        workers=4,
        timeout=30,
        expected_items=8,
        proxy_probe=fake_probe,
        review_runner=fake_runner,
        api_transport=fake_api,
    )
    assert receipt["status"] == launch.PREFLIGHT_STATUS
    assert receipt["successful"] == receipt["attempted"] == 3
    assert receipt["success_rate"] == 1.0
    assert receipt["http_status_counts"] == {"200": 3}
    assert receipt["proxy"]["luna"]["metadata_sha256"] == "a" * 64
    assert receipt["credential_persisted"] is False
    assert secret not in (output / "preflight_receipt.json").read_text()
    validated = launch.validate_preflight_receipt(
        path=output / "preflight_receipt.json",
        public_inventory=inventory,
        proxy_version="proxy-test-v1",
    )
    assert validated["status"] == launch.PREFLIGHT_STATUS
