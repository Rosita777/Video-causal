from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path

import pytest

from scripts import build_causal_role_erasure_7mechanism_review_process_receipt_v2 as process


HEADER = [
    "audit_id",
    "anonymous_review_id",
    "assignment_sha256",
    "field",
    "case_kind",
    "composite_path",
    "composite_sha256",
    *process.SCORE_FIELDS,
]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADER)
        writer.writeheader()
        writer.writerows(rows)


def fixture(tmp_path: Path):
    project = tmp_path / "project"
    project.mkdir()
    instructions = project / "docs/reviewer.md"
    instructions.parent.mkdir()
    instructions.write_text("public instructions\n", encoding="utf-8")
    frozen_builder = (
        project
        / "scripts/build_causal_role_erasure_7mechanism_review_process_receipt_v2.py"
    )
    frozen_builder.parent.mkdir()
    frozen_builder.write_bytes(Path(process.__file__).read_bytes())
    pre_metric_root = tmp_path / "pre_metric"
    pre_metric_root.mkdir()
    environment_receipt = pre_metric_root / "evaluation_environment_receipt_v1.json"
    write_json(environment_receipt, {"status": "synthetic-bound-environment"})
    pre_metric = process.with_self_digest(
        {
            "schema_version": 1,
            "protocol": process.PRE_METRIC_PROTOCOL,
            "protocol_version": process.PROTOCOL_VERSION,
            "status": process.PRE_METRIC_STATUS,
            "implementation_commit": {
                "commit": "a" * 40,
                "tree": "b" * 40,
                "worktree_clean": True,
            },
            "components": [
                {
                    "role": "review_process_receipt_builder",
                    "path": frozen_builder.relative_to(project).as_posix(),
                    "sha256": sha(frozen_builder),
                    "size_bytes": frozen_builder.stat().st_size,
                },
                {
                    "role": "reviewer_instructions",
                    "path": instructions.relative_to(project).as_posix(),
                    "sha256": sha(instructions),
                    "size_bytes": instructions.stat().st_size,
                },
            ],
            "reproducibility_bindings": {
                "evaluation_environment_receipt": {
                    "root_id": "pre_metric_root",
                    "path": environment_receipt.name,
                    "sha256": sha(environment_receipt),
                    "size_bytes": environment_receipt.stat().st_size,
                }
            },
        },
        "manifest_sha256",
    )
    pre_metric_path = pre_metric_root / "pre_metric_code_freeze_v2.json"
    pre_metric_path.write_bytes(process.canonical_json_bytes(pre_metric))
    review = tmp_path / "review"
    public = review / "audit/public"
    media = public / "media/a.jpg"
    media.parent.mkdir(parents=True)
    media.write_bytes(b"fake-jpeg")
    blank = {
        "audit_id": "audit_1",
        "anonymous_review_id": "rv_1",
        "assignment_sha256": "a" * 64,
        "field": "video_quality",
        "case_kind": "causal",
        "composite_path": "media/a.jpg",
        "composite_sha256": sha(media),
        **{field: "" for field in process.SCORE_FIELDS},
    }
    queue = public / "human_audit_queue.csv"
    write_csv(queue, [blank])
    media_manifest = public / "media_manifest.json"
    write_json(media_manifest, {"count": 1, "items": [{"path": "media/a.jpg", "sha256": sha(media), "size_bytes": media.stat().st_size}]})
    audit_manifest = review / "audit/audit_manifest.json"
    write_json(
        audit_manifest,
        {
            "protocol": process.AUDIT_PROTOCOL,
            "status": "initial_human_audit_frozen",
            "counts": {"initial_audit_atoms": 1},
            "artifacts": {
                "human_queue": {"path": "public/human_audit_queue.csv", "sha256": sha(queue), "size_bytes": queue.stat().st_size},
                "media_manifest": {"path": "public/media_manifest.json", "sha256": sha(media_manifest), "size_bytes": media_manifest.stat().st_size},
            },
        },
    )
    delivery_1 = tmp_path / "delivery_1"
    delivery_2 = tmp_path / "delivery_2"
    shutil.copytree(public, delivery_1)
    shutil.copytree(public, delivery_2)
    return project, pre_metric_path, instructions, review, public, audit_manifest, blank, delivery_1, delivery_2


def test_public_only_delivery_and_complete_review_process(tmp_path: Path):
    project, pre_metric, instructions, review, public, audit_manifest, blank, delivery_1, delivery_2 = fixture(tmp_path)
    receipt_1 = tmp_path / "process/reviewer_1_delivery.json"
    receipt_2 = tmp_path / "process/reviewer_2_delivery.json"
    process.freeze_delivery(project_root=project, pre_metric_path=pre_metric, review_root=review, audit_manifest_path=audit_manifest, public_root=public, reviewer_id="reviewer_1", reviewer_instructions=instructions, delivery_root=delivery_1, output_receipt=receipt_1)
    process.freeze_delivery(project_root=project, pre_metric_path=pre_metric, review_root=review, audit_manifest_path=audit_manifest, public_root=public, reviewer_id="reviewer_2", reviewer_instructions=instructions, delivery_root=delivery_2, output_receipt=receipt_2)

    row_1 = {**blank, "reviewer_1_score": "2", "reviewer_1_notes": "frame 12 is clear"}
    row_2 = {**blank, "reviewer_2_score": "1", "reviewer_2_notes": "ambiguous at frame 12"}
    reviewer_1 = tmp_path / "process/reviewer_1.csv"
    reviewer_2 = tmp_path / "process/reviewer_2.csv"
    completed = tmp_path / "process/completed_human.csv"
    write_csv(reviewer_1, [row_1])
    write_csv(reviewer_2, [row_2])
    write_csv(completed, [{**row_1, "reviewer_2_score": "1", "reviewer_2_notes": row_2["reviewer_2_notes"], "adjudicator_score": "2", "adjudicator_notes": "resolved from visible frames"}])
    attestation = tmp_path / "process/independence.json"
    write_json(
        attestation,
        {
            "schema_version": 1,
            "protocol": process.ATTESTATION_PROTOCOL,
            "protocol_version": process.PROTOCOL_VERSION,
            "status": process.ATTESTATION_STATUS,
            "reviewer_1_completed_sha256": sha(reviewer_1),
            "reviewer_2_completed_sha256": sha(reviewer_2),
            "reviewer_1_delivery_receipt_sha256": sha(receipt_1),
            "reviewer_2_delivery_receipt_sha256": sha(receipt_2),
            "reviewer_declarations": {
                reviewer_id: {
                    "did_not_access_peer_labels": True,
                    "did_not_access_answer_key": True,
                    "did_not_access_method_mapping": True,
                    "did_not_access_private_directory": True,
                    "did_not_access_parent_workspace": True,
                }
                for reviewer_id in ("reviewer_1", "reviewer_2")
            },
            "both_reviews_frozen_before_merge": True,
            "coordinator_merged_only_after_both_frozen": True,
        },
    )
    output = tmp_path / "process/review_process_receipt_v2.json"
    result = process.freeze_process(
        project_root=project,
        pre_metric_path=pre_metric,
        review_root=review,
        audit_manifest_path=audit_manifest,
        public_root=public,
        reviewer_instructions=instructions,
        reviewer_1_delivery_receipt=receipt_1,
        reviewer_2_delivery_receipt=receipt_2,
        reviewer_1_completed=reviewer_1,
        reviewer_2_completed=reviewer_2,
        independence_attestation=attestation,
        completed_human=completed,
        output_receipt=output,
    )
    assert result["status"] == process.PROCESS_STATUS
    assert result["counts"] == {"audit_atoms": 1, "human_agreements": 0, "human_disagreements": 1, "adjudicated_disagreements": 1, "unresolved_disagreements": 0}
    assert result["assertions"]["global_answer_key_blindness_claimed"] is False
    assert result["inputs"]["completed_human"]["root_id"] == "review_process_root"
    process.validate_self_digest(result, "receipt_sha256", "process receipt")

    original_receipt_1 = receipt_1.read_bytes()
    original_attestation = attestation.read_bytes()
    incomplete_attestation = json.loads(attestation.read_text())
    incomplete_attestation["reviewer_declarations"]["reviewer_1"][
        "did_not_access_answer_key"
    ] = False
    write_json(attestation, incomplete_attestation)
    with pytest.raises(process.ReviewProcessError, match="attestation is incomplete"):
        process.freeze_process(
            project_root=project,
            pre_metric_path=pre_metric,
            review_root=review,
            audit_manifest_path=audit_manifest,
            public_root=public,
            reviewer_instructions=instructions,
            reviewer_1_delivery_receipt=receipt_1,
            reviewer_2_delivery_receipt=receipt_2,
            reviewer_1_completed=reviewer_1,
            reviewer_2_completed=reviewer_2,
            independence_attestation=attestation,
            completed_human=completed,
            output_receipt=tmp_path / "process/incomplete_attestation.json",
        )
    attestation.write_bytes(original_attestation)
    forged = json.loads(receipt_1.read_text())
    forged["delivery_queue_sha256"] = "0" * 64
    del forged["receipt_sha256"]
    forged = process.with_self_digest(forged, "receipt_sha256")
    write_json(receipt_1, forged)
    attested = json.loads(attestation.read_text())
    attested["reviewer_1_delivery_receipt_sha256"] = sha(receipt_1)
    write_json(attestation, attested)
    with pytest.raises(process.ReviewProcessError, match="deterministic projection"):
        process.freeze_process(
            project_root=project,
            pre_metric_path=pre_metric,
            review_root=review,
            audit_manifest_path=audit_manifest,
            public_root=public,
            reviewer_instructions=instructions,
            reviewer_1_delivery_receipt=receipt_1,
            reviewer_2_delivery_receipt=receipt_2,
            reviewer_1_completed=reviewer_1,
            reviewer_2_completed=reviewer_2,
            independence_attestation=attestation,
            completed_human=completed,
            output_receipt=tmp_path / "process/forged_should_not_exist.json",
        )
    receipt_1.write_bytes(original_receipt_1)
    attestation.write_bytes(original_attestation)

    missing = tmp_path / "process/completed_missing_adjudication.csv"
    write_csv(missing, [{**row_1, "reviewer_2_score": "1", "reviewer_2_notes": row_2["reviewer_2_notes"]}])
    with pytest.raises(process.ReviewProcessError, match="lacks adjudication"):
        process.freeze_process(
            project_root=project,
            pre_metric_path=pre_metric,
            review_root=review,
            audit_manifest_path=audit_manifest,
            public_root=public,
            reviewer_instructions=instructions,
            reviewer_1_delivery_receipt=receipt_1,
            reviewer_2_delivery_receipt=receipt_2,
            reviewer_1_completed=reviewer_1,
            reviewer_2_completed=reviewer_2,
            independence_attestation=attestation,
            completed_human=missing,
            output_receipt=tmp_path / "process/should_not_exist.json",
        )


def test_process_refuses_missing_adjudication_and_nonpublic_delivery(tmp_path: Path):
    project, pre_metric, instructions, review, public, audit_manifest, blank, delivery_1, _ = fixture(tmp_path)
    (delivery_1 / "private_key.json").write_text("no\n", encoding="utf-8")
    with pytest.raises(process.ReviewProcessError, match="file inventory differs"):
        process.freeze_delivery(project_root=project, pre_metric_path=pre_metric, review_root=review, audit_manifest_path=audit_manifest, public_root=public, reviewer_id="reviewer_1", reviewer_instructions=instructions, delivery_root=delivery_1, output_receipt=tmp_path / "receipt.json")


def test_expanded_audit_status_uses_total_atom_count(tmp_path: Path):
    project, pre_metric, instructions, review, public, audit_manifest, blank, delivery_1, delivery_2 = fixture(tmp_path)
    source_row = {
        **blank,
        "reviewer_1_score": "2",
        "reviewer_1_notes": "frozen r1",
        "reviewer_2_score": "1",
        "reviewer_2_notes": "frozen r2",
        "adjudicator_score": "2",
        "adjudicator_notes": "frozen adjudication",
    }
    new_row = {**blank, "audit_id": "audit_2", "anonymous_review_id": "rv_2"}
    queue = public / "human_audit_queue.csv"
    write_csv(queue, [source_row, new_row])
    value = json.loads(audit_manifest.read_text())
    value["status"] = "expanded_human_audit_frozen"
    value["counts"] = {"initial_audit_atoms": 1, "total_audit_atoms": 2}
    value["artifacts"]["human_queue"] = {
        "path": "public/human_audit_queue.csv",
        "sha256": sha(queue),
        "size_bytes": queue.stat().st_size,
    }
    write_json(audit_manifest, value)
    write_csv(
        delivery_1 / "human_audit_queue.csv",
        [
            {**blank, "reviewer_1_score": "2", "reviewer_1_notes": "frozen r1"},
            new_row,
        ],
    )
    write_csv(
        delivery_2 / "human_audit_queue.csv",
        [
            {**blank, "reviewer_2_score": "1", "reviewer_2_notes": "frozen r2"},
            new_row,
        ],
    )
    receipt = process.freeze_delivery(project_root=project, pre_metric_path=pre_metric, review_root=review, audit_manifest_path=audit_manifest, public_root=public, reviewer_id="reviewer_1", reviewer_instructions=instructions, delivery_root=delivery_1, output_receipt=tmp_path / "expanded_receipt.json")
    assert receipt["audit_atom_count"] == 2
    assert receipt["projection_mode"].startswith("reviewer_specific")
    process.freeze_delivery(project_root=project, pre_metric_path=pre_metric, review_root=review, audit_manifest_path=audit_manifest, public_root=public, reviewer_id="reviewer_2", reviewer_instructions=instructions, delivery_root=delivery_2, output_receipt=tmp_path / "expanded_receipt_2.json")
    generated = tmp_path / "generated_reviewer_1"
    generated_receipt = tmp_path / "generated_reviewer_1_receipt.json"
    process.prepare_delivery(project_root=project, pre_metric_path=pre_metric, review_root=review, audit_manifest_path=audit_manifest, public_root=public, reviewer_id="reviewer_1", reviewer_instructions=instructions, delivery_root=generated, output_receipt=generated_receipt)
    with (generated / "human_audit_queue.csv").open(newline="", encoding="utf-8") as handle:
        generated_rows = list(csv.DictReader(handle))
    assert generated_rows[0]["reviewer_1_score"] == "2"
    assert generated_rows[0]["reviewer_2_score"] == generated_rows[0]["adjudicator_score"] == ""
    assert all(generated_rows[1][field] == "" for field in process.SCORE_FIELDS)
    exposed = tmp_path / "exposed"
    shutil.copytree(public, exposed)
    with pytest.raises(process.ReviewProcessError, match="peer/adjudicator"):
        process.freeze_delivery(project_root=project, pre_metric_path=pre_metric, review_root=review, audit_manifest_path=audit_manifest, public_root=public, reviewer_id="reviewer_1", reviewer_instructions=instructions, delivery_root=exposed, output_receipt=tmp_path / "exposed_receipt.json")
