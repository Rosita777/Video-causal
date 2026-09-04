from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

from PIL import Image

from scripts import canonicalize_causal_role_erasure_7mechanism_formal_v1 as canonical
from scripts import compute_causal_role_erasure_7mechanism_formal_metrics_v1 as metrics


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FORMAL_CASES = PROJECT_ROOT / "data" / "causal_role_erasure_7mechanism_main_v2" / "formal_cases.csv"
IDENTIFICATION = PROJECT_ROOT / "data" / "causal_role_erasure_7mechanism_main_v2" / "identification_subset.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(canonical.canonical_json_bytes(row) for row in rows))


def artifact_ref(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "sha256": canonical.sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def write_public_pass(root: Path, pass_id: str, assignments, panel: Path):
    media = root / "media"
    media.mkdir(parents=True)
    ordered = list(assignments if pass_id == "A" else reversed(assignments))
    for row in ordered:
        os.link(panel, media / f"{row['anonymous_review_id']}.jpg")
    assignments_path = root / "assignments.jsonl"
    blank_path = root / "scores.jsonl"
    write_jsonl(assignments_path, ordered)
    blanks = []
    for row in ordered:
        fields = canonical.FIELDS_BY_KIND[row["case_kind"]]
        blanks.append(
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
    write_jsonl(blank_path, blanks)
    pass_manifest = {
        "protocol": canonical.REVIEW_PROTOCOL,
        "schema_version": 1,
        "pass_id": pass_id,
        "item_count": 2448,
        "ordering_commitment": hashlib.sha256(f"order-{pass_id}".encode()).hexdigest(),
        "panel_windows": [[0, 12], [9, 21], [18, 30], [27, 39], [36, 48]],
        "assignments": {"path": "assignments.jsonl", "sha256": canonical.sha256_file(assignments_path)},
        "blank_scores": {"path": "scores.jsonl", "sha256": canonical.sha256_file(blank_path)},
        "composite_contract": {
            "path_base": "pass_root",
            "directory": "media",
            "format": "jpeg",
            "width": canonical.transport.COMPOSITE_WIDTH,
            "height": canonical.transport.COMPOSITE_HEIGHT,
            "jpeg_quality": canonical.transport.JPEG_QUALITY,
            "tile_width": 112,
            "tile_height": 64,
            "tile_fit": "deterministic_center_crop_no_letterbox",
            "resampling": "Pillow.Image.Resampling.LANCZOS",
            "max_bytes": canonical.transport.MAX_IMAGE_BYTES,
            "one_image_per_assignment": True,
        },
    }
    manifest_path = root / "pass_manifest.json"
    manifest_path.write_text(json.dumps(pass_manifest), encoding="utf-8")
    return assignments_path, blank_path, manifest_path, ordered


def write_merge(
    root: Path,
    pass_id: str,
    rows,
    assignments_path: Path,
    blank_path: Path,
    pass_manifest_path: Path,
) -> Path:
    scores = root / "completed_scores.jsonl"
    ordered_ids = [row["anonymous_review_id"] for row in canonical.load_jsonl(assignments_path, "assignments")]
    rows_by_id = {row["anonymous_review_id"]: row for row in rows}
    ordered_rows = [rows_by_id[review_id] for review_id in ordered_ids]
    write_jsonl(scores, ordered_rows)
    run_registration = root / "source_run" / "run_registration.json"
    run_summary = root / "source_run" / "run_summary.json"
    run_registration.parent.mkdir(parents=True)
    run_registration.write_text(json.dumps({"status": "completed"}), encoding="utf-8")
    run_summary.write_text(json.dumps({"status": "completed"}), encoding="utf-8")
    raw_registry = root / "raw_response_registry.json"
    raw_registry.write_text(
        json.dumps(
            {
                row["anonymous_review_id"]: {
                    "model": "gpt-5.6-luna",
                    "observed_model": "gpt-5.6-luna",
                }
                for row in ordered_rows
            }
        ),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "workflow_version": canonical.TRANSPORT_WORKFLOW,
        "status": "complete_schema_valid_public_pass_review",
        "pass_id": pass_id,
        "evaluation_code_registry_sha256": metrics.evaluation_code.build_registry(
            PROJECT_ROOT
        )["registry_sha256"],
        "row_count": 2448,
        "model": "gpt-5.6-luna",
        "reasoning_effort": "low",
        "temperature": 1.0,
        "max_output_tokens": 1600,
        "store": False,
        "proxy_version": "synthetic-test-proxy",
        "endpoint": "http://127.0.0.1:4141/v1/responses",
        "pass_manifest": artifact_ref(pass_manifest_path),
        "assignments": artifact_ref(assignments_path),
        "blank_scores": artifact_ref(blank_path),
        "source_runs": [
            {
                "run_registration": artifact_ref(run_registration),
                "run_summary": artifact_ref(run_summary),
            }
        ],
        "scientific_zero_fallbacks": 0,
        "completed_scores": {
            "path": "completed_scores.jsonl",
            "sha256": canonical.sha256_file(scores),
        },
        "raw_response_registry": artifact_ref(raw_registry),
    }
    (root / "merge_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return scores


def synthetic_inputs(root: Path):
    cases = read_csv(FORMAL_CASES)
    identification_ids = {row["case_id"] for row in read_csv(IDENTIFICATION)}
    panel = root / "panel.jpg"
    panel.parent.mkdir(parents=True, exist_ok=True)
    Image.new(
        "RGB",
        (canonical.transport.COMPOSITE_WIDTH, canonical.transport.COMPOSITE_HEIGHT),
        (100, 120, 140),
    ).save(panel, format="JPEG", quality=canonical.transport.JPEG_QUALITY)
    panel_sha = canonical.sha256_file(panel)
    assignments = []
    key = []
    original_key = []
    a_rows = []
    b_rows = []
    counter = 0

    def values(stream: str, kind: str):
        if kind == "specificity":
            return {
                "protected_object_visibility": 2,
                "noncausal_role_adherence": 2,
                "receiver_preservation": 2,
                "video_quality": 2,
            }
        if stream in ("wan_original", "cogvideox_original"):
            source, footprint = 2, 2
        elif stream == "V4":
            source, footprint = 0, 0
        else:
            source, footprint = 2, 0
        return {
            "source_visibility": source,
            "footprint_visibility": footprint,
            "receiver_preservation": 2,
            "video_quality": 2,
        }

    def add(stream: str, case: dict[str, str], partition: str):
        nonlocal counter
        review_id = f"anon_{counter:04d}"
        counter += 1
        assignment = {
            "anonymous_review_id": review_id,
            "case_kind": case["case_kind"],
            "mechanism_name": case["mechanism_name"],
            "prompt": case["prompt"],
            "source_object": case["source_object"],
            "receiver": case["receiver"],
            "expected_trigger": case["expected_trigger"],
            "expected_footprint": case["expected_footprint"],
            "expected_counterfactual_state": case["expected_counterfactual_state"],
            "protected_object": case["protected_object"],
            "specificity_subtype": case["specificity_subtype"],
            "acceptable_alternative_cause": case["acceptable_alternative_cause"],
            "composite_path": f"media/{review_id}.jpg",
            "composite_sha256": panel_sha,
        }
        assignment_sha = hashlib.sha256(canonical.canonical_json_bytes(assignment)).hexdigest()
        assignment["assignment_sha256"] = assignment_sha
        assignments.append(assignment)
        binding = {
            "anonymous_review_id": review_id,
            "evaluation_partition": partition,
            "stream": stream,
            "backbone_family": "cogvideox" if stream in metrics.COG_STREAMS else "wan",
            "case_id": case["case_id"],
            "formal_global_index": int(case["global_case_index"]),
            "mechanism": case["mechanism"],
            "case_kind": case["case_kind"],
            "generalization_group": case["generalization_group"],
            "source_membership": case["source_membership"],
            "prompt_style": case["prompt_style"],
            "footprint_lexicalization": case["footprint_lexicalization"],
            "specificity_subtype": case["specificity_subtype"],
            "m6_pair_id": case["m6_pair_id"],
            "source_id": case["source_id"],
            "receiver_id": case["receiver_id"],
            "seed": int(case["seed"]),
        }
        key.append(binding)
        if stream in ("wan_original", "cogvideox_original"):
            original_key.append(binding)
        scores = values(stream, case["case_kind"])
        fields = tuple(scores)
        base = {
            "anonymous_review_id": review_id,
            "assignment_sha256": assignment_sha,
            "case_kind": case["case_kind"],
            "scores": scores,
            "confidence": {field: 0.95 for field in fields},
            "evidence_frames": {field: [12] for field in fields},
            "evidence_observations": {field: [f"visible evidence for {field}"] for field in fields},
            "unusable_reason": "",
            "status": "completed",
        }
        a_rows.append(json.loads(json.dumps(base)))
        b_rows.append(json.loads(json.dumps(base)))

    for stream in metrics.MAIN_STREAMS:
        for case in cases:
            add(stream, case, "main")
    selected = [case for case in cases if case["case_id"] in identification_ids]
    for stream in metrics.IDENTIFICATION_STREAMS:
        for case in selected:
            add(stream, case, "identification")
    assert counter == 2448 and len(original_key) == 588

    # Exercise all mandatory audit routes without changing the overall result.
    by_stream_case = {(row["stream"], row["case_id"]): row["anonymous_review_id"] for row in key}
    ids = [by_stream_case[("matched_control", cases[index]["case_id"])] for index in range(3)]
    a_by = {row["anonymous_review_id"]: row for row in a_rows}
    b_by = {row["anonymous_review_id"]: row for row in b_rows}
    b_by[ids[0]]["scores"]["source_visibility"] = 1  # VLM disagreement + partial.
    a_by[ids[1]]["scores"]["source_visibility"] = 1
    b_by[ids[1]]["scores"]["source_visibility"] = 1  # agreed partial.
    a_by[ids[2]]["confidence"]["source_visibility"] = 0.5  # low confidence.
    unusable_id = by_stream_case[("matched_control", cases[3]["case_id"])]
    a_by[unusable_id]["scores"]["receiver_preservation"] = 0
    b_by[unusable_id]["scores"]["receiver_preservation"] = 0

    assignments_path, blank_a, manifest_a, _ = write_public_pass(
        root / "public" / "pass_a", "A", assignments, panel
    )
    assignments_b, blank_b, manifest_b, _ = write_public_pass(
        root / "public" / "pass_b", "B", assignments, panel
    )
    full_key = root / "private" / "full_key.jsonl"
    original = root / "private" / "original_only_key.jsonl"
    audit_strata_key = root / "private" / "audit_strata_key.jsonl"
    write_jsonl(full_key, key)
    write_jsonl(original, original_key)
    stream_tokens = {
        (row["evaluation_partition"], row["stream"]): "ss_" + hashlib.sha256(
            f"{row['evaluation_partition']}|{row['stream']}".encode()
        ).hexdigest()[:32]
        for row in key
    }
    audit_strata = [
        {
            "anonymous_review_id": row["anonymous_review_id"],
            "case_kind": row["case_kind"],
            "mechanism": row["mechanism"],
            "stream_stratum": stream_tokens[(row["evaluation_partition"], row["stream"])],
        }
        for row in key
    ]
    write_jsonl(audit_strata_key, sorted(audit_strata, key=lambda row: row["anonymous_review_id"]))
    commitments = root / "public" / "key_commitments.json"
    code_registry = metrics.evaluation_code.build_registry(PROJECT_ROOT)
    code_registry_path = root / "public" / "evaluation_code_registry.json"
    code_registry_path.write_bytes(canonical.canonical_json_bytes(code_registry))
    commitments.write_text(
        json.dumps(
            {
                "protocol": canonical.REVIEW_PROTOCOL,
                "schema_version": 1,
                "commitment_scheme": "sha256(canonical-jsonl-bytes)",
                "evaluation_code_registry": {
                    "path": "evaluation_code_registry.json",
                    "sha256": canonical.sha256_file(code_registry_path),
                    "registry_sha256": code_registry["registry_sha256"],
                },
                "tier_0_audit_strata": {"row_count": 2448, "sha256": canonical.sha256_file(audit_strata_key)},
                "tier_1_original_only": {"row_count": 588, "sha256": canonical.sha256_file(original)},
                "tier_2_full": {"row_count": 2448, "sha256": canonical.sha256_file(full_key)},
            }
        ),
        encoding="utf-8",
    )
    scores_a = write_merge(
        root / "merge_a", "A", a_rows, assignments_path, blank_a, manifest_a
    )
    scores_b = write_merge(
        root / "merge_b", "B", b_rows, assignments_b, blank_b, manifest_b
    )
    return {
        "cases": cases,
        "assignments": assignments_path,
        "full_key": full_key,
        "original_key": original,
        "audit_strata_key": audit_strata_key,
        "key_commitments": commitments,
        "scores_a": scores_a,
        "scores_b": scores_b,
        "key_rows": key,
        "unusable_id": unusable_id,
    }


def complete_human_queue(
    queue_path: Path,
    audit_key_path: Path,
    output_path: Path,
    *,
    force_error_audit_ids: set[str] | None = None,
) -> None:
    force_error_audit_ids = force_error_audit_ids or set()
    atoms = {
        row["audit_id"]: row
        for row in canonical.load_jsonl(audit_key_path, "audit key")
    }
    rows = canonical.load_csv(queue_path, canonical.HUMAN_QUEUE_FIELDS, "human queue")
    for row in rows:
        score = int(atoms[row["audit_id"]]["a_score"])
        if row["audit_id"] in force_error_audit_ids:
            score = 2 if score == 0 else 0
        row["reviewer_1_score"] = str(score)
        row["reviewer_2_score"] = str(score)
        row["reviewer_1_notes"] = "independent review 1"
        row["reviewer_2_notes"] = "independent review 2"
    output_path.write_bytes(canonical.csv_bytes(rows, canonical.HUMAN_QUEUE_FIELDS))


def test_end_to_end_audit_expansion_canonical_freeze_and_registered_metrics(tmp_path):
    inputs = synthetic_inputs(tmp_path / "inputs")
    audit = tmp_path / "audit"
    audit_manifest = canonical.build_audit_package(
        scores_a_path=inputs["scores_a"],
        scores_b_path=inputs["scores_b"],
        assignments_path=inputs["assignments"],
        audit_strata_key_path=inputs["audit_strata_key"],
        key_commitments_path=inputs["key_commitments"],
        output_root=audit,
    )
    assert audit_manifest["counts"]["videos"] == 2448
    assert audit_manifest["counts"]["atoms"] == 9792
    assert audit_manifest["counts"]["reasons"]["vlm_disagreement"] >= 1
    assert audit_manifest["counts"]["reasons"]["partial"] >= 1
    assert audit_manifest["counts"]["reasons"]["low_confidence"] >= 1
    public_queue_text = (audit / "public" / "human_audit_queue.csv").read_text(encoding="utf-8")
    assert str(tmp_path) not in public_queue_text
    assert "../" not in public_queue_text
    for hidden_method in (*metrics.MAIN_STREAMS, *metrics.IDENTIFICATION_STREAMS):
        assert hidden_method not in public_queue_text
    public_rows = canonical.load_csv(
        audit / "public" / "human_audit_queue.csv",
        canonical.HUMAN_QUEUE_FIELDS,
        "audit queue",
    )
    for row in public_rows:
        image = audit / "public" / row["composite_path"]
        assert image.is_file() and not image.is_symlink()
        assert canonical.sha256_file(image) == row["composite_sha256"]

    audit_atoms = canonical.load_jsonl(audit / "private" / "audit_key.jsonl", "audit key")
    unusable_atoms = [
        row for row in audit_atoms if row["anonymous_review_id"] == inputs["unusable_id"]
    ]
    assert len(unusable_atoms) == 4
    assert all("unusable_output" in row["reason_codes"] for row in unusable_atoms)
    v4_ids = {
        row["anonymous_review_id"]
        for row in inputs["key_rows"]
        if row["stream"] == "V4"
    }
    target = next(
        row
        for row in audit_atoms
        if row["anonymous_review_id"] in v4_ids
        and row["a_score"] == row["b_score"]
        and row["a_score"] != 1
        and row["a_confidence"] >= 0.75
        and row["b_confidence"] >= 0.75
    )
    forced = {
        row["audit_id"]
        for row in audit_atoms
        if row["anonymous_review_id"] in v4_ids
        and row["mechanism"] == target["mechanism"]
        and row["field"] == target["field"]
        and row["a_score"] == row["b_score"]
        and row["a_score"] != 1
        and row["a_confidence"] >= 0.75
        and row["b_confidence"] >= 0.75
    }
    assert len(forced) >= 3
    completed_initial = tmp_path / "completed_initial.csv"
    complete_human_queue(
        audit / "public" / "human_audit_queue.csv",
        audit / "private" / "audit_key.jsonl",
        completed_initial,
        force_error_audit_ids=forced,
    )
    expanded = tmp_path / "expanded"
    expanded_manifest = canonical.expand_audit_package(
        audit_root=audit,
        completed_human_path=completed_initial,
        output_root=expanded,
    )
    assert expanded_manifest["counts"]["expanded_strata"] >= 1
    assert expanded_manifest["counts"]["added_audit_atoms"] >= 1
    completed_expanded = tmp_path / "completed_expanded.csv"
    complete_human_queue(
        expanded / "public" / "human_audit_queue.csv",
        expanded / "private" / "audit_key.jsonl",
        completed_expanded,
    )
    frozen = tmp_path / "canonical"
    freeze_manifest = canonical.freeze_canonical_scores(
        audit_root=expanded,
        completed_human_path=completed_expanded,
        scores_a_path=inputs["scores_a"],
        scores_b_path=inputs["scores_b"],
        output_root=frozen,
    )
    assert freeze_manifest["row_count"] == 2448
    assert freeze_manifest["full_method_key_opened_for_scoring"] is False
    eligibility_root = tmp_path / "eligibility"
    eligibility_manifest = canonical.freeze_original_eligibility(
        canonical_root=frozen,
        original_key_path=inputs["original_key"],
        key_commitments_path=inputs["key_commitments"],
        output_root=eligibility_root,
    )
    assert eligibility_manifest["full_method_key_opened"] is False
    assert len(read_csv(eligibility_root / "original_eligibility.csv")) == 336
    assert sum(int(row["shared_eligible"]) for row in read_csv(eligibility_root / "shared_capability_subsets.csv")) == 168
    rogue_original = tmp_path / "rogue_original_key.jsonl"
    rogue_rows = canonical.load_jsonl(inputs["original_key"], "Original key")
    rogue_rows[0] = dict(rogue_rows[0])
    rogue_rows[0]["stream"] = "matched_control"
    write_jsonl(rogue_original, rogue_rows)
    try:
        canonical.freeze_original_eligibility(
            canonical_root=frozen,
            original_key_path=rogue_original,
            key_commitments_path=inputs["key_commitments"],
            output_root=tmp_path / "rogue_eligibility",
        )
    except canonical.CanonicalizationError as exc:
        assert "precommitted SHA" in str(exc)
    else:
        raise AssertionError("post-canonical Original-key replacement was accepted")

    output = tmp_path / "metrics"
    results = metrics.compute_formal_metrics(
        canonical_scores_path=frozen / "canonical_scores.jsonl",
        full_key_path=inputs["full_key"],
        key_commitments_path=inputs["key_commitments"],
        original_eligibility_path=eligibility_root / "original_eligibility.csv",
        shared_subsets_path=eligibility_root / "shared_capability_subsets.csv",
        formal_cases_path=FORMAL_CASES,
        output_root=output,
    )
    assert results["bootstrap"] == {
        "replicates": 10_000,
        "headline_pcg64_seed": 8_202_601,
        "identification_pcg64_seed": 8_202_602,
        "interval": "unadjusted_percentile_2.5_97.5",
    }
    assert len(results["headline_contrasts"]) == 5
    assert len(results["identification_contrasts"]) == 2
    assert results["claims"] == {
        "outperforms_all_registered_baselines": True,
        "retaining_allowed_by_all_primary_noninferiority_margins": True,
        "role_conditioned_novelty_supported": True,
    }
    mechanism = read_csv(output / "mechanism_metrics.csv")
    macro = {row["stream"]: row for row in read_csv(output / "macro_metrics.csv")}
    assert len(mechanism) == 56
    assert float(macro["V4"]["ces"]) > float(macro["matched_control"]["ces"])
    assert len(read_csv(output / "m6_pairs.csv")) == 336
    assert len(json.loads((output / "mechanism_metrics.json").read_text())) == 56
    assert len(json.loads((output / "m6_pairs.json").read_text())) == 336


def test_human_disagreement_requires_third_adjudicator(tmp_path):
    assert canonical.resolved_human_score(
        {
            "reviewer_1_score": "0",
            "reviewer_2_score": "1",
            "adjudicator_score": "2",
        }
    ) == 2
    inputs = synthetic_inputs(tmp_path / "inputs")
    audit = tmp_path / "audit"
    canonical.build_audit_package(
        scores_a_path=inputs["scores_a"],
        scores_b_path=inputs["scores_b"],
        assignments_path=inputs["assignments"],
        audit_strata_key_path=inputs["audit_strata_key"],
        key_commitments_path=inputs["key_commitments"],
        output_root=audit,
    )
    first_media = next((audit / "public" / "media").iterdir())
    original_media = first_media.read_bytes()
    first_media.write_bytes(b"corrupt")
    try:
        canonical._load_audit_root(audit)
    except canonical.CanonicalizationError as exc:
        assert "composite differs" in str(exc)
    else:
        raise AssertionError("post-package audit media corruption was accepted")
    first_media.write_bytes(original_media)
    queue = canonical.load_csv(audit / "public" / "human_audit_queue.csv", canonical.HUMAN_QUEUE_FIELDS, "queue")
    key = {row["audit_id"]: row for row in canonical.load_jsonl(audit / "private" / "audit_key.jsonl", "key")}
    for row in queue:
        score = str(key[row["audit_id"]]["a_score"])
        row["reviewer_1_score"] = score
        row["reviewer_2_score"] = score
    tampered_rows = [dict(row) for row in queue]
    tampered_rows[0]["prompt"] += " silently changed"
    tampered = tmp_path / "tampered_human.csv"
    tampered.write_bytes(canonical.csv_bytes(tampered_rows, canonical.HUMAN_QUEUE_FIELDS))
    try:
        canonical.expand_audit_package(
            audit_root=audit,
            completed_human_path=tampered,
            output_root=tmp_path / "tampered_out",
        )
    except canonical.CanonicalizationError as exc:
        assert "public context changed" in str(exc)
    else:
        raise AssertionError("tampered human-review context was accepted")
    queue[0]["reviewer_2_score"] = "2" if queue[0]["reviewer_1_score"] != "2" else "0"
    completed = tmp_path / "human.csv"
    completed.write_bytes(canonical.csv_bytes(queue, canonical.HUMAN_QUEUE_FIELDS))
    try:
        canonical.expand_audit_package(audit_root=audit, completed_human_path=completed, output_root=tmp_path / "no")
    except canonical.CanonicalizationError as exc:
        assert "requires adjudication" in str(exc)
    else:
        raise AssertionError("missing adjudication was accepted")


def test_holm_and_shared_subset_nonestimability_are_fail_closed(tmp_path):
    adjusted = metrics.holm_adjust({"a": 0.001, "b": 0.02, "c": 0.04})
    assert adjusted == {"a": 0.003, "b": 0.04, "c": 0.04}

    inputs = synthetic_inputs(tmp_path / "inputs")
    # This unit directly exercises the registered external estimability rule on
    # canonical-shaped synthetic rows without rerunning the human stages.
    key = canonical.load_jsonl(inputs["full_key"], "key")
    a = canonical.load_merged_scores(inputs["scores_a"], "A")
    eligibility = []
    for binding in key:
        if binding["stream"] in ("wan_original", "cogvideox_original") and binding["case_kind"] == "causal":
            eligibility.append(
                {
                    "case_id": binding["case_id"],
                    "backbone_family": binding["backbone_family"],
                    "eligible": "1",
                }
            )
    case_rows = metrics.build_case_metrics(a, key, eligibility, read_csv(FORMAL_CASES))
    shared = []
    for case in read_csv(FORMAL_CASES):
        if case["case_kind"] == "causal":
            shared.append(
                {
                    "case_id": case["case_id"],
                    "mechanism": case["mechanism"],
                    "shared_eligible": "0" if case["mechanism"] == "water_impact" and int(case["global_case_index"]) >= 11 else "1",
                }
            )
    headline, _, claim = metrics.headline_contrasts(case_rows, shared)
    assert {row["contrast"]: row["estimable"] for row in headline}["matched_control"] == 1
    assert {row["contrast"]: row["estimable"] for row in headline}["negative_prompt"] == 0
    assert claim is False

    # The registered identification condition is the equal-weight two-mechanism
    # contrast, not positivity in each individual mechanism.
    identification_case_rows = json.loads(json.dumps(case_rows))
    for row in identification_case_rows:
        if row["case_kind"] != "causal":
            continue
        if row["stream"] == "V4" and row["case_id"] in {
            item["case_id"]
            for item in identification_case_rows
            if item["evaluation_partition"] == "identification"
        }:
            row["ces"] = 0.4 if row["mechanism"] == "water_impact" else 1.0
        if row["evaluation_partition"] == "identification":
            row["ces"] = 1.0 if row["mechanism"] == "water_impact" else 0.0
    identification, supported = metrics.identification_table(identification_case_rows)
    assert supported is True
    assert all(row["implicit_water_point_estimate"] < 0 for row in identification)
    assert all(row["implicit_fracture_point_estimate"] > 0 for row in identification)
    assert all(row["implicit_two_mechanism_point_estimate"] > 0 for row in identification)
    reversed_result, reversed_supported = metrics.identification_table(
        list(reversed(identification_case_rows))
    )
    assert reversed_result == identification
    assert reversed_supported is supported
