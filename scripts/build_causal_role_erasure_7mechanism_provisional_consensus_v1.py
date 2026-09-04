#!/usr/bin/env python3
"""Export a post-unblinding VLM-only exploratory consensus.

This is deliberately separate from the human canonicalization workflow.  It
accepts only public assignments, the two completed blinded VLM passes, and the
public key commitments.  It has no option for an Original-only or full method
key and never computes method-level metrics.

The full method key was already viewed by a separate read-only audit before
this exporter was added.  Consequently, neither its decision nor its output
may be represented as pre-unblinding, preregistered, or confirmatory.  The
exporter itself still accepts only blinded public inputs and never reads or
uses method identities.

The output is fresh-only.  A self-committed decision record is written before
the consensus rows are materialized.  Every atomic score is the arithmetic
mean of pass A and pass B, so the only possible values are
``0, 0.5, 1, 1.5, 2``.  Human-canonical scores remain the sole final scientific
authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    import canonicalize_causal_role_erasure_7mechanism_formal_v1 as canonical
    import causal_role_erasure_7mechanism_evaluation_code_registry_v1 as evaluation_code
except ModuleNotFoundError:  # imported as ``scripts.<module>`` in tests
    from scripts import canonicalize_causal_role_erasure_7mechanism_formal_v1 as canonical
    from scripts import causal_role_erasure_7mechanism_evaluation_code_registry_v1 as evaluation_code


PROTOCOL = "causal_role_erasure_7m_post_unblinding_vlm_ab_mean_v1"
STATUS = "post_unblinding_exploratory_vlm_ab_mean_not_human_calibrated"
SCIENTIFIC_LABEL = "POST_UNBLINDING_EXPLORATORY_PRELIMINARY"
USE_RESTRICTION = "WRITING_PREVIEW_ONLY_NOT_FOR_FINAL_SCIENTIFIC_CLAIMS"
FINAL_AUTHORITY = "FINAL_HUMAN_CANONICAL_SCORES_REMAIN_PRIMARY"
GLOBAL_UNBLINDING_STATE = (
    "ALREADY_OPENED_BY_SEPARATE_READ_ONLY_AUDIT_BEFORE_THIS_PROVISIONAL_EXPORT"
)
EXPECTED_ITEMS = 2448
EXPECTED_ORIGINAL_KEY_ITEMS = 588
ALLOWED_MEANS = (0, 0.5, 1, 1.5, 2)
IMPLEMENTATION_RELATIVE_PATH = (
    "scripts/build_causal_role_erasure_7mechanism_provisional_consensus_v1.py"
)
COMMITMENT_FIELDS = {
    "blind_key_sha256",
    "commitment_scheme",
    "evaluation_code_registry",
    "generation_ledger",
    "protocol",
    "schema_version",
    "tier_0_audit_strata",
    "tier_1_original_only",
    "tier_2_full",
}
PROVISIONAL_ROW_FIELDS = {
    "anonymous_review_id",
    "assignment_sha256",
    "case_kind",
    "scores",
    "source_scores",
    "source_score_record_sha256",
    "consensus_rule",
    "status",
    "scientific_label",
    "use_restriction",
    "final_authority",
}


class ProvisionalConsensusError(ValueError):
    """Public inputs or requested output violate the provisional protocol."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ProvisionalConsensusError(message)


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def regular_file(path: Path, label: str) -> None:
    require(
        path.is_file() and not path.is_symlink(),
        f"{label} missing or symlinked: {path}",
    )


def load_json(path: Path, label: str) -> dict[str, Any]:
    regular_file(path, label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProvisionalConsensusError(f"{label} is invalid JSON") from exc
    require(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def write_bytes(path: Path, value: bytes, mode: int = 0o444) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(value)
    path.chmod(mode)


def write_json(path: Path, value: Any, mode: int = 0o444) -> None:
    write_bytes(path, canonical_json_bytes(value), mode)


def write_jsonl(
    path: Path, rows: Iterable[Mapping[str, Any]], mode: int = 0o444
) -> None:
    write_bytes(
        path,
        b"".join(canonical_json_bytes(dict(row)) for row in rows),
        mode,
    )


def file_ref(path: Path) -> dict[str, Any]:
    regular_file(path, "referenced artifact")
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def relative_ref(root: Path, path: Path) -> dict[str, Any]:
    regular_file(path, "output artifact")
    return {
        "path": str(path.relative_to(root)),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _valid_sha(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def load_public_commitments(
    path: Path, *, expected_items: int = EXPECTED_ITEMS
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    """Validate only the public commitment record and its code registry."""

    value = load_json(path, "public key commitments")
    require(set(value) == COMMITMENT_FIELDS, "public key commitment schema changed")
    require(
        value["protocol"] == canonical.REVIEW_PROTOCOL
        and value["schema_version"] == 1
        and value["commitment_scheme"] == "sha256(canonical-jsonl-bytes)",
        "public key commitment identity changed",
    )
    require(_valid_sha(value["blind_key_sha256"]), "blind-key commitment is invalid")

    expected_counts = {
        "tier_0_audit_strata": expected_items,
        "tier_1_original_only": EXPECTED_ORIGINAL_KEY_ITEMS,
        "tier_2_full": expected_items,
        "generation_ledger": expected_items,
    }
    for name, count in expected_counts.items():
        entry = value.get(name)
        require(
            isinstance(entry, dict)
            and set(entry) == {"row_count", "sha256"}
            and entry["row_count"] == count
            and _valid_sha(entry["sha256"]),
            f"public {name} commitment changed",
        )

    registry_ref = value.get("evaluation_code_registry")
    require(
        isinstance(registry_ref, dict)
        and set(registry_ref) == {"path", "sha256", "registry_sha256"}
        and registry_ref["path"] == "evaluation_code_registry.json"
        and _valid_sha(registry_ref["sha256"])
        and _valid_sha(registry_ref["registry_sha256"]),
        "public evaluation-code registry reference changed",
    )
    registry_path = path.parent / "evaluation_code_registry.json"
    regular_file(registry_path, "evaluation code registry")
    require(
        sha256_file(registry_path) == registry_ref["sha256"],
        "evaluation code registry file hash changed",
    )
    registry = load_json(registry_path, "evaluation code registry")
    try:
        evaluation_code.validate_registry(
            registry, Path(__file__).resolve().parents[1]
        )
    except ValueError as exc:
        raise ProvisionalConsensusError(str(exc)) from exc
    require(
        registry["registry_sha256"] == registry_ref["registry_sha256"],
        "evaluation code registry self-commitment changed",
    )
    return value, registry_path, registry


def _score_record_sha256(row: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json_bytes(dict(row)))


def _mean_ordinal(left: int, right: int) -> int | float:
    total = left + right
    value: int | float = total // 2 if total % 2 == 0 else total / 2
    require(value in ALLOWED_MEANS, "internal arithmetic mean escaped frozen domain")
    return value


def build_provisional_rows(
    scores_a: Sequence[Mapping[str, Any]],
    scores_b: Sequence[Mapping[str, Any]],
    assignments: Mapping[str, Mapping[str, Any]],
    *,
    expected_items: int = EXPECTED_ITEMS,
) -> list[dict[str, Any]]:
    """Strictly join blinded rows and compute field-wise arithmetic means."""

    require(
        len(scores_a) == len(scores_b) == len(assignments) == expected_items,
        "A/B/assignment row counts differ from the frozen item count",
    )
    for pass_id, rows in (("A", scores_a), ("B", scores_b)):
        for index, row in enumerate(rows):
            try:
                canonical.validate_score_row(row, f"pass {pass_id} row {index}")
            except canonical.CanonicalizationError as exc:
                raise ProvisionalConsensusError(str(exc)) from exc

    a_by_id = {str(row["anonymous_review_id"]): row for row in scores_a}
    b_by_id = {str(row["anonymous_review_id"]): row for row in scores_b}
    require(len(a_by_id) == expected_items, "pass A contains duplicate anonymous IDs")
    require(len(b_by_id) == expected_items, "pass B contains duplicate anonymous IDs")
    require(
        set(a_by_id) == set(b_by_id) == set(assignments),
        "A/B/assignment anonymous inventories differ (missing or extra row)",
    )
    require(
        [str(row["anonymous_review_id"]) for row in scores_a]
        != [str(row["anonymous_review_id"]) for row in scores_b],
        "A/B review orders are not independently randomized",
    )

    provisional: list[dict[str, Any]] = []
    for review_id in sorted(a_by_id):
        left = a_by_id[review_id]
        right = b_by_id[review_id]
        assignment = assignments[review_id]
        require(
            left["assignment_sha256"]
            == right["assignment_sha256"]
            == assignment.get("assignment_sha256"),
            f"{review_id}: A/B/public assignment binding mismatch",
        )
        require(
            left["case_kind"]
            == right["case_kind"]
            == assignment.get("case_kind"),
            f"{review_id}: A/B/public case_kind mismatch",
        )
        fields = canonical.FIELDS_BY_KIND.get(str(left["case_kind"]))
        require(fields is not None, f"{review_id}: unknown case_kind")
        require(
            set(left["scores"]) == set(right["scores"]) == set(fields),
            f"{review_id}: A/B score-field inventory mismatch",
        )
        means = {
            field: _mean_ordinal(left["scores"][field], right["scores"][field])
            for field in fields
        }
        row = {
            "anonymous_review_id": review_id,
            "assignment_sha256": left["assignment_sha256"],
            "case_kind": left["case_kind"],
            "scores": means,
            "source_scores": {
                "pass_a": {field: left["scores"][field] for field in fields},
                "pass_b": {field: right["scores"][field] for field in fields},
            },
            "source_score_record_sha256": {
                "pass_a": _score_record_sha256(left),
                "pass_b": _score_record_sha256(right),
            },
            "consensus_rule": "fieldwise_arithmetic_mean_of_blinded_pass_a_and_pass_b",
            "status": STATUS,
            "scientific_label": SCIENTIFIC_LABEL,
            "use_restriction": USE_RESTRICTION,
            "final_authority": FINAL_AUTHORITY,
        }
        require(set(row) == PROVISIONAL_ROW_FIELDS, "internal provisional row schema changed")
        provisional.append(row)
    require(
        len(provisional) == expected_items,
        "provisional score inventory is incomplete",
    )
    return provisional


def _fresh_staging(output_root: Path) -> Path:
    require(
        not output_root.exists() and not output_root.is_symlink(),
        f"fresh-only output already exists: {output_root}",
    )
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.tmp-", dir=output_root.parent)
    )
    staging.chmod(0o700)
    return staging


def build_provisional_consensus(
    *,
    scores_a_path: Path,
    scores_b_path: Path,
    assignments_path: Path,
    key_commitments_path: Path,
    output_root: Path,
    expected_items: int = EXPECTED_ITEMS,
) -> dict[str, Any]:
    """Freeze the decision first, then materialize a blind provisional ledger."""

    require(
        scores_a_path.resolve() != scores_b_path.resolve(),
        "pass A and pass B score paths must differ",
    )
    staging = _fresh_staging(output_root)
    try:
        commitments, registry_path, registry = load_public_commitments(
            key_commitments_path, expected_items=expected_items
        )
        try:
            scores_a = canonical.load_merged_scores(
                scores_a_path,
                "A",
                expected_items,
                expected_assignments_path=assignments_path,
            )
            scores_b = canonical.load_merged_scores(
                scores_b_path,
                "B",
                expected_items,
            )
            assignments = canonical.load_assignments(assignments_path, expected_items)
        except canonical.CanonicalizationError as exc:
            raise ProvisionalConsensusError(str(exc)) from exc

        merge_a_path = scores_a_path.parent / "merge_manifest.json"
        merge_b_path = scores_b_path.parent / "merge_manifest.json"
        merge_a = load_json(merge_a_path, "pass A merge manifest")
        merge_b = load_json(merge_b_path, "pass B merge manifest")
        registry_sha = registry["registry_sha256"]
        require(
            merge_a.get("evaluation_code_registry_sha256")
            == merge_b.get("evaluation_code_registry_sha256")
            == commitments["evaluation_code_registry"]["registry_sha256"]
            == registry_sha,
            "A/B/key-commitment evaluation code registry bindings differ",
        )

        implementation_path = Path(__file__).resolve()
        inputs = {
            "scores_a": file_ref(scores_a_path),
            "scores_a_merge_manifest": file_ref(merge_a_path),
            "scores_b": file_ref(scores_b_path),
            "scores_b_merge_manifest": file_ref(merge_b_path),
            "public_assignments": file_ref(assignments_path),
            "public_key_commitments": file_ref(key_commitments_path),
            "evaluation_code_registry": file_ref(registry_path),
            "provisional_implementation": file_ref(implementation_path),
        }
        decision_body = {
            "schema_version": 1,
            "protocol": PROTOCOL,
            "status": STATUS,
            "scientific_label": SCIENTIFIC_LABEL,
            "use_restriction": USE_RESTRICTION,
            "final_authority": FINAL_AUTHORITY,
            "global_unblinding_state": GLOBAL_UNBLINDING_STATE,
            "protocol_deviation": {
                "occurred": True,
                "code": "full_method_key_access_preceded_provisional_rule_freeze",
                "scope": "a_separate_read_only_audit_opened_the_full_key",
                "consequence": (
                    "this_export_is_post_unblinding_exploratory_only_and_cannot_be_"
                    "described_as_preregistered_or_confirmatory"
                ),
            },
            "decision": {
                "unit": "one anonymous video and one rubric field",
                "rule": "fieldwise_arithmetic_mean_of_blinded_pass_a_and_pass_b",
                "formula": "(pass_a_score + pass_b_score) / 2",
                "allowed_input_values": [0, 1, 2],
                "allowed_output_values": list(ALLOWED_MEANS),
                "tie_breaking": "none",
                "rounding": "none",
                "human_override_in_this_artifact": False,
            },
            "expected_items": expected_items,
            "input_bindings_sha256": sha256_bytes(canonical_json_bytes(inputs)),
            "evaluation_code_registry_sha256": registry_sha,
            "inputs": inputs,
            "sequence_contract": [
                "validate_public_blinded_inputs",
                "freeze_this_decision_record",
                "materialize_provisional_scores",
                "write_content_addressed_receipt",
            ],
            "key_and_metric_boundaries": {
                "full_method_key_is_an_input": False,
                "full_method_key_read_by_this_builder": False,
                "method_identities_used_in_consensus": False,
                "global_full_method_key_access_had_already_occurred": True,
                "original_only_key_is_an_input": False,
                "method_metrics_computed": False,
                "canonical_scores_read_or_written": False,
            },
        }
        decision = {
            **decision_body,
            "decision_body_sha256": sha256_bytes(canonical_json_bytes(decision_body)),
        }
        decision_path = staging / "provisional_decision.json"
        # This exclusive write intentionally precedes arithmetic materialization.
        write_json(decision_path, decision)

        rows = build_provisional_rows(
            scores_a,
            scores_b,
            assignments,
            expected_items=expected_items,
        )
        scores_path = staging / "provisional_scores.jsonl"
        write_jsonl(scores_path, rows)
        atomic_count = sum(len(row["scores"]) for row in rows)
        receipt = {
            "schema_version": 1,
            "protocol": PROTOCOL,
            "status": STATUS,
            "completion_status": "complete_post_unblinding_exploratory_vlm_ab_mean",
            "scientific_label": SCIENTIFIC_LABEL,
            "use_restriction": USE_RESTRICTION,
            "final_authority": FINAL_AUTHORITY,
            "global_unblinding_state": GLOBAL_UNBLINDING_STATE,
            "protocol_deviation": decision_body["protocol_deviation"],
            "decision_frozen_before_score_materialization": True,
            "evaluation_code_registry_sha256": registry_sha,
            "counts": {
                "videos": len(rows),
                "atomic_scores": atomic_count,
                "pass_a_rows": len(scores_a),
                "pass_b_rows": len(scores_b),
            },
            "inputs": inputs,
            "artifacts": {
                "provisional_decision": relative_ref(staging, decision_path),
                "provisional_scores": relative_ref(staging, scores_path),
            },
            "key_and_metric_boundaries": decision_body["key_and_metric_boundaries"],
        }
        receipt_path = staging / "provisional_receipt.json"
        write_json(receipt_path, receipt)
        require(not output_root.exists(), f"output appeared during build: {output_root}")
        os.replace(staging, output_root)
        return receipt
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _resolve(root: Path, value: Path) -> Path:
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--scores-a", type=Path, required=True)
    parser.add_argument("--scores-b", type=Path, required=True)
    parser.add_argument("--assignments", type=Path, required=True)
    parser.add_argument("--key-commitments", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    root = args.project_root.resolve()
    try:
        receipt = build_provisional_consensus(
            scores_a_path=_resolve(root, args.scores_a),
            scores_b_path=_resolve(root, args.scores_b),
            assignments_path=_resolve(root, args.assignments),
            key_commitments_path=_resolve(root, args.key_commitments),
            output_root=_resolve(root, args.output_root),
        )
    except ProvisionalConsensusError as exc:
        parser.exit(2, f"provisional consensus refused: {exc}\n")
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "completion_status": receipt["completion_status"],
                "counts": receipt["counts"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
