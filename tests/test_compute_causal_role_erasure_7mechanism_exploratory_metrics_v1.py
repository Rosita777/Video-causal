from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from scripts import build_causal_role_erasure_7mechanism_provisional_consensus_v1 as consensus
from scripts import canonicalize_causal_role_erasure_7mechanism_formal_v1 as canonical
from scripts import compute_causal_role_erasure_7mechanism_exploratory_metrics_v1 as exploratory
from tests.test_causal_role_erasure_7mechanism_formal_metrics_v1 import (
    FORMAL_CASES,
    read_csv,
    synthetic_inputs,
)


def _prepare_provisional(
    tmp_path: Path, *, make_water_shared_subset_nonestimable: bool = False
):
    inputs = synthetic_inputs(tmp_path / "review_package")
    commitments_path = inputs["key_commitments"]
    commitments = json.loads(commitments_path.read_text(encoding="utf-8"))
    commitments["generation_ledger"] = {"row_count": 2448, "sha256": "7" * 64}
    commitments["blind_key_sha256"] = "8" * 64
    commitments_path.write_bytes(exploratory.canonical_json_bytes(commitments))
    if make_water_shared_subset_nonestimable:
        original = canonical.load_jsonl(inputs["original_key"], "Original key")
        targets = {
            row["anonymous_review_id"]
            for row in original
            if row["stream"] == "wan_original"
            and row["mechanism"] == "water_impact"
            and row["case_kind"] == "causal"
        }
        assert len(targets) == 24
        scores_b_path = inputs["scores_b"]
        rows = canonical.load_jsonl(scores_b_path, "pass B scores")
        for row in rows:
            if row["anonymous_review_id"] in targets:
                row["scores"]["source_visibility"] = 1
        scores_b_path.write_bytes(
            b"".join(exploratory.canonical_json_bytes(row) for row in rows)
        )
        merge_path = scores_b_path.parent / "merge_manifest.json"
        merge = json.loads(merge_path.read_text(encoding="utf-8"))
        merge["completed_scores"]["sha256"] = exploratory.sha256_file(scores_b_path)
        merge_path.write_bytes(exploratory.canonical_json_bytes(merge))
    provisional_root = tmp_path / "provisional"
    consensus.build_provisional_consensus(
        scores_a_path=inputs["scores_a"],
        scores_b_path=inputs["scores_b"],
        assignments_path=inputs["assignments"],
        key_commitments_path=commitments_path,
        output_root=provisional_root,
    )
    return inputs, provisional_root


def test_end_to_end_is_not_canonical_and_fail_closes_small_shared_subsets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    inputs, provisional_root = _prepare_provisional(
        tmp_path, make_water_shared_subset_nonestimable=True
    )
    output = tmp_path / "exploratory"

    opened_after_checkpoint = []
    original_opener = exploratory._open_full_key_after_checkpoint

    def checked_opener(path, commitments, checkpoint_path, provisional_by_id, original_rows):
        assert checkpoint_path.is_file()
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        assert checkpoint["full_method_key_opened_by_this_process"] is False
        assert checkpoint["counts"]["shared_eligible_cases"] == 144
        opened_after_checkpoint.append(True)
        return original_opener(
            path, commitments, checkpoint_path, provisional_by_id, original_rows
        )

    monkeypatch.setattr(exploratory, "_open_full_key_after_checkpoint", checked_opener)
    results = exploratory.compute_exploratory_metrics(
        provisional_scores_path=provisional_root / "provisional_scores.jsonl",
        original_key_path=inputs["original_key"],
        full_key_path=inputs["full_key"],
        key_commitments_path=inputs["key_commitments"],
        formal_cases_path=FORMAL_CASES,
        output_root=output,
    )

    assert opened_after_checkpoint == [True]
    assert results["status"] == exploratory.STATUS
    assert results["completion_status"] == exploratory.COMPLETION_STATUS
    assert results["canonical_status"] == "NOT_CANONICAL"
    assert results["formal_claims_permitted"] is False
    assert results["protocol_deviation"]["occurred"] is True
    assert results["sequence"] == {
        "original_only_checkpoint_materialized_before_full_key_opened_by_this_process": True,
        "global_pre_unblinding_preregistration": False,
    }
    assert results["estimability"]["cross_backbone_contrasts_estimable"] is False
    assert results["nonlinear_order"] == exploratory.NONLINEAR_ORDER

    headline = read_csv(output / "headline_contrasts.csv")
    by_contrast = {row["contrast"]: row for row in headline}
    assert by_contrast["matched_control"]["estimable"] == "1"
    for name in (
        "negative_prompt",
        "videoeraser_official",
        "t2vunlearning_adapted",
        "safree_cogvideox",
    ):
        assert by_contrast[name]["estimable"] == "0"
        assert by_contrast[name]["exploratory_positive_threshold_pattern"] == "0"
        assert by_contrast[name]["formal_win_claim"] == "NOT_PERMITTED"
    assert all("registered_win" not in row for row in headline)

    headline_mechanism = read_csv(output / "headline_mechanism_contrasts.csv")
    water_external = [
        row
        for row in headline_mechanism
        if row["mechanism"] == "water_impact"
        and row["contrast"] != "matched_control"
    ]
    assert len(water_external) == 4
    assert all(row["estimability_status"] == "not_estimable_fail_closed" for row in water_external)
    assert all(row["point_estimate"] == "" for row in water_external)

    mechanism = read_csv(output / "mechanism_metrics.csv")
    macro = read_csv(output / "macro_metrics.csv")
    assert len(mechanism) == 56 and len(macro) == 8
    assert all("eligible_source_clear_to_partial_rate" not in row for row in mechanism)
    assert all("eligible_source_clear_to_partial_rate" not in row for row in macro)
    assert all(row["canonical_status"] == "NOT_CANONICAL" for row in mechanism)
    assert not (output / "secondary_metrics.csv").exists()
    assert len(read_csv(output / "per_case_metrics.csv")) == 2448
    assert len(read_csv(output / "m6_pairs.csv")) == 336
    sensitivity_csv = read_csv(output / "ab_order_sensitivity_macro.csv")
    sensitivity_json = json.loads(
        (output / "ab_order_sensitivity_macro.json").read_text(encoding="utf-8")
    )
    assert len(sensitivity_csv) == len(sensitivity_json) == 8
    sensitivity_json_by_stream = {row["stream"]: row for row in sensitivity_json}
    for row in sensitivity_csv:
        raw = sensitivity_json_by_stream[row["stream"]]
        assert row["sensitivity_only"] == str(raw["sensitivity_only"])
        assert row["replaces_primary"] == str(raw["replaces_primary"])
        assert float(row["pass_a_ces"]) == raw["pass_a_ces"]
        assert float(row["pass_b_ces"]) == raw["pass_b_ces"]
        assert float(row["primary_atomic_mean_ces"]) == raw[
            "primary_atomic_mean_ces"
        ]
        assert raw["canonical_status"] == "NOT_CANONICAL"
        assert raw["formal_claims_permitted"] is False
    assert any(
        row["mean_of_pass_macros_minus_primary_original_capability_coverage"]
        != 0
        for row in sensitivity_json
    )

    manifest = json.loads(
        (output / "exploratory_metrics_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == exploratory.STATUS
    assert manifest["completion_status"] == exploratory.COMPLETION_STATUS
    assert manifest["nonlinear_order"] == exploratory.NONLINEAR_ORDER
    assert manifest["full_method_key_opened_by_this_process_after_original_only_checkpoint"] is True
    assert manifest["all_inputs_revalidated_immediately_before_publish"] is True
    snapshot = output / "implementation_snapshot.py"
    assert stat.S_IMODE(snapshot.stat().st_mode) == 0o444
    assert snapshot.read_bytes() == Path(exploratory.__file__).read_bytes()
    assert manifest["code_provenance"]["implementation_snapshot"] == {
        "path": "implementation_snapshot.py",
        "sha256": exploratory.sha256_file(snapshot),
        "size_bytes": snapshot.stat().st_size,
    }
    for name, formats in manifest["tables"].items():
        expected_rows = len(json.loads((output / formats["json"]["path"]).read_text()))
        assert formats["csv"]["row_count"] == expected_rows
        assert formats["json"]["row_count"] == expected_rows
        for ref in formats.values():
            path = output / ref["path"]
            assert ref["sha256"] == exploratory.sha256_file(path)
            assert ref["size_bytes"] == path.stat().st_size
    assert stat.S_IMODE((output / "results.json").stat().st_mode) == 0o444
    with pytest.raises(exploratory.ExploratoryMetricsError, match="fresh-only"):
        exploratory.compute_exploratory_metrics(
            provisional_scores_path=provisional_root / "provisional_scores.jsonl",
            original_key_path=inputs["original_key"],
            full_key_path=inputs["full_key"],
            key_commitments_path=inputs["key_commitments"],
            formal_cases_path=FORMAL_CASES,
            output_root=output,
        )


def test_full_key_commitment_is_checked_only_after_original_checkpoint(tmp_path: Path):
    inputs, provisional_root = _prepare_provisional(tmp_path)
    full_key = inputs["full_key"]
    full_key.write_bytes(full_key.read_bytes() + b"\n")
    output = tmp_path / "refused"
    with pytest.raises(exploratory.ExploratoryMetricsError, match="full method key differs"):
        exploratory.compute_exploratory_metrics(
            provisional_scores_path=provisional_root / "provisional_scores.jsonl",
            original_key_path=inputs["original_key"],
            full_key_path=full_key,
            key_commitments_path=inputs["key_commitments"],
            formal_cases_path=FORMAL_CASES,
            output_root=output,
        )
    assert not output.exists()


def test_forged_provisional_and_receipt_cannot_replace_authoritative_ab_rows(
    tmp_path: Path,
):
    inputs, provisional_root = _prepare_provisional(tmp_path)
    scores_path = provisional_root / "provisional_scores.jsonl"
    rows = canonical.load_jsonl(scores_path, "provisional scores")
    target = next(row for row in rows if row["case_kind"] == "causal")
    target["source_scores"]["pass_b"]["source_visibility"] = 1
    target["scores"]["source_visibility"] = (
        target["source_scores"]["pass_a"]["source_visibility"] + 1
    ) / 2
    scores_path.chmod(0o644)
    scores_path.write_bytes(
        b"".join(exploratory.canonical_json_bytes(row) for row in rows)
    )
    scores_path.chmod(0o444)
    receipt_path = provisional_root / "provisional_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["artifacts"]["provisional_scores"]["sha256"] = exploratory.sha256_file(
        scores_path
    )
    receipt["artifacts"]["provisional_scores"]["size_bytes"] = scores_path.stat().st_size
    receipt_path.chmod(0o644)
    receipt_path.write_bytes(exploratory.canonical_json_bytes(receipt))
    receipt_path.chmod(0o444)

    with pytest.raises(
        exploratory.ExploratoryMetricsError,
        match="source scores differ from authoritative A/B rows",
    ):
        exploratory.compute_exploratory_metrics(
            provisional_scores_path=scores_path,
            original_key_path=inputs["original_key"],
            full_key_path=inputs["full_key"],
            key_commitments_path=inputs["key_commitments"],
            formal_cases_path=FORMAL_CASES,
            output_root=tmp_path / "forged_refused",
        )
