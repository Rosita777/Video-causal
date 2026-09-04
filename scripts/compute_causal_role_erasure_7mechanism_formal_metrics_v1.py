#!/usr/bin/env python3
"""Compute frozen CES/SU tables and inference for the seven-mechanism study."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

try:
    import causal_role_erasure_7mechanism_evaluation_code_registry_v1 as evaluation_code
except ModuleNotFoundError:
    from scripts import causal_role_erasure_7mechanism_evaluation_code_registry_v1 as evaluation_code


PROTOCOL = "causal_role_erasure_7m_formal_metrics_v1"
CANONICAL_PROTOCOL = "causal_role_erasure_7m_human_canonicalization_v1"
REVIEW_PACKAGE_PROTOCOL = "causal_role_erasure_7m_anonymous_review_v1"
PROTOCOL_VERSION = "causal_role_erasure_7m_single_seed_v2"
MECHANISMS = (
    "water_impact",
    "rigid_collision",
    "brittle_fracture",
    "powder_impact",
    "elastic_deformation",
    "material_release",
    "surface_trace",
)
MAIN_STREAMS = (
    "wan_original",
    "matched_control",
    "V4",
    "cogvideox_original",
    "negative_prompt",
    "videoeraser_official",
    "t2vunlearning_adapted",
    "safree_cogvideox",
)
IDENTIFICATION_STREAMS = ("generic_paraphrase", "bystander_token")
WAN_STREAMS = ("wan_original", "matched_control", "V4", *IDENTIFICATION_STREAMS)
COG_STREAMS = (
    "cogvideox_original",
    "negative_prompt",
    "videoeraser_official",
    "t2vunlearning_adapted",
    "safree_cogvideox",
)
HEADLINE_STREAMS = (
    ("matched_control", "matched_control"),
    ("negative_prompt", "negative_prompt"),
    ("videoeraser_official", "videoeraser_official"),
    ("t2vunlearning_adapted", "t2vunlearning_adapted"),
    ("safree_cogvideox", "safree_cogvideox"),
)
BOOTSTRAP_REPLICATES = 10_000
HEADLINE_SEED = 8_202_601
IDENTIFICATION_SEED = 8_202_602
MIN_SHARED_CASES_PER_MECHANISM = 12
NI_MARGINS = {
    "specificity_utility": -0.10,
    "receiver_preservation": -0.10,
    "video_quality": -0.10,
    "usable_fraction": -0.05,
}


class FormalMetricsError(ValueError):
    """The canonical table, private key, or registered statistic is invalid."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise FormalMetricsError(message)


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def regular_file(path: Path, label: str) -> None:
    require(path.is_file() and not path.is_symlink(), f"{label} missing or symlinked: {path}")


def load_json(path: Path, label: str) -> dict[str, Any]:
    regular_file(path, label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FormalMetricsError(f"{label} is invalid JSON") from exc
    require(isinstance(value, dict), f"{label} must be an object")
    return value


def load_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    regular_file(path, label)
    rows = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        require(line and line.strip() == line, f"{label} row {index} is blank/noncanonical")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise FormalMetricsError(f"{label} row {index} is invalid JSON") from exc
        require(isinstance(row, dict), f"{label} row {index} is not an object")
        rows.append(row)
    return rows


def load_csv(path: Path, label: str) -> list[dict[str, str]]:
    regular_file(path, label)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames is not None, f"{label} has no header")
        return [dict(row) for row in reader]


def write_bytes(path: Path, value: bytes, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(value)
    path.chmod(mode)


def write_json(path: Path, value: Any) -> None:
    write_bytes(path, canonical_json_bytes(value))


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    write_bytes(path, b"".join(canonical_json_bytes(dict(row)) for row in rows))


def csv_bytes(rows: Sequence[Mapping[str, Any]], fields: Sequence[str] | None = None) -> bytes:
    import io

    if fields is None:
        ordered: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for field in row:
                if field not in seen:
                    seen.add(field)
                    ordered.append(field)
        fields = tuple(ordered)
    else:
        fields = tuple(fields)
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(fields), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def relative_ref(root: Path, path: Path) -> dict[str, Any]:
    return {"path": str(path.relative_to(root)), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}


def _resolve_ref(root: Path, ref: Any, label: str, expected: Path | None = None) -> Path:
    require(isinstance(ref, dict) and set(ref) >= {"path", "sha256"}, f"{label} reference malformed")
    value = Path(str(ref["path"]))
    path = value if value.is_absolute() else root / value
    regular_file(path, label)
    path = path.resolve()
    if expected is not None:
        require(path == expected.resolve(), f"{label} path mismatch")
    require(sha256_file(path) == ref["sha256"], f"{label} SHA mismatch")
    return path


def _mean(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    require(bool(rows), f"cannot average empty {field}")
    return float(sum(float(row[field]) for row in rows) / len(rows))


def load_canonical_inputs(
    *,
    canonical_scores_path: Path,
    full_key_path: Path,
    key_commitments_path: Path,
    original_eligibility_path: Path,
    shared_subsets_path: Path,
    formal_cases_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    manifest = load_json(canonical_scores_path.parent / "canonical_manifest.json", "canonical manifest")
    require(
        manifest.get("protocol") == CANONICAL_PROTOCOL
        and manifest.get("status")
        == "canonical_anonymous_scores_frozen_before_answer_key_opening"
        and manifest.get("row_count") == 2448
        and manifest.get("full_method_key_opened_for_scoring") is False,
        "canonical manifest identity/status mismatch",
    )
    _resolve_ref(
        canonical_scores_path.parent,
        manifest["inputs"]["key_commitments"],
        "pre-canonical key commitments",
        key_commitments_path,
    )
    commitments = load_json(key_commitments_path, "public key commitments")
    code_registry_path = key_commitments_path.parent / "evaluation_code_registry.json"
    code_registry = load_json(code_registry_path, "evaluation code registry")
    try:
        evaluation_code.validate_registry(
            code_registry, Path(__file__).resolve().parents[1]
        )
    except ValueError as exc:
        raise FormalMetricsError(str(exc)) from exc
    require(
        commitments.get("protocol") == REVIEW_PACKAGE_PROTOCOL
        and commitments.get("schema_version") == 1
        and commitments.get("commitment_scheme") == "sha256(canonical-jsonl-bytes)"
        and commitments.get("tier_2_full", {}).get("row_count") == 2448
        and commitments.get("tier_2_full", {}).get("sha256") == sha256_file(full_key_path),
        "full method key differs from its pre-canonical commitment",
    )
    require(
        commitments.get("evaluation_code_registry", {}).get("sha256")
        == sha256_file(code_registry_path)
        and commitments.get("evaluation_code_registry", {}).get(
            "registry_sha256"
        )
        == code_registry["registry_sha256"]
        and manifest.get("evaluation_code_registry_sha256")
        == code_registry["registry_sha256"],
        "canonical evaluation-code registry binding changed",
    )
    audit_strata_path = _resolve_ref(
        canonical_scores_path.parent,
        manifest["inputs"]["audit_strata_key"],
        "pre-canonical opaque audit-strata key",
    )
    require(
        commitments.get("tier_0_audit_strata", {}).get("row_count") == 2448
        and commitments.get("tier_0_audit_strata", {}).get("sha256")
        == sha256_file(audit_strata_path),
        "opaque audit-strata key differs from its commitment",
    )
    _resolve_ref(canonical_scores_path.parent, manifest["artifacts"]["canonical_scores"], "canonical scores", canonical_scores_path)
    eligibility_manifest_path = original_eligibility_path.parent / "eligibility_manifest.json"
    eligibility_manifest = load_json(eligibility_manifest_path, "eligibility manifest")
    require(
        eligibility_manifest.get("protocol") == CANONICAL_PROTOCOL
        and eligibility_manifest.get("status")
        == "original_eligibility_and_shared_subsets_frozen_before_full_key_opening"
        and eligibility_manifest.get("full_method_key_opened") is False,
        "eligibility manifest identity/status mismatch",
    )
    require(
        eligibility_manifest.get("evaluation_code_registry_sha256")
        == code_registry["registry_sha256"],
        "eligibility evaluation-code registry binding changed",
    )
    _resolve_ref(
        eligibility_manifest_path.parent,
        eligibility_manifest["inputs"]["canonical_manifest"],
        "eligibility canonical manifest",
        canonical_scores_path.parent / "canonical_manifest.json",
    )
    _resolve_ref(
        eligibility_manifest_path.parent,
        eligibility_manifest["inputs"]["key_commitments"],
        "eligibility key commitments",
        key_commitments_path,
    )
    original_key_path = _resolve_ref(
        eligibility_manifest_path.parent,
        eligibility_manifest["inputs"]["original_only_key"],
        "eligibility Original-only key",
    )
    _resolve_ref(eligibility_manifest_path.parent, eligibility_manifest["artifacts"]["original_eligibility"], "Original eligibility", original_eligibility_path)
    _resolve_ref(eligibility_manifest_path.parent, eligibility_manifest["artifacts"]["shared_capability_subsets"], "shared subsets", shared_subsets_path)
    canonical = load_jsonl(canonical_scores_path, "canonical scores")
    key = load_jsonl(full_key_path, "full method key")
    audit_strata = load_jsonl(audit_strata_path, "opaque audit-strata key")
    original_key = load_jsonl(original_key_path, "Original-only key")
    eligibility = load_csv(original_eligibility_path, "Original eligibility")
    shared = load_csv(shared_subsets_path, "shared capability subsets")
    cases = load_csv(formal_cases_path, "formal cases")
    require(len(canonical) == len(key) == 2448, "canonical/full-key inventory must be exactly 2448")
    key_by_id = {row.get("anonymous_review_id"): row for row in key}
    require(
        commitments.get("tier_1_original_only", {}).get("row_count") == 588
        and commitments.get("tier_1_original_only", {}).get("sha256")
        == sha256_file(original_key_path)
        and len(original_key) == 588
        and all(row == key_by_id.get(row.get("anonymous_review_id")) for row in original_key),
        "Original-only key is not the exact precommitted Original subset of the full key",
    )
    require(
        Counter(row.get("stream") for row in original_key)
        == {"wan_original": 294, "cogvideox_original": 294}
        and all(
            row.get("backbone_family")
            == ("wan" if row.get("stream") == "wan_original" else "cogvideox")
            for row in original_key
        ),
        "Original-only key stream/backbone inventory changed",
    )
    strata_by_id = {row.get("anonymous_review_id"): row for row in audit_strata}
    require(
        len(audit_strata) == len(strata_by_id) == 2448
        and set(strata_by_id) == set(key_by_id),
        "opaque audit-strata/full-key ID inventory differs",
    )
    token_to_stream: dict[str, tuple[str, str]] = {}
    stream_to_token: dict[tuple[str, str], str] = {}
    for review_id, binding in key_by_id.items():
        stratum = strata_by_id[review_id]
        require(
            stratum.get("mechanism") == binding.get("mechanism")
            and stratum.get("case_kind") == binding.get("case_kind"),
            f"opaque audit stratum semantic mismatch: {review_id}",
        )
        token = str(stratum.get("stream_stratum", ""))
        stream = (str(binding.get("evaluation_partition", "")), str(binding.get("stream", "")))
        require(token not in token_to_stream or token_to_stream[token] == stream, "one opaque token maps to multiple streams")
        require(stream not in stream_to_token or stream_to_token[stream] == token, "one stream maps to multiple opaque tokens")
        token_to_stream[token] = stream
        stream_to_token[stream] = token
    require(len(token_to_stream) == len(stream_to_token) == 10, "opaque stream strata do not bijectively recover ten streams")
    require(len(eligibility) == 336 and len(shared) == 168 and len(cases) == 294, "case/eligibility inventory count changed")
    ids = [str(row.get("anonymous_review_id", "")) for row in canonical]
    key_ids = [str(row.get("anonymous_review_id", "")) for row in key]
    require(len(set(ids)) == len(ids) and set(ids) == set(key_ids), "canonical/full-key anonymous IDs differ")
    require(len(set(key_ids)) == len(key_ids), "full-key anonymous IDs repeat")
    canonical_by_id = {row["anonymous_review_id"]: row for row in canonical}
    for index, row in enumerate(canonical):
        require(
            set(row)
            == {
                "anonymous_review_id",
                "assignment_sha256",
                "case_kind",
                "scores",
                "canonical_sources",
                "status",
            }
            and row["status"] == "canonical_frozen",
            f"canonical row {index} schema/status changed",
        )
        expected_fields = (
            {"source_visibility", "footprint_visibility", "receiver_preservation", "video_quality"}
            if row["case_kind"] == "causal"
            else {"protected_object_visibility", "noncausal_role_adherence", "receiver_preservation", "video_quality"}
            if row["case_kind"] == "specificity"
            else set()
        )
        require(
            expected_fields
            and isinstance(row["scores"], dict)
            and set(row["scores"]) == expected_fields
            and all(type(value) is int and value in (0, 1, 2) for value in row["scores"].values())
            and isinstance(row["canonical_sources"], dict)
            and set(row["canonical_sources"]) == expected_fields,
            f"canonical row {index} atomic schema changed",
        )
    stream_counts = Counter(
        (row.get("evaluation_partition"), row.get("stream")) for row in key
    )
    require(
        stream_counts
        == Counter(
            {**{("main", stream): 294 for stream in MAIN_STREAMS}, **{("identification", stream): 48 for stream in IDENTIFICATION_STREAMS}}
        ),
        "full key stream inventory changed",
    )
    require({row.get("protocol_version") for row in cases} == {PROTOCOL_VERSION}, "formal case protocol changed")
    require(
        Counter((row["mechanism"], row["case_kind"]) for row in cases)
        == Counter({**{(mechanism, "causal"): 24 for mechanism in MECHANISMS}, **{(mechanism, "specificity"): 18 for mechanism in MECHANISMS}}),
        "formal case mechanism/kind balance changed",
    )
    eligibility_pairs = [(row["case_id"], row["backbone_family"]) for row in eligibility]
    require(len(set(eligibility_pairs)) == 336, "Original eligibility pairs repeat")
    require(all(row["eligible"] in ("0", "1") for row in eligibility), "Original eligibility is not binary")
    eligibility_by_id = {row["anonymous_review_id"]: row for row in eligibility}
    original_causal = [row for row in original_key if row["case_kind"] == "causal"]
    require(
        len(eligibility_by_id) == len(original_causal) == 336
        and set(eligibility_by_id)
        == {row["anonymous_review_id"] for row in original_causal},
        "Original eligibility anonymous inventory changed",
    )
    for binding in original_causal:
        review_id = binding["anonymous_review_id"]
        score = canonical_by_id[review_id]["scores"]
        observed = eligibility_by_id[review_id]
        recomputed = int(
            score["source_visibility"] == 2
            and score["footprint_visibility"] >= 1
            and score["receiver_preservation"] >= 1
            and score["video_quality"] >= 1
        )
        expected_backbone = "wan" if binding["stream"] == "wan_original" else "cogvideox"
        require(
            observed["backbone_family"] == expected_backbone
            and observed["case_id"] == binding["case_id"]
            and observed["mechanism"] == binding["mechanism"]
            and int(observed["eligible"]) == recomputed
            and all(int(observed[field]) == score[field] for field in ("source_visibility", "footprint_visibility", "receiver_preservation", "video_quality")),
            f"Original eligibility does not independently reproduce canonical scores: {review_id}",
        )
    eligibility_map = {(row["case_id"], row["backbone_family"]): row["eligible"] for row in eligibility}
    shared_by_case = {row["case_id"]: row for row in shared}
    require(len(shared_by_case) == 168, "shared capability case IDs repeat")
    for row in shared:
        pair = (row["case_id"], row["mechanism"])
        require(
            row["wan_original_eligible"] in ("0", "1")
            and row["cogvideox_original_eligible"] in ("0", "1")
            and row["wan_original_eligible"] == eligibility_map[(row["case_id"], "wan")]
            and row["cogvideox_original_eligible"] == eligibility_map[(row["case_id"], "cogvideox")]
            and row["shared_eligible"]
            == str(int(row["wan_original_eligible"] == "1" and row["cogvideox_original_eligible"] == "1")),
            f"shared capability row changed: {pair}",
        )
    return canonical, key, eligibility, shared, cases


def build_case_metrics(
    canonical: Sequence[Mapping[str, Any]],
    key: Sequence[Mapping[str, Any]],
    eligibility: Sequence[Mapping[str, str]],
    cases: Sequence[Mapping[str, str]],
) -> list[dict[str, Any]]:
    canonical_by_id = {row["anonymous_review_id"]: row for row in canonical}
    eligibility_by_pair = {
        (row["case_id"], row["backbone_family"]): int(row["eligible"])
        for row in eligibility
    }
    case_by_id = {row["case_id"]: row for row in cases}
    rows: list[dict[str, Any]] = []
    seen_bindings: set[tuple[str, str, str]] = set()
    for binding in key:
        review_id = binding["anonymous_review_id"]
        score_row = canonical_by_id[review_id]
        case = case_by_id.get(binding["case_id"])
        require(case is not None, f"full key case is absent from formal table: {binding['case_id']}")
        identity = (binding["evaluation_partition"], binding["stream"], binding["case_id"])
        require(identity not in seen_bindings, f"full key repeats a stream/case binding: {identity}")
        seen_bindings.add(identity)
        for field in (
            "mechanism",
            "case_kind",
            "generalization_group",
            "source_membership",
            "prompt_style",
            "footprint_lexicalization",
            "specificity_subtype",
            "m6_pair_id",
            "source_id",
            "receiver_id",
        ):
            require(str(binding.get(field, "")) == str(case.get(field, "")), f"{review_id}: formal {field} binding mismatch")
        require(
            score_row["case_kind"] == binding["case_kind"]
            and int(binding["formal_global_index"]) == int(case["global_case_index"])
            and int(binding["seed"]) == int(case["seed"]),
            f"{review_id}: formal score-kind/index/seed binding mismatch",
        )
        scores = score_row["scores"]
        backbone = "cogvideox" if binding["stream"] in COG_STREAMS else "wan"
        usable = int(scores["receiver_preservation"] >= 1 and scores["video_quality"] >= 1)
        common: dict[str, Any] = {
            "anonymous_review_id": review_id,
            "evaluation_partition": binding["evaluation_partition"],
            "stream": binding["stream"],
            "backbone_family": backbone,
            "case_id": binding["case_id"],
            "mechanism": binding["mechanism"],
            "case_kind": binding["case_kind"],
            "seed": binding["seed"],
            "m6_pair_id": binding.get("m6_pair_id", ""),
            "footprint_lexicalization": binding.get("footprint_lexicalization", ""),
            "source_membership": binding.get("source_membership", ""),
            "generalization_group": binding.get("generalization_group", ""),
            "usable": usable,
            "receiver_preservation_norm": scores["receiver_preservation"] / 2,
            "video_quality_norm": scores["video_quality"] / 2,
            "original_eligible": "",
            "ces": "",
            "su": "",
            "source_absent": "",
            "footprint_absent": "",
            "strict_success": "",
            **scores,
        }
        if binding["case_kind"] == "causal":
            e_b = eligibility_by_pair[(binding["case_id"], backbone)]
            ces = e_b * usable * ((2 - scores["source_visibility"]) + (2 - scores["footprint_visibility"])) / 4
            common.update(
                {
                    "original_eligible": e_b,
                    "ces": ces,
                    "source_absent": int(scores["source_visibility"] == 0),
                    "footprint_absent": int(scores["footprint_visibility"] == 0),
                    "strict_success": int(
                        e_b == 1
                        and scores["source_visibility"] == 0
                        and scores["footprint_visibility"] == 0
                        and scores["receiver_preservation"] == 2
                        and scores["video_quality"] == 2
                    ),
                }
            )
        else:
            common["su"] = usable * (
                scores["protected_object_visibility"] + scores["noncausal_role_adherence"]
            ) / 4
        rows.append(common)
    counts = Counter((row["evaluation_partition"], row["stream"]) for row in rows)
    require(
        counts
        == Counter(
            {**{("main", stream): 294 for stream in MAIN_STREAMS}, **{("identification", stream): 48 for stream in IDENTIFICATION_STREAMS}}
        ),
        "de-anonymized stream inventory differs from 8x294 + 2x48",
    )
    formal_ids = set(case_by_id)
    for stream in MAIN_STREAMS:
        require(
            {row["case_id"] for row in rows if row["evaluation_partition"] == "main" and row["stream"] == stream}
            == formal_ids,
            f"{stream}: formal case coverage changed",
        )
    identification_sets = [
        {row["case_id"] for row in rows if row["evaluation_partition"] == "identification" and row["stream"] == stream}
        for stream in IDENTIFICATION_STREAMS
    ]
    require(
        identification_sets[0] == identification_sets[1]
        and len(identification_sets[0]) == 48,
        "identification control case sets differ",
    )
    return rows


def mechanism_and_macro_tables(rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    main = [row for row in rows if row["evaluation_partition"] == "main"]
    by_stream_case = {(row["stream"], row["case_id"]): row for row in main}
    mechanism_rows: list[dict[str, Any]] = []
    for stream in MAIN_STREAMS:
        for mechanism in MECHANISMS:
            block = [row for row in main if row["stream"] == stream and row["mechanism"] == mechanism]
            causal = [row for row in block if row["case_kind"] == "causal"]
            spec = [row for row in block if row["case_kind"] == "specificity"]
            require(len(causal) == 24 and len(spec) == 18, f"{stream}/{mechanism}: expected 24 causal + 18 specificity")
            original_stream = "cogvideox_original" if stream in COG_STREAMS else "wan_original"
            original_rows = [by_stream_case[(original_stream, row["case_id"])] for row in causal]
            eligible_pairs = [(row, original) for row, original in zip(causal, original_rows) if original["original_eligible"] == 1]
            mechanism_rows.append(
                {
                    "stream": stream,
                    "mechanism": mechanism,
                    "causal_n": 24,
                    "specificity_n": 18,
                    "ces": _mean(causal, "ces"),
                    "specificity_utility": _mean(spec, "su"),
                    "original_capability_coverage": _mean(causal, "original_eligible"),
                    "causal_usable_fraction": _mean(causal, "usable"),
                    "specificity_usable_fraction": _mean(spec, "usable"),
                    "causal_receiver_preservation": _mean(causal, "receiver_preservation_norm"),
                    "causal_video_quality": _mean(causal, "video_quality_norm"),
                    "specificity_receiver_preservation": _mean(spec, "receiver_preservation_norm"),
                    "specificity_video_quality": _mean(spec, "video_quality_norm"),
                    "source_absent_rate": _mean(causal, "source_absent"),
                    "footprint_absent_rate": _mean(causal, "footprint_absent"),
                    "strict_success_rate": _mean(causal, "strict_success"),
                    "eligible_source_clear_to_partial_rate": (
                        sum(row["source_visibility"] == 1 for row, _ in eligible_pairs) / len(eligible_pairs)
                        if eligible_pairs
                        else ""
                    ),
                    "eligible_source_clear_to_absent_rate": (
                        sum(row["source_visibility"] == 0 for row, _ in eligible_pairs) / len(eligible_pairs)
                        if eligible_pairs
                        else ""
                    ),
                }
            )
    macro_rows: list[dict[str, Any]] = []
    metric_fields = tuple(field for field in mechanism_rows[0] if field not in ("stream", "mechanism", "causal_n", "specificity_n"))
    for stream in MAIN_STREAMS:
        block = [row for row in mechanism_rows if row["stream"] == stream]
        require(len(block) == 7, f"{stream}: mechanism macro is incomplete")
        macro = {"stream": stream, "mechanism_count": 7}
        for field in metric_fields:
            values = [float(row[field]) for row in block if row[field] != ""]
            macro[field] = sum(values) / len(values) if len(values) == 7 else ""
        macro_rows.append(macro)
    return mechanism_rows, macro_rows


def bootstrap_stratified(
    groups: Mapping[str, Sequence[float]],
    *,
    seed: int,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> np.ndarray:
    require(tuple(groups) == tuple(mechanism for mechanism in MECHANISMS if mechanism in groups), "bootstrap groups are not in frozen mechanism order")
    require(groups and all(len(values) > 0 for values in groups.values()), "bootstrap groups are empty")
    require(replicates == BOOTSTRAP_REPLICATES, "formal inference requires exactly 10,000 bootstrap replicates")
    rng = np.random.Generator(np.random.PCG64(seed))
    mechanism_samples = []
    for values in groups.values():
        array = np.asarray(values, dtype=np.float64)
        indices = rng.integers(0, len(array), size=(replicates, len(array)))
        mechanism_samples.append(array[indices].mean(axis=1))
    return np.stack(mechanism_samples, axis=0).mean(axis=0)


def bootstrap_summary(
    groups: Mapping[str, Sequence[float]],
    *,
    seed: int,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    samples = bootstrap_stratified(groups, seed=seed, replicates=replicates)
    point = float(np.mean([np.mean(values) for values in groups.values()]))
    return {
        "point_estimate": point,
        "ci95_lower": float(np.quantile(samples, 0.025)),
        "ci95_upper": float(np.quantile(samples, 0.975)),
        "one_sided_p": float((1 + np.count_nonzero(samples <= 0)) / (replicates + 1)),
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
    }


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(p_values, key=lambda name: (p_values[name], name))
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for rank, name in enumerate(ordered):
        running = max(running, min(1.0, (total - rank) * p_values[name]))
        adjusted[name] = running
    return adjusted


def headline_contrasts(
    rows: Sequence[Mapping[str, Any]],
    shared: Sequence[Mapping[str, str]],
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool]:
    main_causal = [row for row in rows if row["evaluation_partition"] == "main" and row["case_kind"] == "causal"]
    by = {(row["stream"], row["case_id"]): row for row in main_causal}
    shared_ids = {row["case_id"] for row in shared if int(row["shared_eligible"]) == 1}
    headline: list[dict[str, Any]] = []
    mechanism_output: list[dict[str, Any]] = []
    for name, baseline in HEADLINE_STREAMS:
        groups: dict[str, list[float]] = {}
        all_estimable = True
        for mechanism in MECHANISMS:
            case_ids = sorted({row["case_id"] for row in main_causal if row["stream"] == "V4" and row["mechanism"] == mechanism})
            if baseline == "matched_control":
                selected = case_ids
                values = [float(by[("V4", case_id)]["ces"]) - float(by[(baseline, case_id)]["ces"]) for case_id in selected]
                estimable = len(values) == 24
            else:
                selected = [case_id for case_id in case_ids if case_id in shared_ids]
                values = [
                    (float(by[("V4", case_id)]["ces"]) - float(by[("wan_original", case_id)]["ces"]))
                    - (float(by[(baseline, case_id)]["ces"]) - float(by[("cogvideox_original", case_id)]["ces"]))
                    for case_id in selected
                ]
                estimable = len(values) >= MIN_SHARED_CASES_PER_MECHANISM
            mechanism_output.append(
                {
                    "contrast": name,
                    "mechanism": mechanism,
                    "n": len(values),
                    "estimable": int(estimable),
                    "point_estimate": float(np.mean(values)) if values else "",
                }
            )
            all_estimable = all_estimable and estimable
            if estimable:
                groups[mechanism] = values
        if all_estimable and len(groups) == 7:
            summary = bootstrap_summary(groups, seed=HEADLINE_SEED, replicates=replicates)
            headline.append({"contrast": name, "estimable": 1, **summary})
        else:
            headline.append(
                {
                    "contrast": name,
                    "estimable": 0,
                    "point_estimate": "",
                    "ci95_lower": "",
                    "ci95_upper": "",
                    "one_sided_p": 1.0,
                    "bootstrap_replicates": replicates,
                    "bootstrap_seed": HEADLINE_SEED,
                }
            )
    adjusted = holm_adjust({row["contrast"]: float(row["one_sided_p"]) for row in headline})
    for row in headline:
        row["holm_adjusted_p"] = adjusted[row["contrast"]]
        row["registered_win"] = int(
            row["estimable"] == 1
            and float(row["point_estimate"]) > 0
            and float(row["ci95_lower"]) > 0
            and row["holm_adjusted_p"] < 0.05
        )
    return headline, mechanism_output, all(row["registered_win"] == 1 for row in headline)


def noninferiority_table(
    rows: Sequence[Mapping[str, Any]],
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> tuple[list[dict[str, Any]], bool]:
    main = [row for row in rows if row["evaluation_partition"] == "main" and row["stream"] in ("V4", "matched_control")]
    by = {(row["stream"], row["case_id"]): row for row in main}
    specs = (
        ("specificity_utility", "specificity", "su", NI_MARGINS["specificity_utility"], "primary"),
        ("receiver_preservation", "causal", "receiver_preservation_norm", NI_MARGINS["receiver_preservation"], "primary"),
        ("video_quality", "causal", "video_quality_norm", NI_MARGINS["video_quality"], "primary"),
        ("usable_fraction", "causal", "usable", NI_MARGINS["usable_fraction"], "primary"),
        ("specificity_receiver_preservation", "specificity", "receiver_preservation_norm", NI_MARGINS["receiver_preservation"], "secondary"),
        ("specificity_video_quality", "specificity", "video_quality_norm", NI_MARGINS["video_quality"], "secondary"),
        ("specificity_usable_fraction", "specificity", "usable", NI_MARGINS["usable_fraction"], "secondary"),
    )
    output = []
    for name, kind, field, margin, role in specs:
        groups: dict[str, list[float]] = {}
        for mechanism in MECHANISMS:
            cases = sorted(row["case_id"] for row in main if row["stream"] == "V4" and row["mechanism"] == mechanism and row["case_kind"] == kind)
            expected = 24 if kind == "causal" else 18
            require(len(cases) == expected, f"{name}/{mechanism}: paired inventory mismatch")
            groups[mechanism] = [float(by[("V4", case_id)][field]) - float(by[("matched_control", case_id)][field]) for case_id in cases]
        summary = bootstrap_summary(groups, seed=HEADLINE_SEED, replicates=replicates)
        output.append(
            {
                "metric": name,
                "role": role,
                "margin": margin,
                **summary,
                "noninferior": int(summary["ci95_lower"] > margin),
            }
        )
    allowed = all(row["noninferior"] == 1 for row in output if row["role"] == "primary")
    return output, allowed


def identification_table(
    rows: Sequence[Mapping[str, Any]],
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> tuple[list[dict[str, Any]], bool]:
    by = {(row["stream"], row["case_id"]): row for row in rows}
    controls = [row for row in rows if row["evaluation_partition"] == "identification"]
    causal_results: list[dict[str, Any]] = []
    for control in IDENTIFICATION_STREAMS:
        groups: dict[str, list[float]] = {}
        implicit_values: list[float] = []
        implicit_by_mechanism: dict[str, list[float]] = defaultdict(list)
        for mechanism in ("water_impact", "brittle_fracture"):
            control_rows = sorted(
                (
                    row
                    for row in controls
                    if row["stream"] == control
                    and row["mechanism"] == mechanism
                    and row["case_kind"] == "causal"
                ),
                key=lambda row: row["case_id"],
            )
            require(len(control_rows) == 12, f"{control}/{mechanism}: expected 12 identification causal cases")
            values = [float(by[("V4", row["case_id"])]["ces"]) - float(row["ces"]) for row in control_rows]
            groups[mechanism] = values
            implicit_by_mechanism[mechanism] = [
                float(by[("V4", row["case_id"])]["ces"]) - float(row["ces"])
                for row in control_rows
                if row["footprint_lexicalization"] == "implicit"
            ]
            require(len(implicit_by_mechanism[mechanism]) == 6, f"{control}/{mechanism}: expected six implicit cases")
            implicit_values.append(float(np.mean(implicit_by_mechanism[mechanism])))
        summary = bootstrap_summary(groups, seed=IDENTIFICATION_SEED, replicates=replicates)
        spec_groups: dict[str, list[float]] = {}
        for mechanism in ("water_impact", "brittle_fracture"):
            spec_rows = sorted(
                (
                    row
                    for row in controls
                    if row["stream"] == control
                    and row["mechanism"] == mechanism
                    and row["case_kind"] == "specificity"
                ),
                key=lambda row: row["case_id"],
            )
            require(len(spec_rows) == 12, f"{control}/{mechanism}: expected 12 identification specificity cases")
            spec_groups[mechanism] = [float(by[("V4", row["case_id"])]["su"]) - float(row["su"]) for row in spec_rows]
        spec_summary = bootstrap_summary(spec_groups, seed=IDENTIFICATION_SEED, replicates=replicates)
        causal_results.append(
            {
                "control": control,
                **summary,
                "implicit_two_mechanism_point_estimate": float(np.mean(implicit_values)),
                "implicit_water_point_estimate": implicit_values[0],
                "implicit_fracture_point_estimate": implicit_values[1],
                "specificity_margin": -0.10,
                "specificity_point_estimate": spec_summary["point_estimate"],
                "specificity_ci95_lower": spec_summary["ci95_lower"],
                "specificity_ci95_upper": spec_summary["ci95_upper"],
                "specificity_noninferior": int(spec_summary["ci95_lower"] > -0.10),
            }
        )
    adjusted = holm_adjust({row["control"]: row["one_sided_p"] for row in causal_results})
    for row in causal_results:
        row["holm_adjusted_p"] = adjusted[row["control"]]
        row["novelty_condition_pass"] = int(
            row["point_estimate"] > 0
            and row["ci95_lower"] > 0
            and row["holm_adjusted_p"] < 0.05
            and row["implicit_two_mechanism_point_estimate"] > 0
            and row["specificity_noninferior"] == 1
        )
    return causal_results, all(row["novelty_condition_pass"] == 1 for row in causal_results)


def main_role_conditioned_evidence(
    rows: Sequence[Mapping[str, Any]],
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> tuple[list[dict[str, Any]], bool]:
    by = {
        (row["stream"], row["case_id"]): row
        for row in rows
        if row["evaluation_partition"] == "main" and row["case_kind"] == "causal"
    }
    definitions = (
        ("implicit_footprint", lambda row: row["footprint_lexicalization"] == "implicit"),
        ("held_out_source", lambda row: row["source_membership"] == "eval_holdout"),
    )
    output = []
    for name, predicate in definitions:
        groups: dict[str, list[float]] = {}
        counts: dict[str, int] = {}
        for mechanism in MECHANISMS:
            selected = sorted(
                (
                    row
                    for row in by.values()
                    if row["stream"] == "V4"
                    and row["mechanism"] == mechanism
                    and predicate(row)
                ),
                key=lambda row: row["case_id"],
            )
            require(selected, f"{name}/{mechanism}: registered subset is empty")
            groups[mechanism] = [
                float(row["ces"])
                - float(by[("matched_control", row["case_id"])]["ces"])
                for row in selected
            ]
            counts[mechanism] = len(selected)
        summary = bootstrap_summary(groups, seed=HEADLINE_SEED, replicates=replicates)
        output.append(
            {
                "subset": name,
                "per_mechanism_n": counts,
                **summary,
                "registered_positive": int(summary["point_estimate"] > 0),
            }
        )
    return output, all(row["registered_positive"] == 1 for row in output)


def m6_table(rows: Sequence[Mapping[str, Any]], cases: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    pairs: dict[str, dict[str, str]] = defaultdict(dict)
    for case in cases:
        pair_id = case.get("m6_pair_id", "")
        if pair_id:
            require(case["case_kind"] not in pairs[pair_id], f"duplicate M6 {pair_id}/{case['case_kind']}")
            pairs[pair_id][case["case_kind"]] = case["case_id"]
    require(len(pairs) == 42 and all(set(pair) == {"causal", "specificity"} for pair in pairs.values()), "M6 inventory must be 6 pairs x 7 mechanisms")
    by = {(row["stream"], row["case_id"]): row for row in rows if row["evaluation_partition"] == "main"}
    output = []
    for stream in MAIN_STREAMS:
        for pair_id, pair in sorted(pairs.items()):
            causal = by[(stream, pair["causal"])]
            specificity = by[(stream, pair["specificity"])]
            output.append(
                {
                    "stream": stream,
                    "mechanism": causal["mechanism"],
                    "m6_pair_id": pair_id,
                    "causal_case_id": pair["causal"],
                    "specificity_case_id": pair["specificity"],
                    "causal_ces": causal["ces"],
                    "specificity_utility": specificity["su"],
                    "pair_mean": (float(causal["ces"]) + float(specificity["su"])) / 2,
                }
            )
    require(len(output) == 336, "M6 output must contain 8x7x6 rows")
    return output


def compute_formal_metrics(
    *,
    canonical_scores_path: Path,
    full_key_path: Path,
    key_commitments_path: Path,
    original_eligibility_path: Path,
    shared_subsets_path: Path,
    formal_cases_path: Path,
    output_root: Path,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    require(bootstrap_replicates == BOOTSTRAP_REPLICATES, "formal scoring requires 10,000 bootstrap replicates")
    canonical, key, eligibility, shared, cases = load_canonical_inputs(
        canonical_scores_path=canonical_scores_path,
        full_key_path=full_key_path,
        key_commitments_path=key_commitments_path,
        original_eligibility_path=original_eligibility_path,
        shared_subsets_path=shared_subsets_path,
        formal_cases_path=formal_cases_path,
    )
    case_metrics = build_case_metrics(canonical, key, eligibility, cases)
    mechanism, macro = mechanism_and_macro_tables(case_metrics)
    secondary_fields = (
        "stream",
        "mechanism",
        "original_capability_coverage",
        "causal_usable_fraction",
        "specificity_usable_fraction",
        "causal_receiver_preservation",
        "causal_video_quality",
        "specificity_receiver_preservation",
        "specificity_video_quality",
        "source_absent_rate",
        "footprint_absent_rate",
        "strict_success_rate",
        "eligible_source_clear_to_partial_rate",
        "eligible_source_clear_to_absent_rate",
    )
    secondary = [{field: row[field] for field in secondary_fields} for row in mechanism]
    headline, headline_mechanism, all_baselines = headline_contrasts(case_metrics, shared, replicates=bootstrap_replicates)
    noninferiority, retention = noninferiority_table(case_metrics, replicates=bootstrap_replicates)
    identification, identification_novelty = identification_table(case_metrics, replicates=bootstrap_replicates)
    role_evidence, main_role_positive = main_role_conditioned_evidence(
        case_metrics, replicates=bootstrap_replicates
    )
    matched_control_win = next(
        row["registered_win"] == 1
        for row in headline
        if row["contrast"] == "matched_control"
    )
    novelty = (
        identification_novelty
        and main_role_positive
        and retention
        and matched_control_win
    )
    m6 = m6_table(case_metrics, cases)
    output_root = output_root.resolve()
    require(not output_root.exists() and not output_root.is_symlink(), f"fresh-only output exists: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.tmp-", dir=output_root.parent))
    staging.chmod(0o700)
    try:
        tables = {
            "per_case_metrics": ("per_case_metrics.csv", case_metrics),
            "mechanism_metrics": ("mechanism_metrics.csv", mechanism),
            "macro_metrics": ("macro_metrics.csv", macro),
            "headline_contrasts": ("headline_contrasts.csv", headline),
            "headline_mechanism_contrasts": ("headline_mechanism_contrasts.csv", headline_mechanism),
            "noninferiority": ("noninferiority.csv", noninferiority),
            "identification": ("identification_contrasts.csv", identification),
            "role_conditioned_evidence": ("role_conditioned_evidence.csv", role_evidence),
            "m6_pairs": ("m6_pairs.csv", m6),
            "secondary_metrics": ("secondary_metrics.csv", secondary),
        }
        refs = {}
        for name, (filename, values) in tables.items():
            csv_path = staging / filename
            json_path = staging / filename.replace(".csv", ".json")
            write_bytes(csv_path, csv_bytes(values))
            write_json(json_path, values)
            refs[name] = {
                "csv": relative_ref(staging, csv_path),
                "json": relative_ref(staging, json_path),
            }
        results = {
            "schema_version": 1,
            "protocol": PROTOCOL,
            "status": "formal_metrics_complete",
            "evaluation_code_registry_sha256": evaluation_code.build_registry(
                Path(__file__).resolve().parents[1]
            )["registry_sha256"],
            "aggregation": "case_then_mechanism_mean_then_equal_weight_7_mechanism_macro",
            "metric_definitions": {
                "original_eligibility": "source_visibility==2 and footprint_visibility>=1 and receiver_preservation>=1 and video_quality>=1",
                "usable": "receiver_preservation>=1 and video_quality>=1",
                "CES": "E_b * usable * ((2-source_visibility)+(2-footprint_visibility))/4",
                "SU": "usable * (protected_object_visibility+noncausal_role_adherence)/4",
                "external_headline": "(V4-WanOriginal)-(baseline-CogVideoXOriginal) on pre-frozen shared-capability cases",
                "fixed_denominator_policy": "Original-ineligible causal cases remain present and contribute zero CES",
            },
            "bootstrap": {
                "replicates": BOOTSTRAP_REPLICATES,
                "headline_pcg64_seed": HEADLINE_SEED,
                "identification_pcg64_seed": IDENTIFICATION_SEED,
                "interval": "unadjusted_percentile_2.5_97.5",
            },
            "holm_families": {"headline": 5, "identification": 2},
            "claims": {
                "outperforms_all_registered_baselines": all_baselines,
                "retaining_allowed_by_all_primary_noninferiority_margins": retention,
                "role_conditioned_novelty_supported": novelty,
            },
            "headline_contrasts": headline,
            "noninferiority": noninferiority,
            "identification_contrasts": identification,
            "role_conditioned_main_evidence": role_evidence,
            "tables": refs,
        }
        results_path = staging / "results.json"
        write_json(results_path, results)
        manifest = {
            "schema_version": 1,
            "protocol": PROTOCOL,
            "status": "frozen_after_registered_10000_bootstrap_scoring",
            "evaluation_code_registry_sha256": results[
                "evaluation_code_registry_sha256"
            ],
            "inputs": {
                "canonical_scores": {"path": str(canonical_scores_path), "sha256": sha256_file(canonical_scores_path)},
                "full_method_key": {"path": str(full_key_path), "sha256": sha256_file(full_key_path)},
                "key_commitments": {"path": str(key_commitments_path), "sha256": sha256_file(key_commitments_path)},
                "evaluation_code_registry": {
                    "path": str(key_commitments_path.parent / "evaluation_code_registry.json"),
                    "sha256": sha256_file(
                        key_commitments_path.parent
                        / "evaluation_code_registry.json"
                    ),
                },
                "original_eligibility": {"path": str(original_eligibility_path), "sha256": sha256_file(original_eligibility_path)},
                "shared_capability_subsets": {"path": str(shared_subsets_path), "sha256": sha256_file(shared_subsets_path)},
                "formal_cases": {"path": str(formal_cases_path), "sha256": sha256_file(formal_cases_path)},
            },
            "results": relative_ref(staging, results_path),
            "tables": refs,
        }
        write_json(staging / "score_manifest.json", manifest)
        os.replace(staging, output_root)
        return results
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--canonical-scores", type=Path, required=True)
    parser.add_argument("--full-key", type=Path, required=True)
    parser.add_argument("--key-commitments", type=Path, required=True)
    parser.add_argument("--original-eligibility", type=Path, required=True)
    parser.add_argument("--shared-subsets", type=Path, required=True)
    parser.add_argument("--formal-cases", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = compute_formal_metrics(
            canonical_scores_path=args.canonical_scores.resolve(),
            full_key_path=args.full_key.resolve(),
            key_commitments_path=args.key_commitments.resolve(),
            original_eligibility_path=args.original_eligibility.resolve(),
            shared_subsets_path=args.shared_subsets.resolve(),
            formal_cases_path=args.formal_cases.resolve(),
            output_root=args.output_root.resolve(),
        )
    except FormalMetricsError as exc:
        parser.exit(2, f"formal metrics refused: {exc}\n")
    print(json.dumps(result["claims"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
