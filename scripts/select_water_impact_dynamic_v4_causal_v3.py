#!/usr/bin/env python3
"""Deterministic greedy/exact selector for the v4_dev72_v3 causal graph."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    import water_impact_dynamic_v4_eval_protocol_v3 as protocol
    import build_water_impact_dynamic_v4_causal_candidates_v3 as builder
except ModuleNotFoundError:  # imported as scripts.select_...
    from scripts import water_impact_dynamic_v4_eval_protocol_v3 as protocol
    from scripts import build_water_impact_dynamic_v4_causal_candidates_v3 as builder


ELIGIBILITY_HEADER = (
    "candidate_id",
    "semantic_case_id",
    "group",
    "prompt_variant",
    "source_visibility",
    "footprint_visibility",
    "receiver",
    "quality",
    "causal_link",
    "eligible",
)
SELECTED_PROTOCOL = "water_impact_dynamic_v4_causal_selected_manifest_v3"
UNIT_PROTOCOL = "water_impact_dynamic_v4_causal_unit_manifest_v3"
FORBIDDEN_PROTOCOL = "water_impact_dynamic_v4_forbidden_seed_inventory_v3"


class PreflightDatasetInvalid(RuntimeError):
    def __init__(self, reason_code: str, message: str):
        super().__init__(message)
        self.reason_code = reason_code


def _score(value: str, label: str) -> int:
    protocol.require(value in {"0", "1", "2"}, f"eligibility {label} invalid")
    return int(value)


def validate_eligibility_rows(
    rows: Sequence[Mapping[str, str]], candidates: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, Any], ...]:
    protocol.require(len(rows) == protocol.CANDIDATE_COUNT, "eligibility table must contain 576 rows")
    by_id = {str(row["case_id"]): row for row in candidates}
    protocol.require(len(by_id) == protocol.CANDIDATE_COUNT, "candidate ID inventory invalid")
    seen: set[str] = set()
    output: list[dict[str, Any]] = []
    for raw in rows:
        protocol.require(tuple(raw.keys()) == ELIGIBILITY_HEADER, "eligibility header/order is not exact")
        candidate_id = raw["candidate_id"]
        protocol.require(candidate_id == raw["semantic_case_id"] and candidate_id in by_id and candidate_id not in seen, "eligibility candidate binding invalid")
        seen.add(candidate_id)
        candidate = by_id[candidate_id]
        protocol.require(raw["group"] == candidate["group"] and raw["prompt_variant"] == candidate["prompt_variant"], "eligibility group/variant drift")
        source = _score(raw["source_visibility"], "source_visibility")
        footprint = _score(raw["footprint_visibility"], "footprint_visibility")
        receiver = _score(raw["receiver"], "receiver")
        quality = _score(raw["quality"], "quality")
        causal = _score(raw["causal_link"], "causal_link")
        expected = source == 2 and footprint >= 1 and receiver >= 1 and quality >= 1 and causal == 2
        protocol.require(raw["eligible"] == ("yes" if expected else "no"), "eligibility formula mismatch")
        output.append(
            {
                "candidate_id": candidate_id,
                "group": raw["group"],
                "prompt_variant": raw["prompt_variant"],
                "eligible": expected,
                "source_visibility": source,
                "footprint_visibility": footprint,
                "receiver": receiver,
                "quality": quality,
                "causal_link": causal,
            }
        )
    protocol.require(seen == set(by_id), "eligibility table coverage mismatch")
    return tuple(output)


def load_eligibility_csv(
    path: Path, *, private_root: Path | None = None
) -> tuple[dict[str, str], ...]:
    protocol.reject_forbidden_path(path)
    if private_root is not None:
        protocol.validate_private_path(private_root, path)
    elif not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"missing non-symlink eligibility CSV: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        protocol.require(tuple(reader.fieldnames or ()) == ELIGIBILITY_HEADER, "eligibility CSV header is not exact")
        return tuple(dict(row) for row in reader)


def _eligible_options(
    rows: Sequence[Mapping[str, Any]],
    *,
    group: str,
    forced: frozenset[str],
    excluded: frozenset[str],
) -> tuple[Mapping[str, Any], ...]:
    return tuple(
        row
        for row in rows
        if row["group"] == group
        and row["eligible"]
        and row["case_id"] not in excluded
        and (row["case_id"] in forced or row["case_id"] not in forced)
    )


def _validate_forced_common(
    options: Sequence[Mapping[str, Any]], forced: frozenset[str]
) -> tuple[Mapping[str, Any], ...] | None:
    by_id = {str(row["case_id"]): row for row in options}
    if not forced <= set(by_id):
        return None
    forced_rows = tuple(by_id[item] for item in sorted(forced))
    if len({row["receiver_id"] for row in forced_rows}) != len(forced_rows):
        return None
    if sum(row["prompt_variant"] == "direct" for row in forced_rows) > 4:
        return None
    if sum(row["prompt_variant"] == "natural" for row in forced_rows) > 4:
        return None
    return forced_rows


def g1_completion(
    options: Sequence[Mapping[str, Any]], forced: frozenset[str]
) -> tuple[str, ...] | None:
    forced_rows = _validate_forced_common(options, forced)
    if forced_rows is None or len(forced_rows) > 8:
        return None
    forced_heads = [str(row["physical_anchor_id"]) for row in forced_rows]
    if len(set(forced_heads)) != len(forced_heads):
        return None
    used_receivers = frozenset(str(row["receiver_id"]) for row in forced_rows)
    direct_used = sum(row["prompt_variant"] == "direct" for row in forced_rows)
    by_head: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in options:
        if row["case_id"] not in forced:
            by_head[str(row["physical_anchor_id"])].append(row)
    head_order = sorted(
        (head for head in by_head if head not in set(forced_heads)),
        key=lambda head: (len(by_head[head]), head),
    )
    for values in by_head.values():
        values.sort(key=lambda row: (str(row["selection_rank_sha256"]), str(row["case_id"])))

    target_more = 8 - len(forced_rows)
    direct_more = 4 - direct_used

    @lru_cache(maxsize=None)
    def search(
        index: int,
        selected: int,
        direct: int,
        receivers: tuple[str, ...],
    ) -> tuple[str, ...] | None:
        if selected == target_more:
            return () if direct == direct_more else None
        if index == len(head_order) or len(head_order) - index < target_more - selected:
            return None
        if direct > direct_more or direct + (target_more - selected) < direct_more:
            return None
        used = set(receivers)
        head = head_order[index]
        for row in by_head[head]:
            receiver_id = str(row["receiver_id"])
            is_direct = row["prompt_variant"] == "direct"
            if receiver_id in used or direct + int(is_direct) > direct_more:
                continue
            tail = search(
                index + 1,
                selected + 1,
                direct + int(is_direct),
                tuple(sorted((*used, receiver_id))),
            )
            if tail is not None:
                return (str(row["case_id"]), *tail)
        return search(index + 1, selected, direct, receivers)

    tail = search(0, 0, 0, tuple(sorted(used_receivers)))
    if tail is None:
        return None
    return tuple(str(row["case_id"]) for row in forced_rows) + tail


def _fixed_anchor_completion(
    options: Sequence[Mapping[str, Any]],
    forced: frozenset[str],
    *,
    group: str,
    receiver_matching: bool,
) -> tuple[str, ...] | None:
    forced_rows = _validate_forced_common(options, forced)
    if forced_rows is None or len(forced_rows) > 8:
        return None
    forced_by_anchor: dict[str, Mapping[str, Any]] = {}
    for row in forced_rows:
        anchor = str(row["physical_anchor_id"])
        if anchor in forced_by_anchor:
            return None
        forced_by_anchor[anchor] = row
    by_anchor_variant: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in options:
        by_anchor_variant[(str(row["physical_anchor_id"]), str(row["prompt_variant"]))].append(row)
    anchors = sorted({anchor for anchor, _ in by_anchor_variant})
    if len(anchors) != 8:
        return None
    for values in by_anchor_variant.values():
        values.sort(key=lambda row: (str(row["selection_rank_sha256"]), str(row["case_id"])))

    forced_direct = {anchor for anchor, row in forced_by_anchor.items() if row["prompt_variant"] == "direct"}
    forced_natural = set(forced_by_anchor) - forced_direct
    if len(forced_direct) > 4 or len(forced_natural) > 4:
        return None
    for direct_anchors_tuple in itertools.combinations(anchors, 4):
        direct_anchors = set(direct_anchors_tuple)
        if not forced_direct <= direct_anchors or forced_natural & direct_anchors:
            continue
        assigned = {anchor: ("direct" if anchor in direct_anchors else "natural") for anchor in anchors}
        if any(forced_by_anchor[anchor]["prompt_variant"] != assigned[anchor] for anchor in forced_by_anchor):
            continue
        selected: dict[str, Mapping[str, Any]] = dict(forced_by_anchor)
        used_receivers = {str(row["receiver_id"]) for row in forced_rows}
        remaining = [anchor for anchor in anchors if anchor not in selected]
        remaining.sort(key=lambda anchor: len(by_anchor_variant[(anchor, assigned[anchor])]))

        def match(index: int) -> bool:
            if index == len(remaining):
                return True
            anchor = remaining[index]
            for row in by_anchor_variant[(anchor, assigned[anchor])]:
                receiver_id = str(row["receiver_id"])
                if receiver_matching and receiver_id in used_receivers:
                    continue
                selected[anchor] = row
                if receiver_matching:
                    used_receivers.add(receiver_id)
                if match(index + 1):
                    return True
                if receiver_matching:
                    used_receivers.remove(receiver_id)
                del selected[anchor]
            return False

        if match(0):
            return tuple(str(selected[anchor]["case_id"]) for anchor in anchors)
    return None


def group_completion(
    group: str,
    rows: Sequence[Mapping[str, Any]],
    forced: frozenset[str],
    excluded: frozenset[str],
) -> tuple[str, ...] | None:
    group_forced = frozenset(
        case_id
        for case_id in forced
        if any(row["case_id"] == case_id and row["group"] == group for row in rows)
    )
    if len(group_forced) != sum(
        1 for case_id in forced if any(row["case_id"] == case_id and row["group"] == group for row in rows)
    ):
        return None
    options = tuple(
        row
        for row in rows
        if row["group"] == group and row["eligible"] and row["case_id"] not in excluded
    )
    if group == protocol.GROUPS[0]:
        return g1_completion(options, group_forced)
    if group == protocol.GROUPS[1]:
        return _fixed_anchor_completion(options, group_forced, group=group, receiver_matching=False)
    if group == protocol.GROUPS[2]:
        return _fixed_anchor_completion(options, group_forced, group=group, receiver_matching=True)
    raise ValueError("unknown group")


def exact_completion(
    rows: Sequence[Mapping[str, Any]],
    forced: frozenset[str],
    excluded: frozenset[str],
) -> tuple[str, ...] | None:
    protocol.require(
        all(type(row.get("eligible")) is bool for row in rows),
        "selector eligibility must be strict booleans",
    )
    all_ids = {str(row["case_id"]) for row in rows}
    if forced & excluded or not forced <= all_ids:
        return None
    pieces: list[str] = []
    for group in protocol.GROUPS:
        completion = group_completion(group, rows, forced, excluded)
        if completion is None:
            return None
        pieces.extend(completion)
    by_id = {str(row["case_id"]): row for row in rows}
    selected = [by_id[item] for item in pieces]
    if len(selected) != protocol.SELECTED_COUNT or len({row["receiver_id"] for row in selected}) != protocol.SELECTED_COUNT:
        return None
    if len({row["source_head_lemma"] for row in selected if row["group"] in protocol.GROUPS[:2]}) != 16:
        return None
    return tuple(pieces)


def greedy_select(
    candidates: Sequence[Mapping[str, Any]],
    eligibility: Sequence[Mapping[str, Any]],
    selector_salt: str,
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    protocol.require(len(candidates) == protocol.CANDIDATE_COUNT, "selector candidate count invalid")
    protocol.require(
        all(type(row.get("eligible")) is bool for row in eligibility),
        "selector eligibility must be strict booleans",
    )
    eligible_by_id = {
        str(row["candidate_id"]): row["eligible"] for row in eligibility
    }
    protocol.require(len(eligible_by_id) == protocol.CANDIDATE_COUNT, "selector eligibility coverage invalid")
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        row = dict(candidate)
        case_id = str(row["case_id"])
        protocol.require(case_id in eligible_by_id, "selector candidate missing eligibility")
        row["eligible"] = eligible_by_id[case_id]
        row["selection_rank_sha256"] = protocol.selection_rank(candidate, selector_salt)
        rows.append(row)
    ranks = [row["selection_rank_sha256"] for row in rows]
    if len(set(ranks)) != len(ranks):
        raise PreflightDatasetInvalid("selection_rank_tie", "rank tie invalidates v3")
    rows.sort(key=lambda row: row["selection_rank_sha256"])
    if exact_completion(rows, frozenset(), frozenset()) is None:
        raise PreflightDatasetInvalid("global_subset_infeasible", "no feasible v3 selector completion")
    forced: frozenset[str] = frozenset()
    excluded: frozenset[str] = frozenset(row["case_id"] for row in rows if not row["eligible"])
    decisions: list[dict[str, Any]] = []
    for row in rows:
        case_id = str(row["case_id"])
        if case_id in excluded:
            decisions.append({"case_id": case_id, "decision": "excluded_ineligible", "selection_rank_sha256": row["selection_rank_sha256"]})
            continue
        tentative = forced | {case_id}
        completion = exact_completion(rows, tentative, excluded)
        if completion is not None:
            forced = tentative
            decisions.append({"case_id": case_id, "decision": "included", "selection_rank_sha256": row["selection_rank_sha256"]})
        else:
            excluded = excluded | {case_id}
            decisions.append({"case_id": case_id, "decision": "excluded_no_completion", "selection_rank_sha256": row["selection_rank_sha256"]})
        if len(forced) == protocol.SELECTED_COUNT:
            break
    completion = exact_completion(rows, forced, excluded)
    if len(forced) != protocol.SELECTED_COUNT or completion is None or set(completion) != set(forced):
        raise PreflightDatasetInvalid("global_subset_infeasible", "greedy selector did not finish exact subset")
    by_id = {str(row["case_id"]): row for row in rows}
    selected = tuple(dict(by_id[item]) for item in sorted(forced, key=lambda item: by_id[item]["selection_rank_sha256"]))
    validate_selected_rows(selected)
    return selected, tuple(decisions)


def validate_selected_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    protocol.require(len(rows) == protocol.SELECTED_COUNT, "selected manifest must contain 24 rows")
    protocol.require(len({row["case_id"] for row in rows}) == protocol.SELECTED_COUNT, "selected case IDs repeat")
    protocol.require(len({row["receiver_id"] for row in rows}) == protocol.SELECTED_COUNT, "selected receivers repeat")
    groups = Counter(str(row["group"]) for row in rows)
    protocol.require(groups == Counter({group: 8 for group in protocol.GROUPS}), "selected group quota mismatch")
    for group in protocol.GROUPS:
        variants = Counter(str(row["prompt_variant"]) for row in rows if row["group"] == group)
        protocol.require(variants == Counter({"direct": 4, "natural": 4}), "selected variant quota mismatch")
    g1 = [row for row in rows if row["group"] == protocol.GROUPS[0]]
    protocol.require(len({row["physical_anchor_id"] for row in g1}) == 8, "G1 selected heads repeat")
    for group in protocol.GROUPS[1:]:
        subset = [row for row in rows if row["group"] == group]
        protocol.require(len({row["physical_anchor_id"] for row in subset}) == 8, f"{group} anchor coverage invalid")
    holdout_heads = {row["source_head_lemma"] for row in rows if row["group"] in protocol.GROUPS[:2]}
    protocol.require(len(holdout_heads) == 16, "selected holdout heads not unique")


def build_private_outputs(
    selected: Sequence[Mapping[str, Any]],
    *,
    evaluation_salt: str,
    screening_seed: int,
    forbidden_seeds: set[int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    validate_selected_rows(selected)
    protocol.require(isinstance(screening_seed, int) and not isinstance(screening_seed, bool) and 0 <= screening_seed < 2**32, "screening seed must be uint32")
    selected_rows = [
        {key: value for key, value in row.items() if key not in {"eligible"}}
        for row in selected
    ]
    selected_payload = {
        "protocol": SELECTED_PROTOCOL,
        "dataset_version": protocol.DATASET_VERSION,
        "status": "selected",
        "selected_count": protocol.SELECTED_COUNT,
        "selected": selected_rows,
    }
    units: list[dict[str, Any]] = []
    seeds: set[int] = set()
    for row in selected:
        for replicate in protocol.REPLICATES:
            seed = protocol.derive_evaluation_seed(evaluation_salt, str(row["case_id"]), replicate)
            protocol.require(seed not in seeds and seed not in forbidden_seeds and seed != screening_seed, "evaluation seed collision")
            seeds.add(seed)
            units.append(
                {
                    "unit_id": f"{row['case_id']}:r{replicate}",
                    "semantic_case_id": row["case_id"],
                    "replicate": replicate,
                    "seed": seed,
                    "group": row["group"],
                    "prompt_variant": row["prompt_variant"],
                    "canonical_prompt": row["canonical_prompt"],
                }
            )
    protocol.require(len(units) == protocol.UNIT_COUNT and len(seeds) == protocol.UNIT_COUNT, "U72 seed inventory invalid")
    unit_payload = {
        "protocol": UNIT_PROTOCOL,
        "dataset_version": protocol.DATASET_VERSION,
        "status": "frozen",
        "unit_count": protocol.UNIT_COUNT,
        "units": units,
    }
    return selected_payload, unit_payload


def build_selector_summary(
    *,
    eligibility: Sequence[Mapping[str, Any]],
    selected_payload: Mapping[str, Any],
    unit_payload: Mapping[str, Any],
    stage0_registry_sha256: str,
    screening_freeze_sha256: str,
    eligibility_table_sha256: str,
) -> dict[str, Any]:
    for value in (stage0_registry_sha256, screening_freeze_sha256, eligibility_table_sha256):
        protocol.require(protocol.is_hex64(value), "selector summary input hash invalid")
    protocol.require(
        all(type(row.get("eligible")) is bool for row in eligibility),
        "selector summary eligibility must be strict booleans",
    )
    cells = Counter(f"{row['group']}:{row['prompt_variant']}" for row in eligibility if row["eligible"])
    rank_tuple = [row["selection_rank_sha256"] for row in selected_payload["selected"]]
    payload = {
        "protocol": protocol.SELECTOR_SUMMARY_PROTOCOL,
        "dataset_version": protocol.DATASET_VERSION,
        "status": "selected",
        "candidate_count": protocol.CANDIDATE_COUNT,
        "eligible_count": sum(bool(row["eligible"]) for row in eligibility),
        "cell_eligible_counts": {f"{group}:{variant}": cells[f"{group}:{variant}"] for group, variant in protocol.CELL_ORDER},
        "selected_count": protocol.SELECTED_COUNT,
        "unit_count": protocol.UNIT_COUNT,
        "stage0_registry_sha256": stage0_registry_sha256,
        "screening_freeze_sha256": screening_freeze_sha256,
        "eligibility_table_sha256": eligibility_table_sha256,
        "selected_case_manifest_sha256": protocol.sha256_bytes(protocol.canonical_json_bytes(selected_payload)),
        "unit_manifest_sha256": protocol.sha256_bytes(protocol.canonical_json_bytes(unit_payload)),
        "selection_rank_tuple_sha256": protocol.sha256_bytes(protocol.canonical_json_bytes(rank_tuple)),
        "constraints": {
            "cell_quota_pass": True,
            "g1_distinct_head_pass": True,
            "g2_anchor_coverage_pass": True,
            "g3_anchor_coverage_pass": True,
            "original_source_coverage_pass": True,
            "holdout_head_uniqueness_pass": True,
            "receiver_uniqueness_pass": True,
            "rank_tie_free": True,
            "seed_contract_pass": True,
        },
    }
    protocol.validate_selector_summary(payload)
    return payload


def validate_forbidden_seed_inventory(payload: Mapping[str, Any]) -> set[int]:
    protocol.require_exact_keys(payload, {"protocol", "dataset", "status", "seed_encoding", "source_commitments", "seeds"}, "forbidden seed inventory")
    protocol.require(payload["protocol"] == FORBIDDEN_PROTOCOL and payload["dataset"] == protocol.DATASET and payload["status"] == "frozen_by_independent_seed_auditor", "forbidden seed inventory protocol/status mismatch")
    protocol.require(payload["seed_encoding"] == "nonnegative JSON integer below 2^63", "forbidden seed encoding mismatch")
    sources = payload["source_commitments"]
    protocol.require(isinstance(sources, list) and sources, "forbidden seed sources missing")
    source_names: list[str] = []
    for source in sources:
        protocol.require_exact_keys(source, {"name", "sha256", "seed_count"}, "forbidden seed source")
        protocol.require(isinstance(source["name"], str) and source["name"].strip() and protocol.is_hex64(source["sha256"]), "forbidden seed source invalid")
        protocol.require(isinstance(source["seed_count"], int) and not isinstance(source["seed_count"], bool) and source["seed_count"] >= 0, "forbidden seed source count invalid")
        source_names.append(source["name"])
    protocol.require(
        source_names == sorted(source_names) and len(set(source_names)) == len(source_names),
        "forbidden seed source names must be unique and sorted",
    )
    values = payload["seeds"]
    protocol.require(isinstance(values, list) and values and all(isinstance(value, int) and not isinstance(value, bool) and 0 <= value < 2**63 for value in values), "forbidden seeds invalid")
    protocol.require(values == sorted(values) and len(set(values)) == len(values), "forbidden seeds must be unique and sorted")
    protocol.require(
        sum(source["seed_count"] for source in sources) >= len(values),
        "forbidden seed inventory exceeds audited source counts",
    )
    return set(values)


def _read_secret(
    path: Path, *, hex_value: bool, private_root: Path | None = None
) -> str | int:
    if private_root is not None:
        protocol.validate_private_path(private_root, path)
    elif not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"missing non-symlink secret: {path}")
    raw = path.read_bytes()
    try:
        value = raw.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ValueError("secret is not ASCII") from exc
    protocol.require(value.endswith("\n") and value.count("\n") == 1, "secret must have one trailing LF")
    stripped = value[:-1]
    if hex_value:
        return protocol.validate_lower_hex_salt(stripped, "secret")
    protocol.require(stripped == str(int(stripped)) and 0 <= int(stripped) < 2**32, "screening seed invalid")
    return int(stripped)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--candidate-graph", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--eligibility", type=Path, required=True)
    parser.add_argument("--selector-salt", type=Path, required=True)
    parser.add_argument("--evaluation-salt", type=Path, required=True)
    parser.add_argument("--screening-seed", type=Path, required=True)
    parser.add_argument("--forbidden-seeds", type=Path, required=True)
    parser.add_argument("--stage0-registry-sha256", required=True)
    parser.add_argument("--screening-freeze-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    build_parser().parse_args(argv)
    raise RuntimeError(
        "formal selector execution not implemented; Stage-0 wrapper and "
        "screening-freeze provenance must be completed first"
    )


if __name__ == "__main__":
    raise SystemExit(main())
