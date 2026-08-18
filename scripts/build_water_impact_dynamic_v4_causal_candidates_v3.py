#!/usr/bin/env python3
"""Build and validate the frozen 576-edge v4_dev72_v3 causal graph.

The builder consumes only v3 private ontology files plus the two immutable v2
public training upstreams.  It does not generate media and does not authorize
Stage 0.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import re
import stat
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

try:
    import water_impact_dynamic_v4_eval_protocol_v3 as protocol
except ModuleNotFoundError:  # imported as scripts.build_...
    from scripts import water_impact_dynamic_v4_eval_protocol_v3 as protocol


HOLDOUT_PROTOCOL = "water_impact_dynamic_v4_eval_holdout_source_ontology_v3"
RECEIVER_PROTOCOL = "water_impact_dynamic_v4_eval_receiver_ontology_v3"
HISTORICAL_PROTOCOL = "water_impact_dynamic_v4_historical_receiver_anchors_v3"
GRAPH_ASSIGNMENT_DOMAIN = protocol.GRAPH_ASSIGNMENT_DOMAIN

HOLDOUT_TOP_KEYS = {
    "protocol",
    "dataset_version",
    "source_count",
    "sources",
    "curation_audit",
    "disjointness_commitment",
}
HOLDOUT_ROW_KEYS = {
    "source_id",
    "source_phrase",
    "normalized_phrase",
    "head_lemma",
    "origin",
    "food_status",
    "shape_class",
    "color_family",
    "material_family",
    "texture_class",
    "impact_plausibility",
    "physical_audit_status",
    "curator",
    "curation_stratum",
    "group_pool",
    "head_ordinal",
}
IMPACT_KEYS = {
    "verdict",
    "compact_and_rigid",
    "natural_drop_entry",
    "visible_brief_splash_or_ripple_plausible",
    "predominantly_buoyant_or_windborne",
    "flexible_or_film_like",
    "fragile",
    "powder",
    "loose_aggregate",
    "porous",
    "food_or_produce",
    "negative_buoyancy",
    "visually_recognizable",
    "entity_state",
    "material",
    "density_g_cm3",
    "mass_g",
    "dimensions_cm",
    "size_class",
    "source_specific_feature",
    "curator_note",
}
RECEIVER_TOP_KEYS = {
    "protocol",
    "dataset_version",
    "receiver_count",
    "pools",
    "receivers",
    "curation_audit",
    "disjointness_commitment",
}
RECEIVER_ROW_KEYS = {
    "receiver_id",
    "receiver_phrase",
    "normalized_phrase",
    "head_lemma",
    "receiver_type",
    "pool",
    "receiver_ordinal",
    "curator_note",
    "curator",
}
HISTORICAL_TOP_KEYS = {
    "protocol",
    "dataset_version",
    "anchor_count",
    "training_receiver_inventory_sha256",
    "v2_disjointness_commitment",
    "anchors",
}
HISTORICAL_ROW_KEYS = {
    "anchor_id",
    "receiver_id",
    "receiver_phrase",
    "normalized_phrase",
    "head_lemma",
    "historical_training_binding_sha256",
}
GRAPH_TOP_KEYS = {
    "protocol",
    "dataset_version",
    "status",
    "candidate_count",
    "cell_counts",
    "topology",
    "graph_assignment_salt_sha256",
    "r1",
    "r3",
    "anchors",
    "edges",
    "graph_sha256",
}
CANDIDATE_TOP_KEYS = {
    "protocol",
    "dataset_version",
    "status",
    "candidate_count",
    "cell_counts",
    "graph_sha256",
    "candidates",
}

PHRASE_PATTERN = re.compile(r"[a-z0-9]+(?: [a-z0-9]+)*\Z")
SOURCE_EVENT_ROOTS = (
    "water", "drop", "splash", "ripple", "impact", "fall", "wave", "sink",
    "plunge", "pour", "spray", "collision", "contact", "enter", "entry",
    "cavity",
)
AMBIGUOUS_SIZE_WORDS = {"small", "miniature", "tiny", "little"}
PROHIBITED_SOURCE_WORDS = {
    "food", "fruit", "vegetable", "produce", "apple", "lime", "berry",
    "walnut", "strawberry", "bread", "candy", "powder", "foam", "sponge",
    "paper", "cloth", "fabric", "leaf", "feather", "cork", "shell", "flake",
    "pellet", "granule",
}
RECEIVER_BOUNDARY_WORDS = {
    "edge", "edges", "edged", "boundary", "bounded", "rim", "rimmed",
    "center", "middle", "point", "area", "margins",
}
RECEIVER_STILL_WATER_WORDS = {"still", "calm", "quiet"}
EVENT_TOKENS = {
    "drop", "fall", "falling", "splash", "ripple", "impact", "collision",
    "contact", "enter", "entry", "wave", "spray",
}


def normalize_phrase(value: str) -> str:
    text = value.casefold()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def factual_prompt(source: str, receiver: str, variant: str) -> str:
    protocol.require(variant in protocol.PROMPT_VARIANTS, "prompt variant invalid")
    if variant == "direct":
        event = (
            f"{source.capitalize()} falls naturally from above, enters the center of the water in "
            f"{receiver}, and makes contact. The contact produces a visible brief splash "
            "followed by circular ripples spreading across the water."
        )
    else:
        event = (
            f"In a single natural motion, {source} drops into the center of the water in "
            f"{receiver}. After it touches the water, a short splash rises and expanding "
            "ripples travel outward."
        )
    return (
        "A simple realistic close-up video in one continuous shot. "
        f"{event} A soft reflected highlight moves slowly from left to right across "
        "the water and receiver throughout the shot. The viewpoint, receiver, and "
        "background geometry remain stable."
    )


def _validate_nonempty_object(value: Any, label: str) -> Mapping[str, Any]:
    protocol.require(isinstance(value, dict) and value, f"{label} must be a nonempty object")
    protocol.require(not protocol.contains_placeholder(value), f"{label} contains placeholder")
    return value


def _normalized_head(value: Any, label: str) -> str:
    protocol.require(
        isinstance(value, str) and bool(value.strip()),
        f"{label} head invalid",
    )
    normalized = normalize_phrase(value)
    protocol.require(
        bool(normalized) and PHRASE_PATTERN.fullmatch(normalized) is not None,
        f"{label} normalized head invalid",
    )
    return normalized


def _head_span_count(normalized_text: str, normalized_head: str) -> int:
    text_tokens = normalized_text.split()
    head_tokens = normalized_head.split()
    width = len(head_tokens)
    return sum(
        text_tokens[index : index + width] == head_tokens
        for index in range(len(text_tokens) - width + 1)
    )


def _validate_identity_fields(
    row: Mapping[str, Any], *, prefix: str
) -> str:
    identifier = row[f"{prefix}_id"]
    phrase = row[f"{prefix}_phrase"]
    normalized = row["normalized_phrase"]
    head = row["head_lemma"]
    protocol.require(isinstance(identifier, str) and identifier and identifier.strip() == identifier, f"{prefix} id invalid")
    protocol.require(isinstance(phrase, str) and phrase and phrase.strip() == phrase, f"{prefix} phrase invalid")
    protocol.require(normalized == normalize_phrase(phrase) and PHRASE_PATTERN.fullmatch(normalized) is not None, f"{prefix} normalization mismatch")
    normalized_head = _normalized_head(head, prefix)
    protocol.require(
        _head_span_count(normalized, normalized_head) == 1,
        f"{prefix} normalized head must occur exactly once as a contiguous whole-token span",
    )
    return normalized_head


def _validate_impact(value: Any) -> None:
    audit = protocol.require_exact_keys(value, IMPACT_KEYS, "impact plausibility")
    protocol.require(audit["verdict"] == "pass", "impact plausibility verdict failed")
    for key in (
        "compact_and_rigid",
        "natural_drop_entry",
        "visible_brief_splash_or_ripple_plausible",
        "negative_buoyancy",
        "visually_recognizable",
    ):
        protocol.require(audit[key] is True, f"impact required boolean failed: {key}")
    for key in (
        "predominantly_buoyant_or_windborne",
        "flexible_or_film_like",
        "fragile",
        "powder",
        "loose_aggregate",
        "porous",
        "food_or_produce",
    ):
        protocol.require(audit[key] is False, f"impact forbidden boolean true: {key}")
    protocol.require(audit["entity_state"] in {"solid_one_piece", "rigid_locked_assembly"}, "impact entity state invalid")
    protocol.require(isinstance(audit["density_g_cm3"], (int, float)) and not isinstance(audit["density_g_cm3"], bool) and 3.0 <= audit["density_g_cm3"] <= 20.0, "impact density invalid")
    protocol.require(isinstance(audit["mass_g"], int) and not isinstance(audit["mass_g"], bool) and 350 <= audit["mass_g"] <= 1200, "impact mass invalid")
    dims = audit["dimensions_cm"]
    protocol.require(isinstance(dims, list) and len(dims) == 3 and all(isinstance(item, (int, float)) and not isinstance(item, bool) and 2.5 <= item <= 15.0 for item in dims), "impact dimensions invalid")
    protocol.require(max(float(item) for item in dims) >= 8.0, "impact dimensions lack an explicit palm-sized extent")
    protocol.require(float(audit["mass_g"]) <= float(audit["density_g_cm3"]) * math.prod(float(item) for item in dims), "impact mass exceeds bounding-volume capacity")
    for key in ("material", "source_specific_feature", "curator_note"):
        protocol.require(isinstance(audit[key], str) and audit[key].strip(), f"impact {key} invalid")
    protocol.require(audit["size_class"] == "palm_sized_explicit", "impact size class invalid")


def validate_holdout_ontology(payload: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    protocol.require_exact_keys(payload, HOLDOUT_TOP_KEYS, "holdout ontology")
    protocol.require(payload["protocol"] == HOLDOUT_PROTOCOL and payload["dataset_version"] == protocol.DATASET_VERSION and payload["source_count"] == 48, "holdout ontology protocol/count mismatch")
    _validate_nonempty_object(payload["curation_audit"], "holdout curation audit")
    protocol.require(protocol.is_hex64(payload["disjointness_commitment"]), "holdout disjointness commitment invalid")
    rows = payload["sources"]
    protocol.require(isinstance(rows, list) and len(rows) == 48, "holdout source inventory must contain 48 rows")
    ids: set[str] = set()
    heads: set[str] = set()
    pool_ordinals: dict[str, set[int]] = {"G1": set(), "G2": set()}
    output: list[Mapping[str, Any]] = []
    for row in rows:
        protocol.require_exact_keys(row, HOLDOUT_ROW_KEYS, "holdout source row")
        normalized_head = _validate_identity_fields(row, prefix="source")
        protocol.require(row["source_id"] not in ids and normalized_head not in heads, "holdout IDs/heads are not unique")
        ids.add(row["source_id"])
        heads.add(normalized_head)
        protocol.require(row["physical_audit_status"] == "strict_physical_pass_v3", "holdout physical status invalid")
        _validate_impact(row["impact_plausibility"])
        tokens = set(row["normalized_phrase"].split())
        identity = f"{row['source_id']} {row['normalized_phrase']} {row['head_lemma']}"
        protocol.require(
            not any(root in identity for root in SOURCE_EVENT_ROOTS),
            "holdout source contains event/mechanism language",
        )
        protocol.require(
            not tokens & (AMBIGUOUS_SIZE_WORDS | PROHIBITED_SOURCE_WORDS),
            "holdout source contains ambiguous/prohibited category language",
        )
        protocol.require(row["food_status"] == "non_food", "holdout source food status invalid")
        protocol.require(
            _head_span_count(
                normalize_phrase(row["impact_plausibility"]["curator_note"]),
                normalized_head,
            )
            == 1,
            "holdout physical note is not identity-specific",
        )
        protocol.require(row["group_pool"] in pool_ordinals, "holdout group pool invalid")
        ordinal = row["head_ordinal"]
        protocol.require(isinstance(ordinal, int) and not isinstance(ordinal, bool) and 0 <= ordinal < 24 and ordinal not in pool_ordinals[row["group_pool"]], "holdout head ordinal invalid")
        pool_ordinals[row["group_pool"]].add(ordinal)
        for key in ("origin", "food_status", "shape_class", "color_family", "material_family", "texture_class", "curator", "curation_stratum"):
            protocol.require(isinstance(row[key], str) and row[key].strip(), f"holdout {key} invalid")
        output.append(row)
    protocol.require(all(values == set(range(24)) for values in pool_ordinals.values()), "holdout pool ordinals incomplete")
    protocol.require(not protocol.contains_placeholder(payload), "holdout ontology contains placeholder")
    return tuple(output)


def validate_receiver_ontology(payload: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    protocol.require_exact_keys(payload, RECEIVER_TOP_KEYS, "receiver ontology")
    protocol.require(payload["protocol"] == RECEIVER_PROTOCOL and payload["dataset_version"] == protocol.DATASET_VERSION and payload["receiver_count"] == 56, "receiver ontology protocol/count mismatch")
    protocol.require(payload["pools"] == {"R1": 24, "R3": 32}, "receiver pool sizes mismatch")
    _validate_nonempty_object(payload["curation_audit"], "receiver curation audit")
    protocol.require(protocol.is_hex64(payload["disjointness_commitment"]), "receiver disjointness commitment invalid")
    rows = payload["receivers"]
    protocol.require(isinstance(rows, list) and len(rows) == 56, "receiver ontology must contain 56 rows")
    ids: set[str] = set()
    phrases: set[str] = set()
    heads: set[str] = set()
    receiver_types: set[str] = set()
    ordinals: dict[str, set[int]] = {"R1": set(), "R3": set()}
    output: list[Mapping[str, Any]] = []
    for row in rows:
        protocol.require_exact_keys(row, RECEIVER_ROW_KEYS, "receiver row")
        normalized_head = _validate_identity_fields(row, prefix="receiver")
        protocol.require(
            row["receiver_id"] not in ids
            and row["normalized_phrase"] not in phrases
            and normalized_head not in heads
            and row["receiver_type"] not in receiver_types,
            "receiver identity/head/type is not unique",
        )
        ids.add(row["receiver_id"])
        phrases.add(row["normalized_phrase"])
        heads.add(normalized_head)
        receiver_types.add(row["receiver_type"])
        pool = row["pool"]
        protocol.require(pool in ordinals, "receiver pool invalid")
        ordinal = row["receiver_ordinal"]
        limit = 24 if pool == "R1" else 32
        protocol.require(isinstance(ordinal, int) and not isinstance(ordinal, bool) and 0 <= ordinal < limit and ordinal not in ordinals[pool], "receiver ordinal invalid")
        ordinals[pool].add(ordinal)
        for key in ("receiver_type", "curator_note", "curator"):
            protocol.require(isinstance(row[key], str) and row[key].strip(), f"receiver {key} invalid")
        tokens = set(row["normalized_phrase"].split())
        protocol.require(
            {"water", "open", "unobstructed", "landing"} <= tokens
            and bool(tokens & RECEIVER_STILL_WATER_WORDS),
            "receiver lacks required water/open/unobstructed/landing/stillness language",
        )
        protocol.require(
            bool(tokens & RECEIVER_BOUNDARY_WORDS),
            "receiver lacks a clearly bounded water surface",
        )
        protocol.require(
            not tokens & EVENT_TOKENS,
            "receiver contains event language",
        )
        protocol.require(
            _head_span_count(normalize_phrase(row["curator_note"]), normalized_head)
            == 1,
            "receiver note is not identity-specific",
        )
        output.append(row)
    protocol.require(ordinals["R1"] == set(range(24)) and ordinals["R3"] == set(range(32)), "receiver pool ordinals incomplete")
    protocol.require(not protocol.contains_placeholder(payload), "receiver ontology contains placeholder")
    return tuple(output)


def validate_historical_anchors(payload: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    protocol.require_exact_keys(payload, HISTORICAL_TOP_KEYS, "historical anchors")
    protocol.require(payload["protocol"] == HISTORICAL_PROTOCOL and payload["dataset_version"] == protocol.DATASET_VERSION and payload["anchor_count"] == 8, "historical anchor protocol/count mismatch")
    protocol.require(protocol.is_hex64(payload["training_receiver_inventory_sha256"]) and protocol.is_hex64(payload["v2_disjointness_commitment"]), "historical anchor commitment invalid")
    rows = payload["anchors"]
    protocol.require(isinstance(rows, list) and len(rows) == 8, "historical anchor inventory must contain eight rows")
    ids: set[str] = set()
    anchor_ids: set[str] = set()
    phrases: set[str] = set()
    heads: set[str] = set()
    output: list[Mapping[str, Any]] = []
    for row in rows:
        protocol.require_exact_keys(row, HISTORICAL_ROW_KEYS, "historical anchor row")
        normalized_head = _validate_identity_fields(row, prefix="receiver")
        protocol.require(
            isinstance(row["anchor_id"], str)
            and bool(row["anchor_id"])
            and row["anchor_id"].strip() == row["anchor_id"],
            "historical anchor id invalid",
        )
        protocol.require(
            row["anchor_id"] not in anchor_ids
            and row["receiver_id"] not in ids
            and row["normalized_phrase"] not in phrases
            and normalized_head not in heads,
            "historical anchor/receiver identity repeated",
        )
        anchor_ids.add(row["anchor_id"])
        ids.add(row["receiver_id"])
        phrases.add(row["normalized_phrase"])
        heads.add(normalized_head)
        protocol.require(protocol.is_hex64(row["historical_training_binding_sha256"]), "historical training binding invalid")
        output.append(row)
    protocol.require(not protocol.contains_placeholder(payload), "historical anchors contain placeholder")
    return tuple(output)


def validate_templates_and_fields(
    templates_path: Path,
    field_rules_path: Path,
    *,
    private_root: Path | None = None,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    if private_root is not None:
        protocol.validate_private_path(private_root, templates_path)
        protocol.validate_private_path(private_root, field_rules_path)
    protocol.require(protocol.sha256_file(templates_path) == protocol.V2_TEMPLATE_SHA256, "v3 templates are not byte-equal to frozen construct")
    protocol.require(protocol.sha256_file(field_rules_path) == protocol.V2_FIELD_RULES_SHA256, "v3 field rules are not byte-equal to frozen construct")
    templates = protocol.load_json(templates_path, private_root=private_root)
    fields = protocol.load_json(field_rules_path, private_root=private_root)
    expected_templates = {
        variant: factual_prompt("{source_phrase}", "{receiver_phrase}", variant)
        for variant in protocol.PROMPT_VARIANTS
    }
    protocol.require(templates.get("prompt_templates") == expected_templates, "template bytes do not encode the canonical prompt builder")
    protocol.require(templates.get("template_fill_rules") == {
        "direct": {"source_phrase": "python_str_capitalize", "receiver_phrase": "identity"},
        "natural": {"source_phrase": "identity", "receiver_phrase": "identity"},
    }, "template fill rules invalid")
    return templates, fields


def _permuted_receivers(
    rows: Sequence[Mapping[str, Any]], pool: str, salt: str
) -> tuple[Mapping[str, Any], ...]:
    protocol.validate_lower_hex_salt(salt, "graph assignment salt")
    selected = [row for row in rows if row["pool"] == pool]
    ranked = [
        (
            hashlib.sha256(
                GRAPH_ASSIGNMENT_DOMAIN.encode("ascii")
                + b"\x00"
                + salt.encode("ascii")
                + b"\x00"
                + pool.encode("ascii")
                + b"\x00"
                + row["receiver_id"].encode("utf-8")
            ).hexdigest(),
            row,
        )
        for row in selected
    ]
    digests = [digest for digest, _ in ranked]
    protocol.require(
        len(set(digests)) == len(digests),
        f"{pool} receiver permutation rank tie invalidates v3",
    )
    return tuple(row for _, row in sorted(ranked, key=lambda item: item[0]))


def _permutation_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    return protocol.sha256_bytes(
        protocol.canonical_json_bytes([str(row["receiver_id"]) for row in rows])
    )


def _original_sources(bank: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    entries = bank.get("entries")
    protocol.require(isinstance(entries, list) and len(entries) == 64, "source bank entries invalid")
    originals = sorted(
        (row for row in entries if row.get("membership") == "original_training_source"),
        key=lambda row: row.get("bank_index", -1),
    )
    protocol.require(len(originals) == 8 and [row["bank_index"] for row in originals] == list(range(8)), "source bank original-eight binding invalid")
    required = {"bank_index", "head_lemma", "membership", "normalized_phrase", "physical_audit_status", "source_id", "source_phrase"}
    protocol.require(all(set(row) == required for row in originals), "original source row fields invalid")
    return tuple(originals)


def _edge(
    *,
    case_id: str,
    group: str,
    variant: str,
    anchor_id: str,
    edge_ordinal: int,
    source_membership: str,
    source: Mapping[str, Any],
    receiver_membership: str,
    receiver: Mapping[str, Any],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "case_id": case_id,
        "group": group,
        "prompt_variant": variant,
        "physical_anchor_id": anchor_id,
        "edge_ordinal": edge_ordinal,
        "source_membership": source_membership,
        "source_id": source["source_id"],
        "source_phrase": source["source_phrase"],
        "source_head_lemma": source["head_lemma"],
        "receiver_membership": receiver_membership,
        "receiver_id": receiver["receiver_id"],
        "receiver_phrase": receiver["receiver_phrase"],
        "canonical_prompt": factual_prompt(source["source_phrase"], receiver["receiver_phrase"], variant),
    }
    row["canonical_record_sha256"] = protocol.sha256_bytes(protocol.canonical_json_bytes(row))
    protocol.candidate_record_bytes(row)
    return row


def graph_topology() -> dict[str, Any]:
    return {
        "cell_order": [f"{group}:{variant}" for group, variant in protocol.CELL_ORDER],
        "G1": {
            "physical_anchors": 24,
            "direct_edges_per_anchor": 2,
            "natural_edges_per_anchor": 7,
            "selected_heads": 8,
            "max_selected_per_head": 1,
        },
        "G2": {
            "physical_anchors": 8,
            "heads_per_anchor": 3,
            "direct_edges_per_head": 1,
            "natural_edges_per_head": 1,
            "selected_per_anchor": 1,
        },
        "G3": {
            "physical_anchors": 8,
            "direct_edges_per_anchor": 12,
            "natural_edges_per_anchor": 27,
            "selected_per_anchor": 1,
        },
        "selected_per_group": 8,
        "selected_variants_per_group": {"direct": 4, "natural": 4},
    }


def build_candidate_graph(
    *,
    holdout_payload: Mapping[str, Any],
    receiver_payload: Mapping[str, Any],
    historical_payload: Mapping[str, Any],
    source_bank_payload: Mapping[str, Any],
    graph_assignment_salt: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    holdout = validate_holdout_ontology(holdout_payload)
    receivers = validate_receiver_ontology(receiver_payload)
    historical = validate_historical_anchors(historical_payload)
    originals = _original_sources(source_bank_payload)
    r1 = _permuted_receivers(receivers, "R1", graph_assignment_salt)
    r3 = _permuted_receivers(receivers, "R3", graph_assignment_salt)
    g1_sources = sorted((row for row in holdout if row["group_pool"] == "G1"), key=lambda row: row["head_ordinal"])
    g2_sources = sorted((row for row in holdout if row["group_pool"] == "G2"), key=lambda row: row["head_ordinal"])

    edges: list[dict[str, Any]] = []
    anchor_records: list[dict[str, Any]] = []
    for head_index, source in enumerate(g1_sources):
        anchor_id = f"g1h{head_index:02d}"
        anchor_records.append({"group": protocol.GROUPS[0], "physical_anchor_id": anchor_id, "source_id": source["source_id"]})
        edge_ordinal = 0
        for variant, offsets in (("direct", protocol.R1_DIRECT_OFFSETS), ("natural", protocol.R1_NATURAL_OFFSETS)):
            for offset in offsets:
                edges.append(
                    _edge(
                        case_id=f"v4v3_g1_h{head_index:02d}_{variant}_e{edge_ordinal:02d}",
                        group=protocol.GROUPS[0],
                        variant=variant,
                        anchor_id=anchor_id,
                        edge_ordinal=edge_ordinal,
                        source_membership="holdout_source",
                        source=source,
                        receiver_membership="new_receiver",
                        receiver=r1[(head_index + offset) % 24],
                    )
                )
                edge_ordinal += 1

    for anchor_index, historical_receiver in enumerate(historical):
        anchor_id = f"g2a{anchor_index}"
        anchor_records.append({"group": protocol.GROUPS[1], "physical_anchor_id": anchor_id, "receiver_id": historical_receiver["receiver_id"]})
        edge_ordinal = 0
        receiver = {
            "receiver_id": historical_receiver["receiver_id"],
            "receiver_phrase": historical_receiver["receiver_phrase"],
        }
        for head_offset in range(3):
            source = g2_sources[3 * anchor_index + head_offset]
            for variant in protocol.PROMPT_VARIANTS:
                edges.append(
                    _edge(
                        case_id=f"v4v3_g2_a{anchor_index}_{variant}_h{head_offset}",
                        group=protocol.GROUPS[1],
                        variant=variant,
                        anchor_id=anchor_id,
                        edge_ordinal=edge_ordinal,
                        source_membership="holdout_source",
                        source=source,
                        receiver_membership="seen_receiver",
                        receiver=receiver,
                    )
                )
                edge_ordinal += 1

    for anchor_index, source in enumerate(originals):
        anchor_id = f"g3a{anchor_index}"
        anchor_records.append({"group": protocol.GROUPS[2], "physical_anchor_id": anchor_id, "source_id": source["source_id"]})
        direct_indices = sorted(
            {
                4 * ((anchor_index + block_offset) % 8) + offset
                for block_offset in (1, 2, 3)
                for offset in range(4)
            }
        )
        natural_indices = sorted(
            set(range(32))
            - {4 * anchor_index + offset for offset in range(4)}
            - {4 * ((anchor_index + 4) % 8) + (anchor_index % 4)}
        )
        edge_ordinal = 0
        for variant, receiver_indices in (("direct", direct_indices), ("natural", natural_indices)):
            for receiver_index in receiver_indices:
                edges.append(
                    _edge(
                        case_id=f"v4v3_g3_a{anchor_index}_{variant}_e{edge_ordinal:02d}",
                        group=protocol.GROUPS[2],
                        variant=variant,
                        anchor_id=anchor_id,
                        edge_ordinal=edge_ordinal,
                        source_membership="original_source",
                        source=source,
                        receiver_membership="new_receiver",
                        receiver=r3[receiver_index],
                    )
                )
                edge_ordinal += 1

    cell_counts = Counter((row["group"], row["prompt_variant"]) for row in edges)
    protocol.require(len(edges) == protocol.CANDIDATE_COUNT and dict(cell_counts) == protocol.CELL_COUNTS, "candidate graph cell counts mismatch")
    protocol.require(len({row["case_id"] for row in edges}) == protocol.CANDIDATE_COUNT, "candidate case IDs are not unique")
    protocol.require(len({row["canonical_record_sha256"] for row in edges}) == protocol.CANDIDATE_COUNT, "candidate canonical records are not unique")

    graph: dict[str, Any] = {
        "protocol": protocol.GRAPH_PROTOCOL,
        "dataset_version": protocol.DATASET_VERSION,
        "status": "frozen_before_original_screening",
        "candidate_count": protocol.CANDIDATE_COUNT,
        "cell_counts": {f"{group}:{variant}": protocol.CELL_COUNTS[(group, variant)] for group, variant in protocol.CELL_ORDER},
        "topology": graph_topology(),
        "graph_assignment_salt_sha256": protocol.sha256_bytes(
            (graph_assignment_salt + "\n").encode("ascii")
        ),
        "r1": {
            "pool": "R1",
            "receiver_count": 24,
            "receiver_ids": [row["receiver_id"] for row in r1],
            "permutation_sha256": _permutation_sha256(r1),
        },
        "r3": {
            "pool": "R3",
            "receiver_count": 32,
            "receiver_ids": [row["receiver_id"] for row in r3],
            "permutation_sha256": _permutation_sha256(r3),
        },
        "anchors": anchor_records,
        "edges": edges,
    }
    graph["graph_sha256"] = protocol.sha256_bytes(protocol.canonical_json_bytes(graph))
    manifest = {
        "protocol": protocol.CANDIDATE_PROTOCOL,
        "dataset_version": protocol.DATASET_VERSION,
        "status": "frozen_before_original_screening",
        "candidate_count": protocol.CANDIDATE_COUNT,
        "cell_counts": graph["cell_counts"],
        "graph_sha256": graph["graph_sha256"],
        "candidates": [dict(row) for row in edges],
    }
    validate_graph_payload(graph)
    validate_candidate_projection(graph, manifest)
    return graph, manifest


def validate_graph_payload(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    protocol.require_exact_keys(payload, GRAPH_TOP_KEYS, "candidate graph")
    protocol.require(payload["protocol"] == protocol.GRAPH_PROTOCOL and payload["dataset_version"] == protocol.DATASET_VERSION and payload["status"] == "frozen_before_original_screening", "candidate graph protocol/status mismatch")
    protocol.require(payload["candidate_count"] == protocol.CANDIDATE_COUNT and payload["cell_counts"] == {f"{g}:{v}": protocol.CELL_COUNTS[(g, v)] for g, v in protocol.CELL_ORDER}, "candidate graph counts mismatch")
    protocol.require(payload["topology"] == graph_topology(), "candidate graph topology mismatch")
    protocol.require(protocol.is_hex64(payload["graph_assignment_salt_sha256"]), "graph assignment salt-file hash invalid")
    protocol.require(set(payload["r1"]) == {"pool", "receiver_count", "receiver_ids", "permutation_sha256"} and payload["r1"].get("pool") == "R1" and payload["r1"].get("receiver_count") == 24 and isinstance(payload["r1"].get("receiver_ids"), list) and len(payload["r1"]["receiver_ids"]) == 24 and len(set(payload["r1"]["receiver_ids"])) == 24 and payload["r1"]["permutation_sha256"] == protocol.sha256_bytes(protocol.canonical_json_bytes(payload["r1"]["receiver_ids"])), "R1 inventory/permutation invalid")
    protocol.require(set(payload["r3"]) == {"pool", "receiver_count", "receiver_ids", "permutation_sha256"} and payload["r3"].get("pool") == "R3" and payload["r3"].get("receiver_count") == 32 and isinstance(payload["r3"].get("receiver_ids"), list) and len(payload["r3"]["receiver_ids"]) == 32 and len(set(payload["r3"]["receiver_ids"])) == 32 and payload["r3"]["permutation_sha256"] == protocol.sha256_bytes(protocol.canonical_json_bytes(payload["r3"]["receiver_ids"])), "R3 inventory/permutation invalid")
    protocol.require(set(payload["r1"]["receiver_ids"]).isdisjoint(payload["r3"]["receiver_ids"]), "R1/R3 receiver pools overlap")
    protocol.require(isinstance(payload["anchors"], list) and len(payload["anchors"]) == 40, "anchor inventory invalid")
    edges = payload["edges"]
    protocol.require(isinstance(edges, list) and len(edges) == protocol.CANDIDATE_COUNT, "candidate edge inventory invalid")
    counts: Counter[tuple[str, str]] = Counter()
    for row in edges:
        protocol.require_exact_keys(row, protocol.GRAPH_EDGE_KEYS, "candidate edge")
        protocol.candidate_record_bytes(row)
        protocol.require(row["group"] in protocol.GROUPS and row["prompt_variant"] in protocol.PROMPT_VARIANTS, "candidate group/variant invalid")
        protocol.require(row["canonical_prompt"] == factual_prompt(row["source_phrase"], row["receiver_phrase"], row["prompt_variant"]), "candidate canonical prompt mismatch")
        counts[(row["group"], row["prompt_variant"])] += 1
    protocol.require(dict(counts) == protocol.CELL_COUNTS, "candidate edge cell counts invalid")
    r1_ids = tuple(payload["r1"]["receiver_ids"])
    r3_ids = tuple(payload["r3"]["receiver_ids"])
    by_case = {str(row["case_id"]): row for row in edges}
    protocol.require(len(by_case) == protocol.CANDIDATE_COUNT, "candidate case IDs repeat")

    g1_heads: set[str] = set()
    for head_index in range(24):
        anchor = f"g1h{head_index:02d}"
        source_ids: set[str] = set()
        ordinal = 0
        for variant, offsets in (("direct", protocol.R1_DIRECT_OFFSETS), ("natural", protocol.R1_NATURAL_OFFSETS)):
            for offset in offsets:
                case_id = f"v4v3_g1_h{head_index:02d}_{variant}_e{ordinal:02d}"
                protocol.require(case_id in by_case, "G1 canonical case ID missing")
                row = by_case[case_id]
                protocol.require(
                    row["group"] == protocol.GROUPS[0]
                    and row["prompt_variant"] == variant
                    and row["physical_anchor_id"] == anchor
                    and row["edge_ordinal"] == ordinal
                    and row["source_membership"] == "holdout_source"
                    and row["receiver_membership"] == "new_receiver"
                    and row["receiver_id"] == r1_ids[(head_index + offset) % 24],
                    "G1 graph incidence mismatch",
                )
                source_ids.add(str(row["source_id"]))
                g1_heads.add(normalize_phrase(str(row["source_head_lemma"])))
                ordinal += 1
        protocol.require(len(source_ids) == 1, "G1 anchor contains multiple sources")

    g2_heads: set[str] = set()
    g2_receivers: set[str] = set()
    for anchor_index in range(8):
        anchor = f"g2a{anchor_index}"
        anchor_receiver: set[str] = set()
        anchor_heads: set[str] = set()
        ordinal = 0
        for head_offset in range(3):
            head_source: set[str] = set()
            for variant in protocol.PROMPT_VARIANTS:
                case_id = f"v4v3_g2_a{anchor_index}_{variant}_h{head_offset}"
                protocol.require(case_id in by_case, "G2 canonical case ID missing")
                row = by_case[case_id]
                protocol.require(
                    row["group"] == protocol.GROUPS[1]
                    and row["prompt_variant"] == variant
                    and row["physical_anchor_id"] == anchor
                    and row["edge_ordinal"] == ordinal
                    and row["source_membership"] == "holdout_source"
                    and row["receiver_membership"] == "seen_receiver",
                    "G2 graph incidence mismatch",
                )
                anchor_receiver.add(str(row["receiver_id"]))
                anchor_heads.add(normalize_phrase(str(row["source_head_lemma"])))
                head_source.add(str(row["source_id"]))
                ordinal += 1
            protocol.require(len(head_source) == 1, "G2 head variants do not bind one source")
        protocol.require(len(anchor_receiver) == 1 and len(anchor_heads) == 3, "G2 anchor receiver/head structure invalid")
        g2_receivers.update(anchor_receiver)
        g2_heads.update(anchor_heads)

    g3_sources: set[str] = set()
    for anchor_index in range(8):
        anchor = f"g3a{anchor_index}"
        direct_indices = sorted(
            {
                4 * ((anchor_index + block_offset) % 8) + offset
                for block_offset in (1, 2, 3)
                for offset in range(4)
            }
        )
        natural_indices = sorted(
            set(range(32))
            - {4 * anchor_index + offset for offset in range(4)}
            - {4 * ((anchor_index + 4) % 8) + (anchor_index % 4)}
        )
        source_ids: set[str] = set()
        ordinal = 0
        for variant, receiver_indices in (("direct", direct_indices), ("natural", natural_indices)):
            for receiver_index in receiver_indices:
                case_id = f"v4v3_g3_a{anchor_index}_{variant}_e{ordinal:02d}"
                protocol.require(case_id in by_case, "G3 canonical case ID missing")
                row = by_case[case_id]
                protocol.require(
                    row["group"] == protocol.GROUPS[2]
                    and row["prompt_variant"] == variant
                    and row["physical_anchor_id"] == anchor
                    and row["edge_ordinal"] == ordinal
                    and row["source_membership"] == "original_source"
                    and row["receiver_membership"] == "new_receiver"
                    and row["receiver_id"] == r3_ids[receiver_index],
                    "G3 graph incidence mismatch",
                )
                source_ids.add(str(row["source_id"]))
                ordinal += 1
        protocol.require(len(source_ids) == 1, "G3 anchor contains multiple sources")
        g3_sources.update(source_ids)
    protocol.require(len(g1_heads) == 24 and len(g2_heads) == 24 and g1_heads.isdisjoint(g2_heads), "G1/G2 holdout head partitions invalid")
    protocol.require(len(g2_receivers) == 8 and g2_receivers.isdisjoint(r1_ids) and g2_receivers.isdisjoint(r3_ids), "G2/R1/R3 receiver pools are not disjoint")
    protocol.require(len(g3_sources) == 8, "G3 original-source coverage invalid")
    expected_anchors: list[dict[str, Any]] = []
    for head_index in range(24):
        first = by_case[f"v4v3_g1_h{head_index:02d}_direct_e00"]
        expected_anchors.append({
            "group": protocol.GROUPS[0],
            "physical_anchor_id": f"g1h{head_index:02d}",
            "source_id": first["source_id"],
        })
    for anchor_index in range(8):
        first = by_case[f"v4v3_g2_a{anchor_index}_direct_h0"]
        expected_anchors.append({
            "group": protocol.GROUPS[1],
            "physical_anchor_id": f"g2a{anchor_index}",
            "receiver_id": first["receiver_id"],
        })
    for anchor_index in range(8):
        first = by_case[f"v4v3_g3_a{anchor_index}_direct_e00"]
        expected_anchors.append({
            "group": protocol.GROUPS[2],
            "physical_anchor_id": f"g3a{anchor_index}",
            "source_id": first["source_id"],
        })
    protocol.require(payload["anchors"] == expected_anchors, "candidate anchor inventory mismatch")
    base = dict(payload)
    digest = base.pop("graph_sha256")
    protocol.require(protocol.is_hex64(digest) and protocol.sha256_bytes(protocol.canonical_json_bytes(base)) == digest, "candidate graph self-hash mismatch")
    protocol.require(not protocol.contains_placeholder(payload), "candidate graph contains placeholder")
    return payload


def validate_graph_against_inputs(
    graph: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    holdout_payload: Mapping[str, Any],
    receiver_payload: Mapping[str, Any],
    historical_payload: Mapping[str, Any],
    source_bank_payload: Mapping[str, Any],
    graph_assignment_salt: str,
) -> None:
    expected_graph, expected_manifest = build_candidate_graph(
        holdout_payload=holdout_payload,
        receiver_payload=receiver_payload,
        historical_payload=historical_payload,
        source_bank_payload=source_bank_payload,
        graph_assignment_salt=graph_assignment_salt,
    )
    protocol.require(graph == expected_graph, "candidate graph differs from deterministic ontology reconstruction")
    protocol.require(manifest == expected_manifest, "candidate manifest differs from deterministic graph projection")


def validate_candidate_projection(
    graph: Mapping[str, Any], manifest: Mapping[str, Any]
) -> Mapping[str, Any]:
    validate_graph_payload(graph)
    protocol.require_exact_keys(manifest, CANDIDATE_TOP_KEYS, "candidate manifest")
    protocol.require(manifest["protocol"] == protocol.CANDIDATE_PROTOCOL and manifest["dataset_version"] == protocol.DATASET_VERSION and manifest["status"] == "frozen_before_original_screening", "candidate manifest protocol/status mismatch")
    protocol.require(manifest["candidate_count"] == protocol.CANDIDATE_COUNT and manifest["cell_counts"] == graph["cell_counts"] and manifest["graph_sha256"] == graph["graph_sha256"], "candidate manifest graph binding mismatch")
    protocol.require(manifest["candidates"] == graph["edges"], "candidate manifest is not exact graph projection")
    protocol.require(not protocol.contains_placeholder(manifest), "candidate manifest contains placeholder")
    return manifest


def _load_salt(path: Path, private_root: Path) -> str:
    protocol.validate_private_path(private_root, path)
    raw = path.read_bytes()
    try:
        value = raw.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ValueError("graph salt is not ASCII") from exc
    protocol.require(value.endswith("\n") and value.count("\n") == 1, "graph salt must have one trailing LF")
    return protocol.validate_lower_hex_salt(value[:-1], "graph salt")


def _require_v2_hashes_unchanged(
    project_root: Path, before: Mapping[str, str]
) -> dict[str, str]:
    observed = protocol.validate_v2_public_inputs(project_root)
    protocol.require(
        observed == dict(before),
        "allowed v2 public inputs changed during candidate construction",
    )
    return observed


def _write_graph_manifest_transaction(
    graph_path: Path,
    graph: Mapping[str, Any],
    manifest_path: Path,
    manifest: Mapping[str, Any],
    *,
    post_link_check: Callable[[], None],
    ownership_sink: list[tuple[Path, tuple[int, int]]] | None = None,
) -> None:
    payload_targets = ((graph_path, graph), (manifest_path, manifest))
    protocol.require(
        graph_path.parent == manifest_path.parent,
        "graph/manifest outputs must share one private parent",
    )
    for path, _ in payload_targets:
        if os.path.lexists(path):
            raise FileExistsError(f"refusing to overwrite builder output: {path}")
    targets = tuple(
        (path, protocol.canonical_json_bytes(dict(payload)))
        for path, payload in payload_targets
    )
    temporaries: list[tuple[Path, tuple[int, int], Path, bytes]] = []
    owned_targets: list[tuple[Path, tuple[int, int]]] = []
    namespace_changed = False

    def unlink_if_owned(path: Path, inode: tuple[int, int]) -> bool:
        try:
            info = os.lstat(path)
        except FileNotFoundError:
            return False
        if (
            not stat.S_ISREG(info.st_mode)
            or (info.st_dev, info.st_ino) != inode
        ):
            return False
        path.unlink()
        return True

    def fsync_parent() -> None:
        parent_descriptor = os.open(
            graph_path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)

    def require_owned_readback(
        path: Path, inode: tuple[int, int], expected: bytes
    ) -> None:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            info = os.fstat(descriptor)
            protocol.require(
                stat.S_ISREG(info.st_mode)
                and (info.st_dev, info.st_ino) == inode
                and info.st_nlink == 1,
                f"builder output inode changed before readback: {path}",
            )
            with os.fdopen(descriptor, "rb") as handle:
                descriptor = -1
                observed = handle.read()
        finally:
            if descriptor >= 0:
                os.close(descriptor)
        protocol.require(
            observed == expected, f"builder output readback mismatch: {path}"
        )
        current = os.lstat(path)
        protocol.require(
            stat.S_ISREG(current.st_mode)
            and (current.st_dev, current.st_ino) == inode
            and current.st_nlink == 1,
            f"builder output inode changed after readback: {path}",
        )

    def create_tracked_temporary(target: Path, raw: bytes) -> None:
        """Create/write one temp after publishing its inode to the outer scope."""

        nonlocal namespace_changed
        descriptor = -1
        temporary: Path | None = None
        try:
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{target.name}.", dir=target.parent
            )
            temporary = Path(temporary_name)
            namespace_changed = True
            try:
                opened = os.fstat(descriptor)
            except BaseException:
                # Recover the authoritative descriptor identity through a
                # separate syscall wrapper.  This keeps one-shot injected
                # BaseExceptions between creation and tracking inside the
                # transaction without ever trusting the pathname inode.
                try:
                    opened = os.stat(descriptor)
                    inode = (opened.st_dev, opened.st_ino)
                    temporaries.append((temporary, inode, target, raw))
                except BaseException:
                    pass
                raise
            inode = (opened.st_dev, opened.st_ino)
            temporaries.append((temporary, inode, target, raw))
            os.fchmod(descriptor, 0o600)
            handle = os.fdopen(descriptor, "wb")
            descriptor = -1
            with handle:
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
        finally:
            if descriptor >= 0:
                try:
                    os.close(descriptor)
                except BaseException:
                    pass
            # No pathname-only cleanup is permitted: until the descriptor
            # identity is tracked, that name could already denote a foreign
            # replacement.  A tracked temp is cleaned by the outer rollback.

    try:
        for target, raw in targets:
            create_tracked_temporary(target, raw)
        for temporary, inode, target, _ in temporaries:
            # Register the temporary's identity against the eventual target in
            # both rollback scopes *before* link(2).  In particular, the outer
            # preparer must own a successfully linked target before this
            # function can return, so a KeyboardInterrupt cannot land in a
            # function-return -> caller-registration gap.
            ownership = (target, inode)
            owned_targets.append(ownership)
            if ownership_sink is not None:
                ownership_sink.append(ownership)
            os.link(temporary, target)
            namespace_changed = True
        for temporary, inode, target, _ in temporaries:
            target_info = os.lstat(target)
            protocol.require(
                stat.S_ISREG(target_info.st_mode)
                and (target_info.st_dev, target_info.st_ino) == inode,
                f"builder output inode changed during publication: {target}",
            )
            protocol.require(
                unlink_if_owned(temporary, inode),
                f"builder temporary inode changed: {temporary}",
            )
        fsync_parent()
        for _, inode, target, raw in temporaries:
            require_owned_readback(target, inode, raw)

        # Producer validation, target hashing/readback, and its success record
        # remain inside the rollback boundary.  This is deliberately the last
        # failure-capable producer operation.
        post_link_check()
    except BaseException:
        for target, inode in reversed(owned_targets):
            try:
                namespace_changed = (
                    unlink_if_owned(target, inode) or namespace_changed
                )
            except BaseException:
                pass
        for temporary, inode, _, _ in reversed(temporaries):
            try:
                namespace_changed = (
                    unlink_if_owned(temporary, inode) or namespace_changed
                )
            except BaseException:
                pass
        if namespace_changed:
            try:
                fsync_parent()
            except BaseException:
                pass
        raise
    finally:
        removed = False
        for temporary, inode, _, _ in temporaries:
            try:
                removed = unlink_if_owned(temporary, inode) or removed
            except BaseException:
                pass
        if removed:
            try:
                fsync_parent()
            except BaseException:
                pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--holdout-ontology", type=Path, required=True)
    parser.add_argument("--receiver-ontology", type=Path, required=True)
    parser.add_argument("--historical-anchors", type=Path, required=True)
    parser.add_argument("--templates", type=Path, required=True)
    parser.add_argument("--field-rules", type=Path, required=True)
    parser.add_argument("--graph-salt", type=Path, required=True)
    parser.add_argument("--source-bank", type=Path, required=True)
    parser.add_argument("--source-mapping", type=Path, required=True)
    parser.add_argument("--graph-output", type=Path, required=True)
    parser.add_argument("--candidate-output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project_root = protocol.validate_project_root(args.project_root)
    v2_before = protocol.validate_v2_public_inputs(project_root)
    expected_private_names = {
        "holdout_ontology": "eval_holdout_source_ontology_private48_v3.json",
        "receiver_ontology": "receiver_ontology_private56_v3.json",
        "historical_anchors": "historical_receiver_anchors_private8_v3.json",
        "templates": "causal_stage0_templates_private_v3.json",
        "field_rules": "causal_stage0_field_rules_private_v3.json",
        "graph_salt": "causal_graph_assignment_salt_v3.txt",
        "graph_output": "causal_stage0_candidate_graph_private576_v3.json",
        "candidate_output": "causal_stage0_candidates_private576_v3.json",
    }
    for argument, expected_name in expected_private_names.items():
        protocol.require(
            getattr(args, argument).name == expected_name,
            f"{argument} basename must be exactly {expected_name}",
        )
    for supplied, expected in ((args.source_bank, protocol.V2_BANK), (args.source_mapping, protocol.V2_MAPPING)):
        relative = protocol.validate_runtime_read_path(
            project_root, supplied, allow_v2=True
        )
        protocol.require(
            relative == expected.as_posix(), "v2 upstream path is not exact"
        )
    bank = protocol.load_json(args.source_bank, project_root=project_root, allow_v2=True)
    protocol.load_json(args.source_mapping, project_root=project_root, allow_v2=True)
    holdout = protocol.load_json(args.holdout_ontology, private_root=args.private_root)
    receivers = protocol.load_json(args.receiver_ontology, private_root=args.private_root)
    historical = protocol.load_json(args.historical_anchors, private_root=args.private_root)
    validate_templates_and_fields(args.templates, args.field_rules, private_root=args.private_root)
    salt = _load_salt(args.graph_salt, args.private_root)
    for output in (args.graph_output, args.candidate_output):
        protocol.validate_private_output_path(args.private_root, output)
    graph, manifest = build_candidate_graph(
        holdout_payload=holdout,
        receiver_payload=receivers,
        historical_payload=historical,
        source_bank_payload=bank,
        graph_assignment_salt=salt,
    )
    _require_v2_hashes_unchanged(project_root, v2_before)

    def validate_and_publish_producer_record() -> None:
        _require_v2_hashes_unchanged(project_root, v2_before)
        graph_file_sha256 = protocol.sha256_file(args.graph_output)
        candidate_file_sha256 = protocol.sha256_file(args.candidate_output)
        protocol.require(
            graph_file_sha256
            == protocol.sha256_bytes(protocol.canonical_json_bytes(dict(graph))),
            "candidate graph producer readback mismatch",
        )
        protocol.require(
            candidate_file_sha256
            == protocol.sha256_bytes(protocol.canonical_json_bytes(dict(manifest))),
            "candidate manifest producer readback mismatch",
        )
        producer_record = {
            "status": "built_not_authorized",
            "candidate_count": protocol.CANDIDATE_COUNT,
            "graph_sha256": graph["graph_sha256"],
            "graph_file_sha256": graph_file_sha256,
            "candidate_file_sha256": candidate_file_sha256,
        }
        print(
            protocol.canonical_json_bytes(producer_record).decode("ascii"), end=""
        )

    _write_graph_manifest_transaction(
        args.graph_output,
        graph,
        args.candidate_output,
        manifest,
        post_link_check=validate_and_publish_producer_record,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
