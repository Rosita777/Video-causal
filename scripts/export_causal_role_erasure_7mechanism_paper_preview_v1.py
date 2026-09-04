#!/usr/bin/env python3
"""Export writing-preview tables from post-unblinding exploratory CSVs.

This is deliberately a formatter, not an evaluator.  It reads only the ten
aggregate CSV inputs named in ``INPUT_CANDIDATES`` and never accepts, resolves,
or reads an anonymity/method key.  Outputs are fresh-only and carry an explicit
NOT_CANONICAL writing-preview label both in metadata and in every detached CSV.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


PROTOCOL = "causal_role_erasure_7m_paper_preview_tables_v1"
SCIENTIFIC_LABEL = "POST_UNBLINDING_EXPLORATORY_PRELIMINARY"
USE_RESTRICTION = "WRITING_PREVIEW_ONLY_NOT_FOR_FINAL_SCIENTIFIC_CLAIMS"
SOURCE_USE_RESTRICTION = USE_RESTRICTION
CANONICAL_STATUS = "NOT_CANONICAL"
FINAL_AUTHORITY = "FINAL_HUMAN_CANONICAL_SCORES_REMAIN_PRIMARY"

STREAMS = (
    "wan_original",
    "matched_control",
    "V4",
    "cogvideox_original",
    "negative_prompt",
    "videoeraser_official",
    "t2vunlearning_adapted",
    "safree_cogvideox",
)
DISPLAY_NAMES = {
    "wan_original": "Original (Wan)",
    "matched_control": "Matched Control",
    "V4": "Ours (V4)",
    "cogvideox_original": "Original (CogVideoX)",
    "negative_prompt": "Negative Prompt",
    "videoeraser_official": "VideoEraser",
    "t2vunlearning_adapted": "T2VUnlearning",
    "safree_cogvideox": "SAFREE",
}
WAN_STREAMS = {"wan_original", "matched_control", "V4"}
COG_STREAMS = set(STREAMS) - WAN_STREAMS
MECHANISMS = (
    "water_impact",
    "rigid_collision",
    "brittle_fracture",
    "powder_impact",
    "elastic_deformation",
    "material_release",
    "surface_trace",
)
MECHANISM_NAMES = {
    "water_impact": "Water impact",
    "rigid_collision": "Rigid collision",
    "brittle_fracture": "Brittle fracture",
    "powder_impact": "Powder impact",
    "elastic_deformation": "Elastic deformation",
    "material_release": "Material release",
    "surface_trace": "Surface trace",
}
HEADLINE_CONTRASTS = (
    "matched_control",
    "negative_prompt",
    "videoeraser_official",
    "t2vunlearning_adapted",
    "safree_cogvideox",
)
NONINFERIORITY_METRICS = (
    "specificity_utility",
    "receiver_preservation",
    "video_quality",
    "usable_fraction",
    "specificity_receiver_preservation",
    "specificity_video_quality",
    "specificity_usable_fraction",
)
IDENTIFICATION_CONTROLS = ("generic_paraphrase", "bystander_token")
ROLE_SUBSETS = ("implicit_footprint", "held_out_source")
SENSITIVITY_METRICS = (
    "original_capability_coverage",
    "causal_usable_fraction",
    "specificity_usable_fraction",
    "ces",
    "specificity_utility",
)

# Compatibility is intentionally narrow: all selected files must be regular,
# direct children of the metrics directory.  No recursive discovery occurs.
INPUT_CANDIDATES = {
    "macro": ("macro_metrics.csv", "macro.csv"),
    "mechanism": ("mechanism_metrics.csv", "metrics_by_mechanism.csv"),
    "headline": ("headline_contrasts.csv", "headline.csv"),
    "headline_mechanism": (
        "headline_mechanism_contrasts.csv",
        "headline_contrasts_by_mechanism.csv",
    ),
    "noninferiority": ("noninferiority.csv", "noninferiority_metrics.csv"),
    "identification": (
        "identification_contrasts.csv",
        "identification.csv",
    ),
    "role": ("role_conditioned_evidence.csv", "role_conditioned.csv"),
    "eligibility": (
        "original_mean_rule_eligibility.csv",
        "original_eligibility.csv",
    ),
    "shared": ("shared_capability_subsets.csv", "shared_subsets.csv"),
    "sensitivity": (
        "ab_order_sensitivity_macro.csv",
        "order_sensitivity_macro.csv",
    ),
}

NOTICE = (
    "**POST_UNBLINDING_EXPLORATORY_PRELIMINARY — "
    "WRITING_PREVIEW_ONLY_NOT_FOR_FINAL_SCIENTIFIC_CLAIMS — "
    "NOT CANONICAL. Final human-canonical scores remain the sole final authority.**"
)


class PaperPreviewError(ValueError):
    """An input or output violates the writing-preview boundary."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PaperPreviewError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_ref(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def relative_ref(root: Path, path: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(root)),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def verify_refs_unchanged(
    paths: Mapping[str, Path],
    initial_refs: Mapping[str, Mapping[str, Any]],
    *,
    label: str,
) -> None:
    """Fail closed if any bound regular file changes before publication."""

    require(set(paths) == set(initial_refs), f"{label}: reference inventory changed")
    for name, path in sorted(paths.items()):
        require(path.is_file() and not path.is_symlink(), f"{label}/{name}: file disappeared or became a symlink")
        observed = file_ref(path)
        expected = initial_refs[name]
        require(
            observed["path"] == expected.get("path")
            and observed["sha256"] == expected.get("sha256")
            and observed["size_bytes"] == expected.get("size_bytes"),
            f"{label}/{name}: file changed during export",
        )


def discover_inputs(metrics_root: Path) -> dict[str, Path]:
    require(metrics_root.is_dir() and not metrics_root.is_symlink(), f"metrics root is not a regular directory: {metrics_root}")
    resolved: dict[str, Path] = {}
    for label, candidates in INPUT_CANDIDATES.items():
        matches = [metrics_root / name for name in candidates if (metrics_root / name).is_file()]
        require(matches, f"missing {label} CSV; tried: {', '.join(candidates)}")
        require(len(matches) == 1, f"ambiguous {label} CSV: {', '.join(path.name for path in matches)}")
        path = matches[0]
        require(not path.is_symlink() and path.parent.resolve() == metrics_root.resolve(), f"{label} CSV must be a direct regular child")
        resolved[label] = path
    return resolved


def read_csv(path: Path, label: str) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames is not None, f"{label} has no header")
        rows = [dict(row) for row in reader]
    require(rows, f"{label} is empty")
    return list(reader.fieldnames), rows


def require_fields(fields: Sequence[str], required: Sequence[str], label: str) -> None:
    missing = [field for field in required if field not in fields]
    require(not missing, f"{label} is missing fields: {', '.join(missing)}")


def validate_labels(rows: Sequence[Mapping[str, str]], label: str) -> None:
    for index, row in enumerate(rows):
        require(row.get("canonical_status") == CANONICAL_STATUS, f"{label} row {index}: canonical status is not NOT_CANONICAL")
        require(row.get("scientific_label") == SCIENTIFIC_LABEL, f"{label} row {index}: scientific label changed")
        require(row.get("use_restriction") == SOURCE_USE_RESTRICTION, f"{label} row {index}: source use restriction changed")
        require(row.get("final_authority") == FINAL_AUTHORITY, f"{label} row {index}: final authority changed")
        require(
            row.get("formal_claims_permitted") == "False",
            f"{label} row {index}: formal_claims_permitted must be the CSV boolean False",
        )


def finite_float(value: str, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise PaperPreviewError(f"{label} is not numeric: {value!r}") from exc
    require(math.isfinite(result), f"{label} is not finite")
    return result


def integer(value: str, label: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise PaperPreviewError(f"{label} is not an integer: {value!r}") from exc
    return result


def exact_inventory(rows: Sequence[Mapping[str, str]], field: str, expected: Sequence[str], label: str) -> None:
    observed = [row.get(field, "") for row in rows]
    require(len(observed) == len(expected), f"{label}: expected {len(expected)} rows, observed {len(observed)}")
    require(Counter(observed) == Counter(expected), f"{label}: {field} inventory changed")


def load_and_validate(metrics_root: Path) -> tuple[dict[str, Path], dict[str, list[dict[str, str]]]]:
    paths = discover_inputs(metrics_root)
    loaded: dict[str, list[dict[str, str]]] = {}
    fields_by_label: dict[str, list[str]] = {}
    for label, path in paths.items():
        fields, rows = read_csv(path, label)
        require_fields(
            fields,
            ("canonical_status", "scientific_label", "use_restriction", "final_authority", "formal_claims_permitted"),
            label,
        )
        validate_labels(rows, label)
        fields_by_label[label] = fields
        loaded[label] = rows

    require_fields(
        fields_by_label["macro"],
        (
            "stream", "mechanism_count", "ces", "specificity_utility",
            "original_capability_coverage", "causal_usable_fraction",
            "specificity_usable_fraction", "causal_receiver_preservation",
            "causal_video_quality", "source_absent_rate", "footprint_absent_rate",
            "strict_success_rate", "eligible_source_clear_to_absent_rate",
        ),
        "macro",
    )
    exact_inventory(loaded["macro"], "stream", STREAMS, "macro")
    require(all(row["mechanism_count"] == "7" for row in loaded["macro"]), "macro mechanism count changed")

    require_fields(fields_by_label["mechanism"], ("stream", "mechanism", "ces"), "mechanism")
    observed_pairs = Counter((row["stream"], row["mechanism"]) for row in loaded["mechanism"])
    expected_pairs = Counter((stream, mechanism) for stream in STREAMS for mechanism in MECHANISMS)
    require(observed_pairs == expected_pairs, "mechanism metrics must be the complete 8x7 inventory")

    require_fields(
        fields_by_label["headline"],
        ("contrast", "estimable", "point_estimate", "ci95_lower", "ci95_upper", "one_sided_p", "holm_adjusted_p"),
        "headline",
    )
    exact_inventory(loaded["headline"], "contrast", HEADLINE_CONTRASTS, "headline")

    require_fields(fields_by_label["headline_mechanism"], ("contrast", "mechanism", "n", "estimable", "point_estimate"), "headline mechanism")
    require(
        Counter((row["contrast"], row["mechanism"]) for row in loaded["headline_mechanism"])
        == Counter((contrast, mechanism) for contrast in HEADLINE_CONTRASTS for mechanism in MECHANISMS),
        "headline mechanism contrasts must be the complete 5x7 inventory",
    )

    require_fields(
        fields_by_label["noninferiority"],
        ("metric", "role", "margin", "point_estimate", "ci95_lower", "ci95_upper", "one_sided_p", "exploratory_noninferiority_threshold_pattern"),
        "noninferiority",
    )
    exact_inventory(loaded["noninferiority"], "metric", NONINFERIORITY_METRICS, "noninferiority")

    require_fields(
        fields_by_label["identification"],
        ("control", "point_estimate", "ci95_lower", "ci95_upper", "one_sided_p", "holm_adjusted_p", "exploratory_novelty_threshold_pattern"),
        "identification",
    )
    exact_inventory(loaded["identification"], "control", IDENTIFICATION_CONTROLS, "identification")

    require_fields(
        fields_by_label["role"],
        ("subset", "per_mechanism_n", "point_estimate", "ci95_lower", "ci95_upper", "one_sided_p", "exploratory_positive_point_pattern"),
        "role conditioned",
    )
    exact_inventory(loaded["role"], "subset", ROLE_SUBSETS, "role conditioned")

    require_fields(fields_by_label["eligibility"], ("backbone_family", "case_id", "mechanism", "eligible"), "eligibility")
    require(len(loaded["eligibility"]) == 336, "Original eligibility must contain 336 causal rows")
    require(
        Counter((row["backbone_family"], row["mechanism"]) for row in loaded["eligibility"])
        == Counter((backbone, mechanism) for backbone in ("wan", "cogvideox") for mechanism in MECHANISMS for _ in range(24)),
        "Original eligibility must contain 24 rows per backbone/mechanism",
    )
    require(all(integer(row["eligible"], "eligible") in (0, 1) for row in loaded["eligibility"]), "eligibility flag outside {0,1}")

    require_fields(fields_by_label["shared"], ("case_id", "mechanism", "wan_original_eligible", "cogvideox_original_eligible", "shared_eligible"), "shared")
    require(len(loaded["shared"]) == 168, "shared capability table must contain 168 semantic cases")
    require(Counter(row["mechanism"] for row in loaded["shared"]) == Counter(mechanism for mechanism in MECHANISMS for _ in range(24)), "shared table must contain 24 rows per mechanism")
    require(all(integer(row["shared_eligible"], "shared eligible") in (0, 1) for row in loaded["shared"]), "shared flag outside {0,1}")

    require_fields(
        fields_by_label["sensitivity"],
        (
            "stream", "sensitivity_only", "replaces_primary",
            "primary_nonlinear_order", "sensitivity_order",
            *tuple(
                field
                for metric in SENSITIVITY_METRICS
                for field in (
                    f"pass_a_{metric}", f"pass_b_{metric}",
                    f"mean_of_pass_macros_{metric}",
                    f"primary_atomic_mean_{metric}",
                    f"mean_of_pass_macros_minus_primary_{metric}",
                )
            ),
        ),
        "A/B order sensitivity",
    )
    exact_inventory(loaded["sensitivity"], "stream", STREAMS, "A/B order sensitivity")
    for index, row in enumerate(loaded["sensitivity"]):
        require(row["sensitivity_only"] == "True", f"sensitivity row {index}: sensitivity_only must be True")
        require(row["replaces_primary"] == "False", f"sensitivity row {index}: replaces_primary must be False")
        for metric in SENSITIVITY_METRICS:
            pass_a = finite_float(row[f"pass_a_{metric}"], f"sensitivity row {index} pass A {metric}")
            pass_b = finite_float(row[f"pass_b_{metric}"], f"sensitivity row {index} pass B {metric}")
            pass_mean = finite_float(row[f"mean_of_pass_macros_{metric}"], f"sensitivity row {index} pass-wise {metric} mean")
            primary = finite_float(row[f"primary_atomic_mean_{metric}"], f"sensitivity row {index} primary {metric}")
            difference = finite_float(
                row[f"mean_of_pass_macros_minus_primary_{metric}"],
                f"sensitivity row {index} {metric} difference",
            )
            require(math.isclose(pass_mean, (pass_a + pass_b) / 2, abs_tol=1e-12), f"sensitivity row {index}: pass-wise {metric} mean changed")
            require(math.isclose(difference, pass_mean - primary, abs_tol=1e-12), f"sensitivity row {index}: {metric} difference changed")

    # Validate numeric data that this formatter will expose.
    for label in ("macro", "mechanism"):
        for index, row in enumerate(loaded[label]):
            finite_float(row["ces"], f"{label} row {index} CES")
    for label in ("headline", "headline_mechanism"):
        for index, row in enumerate(loaded[label]):
            estimable = integer(row["estimable"], f"{label} row {index} estimable")
            require(estimable in (0, 1), f"{label} row {index}: estimable outside {{0,1}}")
            if estimable:
                finite_float(row["point_estimate"], f"{label} row {index} point estimate")
    return paths, loaded


def preview_labels() -> dict[str, str]:
    return {
        "canonical_status": CANONICAL_STATUS,
        "scientific_label": SCIENTIFIC_LABEL,
        "use_restriction": USE_RESTRICTION,
        "final_authority": FINAL_AUTHORITY,
    }


def fmt(value: str | float, digits: int = 4) -> str:
    number = finite_float(str(value), "display value")
    return f"{number:.{digits}f}"


def signed(value: float, digits: int = 4) -> str:
    return f"{value:+.{digits}f}"


def md_escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]], align_right_from: int = 1) -> str:
    alignment = ["---" if index < align_right_from else "---:" for index in range(len(headers))]
    lines = [
        "| " + " | ".join(md_escape(value) for value in headers) + " |",
        "| " + " | ".join(alignment) + " |",
    ]
    lines.extend("| " + " | ".join(md_escape(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def csv_bytes(rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> bytes:
    require(bool(rows), "cannot write an empty preview table")
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def write_new(path: Path, value: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(value)
    path.chmod(0o444)


def markdown_document(title: str, body: str, notes: Sequence[str]) -> bytes:
    lines = [f"# {title}", "", f"> {NOTICE}", "", body]
    if notes:
        lines.extend(["", "Notes:", ""] + [f"- {note}" for note in notes])
    lines.append("")
    return "\n".join(lines).encode("utf-8")


def build_overall(macro_rows: Sequence[Mapping[str, str]]) -> tuple[list[dict[str, str]], bytes]:
    by_stream = {row["stream"]: row for row in macro_rows}
    output: list[dict[str, str]] = []
    labels = preview_labels()
    numeric_fields = (
        "ces", "specificity_utility", "original_capability_coverage",
        "causal_usable_fraction", "specificity_usable_fraction",
        "causal_receiver_preservation", "causal_video_quality",
        "source_absent_rate", "footprint_absent_rate", "strict_success_rate",
        "eligible_source_clear_to_absent_rate",
    )
    for stream in STREAMS:
        source = by_stream[stream]
        backbone = "Wan" if stream in WAN_STREAMS else "CogVideoX"
        original = by_stream["wan_original" if stream in WAN_STREAMS else "cogvideox_original"]
        row = {
            **labels,
            "stream": stream,
            "method": DISPLAY_NAMES[stream],
            "backbone": backbone,
            "delta_ces_vs_backbone_original_descriptive": signed(
                finite_float(source["ces"], "CES") - finite_float(original["ces"], "Original CES")
            ),
            "delta_scope": "WITHIN_BACKBONE_DESCRIPTIVE_NOT_REGISTERED_CROSS_BACKBONE_INFERENCE",
        }
        row.update({field: fmt(source[field]) for field in numeric_fields})
        output.append(row)

    headers = (
        "Method", "Backbone", "CES ↑", "Δ CES vs backbone Original†",
        "Specificity utility ↑", "Original capability", "Causal usable",
        "Specificity usable", "Causal receiver", "Causal video quality",
        "Source absent", "Footprint absent", "Strict success",
    )
    table_rows = [
        (
            row["method"], row["backbone"], row["ces"],
            row["delta_ces_vs_backbone_original_descriptive"],
            row["specificity_utility"], row["original_capability_coverage"],
            row["causal_usable_fraction"], row["specificity_usable_fraction"],
            row["causal_receiver_preservation"], row["causal_video_quality"],
            row["source_absent_rate"], row["footprint_absent_rate"],
            row["strict_success_rate"],
        )
        for row in output
    ]
    body = markdown_table(headers, table_rows)
    document = markdown_document(
        "Overall results (8 methods)",
        body,
        (
            "All metrics are shown on a 0–1 scale; values are VLM A/B-mean writing previews.",
            "† Δ CES is a descriptive within-backbone subtraction from the matching Original. It is not a registered cross-backbone inferential contrast.",
            "No row authorizes a formal win claim.",
        ),
    )
    return output, document


def build_ces_wide(
    mechanism_rows: Sequence[Mapping[str, str]],
    macro_rows: Sequence[Mapping[str, str]],
) -> tuple[list[dict[str, str]], bytes]:
    by_pair = {(row["stream"], row["mechanism"]): row for row in mechanism_rows}
    macro_by_stream = {row["stream"]: row for row in macro_rows}
    labels = preview_labels()
    output: list[dict[str, str]] = []
    for stream in STREAMS:
        row = {**labels, "stream": stream, "method": DISPLAY_NAMES[stream]}
        row.update({mechanism: fmt(by_pair[(stream, mechanism)]["ces"]) for mechanism in MECHANISMS})
        row["macro_ces"] = fmt(macro_by_stream[stream]["ces"])
        calculated = sum(finite_float(by_pair[(stream, mechanism)]["ces"], "mechanism CES") for mechanism in MECHANISMS) / 7
        require(math.isclose(calculated, finite_float(macro_by_stream[stream]["ces"], "macro CES"), abs_tol=1e-12), f"{stream}: macro CES is not the equal-weight seven-mechanism mean")
        output.append(row)

    headers = ["Method", *[MECHANISM_NAMES[value] for value in MECHANISMS], "Macro"]
    table_rows = [[row["method"], *[row[value] for value in MECHANISMS], row["macro_ces"]] for row in output]
    document = markdown_document(
        "CES by mechanism (8 × 7 + macro)",
        markdown_table(headers, table_rows),
        (
            "Macro is the equal-weight mean over the seven mechanisms.",
            "This is a descriptive writing preview, not a human-canonical table.",
        ),
    )
    return output, document


def _count_cell(numerator: int, denominator: int) -> str:
    return f"{numerator}/{denominator} ({numerator / denominator:.4f})"


def build_capability(
    eligibility_rows: Sequence[Mapping[str, str]],
    shared_rows: Sequence[Mapping[str, str]],
) -> tuple[list[dict[str, str]], bytes]:
    labels = preview_labels()
    output: list[dict[str, str]] = []
    definitions = (
        ("wan_original", "Original (Wan)", eligibility_rows, "wan", "eligible"),
        ("cogvideox_original", "Original (CogVideoX)", eligibility_rows, "cogvideox", "eligible"),
        ("shared", "Shared (both Originals)", shared_rows, None, "shared_eligible"),
    )
    for row_id, display_name, source_rows, backbone, field in definitions:
        row = {**labels, "row_id": row_id, "capability_set": display_name}
        total_n = 0
        total_d = 0
        for mechanism in MECHANISMS:
            selected = [
                value for value in source_rows
                if value["mechanism"] == mechanism
                and (backbone is None or value["backbone_family"] == backbone)
            ]
            require(len(selected) == 24, f"{row_id}/{mechanism}: expected 24 capability rows")
            numerator = sum(integer(value[field], field) for value in selected)
            row[mechanism] = _count_cell(numerator, 24)
            total_n += numerator
            total_d += 24
        row["all_7_mechanisms"] = _count_cell(total_n, total_d)
        output.append(row)

    # The shared file redundantly contains each backbone flag; require exact
    # agreement with the independently supplied Original-eligibility table.
    original_by_pair = {
        (row["backbone_family"], row["case_id"]): integer(row["eligible"], "eligible")
        for row in eligibility_rows
    }
    for row in shared_rows:
        case_id = row["case_id"]
        require(integer(row["wan_original_eligible"], "wan original eligible") == original_by_pair[("wan", case_id)], f"{case_id}: Wan capability mismatch")
        require(integer(row["cogvideox_original_eligible"], "CogVideoX original eligible") == original_by_pair[("cogvideox", case_id)], f"{case_id}: CogVideoX capability mismatch")
        expected_shared = original_by_pair[("wan", case_id)] * original_by_pair[("cogvideox", case_id)]
        require(integer(row["shared_eligible"], "shared eligible") == expected_shared, f"{case_id}: shared capability mismatch")

    headers = ["Capability set", *[MECHANISM_NAMES[value] for value in MECHANISMS], "All 7"]
    table_rows = [[row["capability_set"], *[row[value] for value in MECHANISMS], row["all_7_mechanisms"]] for row in output]
    document = markdown_document(
        "Original capability and shared subsets (2 × 7 + shared)",
        markdown_table(headers, table_rows),
        (
            "Cells are eligible/total (rate). Each mechanism denominator is 24; the all-mechanism denominator is 168.",
            "Shared requires both Wan Original and CogVideoX Original to be eligible for the same semantic case.",
        ),
    )
    return output, document


def build_sensitivity(
    sensitivity_rows: Sequence[Mapping[str, str]],
) -> tuple[list[dict[str, str]], bytes]:
    """Format the A/B nonlinear-order sensitivity without replacing primary."""

    labels = preview_labels()
    by_stream = {row["stream"]: row for row in sensitivity_rows}
    output: list[dict[str, str]] = []
    for stream in STREAMS:
        source = by_stream[stream]
        output.append(
            {
                **labels,
                "stream": stream,
                "method": DISPLAY_NAMES[stream],
                "primary_atomic_mean_ces": fmt(source["primary_atomic_mean_ces"]),
                "pass_a_macro_ces": fmt(source["pass_a_ces"]),
                "pass_b_macro_ces": fmt(source["pass_b_ces"]),
                "pass_wise_mean_macro_ces": fmt(source["mean_of_pass_macros_ces"]),
                "pass_wise_mean_minus_primary_ces": signed(
                    finite_float(
                        source["mean_of_pass_macros_minus_primary_ces"],
                        "sensitivity CES difference",
                    )
                ),
                "sensitivity_only": "True",
                "replaces_primary": "False",
                "primary_nonlinear_order": source["primary_nonlinear_order"],
                "sensitivity_order": source["sensitivity_order"],
            }
        )

    headers = (
        "Method", "Primary CES", "Pass A macro CES", "Pass B macro CES",
        "Pass-wise mean CES", "Difference†",
    )
    table_rows = [
        (
            row["method"], row["primary_atomic_mean_ces"],
            row["pass_a_macro_ces"], row["pass_b_macro_ces"],
            row["pass_wise_mean_macro_ces"],
            row["pass_wise_mean_minus_primary_ces"],
        )
        for row in output
    ]
    document = markdown_document(
        "A/B nonlinear-order sensitivity (CES)",
        markdown_table(headers, table_rows),
        (
            "This is a sensitivity analysis only and does not replace the primary atomic-mean-first result.",
            "Primary CES applies the A/B atomic mean before eligibility, usability, CES, and macro aggregation.",
            "Pass-wise mean independently applies the nonlinear pipeline to pass A and pass B, then averages their macro CES values.",
            "† Difference = pass-wise mean CES − Primary CES.",
        ),
    )
    return output, document


def _contrast_name(value: str) -> str:
    return DISPLAY_NAMES.get(value, value.replace("_", " ").title())


def _estimate_fields(row: Mapping[str, str], estimable: bool, *, has_ci: bool, has_p: bool) -> dict[str, str]:
    if not estimable:
        return {
            "point_estimate": "NE",
            "ci95_lower": "—",
            "ci95_upper": "—",
            "one_sided_p": "—",
            "holm_adjusted_p": "—",
        }
    return {
        "point_estimate": fmt(row["point_estimate"]),
        "ci95_lower": fmt(row["ci95_lower"]) if has_ci else "—",
        "ci95_upper": fmt(row["ci95_upper"]) if has_ci else "—",
        "one_sided_p": fmt(row["one_sided_p"]) if has_p else "—",
        "holm_adjusted_p": fmt(row["holm_adjusted_p"]) if has_p and row.get("holm_adjusted_p", "") else "—",
    }


def build_headline(
    headline_rows: Sequence[Mapping[str, str]],
    headline_mechanism_rows: Sequence[Mapping[str, str]],
    noninferiority_rows: Sequence[Mapping[str, str]],
    identification_rows: Sequence[Mapping[str, str]],
    role_rows: Sequence[Mapping[str, str]],
) -> tuple[list[dict[str, str]], bytes]:
    labels = preview_labels()
    output: list[dict[str, str]] = []

    headline_by_name = {row["contrast"]: row for row in headline_rows}
    for contrast in HEADLINE_CONTRASTS:
        source = headline_by_name[contrast]
        estimable = integer(source["estimable"], "headline estimable") == 1
        output.append(
            {
                **labels,
                "section": "headline_overall",
                "test": f"ours_vs_{contrast}",
                "display_name": f"Ours (V4) vs {_contrast_name(contrast)}",
                "mechanism": "macro",
                "n": "—",
                "estimability": "estimable" if estimable else "NE",
                **_estimate_fields(source, estimable, has_ci=True, has_p=True),
                "margin": "—",
                "exploratory_threshold_pattern": source.get("exploratory_positive_threshold_pattern", "0") if estimable else "0",
                "interpretation": "EXPLORATORY_ONLY_FORMAL_WIN_NOT_PERMITTED",
            }
        )

    mechanism_by_pair = {(row["contrast"], row["mechanism"]): row for row in headline_mechanism_rows}
    for contrast in HEADLINE_CONTRASTS:
        for mechanism in MECHANISMS:
            source = mechanism_by_pair[(contrast, mechanism)]
            estimable = integer(source["estimable"], "mechanism estimable") == 1
            output.append(
                {
                    **labels,
                    "section": "headline_mechanism_diagnostic",
                    "test": f"ours_vs_{contrast}",
                    "display_name": f"Ours (V4) vs {_contrast_name(contrast)}",
                    "mechanism": mechanism,
                    "n": source["n"],
                    "estimability": "estimable" if estimable else "NE",
                    **_estimate_fields(source, estimable, has_ci=False, has_p=False),
                    "margin": "—",
                    "exploratory_threshold_pattern": "—",
                    "interpretation": "MECHANISM_DIAGNOSTIC_ONLY_NO_SEPARATE_INFERENCE",
                }
            )

    ni_by_metric = {row["metric"]: row for row in noninferiority_rows}
    for metric in NONINFERIORITY_METRICS:
        source = ni_by_metric[metric]
        fields = _estimate_fields(source, True, has_ci=True, has_p=True)
        output.append(
            {
                **labels,
                "section": "noninferiority",
                "test": metric,
                "display_name": metric.replace("_", " ").title(),
                "mechanism": "macro",
                "n": "—",
                "estimability": "estimable",
                **fields,
                "margin": fmt(source["margin"]),
                "exploratory_threshold_pattern": source["exploratory_noninferiority_threshold_pattern"],
                "interpretation": f"EXPLORATORY_{source['role'].upper()}_NONINFERIORITY_PATTERN",
            }
        )

    identification_by_control = {row["control"]: row for row in identification_rows}
    for control in IDENTIFICATION_CONTROLS:
        source = identification_by_control[control]
        output.append(
            {
                **labels,
                "section": "identification",
                "test": control,
                "display_name": control.replace("_", " ").title(),
                "mechanism": "water_impact+brittle_fracture",
                "n": "—",
                "estimability": "estimable",
                **_estimate_fields(source, True, has_ci=True, has_p=True),
                "margin": fmt(source.get("specificity_margin", "-0.1")),
                "exploratory_threshold_pattern": source["exploratory_novelty_threshold_pattern"],
                "interpretation": "EXPLORATORY_IDENTIFICATION_PATTERN",
            }
        )

    role_by_subset = {row["subset"]: row for row in role_rows}
    for subset in ROLE_SUBSETS:
        source = role_by_subset[subset]
        output.append(
            {
                **labels,
                "section": "role_conditioned",
                "test": subset,
                "display_name": subset.replace("_", " ").title(),
                "mechanism": "macro",
                "n": source["per_mechanism_n"],
                "estimability": "estimable",
                **_estimate_fields(source, True, has_ci=True, has_p=True),
                "margin": "—",
                "exploratory_threshold_pattern": source["exploratory_positive_point_pattern"],
                "interpretation": "EXPLORATORY_ROLE_CONDITIONED_PATTERN",
            }
        )

    def compact(rows: Sequence[Mapping[str, str]]) -> list[list[str]]:
        return [
            [
                row["display_name"], row["mechanism"], row["n"],
                row["estimability"], row["point_estimate"],
                f"[{row['ci95_lower']}, {row['ci95_upper']}]" if row["ci95_lower"] != "—" else "—",
                row["one_sided_p"], row["holm_adjusted_p"], row["margin"],
                row["exploratory_threshold_pattern"],
            ]
            for row in rows
        ]

    common_headers = ("Test", "Mechanism", "n", "Estimability", "Estimate", "95% CI", "one-sided p", "Holm p", "Margin", "Pattern")
    overall = [row for row in output if row["section"] == "headline_overall"]
    per_mechanism = [row for row in output if row["section"] == "headline_mechanism_diagnostic"]
    ni = [row for row in output if row["section"] == "noninferiority"]
    identification = [row for row in output if row["section"] == "identification"]
    role = [row for row in output if row["section"] == "role_conditioned"]
    body = "\n\n".join(
        (
            "## Headline CES contrasts\n\n" + markdown_table(common_headers, compact(overall), align_right_from=2),
            "## Per-mechanism headline diagnostics\n\n" + markdown_table(common_headers, compact(per_mechanism), align_right_from=2),
            "## Noninferiority patterns\n\n" + markdown_table(common_headers, compact(ni), align_right_from=2),
            "## Identification controls\n\n" + markdown_table(common_headers, compact(identification), align_right_from=2),
            "## Role-conditioned evidence\n\n" + markdown_table(common_headers, compact(role), align_right_from=2),
        )
    )
    document = markdown_document(
        "Headline presentation tables",
        body,
        (
            "NE means not estimable under the frozen minimum-shared-case rule; estimate is NE and inference fields are shown as —.",
            "In particular, a non-estimable row is never rendered as p=1.0000.",
            "Pattern flags are exploratory threshold diagnostics, not registered or formal claims.",
            "Per-mechanism rows are descriptive diagnostics without separate confidence intervals or hypothesis tests.",
        ),
    )
    return output, document


def export_paper_preview(*, metrics_root: Path, output_root: Path) -> dict[str, Any]:
    metrics_root = metrics_root.resolve()
    output_root = output_root.resolve()
    require(not output_root.exists() and not output_root.is_symlink(), f"fresh-only output already exists: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    inputs, tables = load_and_validate(metrics_root)
    input_refs = {label: file_ref(path) for label, path in sorted(inputs.items())}
    implementation_path = Path(__file__).resolve()
    require(
        implementation_path.is_file() and not implementation_path.is_symlink(),
        "exporter implementation must be a regular, non-symlinked file",
    )
    implementation_bytes = implementation_path.read_bytes()
    implementation_ref = file_ref(implementation_path)

    staging = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.tmp-", dir=output_root.parent))
    try:
        overall, overall_md = build_overall(tables["macro"])
        ces, ces_md = build_ces_wide(tables["mechanism"], tables["macro"])
        capability, capability_md = build_capability(tables["eligibility"], tables["shared"])
        sensitivity, sensitivity_md = build_sensitivity(tables["sensitivity"])
        headline, headline_md = build_headline(
            tables["headline"], tables["headline_mechanism"], tables["noninferiority"],
            tables["identification"], tables["role"],
        )

        artifacts: dict[str, Path] = {}
        specs = (
            ("overall_csv", "overall_8_methods.csv", csv_bytes(overall, tuple(overall[0]))),
            ("overall_markdown", "overall_8_methods.md", overall_md),
            ("ces_csv", "ces_by_mechanism_8x7_plus_macro.csv", csv_bytes(ces, tuple(ces[0]))),
            ("ces_markdown", "ces_by_mechanism_8x7_plus_macro.md", ces_md),
            ("capability_csv", "original_capability_2x7_shared.csv", csv_bytes(capability, tuple(capability[0]))),
            ("capability_markdown", "original_capability_2x7_shared.md", capability_md),
            ("sensitivity_csv", "ab_order_sensitivity_ces.csv", csv_bytes(sensitivity, tuple(sensitivity[0]))),
            ("sensitivity_markdown", "ab_order_sensitivity_ces.md", sensitivity_md),
            ("headline_csv", "headline_presentation.csv", csv_bytes(headline, tuple(headline[0]))),
            ("headline_markdown", "headline_presentation.md", headline_md),
        )
        for label, filename, content in specs:
            path = staging / filename
            write_new(path, content)
            artifacts[label] = path

        label_payload = {
            "schema_version": 1,
            "protocol": PROTOCOL,
            **preview_labels(),
            "formal_claims_permitted": False,
            "human_calibrated": False,
            "source_use_restriction": SOURCE_USE_RESTRICTION,
            "input_boundary": "TEN_AGGREGATE_CSV_FILES_ONLY_NO_KEY_INPUTS",
        }
        label_path = staging / "PREVIEW_LABEL.json"
        write_new(label_path, json_bytes(label_payload))
        artifacts["preview_label"] = label_path

        implementation_snapshot_path = staging / "implementation_snapshot.py"
        write_new(implementation_snapshot_path, implementation_bytes)
        require(
            sha256_file(implementation_snapshot_path) == implementation_ref["sha256"],
            "exporter implementation snapshot differs from its source",
        )
        artifacts["implementation_snapshot"] = implementation_snapshot_path

        readme = "\n".join(
            (
                "# 04 paper tables — writing preview only",
                "",
                f"> {NOTICE}",
                "",
                "These tables format the post-unblinding VLM A/B-mean aggregate metrics for drafting and internal presentation.",
                "They are not human calibrated, do not authorize formal scientific claims, and must not be cited as final results.",
                "The exporter consumes only the ten aggregate CSV inputs listed in `paper_preview_manifest.json`; it does not accept or read key files.",
                "",
                "Artifacts:",
                "",
                "- `overall_8_methods.*`: eight-method macro table, including a descriptive within-backbone CES delta.",
                "- `ces_by_mechanism_8x7_plus_macro.*`: CES for eight methods across seven mechanisms plus equal-weight macro.",
                "- `original_capability_2x7_shared.*`: two Original capability rows and their shared subset.",
                "- `ab_order_sensitivity_ces.*`: sensitivity-only comparison of primary versus pass-wise nonlinear aggregation; it does not replace the primary result.",
                "- `headline_presentation.*`: headline, per-mechanism, noninferiority, identification, and role-conditioned previews.",
                "- `PREVIEW_LABEL.json`: machine-readable evidence boundary.",
                "- `implementation_snapshot.py`: read-only hash-bound snapshot of the exporter that produced this directory.",
                "",
                "Final authority: human-canonical scores and tables produced by the frozen formal workflow.",
                "",
            )
        ).encode("utf-8")
        readme_path = staging / "README.md"
        write_new(readme_path, readme)
        artifacts["readme"] = readme_path

        manifest = {
            "schema_version": 1,
            "protocol": PROTOCOL,
            **preview_labels(),
            "formal_claims_permitted": False,
            "human_calibrated": False,
            "exporter_implementation": {
                "source": implementation_ref,
                "snapshot": relative_ref(staging, implementation_snapshot_path),
                "source_matches_snapshot": True,
            },
            "input_boundary": {
                "policy": "TEN_AGGREGATE_CSV_FILES_ONLY_NO_KEY_INPUTS",
                "files": input_refs,
            },
            "tables": {
                "overall_rows": len(overall),
                "ces_rows": len(ces),
                "capability_rows": len(capability),
                "sensitivity_rows": len(sensitivity),
                "headline_long_rows": len(headline),
            },
            "artifacts": {label: relative_ref(staging, path) for label, path in sorted(artifacts.items())},
        }
        manifest_path = staging / "paper_preview_manifest.json"
        write_new(manifest_path, json_bytes(manifest))

        # The parsed tables and manifest are bound to the initial bytes.  Rehash
        # every allowed aggregate input immediately before the atomic publish to
        # close the read/build/publish TOCTOU window.  Bind the implementation in
        # the same way so the recorded source SHA cannot race the snapshot.
        verify_refs_unchanged(inputs, input_refs, label="aggregate input")
        verify_refs_unchanged(
            {"exporter": implementation_path},
            {"exporter": implementation_ref},
            label="implementation",
        )
        require(not output_root.exists(), f"output appeared during build: {output_root}")
        os.replace(staging, output_root)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--metrics-root", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        help="fresh-only destination; defaults to a sibling 04_paper_tables directory",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    metrics_root = args.metrics_root.resolve()
    output_root = args.output_root.resolve() if args.output_root else metrics_root.parent / "04_paper_tables"
    try:
        manifest = export_paper_preview(metrics_root=metrics_root, output_root=output_root)
    except PaperPreviewError as exc:
        raise SystemExit(f"error: {exc}") from exc
    print(
        json.dumps(
            {
                "status": "paper_preview_tables_written",
                "output_root": str(output_root),
                "scientific_label": manifest["scientific_label"],
                "use_restriction": manifest["use_restriction"],
                "final_authority": manifest["final_authority"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
