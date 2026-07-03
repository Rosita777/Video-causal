#!/usr/bin/env python3
"""Score Method C0 counterfactual-grid VLM predictions."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any, Iterable, Sequence


VARIANTS = ["original", "remove_target", "footprint_only", "target_only"]
EXPECTED_STATES = {
    "original": (True, True),
    "remove_target": (False, False),
    "footprint_only": (False, True),
    "target_only": (True, False),
}

TRUE_LABELS = {"1", "true", "t", "yes", "y", "present", "visible"}
FALSE_LABELS = {"0", "false", "f", "no", "n", "absent", "not_visible", "none"}

VARIANT_SCORE_FIELDS = [
    "item_key",
    "item_id",
    "item_index",
    "pair_id",
    "mechanism_type",
    "baseline",
    "variant_role",
    "target_concept",
    "expected_effect",
    "expected_target_visible",
    "expected_footprint_visible",
    "observed_target_visible",
    "observed_footprint_visible",
    "target_match",
    "footprint_match",
    "quality_ok",
    "variant_pass",
    "video_quality",
    "confidence",
    "final_label",
    "notes",
]

ITEM_SCORE_FIELDS = [
    "item_key",
    "item_id",
    "item_index",
    "pair_id",
    "mechanism_type",
    "original_valid",
    "counterfactual_pass",
    "c0_grid_pass",
    "missing_variants",
    "failed_variants",
    "failure_mode",
    "original_pass",
    "remove_target_pass",
    "footprint_only_pass",
    "target_only_pass",
    "original_observed",
    "remove_target_observed",
    "footprint_only_observed",
    "target_only_observed",
]


def normalize_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if not text:
        return None
    if text in TRUE_LABELS:
        return True
    if text in FALSE_LABELS:
        return False
    return None


def bool_to_text(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    return str(value)


def item_key(row: dict[str, str]) -> str:
    return (
        str(row.get("pair_id", "")).strip()
        or str(row.get("item_id", "")).strip()
        or str(row.get("item_index", "")).strip()
    )


def variant_role(row: dict[str, str]) -> str:
    return str(row.get("variant_role") or row.get("baseline") or "").strip()


def observed_footprint_value(row: dict[str, str]) -> object:
    if "footprint_visible" in row:
        return row.get("footprint_visible")
    return row.get("causal_effect_visible")


def score_prediction_rows(
    rows: Iterable[dict[str, str]],
    *,
    require_video_quality: bool = True,
) -> list[dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    for row in rows:
        role = variant_role(row)
        if role not in EXPECTED_STATES:
            continue
        expected_target, expected_footprint = EXPECTED_STATES[role]
        observed_target = normalize_bool(row.get("target_visible"))
        observed_footprint = normalize_bool(observed_footprint_value(row))
        target_match = observed_target is expected_target
        footprint_match = observed_footprint is expected_footprint
        quality_value = normalize_bool(row.get("video_quality"))
        quality_ok = True
        if require_video_quality and quality_value is False:
            quality_ok = False
        variant_pass = target_match and footprint_match and quality_ok
        scored.append(
            {
                "item_key": item_key(row),
                "item_id": str(row.get("item_id", "")),
                "item_index": str(row.get("item_index", "")),
                "pair_id": str(row.get("pair_id", "")),
                "mechanism_type": str(row.get("mechanism_type", "")),
                "baseline": str(row.get("baseline", "")),
                "variant_role": role,
                "target_concept": str(row.get("target_concept", "")),
                "expected_effect": str(row.get("expected_effect", "")),
                "expected_target_visible": expected_target,
                "expected_footprint_visible": expected_footprint,
                "observed_target_visible": observed_target,
                "observed_footprint_visible": observed_footprint,
                "target_match": target_match,
                "footprint_match": footprint_match,
                "quality_ok": quality_ok,
                "variant_pass": variant_pass,
                "video_quality": str(row.get("video_quality", "")),
                "confidence": str(row.get("confidence", "")),
                "final_label": str(row.get("final_label", "")),
                "notes": str(row.get("notes", "")),
            }
        )
    return scored


def observed_state(row: dict[str, Any] | None) -> str:
    if row is None:
        return ""
    target = bool_to_text(row.get("observed_target_visible"))
    footprint = bool_to_text(row.get("observed_footprint_visible"))
    return f"target={target};footprint={footprint}"


def aggregate_item_scores(scored_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for row in scored_rows:
        groups.setdefault(str(row["item_key"]), []).append(row)

    item_scores: list[dict[str, Any]] = []
    for key, rows in groups.items():
        by_variant = {str(row["variant_role"]): row for row in rows}
        missing = [variant for variant in VARIANTS if variant not in by_variant]
        pass_flags = {
            variant: bool(by_variant.get(variant, {}).get("variant_pass", False))
            for variant in VARIANTS
        }
        original_valid = pass_flags["original"] and "original" not in missing
        counterfactual_variants = ["remove_target", "footprint_only", "target_only"]
        counterfactual_pass = all(
            pass_flags[variant] and variant not in missing for variant in counterfactual_variants
        )
        c0_grid_pass = original_valid and counterfactual_pass and not missing
        failed_counterfactuals = [
            variant
            for variant in counterfactual_variants
            if variant in by_variant and not pass_flags[variant]
        ]
        if missing:
            failure_mode = "missing_variants"
        elif not original_valid:
            failure_mode = "invalid_original"
        elif failed_counterfactuals:
            failure_mode = "failed:" + ",".join(failed_counterfactuals)
        else:
            failure_mode = "pass"

        first = rows[0]
        item_scores.append(
            {
                "item_key": key,
                "item_id": first.get("item_id", ""),
                "item_index": first.get("item_index", ""),
                "pair_id": first.get("pair_id", ""),
                "mechanism_type": first.get("mechanism_type", ""),
                "original_valid": original_valid,
                "counterfactual_pass": counterfactual_pass,
                "c0_grid_pass": c0_grid_pass,
                "missing_variants": ",".join(missing),
                "failed_variants": ",".join(failed_counterfactuals),
                "failure_mode": failure_mode,
                "original_pass": pass_flags["original"],
                "remove_target_pass": pass_flags["remove_target"],
                "footprint_only_pass": pass_flags["footprint_only"],
                "target_only_pass": pass_flags["target_only"],
                "original_observed": observed_state(by_variant.get("original")),
                "remove_target_observed": observed_state(by_variant.get("remove_target")),
                "footprint_only_observed": observed_state(by_variant.get("footprint_only")),
                "target_only_observed": observed_state(by_variant.get("target_only")),
            }
        )
    return sorted(item_scores, key=item_sort_key)


def item_sort_key(row: dict[str, Any]) -> tuple[int, str]:
    index = str(row.get("item_index", "")).strip()
    if index.isdigit():
        return int(index), str(row.get("item_key", ""))
    return 10**9, str(row.get("item_key", ""))


def summarize_scores(
    variant_scores: Sequence[dict[str, Any]],
    item_scores: Sequence[dict[str, Any]],
    *,
    require_video_quality: bool,
) -> dict[str, Any]:
    variant_totals = Counter(str(row["variant_role"]) for row in variant_scores)
    variant_passes = Counter(
        str(row["variant_role"]) for row in variant_scores if row["variant_pass"]
    )
    failure_modes = Counter(str(row["failure_mode"]) for row in item_scores)
    return {
        "total_variant_rows": len(variant_scores),
        "total_items": len(item_scores),
        "original_valid_items": sum(bool(row["original_valid"]) for row in item_scores),
        "counterfactual_pass_items": sum(
            bool(row["counterfactual_pass"]) for row in item_scores
        ),
        "c0_grid_pass_items": sum(bool(row["c0_grid_pass"]) for row in item_scores),
        "variant_total_counts": {variant: variant_totals.get(variant, 0) for variant in VARIANTS},
        "variant_pass_counts": {variant: variant_passes.get(variant, 0) for variant in VARIANTS},
        "failure_mode_counts": dict(sorted(failure_modes.items())),
        "require_video_quality": require_video_quality,
    }


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def csv_row(row: dict[str, Any], fields: Sequence[str]) -> dict[str, str]:
    return {field: bool_to_text(row.get(field, "")) for field in fields}


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow(csv_row(row, fields))


def write_outputs(
    output_dir: Path,
    variant_scores: Sequence[dict[str, Any]],
    item_scores: Sequence[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "c0_variant_scores.csv", variant_scores, VARIANT_SCORE_FIELDS)
    write_csv(output_dir / "c0_item_scores.csv", item_scores, ITEM_SCORE_FIELDS)
    (output_dir / "c0_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--predictions-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--allow-low-quality",
        action="store_true",
        help="Do not fail a variant solely because video_quality is no.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.predictions_csv.exists():
        parser.exit(2, f"{args.predictions_csv}: file not found\n")
    require_quality = not args.allow_low_quality
    predictions = read_csv(args.predictions_csv)
    variant_scores = score_prediction_rows(predictions, require_video_quality=require_quality)
    item_scores = aggregate_item_scores(variant_scores)
    summary = summarize_scores(
        variant_scores,
        item_scores,
        require_video_quality=require_quality,
    )
    write_outputs(args.output_dir, variant_scores, item_scores, summary)
    print(
        "Wrote "
        f"{len(variant_scores)} variant rows and {len(item_scores)} item rows to "
        f"{args.output_dir}"
    )
    print(
        "C0 grid pass: "
        f"{summary['c0_grid_pass_items']}/{summary['total_items']} items; "
        f"original valid: {summary['original_valid_items']}/{summary['total_items']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
