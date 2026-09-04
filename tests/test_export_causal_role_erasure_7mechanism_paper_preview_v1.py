from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts import export_causal_role_erasure_7mechanism_paper_preview_v1 as preview


LABELS = {
    "analysis_status": "post_unblinding_exploratory_vlm_ab_mean_not_human_calibrated",
    "canonical_status": preview.CANONICAL_STATUS,
    "scientific_label": preview.SCIENTIFIC_LABEL,
    "use_restriction": preview.SOURCE_USE_RESTRICTION,
    "final_authority": preview.FINAL_AUTHORITY,
    "formal_claims_permitted": False,
}


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def labelled(**values: object) -> dict[str, object]:
    return {**LABELS, **values}


def build_fixture(root: Path) -> None:
    root.mkdir()
    base = {stream: 0.05 + index * 0.03 for index, stream in enumerate(preview.STREAMS)}
    macro = []
    mechanism = []
    for stream in preview.STREAMS:
        mechanism_values = []
        for index, name in enumerate(preview.MECHANISMS):
            value = base[stream] + (index - 3) * 0.002
            mechanism_values.append(value)
            mechanism.append(labelled(stream=stream, mechanism=name, ces=value))
        macro.append(
            labelled(
                stream=stream,
                mechanism_count=7,
                ces=sum(mechanism_values) / 7,
                specificity_utility=0.9,
                original_capability_coverage=0.7,
                causal_usable_fraction=0.95,
                specificity_usable_fraction=0.96,
                causal_receiver_preservation=0.91,
                causal_video_quality=0.92,
                source_absent_rate=0.2,
                footprint_absent_rate=0.3,
                strict_success_rate=0.1,
                eligible_source_clear_to_absent_rate=0.25,
            )
        )
    write_csv(root / "macro_metrics.csv", macro)
    write_csv(root / "mechanism_metrics.csv", mechanism)

    headline = []
    headline_mechanism = []
    for contrast_index, contrast in enumerate(preview.HEADLINE_CONTRASTS):
        estimable = int(contrast == "matched_control")
        headline.append(
            labelled(
                contrast=contrast,
                estimable=estimable,
                point_estimate=0.04 if estimable else "",
                ci95_lower=0.01 if estimable else "",
                ci95_upper=0.07 if estimable else "",
                one_sided_p=0.01 if estimable else 1.0,
                holm_adjusted_p=0.05 if estimable else 1.0,
                exploratory_positive_threshold_pattern=1 if estimable else 0,
            )
        )
        for mechanism_index, mechanism_name in enumerate(preview.MECHANISMS):
            mechanism_estimable = int(contrast_index == 0 or mechanism_index == 0)
            headline_mechanism.append(
                labelled(
                    contrast=contrast,
                    mechanism=mechanism_name,
                    n=24 if contrast_index == 0 else 12,
                    estimable=mechanism_estimable,
                    point_estimate=0.02 if mechanism_estimable else "",
                )
            )
    write_csv(root / "headline_contrasts.csv", headline)
    write_csv(root / "headline_mechanism_contrasts.csv", headline_mechanism)

    write_csv(
        root / "noninferiority.csv",
        [
            labelled(
                metric=metric,
                role="primary" if index < 4 else "secondary",
                margin=-0.1,
                point_estimate=0.0,
                ci95_lower=-0.02,
                ci95_upper=0.02,
                one_sided_p=1.0,
                exploratory_noninferiority_threshold_pattern=1,
            )
            for index, metric in enumerate(preview.NONINFERIORITY_METRICS)
        ],
    )
    write_csv(
        root / "identification_contrasts.csv",
        [
            labelled(
                control=control,
                point_estimate=0.1,
                ci95_lower=0.01,
                ci95_upper=0.2,
                one_sided_p=0.01,
                holm_adjusted_p=0.02,
                specificity_margin=-0.1,
                exploratory_novelty_threshold_pattern=1,
            )
            for control in preview.IDENTIFICATION_CONTROLS
        ],
    )
    write_csv(
        root / "role_conditioned_evidence.csv",
        [
            labelled(
                subset=subset,
                per_mechanism_n="{'water_impact': 12}",
                point_estimate=0.05,
                ci95_lower=0.01,
                ci95_upper=0.09,
                one_sided_p=0.01,
                exploratory_positive_point_pattern=1,
            )
            for subset in preview.ROLE_SUBSETS
        ],
    )

    sensitivity = []
    for stream_index, stream in enumerate(preview.STREAMS):
        row = labelled(
            stream=stream,
            sensitivity_only=True,
            replaces_primary=False,
            primary_nonlinear_order="atomic A/B mean before thresholds and aggregation",
            sensitivity_order="per-pass thresholds and aggregation before macro mean",
        )
        for metric_index, metric in enumerate(preview.SENSITIVITY_METRICS):
            primary = 0.2 + stream_index * 0.01 + metric_index * 0.001
            pass_a = primary + 0.02
            pass_b = primary
            pass_mean = (pass_a + pass_b) / 2
            row[f"pass_a_{metric}"] = pass_a
            row[f"pass_b_{metric}"] = pass_b
            row[f"mean_of_pass_macros_{metric}"] = pass_mean
            row[f"primary_atomic_mean_{metric}"] = primary
            row[f"mean_of_pass_macros_minus_primary_{metric}"] = pass_mean - primary
        sensitivity.append(row)
    write_csv(root / "ab_order_sensitivity_macro.csv", sensitivity)

    eligibility = []
    shared = []
    for mechanism_index, mechanism_name in enumerate(preview.MECHANISMS):
        for case_index in range(24):
            case_id = f"case_{mechanism_index:02d}_{case_index:02d}"
            wan = int(case_index < 18)
            cog = int(case_index < 16)
            for backbone, eligible in (("wan", wan), ("cogvideox", cog)):
                eligibility.append(
                    labelled(
                        backbone_family=backbone,
                        case_id=case_id,
                        mechanism=mechanism_name,
                        eligible=eligible,
                    )
                )
            shared.append(
                labelled(
                    case_id=case_id,
                    mechanism=mechanism_name,
                    wan_original_eligible=wan,
                    cogvideox_original_eligible=cog,
                    shared_eligible=wan * cog,
                )
            )
    write_csv(root / "original_mean_rule_eligibility.csv", eligibility)
    write_csv(root / "shared_capability_subsets.csv", shared)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_exports_labelled_preview_and_never_renders_nonestimable_p_as_one(tmp_path: Path):
    metrics = tmp_path / "03_deanonymized_metrics"
    build_fixture(metrics)
    # A conspicuous non-input file demonstrates that discovery is allowlisted,
    # direct-child-only, and does not scan arbitrary artifacts.
    (metrics / "full_key.jsonl").write_text("must not be read\n", encoding="utf-8")
    output = tmp_path / "04_paper_tables"

    manifest = preview.export_paper_preview(metrics_root=metrics, output_root=output)

    assert manifest["scientific_label"] == preview.SCIENTIFIC_LABEL
    assert manifest["use_restriction"] == preview.SOURCE_USE_RESTRICTION
    assert len(read_csv(output / "overall_8_methods.csv")) == 8
    assert len(read_csv(output / "ces_by_mechanism_8x7_plus_macro.csv")) == 8
    assert len(read_csv(output / "original_capability_2x7_shared.csv")) == 3
    sensitivity = read_csv(output / "ab_order_sensitivity_ces.csv")
    assert len(sensitivity) == 8
    assert sensitivity[0]["primary_atomic_mean_ces"] == "0.2030"
    assert sensitivity[0]["pass_wise_mean_macro_ces"] == "0.2130"
    assert sensitivity[0]["pass_wise_mean_minus_primary_ces"] == "+0.0100"
    headline = read_csv(output / "headline_presentation.csv")
    assert len(headline) == 51
    external = [
        row for row in headline
        if row["section"] == "headline_overall"
        and row["test"] == "ours_vs_negative_prompt"
    ][0]
    assert external["estimability"] == "NE"
    assert external["point_estimate"] == "NE"
    assert external["one_sided_p"] == "—"
    assert external["holm_adjusted_p"] == "—"

    overall = {row["stream"]: row for row in read_csv(output / "overall_8_methods.csv")}
    assert overall["V4"]["method"] == "Ours (V4)"
    assert overall["V4"]["delta_ces_vs_backbone_original_descriptive"] == "+0.0600"
    overall_md = (output / "overall_8_methods.md").read_text(encoding="utf-8")
    assert "Causal receiver" in overall_md
    assert "Causal video quality" in overall_md
    capability = {row["row_id"]: row for row in read_csv(output / "original_capability_2x7_shared.csv")}
    assert capability["shared"]["water_impact"] == "16/24 (0.6667)"

    headline_md = (output / "headline_presentation.md").read_text(encoding="utf-8")
    assert "Ours (V4) vs Negative Prompt | macro | — | NE | NE | — | — | —" in headline_md
    readme = (output / "README.md").read_text(encoding="utf-8")
    assert preview.SCIENTIFIC_LABEL in readme
    assert "WRITING_PREVIEW_ONLY" in readme
    assert "Final human-canonical scores" in readme
    assert "does not replace the primary result" in readme
    assert "full_key.jsonl" not in json.dumps(manifest)
    assert manifest["input_boundary"]["policy"] == "TEN_AGGREGATE_CSV_FILES_ONLY_NO_KEY_INPUTS"
    assert len(manifest["input_boundary"]["files"]) == 10
    assert manifest["tables"]["sensitivity_rows"] == 8

    sensitivity_md = (output / "ab_order_sensitivity_ces.md").read_text(encoding="utf-8")
    assert "This is a sensitivity analysis only and does not replace the primary" in sensitivity_md

    snapshot = output / "implementation_snapshot.py"
    assert snapshot.stat().st_mode & 0o222 == 0
    assert preview.sha256_file(snapshot) == manifest["exporter_implementation"]["source"]["sha256"]
    assert manifest["exporter_implementation"]["snapshot"]["sha256"] == manifest["exporter_implementation"]["source"]["sha256"]

    with pytest.raises(preview.PaperPreviewError, match="fresh-only"):
        preview.export_paper_preview(metrics_root=metrics, output_root=output)


def test_final_rehash_detects_input_toctou(tmp_path: Path):
    metrics = tmp_path / "03_deanonymized_metrics"
    build_fixture(metrics)
    paths = preview.discover_inputs(metrics)
    initial = {label: preview.file_ref(path) for label, path in paths.items()}
    with (metrics / "macro_metrics.csv").open("a", encoding="utf-8") as handle:
        handle.write("\n")
    with pytest.raises(preview.PaperPreviewError, match="changed during export"):
        preview.verify_refs_unchanged(paths, initial, label="aggregate input")


def test_requires_boolean_false_formal_claims_annotation(tmp_path: Path):
    metrics = tmp_path / "03_deanonymized_metrics"
    build_fixture(metrics)
    macro_path = metrics / "macro_metrics.csv"
    macro = read_csv(macro_path)
    macro[0]["formal_claims_permitted"] = "0"
    write_csv(macro_path, macro)

    with pytest.raises(
        preview.PaperPreviewError,
        match="formal_claims_permitted must be the CSV boolean False",
    ):
        preview.export_paper_preview(
            metrics_root=metrics,
            output_root=tmp_path / "04_paper_tables",
        )
