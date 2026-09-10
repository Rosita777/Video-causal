from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

import pytest

from scripts import export_causal_role_erasure_7mechanism_paper_tables_v1 as exporter
from scripts import compute_causal_role_erasure_7mechanism_formal_metrics_v1 as legacy_metrics


REPO_ROOT = Path(__file__).resolve().parents[1]
FORMAL_CASES = REPO_ROOT / "data" / "causal_role_erasure_7mechanism_main_v2" / "formal_cases.csv"
IDENTIFICATION = REPO_ROOT / "data" / "causal_role_erasure_7mechanism_main_v2" / "identification_subset.csv"
AMENDMENT = REPO_ROOT / "data" / "causal_role_erasure_7mechanism_main_v2" / "evaluation_provenance_amendment_v2.json"
REGISTRY = REPO_ROOT.parent / "causal7m_formal_review_launch_v5" / "review_package" / "public" / "evaluation_code_registry.json"


def json_bytes(value):
    return exporter.canonical_json_bytes(value)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json_bytes(value))


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(json_bytes(row) for row in rows))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows, fields) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def ref(root_id: str, root: Path, path: Path) -> dict[str, object]:
    return {
        "root_id": root_id,
        "path": path.relative_to(root).as_posix(),
        "sha256": exporter.sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def with_self_digest(value: dict[str, object]) -> dict[str, object]:
    value = dict(value)
    value["manifest_sha256"] = exporter.self_digest(value)
    return value


def base_scores(stream: str, kind: str) -> dict[str, object]:
    if kind == "specificity":
        return {
            "usable": 1,
            "receiver_preservation_norm": 1.0,
            "video_quality_norm": 1.0,
            "original_eligible": "",
            "ces": "",
            "su": 1.0,
            "source_absent": "",
            "footprint_absent": "",
            "strict_success": "",
            "source_visibility": "",
            "footprint_visibility": "",
            "receiver_preservation": 2,
            "video_quality": 2,
            "protected_object_visibility": 2,
            "noncausal_role_adherence": 2,
        }
    visibility = {
        "wan_original": (2, 2),
        "matched_control": (2, 1),
        "V4": (0, 0),
        "cogvideox_original": (2, 2),
        "negative_prompt": (1, 1),
        "videoeraser_official": (1, 0),
        "t2vunlearning_adapted": (0, 1),
        "safree_cogvideox": (1, 1),
        "generic_paraphrase": (1, 1),
        "bystander_token": (1, 1),
    }[stream]
    source, footprint = visibility
    ces = ((2 - source) + (2 - footprint)) / 4
    return {
        "usable": 1,
        "receiver_preservation_norm": 1.0,
        "video_quality_norm": 1.0,
        "original_eligible": 1,
        "ces": ces,
        "su": "",
        "source_absent": int(source == 0),
        "footprint_absent": int(footprint == 0),
        "strict_success": int(source == 0 and footprint == 0),
        "source_visibility": source,
        "footprint_visibility": footprint,
        "receiver_preservation": 2,
        "video_quality": 2,
        "protected_object_visibility": "",
        "noncausal_role_adherence": "",
    }


def build_tables(cases: list[dict[str, str]], identification_ids: set[str], *, ne_external: bool):
    per_case = []
    counter = 0
    for partition, streams, selected in (
        ("main", exporter.STREAMS, cases),
        ("identification", exporter.IDENTIFICATION_CONTROLS, [row for row in cases if row["case_id"] in identification_ids]),
    ):
        for stream in streams:
            for case in selected:
                scores = base_scores(stream, case["case_kind"])
                per_case.append(
                    {
                        "anonymous_review_id": f"synthetic_{counter:04d}",
                        "evaluation_partition": partition,
                        "stream": stream,
                        "backbone_family": "cogvideox" if stream in exporter.COG_STREAMS else "wan",
                        "case_id": case["case_id"],
                        "mechanism": case["mechanism"],
                        "case_kind": case["case_kind"],
                        "seed": int(case["seed"]),
                        "m6_pair_id": case["m6_pair_id"],
                        "footprint_lexicalization": case["footprint_lexicalization"],
                        "source_membership": case["source_membership"],
                        "generalization_group": case["generalization_group"],
                        **scores,
                    }
                )
                counter += 1
    assert len(per_case) == 2448

    mechanism = []
    for stream in exporter.STREAMS:
        for mechanism_name in exporter.MECHANISMS:
            block = [row for row in per_case if row["evaluation_partition"] == "main" and row["stream"] == stream and row["mechanism"] == mechanism_name]
            causal = [row for row in block if row["case_kind"] == "causal"]
            specificity = [row for row in block if row["case_kind"] == "specificity"]
            mechanism.append(
                {
                    "stream": stream,
                    "mechanism": mechanism_name,
                    "causal_n": 24,
                    "specificity_n": 18,
                    "ces": sum(row["ces"] for row in causal) / 24,
                    "specificity_utility": 1.0,
                    "original_capability_coverage": 1.0,
                    "causal_usable_fraction": 1.0,
                    "specificity_usable_fraction": 1.0,
                    "causal_receiver_preservation": 1.0,
                    "causal_video_quality": 1.0,
                    "specificity_receiver_preservation": 1.0,
                    "specificity_video_quality": 1.0,
                    "source_absent_rate": sum(row["source_absent"] for row in causal) / 24,
                    "footprint_absent_rate": sum(row["footprint_absent"] for row in causal) / 24,
                    "strict_success_rate": sum(row["strict_success"] for row in causal) / 24,
                    "eligible_source_clear_to_partial_rate": sum(row["source_visibility"] == 1 for row in causal) / 24,
                    "eligible_source_clear_to_absent_rate": sum(row["source_visibility"] == 0 for row in causal) / 24,
                }
            )
    macro = []
    metric_fields = exporter.TABLE_SCHEMAS["macro_metrics"][2:]
    for stream in exporter.STREAMS:
        block = [row for row in mechanism if row["stream"] == stream]
        macro.append({"stream": stream, "mechanism_count": 7, **{field: sum(row[field] for row in block) / 7 for field in metric_fields}})

    headline = []
    for contrast in exporter.HEADLINE_CONTRASTS:
        is_ne = ne_external and contrast == "negative_prompt"
        point = 0.75 if contrast == "matched_control" else 0.50
        headline.append(
            {
                "contrast": contrast,
                "estimable": int(not is_ne),
                "point_estimate": "" if is_ne else point,
                "ci95_lower": "" if is_ne else point - 0.05,
                "ci95_upper": "" if is_ne else point + 0.05,
                "one_sided_p": 1.0 if is_ne else 0.0001,
                "bootstrap_replicates": 10_000,
                "bootstrap_seed": exporter.HEADLINE_SEED,
                "holm_adjusted_p": 1.0 if is_ne else 0.0005,
                "registered_win": int(not is_ne),
            }
        )
    headline_mechanism = []
    for contrast in exporter.HEADLINE_CONTRASTS:
        for mechanism_name in exporter.MECHANISMS:
            is_ne = ne_external and contrast == "negative_prompt" and mechanism_name == "water_impact"
            headline_mechanism.append(
                {
                    "contrast": contrast,
                    "mechanism": mechanism_name,
                    "n": 24 if contrast == "matched_control" else (11 if is_ne else 24),
                    "estimable": int(not is_ne),
                    "point_estimate": "" if is_ne else (0.75 if contrast == "matched_control" else 0.50),
                }
            )

    noninferiority = []
    for index, metric in enumerate(exporter.NI_METRICS):
        margin = -0.05 if metric in ("usable_fraction", "specificity_usable_fraction") else -0.10
        noninferiority.append(
            {
                "metric": metric,
                "role": "primary" if index < 4 else "secondary",
                "margin": margin,
                "point_estimate": 0.0,
                "ci95_lower": -0.01,
                "ci95_upper": 0.01,
                "one_sided_p": 1.0,
                "bootstrap_replicates": 10_000,
                "bootstrap_seed": exporter.HEADLINE_SEED,
                "noninferior": 1,
            }
        )
    identification = []
    for control in exporter.IDENTIFICATION_CONTROLS:
        identification.append(
            {
                "control": control,
                "point_estimate": 0.50,
                "ci95_lower": 0.45,
                "ci95_upper": 0.55,
                "one_sided_p": 0.0001,
                "bootstrap_replicates": 10_000,
                "bootstrap_seed": exporter.IDENTIFICATION_SEED,
                "implicit_two_mechanism_point_estimate": 0.50,
                "implicit_water_point_estimate": 0.50,
                "implicit_fracture_point_estimate": 0.50,
                "specificity_margin": -0.10,
                "specificity_point_estimate": 0.0,
                "specificity_ci95_lower": -0.01,
                "specificity_ci95_upper": 0.01,
                "specificity_noninferior": 1,
                "holm_adjusted_p": 0.0002,
                "novelty_condition_pass": 1,
            }
        )
    role = []
    for subset, n in (("implicit_footprint", 12), ("held_out_source", 16)):
        role.append(
            {
                "subset": subset,
                "per_mechanism_n": {mechanism_name: n for mechanism_name in exporter.MECHANISMS},
                "point_estimate": 0.75,
                "ci95_lower": 0.70,
                "ci95_upper": 0.80,
                "one_sided_p": 0.0001,
                "bootstrap_replicates": 10_000,
                "bootstrap_seed": exporter.HEADLINE_SEED,
                "registered_positive": 1,
            }
        )

    by_case = {(row["evaluation_partition"], row["stream"], row["case_id"]): row for row in per_case}
    pair_meta: dict[str, dict[str, str]] = {}
    for case in cases:
        if case["m6_pair_id"]:
            pair_meta.setdefault(case["m6_pair_id"], {})[case["case_kind"]] = case["case_id"]
    m6 = []
    for stream in exporter.STREAMS:
        for pair_id in sorted(pair_meta):
            causal = by_case[("main", stream, pair_meta[pair_id]["causal"])]
            spec = by_case[("main", stream, pair_meta[pair_id]["specificity"])]
            m6.append(
                {
                    "stream": stream,
                    "mechanism": causal["mechanism"],
                    "m6_pair_id": pair_id,
                    "causal_case_id": causal["case_id"],
                    "specificity_case_id": spec["case_id"],
                    "causal_ces": causal["ces"],
                    "specificity_utility": spec["su"],
                    "pair_mean": (causal["ces"] + spec["su"]) / 2,
                }
            )
    secondary = [{field: row[field] for field in exporter.TABLE_SCHEMAS["secondary_metrics"]} for row in mechanism]
    return {
        "per_case_metrics": per_case,
        "mechanism_metrics": mechanism,
        "macro_metrics": macro,
        "headline_contrasts": headline,
        "headline_mechanism_contrasts": headline_mechanism,
        "noninferiority": noninferiority,
        "identification": identification,
        "role_conditioned_evidence": role,
        "m6_pairs": m6,
        "secondary_metrics": secondary,
    }


def build_fixture(tmp_path: Path, *, ne_external: bool = False) -> dict[str, Path]:
    project_root = tmp_path / "project"
    project_root.mkdir(parents=True)
    registry_value = json.loads(REGISTRY.read_text(encoding="utf-8"))
    copied = {
        "scripts/export_causal_role_erasure_7mechanism_paper_tables_v1.py",
        "scripts/compute_causal_role_erasure_7mechanism_formal_metrics_v2.py",
        "scripts/compute_causal_role_erasure_7mechanism_formal_metrics_v1.py",
        "scripts/causal_role_erasure_7mechanism_evaluation_code_registry_v1.py",
        "data/causal_role_erasure_7mechanism_main_v2/formal_cases.csv",
        "data/causal_role_erasure_7mechanism_main_v2/identification_subset.csv",
        "data/causal_role_erasure_7mechanism_main_v2/evaluation_provenance_amendment_v2.json",
        *(row["path"] for row in registry_value["files"]),
    }
    for relative in copied:
        source = REPO_ROOT / relative
        target = project_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)

    review_root = tmp_path / "review"
    eligibility_root = tmp_path / "eligibility"
    metrics_root = tmp_path / "metrics"
    pre_metric_root = tmp_path / "pre_metric"
    for root in (review_root, eligibility_root, metrics_root, pre_metric_root):
        root.mkdir(parents=True)

    formal_cases = project_root / "data/causal_role_erasure_7mechanism_main_v2/formal_cases.csv"
    amendment_path = project_root / "data/causal_role_erasure_7mechanism_main_v2/evaluation_provenance_amendment_v2.json"
    cases = read_csv(formal_cases)
    identification_ids = {row["case_id"] for row in read_csv(IDENTIFICATION)}
    tables = build_tables(cases, identification_ids, ne_external=ne_external)

    eligibility = []
    shared = []
    causal_cases = [row for row in cases if row["case_kind"] == "causal"]
    for index, case in enumerate(causal_cases):
        for backbone, stream in (("wan", "wan_original"), ("cogvideox", "cogvideox_original")):
            per_case_row = next(
                row for row in tables["per_case_metrics"]
                if row["evaluation_partition"] == "main"
                and row["stream"] == stream
                and row["case_id"] == case["case_id"]
            )
            eligibility.append(
                {
                    "anonymous_review_id": per_case_row["anonymous_review_id"],
                    "backbone_family": backbone,
                    "case_id": case["case_id"],
                    "mechanism": case["mechanism"],
                    "eligible": 1,
                    "source_visibility": 2,
                    "footprint_visibility": 2,
                    "receiver_preservation": 2,
                    "video_quality": 2,
                }
            )
        shared.append(
            {
                "case_id": case["case_id"],
                "mechanism": case["mechanism"],
                "wan_original_eligible": 1,
                "cogvideox_original_eligible": 1,
                "shared_eligible": 1,
            }
        )
    per_case = tables["per_case_metrics"]
    mechanism, macro = legacy_metrics.mechanism_and_macro_tables(per_case)
    headline, headline_mechanism, all_baselines = legacy_metrics.headline_contrasts(
        per_case, shared, replicates=exporter.BOOTSTRAP_REPLICATES
    )
    noninferiority, retention = legacy_metrics.noninferiority_table(
        per_case, replicates=exporter.BOOTSTRAP_REPLICATES
    )
    identification, identification_novelty = legacy_metrics.identification_table(
        per_case, replicates=exporter.BOOTSTRAP_REPLICATES
    )
    role, role_positive = legacy_metrics.main_role_conditioned_evidence(
        per_case, replicates=exporter.BOOTSTRAP_REPLICATES
    )
    m6 = legacy_metrics.m6_table(per_case, cases)
    tables.update(
        {
            "mechanism_metrics": mechanism,
            "macro_metrics": macro,
            "headline_contrasts": headline,
            "headline_mechanism_contrasts": headline_mechanism,
            "noninferiority": noninferiority,
            "identification": identification,
            "role_conditioned_evidence": role,
            "m6_pairs": m6,
            "secondary_metrics": [
                {field: row[field] for field in exporter.TABLE_SCHEMAS["secondary_metrics"]}
                for row in mechanism
            ],
        }
    )

    table_refs = {}
    for name in exporter.TABLE_NAMES:
        basename = exporter.TABLE_BASENAMES[name]
        json_path = metrics_root / f"{basename}.json"
        csv_path = metrics_root / f"{basename}.csv"
        write_json(json_path, tables[name])
        write_csv(csv_path, tables[name], exporter.CSV_SCHEMAS[name])
        table_refs[name] = {
            "csv": ref("metrics_root", metrics_root, csv_path),
            "json": ref("metrics_root", metrics_root, json_path),
        }
    eligibility_path = eligibility_root / "original_eligibility.csv"
    shared_path = eligibility_root / "shared_capability_subsets.csv"
    write_csv(eligibility_path, eligibility, exporter.ELIGIBILITY_SCHEMA)
    write_csv(shared_path, shared, exporter.SHARED_SCHEMA)
    write_json(eligibility_root / "eligibility_manifest.json", {"schema_version": 2, "protocol": exporter.CANONICAL_PROTOCOL, "status": exporter.ELIGIBILITY_STATUS})
    public_root = review_root / "review_package" / "public"
    public_root.mkdir(parents=True)
    write_json(public_root / "key_commitments.json", {"synthetic": True})
    shutil.copyfile(REGISTRY, public_root / "evaluation_code_registry.json")

    results = {
        "schema_version": 1,
        "protocol": exporter.LEGACY_RESULTS_PROTOCOL,
        "status": exporter.LEGACY_RESULTS_STATUS,
        "evaluation_code_registry_sha256": exporter.REGISTRY_SHA256,
        "aggregation": "case_then_mechanism_mean_then_equal_weight_7_mechanism_macro",
        "metric_definitions": {
            "original_eligibility": "source_visibility==2 and footprint_visibility>=1 and receiver_preservation>=1 and video_quality>=1",
            "usable": "receiver_preservation>=1 and video_quality>=1",
            "CES": "E_b * usable * ((2-source_visibility)+(2-footprint_visibility))/4",
            "SU": "usable * (protected_object_visibility+noncausal_role_adherence)/4",
            "external_headline": "(V4-WanOriginal)-(baseline-CogVideoXOriginal) on pre-frozen shared-capability cases",
            "fixed_denominator_policy": "Original-ineligible causal cases remain present and contribute zero CES",
        },
        "bootstrap": {"replicates": 10_000, "headline_pcg64_seed": exporter.HEADLINE_SEED, "identification_pcg64_seed": exporter.IDENTIFICATION_SEED, "interval": "unadjusted_percentile_2.5_97.5"},
        "holm_families": {"headline": 5, "identification": 2},
        "claims": {
            "outperforms_all_registered_baselines": bool(all_baselines),
            "retaining_allowed_by_all_primary_noninferiority_margins": bool(retention),
            "role_conditioned_novelty_supported": bool(
                identification_novelty
                and role_positive
                and retention
                and next(row["registered_win"] == 1 for row in headline if row["contrast"] == "matched_control")
            ),
        },
        "headline_contrasts": tables["headline_contrasts"],
        "noninferiority": tables["noninferiority"],
        "identification_contrasts": tables["identification"],
        "role_conditioned_main_evidence": tables["role_conditioned_evidence"],
        "tables": {
            name: {
                kind: {key: value for key, value in table_refs[name][kind].items() if key != "root_id"}
                for kind in ("csv", "json")
            }
            for name in exporter.TABLE_NAMES
        },
    }
    results_path = metrics_root / "results.json"
    write_json(results_path, results)

    exporter_path = project_root / "scripts" / "export_causal_role_erasure_7mechanism_paper_tables_v1.py"
    pre_metric = with_self_digest(
        {
            "schema_version": 1,
            "protocol": exporter.PRE_METRIC_PROTOCOL,
            "protocol_version": "causal_role_erasure_7m_single_seed_v2",
            "status": exporter.PRE_METRIC_STATUS,
            "deviation": {"global_key_opened": True},
            "v1_authority": {"bound": True},
            "source_bindings": {"synthetic": True},
            "components": [
                {
                    "path": "scripts/export_causal_role_erasure_7mechanism_paper_tables_v1.py",
                    "sha256": exporter.sha256_file(exporter_path),
                    "size_bytes": exporter_path.stat().st_size,
                    "role": "final_table_exporter",
                }
            ],
            "reproducibility_bindings": {"synthetic": True},
            "pending_outputs": [{"name": "human_reviews", "status": "not_materialized_at_pre_metric_freeze"}],
            "implementation_commit": {"commit": "1" * 40, "tree": "2" * 40, "worktree_clean": True},
        }
    )
    pre_metric_path = pre_metric_root / "pre_metric_code_freeze_v2.json"
    write_json(pre_metric_path, pre_metric)
    inputs = {
        "canonical_manifest": {"root_id": "canonical_root", "path": "canonical_manifest.json", "sha256": "5" * 64, "size_bytes": 1},
        "canonical_scores": {"root_id": "canonical_root", "path": "canonical_scores.jsonl", "sha256": "6" * 64, "size_bytes": 1},
        "eligibility_manifest": ref("eligibility_root", eligibility_root, eligibility_root / "eligibility_manifest.json"),
        "original_eligibility": ref("eligibility_root", eligibility_root, eligibility_path),
        "shared_capability_subsets": ref("eligibility_root", eligibility_root, shared_path),
        "full_method_key": {"root_id": "review_root", "path": "review_package/private/full_key.jsonl", "sha256": "7" * 64, "size_bytes": 1},
        "key_commitments": ref("review_root", review_root, public_root / "key_commitments.json"),
        "formal_cases": ref("repo_root", project_root, formal_cases),
    }
    wrapper = project_root / "scripts" / "compute_causal_role_erasure_7mechanism_formal_metrics_v2.py"
    score = with_self_digest(
        {
            "schema_version": 2,
            "protocol": exporter.SCORE_PROTOCOL,
            "status": exporter.SCORE_STATUS,
            "provenance_amendment": ref("amendment_root", amendment_path.parent, amendment_path),
            "pre_metric_code_freeze": ref("pre_metric_root", pre_metric_root, pre_metric_path),
            "evaluation_code_registry": {**ref("review_root", review_root, public_root / "evaluation_code_registry.json"), "registry_sha256": exporter.REGISTRY_SHA256},
            "wrapper_implementation": ref("repo_root", project_root, wrapper),
            "registered_v1_derivation": {
                "implementation": ref(
                    "repo_root",
                    project_root,
                    project_root / "scripts" / "compute_causal_role_erasure_7mechanism_formal_metrics_v1.py",
                ),
                "entry_point": "compute_formal_metrics",
                "legacy_manifest_sha256": "4" * 64,
                "legacy_manifest_authoritative": False,
                "scientific_payload_byte_identical": True,
                "payload_sha256": {
                    "results": exporter.sha256_file(results_path),
                    **{
                        f"{name}.{suffix}": table_refs[name][suffix]["sha256"]
                        for name in exporter.TABLE_NAMES
                        for suffix in ("csv", "json")
                    },
                },
            },
            "key_access": {
                "global_full_method_key_opened_before_human_canonicalization": True,
                "global_pre_unblinding_claim": False,
                "tier_2_full_method_key_read_by_this_stage": True,
            },
            "inputs": inputs,
            "results": ref("metrics_root", metrics_root, results_path),
            "tables": table_refs,
        }
    )
    score_path = metrics_root / "score_manifest.json"
    write_json(score_path, score)
    return {
        "project_root": project_root,
        "review_root": review_root,
        "eligibility_root": eligibility_root,
        "metrics_root": metrics_root,
        "pre_metric": pre_metric_path,
        "score": score_path,
        "amendment": amendment_path,
        "output": project_root / exporter.OUTPUT_RELATIVE,
    }


def run_export(fixture: dict[str, object], *, output: Path | None = None, mutate_during_build=None):
    exporter_copy = fixture["project_root"] / "scripts/export_causal_role_erasure_7mechanism_paper_tables_v1.py"
    authority_validator = mock.Mock()
    fixture["authority_validator"] = authority_validator
    with ExitStack() as stack:
        stack.enter_context(mock.patch.object(exporter, "__file__", str(exporter_copy)))
        stack.enter_context(mock.patch.object(exporter, "validate_v2_authority_envelope", authority_validator))
        stack.enter_context(mock.patch.object(exporter.provenance_v2, "load_registered_metrics", return_value=legacy_metrics))
        if mutate_during_build is not None:
            original_build_outputs = exporter.build_outputs

            def build_then_mutate(*args, **kwargs):
                result = original_build_outputs(*args, **kwargs)
                mutate_during_build()
                return result

            stack.enter_context(mock.patch.object(exporter, "build_outputs", build_then_mutate))
        return exporter.export_tables(
            repo_root=fixture["project_root"],
            review_root=fixture["review_root"],
            eligibility_root=fixture["eligibility_root"],
            score_manifest_path=fixture["score"],
            pre_metric_path=fixture["pre_metric"],
            amendment_path=fixture["amendment"],
            output_root=output or fixture["output"],
        )


def reseal_table(fixture: dict[str, object], name: str, rows) -> None:
    metrics_root = fixture["metrics_root"]
    basename = exporter.TABLE_BASENAMES[name]
    json_path = metrics_root / f"{basename}.json"
    csv_path = metrics_root / f"{basename}.csv"
    write_json(json_path, rows)
    write_csv(csv_path, rows, exporter.CSV_SCHEMAS[name])
    result = json.loads((metrics_root / "results.json").read_text())
    for kind, path in (("csv", csv_path), ("json", json_path)):
        result["tables"][name][kind] = {
            "path": path.name,
            "sha256": exporter.sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    write_json(metrics_root / "results.json", result)
    score = json.loads(fixture["score"].read_text())
    for kind, path in (("csv", csv_path), ("json", json_path)):
        score["tables"][name][kind] = ref("metrics_root", metrics_root, path)
        score["registered_v1_derivation"]["payload_sha256"][f"{name}.{kind}"] = exporter.sha256_file(path)
    results_path = metrics_root / "results.json"
    score["results"] = ref("metrics_root", metrics_root, results_path)
    score["registered_v1_derivation"]["payload_sha256"]["results"] = exporter.sha256_file(results_path)
    score["manifest_sha256"] = exporter.self_digest(score)
    write_json(fixture["score"], score)


def test_exports_complete_deterministic_human_canonical_tree(tmp_path: Path):
    fixture = build_fixture(tmp_path / "fixture_a")
    second_fixture = build_fixture(tmp_path / "fixture_b")
    first = fixture["output"]
    second = second_fixture["output"]
    manifest = run_export(fixture)
    run_export(second_fixture)

    expected = {
        "main/tab_main_mechanisms.tex",
        "main/tab_main_macro_guardrails.tex",
        "main/tab_identification_controls.tex",
        "main/tab_generalization.tex",
        *{f"appendix/tab_e{index}_{name}.tex" for index, name in ()},
        "appendix/tab_e1_mechanism_decomposition.tex",
        "appendix/tab_e2_guardrail_breakdown.tex",
        "appendix/tab_e3_capability.tex",
        "appendix/tab_e4_headline_contrasts.tex",
        "appendix/tab_e5_noninferiority.tex",
        "appendix/tab_e6_secondary_outcomes.tex",
        "appendix/tab_e7_m6_pairs.tex",
        "appendix/tab_f1_identification_causal.tex",
        "appendix/tab_f2_identification_specificity.tex",
        "appendix/tab_f3_role_conditioned.tex",
        "figure_data/fig_f1_m6_pairs.csv",
        "figure_data/fig_f2_mechanism_contrasts.csv",
        "implementation_snapshot.py",
    }
    assert set(manifest["artifacts"]) == expected
    assert manifest["canonical_status"] == "HUMAN_CANONICAL"
    assert manifest["format"]["metric_digits"] == 3
    assert manifest["manifest_sha256"] == exporter.self_digest(manifest)
    assert manifest["manifest_location"] == {"root_id": "paper_table_root", "path": "paper_tables_manifest.json"}
    assert fixture["authority_validator"].call_count == 1
    assert not (fixture["review_root"] / "review_package/private/full_key.jsonl").exists()
    assert len(read_csv(first / "figure_data" / "fig_f1_m6_pairs.csv")) == 42
    assert len(read_csv(first / "figure_data" / "fig_f2_mechanism_contrasts.csv")) == 7
    assert all(path.stat().st_mode & 0o222 == 0 for path in first.rglob("*") if path.is_file())
    on_disk_manifest = json.loads((first / "paper_tables_manifest.json").read_text())
    assert on_disk_manifest == manifest
    source_roots = {
        "paper_table_root": first,
        "metrics_root": fixture["metrics_root"],
        "eligibility_root": fixture["eligibility_root"],
        "repo_root": fixture["project_root"],
    }
    for relative, artifact in manifest["artifacts"].items():
        assert artifact["root_id"] == "paper_table_root"
        path = first / artifact["path"]
        assert path.relative_to(first).as_posix() == relative
        assert path.stat().st_size == artifact["size_bytes"]
        assert exporter.sha256_file(path) == artifact["sha256"]
        assert artifact["sources"]
        assert len({source["name"] for source in artifact["sources"]}) == len(artifact["sources"])
        assert all("root_id" in source and "path" in source and "sha256" in source for source in artifact["sources"])
        for source in artifact["sources"]:
            source_path = source_roots[source["root_id"]] / source["path"]
            assert source_path.is_file()
            assert exporter.sha256_file(source_path) == source["sha256"]
    assert {source["name"] for source in manifest["artifacts"]["main/tab_main_mechanisms.tex"]["sources"]} == {
        "mechanism_metrics.csv", "mechanism_metrics.json", "macro_metrics.csv", "macro_metrics.json"
    }
    assert {source["name"] for source in manifest["artifacts"]["main/tab_generalization.tex"]["sources"]} == {
        "role_conditioned_evidence.csv", "role_conditioned_evidence.json", "formal_cases"
    }

    first_files = {path.relative_to(first): path.read_bytes() for path in first.rglob("*") if path.is_file()}
    second_files = {path.relative_to(second): path.read_bytes() for path in second.rglob("*") if path.is_file()}
    assert first_files == second_files
    for path, raw in first_files.items():
        if path.suffix == ".tex":
            text = raw.decode()
            assert "AUTO-GENERATED HUMAN-CANONICAL" in text
            assert "score_manifest.json sha256=" in text
            assert "/Users/" not in text
            assert "Ours (V4)" not in text
            assert not any(sentinel in text for sentinel in exporter.PREVIEW_SENTINELS)
    assert r"\textit{Wan backbone}" in (first / "main/tab_main_mechanisms.tex").read_text()
    assert r"\textit{CogVideoX backbone}" in (first / "main/tab_main_mechanisms.tex").read_text()
    assert "Backbone & Method" in (first / "appendix/tab_e1_mechanism_decomposition.tex").read_text()

    with pytest.raises(exporter.FinalTableError, match="fresh-only"):
        run_export(fixture)


def test_nonestimable_is_ne_and_never_renders_p_one(tmp_path: Path):
    fixture = build_fixture(tmp_path / "fixture")
    tables = {
        name: json.loads((fixture["metrics_root"] / f"{exporter.TABLE_BASENAMES[name]}.json").read_text())
        for name in exporter.TABLE_NAMES
    }
    row = next(item for item in tables["headline_contrasts"] if item["contrast"] == "negative_prompt")
    row.update({"estimable": 0, "point_estimate": "", "ci95_lower": "", "ci95_upper": "", "one_sided_p": 1.0, "holm_adjusted_p": 1.0, "registered_win": 0})
    eligibility = read_csv(fixture["eligibility_root"] / "original_eligibility.csv")
    shared = read_csv(fixture["eligibility_root"] / "shared_capability_subsets.csv")
    formal_cases = read_csv(fixture["project_root"] / "data/causal_role_erasure_7mechanism_main_v2/formal_cases.csv")
    outputs, _ = exporter.build_outputs(
        tables,
        eligibility,
        shared,
        formal_cases,
    )
    main = outputs["main/tab_main_macro_guardrails.tex"].decode()
    appendix = outputs["appendix/tab_e4_headline_contrasts.tex"].decode()
    assert "Negative Prompt & NE & -- & -- & Not estimable" in main
    assert "Negative Prompt & NE & -- & -- & -- & Not estimable" in appendix
    assert "Negative Prompt & NE & -- & 1.000" not in main + appendix

    tampered_cases = [dict(item) for item in formal_cases]
    overlap_case = next(
        item for item in tampered_cases
        if item["case_kind"] == "causal"
        and item["footprint_lexicalization"] == "implicit"
        and item["source_membership"] == "eval_holdout"
    )
    overlap_case["footprint_lexicalization"] = "explicit"
    with pytest.raises(exporter.FinalTableError, match="implicit/held-out overlap"):
        exporter.build_outputs(tables, eligibility, shared, tampered_cases)


def test_rejects_preview_identity_before_opening_metric_values(tmp_path: Path):
    fixture = build_fixture(tmp_path / "fixture")
    score = json.loads(fixture["score"].read_text())
    score["protocol"] = "causal_role_erasure_7m_paper_preview_tables_v1"
    score["manifest_sha256"] = exporter.self_digest(score)
    fixture["score"].write_bytes(json_bytes(score))
    (fixture["metrics_root"] / "macro_metrics.json").unlink()
    with pytest.raises(exporter.FinalTableError, match="protocol changed"):
        run_export(fixture)


def test_rejects_tampered_metric_and_exporter_binding(tmp_path: Path):
    fixture = build_fixture(tmp_path / "fixture")
    macro = fixture["metrics_root"] / "macro_metrics.json"
    macro.write_bytes(macro.read_bytes() + b" ")
    with pytest.raises(exporter.FinalTableError, match="(?:size|SHA) mismatch"):
        run_export(fixture)

    fixture = build_fixture(tmp_path / "fixture_two")
    freeze = json.loads(fixture["pre_metric"].read_text())
    freeze["components"][0]["sha256"] = "0" * 64
    freeze["manifest_sha256"] = exporter.self_digest(freeze)
    fixture["pre_metric"].write_bytes(json_bytes(freeze))
    score = json.loads(fixture["score"].read_text())
    score["pre_metric_code_freeze"] = ref("pre_metric_root", fixture["pre_metric"].parent, fixture["pre_metric"])
    score["manifest_sha256"] = exporter.self_digest(score)
    fixture["score"].write_bytes(json_bytes(score))
    with pytest.raises(exporter.FinalTableError, match="exporter SHA differs"):
        run_export(fixture)


def test_rejects_wrong_output_leaf_and_never_opens_private_inputs(tmp_path: Path):
    fixture = build_fixture(tmp_path / "fixture")
    with pytest.raises(exporter.FinalTableError, match="versioned leaf"):
        run_export(fixture, output=fixture["project_root"] / "paper/tables/final")
    assert not (fixture["review_root"] / "review_package/private/full_key.jsonl").exists()
    run_export(fixture)


def test_public_authority_envelope_has_no_private_root_argument(tmp_path: Path):
    review_root = tmp_path / "review"
    registry = review_root / "review_package/public/evaluation_code_registry.json"
    registry.parent.mkdir(parents=True)
    registry.write_text("{}\n", encoding="utf-8")
    expected_pre_metric = {"public": True}
    execution = mock.Mock(return_value=({"amendment": True}, expected_pre_metric))
    review = mock.Mock(return_value=(registry, {"registry": True}))
    with mock.patch.object(exporter.provenance_v2, "validate_execution_authority", execution), mock.patch.object(
        exporter.provenance_v2, "validate_review_authority", review
    ):
        exporter.validate_v2_authority_envelope(
            repo_root=tmp_path,
            review_root=review_root,
            amendment_path=tmp_path / "amendment.json",
            pre_metric_path=tmp_path / "freeze.json",
            key_commitments_path=review_root / "review_package/public/key_commitments.json",
            expected_pre_metric=expected_pre_metric,
            expected_registry_path=registry,
        )
    assert execution.call_count == review.call_count == 1
    assert set(review.call_args.kwargs) == {"project_root", "amendment", "pre_metric", "key_commitments_path"}


def test_recomputes_all_aggregates_from_manifest_bound_per_case(tmp_path: Path):
    fixture = build_fixture(tmp_path / "fixture")
    path = fixture["metrics_root"] / "per_case_metrics.json"
    rows = json.loads(path.read_text())
    causal = next(row for row in rows if row["case_kind"] == "causal" and row["stream"] == "V4")
    causal["ces"] = 0.75
    reseal_table(fixture, "per_case_metrics", rows)
    with pytest.raises(exporter.FinalTableError, match="complete recomputation"):
        run_export(fixture)


def test_formal_case_conditional_semantic_replicate_schema(tmp_path: Path):
    rows = read_csv(FORMAL_CASES)
    valid = tmp_path / "valid.csv"
    write_csv(valid, rows, exporter.FORMAL_CASES_SCHEMA)
    loaded = exporter.load_formal_cases(valid)
    assert sum(row["semantic_replicate"] == "" for row in loaded) == 126

    bad_specificity = [dict(row) for row in rows]
    next(row for row in bad_specificity if row["case_kind"] == "specificity")["semantic_replicate"] = "0"
    path = tmp_path / "bad_specificity.csv"
    write_csv(path, bad_specificity, exporter.FORMAL_CASES_SCHEMA)
    with pytest.raises(exporter.FinalTableError, match="must be blank for a specificity"):
        exporter.load_formal_cases(path)

    bad_causal = [dict(row) for row in rows]
    next(row for row in bad_causal if row["case_kind"] == "causal")["semantic_replicate"] = ""
    path = tmp_path / "bad_causal.csv"
    write_csv(path, bad_causal, exporter.FORMAL_CASES_SCHEMA)
    with pytest.raises(exporter.FinalTableError, match="not an integer for a causal"):
        exporter.load_formal_cases(path)

    bad_other_integer = [dict(row) for row in rows]
    bad_other_integer[0]["seed_nonce"] = ""
    path = tmp_path / "bad_seed_nonce.csv"
    write_csv(path, bad_other_integer, exporter.FORMAL_CASES_SCHEMA)
    with pytest.raises(exporter.FinalTableError, match="seed_nonce is not an integer"):
        exporter.load_formal_cases(path)

    injected = [dict(row) for row in rows]
    injected[0]["source_membership"] = r"eval_holdout&\input{evil}"
    path = tmp_path / "injected.csv"
    write_csv(path, injected, exporter.FORMAL_CASES_SCHEMA)
    with pytest.raises(exporter.FinalTableError, match="source membership changed"):
        exporter.load_formal_cases(path)


def test_strict_serialization_ranges_tex_escape_and_authority_toctou(tmp_path: Path):
    with pytest.raises(exporter.FinalTableError, match="noncanonical"):
        exporter.csv_matches_json([{"value": "1.0"}], [{"value": 1}], "synthetic")
    with pytest.raises(exporter.FinalTableError, match=r"outside \[0,1\]"):
        exporter.unit(1.01, "bounded")
    assert exporter.tex_escape("a&b_%#${}~^\\") == r"a\&b\_\%\#\$\{\}\textasciitilde{}\textasciicircum{}\textbackslash{}"

    fixture = build_fixture(tmp_path / "fixture")
    macro_json = fixture["metrics_root"] / "macro_metrics.json"
    macro_json.write_bytes(b" " + macro_json.read_bytes())
    with pytest.raises(exporter.FinalTableError, match="not canonical"):
        exporter.load_json_rows(macro_json, "macro_metrics")
    fixture = build_fixture(tmp_path / "fixture_toctou")
    authority = fixture["review_root"] / "review_package/public/key_commitments.json"

    def mutate():
        authority.write_bytes(authority.read_bytes() + b" ")

    with pytest.raises(exporter.FinalTableError, match="changed during export"):
        run_export(fixture, mutate_during_build=mutate)


def test_fixed_precision_p_format_and_no_ni_p_column():
    assert exporter.decimal_text(0.12345) == "0.123"
    assert exporter.decimal_text(0.1235) == "0.124"
    assert exporter.decimal_text(-0.0001) == "0.000"
    assert exporter.p_text(0.0009) == r"$<0.001$"
    assert exporter.p_text(0.001) == "0.001"
    source = Path(exporter.__file__).read_text()
    assert "superiority-to-zero p-values are intentionally not shown" in source


def test_generated_tex_compiles_when_latexmk_is_available(tmp_path: Path):
    latexmk = shutil.which("latexmk")
    pdflatex = shutil.which("pdflatex")
    if latexmk is None or pdflatex is None:
        pytest.skip("LaTeX toolchain is not on PATH")
    fixture = build_fixture(tmp_path / "fixture")
    output = fixture["output"]
    run_export(fixture)
    tex_files = sorted(path.relative_to(output).as_posix() for path in output.rglob("*.tex"))
    harness = tmp_path / "harness.tex"
    harness.write_text(
        "\n".join(
            [
                r"\documentclass[10pt,twocolumn]{article}",
                r"\usepackage{booktabs,graphicx}",
                r"\begin{document}",
                *[rf"\input{{{output.relative_to(tmp_path).as_posix()}/{name}}}\clearpage" for name in tex_files],
                r"\end{document}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    completed = subprocess.run(
        [latexmk, "-pdf", "-interaction=nonstopmode", "-halt-on-error", harness.name],
        cwd=tmp_path,
        env={**os.environ, "SOURCE_DATE_EPOCH": "0"},
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, (completed.stdout + completed.stderr)[-4000:]
    log = (tmp_path / "harness.log").read_text(encoding="utf-8", errors="replace")
    assert "Overfull \\hbox" not in log
