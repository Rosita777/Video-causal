#!/usr/bin/env python3
"""Build or validate the frozen v3c fresh-dev24/sealed-final36 partition."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import water_impact_dynamic_v3c_eval_protocol as protocol


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader.fieldnames), list(reader)


def derive_partition(
    test_rows: list[dict[str, str]], eval12_rows: list[dict[str, str]]
) -> tuple[list[tuple[int, dict[str, str]]], list[tuple[int, dict[str, str]]]]:
    return protocol.derive_expected_partition(test_rows, eval12_rows)


def write_csv(path: Path, source_fields: list[str], selected: list[tuple[int, dict[str, str]]]) -> None:
    fields = ["eval_index", "source_test_index", *source_fields]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for eval_index, (source_index, row) in enumerate(selected):
            writer.writerow(
                {"eval_index": eval_index, "source_test_index": source_index, **row}
            )


def write_prompts(path: Path, selected: list[tuple[int, dict[str, str]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        " | ".join(
            (row["training_prompt"], row["source_object"], row["expected_factual_event"])
        )
        for _, row in selected
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def assignment_sha256(
    eval12_rows: list[dict[str, str]],
    fresh: list[tuple[int, dict[str, str]]],
    final: list[tuple[int, dict[str, str]]],
) -> str:
    assignment = {row["pair_id"]: "exhausted_eval12" for row in eval12_rows}
    assignment.update({row["pair_id"]: "fresh_dev24" for _, row in fresh})
    assignment.update({row["pair_id"]: "sealed_final36" for _, row in final})
    return hashlib.sha256(
        json.dumps(assignment, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def build(project_root: Path) -> dict[str, Any]:
    outputs = tuple(
        protocol.resolve_path(project_root, path)
        for path in (
            protocol.FRESH_DEV_CSV,
            protocol.SEALED_FINAL_CSV,
            protocol.FRESH_DEV_PROMPTS,
            protocol.SEALED_FINAL_PROMPTS,
            protocol.SPLIT_REGISTRY,
        )
    )
    existing = [path for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite frozen split artifacts: {existing}")
    if protocol.resolve_path(project_root, protocol.V3C_RUN).exists():
        raise RuntimeError("refusing to freeze split after the registered v3c run exists")
    test_path = protocol.resolve_path(project_root, protocol.TEST_PAIRS)
    eval12_path = protocol.resolve_path(project_root, protocol.EXHAUSTED_EVAL12)
    if protocol.file_sha256(test_path) != protocol.TEST_PAIRS_SHA256:
        raise ValueError("test_pairs.csv does not match its registered SHA-256")
    if protocol.file_sha256(eval12_path) != protocol.EXHAUSTED_EVAL12_SHA256:
        raise ValueError("eval12.csv does not match its registered SHA-256")
    fields, test_rows = read_csv(test_path)
    _, eval12_rows = read_csv(eval12_path)
    fresh, final = derive_partition(test_rows, eval12_rows)
    write_csv(outputs[0], fields, fresh)
    write_csv(outputs[1], fields, final)
    write_prompts(outputs[2], fresh)
    write_prompts(outputs[3], final)
    registered_files = {
        name: {
            "sha256": protocol.file_sha256(protocol.resolve_path(project_root, name)),
            "row_count": count,
        }
        for name, count in (
            (protocol.FRESH_DEV_CSV, 24),
            (protocol.SEALED_FINAL_CSV, 36),
            (protocol.FRESH_DEV_PROMPTS, 24),
            (protocol.SEALED_FINAL_PROMPTS, 36),
        )
    }
    registry: dict[str, Any] = {
        "protocol": protocol.SPLIT_PROTOCOL,
        "status": "frozen_before_v3c_generation",
        "selection_seed": protocol.SPLIT_SEED,
        "selection_algorithm": (
            "sha256_rank_within_generalization_group_x_prompt_variant; "
            "lowest_4_of_10_to_fresh_dev"
        ),
        "source_test_pairs": protocol.TEST_PAIRS,
        "source_test_pairs_sha256": protocol.TEST_PAIRS_SHA256,
        "excluded_eval12": protocol.EXHAUSTED_EVAL12,
        "excluded_eval12_sha256": protocol.EXHAUSTED_EVAL12_SHA256,
        "fresh_dev_count": 24,
        "sealed_final_count": 36,
        "generalization_groups": list(protocol.GENERALIZATION_GROUPS),
        "prompt_variants": list(protocol.PROMPT_VARIANTS),
        "fresh_dev_per_group": 8,
        "fresh_dev_per_group_variant": 4,
        "assignment_sha256": assignment_sha256(eval12_rows, fresh, final),
        "registered_files": registered_files,
        "blind_seed": protocol.BLIND_SEED,
        "review_semantics": {
            "reviewers": 2,
            "adjudication": "third_blinded_reviewer_for_every_atomic_disagreement",
            "canonical_agreement": "exact_two_reviewer_agreement",
            "canonical_disagreement": "majority_of_three; median_1_for_exact_0_1_2",
            "public_private_packages": "distinct_sibling_directories",
        },
        "gate_spec": protocol.GATE_SPEC,
        "sealed_final_policy": (
            "do_not_generate_inspect_or_score_until_fresh_dev_gate_passes_all_checks"
        ),
        "stage2_policy": (
            "v3c_checkpoint_and_training_artifact_hashes_must_be_registered_before_v3c_generation"
        ),
    }
    outputs[4].parent.mkdir(parents=True, exist_ok=True)
    outputs[4].write_text(
        json.dumps(registry, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return registry


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    if args.validate:
        payload = protocol.validate_split_registration(Path.cwd())
    else:
        payload = build(Path.cwd())
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
