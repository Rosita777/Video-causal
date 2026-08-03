#!/usr/bin/env python3
"""Compare positive-only PCA and discriminative causal subspaces."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch


def stack(group: dict[str, object]) -> tuple[torch.Tensor, list[str]]:
    records = group["records"]
    features = torch.stack([record["features"].float() for record in records])
    names = [str(record["scene_id"]) for record in records]
    return features, names


def covariance(features: torch.Tensor) -> torch.Tensor:
    flat = features.flatten(0, 1)
    return flat.T @ flat / len(flat)


def top_basis(matrix: torch.Tensor, rank: int) -> torch.Tensor:
    values, vectors = torch.linalg.eigh(matrix)
    return vectors[:, -rank:]


def projection_scores(features: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    projected = features @ basis
    numerator = projected.square().sum(dim=-1)
    denominator = features.square().sum(dim=-1).clamp_min(1e-8)
    return (numerator / denominator).mean(dim=1)


def anchor_scores(features: torch.Tensor, basis: torch.Tensor, top_tokens: int = 8) -> torch.Tensor:
    projected = features @ basis
    ratios = projected.square().sum(dim=-1) / features.square().sum(dim=-1).clamp_min(1e-8)
    return torch.topk(ratios, k=min(top_tokens, ratios.shape[1]), dim=1).values.mean(dim=1)


def auc(positive: np.ndarray, negative: np.ndarray) -> float:
    comparisons = (positive[:, None] > negative[None, :]).mean()
    ties = (positive[:, None] == negative[None, :]).mean()
    return float(comparisons + 0.5 * ties)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--negative-weight", type=float, default=1.0)
    parser.add_argument("--collision-manifest", type=Path, required=True)
    parser.add_argument(
        "--heldout-receivers",
        default="paper_cup,short_tin,wide_domino,wood_peg",
    )
    parser.add_argument("--generic-train-count", type=int, default=24)
    args = parser.parse_args()

    payload = torch.load(args.features, map_location="cpu", weights_only=False)
    collision, collision_names = stack(payload["groups"]["collision"])
    generic, generic_names = stack(payload["groups"]["generic"])
    waterdrop, waterdrop_names = stack(payload["groups"]["waterdrop"])
    target_object, _ = stack(payload["groups"]["target_object"])
    with args.collision_manifest.open(newline="", encoding="utf-8") as handle:
        receiver_by_scene = {
            row["scene_id"]: row["receiver_id"] for row in csv.DictReader(handle)
        }
    heldout_receivers = {
        value.strip() for value in args.heldout_receivers.split(",") if value.strip()
    }
    train_indices = [
        index
        for index, name in enumerate(collision_names)
        if receiver_by_scene[name] not in heldout_receivers
    ]
    test_indices = [
        index
        for index, name in enumerate(collision_names)
        if receiver_by_scene[name] in heldout_receivers
    ]
    if not train_indices or not test_indices:
        raise ValueError("Receiver split produced an empty train or test partition")
    if not 0 < args.generic_train_count < len(generic):
        raise ValueError("--generic-train-count must leave non-empty train and test sets")
    collision_train = collision[train_indices]
    collision_test = collision[test_indices]
    collision_train_names = [collision_names[index] for index in train_indices]
    collision_test_names = [collision_names[index] for index in test_indices]
    generic_train = generic[: args.generic_train_count]
    generic_test = generic[args.generic_train_count :]
    generic_train_names = generic_names[: args.generic_train_count]
    generic_test_names = generic_names[args.generic_train_count :]

    positive_cov = covariance(collision_train)
    negative_cov = covariance(generic_train)
    rank = min(args.rank, positive_cov.shape[0])
    pca_basis = top_basis(positive_cov, rank)
    trace_scale = positive_cov.trace() / negative_cov.trace().clamp_min(1e-8)
    discriminative_matrix = positive_cov - args.negative_weight * trace_scale * negative_cov
    discriminative_basis = top_basis(discriminative_matrix, rank)
    object_basis = top_basis(covariance(target_object), min(4, positive_cov.shape[0]))

    rows = []
    summary = {
        "rank": rank,
        "negative_weight": args.negative_weight,
        "heldout_receivers": sorted(heldout_receivers),
        "methods": {},
    }
    groups = {
        "collision_train": (collision_train, collision_train_names),
        "collision_test": (collision_test, collision_test_names),
        "generic_train": (generic_train, generic_train_names),
        "generic_test": (generic_test, generic_test_names),
        "waterdrop": (waterdrop, waterdrop_names),
    }
    for method, basis in (("pca", pca_basis), ("discriminative", discriminative_basis)):
        scores = {}
        for group_name, (features, names) in groups.items():
            values = projection_scores(features, basis).numpy()
            scores[group_name] = values
            rows.extend(
                {
                    "method": method,
                    "group": group_name,
                    "scene_id": name,
                    "projection_ratio": f"{value:.8f}",
                }
                for name, value in zip(names, values, strict=True)
            )
        summary["methods"][method] = {
            group: {
                "mean": float(values.mean()),
                "median": float(np.median(values)),
                "std": float(values.std()),
            }
            for group, values in scores.items()
        }
        summary["methods"][method]["auc_collision_test_vs_generic_test"] = auc(
            scores["collision_test"], scores["generic_test"]
        )
        summary["methods"][method]["auc_collision_test_vs_waterdrop"] = auc(
            scores["collision_test"], scores["waterdrop"]
        )

    anchored_scores = {}
    for group_name, (features, names) in groups.items():
        mechanism = projection_scores(features, discriminative_basis)
        anchor = anchor_scores(features, object_basis)
        values = (mechanism * anchor).numpy()
        anchored_scores[group_name] = values
        rows.extend(
            {
                "method": "anchored_discriminative",
                "group": group_name,
                "scene_id": name,
                "projection_ratio": f"{value:.8f}",
            }
            for name, value in zip(names, values, strict=True)
        )
    summary["methods"]["anchored_discriminative"] = {
        group: {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "std": float(values.std()),
        }
        for group, values in anchored_scores.items()
    }
    summary["methods"]["anchored_discriminative"][
        "auc_collision_test_vs_generic_test"
    ] = auc(anchored_scores["collision_test"], anchored_scores["generic_test"])
    summary["methods"]["anchored_discriminative"][
        "auc_collision_test_vs_waterdrop"
    ] = auc(anchored_scores["collision_test"], anchored_scores["waterdrop"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "scores.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    torch.save(
        {
            "pca_basis": pca_basis.to(torch.float16),
            "discriminative_basis": discriminative_basis.to(torch.float16),
            "object_basis": object_basis.to(torch.float16),
            "source": str(args.features),
            "rank": rank,
        },
        args.output_dir / "bases.pt",
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
