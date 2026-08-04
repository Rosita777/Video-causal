#!/usr/bin/env python3
"""Train a factual-feature object anchor and test temporal causal propagation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch


def normalize(features: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(features.float(), dim=-1)


def record_features(record: dict[str, object]) -> torch.Tensor:
    return normalize(record["factual_features"])


def train_detector(
    target_records: list[dict[str, object]],
    generic_records: list[dict[str, object]],
    steps: int,
    seed: int,
) -> torch.nn.Linear:
    positive_parts = []
    negative_parts = []
    for record in target_records:
        features = record_features(record)
        weights = record["mask_weights"].float()
        cutoff = max(0.25, float(torch.quantile(weights, 0.75)))
        positive_parts.append(features[weights >= cutoff])
        negative_parts.append(normalize(record["background_features"]))
    for record in generic_records:
        negative_parts.append(record_features(record))
        negative_parts.append(normalize(record["background_features"]))
    positives = torch.cat(positive_parts)
    negatives = torch.cat(negative_parts)
    torch.manual_seed(seed)
    detector = torch.nn.Linear(positives.shape[1], 1)
    optimizer = torch.optim.AdamW(detector.parameters(), lr=1e-2, weight_decay=1e-2)
    for _ in range(steps):
        positive_logits = detector(positives).flatten()
        negative_logits = detector(negatives).flatten()
        loss = 0.5 * torch.nn.functional.binary_cross_entropy_with_logits(
            positive_logits, torch.ones_like(positive_logits)
        )
        loss += 0.5 * torch.nn.functional.binary_cross_entropy_with_logits(
            negative_logits, torch.zeros_like(negative_logits)
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return detector.eval()


@torch.no_grad()
def probabilities(record: dict[str, object], detector: torch.nn.Linear) -> torch.Tensor:
    return torch.sigmoid(detector(record_features(record)).flatten())


@torch.no_grad()
def propagate(
    record: dict[str, object], detector: torch.nn.Linear, radius: float, decay: float
) -> tuple[torch.Tensor, torch.Tensor]:
    features = record_features(record)
    positions = record["positions"].float()
    direct = probabilities(record, detector)
    propagated = direct.clone()
    frames = sorted({int(value) for value in positions[:, 0].tolist()})
    for previous_frame, current_frame in zip(frames, frames[1:]):
        previous_indices = torch.where(positions[:, 0] == previous_frame)[0]
        current_indices = torch.where(positions[:, 0] == current_frame)[0]
        affinity = torch.relu(features[previous_indices] @ features[current_indices].T)
        previous_xy = positions[previous_indices, 1:]
        current_xy = positions[current_indices, 1:]
        distance2 = (previous_xy[:, None, :] - current_xy[None, :, :]).square().sum(dim=-1)
        spatial = torch.exp(-distance2 / (2.0 * radius * radius))
        transition = (propagated[previous_indices, None] * affinity * spatial).amax(dim=0)
        propagated[current_indices] = torch.maximum(
            propagated[current_indices], decay * transition
        )
    return direct, propagated


def auc(positive: np.ndarray, negative: np.ndarray) -> float:
    comparisons = (positive[:, None] > negative[None, :]).mean()
    ties = (positive[:, None] == negative[None, :]).mean()
    return float(comparisons + 0.5 * ties)


def sample_score(values: torch.Tensor, top_tokens: int = 8) -> float:
    return float(torch.topk(values, k=min(top_tokens, len(values))).values.mean())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--generic-motion-scores",
        type=Path,
        default=Path("data/generic_preservation32_motion_scores.csv"),
    )
    parser.add_argument("--generic-train-count", type=int, default=16)
    parser.add_argument("--generic-eval-count", type=int, default=8)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--radius", type=float, default=6.0)
    parser.add_argument("--decay", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260803)
    args = parser.parse_args()

    payload = torch.load(args.features, map_location="cpu", weights_only=False)
    groups = payload["groups"]
    with args.generic_motion_scores.open(newline="", encoding="utf-8") as handle:
        ranked_motion = list(csv.DictReader(handle))
    ranked_paths = [str(Path(row["video_path"]).resolve()) for row in ranked_motion]
    generic_by_video = {
        str(Path(record["video_path"]).resolve()): record
        for record in groups["generic"]["records"]
    }
    ranked_generic = [generic_by_video[path] for path in ranked_paths if path in generic_by_video]
    required_generic = args.generic_train_count + args.generic_eval_count
    if len(ranked_generic) < required_generic:
        raise ValueError(
            f"Need {required_generic} ranked generic records, found {len(ranked_generic)}"
        )
    generic_train = ranked_generic[: args.generic_train_count]
    groups["generic"]["records"] = ranked_generic[
        args.generic_train_count : required_generic
    ]
    detector = train_detector(
        groups["target_object"]["records"][:4],
        generic_train,
        args.steps,
        args.seed,
    )

    rows = []
    group_scores: dict[str, dict[str, list[float]]] = {}
    evaluation_groups = [
        name for name in ("collision", "generic", "waterdrop", "other_ball", "negation")
        if name in groups
    ]
    for group_name in evaluation_groups:
        metrics = {name: [] for name in ("direct", "propagated", "direct_coverage", "propagated_coverage")}
        for record in groups[group_name]["records"]:
            direct, propagated = propagate(record, detector, args.radius, args.decay)
            weights = record["mask_weights"].float()
            positions = record["positions"]
            causal_late = (weights >= 0.5) & (positions[:, 0] >= 4)
            direct_coverage = float((direct[causal_late] >= 0.5).float().mean()) if causal_late.any() else 0.0
            propagated_coverage = float((propagated[causal_late] >= 0.5).float().mean()) if causal_late.any() else 0.0
            direct_score = sample_score(direct)
            propagated_score = sample_score(propagated)
            metrics["direct"].append(direct_score)
            metrics["propagated"].append(propagated_score)
            metrics["direct_coverage"].append(direct_coverage)
            metrics["propagated_coverage"].append(propagated_coverage)
            rows.append(
                {
                    "group": group_name,
                    "scene_id": record["scene_id"],
                    "direct_anchor_score": f"{direct_score:.8f}",
                    "propagated_score": f"{propagated_score:.8f}",
                    "direct_late_coverage": f"{direct_coverage:.8f}",
                    "propagated_late_coverage": f"{propagated_coverage:.8f}",
                }
            )
        group_scores[group_name] = metrics

    summary = {
        "radius": args.radius,
        "decay": args.decay,
        "generic_train_count": len(generic_train),
        "generic_eval_count": len(groups["generic"]["records"]),
        "groups": {},
    }
    for group_name, metrics in group_scores.items():
        summary["groups"][group_name] = {
            name: {"mean": float(np.mean(values)), "std": float(np.std(values))}
            for name, values in metrics.items()
        }
    for method in ("direct", "propagated"):
        for negative_group in evaluation_groups:
            if negative_group == "collision":
                continue
            summary[f"auc_collision_vs_{negative_group}_{method}"] = auc(
                np.asarray(group_scores["collision"][method]),
                np.asarray(group_scores[negative_group][method]),
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "scores.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    torch.save(detector.state_dict(), args.output_dir / "object_detector.pt")
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
