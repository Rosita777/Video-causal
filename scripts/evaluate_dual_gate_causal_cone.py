#!/usr/bin/env python3
"""Ablate motion, object, soft-product, and strict causal-cone gates."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch


def normalize(features: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(features.float(), dim=-1)


def linear_detector(
    positives: torch.Tensor,
    negatives: torch.Tensor,
    *,
    steps: int,
    seed: int,
    device: torch.device,
) -> torch.nn.Linear:
    positives = normalize(positives).to(device)
    negatives = normalize(negatives).to(device)
    torch.manual_seed(seed)
    detector = torch.nn.Linear(positives.shape[1], 1, device=device)
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
    return detector.cpu().eval()


def train_object_detector(
    target_records: list[dict[str, object]],
    generic_records: list[dict[str, object]],
    hard_negative_records: list[dict[str, object]],
    *,
    steps: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.nn.Linear, torch.Tensor]:
    positive_parts = []
    negative_parts = []
    for record in target_records:
        weights = record["mask_weights"].float()
        cutoff = max(0.25, float(torch.quantile(weights, 0.75)))
        positive_parts.append(record["factual_features"].float()[weights >= cutoff])
        negative_parts.append(record["background_features"].float())
    for record in generic_records:
        negative_parts.append(record["factual_features"].float())
        negative_parts.append(record["background_features"].float())
    for record in hard_negative_records:
        negative_parts.append(record["factual_features"].float())
        negative_parts.append(record["background_features"].float())
    positives = torch.cat(positive_parts)
    detector = linear_detector(
        positives,
        torch.cat(negative_parts),
        steps=steps,
        seed=seed,
        device=device,
    )
    return detector, positives


def train_mechanism_detector(
    collision_records: list[dict[str, object]],
    generic_records: list[dict[str, object]],
    *,
    steps: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.nn.Linear, torch.Tensor]:
    positives = torch.cat([record["features"].float() for record in collision_records])
    negatives = torch.cat([record["features"].float() for record in generic_records])
    detector = linear_detector(
        positives,
        negatives,
        steps=steps,
        seed=seed,
        device=device,
    )
    return detector, negatives


@torch.no_grad()
def probabilities(features: torch.Tensor, detector: torch.nn.Linear) -> torch.Tensor:
    return torch.sigmoid(detector(normalize(features)).flatten())


@torch.no_grad()
def strict_causal_cone(
    record: dict[str, object],
    object_scores: torch.Tensor,
    mechanism_scores: torch.Tensor,
    *,
    object_threshold: float,
    mechanism_threshold: float,
    radius: float,
    edge_threshold: float,
    seed_window: int,
) -> torch.Tensor:
    features = normalize(record["factual_features"])
    positions = record["positions"].float()
    frames = sorted({int(value) for value in positions[:, 0].tolist()})
    cone = torch.zeros_like(object_scores)

    seed_frame = None
    for frame in frames:
        indices = torch.where(positions[:, 0] == frame)[0]
        joint = (object_scores[indices] >= object_threshold) & (
            mechanism_scores[indices] >= mechanism_threshold
        )
        if joint.any():
            cone[indices[joint]] = 1.0
            seed_frame = frame
            break
    if seed_frame is None:
        return cone

    future_frames = [frame for frame in frames if frame >= seed_frame]
    for previous_frame, current_frame in zip(future_frames, future_frames[1:]):
        previous_indices = torch.where(positions[:, 0] == previous_frame)[0]
        current_indices = torch.where(positions[:, 0] == current_frame)[0]
        affinity = torch.relu(features[previous_indices] @ features[current_indices].T)
        previous_xy = positions[previous_indices, 1:]
        current_xy = positions[current_indices, 1:]
        distance2 = (previous_xy[:, None, :] - current_xy[None, :, :]).square().sum(dim=-1)
        spatial = torch.exp(-distance2 / (2.0 * radius * radius))
        transition = (cone[previous_indices, None] * affinity * spatial).amax(dim=0)
        allowed = mechanism_scores[current_indices] >= mechanism_threshold
        propagated = (transition >= edge_threshold) & allowed
        if current_frame < seed_frame + seed_window:
            direct_seed = (
                (object_scores[current_indices] >= object_threshold) & allowed
            )
            propagated |= direct_seed
        cone[current_indices] = propagated.float()
    return cone


def late_coverage(record: dict[str, object], values: torch.Tensor) -> float:
    weights = record["mask_weights"].float()
    positions = record["positions"]
    causal_late = (weights >= 0.5) & (positions[:, 0] >= 4)
    return float((values[causal_late] >= 0.5).float().mean()) if causal_late.any() else 0.0


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
    parser.add_argument("--collision-manifest", type=Path, required=True)
    parser.add_argument("--generic-motion-scores", type=Path, required=True)
    parser.add_argument("--heldout-receivers", default="paper_cup,short_tin,wide_domino,wood_peg")
    parser.add_argument("--generic-train-count", type=int, default=16)
    parser.add_argument("--generic-eval-count", type=int, default=8)
    parser.add_argument("--other-ball-train-count", type=int, default=3)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--radius", type=float, default=6.0)
    parser.add_argument("--target-train-coverage", type=float, default=0.80)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    payload = torch.load(args.features, map_location="cpu", weights_only=False)
    groups = payload["groups"]
    with args.collision_manifest.open(newline="", encoding="utf-8") as handle:
        receiver_by_scene = {row["scene_id"]: row["receiver_id"] for row in csv.DictReader(handle)}
    heldout_receivers = {value.strip() for value in args.heldout_receivers.split(",") if value.strip()}
    collision_train = [
        record for record in groups["collision"]["records"]
        if receiver_by_scene[record["scene_id"]] not in heldout_receivers
    ]
    collision_eval = [
        record for record in groups["collision"]["records"]
        if receiver_by_scene[record["scene_id"]] in heldout_receivers
    ]

    with args.generic_motion_scores.open(newline="", encoding="utf-8") as handle:
        ranked_paths = [str(Path(row["video_path"]).resolve()) for row in csv.DictReader(handle)]
    generic_by_video = {
        str(Path(record["video_path"]).resolve()): record
        for record in groups["generic"]["records"]
    }
    ranked_generic = [generic_by_video[path] for path in ranked_paths if path in generic_by_video]
    required = args.generic_train_count + args.generic_eval_count
    if len(ranked_generic) < required:
        raise ValueError(f"Need {required} ranked generic records, found {len(ranked_generic)}")
    generic_train = ranked_generic[: args.generic_train_count]
    generic_eval = ranked_generic[args.generic_train_count : required]
    other_ball_train = groups["other_ball"]["records"][: args.other_ball_train_count]
    other_ball_eval = groups["other_ball"]["records"][args.other_ball_train_count :]

    device = torch.device(args.device)
    object_detector, object_positives = train_object_detector(
        groups["target_object"]["records"][:4],
        generic_train,
        other_ball_train,
        steps=args.steps,
        seed=args.seed,
        device=device,
    )
    mechanism_detector, mechanism_negatives = train_mechanism_detector(
        collision_train,
        generic_train,
        steps=args.steps,
        seed=args.seed + 1,
        device=device,
    )
    object_positive_scores = probabilities(object_positives, object_detector)
    mechanism_negative_scores = probabilities(mechanism_negatives, mechanism_detector)

    calibration_records = collision_train + generic_train
    collision_train_ids = {record["scene_id"] for record in collision_train}
    calibration_scores = {
        record["scene_id"]: (
            probabilities(record["factual_features"], object_detector),
            probabilities(record["features"], mechanism_detector),
        )
        for record in calibration_records
    }
    candidates = []
    for object_quantile in (0.05, 0.10, 0.20, 0.30):
        object_threshold = float(torch.quantile(object_positive_scores, object_quantile))
        for mechanism_quantile in (0.90, 0.95, 0.99):
            mechanism_threshold = float(
                torch.quantile(mechanism_negative_scores, mechanism_quantile)
            )
            for radius in (2.0, 4.0, 6.0):
                for edge_threshold in (0.1, 0.3, 0.5, 0.7):
                    for seed_window in (1, 2, 4, 6, 8, 13):
                        collision_coverages = []
                        generic_coverages = []
                        for record in calibration_records:
                            object_scores, mechanism_scores = calibration_scores[record["scene_id"]]
                            cone = strict_causal_cone(
                                record,
                                object_scores,
                                mechanism_scores,
                                object_threshold=object_threshold,
                                mechanism_threshold=mechanism_threshold,
                                radius=radius,
                                edge_threshold=edge_threshold,
                                seed_window=seed_window,
                            )
                            destination = (
                                collision_coverages
                                if record["scene_id"] in collision_train_ids
                                else generic_coverages
                            )
                            destination.append(late_coverage(record, cone))
                        collision_coverage = float(np.mean(collision_coverages))
                        generic_coverage = float(np.mean(generic_coverages))
                        objective = (
                            abs(collision_coverage - args.target_train_coverage)
                            + 2.0 * generic_coverage
                        )
                        candidates.append({
                            "objective": objective,
                            "object_quantile": object_quantile,
                            "mechanism_quantile": mechanism_quantile,
                            "object_threshold": object_threshold,
                            "mechanism_threshold": mechanism_threshold,
                            "radius": radius,
                            "edge_threshold": edge_threshold,
                            "seed_window": seed_window,
                            "collision_train_coverage": collision_coverage,
                            "generic_train_coverage": generic_coverage,
                        })
    candidates.sort(key=lambda item: (item["objective"], item["generic_train_coverage"]))
    selected = candidates[0]
    object_threshold = selected["object_threshold"]
    mechanism_threshold = selected["mechanism_threshold"]

    evaluation = {
        "collision": collision_eval,
        "generic": generic_eval,
        "waterdrop": groups["waterdrop"]["records"],
        "other_ball": other_ball_eval,
        "negation": groups["negation"]["records"],
    }
    methods = ("motion_only", "object_only", "soft_product", "gated_cone")
    group_metrics: dict[str, dict[str, dict[str, list[float]]]] = {}
    rows = []
    for group_name, records in evaluation.items():
        group_metrics[group_name] = {
            method: {"score": [], "coverage": []} for method in methods
        }
        for record in records:
            object_scores = probabilities(record["factual_features"], object_detector)
            mechanism_scores = probabilities(record["features"], mechanism_detector)
            values_by_method = {
                "motion_only": mechanism_scores,
                "object_only": object_scores,
                "soft_product": object_scores * mechanism_scores,
                "gated_cone": strict_causal_cone(
                    record,
                    object_scores,
                    mechanism_scores,
                    object_threshold=object_threshold,
                    mechanism_threshold=mechanism_threshold,
                    radius=selected["radius"],
                    edge_threshold=selected["edge_threshold"],
                    seed_window=selected["seed_window"],
                ),
            }
            for method, values in values_by_method.items():
                score = sample_score(values)
                coverage = late_coverage(record, values)
                group_metrics[group_name][method]["score"].append(score)
                group_metrics[group_name][method]["coverage"].append(coverage)
                rows.append({
                    "group": group_name,
                    "scene_id": record["scene_id"],
                    "method": method,
                    "score": f"{score:.8f}",
                    "late_coverage": f"{coverage:.8f}",
                })

    summary: dict[str, object] = {
        "collision_train_count": len(collision_train),
        "collision_eval_count": len(collision_eval),
        "generic_train_count": len(generic_train),
        "generic_eval_count": len(generic_eval),
        "other_ball_train_count": len(other_ball_train),
        "other_ball_eval_count": len(other_ball_eval),
        "object_threshold": object_threshold,
        "mechanism_threshold": mechanism_threshold,
        "selected_gate": selected,
        "calibration_candidates": candidates[:10],
        "groups": {},
    }
    for group_name, method_metrics in group_metrics.items():
        summary["groups"][group_name] = {}
        for method, metrics in method_metrics.items():
            summary["groups"][group_name][method] = {
                name: {"mean": float(np.mean(values)), "std": float(np.std(values))}
                for name, values in metrics.items()
            }
    for method in methods:
        positive = np.asarray(group_metrics["collision"][method]["score"])
        for negative_group in ("generic", "waterdrop", "other_ball", "negation"):
            negative = np.asarray(group_metrics[negative_group][method]["score"])
            summary[f"auc_collision_vs_{negative_group}_{method}"] = auc(positive, negative)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "scores.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    torch.save(
        {
            "object_detector": object_detector.state_dict(),
            "mechanism_detector": mechanism_detector.state_dict(),
            "object_threshold": object_threshold,
            "mechanism_threshold": mechanism_threshold,
        },
        args.output_dir / "detectors.pt",
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
