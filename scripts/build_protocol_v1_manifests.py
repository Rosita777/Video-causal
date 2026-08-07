#!/usr/bin/env python3
"""Build the frozen Protocol v1 training, preservation, and evaluation manifests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


TRAIN_VARIANTS = ("explicit_full", "concise_full", "contact_only", "natural_implicit")
EVAL_VARIANTS = (
    "eval_explicit_a",
    "eval_explicit_b",
    "eval_implicit_a",
    "eval_implicit_b",
    "eval_compact",
)
GENERALIZATION_GROUPS = ("seen_seen", "unseen_seen", "seen_unseen", "unseen_unseen")

MANIFEST_FIELDS = (
    "protocol_version",
    "sample_id",
    "training_role",
    "mechanism",
    "split",
    "generalization_group",
    "source_id",
    "source_object",
    "source_seen",
    "receiver_id",
    "receiver",
    "receiver_seen",
    "prompt_variant",
    "prompt",
    "target_concept",
    "expected_footprint",
    "expected_counterfactual_state",
    "seed",
    "num_frames",
    "fps",
    "reference_start_inclusive",
    "reference_end_exclusive",
)


PRESERVE_SCENES = (
    ("books", "three closed books resting neatly on a wooden desk"),
    ("vase", "one blue ceramic vase standing on a shelf"),
    ("lamp", "one desk lamp illuminating an otherwise empty table"),
    ("clock", "one wall clock with its second hand moving steadily"),
    ("fan", "one small desk fan spinning at a constant speed"),
    ("train", "one toy train moving smoothly around an empty circular track"),
    ("pendulum", "one metal pendulum swinging gently from side to side"),
    ("conveyor", "a conveyor belt carrying identical sealed boxes without contact or collision"),
    ("walker", "one person walking steadily across an empty room"),
    ("clouds", "thin clouds drifting slowly across a clear sky"),
    ("traffic_light", "one traffic light changing normally from green to yellow to red"),
    ("turntable", "one colored wooden disk rotating smoothly on a turntable"),
)


def clean_prefix(mechanism: str, source: dict[str, str], receiver: dict[str, str]) -> str:
    return (
        f"During the first two seconds, the scene contains {receiver['name']}; {receiver['clean_state']}. "
        "The source object is not visible, and nothing changes."
    )


def causal_action(mechanism: str, source: dict[str, str], receiver: dict[str, str]) -> str:
    if mechanism == "water_impact":
        return f"{source['name']} {source['motion']}, enters the center of the water, and makes contact"
    if mechanism == "rigid_collision":
        return f"{source['name']} {source['motion']} and strikes the nearest receiver object once"
    if mechanism == "brittle_fracture":
        return f"{source['name']} {source['motion']} and strikes {receiver['name']} once"
    if mechanism == "powder_impact":
        return f"{source['name']} {source['motion']} and impacts the center of the material"
    raise KeyError(mechanism)


def fixed_context(mechanism: str) -> str:
    if mechanism == "water_impact":
        return "The camera, lighting, water container, and background remain fixed throughout."
    if mechanism == "rigid_collision":
        return "The camera, tabletop, lighting, and background remain fixed throughout."
    if mechanism == "brittle_fracture":
        return "The camera, support surface, lighting, and background remain fixed throughout."
    if mechanism == "powder_impact":
        return "The camera, tray, lighting, and background remain fixed throughout."
    raise KeyError(mechanism)


def training_prompt(
    mechanism: str,
    spec: dict[str, object],
    source: dict[str, str],
    receiver: dict[str, str],
    variant: str,
) -> str:
    prefix = clean_prefix(mechanism, source, receiver)
    action = causal_action(mechanism, source, receiver)
    footprint = str(spec["footprint"])
    fixed = fixed_context(mechanism)
    if variant == "explicit_full":
        body = f"Then {action}. Only after contact, the event produces {footprint}."
    elif variant == "concise_full":
        body = f"After the quiet opening, {action}; this produces {footprint}."
    elif variant == "contact_only":
        body = f"After the quiet opening, {action}. The event continues naturally after contact."
    elif variant == "natural_implicit":
        body = f"The continuous shot then shows an ordinary physical event in which {action}."
    else:
        raise KeyError(variant)
    return f"A simple realistic fixed-camera video in one continuous shot. {prefix} {body} {fixed}"


def eval_prompt(
    mechanism: str,
    spec: dict[str, object],
    source: dict[str, str],
    receiver: dict[str, str],
    variant: str,
) -> str:
    prefix = clean_prefix(mechanism, source, receiver)
    action = causal_action(mechanism, source, receiver)
    footprint = str(spec["footprint"])
    fixed = "No cuts, camera motion, people, hands, or additional moving objects appear."
    if variant == "eval_explicit_a":
        body = f"Next, {action}. Following that contact, the event produces {footprint}."
    elif variant == "eval_explicit_b":
        body = f"The source arrives only after the still opening: {action}, causing {footprint}."
    elif variant == "eval_implicit_a":
        body = f"The shot then naturally captures the moment when {action}."
    elif variant == "eval_implicit_b":
        body = f"Without any camera change, the scene continues as {action}."
    elif variant == "eval_compact":
        body = f"After two still seconds, {action}; afterward, the event produces {footprint}."
    else:
        raise KeyError(variant)
    return f"A realistic locked-camera video in one continuous shot. {prefix} {body} {fixed}"


def base_row(registry: dict[str, object], sample_id: str) -> dict[str, str]:
    return {
        "protocol_version": str(registry["protocol_version"]),
        "sample_id": sample_id,
        "training_role": "",
        "mechanism": "",
        "split": "",
        "generalization_group": "",
        "source_id": "",
        "source_object": "",
        "source_seen": "",
        "receiver_id": "",
        "receiver": "",
        "receiver_seen": "",
        "prompt_variant": "",
        "prompt": "",
        "target_concept": "",
        "expected_footprint": "",
        "expected_counterfactual_state": "",
        "seed": str(registry["fixed_seed"]),
        "num_frames": str(registry["video_frames"]),
        "fps": str(registry["fps"]),
        "reference_start_inclusive": "0",
        "reference_end_exclusive": str(registry["clean_prefix_frames"]),
    }


def build_train_rows(registry: dict[str, object]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for mechanism, raw_spec in registry["mechanisms"].items():
        spec = dict(raw_spec)
        for source in spec["train_sources"]:
            for receiver in spec["train_receivers"]:
                for variant in TRAIN_VARIANTS:
                    sample_id = f"train_{mechanism}_{source['id']}_{receiver['id']}_{variant}"
                    row = base_row(registry, sample_id)
                    row.update(
                        {
                            "training_role": "erase",
                            "mechanism": mechanism,
                            "split": "train",
                            "source_id": source["id"],
                            "source_object": source["name"],
                            "source_seen": "yes",
                            "receiver_id": receiver["id"],
                            "receiver": receiver["name"],
                            "receiver_seen": "yes",
                            "prompt_variant": variant,
                            "prompt": training_prompt(mechanism, spec, source, receiver, variant),
                            "target_concept": spec["name"],
                            "expected_footprint": spec["footprint"],
                            "expected_counterfactual_state": spec["counterfactual_state"],
                        }
                    )
                    rows.append(row)
    return rows


def preserve_prompt(scene: str, variant: int) -> str:
    endings = (
        "The camera, lighting, background, and all object identities remain unchanged.",
        "The shot is stable, realistic, and free of impacts, breakage, splashes, powder bursts, or collisions.",
        "No object enters from above and no abrupt physical event occurs anywhere in the scene.",
    )
    return f"A realistic fixed-camera video in one continuous shot showing {scene}. {endings[variant]}"


def build_preserve_rows(registry: dict[str, object]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for scene_id, scene in PRESERVE_SCENES:
        for variant in range(3):
            sample_id = f"preserve_{scene_id}_{variant}"
            row = base_row(registry, sample_id)
            row.update(
                {
                    "training_role": "preserve",
                    "mechanism": "generic_preservation",
                    "split": "train",
                    "prompt_variant": f"preserve_{variant}",
                    "prompt": preserve_prompt(scene, variant),
                    "target_concept": "generic preservation",
                    "reference_start_inclusive": "",
                    "reference_end_exclusive": "",
                }
            )
            rows.append(row)
    return rows


def selected_pairs(sources: list[dict[str, str]], receivers: list[dict[str, str]], count: int) -> list[tuple[dict[str, str], dict[str, str]]]:
    pairs = [(source, receiver) for source in sources for receiver in receivers]
    if len(pairs) >= count:
        return pairs[:count]
    return (pairs + pairs[: count - len(pairs)])[:count]


def build_eval_rows(registry: dict[str, object]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for mechanism, raw_spec in registry["mechanisms"].items():
        spec = dict(raw_spec)
        group_inputs = {
            "seen_seen": (spec["train_sources"], spec["train_receivers"], "yes", "yes"),
            "unseen_seen": (spec["test_sources"], spec["train_receivers"], "no", "yes"),
            "seen_unseen": (spec["train_sources"], spec["test_receivers"], "yes", "no"),
            "unseen_unseen": (spec["test_sources"], spec["test_receivers"], "no", "no"),
        }
        mechanism_index = list(registry["mechanisms"]).index(mechanism)
        for group, (sources, receivers, source_seen, receiver_seen) in group_inputs.items():
            for item_index, (source, receiver) in enumerate(selected_pairs(sources, receivers, 5)):
                variant = EVAL_VARIANTS[item_index]
                sample_id = f"eval_{mechanism}_{group}_{item_index:02d}"
                row = base_row(registry, sample_id)
                row.update(
                    {
                        "training_role": "",
                        "mechanism": mechanism,
                        "split": "test",
                        "generalization_group": group,
                        "source_id": source["id"],
                        "source_object": source["name"],
                        "source_seen": source_seen,
                        "receiver_id": receiver["id"],
                        "receiver": receiver["name"],
                        "receiver_seen": receiver_seen,
                        "prompt_variant": variant,
                        "prompt": eval_prompt(mechanism, spec, source, receiver, variant),
                        "target_concept": spec["name"],
                        "expected_footprint": spec["footprint"],
                        "expected_counterfactual_state": spec["counterfactual_state"],
                        "seed": str(int(registry["fixed_seed"]) + mechanism_index),
                    }
                )
                rows.append(row)
    return rows


def validate(registry: dict[str, object], train: list[dict[str, str]], preserve: list[dict[str, str]], evaluation: list[dict[str, str]]) -> None:
    if len(train) != 144 or len(preserve) != 36 or len(evaluation) != 80:
        raise ValueError(f"Unexpected row counts: train={len(train)}, preserve={len(preserve)}, eval={len(evaluation)}")
    ids = [row["sample_id"] for row in train + preserve + evaluation]
    if len(ids) != len(set(ids)):
        raise ValueError("Sample IDs are not unique")
    prompts = [row["prompt"] for row in train + preserve + evaluation]
    if len(prompts) != len(set(prompts)):
        raise ValueError("Prompts are not unique")
    for mechanism, spec in registry["mechanisms"].items():
        train_sources = {item["id"] for item in spec["train_sources"]}
        test_sources = {item["id"] for item in spec["test_sources"]}
        train_receivers = {item["id"] for item in spec["train_receivers"]}
        test_receivers = {item["id"] for item in spec["test_receivers"]}
        if train_sources & test_sources:
            raise ValueError(f"Source leakage in {mechanism}")
        if train_receivers & test_receivers:
            raise ValueError(f"Receiver leakage in {mechanism}")
        mechanism_train = [row for row in train if row["mechanism"] == mechanism]
        mechanism_eval = [row for row in evaluation if row["mechanism"] == mechanism]
        if len(mechanism_train) != 36 or len(mechanism_eval) != 20:
            raise ValueError(f"Unexpected mechanism counts for {mechanism}")
        groups = Counter(row["generalization_group"] for row in mechanism_eval)
        if groups != Counter({group: 5 for group in GENERALIZATION_GROUPS}):
            raise ValueError(f"Unexpected eval groups for {mechanism}: {groups}")


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_prompts(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"# Protocol v1: {path.stem}\n\n")
        for row in rows:
            handle.write(f"{row['prompt']} | {row['target_concept']} | {row['sample_id']}\n")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, default=Path("data/protocol_v1/registry.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/protocol_v1"))
    args = parser.parse_args()

    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    train = build_train_rows(registry)
    preserve = build_preserve_rows(registry)
    evaluation = build_eval_rows(registry)
    validate(registry, train, preserve, evaluation)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "train_erase_manifest": args.output_dir / "train_erase_manifest.csv",
        "preserve_manifest": args.output_dir / "preserve_manifest.csv",
        "eval_manifest": args.output_dir / "eval_manifest.csv",
        "train_erase_prompts": args.output_dir / "train_erase.prompts",
        "preserve_prompts": args.output_dir / "preserve.prompts",
        "eval_prompts": args.output_dir / "eval.prompts",
    }
    write_csv(artifacts["train_erase_manifest"], train)
    write_csv(artifacts["preserve_manifest"], preserve)
    write_csv(artifacts["eval_manifest"], evaluation)
    write_prompts(artifacts["train_erase_prompts"], train)
    write_prompts(artifacts["preserve_prompts"], preserve)
    write_prompts(artifacts["eval_prompts"], evaluation)

    summary = {
        "protocol_version": registry["protocol_version"],
        "counts": {"train_erase": len(train), "preserve": len(preserve), "eval": len(evaluation)},
        "per_mechanism": {
            mechanism: {
                "train_erase": sum(row["mechanism"] == mechanism for row in train),
                "eval": sum(row["mechanism"] == mechanism for row in evaluation),
            }
            for mechanism in registry["mechanisms"]
        },
        "checks": {
            "unique_sample_ids": True,
            "unique_prompts": True,
            "source_train_test_disjoint_within_mechanism": True,
            "receiver_train_test_disjoint_within_mechanism": True,
            "eval_groups_five_each": True,
        },
        "sha256": {name: sha256(path) for name, path in artifacts.items()},
    }
    summary_path = args.output_dir / "manifest_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
