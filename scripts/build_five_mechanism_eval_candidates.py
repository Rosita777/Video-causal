#!/usr/bin/env python3
"""Build the shared Wan/CogVideoX five-mechanism evaluation candidate pool."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


MECHANISMS = {
    "waterdrop_impact": {
        "target": "one large clear water droplet",
        "receivers": [
            ("liquid", "a calm rain barrel filled with water"),
            ("liquid", "a shallow copper basin filled with water"),
            ("liquid", "a clear glass salad bowl filled with water"),
            ("liquid", "a blue ceramic soup bowl filled with water"),
            ("liquid", "a square acrylic tray filled with water"),
            ("liquid", "a white porcelain wash bowl filled with water"),
            ("liquid", "a small granite birdbath filled with water"),
            ("liquid", "a round stainless-steel pan filled with water"),
            ("liquid", "a wide laboratory crystallizing dish filled with water"),
            ("liquid", "a black enamel camping bowl filled with water"),
            ("liquid", "a rectangular ceramic baking dish filled with water"),
            ("liquid", "a clear polycarbonate food container filled with water"),
            ("liquid", "a polished brass basin filled with water"),
            ("liquid", "a shallow stone fountain bowl filled with water"),
            ("liquid", "a white enamel roasting tray filled with water"),
            ("hard", "a dry dark slate tile"),
            ("hard", "a dry white porcelain plate"),
            ("hard", "a dry brushed-steel tray"),
            ("hard", "a dry black granite countertop"),
            ("hard", "a dry clear glass tabletop"),
            ("hard", "a dry glazed blue ceramic tile"),
            ("hard", "a dry polished marble slab"),
            ("hard", "a dry copper sheet"),
            ("hard", "a dry gray concrete paving stone"),
            ("hard", "a dry red plastic cafeteria tray"),
            ("hard", "a dry sealed bamboo board"),
            ("hard", "a dry cream enamel plate"),
            ("hard", "a dry terrazzo floor tile"),
            ("hard", "a dry frosted glass panel"),
            ("hard", "a dry green ceramic saucer"),
        ],
    },
    "red_ball_collision": {
        "target": "one small red rubber ball",
        "receivers": [
            ("upright_row", "three upright lavender wooden blocks"),
            ("upright_row", "three upright orange cork blocks"),
            ("upright_row", "three upright cream foam blocks"),
            ("upright_row", "three upright turquoise toy blocks"),
            ("upright_row", "three upright brown cardboard blocks"),
            ("cup_row", "three upright violet paper cups"),
            ("cup_row", "three upright cream paper cups"),
            ("cup_row", "three upright orange paper cups"),
            ("cup_row", "three upright gray cardboard cups"),
            ("cup_row", "three upright turquoise paper cups"),
            ("tin_row", "three upright copper-colored tins"),
            ("tin_row", "three upright purple aluminum cans"),
            ("tin_row", "three upright cream metal tins"),
            ("tin_row", "three upright orange aluminum cans"),
            ("tin_row", "three upright teal metal tins"),
            ("peg_row", "three upright purple wooden pegs"),
            ("peg_row", "three upright orange wooden pegs"),
            ("peg_row", "three upright cream toy pins"),
            ("peg_row", "three upright turquoise wooden pins"),
            ("peg_row", "three upright brown wooden pegs"),
            ("pawn_row", "three upright violet game pawns"),
            ("pawn_row", "three upright orange game pawns"),
            ("pawn_row", "three upright cream game pawns"),
            ("pawn_row", "three upright turquoise game pawns"),
            ("pawn_row", "three upright brown game pawns"),
            ("domino_row", "three upright purple domino blocks"),
            ("domino_row", "three upright orange domino blocks"),
            ("domino_row", "three upright cream domino blocks"),
            ("domino_row", "three upright turquoise domino blocks"),
            ("domino_row", "three upright brown domino blocks"),
        ],
    },
    "steel_ball_fracture": {
        "target": "one small black steel ball",
        "receivers": [
            ("glass", "a thin clear glass square"),
            ("glass", "a thin frosted glass square"),
            ("glass", "a thin amber glass square"),
            ("glass", "a thin blue glass square"),
            ("glass", "a thin green glass square"),
            ("ceramic", "a flat white ceramic tile"),
            ("ceramic", "a flat blue ceramic tile"),
            ("ceramic", "a flat green ceramic tile"),
            ("ceramic", "a flat gray ceramic tile"),
            ("ceramic", "a flat cream ceramic tile"),
            ("ice", "a thin clear ice sheet"),
            ("ice", "a thin cloudy ice sheet"),
            ("ice", "a thin blue-tinted ice sheet"),
            ("ice", "a thin round ice plate"),
            ("ice", "a thin square ice plate"),
            ("plaster", "a thin white plaster panel"),
            ("plaster", "a thin gray plaster panel"),
            ("plaster", "a thin beige plaster panel"),
            ("plaster", "a thin pink plaster panel"),
            ("plaster", "a thin blue plaster panel"),
            ("cracker", "a large square cream cracker"),
            ("cracker", "a large round wheat cracker"),
            ("cracker", "a large rectangular sesame cracker"),
            ("cracker", "a large pale rice cracker"),
            ("cracker", "a large brown rye cracker"),
            ("brittle_clay", "a thin dry red-clay tile"),
            ("brittle_clay", "a thin dry gray-clay tile"),
            ("brittle_clay", "a thin dry white-clay tile"),
            ("brittle_clay", "a thin dry black-clay tile"),
            ("brittle_clay", "a thin dry yellow-clay tile"),
        ],
    },
    "blue_ball_particles": {
        "target": "one small blue rubber ball",
        "receivers": [
            ("sand", "a shallow tray of fine white sand"),
            ("sand", "a shallow tray of fine black sand"),
            ("sand", "a shallow tray of fine red sand"),
            ("sand", "a shallow tray of fine golden sand"),
            ("sand", "a shallow tray of fine gray sand"),
            ("flour", "a shallow tray of white flour"),
            ("flour", "a shallow tray of whole-wheat flour"),
            ("flour", "a shallow tray of rice flour"),
            ("flour", "a shallow tray of corn flour"),
            ("flour", "a shallow tray of rye flour"),
            ("powder", "a shallow tray of white chalk powder"),
            ("powder", "a shallow tray of blue chalk powder"),
            ("powder", "a shallow tray of pink chalk powder"),
            ("powder", "a shallow tray of yellow chalk powder"),
            ("powder", "a shallow tray of green chalk powder"),
            ("soil", "a shallow tray of dry brown soil"),
            ("soil", "a shallow tray of dry red soil"),
            ("soil", "a shallow tray of dry gray soil"),
            ("soil", "a shallow tray of dry black soil"),
            ("soil", "a shallow tray of dry sandy soil"),
            ("grains", "a shallow tray of uncooked white rice"),
            ("grains", "a shallow tray of uncooked brown rice"),
            ("grains", "a shallow tray of yellow cornmeal"),
            ("grains", "a shallow tray of coarse salt"),
            ("grains", "a shallow tray of tiny beige seeds"),
            ("crumbs", "a shallow tray of fine bread crumbs"),
            ("crumbs", "a shallow tray of dark cookie crumbs"),
            ("crumbs", "a shallow tray of pale cracker crumbs"),
            ("crumbs", "a shallow tray of brown cereal crumbs"),
            ("crumbs", "a shallow tray of white wafer crumbs"),
        ],
    },
    "toy_car_trace": {
        "target": "one small yellow toy car",
        "receivers": [
            ("sand", "a flat bed of fine white sand"),
            ("sand", "a flat bed of fine black sand"),
            ("sand", "a flat bed of fine red sand"),
            ("sand", "a flat bed of fine golden sand"),
            ("sand", "a flat bed of fine gray sand"),
            ("snow", "a flat bed of fresh white snow"),
            ("snow", "a flat bed of soft powdery snow"),
            ("snow", "a flat bed of smooth packed snow"),
            ("snow", "a flat bed of shallow clean snow"),
            ("snow", "a flat bed of fine artificial snow"),
            ("clay", "a flat bed of soft red clay"),
            ("clay", "a flat bed of soft gray clay"),
            ("clay", "a flat bed of soft white clay"),
            ("clay", "a flat bed of soft brown clay"),
            ("clay", "a flat bed of soft blue clay"),
            ("soil", "a flat bed of damp brown soil"),
            ("soil", "a flat bed of damp red soil"),
            ("soil", "a flat bed of damp gray soil"),
            ("soil", "a flat bed of damp black soil"),
            ("soil", "a flat bed of damp sandy soil"),
            ("powder", "a flat bed of white flour"),
            ("powder", "a flat bed of pale chalk powder"),
            ("powder", "a flat bed of pink chalk powder"),
            ("powder", "a flat bed of yellow corn flour"),
            ("powder", "a flat bed of gray cement powder"),
            ("ash", "a flat bed of pale wood ash"),
            ("ash", "a flat bed of dark charcoal ash"),
            ("ash", "a flat bed of fine gray ash"),
            ("ash", "a flat bed of light fireplace ash"),
            ("ash", "a flat bed of smooth volcanic ash"),
        ],
    },
}


def effect_for(mechanism: str, family: str) -> str:
    if mechanism == "waterdrop_impact":
        return "a small splash and circular ripples appear" if family == "liquid" else "the droplet spreads into a visible wet spot with tiny outward splashes"
    if mechanism == "red_ball_collision":
        return "the leftmost receiver falls onto its side only after contact"
    if mechanism == "steel_ball_fracture":
        return "cracks spread outward from the impact point and small fragments separate"
    if mechanism == "blue_ball_particles":
        return "particles scatter outward and a small impact depression remains"
    return "two parallel wheel tracks remain behind the car"


def prompt_for(mechanism: str, target: str, receiver: str, effect: str) -> str:
    common = "A realistic fixed-camera close-up video in one continuous shot."
    if mechanism == "waterdrop_impact":
        return f"{common} At first {receiver} is completely still and undisturbed. Then {target} falls visibly from above and contacts the surface. Only after contact, {effect}. The receiver, camera, lighting, and background remain unchanged."
    if mechanism == "red_ball_collision":
        return f"{common} At first {receiver} stand completely still on a tabletop. Then {target} rolls visibly from left to right and strikes the leftmost receiver. Only after contact, {effect}. The other receivers, camera, lighting, and background remain unchanged."
    if mechanism == "steel_ball_fracture":
        return f"{common} At first {receiver} lies intact and still on a support. Then {target} falls visibly from above and strikes its center. Only after contact, {effect}. The support, camera, lighting, and background remain unchanged."
    if mechanism == "blue_ball_particles":
        return f"{common} At first {receiver} is flat, smooth, and completely still. Then {target} falls visibly from above and hits the center. Only after contact, {effect}. The tray, camera, lighting, and background remain unchanged."
    return f"{common} At first {receiver} is flat, smooth, and completely unmarked. Then {target} drives slowly from left to right across the surface. Only behind the moving car, {effect}. The surrounding surface, camera, lighting, and background remain unchanged."


def build_rows() -> list[dict[str, str]]:
    rows = []
    global_index = 0
    for mechanism, spec in MECHANISMS.items():
        target = str(spec["target"])
        receivers = list(spec["receivers"])
        if len(receivers) != 30:
            raise ValueError(f"{mechanism}: expected 30 receivers, got {len(receivers)}")
        for mechanism_index, (family, receiver) in enumerate(receivers):
            effect = effect_for(mechanism, family)
            rows.append(
                {
                    "candidate_id": f"fiveeval{global_index:03d}",
                    "mechanism": mechanism,
                    "mechanism_index": str(mechanism_index),
                    "target_concept": target,
                    "receiver_family": family,
                    "receiver": receiver,
                    "expected_footprint": effect,
                    "prompt": prompt_for(mechanism, target, receiver, effect),
                    "generation_repetitions": "1",
                    "candidate_status": "pending_wan_and_cog_screen",
                    "intended_split": "evaluation_candidate_only",
                }
            )
            global_index += 1
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_prompts(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{row['prompt']} | {row['target_concept']} | {row['expected_footprint']}" for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def smoke_rows(rows: list[dict[str, str]], per_mechanism: int = 2) -> list[dict[str, str]]:
    selected = []
    for mechanism in MECHANISMS:
        selected.extend(row for row in rows if row["mechanism"] == mechanism and int(row["mechanism_index"]) < per_mechanism)
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-csv", type=Path, default=Path("data/five_mechanism_eval_candidates_v0.csv"))
    parser.add_argument("--output-prompts", type=Path, default=Path("prompts/five_mechanism_eval_candidates_v0.prompts"))
    parser.add_argument("--output-smoke-prompts", type=Path, default=Path("prompts/five_mechanism_eval_smoke10_v0.prompts"))
    args = parser.parse_args()
    rows = build_rows()
    write_csv(args.output_csv, rows)
    write_prompts(args.output_prompts, rows)
    write_prompts(args.output_smoke_prompts, smoke_rows(rows))
    print(f"Wrote {len(rows)} candidates across {len(MECHANISMS)} mechanisms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
