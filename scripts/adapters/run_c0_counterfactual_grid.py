#!/usr/bin/env python3
"""Run the Method C0 paired counterfactual prompt grid."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from adapters.zeroscope_adapter_common import encode_cfg, load_zeroscope_pipe  # noqa: E402
from adapters.run_zeroscope_attention_probe import run_attention_recording_pipeline  # noqa: E402
from build_mvp0_causal_chain_probe import scene_context  # noqa: E402


BASELINE = "c0_counterfactual_grid"
VARIANTS = ["original", "remove_target", "footprint_only", "target_only"]
VARIANT_SETS = {
    "all": VARIANTS,
    "original": ["original"],
}
VARIANT_LABELS = {
    "original": "original target plus footprint",
    "remove_target": "remove target and footprint",
    "footprint_only": "footprint without target",
    "target_only": "target without footprint",
}
EXPECTED_STATES = {
    "original": ("yes", "yes"),
    "remove_target": ("no", "no"),
    "footprint_only": ("no", "yes"),
    "target_only": ("yes", "no"),
}
PROMPT_TEMPLATES = ["legacy", "c02_discrete"]
C02_SURFACE_OVERRIDES = {
    "makeup brush": "compact of pink powder",
    "garden rake": "smooth soil bed",
    "hand": "pillow surface",
    "marker pen": "whiteboard surface",
}
C02_FOOTPRINT_OVERRIDES = {
    "makeup brush": ("a pink powder cloud", "pink powder cloud"),
    "garden rake": ("parallel grooves in the soil", "parallel grooves in the soil"),
    "hand": ("a deep dent in the pillow", "deep dent in the pillow"),
    "marker pen": ("a black line on the whiteboard", "black line on the whiteboard"),
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def slugify(text: str, max_length: int = 72) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    if not slug:
        return "prompt"
    return slug[:max_length].rstrip("-") or "prompt"


def generation_config(args: argparse.Namespace) -> dict[str, object]:
    return {
        "seed": args.seed,
        "num_inference_steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "num_frames": args.num_frames,
        "fps": args.fps,
        "height": args.height,
        "width": args.width,
        "dtype": args.dtype,
        "device": args.device,
        "enable_model_cpu_offload": args.enable_model_cpu_offload,
        "enable_sequential_cpu_offload": args.enable_sequential_cpu_offload,
        "vae_slicing": args.vae_slicing,
        "seeds_per_item": args.seeds_per_item,
        "variant_grid": selected_variants(args),
        "item_indices": args.item_indices,
        "prompt_template": args.prompt_template,
    }


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def phrase_key(text: str) -> str:
    return "".join(char for char in str(text).lower() if char.isalnum())


def append_absence_if_missing(prompt: str, phrase: str) -> str:
    prompt = normalize_space(prompt).rstrip(".")
    phrase = normalize_space(phrase).rstrip(".")
    if not phrase:
        return prompt + "."
    if phrase_key(phrase) in phrase_key(prompt):
        return prompt + "."
    return f"{prompt}. The scene has no {phrase}."


def target_only_prompt(item: dict[str, object]) -> str:
    context = scene_context(item).rstrip(".")
    target = normalize_space(str(item.get("target_concept", "target")))
    footprint = normalize_space(str(item.get("causal_footprint", "causal footprint")))
    if not context:
        context = "A realistic fixed-camera video of the same scene"
    return normalize_space(
        f"{context}. The {target} is clearly visible but does not touch, strike, "
        f"collide with, or disturb the scene. The scene has no {footprint}."
    )


def c02_surface_for(item: dict[str, object]) -> str:
    target = normalize_space(str(item.get("target_concept", ""))).lower()
    return C02_SURFACE_OVERRIDES.get(target, "surface")


def c02_footprints_for(item: dict[str, object]) -> tuple[str, str]:
    target = normalize_space(str(item.get("target_concept", ""))).lower()
    default = normalize_space(str(item.get("causal_footprint", "causal footprint")))
    return C02_FOOTPRINT_OVERRIDES.get(target, (default, default))


def c02_scene_anchor() -> str:
    return normalize_space(
        "A realistic fixed-camera close-up video of the same simple scene. "
        "The background, camera framing, and surface stay consistent across the clip."
    )


def c02_discrete_prompt(item: dict[str, object], variant: str) -> tuple[str, str]:
    target = normalize_space(str(item.get("target_concept", "target")))
    visible_footprint, absence_footprint = c02_footprints_for(item)
    surface = c02_surface_for(item)
    anchor = c02_scene_anchor()
    if variant == "original":
        return normalize_space(
            f"{anchor} The {target} is clearly visible and contacts the {surface}. "
            f"After contact, {visible_footprint} is clearly visible."
        ), ""
    if variant == "remove_target":
        return normalize_space(
            f"{anchor} No {target} is present. No visible cause is present. "
            f"The {surface} stays clean and unchanged. There is no {absence_footprint}."
        ), ""
    if variant == "footprint_only":
        return normalize_space(
            f"{anchor} No {target} is present and no visible cause appears in the frame. "
            f"{visible_footprint.capitalize()} is clearly visible on the {surface}. "
            "The scene otherwise stays the same."
        ), ""
    if variant == "target_only":
        return normalize_space(
            f"{anchor} The {target} is clearly visible, but it is separated from the "
            f"{surface} and does not touch, strike, mark, press, disturb, or change it. "
            f"There is no {absence_footprint}."
        ), ""
    raise ValueError(f"unknown variant: {variant}")


def variant_prompt(
    item: dict[str, object],
    variant: str,
    *,
    prompt_template: str = "legacy",
) -> tuple[str, str]:
    if prompt_template == "c02_discrete":
        return c02_discrete_prompt(item, variant)
    target = str(item.get("target_concept", ""))
    footprint = str(item.get("causal_footprint", ""))
    if variant == "original":
        return normalize_space(item.get("generation_prompt") or item.get("source_prompt") or ""), ""
    if variant == "remove_target":
        prompt = normalize_space(item.get("counterfactual_prompt") or "")
        if not prompt:
            context = scene_context(item).rstrip(".")
            prompt = f"{context}. No {target} is present."
        prompt = append_absence_if_missing(prompt, footprint)
        return prompt, ""
    if variant == "footprint_only":
        prompt = normalize_space(item.get("control_prompt") or "")
        if not prompt:
            context = scene_context(item).rstrip(".")
            prompt = (
                f"{context}. {footprint} is visible, with no {target} or visible cause "
                "in the frame."
            )
        if target and phrase_key(target) not in phrase_key(prompt):
            prompt = append_absence_if_missing(prompt, target)
        return normalize_space(prompt), ""
    if variant == "target_only":
        return target_only_prompt(item), ""
    raise ValueError(f"unknown variant: {variant}")


def selected_variants(args: argparse.Namespace) -> list[str]:
    return list(VARIANT_SETS[str(args.variant_set)])


def parse_item_indices(text: str) -> list[int]:
    text = normalize_space(text)
    if not text:
        return []
    values: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(int(part))
    return values


def select_probe_items(probe_items: Sequence[dict], args: argparse.Namespace) -> list[dict]:
    requested = parse_item_indices(str(args.item_indices))
    if not requested:
        return list(probe_items)
    requested_set = set(requested)
    selected = [
        item for item in probe_items if int(item.get("probe_index", -1)) in requested_set
    ]
    order = {probe_index: index for index, probe_index in enumerate(requested)}
    return sorted(selected, key=lambda item: order[int(item.get("probe_index", -1))])


def build_items(args: argparse.Namespace, probe_items: Sequence[dict]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    variants = selected_variants(args)
    for item in probe_items:
        probe_index = int(item.get("probe_index", len(rows)))
        pair_id = str(item.get("pair_id", f"item_{probe_index}"))
        slug = slugify(pair_id)
        for seed_index in range(args.seeds_per_item):
            seed = args.seed + probe_index + seed_index
            for variant in variants:
                prompt, negative_prompt = variant_prompt(
                    item,
                    variant,
                    prompt_template=str(args.prompt_template),
                )
                expected_target, expected_footprint = EXPECTED_STATES[variant]
                if args.seeds_per_item == 1:
                    video_name = f"{probe_index:03d}_{slug}_{variant}_seed{seed}.mp4"
                else:
                    video_name = (
                        f"{probe_index:03d}_{slug}_seed{seed_index:02d}_"
                        f"{variant}_seed{seed}.mp4"
                    )
                video_path = args.output_dir / "videos" / video_name
                rows.append(
                    {
                        "probe_index": probe_index,
                        "pair_id": pair_id,
                        "slice_index": item.get("slice_index", probe_index),
                        "source_index": str(item.get("source_index", "")),
                        "mechanism_type": str(item.get("mechanism_type", "")),
                        "seed_index": seed_index,
                        "variant": variant,
                        "variant_label": VARIANT_LABELS[variant],
                        "variant_role": variant,
                        "prompt": prompt,
                        "negative_prompt": negative_prompt,
                        "source_prompt": str(item.get("source_prompt", "")),
                        "generation_prompt": str(
                            item.get("generation_prompt") or item.get("source_prompt", "")
                        ),
                        "counterfactual_prompt": str(item.get("counterfactual_prompt", "")),
                        "control_prompt": str(item.get("control_prompt", "")),
                        "target_concept": str(item.get("target_concept", "")),
                        "causal_footprint": str(item.get("causal_footprint", "")),
                        "expected_target_visible": expected_target,
                        "expected_footprint_visible": expected_footprint,
                        "seed": seed,
                        "video_path": str(video_path),
                        "clean_video_path": str(item.get("clean_video_path", "")),
                    }
                )
    return rows


def write_manifest(
    args: argparse.Namespace,
    *,
    source_manifest: dict,
    rows: Sequence[dict[str, object]],
) -> Path:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline": BASELINE,
        "model": args.model,
        "dry_run": args.dry_run,
        "probe_manifest": str(args.probe_manifest),
        "source_probe_name": source_manifest.get("probe_name", ""),
        "variant_grid": selected_variants(args),
        "generation": generation_config(args),
        "items": list(rows),
    }
    out = args.output_dir / "generation_manifest.json"
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


def generate_counterfactual_videos(args: argparse.Namespace, rows: Sequence[dict[str, object]]) -> None:
    torch_module, export_to_video, pipe, selected_device = load_zeroscope_pipe(args)
    generator_device = (
        "cuda"
        if str(selected_device).startswith("cuda") and torch_module.cuda.is_available()
        else "cpu"
    )
    for row in rows:
        video_path = Path(str(row["video_path"]))
        video_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_embeds, negative_prompt_embeds = encode_cfg(
            pipe,
            torch_module,
            prompt=str(row["prompt"]),
            negative_prompt=str(row.get("negative_prompt", "")),
            device=selected_device,
        )
        generator = torch_module.Generator(device=generator_device).manual_seed(int(row["seed"]))
        frames = run_attention_recording_pipeline(
            pipe,
            torch_module,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            generator=generator,
            steps=args.steps,
            num_frames=args.num_frames,
            guidance_scale=args.guidance_scale,
            height=args.height,
            width=args.width,
            step_state={"index": -1},
            output_type="np",
            decode_video=True,
        )
        export_to_video(frames[0], str(video_path), fps=args.fps)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--probe-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="models/zeroscope_v2_576w")
    parser.add_argument("--seed", type=int, default=30000)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--guidance-scale", type=float, default=9.0)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--width", type=int, default=432)
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--enable-model-cpu-offload", action="store_true")
    parser.add_argument("--enable-sequential-cpu-offload", action="store_true")
    parser.add_argument("--vae-slicing", action="store_true")
    parser.add_argument("--limit-items", type=int)
    parser.add_argument("--seeds-per-item", type=int, default=1)
    parser.add_argument("--variant-set", choices=sorted(VARIANT_SETS), default="all")
    parser.add_argument("--item-indices", default="")
    parser.add_argument("--prompt-template", choices=PROMPT_TEMPLATES, default="legacy")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.limit_items is not None and args.limit_items <= 0:
        parser.error("--limit-items must be positive")
    if args.seeds_per_item <= 0:
        parser.error("--seeds-per-item must be positive")
    if args.steps <= 0:
        parser.error("--steps must be positive")
    if args.num_frames <= 0:
        parser.error("--num-frames must be positive")
    if args.fps <= 0:
        parser.error("--fps must be positive")
    if args.height <= 0 or args.width <= 0:
        parser.error("--height and --width must be positive")
    try:
        parse_item_indices(str(args.item_indices))
    except ValueError:
        parser.error("--item-indices must be a comma-separated list of integers")

    source_manifest = read_json(args.probe_manifest)
    probe_items = source_manifest.get("items")
    if not isinstance(probe_items, list):
        parser.exit(2, f"{args.probe_manifest}: missing list field 'items'\n")
    probe_items = select_probe_items(probe_items, args)
    if args.limit_items is not None:
        probe_items = probe_items[: args.limit_items]
    rows = build_items(args, probe_items)
    if not args.dry_run:
        generate_counterfactual_videos(args, rows)
    out = write_manifest(args, source_manifest=source_manifest, rows=rows)
    print(f"C0 counterfactual grid manifest written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
