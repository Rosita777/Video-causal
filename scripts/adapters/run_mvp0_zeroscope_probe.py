#!/usr/bin/env python3
"""Plan ZeroScope MVP-0 causal-chain probe conditions."""

from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from adapters.zeroscope_adapter_common import encode_cfg, load_zeroscope_pipe  # noqa: E402


BASELINE = "mvp0_causal_chain_probe"
CONDITIONS = [
    "target_negative",
    "target_footprint_negative",
    "monolithic_counterfactual",
    "cause_steering",
    "mechanism_steering",
    "footprint_steering",
    "full_chain_steering",
    "random_direction",
    "orthogonal_semantic",
]

ORTHOGONAL_SEMANTIC_PAIR = {
    "positive": "A realistic video with birds flying across the sky.",
    "negative": "A realistic video with no birds in the sky.",
}


def slugify(text: str, max_length: int = 72) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    if not slug:
        return "prompt"
    return slug[:max_length].rstrip("-") or "prompt"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


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
        "steering_alpha": args.alpha,
        "timestep_window": list(args.timestep_window),
    }


def parse_timestep_window(value: str) -> tuple[int, int]:
    parts = value.split(":", 1)
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("expected START:END")
    try:
        start = int(parts[0])
        end = int(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("START and END must be integers") from exc
    if start < 0 or end < start:
        raise argparse.ArgumentTypeError("expected 0 <= START <= END")
    return start, end


def condition_prompt(item: dict, condition: str) -> tuple[str, str]:
    target = str(item["target_concept"])
    footprint = str(item["causal_footprint"])
    source_prompt = str(item.get("generation_prompt") or item["source_prompt"])
    if condition == "target_negative":
        return source_prompt, target
    if condition == "target_footprint_negative":
        return source_prompt, f"{target}, {footprint}"
    if condition == "monolithic_counterfactual":
        return str(item["counterfactual_prompt"]), ""
    return source_prompt, f"{target}, {footprint}"


def steering_contract(item: dict, condition: str) -> dict[str, object]:
    if condition == "cause_steering":
        links = ["cause"]
    elif condition == "mechanism_steering":
        links = ["mechanism"]
    elif condition == "footprint_steering":
        links = ["footprint"]
    elif condition == "full_chain_steering":
        links = ["cause", "mechanism", "footprint"]
    elif condition == "random_direction":
        links = ["random"]
    elif condition == "orthogonal_semantic":
        links = ["orthogonal_semantic"]
    else:
        links = []
    minimal_pairs = dict(item.get("minimal_pairs", {}))
    control_type = ""
    control_reference = ""
    if condition == "random_direction":
        control_type = "gaussian_norm_matched"
        control_reference = "footprint"
    if condition == "orthogonal_semantic":
        control_type = "unrelated_semantic_direction_norm_matched"
        control_reference = "footprint"
        if "orthogonal_semantic" not in minimal_pairs:
            minimal_pairs["orthogonal_semantic"] = dict(ORTHOGONAL_SEMANTIC_PAIR)
    return {
        "enabled": bool(links),
        "links": links,
        "minimal_pairs": minimal_pairs,
        "control_type": control_type,
        "control_reference": control_reference,
    }


def _sub_scaled(base, direction, alpha: float):
    try:
        return base - alpha * direction
    except TypeError:
        return [left - alpha * right for left, right in zip(base, direction)]


def _diff(left, right):
    try:
        return left - right
    except TypeError:
        return [a - b for a, b in zip(left, right)]


def normalize_minimal_pair_value(value):
    if isinstance(value, list):
        return value
    return [value]


def normalize_encoded_pair_value(value):
    if isinstance(value, list):
        return value
    return [value]


def _zero_like(value):
    if isinstance(value, list):
        return [part * 0 for part in value]
    try:
        return value * 0
    except TypeError:
        return [part * 0 for part in value]


def _add(left, right):
    if isinstance(left, list):
        return [a + b for a, b in zip(left, right)]
    try:
        return left + right
    except TypeError:
        return [a + b for a, b in zip(left, right)]


def _scale(value, factor: float):
    if isinstance(value, list):
        return [part * factor for part in value]
    try:
        return value * factor
    except TypeError:
        return [part * factor for part in value]


def average_pair_predictions(predictions):
    directions = [_diff(pair["positive"], pair["negative"]) for pair in predictions]
    if not directions:
        return None
    total = directions[0]
    for direction in directions[1:]:
        total = _add(total, direction)
    averaged = _scale(total, 1.0 / len(directions))
    return {"positive": averaged, "negative": _zero_like(averaged)}


def _norm(value):
    try:
        return value.norm()
    except AttributeError:
        return sum(part * part for part in value) ** 0.5


def _scale_to_norm(value, target_norm):
    current_norm = _norm(value)
    try:
        current = float(current_norm.item())
    except AttributeError:
        current = float(current_norm)
    if current == 0.0:
        return value
    scale = target_norm / current
    try:
        return value * scale
    except TypeError:
        return [part * scale for part in value]


def _random_like(torch_module, reference, seed: int):
    if torch_module is not None and hasattr(torch_module, "randn_like"):
        generator = None
        if hasattr(torch_module, "Generator") and hasattr(reference, "device"):
            generator = torch_module.Generator(device=reference.device).manual_seed(seed)
        try:
            return torch_module.randn_like(reference, generator=generator)
        except TypeError:
            return torch_module.randn_like(reference)

    import random

    rng = random.Random(seed)
    return [rng.gauss(0.0, 1.0) for _ in reference]


def synthesize_random_control_prediction(
    torch_module,
    link_predictions: dict[str, dict[str, object]],
    row: dict[str, object],
    *,
    step_index: int,
) -> None:
    links = row.get("steering", {}).get("links", [])
    if "random" not in links:
        return
    reference = link_predictions.get("__random_reference__")
    if not reference:
        return
    reference_direction = _diff(reference["positive"], reference["negative"])
    try:
        target_norm = float(_norm(reference_direction).item())
    except AttributeError:
        target_norm = float(_norm(reference_direction))
    random_direction = _scale_to_norm(
        _random_like(torch_module, reference_direction, int(row.get("seed", 0)) + step_index),
        target_norm,
    )
    if isinstance(random_direction, list):
        negative = [0.0 for _ in random_direction]
    else:
        negative = reference["negative"] * 0
    link_predictions["random"] = {
        "positive": random_direction,
        "negative": negative,
    }


def synthesize_orthogonal_control_prediction(
    torch_module,
    link_predictions: dict[str, dict[str, object]],
    row: dict[str, object],
    *,
    step_index: int,
) -> None:
    links = row.get("steering", {}).get("links", [])
    if "orthogonal_semantic" not in links:
        return
    reference = link_predictions.get("__orthogonal_reference__")
    orthogonal = link_predictions.get("orthogonal_semantic")
    if not reference or not orthogonal:
        return
    reference_direction = _diff(reference["positive"], reference["negative"])
    orthogonal_direction = _diff(orthogonal["positive"], orthogonal["negative"])
    try:
        target_norm = float(_norm(reference_direction).item())
    except AttributeError:
        target_norm = float(_norm(reference_direction))
    scaled_direction = _scale_to_norm(orthogonal_direction, target_norm)
    link_predictions["orthogonal_semantic"] = {
        "positive": scaled_direction,
        "negative": _zero_like(scaled_direction),
    }


def apply_steering_residual(
    main_residual,
    link_predictions: dict[str, dict[str, object]],
    row: dict[str, object],
    *,
    step_index: int,
    alpha: float,
    timestep_window: tuple[int, int],
):
    start, end = timestep_window
    links = list(dict.fromkeys(row.get("steering", {}).get("links", [])))
    if not links or not (start <= step_index <= end):
        return main_residual

    steered = main_residual
    for link in links:
        predictions = link_predictions.get(link)
        if not predictions:
            continue
        direction = _diff(predictions["positive"], predictions["negative"])
        steered = _sub_scaled(steered, direction, alpha)
    return steered


def _guided_noise_pred(pipe, latent_model_input, timestep, embeds, cross_attention_kwargs=None):
    return pipe.unet(
        latent_model_input,
        timestep,
        encoder_hidden_states=embeds,
        cross_attention_kwargs=cross_attention_kwargs,
        return_dict=False,
    )[0]


def apply_cfg(noise_pred, guidance_scale: float):
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    return noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)


def _reshape_for_scheduler(tensor, latents):
    bsz, channel, frames, width, height = latents.shape
    return tensor.permute(0, 2, 1, 3, 4).reshape(bsz * frames, channel, width, height)


def run_steered_pipeline(
    pipe,
    torch_module,
    *,
    row: dict[str, object],
    prompt_embeds,
    negative_prompt_embeds,
    link_embeds: dict[str, dict[str, object]],
    generator,
    steps: int,
    num_frames: int,
    guidance_scale: float,
    height: int,
    width: int,
    alpha: float,
    timestep_window: tuple[int, int],
    output_type: str = "np",
):
    device = pipe._execution_device
    do_classifier_free_guidance = guidance_scale > 1.0
    main_embeds = (
        torch_module.cat([negative_prompt_embeds, prompt_embeds])
        if do_classifier_free_guidance
        else prompt_embeds
    )

    pipe.scheduler.set_timesteps(steps, device=device)
    timesteps = pipe.scheduler.timesteps
    latents = pipe.prepare_latents(
        1,
        pipe.unet.config.in_channels,
        num_frames,
        height,
        width,
        getattr(prompt_embeds, "dtype", None),
        device,
        generator,
        None,
    )
    extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator, 0.0)

    with torch_module.no_grad():
        with pipe.progress_bar(total=steps) as progress_bar:
            for step_index, timestep in enumerate(timesteps):
                latent_model_input = (
                    torch_module.cat([latents] * 2) if do_classifier_free_guidance else latents
                )
                latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, timestep)
                noise_pred = _guided_noise_pred(pipe, latent_model_input, timestep, main_embeds)
                if do_classifier_free_guidance:
                    noise_pred = apply_cfg(noise_pred, guidance_scale)

                link_predictions = {}
                for link, encoded_pairs in link_embeds.items():
                    pair_predictions = []
                    for pair in normalize_encoded_pair_value(encoded_pairs):
                        prediction_pair = {}
                        for side, embeds in pair.items():
                            pred = _guided_noise_pred(pipe, latent_model_input, timestep, embeds)
                            if do_classifier_free_guidance:
                                pred = apply_cfg(pred, guidance_scale)
                            prediction_pair[side] = pred
                        pair_predictions.append(prediction_pair)
                    averaged = average_pair_predictions(pair_predictions)
                    if averaged is not None:
                        link_predictions[link] = averaged

                synthesize_random_control_prediction(
                    torch_module,
                    link_predictions,
                    row,
                    step_index=step_index,
                )
                synthesize_orthogonal_control_prediction(
                    torch_module,
                    link_predictions,
                    row,
                    step_index=step_index,
                )

                noise_pred = apply_steering_residual(
                    noise_pred,
                    link_predictions,
                    row,
                    step_index=step_index,
                    alpha=alpha,
                    timestep_window=timestep_window,
                )

                bsz, channel, frames, latent_width, latent_height = latents.shape
                scheduler_latents = latents.permute(0, 2, 1, 3, 4).reshape(
                    bsz * frames, channel, latent_width, latent_height
                )
                scheduler_noise_pred = noise_pred.permute(0, 2, 1, 3, 4).reshape(
                    bsz * frames, channel, latent_width, latent_height
                )
                latents = pipe.scheduler.step(
                    scheduler_noise_pred,
                    timestep,
                    scheduler_latents,
                    **extra_step_kwargs,
                ).prev_sample
                latents = latents[None, :].reshape(
                    bsz, frames, channel, latent_width, latent_height
                ).permute(0, 2, 1, 3, 4)
                progress_bar.update()

        if output_type == "latent":
            video = latents
        else:
            video_tensor = pipe.decode_latents(latents)
            video = pipe.video_processor.postprocess_video(video=video_tensor, output_type=output_type)
    pipe.maybe_free_model_hooks()
    return video


def encode_pair_embeds(pipe, torch_module, selected_device: str, row: dict[str, object]):
    pair_embeds: dict[str, list[dict[str, object]]] = {}
    minimal_pairs = row.get("steering", {}).get("minimal_pairs", {})
    for link in row.get("steering", {}).get("links", []):
        pair_key = row.get("steering", {}).get("control_reference", "") if link == "random" else link
        pair_value = minimal_pairs.get(pair_key)
        if not pair_value:
            continue
        embed_key = "__random_reference__" if link == "random" else link
        pair_embeds[embed_key] = []
        for pair in normalize_minimal_pair_value(pair_value):
            encoded_pair = {}
            for side in ["positive", "negative"]:
                prompt_embeds, negative_prompt_embeds = encode_cfg(
                    pipe,
                    torch_module,
                    prompt=str(pair[side]),
                    negative_prompt=str(row.get("negative_prompt", "")),
                    device=selected_device,
                )
                encoded_pair[side] = (
                    torch_module.cat([negative_prompt_embeds, prompt_embeds])
                    if float(row.get("steering", {}).get("alpha", 0.0)) >= 0
                    else prompt_embeds
                )
            pair_embeds[embed_key].append(encoded_pair)
    if "orthogonal_semantic" in row.get("steering", {}).get("links", []):
        reference_key = row.get("steering", {}).get("control_reference", "")
        reference_value = minimal_pairs.get(reference_key)
        if reference_value:
            pair_embeds["__orthogonal_reference__"] = []
            for pair in normalize_minimal_pair_value(reference_value):
                encoded_pair = {}
                for side in ["positive", "negative"]:
                    prompt_embeds, negative_prompt_embeds = encode_cfg(
                        pipe,
                        torch_module,
                        prompt=str(pair[side]),
                        negative_prompt=str(row.get("negative_prompt", "")),
                        device=selected_device,
                    )
                    encoded_pair[side] = (
                        torch_module.cat([negative_prompt_embeds, prompt_embeds])
                        if float(row.get("steering", {}).get("alpha", 0.0)) >= 0
                        else prompt_embeds
                    )
                pair_embeds["__orthogonal_reference__"].append(encoded_pair)
    return pair_embeds


def build_items(args: argparse.Namespace, probe_items: list[dict]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for probe_item in probe_items:
        for condition in args.condition:
            prompt, negative_prompt = condition_prompt(probe_item, condition)
            seed = args.seed + int(probe_item["probe_index"])
            slug = slugify(str(probe_item["pair_id"]))
            rows.append(
                {
                    "probe_index": probe_item["probe_index"],
                    "pair_id": probe_item["pair_id"],
                    "slice_index": probe_item["slice_index"],
                    "mechanism_type": probe_item["mechanism_type"],
                    "condition": condition,
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "source_prompt": str(probe_item.get("source_prompt", "")),
                    "generation_prompt": str(
                        probe_item.get("generation_prompt") or probe_item.get("source_prompt", "")
                    ),
                    "target_concept": probe_item["target_concept"],
                    "causal_footprint": probe_item["causal_footprint"],
                    "seed": seed,
                    "video_path": str(
                        args.output_dir
                        / "videos"
                        / f"{int(probe_item['probe_index']):03d}_{slug}_{condition}_seed{seed}.mp4"
                    ),
                    "steering": steering_contract(probe_item, condition),
                }
            )
            rows[-1]["steering"]["alpha"] = args.alpha
            rows[-1]["steering"]["timestep_window"] = list(args.timestep_window)
    return rows


def _token_count(tokenizer, text: str) -> int:
    return len(tokenizer(text, truncation=False).input_ids)


def audit_prompt_lengths(
    rows: list[dict[str, object]],
    tokenizer,
    *,
    limit: int,
    strict: bool,
) -> list[dict[str, object]]:
    violations: list[dict[str, object]] = []
    for row in rows:
        prompts = {
            "prompt": str(row.get("prompt", "")),
            "negative_prompt": str(row.get("negative_prompt", "")),
        }
        for link, pair in row.get("steering", {}).get("minimal_pairs", {}).items():
            if link not in row.get("steering", {}).get("links", []):
                continue
            for pair_index, normalized_pair in enumerate(normalize_minimal_pair_value(pair)):
                for side, prompt in normalized_pair.items():
                    prompts[f"{link}[{pair_index}].{side}"] = str(prompt)
        for field, prompt in prompts.items():
            count = _token_count(tokenizer, prompt)
            if count > limit:
                violations.append(
                    {
                        "pair_id": row.get("pair_id", ""),
                        "condition": row.get("condition", ""),
                        "field": field,
                        "token_count": count,
                        "limit": limit,
                    }
                )
    if violations:
        lines = [
            f"{violation['pair_id']}/{violation['condition']} {violation['field']} "
            f"has {violation['token_count']} tokens > {violation['limit']}"
            for violation in violations
        ]
        message = "Prompt length audit failed:\n" + "\n".join(lines)
        if strict:
            raise ValueError(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)
    return violations


def load_tokenizer(model_path: str):
    from transformers import CLIPTokenizer

    return CLIPTokenizer.from_pretrained(Path(model_path) / "tokenizer")


def write_manifest(args: argparse.Namespace, probe_manifest: dict, rows: list[dict[str, object]]) -> Path:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline": BASELINE,
        "model": args.model,
        "dry_run": args.dry_run,
        "probe_manifest": str(args.probe_manifest),
        "source_probe_name": probe_manifest.get("probe_name", ""),
        "conditions": args.condition,
        "generation": generation_config(args),
        "items": rows,
    }
    out = args.output_dir / "generation_manifest.json"
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


def generate_probe_videos(args: argparse.Namespace, rows: list[dict[str, object]]) -> None:
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
        link_embeds = encode_pair_embeds(pipe, torch_module, selected_device, row)
        generator = torch_module.Generator(device=generator_device).manual_seed(int(row["seed"]))
        frames = run_steered_pipeline(
            pipe,
            torch_module,
            row=row,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            link_embeds=link_embeds,
            generator=generator,
            steps=args.steps,
            num_frames=args.num_frames,
            guidance_scale=args.guidance_scale,
            height=args.height,
            width=args.width,
            alpha=args.alpha,
            timestep_window=args.timestep_window,
        )
        export_to_video(frames[0], str(video_path), fps=args.fps)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--probe-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="models/zeroscope_v2_576w")
    parser.add_argument("--seed", type=int, default=15000)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=9.0)
    parser.add_argument("--num-frames", type=int, default=24)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--width", type=int, default=576)
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--enable-model-cpu-offload", action="store_true")
    parser.add_argument("--enable-sequential-cpu-offload", action="store_true")
    parser.add_argument("--vae-slicing", action="store_true")
    parser.add_argument("--condition", action="append", choices=CONDITIONS, default=[])
    parser.add_argument("--limit-items", type=int)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--timestep-window", type=parse_timestep_window, default=(4, 14))
    parser.add_argument("--strict-prompt-length", action="store_true")
    parser.add_argument("--prompt-token-limit", type=int, default=77)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.condition:
        args.condition = CONDITIONS
    if args.limit_items is not None and args.limit_items <= 0:
        parser.error("--limit-items must be positive")
    if args.steps <= 0:
        parser.error("--steps must be positive")
    if args.num_frames <= 0:
        parser.error("--num-frames must be positive")
    if args.fps <= 0:
        parser.error("--fps must be positive")
    if args.prompt_token_limit <= 0:
        parser.error("--prompt-token-limit must be positive")
    probe_manifest = read_json(args.probe_manifest)
    probe_items = probe_manifest.get("items")
    if not isinstance(probe_items, list):
        parser.exit(2, f"{args.probe_manifest}: missing list field 'items'\n")
    if args.limit_items is not None:
        probe_items = probe_items[: args.limit_items]
    rows = build_items(args, probe_items)
    try:
        tokenizer = load_tokenizer(args.model)
    except ImportError as exc:
        if args.strict_prompt_length:
            parser.exit(2, f"Prompt length audit requires transformers: {exc}\n")
        warnings.warn(
            f"Skipping prompt length audit because transformers is unavailable: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
    else:
        try:
            audit_prompt_lengths(
                rows,
                tokenizer,
                limit=args.prompt_token_limit,
                strict=args.strict_prompt_length,
            )
        except ValueError as exc:
            parser.exit(2, f"{exc}\n")
    if not args.dry_run:
        generate_probe_videos(args, rows)
    out = write_manifest(args, probe_manifest, rows)
    if args.dry_run:
        print(f"Dry-run MVP-0 probe manifest written: {out}")
    else:
        print(f"MVP-0 probe manifest written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
