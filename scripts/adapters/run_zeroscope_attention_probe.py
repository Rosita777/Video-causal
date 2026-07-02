#!/usr/bin/env python3
"""Record ZeroScope cross-attention dependency summaries for causal probes."""

from __future__ import annotations

import csv
import json
import argparse
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence


SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from adapters.zeroscope_adapter_common import encode_cfg, load_zeroscope_pipe  # noqa: E402


PROBE_NAME = "zeroscope_attention_dependency_probe"
B2_CONDITIONS = [
    "baseline",
    "target_mask",
    "footprint_mask",
    "chain_mask",
    "comparison_token_mask",
    "random_token_mask",
]
SPECIAL_TOKENS = {
    "<|startoftext|>",
    "<|endoftext|>",
    "<s>",
    "</s>",
    "[bos]",
    "[eos]",
    "[cls]",
    "[sep]",
    "[pad]",
}


def normalize_token_text(text: str) -> str:
    """Normalize tokenizer fragments into comparable lowercase words."""
    token = str(text).strip().lower()
    if token in SPECIAL_TOKENS:
        return ""
    token = token.replace("</w>", "")
    token = token.lstrip("Ġġ▁")
    token = token.strip()
    if token in SPECIAL_TOKENS:
        return ""
    return token


def _normalized_token_positions(tokens: Sequence[str]) -> list[tuple[int, str]]:
    return [
        (index, normalized)
        for index, token in enumerate(tokens)
        if (normalized := normalize_token_text(token))
    ]


def find_token_indices(tokens: Sequence[str], phrase: str) -> list[int]:
    """Return original token indices for a normalized phrase span.

    The matcher is character based instead of word-window based so CLIP/BPE
    fragments such as ``ripp`` + ``les</w>`` still map back to the full phrase.
    Punctuation-only fragments are allowed as separators but not returned.
    """
    phrase_key = _span_key(phrase)
    if not phrase_key:
        raise ValueError("could not find token span for empty phrase")

    searchable_chars: list[str] = []
    char_to_index: list[int] = []
    word_token_indices: set[int] = set()
    for index, token in enumerate(tokens):
        normalized = normalize_token_text(token)
        if not normalized:
            continue
        for char in normalized:
            if char.isalnum():
                searchable_chars.append(char)
                char_to_index.append(index)
                word_token_indices.add(index)

    searchable = "".join(searchable_chars)
    start = searchable.find(phrase_key)
    if start >= 0:
        end = start + len(phrase_key)
        matched = []
        for index in char_to_index[start:end]:
            if index in word_token_indices and index not in matched:
                matched.append(index)
        return matched

    visible_tokens = [token for _, token in _normalized_token_positions(tokens)]
    raise ValueError(
        f"could not find token span for {phrase!r}; normalized tokens={visible_tokens!r}"
    )


def _span_key(text: str) -> str:
    return "".join(char for char in str(text).lower() if char.isalnum())


def comparison_token_indices(
    tokens: Sequence[str],
    *,
    selected_indices: Iterable[int],
    count: int,
) -> list[int]:
    """Pick non-special comparison token indices outside the selected span."""
    selected = set(selected_indices)
    indices: list[int] = []
    for index, token in enumerate(tokens):
        if index in selected:
            continue
        if not normalize_token_text(token):
            continue
        indices.append(index)
        if len(indices) >= count:
            break
    return indices


def random_token_indices(
    tokens: Sequence[str],
    *,
    excluded_indices: Iterable[int],
    count: int,
    seed: int,
) -> list[int]:
    excluded = set(excluded_indices)
    candidates = [
        index
        for index, token in enumerate(tokens)
        if index not in excluded and normalize_token_text(token)
    ]
    if len(candidates) < count:
        raise ValueError(
            f"not enough random-token candidates: need {count}, have {len(candidates)}"
        )
    rng = random.Random(seed)
    return sorted(rng.sample(candidates, count))


class AttentionSummaryRecorder:
    """Collect compact attention mass summaries for selected key-token spans."""

    def __init__(
        self,
        *,
        target_indices: Sequence[int],
        footprint_indices: Sequence[int],
        comparison_indices: Sequence[int],
    ) -> None:
        self.target_indices = list(target_indices)
        self.footprint_indices = list(footprint_indices)
        self.comparison_indices = list(comparison_indices)
        self.records: list[dict[str, object]] = []

    def record(
        self,
        *,
        module_name: str,
        step_index: int,
        attention_probs,
        query_tokens: int,
        key_tokens: int,
    ) -> None:
        record = {
            "module_name": module_name,
            "step_index": int(step_index),
            "query_tokens": int(query_tokens),
            "key_tokens": int(key_tokens),
            "target_mass": self._mean_attention_mass(attention_probs, self.target_indices),
            "footprint_mass": self._mean_attention_mass(attention_probs, self.footprint_indices),
            "comparison_mass": self._mean_attention_mass(
                attention_probs, self.comparison_indices
            ),
            "all_text_mass": self._mean_attention_mass(
                attention_probs, range(int(key_tokens))
            ),
            "target_indices": list(self.target_indices),
            "footprint_indices": list(self.footprint_indices),
            "comparison_indices": list(self.comparison_indices),
        }
        self.records.append(record)

    def write_jsonl(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for record in self.records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def write_csv(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "module_name",
            "step_index",
            "query_tokens",
            "key_tokens",
            "target_mass",
            "footprint_mass",
            "comparison_mass",
            "all_text_mass",
        ]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for record in self.records:
                writer.writerow({key: record[key] for key in fieldnames})

    @staticmethod
    def _mean_attention_mass(attention_probs, indices: Iterable[int]) -> float:
        selected = list(indices)
        if not selected:
            return 0.0

        if hasattr(attention_probs, "detach"):
            valid = [
                int(index)
                for index in selected
                if 0 <= int(index) < int(attention_probs.shape[-1])
            ]
            if not valid:
                return 0.0
            index_tensor = attention_probs.new_tensor(valid).long()
            selected_probs = attention_probs.detach().float().index_select(-1, index_tensor)
            return float(selected_probs.mean().item())

        data = attention_probs

        rows = list(_attention_rows(data))
        if not rows:
            return 0.0

        total = 0.0
        count = 0
        for row in rows:
            for index in selected:
                if 0 <= index < len(row):
                    total += float(row[index])
                    count += 1
        if count == 0:
            return 0.0
        return total / count


def _attention_rows(data):
    if not isinstance(data, (list, tuple)):
        return
    if data and all(not isinstance(value, (list, tuple)) for value in data):
        yield data
        return
    for value in data:
        yield from _attention_rows(value)


def reweight_attention_columns(attention_probs, *, selected_indices: Sequence[int], scale: float):
    selected = [int(index) for index in selected_indices]
    if not selected or float(scale) == 1.0:
        return attention_probs
    reweighted = attention_probs.clone()
    valid = [index for index in selected if 0 <= index < int(reweighted.shape[-1])]
    if not valid:
        return attention_probs
    reweighted[:, :, valid] = reweighted[:, :, valid] * float(scale)
    row_sum = reweighted.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return reweighted / row_sum


class RecordingAttnProcessor:
    """Diffusers attention processor that records cross-attention probabilities.

    This mirrors the classic non-SDPA AttnProcessor path so attention
    probabilities are observable. It records compact summaries only when
    encoder_hidden_states is present, then returns the ordinary attention output.
    """

    def __init__(
        self,
        *,
        module_name: str,
        recorder: AttentionSummaryRecorder,
        step_getter,
        cfg_text_conditioned_only: bool = True,
        intervention_indices: Sequence[int] | None = None,
        intervention_scale: float = 1.0,
    ):
        self.module_name = module_name
        self.recorder = recorder
        self.step_getter = step_getter
        self.cfg_text_conditioned_only = cfg_text_conditioned_only
        self.intervention_indices = list(intervention_indices or [])
        self.intervention_scale = float(intervention_scale)

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        *args,
        **kwargs,
    ):
        residual = hidden_states

        if getattr(attn, "spatial_norm", None) is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(
            attention_mask, sequence_length, batch_size
        )

        if getattr(attn, "group_norm", None) is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        is_cross_attention = encoder_hidden_states is not None
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif getattr(attn, "norm_cross", False):
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        recorded_slice = self._text_conditioned_slice(
            attention_probs=attention_probs,
            encoder_hidden_states=encoder_hidden_states,
        )
        if is_cross_attention and self.intervention_indices and self.intervention_scale != 1.0:
            if recorded_slice is None:
                attention_probs = reweight_attention_columns(
                    attention_probs,
                    selected_indices=self.intervention_indices,
                    scale=self.intervention_scale,
                )
            else:
                attention_probs = attention_probs.clone()
                attention_probs[recorded_slice] = reweight_attention_columns(
                    attention_probs[recorded_slice],
                    selected_indices=self.intervention_indices,
                    scale=self.intervention_scale,
                )
        hidden_states = attention_probs.bmm(value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if getattr(attn, "residual_connection", False):
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / getattr(attn, "rescale_output_factor", 1.0)

        if is_cross_attention:
            recorded_probs = attention_probs
            recorded_slice = self._text_conditioned_slice(
                attention_probs=attention_probs,
                encoder_hidden_states=encoder_hidden_states,
            )
            if recorded_slice is not None:
                recorded_probs = attention_probs[recorded_slice]
            self.recorder.record(
                module_name=self.module_name,
                step_index=int(self.step_getter()),
                attention_probs=recorded_probs,
                query_tokens=int(recorded_probs.shape[-2]),
                key_tokens=int(recorded_probs.shape[-1]),
            )

        return hidden_states

    def _text_conditioned_slice(self, *, attention_probs, encoder_hidden_states):
        encoder_batch = int(encoder_hidden_states.shape[0])
        attention_batch = int(attention_probs.shape[0])
        if (
            self.cfg_text_conditioned_only
            and encoder_batch > 1
            and encoder_batch % 2 == 0
            and attention_batch % encoder_batch == 0
        ):
            heads_per_batch = attention_batch // encoder_batch
            start = (encoder_batch // 2) * heads_per_batch
            return slice(start, attention_batch)
        return None


def is_cross_attention_processor_name(name: str) -> bool:
    return ".attn2." in name or name.endswith("attn2.processor")


def install_recording_processors(
    unet,
    *,
    recorder: AttentionSummaryRecorder,
    step_getter,
    cfg_text_conditioned_only: bool = True,
    intervention_indices: Sequence[int] | None = None,
    intervention_scale: float = 1.0,
) -> int:
    processors = {}
    installed = 0
    for name, processor in unet.attn_processors.items():
        if is_cross_attention_processor_name(name):
            processors[name] = RecordingAttnProcessor(
                module_name=name,
                recorder=recorder,
                step_getter=step_getter,
                cfg_text_conditioned_only=cfg_text_conditioned_only,
                intervention_indices=intervention_indices,
                intervention_scale=intervention_scale,
            )
            installed += 1
        else:
            processors[name] = processor
    unet.set_attn_processor(processors)
    return installed


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def slugify(text: str, max_length: int = 72) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    if not slug:
        return "prompt"
    return slug[:max_length].rstrip("-") or "prompt"


def load_tokenizer(model_path: str):
    from transformers import CLIPTokenizer

    return CLIPTokenizer.from_pretrained(Path(model_path) / "tokenizer")


def prompt_tokens(tokenizer, prompt: str) -> list[str]:
    encoded = tokenizer(
        prompt,
        truncation=True,
        max_length=getattr(tokenizer, "model_max_length", 77),
    )
    return tokenizer.convert_ids_to_tokens(encoded.input_ids)


def condition_intervention_indices(
    *,
    condition: str,
    target_indices: Sequence[int],
    footprint_indices: Sequence[int],
    comparison_indices: Sequence[int],
    random_indices: Sequence[int],
) -> list[int]:
    if condition == "baseline":
        return []
    if condition == "target_mask":
        return list(target_indices)
    if condition == "footprint_mask":
        return list(footprint_indices)
    if condition == "chain_mask":
        return list(target_indices) + list(footprint_indices)
    if condition == "comparison_token_mask":
        return list(comparison_indices)[: len(target_indices) + len(footprint_indices)]
    if condition == "random_token_mask":
        return list(random_indices)
    raise ValueError(f"unknown condition: {condition}")


def build_items(args: argparse.Namespace, probe_items: Sequence[dict]) -> list[dict[str, object]]:
    tokenizer = load_tokenizer(args.model)
    rows: list[dict[str, object]] = []
    for item in probe_items:
        prompt = str(item.get("generation_prompt") or item.get("source_prompt") or item["prompt"])
        target = str(item["target_concept"])
        footprint = str(item["causal_footprint"])
        tokens = prompt_tokens(tokenizer, prompt)
        target_indices = find_token_indices(tokens, target)
        footprint_indices = find_token_indices(tokens, footprint)
        selected = set(target_indices) | set(footprint_indices)
        comparison_indices = comparison_token_indices(
            tokens,
            selected_indices=selected,
            count=args.comparison_token_count,
        )
        chain_count = len(target_indices) + len(footprint_indices)
        random_indices = random_token_indices(
            tokens,
            excluded_indices=selected,
            count=chain_count,
            seed=args.seed + int(item.get("probe_index", len(rows))),
        )
        slug = slugify(str(item.get("pair_id", item.get("probe_index", len(rows)))))
        item_seed = args.seed + int(item.get("probe_index", len(rows)))
        for condition in args.condition:
            intervention_indices = condition_intervention_indices(
                condition=condition,
                target_indices=target_indices,
                footprint_indices=footprint_indices,
                comparison_indices=comparison_indices,
                random_indices=random_indices,
            )
            row_dir = args.output_dir / f"{len(rows):03d}_{slug}_{condition}"
            rows.append(
                {
                    "probe_index": item.get("probe_index", len(rows)),
                    "pair_id": item.get("pair_id", f"item_{len(rows)}"),
                    "condition": condition,
                    "prompt": prompt,
                    "negative_prompt": str(item.get("negative_prompt", "")),
                    "target_concept": target,
                    "causal_footprint": footprint,
                    "tokens": tokens,
                    "normalized_tokens": [normalize_token_text(token) for token in tokens],
                    "target_indices": target_indices,
                    "footprint_indices": footprint_indices,
                    "comparison_indices": comparison_indices,
                    "random_indices": random_indices,
                    "intervention_indices": intervention_indices,
                    "intervention_scale": 1.0 if condition == "baseline" else args.mask_scale,
                    "seed": item_seed,
                    "video_path": str(row_dir / "sample.mp4"),
                    "attention_trace_path": str(row_dir / "attention_trace.jsonl"),
                    "attention_summary_path": str(row_dir / "attention_summary.csv"),
                }
            )
    return rows


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
        "comparison_token_count": args.comparison_token_count,
        "conditions": list(args.condition),
        "mask_scale": args.mask_scale,
        "record_cross_attention_only": True,
    }


def write_manifest(
    args: argparse.Namespace,
    *,
    source_manifest: dict,
    rows: Sequence[dict[str, object]],
) -> Path:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "probe_name": PROBE_NAME,
        "dry_run": args.dry_run,
        "model": args.model,
        "source_probe_manifest": str(args.probe_manifest),
        "source_probe_name": source_manifest.get("probe_name", ""),
        "generation": generation_config(args),
        "count": len(rows),
        "items": list(rows),
    }
    out = args.output_dir / "generation_manifest.json"
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--probe-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="models/zeroscope_v2_576w")
    parser.add_argument("--seed", type=int, default=22000)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--guidance-scale", type=float, default=9.0)
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--enable-model-cpu-offload", action="store_true")
    parser.add_argument("--enable-sequential-cpu-offload", action="store_true")
    parser.add_argument("--vae-slicing", action="store_true")
    parser.add_argument("--limit-items", type=int)
    parser.add_argument("--comparison-token-count", type=int, default=8)
    parser.add_argument("--condition", action="append", choices=B2_CONDITIONS, default=[])
    parser.add_argument("--mask-scale", type=float, default=0.0)
    parser.add_argument("--skip-video-export", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.limit_items is not None and args.limit_items <= 0:
        parser.error("--limit-items must be positive")
    if args.steps <= 0:
        parser.error("--steps must be positive")
    if args.num_frames <= 0:
        parser.error("--num-frames must be positive")
    if args.comparison_token_count <= 0:
        parser.error("--comparison-token-count must be positive")
    if not 0.0 <= args.mask_scale <= 1.0:
        parser.error("--mask-scale must be between 0 and 1")
    if not args.condition:
        args.condition = ["baseline"]

    source_manifest = read_json(args.probe_manifest)
    probe_items = source_manifest.get("items")
    if not isinstance(probe_items, list):
        parser.exit(2, f"{args.probe_manifest}: missing list field 'items'\n")
    if args.limit_items is not None:
        probe_items = probe_items[: args.limit_items]

    try:
        rows = build_items(args, probe_items)
    except ValueError as exc:
        parser.exit(2, f"{exc}\n")

    if not args.dry_run:
        generate_attention_probe_videos(args, rows)
    manifest_path = write_manifest(args, source_manifest=source_manifest, rows=rows)
    print(f"Attention dependency probe manifest written: {manifest_path}")
    return 0


def generate_attention_probe_videos(args: argparse.Namespace, rows: Sequence[dict[str, object]]) -> None:
    torch_module, export_to_video, pipe, selected_device = load_zeroscope_pipe(args)
    generator_device = (
        "cuda"
        if str(selected_device).startswith("cuda") and torch_module.cuda.is_available()
        else "cpu"
    )
    for row in rows:
        row_dir = Path(str(row["attention_trace_path"])).parent
        row_dir.mkdir(parents=True, exist_ok=True)
        recorder = AttentionSummaryRecorder(
            target_indices=list(row["target_indices"]),
            footprint_indices=list(row["footprint_indices"]),
            comparison_indices=list(row["comparison_indices"]),
        )
        step_state = {"index": -1}
        installed = install_recording_processors(
            pipe.unet,
            recorder=recorder,
            step_getter=lambda: step_state["index"],
            cfg_text_conditioned_only=args.guidance_scale > 1.0,
            intervention_indices=list(row.get("intervention_indices", [])),
            intervention_scale=float(row.get("intervention_scale", 1.0)),
        )
        if installed == 0:
            raise RuntimeError("no cross-attention processors were installed")

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
            step_state=step_state,
            output_type="np",
            decode_video=not args.skip_video_export,
        )
        recorder.write_jsonl(Path(str(row["attention_trace_path"])))
        recorder.write_csv(Path(str(row["attention_summary_path"])))
        if not args.skip_video_export and frames is not None:
            video_path = Path(str(row["video_path"]))
            video_path.parent.mkdir(parents=True, exist_ok=True)
            export_to_video(frames[0], str(video_path), fps=args.fps)


def run_attention_recording_pipeline(
    pipe,
    torch_module,
    *,
    prompt_embeds,
    negative_prompt_embeds,
    generator,
    steps: int,
    num_frames: int,
    guidance_scale: float,
    height: int,
    width: int,
    step_state: dict[str, int],
    output_type: str = "np",
    decode_video: bool = True,
):
    device = pipe._execution_device
    do_classifier_free_guidance = guidance_scale > 1.0
    prompt_condition = (
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
                step_state["index"] = step_index
                latent_model_input = (
                    torch_module.cat([latents] * 2) if do_classifier_free_guidance else latents
                )
                latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, timestep)
                noise_pred = pipe.unet(
                    latent_model_input,
                    timestep,
                    encoder_hidden_states=prompt_condition,
                    return_dict=False,
                )[0]
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
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

        if not decode_video:
            video = None
        else:
            video_tensor = pipe.decode_latents(latents)
            video = pipe.video_processor.postprocess_video(
                video=video_tensor,
                output_type=output_type,
            )
    pipe.maybe_free_model_hooks()
    return video


if __name__ == "__main__":
    raise SystemExit(main())
