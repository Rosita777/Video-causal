#!/usr/bin/env python3
"""VLM prelabeling for clean-source and control review frame strips."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import re
import urllib.request
from pathlib import Path
from typing import Any


RESPONSE_SCHEMA = {
    "target_visible": ["yes", "no", "partial"],
    "effect_visible": ["yes", "no", "partial"],
    "temporal_order_clear": ["yes", "no", "partial"],
    "effect_depends_on_target": ["yes", "no", "partial"],
    "video_quality": ["yes", "no"],
    "clean_source_valid": ["yes", "no"],
    "control_valid": ["yes", "no"],
    "control_type": ["no_cause", "effect_only", "alternative_cause", "unknown"],
    "confidence": "float in [0.0, 1.0]",
    "reason": "short visual evidence",
}

PREDICTION_FIELDS = [
    "sample_id",
    "pair_id",
    "task",
    "target_concept",
    "expected_effect",
    "target_visible",
    "effect_visible",
    "temporal_order_clear",
    "effect_depends_on_target",
    "video_quality",
    "clean_source_valid",
    "control_valid",
    "control_type",
    "confidence",
    "reason",
]

FLAG3 = {"yes", "no", "partial"}
FLAG2 = {"yes", "no"}
CONTROL_TYPES = {"no_cause", "effect_only", "alternative_cause", "unknown"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: missing header")
        return list(reader)


def strip_path_for(row: dict[str, str], frame_strip_dir: Path, row_index: int) -> Path:
    prompt_id = row.get("prompt_id", f"case_{row_index:03d}")
    candidates = sorted(frame_strip_dir.glob(f"{row_index:03d}_*.jpg"))
    if candidates:
        return candidates[0]
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", prompt_id).strip("-")
    return frame_strip_dir / f"{row_index:03d}_{slug}.jpg"


def infer_control_type(pair_id: str, prompt: str) -> str:
    pair_id_lower = pair_id.lower()
    if pair_id_lower.endswith("__no_cause"):
        return "no_cause"
    if pair_id_lower.endswith("__effect_only"):
        return "effect_only"
    if pair_id_lower.endswith("__alternative_cause"):
        return "alternative_cause"
    prompt_lower = prompt.lower()
    if "alternative" in prompt_lower or "similar footprint" in prompt_lower:
        return "alternative_cause"
    if "with no" in prompt_lower and "showing" in prompt_lower:
        return "effect_only"
    return "unknown"


def prompt_for(row: dict[str, str], task: str, control_type: str) -> str:
    common = [
        "You are labeling a 5-frame contact sheet sampled from a generated video.",
        "Inspect frames from left to right in temporal order.",
        "Do not infer from the text prompt alone; answer from visible evidence in the image.",
        f"Task: {task}",
        f"Target concept: {row.get('target_concept', '')}",
        f"Expected causal footprint/effect: {row.get('expected_effect', '')}",
        f"Prompt: {row.get('prompt', '')}",
    ]
    if task == "clean":
        task_lines = [
            "For clean-source screening, mark clean_source_valid=yes only if the target is visible, the expected effect is visible, temporal order is at least partially clear, the effect appears caused by the target, and video quality is judgeable.",
            "Use clean_source_valid=no if the target is absent, the effect is absent, the event is incoherent, or the video is too poor to judge.",
        ]
    else:
        task_lines = [
            f"Expected control type from metadata: {control_type}.",
            "For no_cause controls, control_valid=yes means neither the target nor the expected footprint is visibly present.",
            "For effect_only controls, control_valid=yes means the expected footprint is visible while the target cause is absent and no clear visible cause explains it.",
            "For alternative_cause controls, control_valid=yes means the expected footprint is visible, the target is absent, and a different visible cause plausibly explains the footprint.",
        ]
    return "\n".join(
        [
            *common,
            *task_lines,
            "Return only valid JSON with exactly these keys:",
            json.dumps(RESPONSE_SCHEMA, ensure_ascii=False),
        ]
    )


def image_data_url(path: Path) -> str:
    suffix = path.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def payload_for(row: dict[str, str], *, row_index: int, frame_strip_dir: Path, task: str) -> dict[str, Any]:
    strip_path = strip_path_for(row, frame_strip_dir, row_index)
    control_type = infer_control_type(row.get("pair_id", ""), row.get("prompt", ""))
    return {
        "sample_id": row.get("prompt_id", f"case_{row_index:03d}"),
        "pair_id": row.get("pair_id", ""),
        "task": task,
        "control_type_hint": control_type,
        "image_path": str(strip_path),
        "image_available": strip_path.exists(),
        "target_concept": row.get("target_concept", ""),
        "expected_effect": row.get("expected_effect", ""),
        "prompt": prompt_for(row, task, control_type),
        "response_schema": RESPONSE_SCHEMA,
    }


def request_payload_for(payload: dict[str, Any], *, model: str, temperature: float, max_tokens: int) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": payload["prompt"]},
                    {"type": "image_url", "image_url": {"url": image_data_url(Path(payload["image_path"]))}},
                ],
            }
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }


def urllib_transport(url: str, api_key: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": "Bearer " + api_key, "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def parse_model_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    fence = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        stripped = fence.group(1).strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end >= start:
        stripped = stripped[start : end + 1]
    return json.loads(stripped)


def normalize_flag(value: Any) -> str:
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return str(value).strip().lower()


def normalize_confidence(value: Any) -> str:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    return f"{confidence:.4f}"


def clean_valid_from_atomic(pred: dict[str, str]) -> str:
    if pred["video_quality"] != "yes":
        return "no"
    return "yes" if (
        pred["target_visible"] in {"yes", "partial"}
        and pred["effect_visible"] in {"yes", "partial"}
        and pred["temporal_order_clear"] in {"yes", "partial"}
        and pred["effect_depends_on_target"] in {"yes", "partial"}
    ) else "no"


def normalize_prediction(parsed: dict[str, Any], *, task: str) -> dict[str, str]:
    pred = {
        "target_visible": normalize_flag(parsed.get("target_visible", "")),
        "effect_visible": normalize_flag(parsed.get("effect_visible", "")),
        "temporal_order_clear": normalize_flag(parsed.get("temporal_order_clear", "")),
        "effect_depends_on_target": normalize_flag(parsed.get("effect_depends_on_target", "")),
        "video_quality": normalize_flag(parsed.get("video_quality", parsed.get("quality_ok", ""))),
        "control_type": normalize_flag(parsed.get("control_type", "unknown")),
        "confidence": normalize_confidence(parsed.get("confidence", 0.0)),
        "reason": str(parsed.get("reason", "")),
    }
    for field in ["target_visible", "effect_visible", "temporal_order_clear", "effect_depends_on_target"]:
        if pred[field] not in FLAG3:
            raise ValueError(f"invalid {field}: {pred[field]}")
    if pred["video_quality"] not in FLAG2:
        raise ValueError(f"invalid video_quality: {pred['video_quality']}")
    if pred["control_type"] not in CONTROL_TYPES:
        pred["control_type"] = "unknown"
    if task == "clean":
        clean_valid = normalize_flag(parsed.get("clean_source_valid", ""))
        pred["clean_source_valid"] = clean_valid if clean_valid in FLAG2 else clean_valid_from_atomic(pred)
        pred["control_valid"] = ""
    else:
        control_valid = normalize_flag(parsed.get("control_valid", ""))
        pred["control_valid"] = control_valid if control_valid in FLAG2 else "no"
        pred["clean_source_valid"] = ""
    return pred


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""), encoding="utf-8")


def write_predictions(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PREDICTION_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def load_api_config(args: argparse.Namespace) -> tuple[str, str]:
    url = args.api_url or os.environ.get("OPENAI_BASE_URL", "").rstrip("/")
    key = args.api_key or os.environ.get("OPENAI_API_KEY", "")
    if not url:
        raise SystemExit("missing API URL; pass --api-url or set OPENAI_BASE_URL")
    if not key:
        raise SystemExit("missing API key; pass --api-key or set OPENAI_API_KEY")
    return url.rstrip("/") + "/chat/completions", key


def extract_content(response: dict[str, Any]) -> str:
    return str(response["choices"][0]["message"]["content"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--review-csv", type=Path, required=True)
    parser.add_argument("--frame-strip-dir", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--raw-output", type=Path)
    parser.add_argument("--task", choices=["clean", "control"], required=True)
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--api-url")
    parser.add_argument("--api-key")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=500)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")

    rows = read_csv(args.review_csv)
    if args.limit is not None:
        rows = rows[: args.limit]
    payloads = [payload_for(row, row_index=index, frame_strip_dir=args.frame_strip_dir, task=args.task) for index, row in enumerate(rows)]
    payloads = [payload for payload in payloads if payload["image_available"]]

    if args.dry_run:
        write_jsonl(args.output_jsonl, payloads)
        print(f"Wrote {len(payloads)} dry-run clean/control payloads to {args.output_jsonl}")
        return 0

    api_url, api_key = load_api_config(args)
    predictions: list[dict[str, str]] = []
    raw_rows: list[dict[str, Any]] = []
    for payload in payloads:
        try:
            request_payload = request_payload_for(
                payload,
                model=args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            response = urllib_transport(api_url, api_key, request_payload, args.timeout)
            parsed = parse_model_json(extract_content(response))
            pred = normalize_prediction(parsed, task=args.task)
            output = {
                "sample_id": str(payload["sample_id"]),
                "pair_id": str(payload["pair_id"]),
                "task": str(payload["task"]),
                "target_concept": str(payload["target_concept"]),
                "expected_effect": str(payload["expected_effect"]),
                **pred,
            }
            predictions.append(output)
            raw_rows.append({"payload": payload, "response": response, "parsed": parsed})
        except Exception as exc:
            if not args.continue_on_error:
                raise
            predictions.append(
                {
                    "sample_id": str(payload["sample_id"]),
                    "pair_id": str(payload["pair_id"]),
                    "task": str(payload["task"]),
                    "target_concept": str(payload["target_concept"]),
                    "expected_effect": str(payload["expected_effect"]),
                    "target_visible": "",
                    "effect_visible": "",
                    "temporal_order_clear": "",
                    "effect_depends_on_target": "",
                    "video_quality": "no",
                    "clean_source_valid": "no" if args.task == "clean" else "",
                    "control_valid": "no" if args.task == "control" else "",
                    "control_type": "unknown",
                    "confidence": "0.0000",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
            raw_rows.append({"payload": payload, "error": f"{type(exc).__name__}: {exc}"})

    if args.predictions is None:
        raise SystemExit("--predictions is required when not using --dry-run")
    write_predictions(args.predictions, predictions)
    if args.raw_output:
        write_jsonl(args.raw_output, raw_rows)
    print(f"Wrote {len(predictions)} clean/control predictions to {args.predictions}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
