#!/usr/bin/env python3
"""Build dry-run VLM request payloads from contact-sheet evaluator inputs."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import re
import urllib.request
from pathlib import Path
from typing import Any, Callable


REQUIRED_INPUT_FIELDS = {
    "output_id",
    "item_id",
    "baseline",
    "video_path",
    "sheet_path",
    "sheet_exists",
    "sheet_error",
    "target_concept",
    "expected_effect",
    "source_prompt",
}

RESPONSE_SCHEMA = {
    "target_absent": ["yes", "no", "partial"],
    "effect_visible": ["yes", "no", "partial"],
    "quality_ok": ["yes", "no"],
    "pred_label": ["strict_leakage", "borderline", "target_leakage", "other_failure"],
    "confidence": "float in [0.0, 1.0]",
    "reason": "short explanation",
}

PREDICTION_FIELDS = [
    "item_id",
    "baseline",
    "video_path",
    "target_absent",
    "effect_visible",
    "quality_ok",
    "pred_label",
    "confidence",
    "reason",
]

ALLOWED_TARGET_ABSENT = {"yes", "no", "partial"}
ALLOWED_EFFECT_VISIBLE = {"yes", "no", "partial"}
ALLOWED_QUALITY_OK = {"yes", "no"}
ALLOWED_LABELS = {"strict_leakage", "borderline", "target_leakage", "other_failure"}

Transport = Callable[[str, str, dict[str, Any], int], dict[str, Any]]


def read_inputs(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: missing header")
        missing = sorted(REQUIRED_INPUT_FIELDS - set(reader.fieldnames))
        if missing:
            raise ValueError(f"{path}: missing required columns: {', '.join(missing)}")
        return list(reader)


def prompt_for(row: dict[str, str]) -> str:
    return "\n".join(
        [
            "You are judging a video concept-erasure result from a 5-frame contact sheet.",
            f"Target concept to erase: {row['target_concept']}",
            f"Expected downstream causal effect: {row['expected_effect']}",
            f"Original source prompt: {row.get('source_prompt', '')}",
            "",
            "Inspect the frames in temporal order from left to right.",
            "Answer these questions: whether the target concept is absent, whether the expected effect is visible, and whether the image evidence is good enough to judge.",
            "Then choose the final prediction label using the provided response schema.",
            "Return only valid JSON with exactly these keys and value ranges:",
            '{ "target_absent": "yes|no|partial", "effect_visible": "yes|no|partial", "quality_ok": "yes|no", "pred_label": "strict_leakage|borderline|target_leakage|other_failure", "confidence": 0.0, "reason": "short visual explanation" }',
            "Use strict_leakage only when the target is absent and the expected downstream effect remains visible.",
            "Use target_leakage when the target concept is still visible.",
            "Use borderline when the target/effect separation is ambiguous.",
            "Use other_failure when the effect is absent or image quality is not enough to judge.",
        ]
    )


def payload_for(row: dict[str, str]) -> dict[str, object]:
    return {
        "output_id": row["output_id"],
        "item_id": row["item_id"],
        "baseline": row["baseline"],
        "video_path": row["video_path"],
        "image_path": row["sheet_path"],
        "sheet_available": row["sheet_exists"] == "true",
        "sheet_error": row.get("sheet_error", ""),
        "target_concept": row["target_concept"],
        "expected_effect": row["expected_effect"],
        "prompt": prompt_for(row),
        "response_schema": RESPONSE_SCHEMA,
    }


def image_data_url(path: Path) -> str:
    suffix = path.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def request_payload_for(row: dict[str, str], *, model: str, temperature: float, max_tokens: int) -> dict[str, Any]:
    payload = payload_for(row)
    image_path = Path(str(payload["image_path"]))
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": str(payload["prompt"])},
                    {"type": "image_url", "image_url": {"url": image_data_url(image_path)}},
                ],
            }
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }


def urllib_transport(url: str, api_key: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": "Bearer " + api_key, "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
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
    normalized = str(value).strip().lower()
    if normalized == "true":
        return "yes"
    if normalized == "false":
        return "no"
    return normalized


def normalize_prediction(parsed: dict[str, Any]) -> dict[str, str]:
    target_absent = normalize_flag(parsed.get("target_absent", ""))
    effect_visible = normalize_flag(parsed.get("effect_visible", parsed.get("causal_effect_visible", "")))
    quality_ok = normalize_flag(parsed.get("quality_ok", parsed.get("quality_sufficient", "")))
    pred_label = str(parsed.get("pred_label", parsed.get("label", ""))).strip().lower()
    if target_absent not in ALLOWED_TARGET_ABSENT:
        raise ValueError(f"invalid target_absent: {target_absent}")
    if effect_visible not in ALLOWED_EFFECT_VISIBLE:
        raise ValueError(f"invalid effect_visible: {effect_visible}")
    if quality_ok not in ALLOWED_QUALITY_OK:
        raise ValueError(f"invalid quality_ok: {quality_ok}")
    if pred_label not in ALLOWED_LABELS:
        raise ValueError(f"invalid pred_label: {pred_label}")
    try:
        confidence = float(parsed.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = min(1.0, max(0.0, confidence))
    return {
        "target_absent": target_absent,
        "effect_visible": effect_visible,
        "quality_ok": quality_ok,
        "pred_label": pred_label,
        "confidence": f"{confidence:.4f}",
        "reason": str(parsed.get("reason", "")).strip(),
    }


def message_content(response: dict[str, Any]) -> str:
    choices = response.get("choices", [])
    if not choices:
        return ""
    message = choices[0].get("message", {})
    content = message.get("content", "")
    if isinstance(content, list):
        return "\n".join(str(part.get("text", part)) if isinstance(part, dict) else str(part) for part in content)
    return str(content)


def raw_record_for(row: dict[str, str], model: str, normalized: dict[str, str], response: dict[str, Any]) -> dict[str, Any]:
    choices = response.get("choices", []) if isinstance(response, dict) else []
    first_choice = choices[0] if choices else {}
    return {
        "output_id": row["output_id"],
        "item_id": row["item_id"],
        "baseline": row["baseline"],
        "model": model,
        "parsed": normalized,
        "model_content": message_content(response) if isinstance(response, dict) else "",
        "finish_reason": first_choice.get("finish_reason", ""),
        "usage": response.get("usage", {}) if isinstance(response, dict) else {},
        "skipped": response.get("skipped", False) if isinstance(response, dict) else False,
    }


def filter_rows(rows: list[dict[str, str]], *, include_missing: bool, limit: int | None) -> list[dict[str, str]]:
    filtered = [row for row in rows if include_missing or row.get("sheet_exists") == "true"]
    if limit is not None:
        return filtered[:limit]
    return filtered


def write_jsonl(payloads: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for payload in payloads:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_prediction_rows(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PREDICTION_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def load_api_config(config_file: Path | None, base_url: str | None, api_key: str | None) -> tuple[str, str]:
    file_values: dict[str, str] = {}
    if config_file is not None:
        for line in config_file.read_text(encoding="utf-8").splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            file_values[key.strip()] = value.strip()
    resolved_base_url = base_url or os.environ.get("VLM_BASE_URL") or file_values.get("url")
    resolved_api_key = api_key or os.environ.get("VLM_API_KEY") or file_values.get("key")
    if not resolved_base_url:
        raise ValueError("missing API base URL; pass --base-url, VLM_BASE_URL, or config file url:")
    if not resolved_api_key:
        raise ValueError("missing API key; pass --api-key, VLM_API_KEY, or config file key:")
    return resolved_base_url.rstrip("/"), resolved_api_key


def run_api_mode(
    *,
    inputs_path: Path,
    predictions_path: Path,
    raw_output_path: Path | None,
    base_url: str,
    api_key: str,
    model: str,
    include_missing: bool,
    limit: int | None,
    temperature: float,
    max_tokens: int,
    timeout: int,
    transport: Transport = urllib_transport,
) -> int:
    rows = filter_rows(read_inputs(inputs_path), include_missing=include_missing, limit=limit)
    prediction_rows: list[dict[str, str]] = []
    raw_records: list[dict[str, Any]] = []
    url = base_url.rstrip("/") + "/chat/completions"
    for row in rows:
        if row.get("sheet_exists") != "true":
            parsed = {
                "target_absent": "partial",
                "effect_visible": "partial",
                "quality_ok": "no",
                "pred_label": "other_failure",
                "confidence": 0.0,
                "reason": row.get("sheet_error", "missing contact sheet"),
            }
            response = {"skipped": True, "reason": row.get("sheet_error", "missing contact sheet")}
        else:
            request_payload = request_payload_for(row, model=model, temperature=temperature, max_tokens=max_tokens)
            response = transport(url, api_key, request_payload, timeout)
            content = message_content(response)
            parsed = parse_model_json(content)
        normalized = normalize_prediction(parsed)
        prediction_rows.append(
            {
                "item_id": row["item_id"],
                "baseline": row["baseline"],
                "video_path": row["video_path"],
                **normalized,
            }
        )
        raw_records.append(raw_record_for(row, model, normalized, response))
    write_prediction_rows(prediction_rows, predictions_path)
    if raw_output_path is not None:
        write_jsonl(raw_records, raw_output_path)
    return len(prediction_rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path)
    parser.add_argument("--output-predictions", type=Path)
    parser.add_argument("--raw-output-jsonl", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-api", action="store_true")
    parser.add_argument("--include-missing", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--model", default="openai/gpt-4o")
    parser.add_argument("--base-url")
    parser.add_argument("--api-key")
    parser.add_argument("--api-config-file", type=Path)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument("--timeout", type=int, default=90)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.dry_run == args.run_api:
        parser.error("choose exactly one mode: --dry-run or --run-api")
    if args.limit is not None and args.limit < 0:
        parser.error("--limit must be non-negative")

    try:
        rows = read_inputs(args.inputs)
    except ValueError as exc:
        parser.exit(2, f"{exc}\n")

    if args.dry_run:
        if args.output_jsonl is None:
            parser.error("--dry-run requires --output-jsonl")
        selected = filter_rows(rows, include_missing=args.include_missing, limit=args.limit)
        payloads = [payload_for(row) for row in selected]
        write_jsonl(payloads, args.output_jsonl)
        skipped = len(rows) - len(filter_rows(rows, include_missing=args.include_missing, limit=None))
        print(f"Wrote {len(payloads)} dry-run payloads to {args.output_jsonl} (skipped_missing={skipped})")
        return 0

    if args.output_predictions is None:
        parser.error("--run-api requires --output-predictions")
    try:
        base_url, api_key = load_api_config(args.api_config_file, args.base_url, args.api_key)
        count = run_api_mode(
            inputs_path=args.inputs,
            predictions_path=args.output_predictions,
            raw_output_path=args.raw_output_jsonl,
            base_url=base_url,
            api_key=api_key,
            model=args.model,
            include_missing=args.include_missing,
            limit=args.limit,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
        )
    except ValueError as exc:
        parser.exit(2, f"{exc}\n")
    print(f"Wrote {count} predictions to {args.output_predictions} using {args.model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
