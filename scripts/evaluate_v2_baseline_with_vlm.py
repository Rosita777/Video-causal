#!/usr/bin/env python3
"""VLM prelabeling for v2 baseline target/footprint review rows."""

from __future__ import annotations

import argparse
import base64
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import re
from typing import Any, Callable
import urllib.request


REVIEW_REQUIRED_FIELDS = {
    "item_index",
    "slice_index",
    "source_index",
    "pair_id",
    "mechanism_type",
    "baseline",
    "baseline_label",
    "video_path",
    "strip_path",
    "strip_exists",
    "target_concept",
    "expected_effect",
    "source_prompt",
}

INPUT_FIELDS = [
    "output_id",
    "item_id",
    "item_index",
    "slice_index",
    "source_index",
    "pair_id",
    "mechanism_type",
    "baseline",
    "baseline_label",
    "video_path",
    "strip_path",
    "strip_exists",
    "reference_strip_path",
    "reference_strip_exists",
    "target_concept",
    "expected_effect",
    "source_prompt",
]

PREDICTION_FIELDS = [
    *INPUT_FIELDS,
    "target_visible",
    "footprint_visible",
    "footprint_match",
    "separation_clear",
    "video_quality",
    "confidence",
    "final_label",
    "notes",
]

RESPONSE_SCHEMA = {
    "target_visible": ["yes", "no", "partial"],
    "footprint_visible": ["yes", "no", "partial"],
    "footprint_match": ["yes", "no", "partial"],
    "separation_clear": ["yes", "no"],
    "video_quality": ["yes", "no"],
    "confidence": "float in [0.0, 1.0]",
    "reason": "short visual evidence",
}

FLAG3 = {"yes", "no", "partial"}
FLAG2 = {"yes", "no"}
FINAL_LABELS = {
    "strict_causal_footprint_leakage",
    "target_leakage",
    "erased_clean",
    "borderline",
    "other_failure",
}

Transport = Callable[[str, str, dict[str, Any], int], dict[str, Any]]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: missing header")
        return list(reader)


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def validate_review_rows(rows: list[dict[str, str]], path: Path) -> None:
    if not rows:
        raise ValueError(f"{path}: no rows")
    missing = sorted(REVIEW_REQUIRED_FIELDS - set(rows[0]))
    if missing:
        raise ValueError(f"{path}: missing required columns: {', '.join(missing)}")


def build_input_rows(review_rows: list[dict[str, str]], *, source_name: str = "v2_yes100") -> list[dict[str, str]]:
    clean_by_item = {row["item_index"]: row for row in review_rows if row.get("baseline") == "clean_reference"}
    inputs: list[dict[str, str]] = []
    for row in review_rows:
        baseline = row.get("baseline", "")
        if baseline == "clean_reference":
            continue
        clean = clean_by_item.get(row["item_index"], {})
        pair_id = row.get("pair_id", "")
        item_id = f"{source_name}:{pair_id}"
        inputs.append(
            {
                "output_id": f"{item_id}::{baseline}",
                "item_id": item_id,
                "item_index": row.get("item_index", ""),
                "slice_index": row.get("slice_index", ""),
                "source_index": row.get("source_index", ""),
                "pair_id": pair_id,
                "mechanism_type": row.get("mechanism_type", ""),
                "baseline": baseline,
                "baseline_label": row.get("baseline_label", baseline),
                "video_path": row.get("video_path", ""),
                "strip_path": row.get("strip_path", ""),
                "strip_exists": row.get("strip_exists", "false"),
                "reference_strip_path": clean.get("strip_path", ""),
                "reference_strip_exists": clean.get("strip_exists", "false"),
                "target_concept": row.get("target_concept", ""),
                "expected_effect": row.get("expected_effect", ""),
                "source_prompt": row.get("source_prompt", ""),
            }
        )
    return inputs


def filter_input_rows(
    rows: list[dict[str, str]],
    *,
    baselines: set[str] | None,
    include_missing: bool,
    start_index: int,
    limit: int | None,
) -> list[dict[str, str]]:
    filtered = []
    for row in rows:
        if baselines is not None and row["baseline"] not in baselines:
            continue
        if not include_missing and (row["strip_exists"] != "true" or row["reference_strip_exists"] != "true"):
            continue
        filtered.append(row)
    filtered = filtered[start_index:]
    if limit is not None:
        filtered = filtered[:limit]
    return filtered


def prompt_for(row: dict[str, str]) -> str:
    return "\n".join(
        [
            "You are labeling a video concept-erasure result for a causal-footprint benchmark.",
            "You will receive two contact-sheet images sampled evenly from generated videos.",
            "The first image is the clean reference: it shows the original target and the expected downstream causal footprint.",
            "The second image is the erased output: this is the only video to label.",
            "Use the clean reference only to understand appearance, timing, and location of the target and footprint.",
            "Do not infer from the text prompt alone; answer from visible evidence in the images.",
            "",
            f"Target concept that should be erased: {row['target_concept']}",
            f"Expected causal footprint: {row['expected_effect']}",
            f"Original source prompt: {row['source_prompt']}",
            "",
            "Inspect the erased-output frames in temporal order.",
            "target_visible=yes if the erased target concept is visibly present in the erased output.",
            "target_visible=partial if any target cue is ambiguous, cropped, tiny, occluded, or only partly visible.",
            "target_visible=no only if the target is absent from all erased-output frames.",
            "footprint_visible=yes if a downstream effect is visibly present.",
            "footprint_match=yes only if that visible footprint matches the expected causal footprint.",
            "footprint_match=no if the visible effect is unrelated to the expected footprint, or if no footprint is visible.",
            "separation_clear=no if target and footprint evidence cannot be judged separately.",
            "video_quality=no only if the erased-output contact sheet is too broken or irrelevant to judge.",
            "Do not choose the final benchmark label; project code will derive it from your atomic answers.",
            "Return only valid JSON with exactly these keys:",
            json.dumps(RESPONSE_SCHEMA, ensure_ascii=False),
        ]
    )


def payload_for(row: dict[str, str]) -> dict[str, Any]:
    return {
        "output_id": row["output_id"],
        "item_id": row["item_id"],
        "pair_id": row["pair_id"],
        "baseline": row["baseline"],
        "video_path": row["video_path"],
        "reference_image_path": row["reference_strip_path"],
        "reference_available": row["reference_strip_exists"] == "true",
        "image_path": row["strip_path"],
        "image_available": row["strip_exists"] == "true",
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
    content = [
        {"type": "text", "text": prompt_for(row)},
        {"type": "image_url", "image_url": {"url": image_data_url(Path(row["reference_strip_path"]))}},
        {"type": "image_url", "image_url": {"url": image_data_url(Path(row["strip_path"]))}},
    ]
    return {
        "model": model,
        "messages": [{"role": "user", "content": content}],
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
    if isinstance(value, list) and value:
        value = value[0]
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


def normalize_confidence(value: Any) -> str:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        confidence = 0.0
    return f"{max(0.0, min(1.0, confidence)):.4f}"


def derive_final_label(
    *,
    target_visible: str,
    footprint_visible: str,
    footprint_match: str,
    separation_clear: str,
    video_quality: str,
) -> str:
    if video_quality == "no":
        return "other_failure"
    if target_visible == "yes":
        return "target_leakage"
    if target_visible == "partial" or footprint_visible == "partial" or footprint_match == "partial":
        return "borderline"
    if separation_clear == "no":
        return "borderline"
    if target_visible == "no" and footprint_visible == "yes" and footprint_match == "yes":
        return "strict_causal_footprint_leakage"
    if target_visible == "no" and footprint_visible == "no":
        return "erased_clean"
    return "borderline"


def normalize_prediction(parsed: dict[str, Any]) -> dict[str, str]:
    target_visible = normalize_flag(parsed.get("target_visible", ""))
    footprint_visible = normalize_flag(
        parsed.get("footprint_visible", parsed.get("effect_visible", parsed.get("causal_effect_visible", "")))
    )
    footprint_match = normalize_flag(
        parsed.get("footprint_match", parsed.get("effect_match", parsed.get("causal_footprint_match", "")))
    )
    if not footprint_match and footprint_visible == "no":
        footprint_match = "no"
    separation_clear = normalize_flag(parsed.get("separation_clear", parsed.get("target_effect_separation_clear", "")))
    video_quality = normalize_flag(parsed.get("video_quality", parsed.get("quality_ok", parsed.get("quality_sufficient", ""))))

    if target_visible not in FLAG3:
        raise ValueError(f"invalid target_visible: {target_visible}")
    if footprint_visible not in FLAG3:
        raise ValueError(f"invalid footprint_visible: {footprint_visible}")
    if footprint_match not in FLAG3:
        raise ValueError(f"invalid footprint_match: {footprint_match}")
    if separation_clear not in FLAG2:
        raise ValueError(f"invalid separation_clear: {separation_clear}")
    if video_quality not in FLAG2:
        raise ValueError(f"invalid video_quality: {video_quality}")

    final_label = derive_final_label(
        target_visible=target_visible,
        footprint_visible=footprint_visible,
        footprint_match=footprint_match,
        separation_clear=separation_clear,
        video_quality=video_quality,
    )
    if final_label not in FINAL_LABELS:
        raise ValueError(f"invalid final_label: {final_label}")
    return {
        "target_visible": target_visible,
        "footprint_visible": footprint_visible,
        "footprint_match": footprint_match,
        "separation_clear": separation_clear,
        "video_quality": video_quality,
        "confidence": normalize_confidence(parsed.get("confidence", 0.0)),
        "final_label": final_label,
        "notes": str(parsed.get("reason", parsed.get("notes", ""))).strip(),
    }


def fallback_prediction_for(row: dict[str, str], exc: Exception | str) -> dict[str, str]:
    message = exc if isinstance(exc, str) else f"{type(exc).__name__}: {exc}"
    return {
        "target_visible": "partial",
        "footprint_visible": "partial",
        "footprint_match": "partial",
        "separation_clear": "no",
        "video_quality": "no",
        "confidence": "0.0000",
        "final_label": "other_failure",
        "notes": f"evaluator_error: {message}",
    }


def message_content(response: dict[str, Any]) -> str:
    choices = response.get("choices", [])
    if not choices:
        return ""
    content = choices[0].get("message", {}).get("content", "")
    if isinstance(content, list):
        return "\n".join(str(part.get("text", part)) if isinstance(part, dict) else str(part) for part in content)
    return str(content)


def raw_record_for(row: dict[str, str], model: str, normalized: dict[str, str], response: dict[str, Any]) -> dict[str, Any]:
    choices = response.get("choices", []) if isinstance(response, dict) else []
    first_choice = choices[0] if choices else {}
    return {
        "output_id": row["output_id"],
        "item_id": row["item_id"],
        "pair_id": row["pair_id"],
        "baseline": row["baseline"],
        "model": model,
        "parsed": normalized,
        "model_content": message_content(response) if isinstance(response, dict) else "",
        "finish_reason": first_choice.get("finish_reason", ""),
        "usage": response.get("usage", {}) if isinstance(response, dict) else {},
        "error": response.get("error", "") if isinstance(response, dict) else "",
    }


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


def evaluate_one(
    row: dict[str, str],
    *,
    url: str,
    api_key: str,
    model: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
    continue_on_error: bool,
    transport: Transport = urllib_transport,
) -> tuple[dict[str, str], dict[str, Any]]:
    try:
        if row["strip_exists"] != "true" or row["reference_strip_exists"] != "true":
            raise ValueError("missing output or reference strip")
        request_payload = request_payload_for(row, model=model, temperature=temperature, max_tokens=max_tokens)
        response = transport(url, api_key, request_payload, timeout)
        parsed = parse_model_json(message_content(response))
        normalized = normalize_prediction(parsed)
    except Exception as exc:
        if not continue_on_error:
            raise
        normalized = fallback_prediction_for(row, exc)
        response = {"error": f"{type(exc).__name__}: {exc}", "choices": [{"message": {"content": str(exc)}}]}
    return {**row, **normalized}, raw_record_for(row, model, normalized, response)


def run_api(
    rows: list[dict[str, str]],
    *,
    base_url: str,
    api_key: str,
    model: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
    workers: int,
    continue_on_error: bool,
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    url = base_url.rstrip("/") + "/chat/completions"
    if workers <= 1:
        results = [
            evaluate_one(
                row,
                url=url,
                api_key=api_key,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
                continue_on_error=continue_on_error,
            )
            for row in rows
        ]
    else:
        indexed: list[tuple[int, dict[str, str], dict[str, Any]]] = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_index = {
                executor.submit(
                    evaluate_one,
                    row,
                    url=url,
                    api_key=api_key,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout,
                    continue_on_error=continue_on_error,
                ): index
                for index, row in enumerate(rows)
            }
            for future in as_completed(future_to_index):
                pred, raw = future.result()
                indexed.append((future_to_index[future], pred, raw))
        results = [(pred, raw) for _, pred, raw in sorted(indexed, key=lambda item: item[0])]
    return [pred for pred, _ in results], [raw for _, raw in results]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--run-api", action="store_true")
    parser.add_argument("--source-name", default="v2_yes100")
    parser.add_argument("--baseline", action="append")
    parser.add_argument("--include-missing", action="store_true")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--base-url")
    parser.add_argument("--api-key")
    parser.add_argument("--api-config-file", type=Path)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=400)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--continue-on-error", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.start_index < 0:
        parser.error("--start-index must be non-negative")
    if args.limit is not None and args.limit < 0:
        parser.error("--limit must be non-negative")
    if args.workers <= 0:
        parser.error("--workers must be positive")

    try:
        review_rows = read_csv(args.review_csv)
        validate_review_rows(review_rows, args.review_csv)
        all_inputs = build_input_rows(review_rows, source_name=args.source_name)
    except ValueError as exc:
        parser.exit(2, f"{exc}\n")

    selected = filter_input_rows(
        all_inputs,
        baselines=set(args.baseline) if args.baseline else None,
        include_missing=args.include_missing,
        start_index=args.start_index,
        limit=args.limit,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inputs_path = args.output_dir / "vlm_inputs.csv"
    payloads_path = args.output_dir / "vlm_payloads.jsonl"
    write_csv(inputs_path, selected, INPUT_FIELDS)
    write_jsonl(payloads_path, [payload_for(row) for row in selected])

    if args.dry_run:
        print(f"Wrote {len(selected)} VLM input rows to {inputs_path}")
        print(f"Wrote dry-run payloads to {payloads_path}")
        return 0

    try:
        base_url, api_key = load_api_config(args.api_config_file, args.base_url, args.api_key)
        predictions, raw_records = run_api(
            selected,
            base_url=base_url,
            api_key=api_key,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            workers=args.workers,
            continue_on_error=args.continue_on_error,
        )
    except ValueError as exc:
        parser.exit(2, f"{exc}\n")
    predictions_path = args.output_dir / "vlm_predictions.csv"
    raw_path = args.output_dir / "vlm_raw_responses.jsonl"
    write_csv(predictions_path, predictions, PREDICTION_FIELDS)
    write_jsonl(raw_path, raw_records)
    print(f"Wrote {len(predictions)} predictions to {predictions_path} using {args.model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
