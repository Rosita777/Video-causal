#!/usr/bin/env python3
"""Chunked full-frame VLM prelabeling for clean-source causal validity."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import math
import os
import re
import time
import urllib.request
from pathlib import Path
from typing import Any


CHUNK_RESPONSE_SCHEMA = {
    "target_visible": ["yes", "no", "partial"],
    "target_frame_ids": ["frame ids such as f003"],
    "effect_visible": ["yes", "no", "partial"],
    "effect_frame_ids": ["frame ids such as f014"],
    "local_temporal_order": ["yes", "no", "partial"],
    "local_causal_transition": ["yes", "no", "partial"],
    "video_quality": ["yes", "no"],
    "confidence": "float in [0.0, 1.0]",
    "reason": "short visual evidence from this chunk only",
}

CHUNK_FIELDS = [
    "sample_id",
    "pair_id",
    "row_index",
    "chunk_index",
    "chunk_total",
    "frame_start",
    "frame_end",
    "frame_ids",
    "sheet_path",
    "video_path",
    "target_concept",
    "expected_effect",
    "prompt",
]

CHUNK_PREDICTION_FIELDS = [
    "sample_id",
    "pair_id",
    "row_index",
    "chunk_index",
    "frame_start",
    "frame_end",
    "target_visible",
    "target_frame_ids",
    "effect_visible",
    "effect_frame_ids",
    "local_temporal_order",
    "local_causal_transition",
    "video_quality",
    "confidence",
    "reason",
]

AGGREGATE_FIELDS = [
    "sample_id",
    "pair_id",
    "target_concept",
    "expected_effect",
    "chunks_seen",
    "target_present",
    "effect_present",
    "temporal_order_support",
    "causal_transition_support",
    "video_quality",
    "first_target_frame",
    "first_effect_frame",
    "rule_clean_source_candidate",
    "confidence",
    "reason",
]

FLAG3 = {"yes", "no", "partial"}
FLAG2 = {"yes", "no"}


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
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )


def append_jsonl_row(handle: Any, row: dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    handle.flush()


def selected_rows(rows: list[dict[str, str]], *, start_index: int, limit: int | None) -> list[tuple[int, dict[str, str]]]:
    selected = list(enumerate(rows))[start_index:]
    if limit is not None:
        selected = selected[:limit]
    return selected


def safe_stem(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", text).strip("-")
    return cleaned[:80] or "case"


def chunk_ranges(frame_count: int, chunk_count: int = 5, overlap: int = 3) -> list[tuple[int, int]]:
    if frame_count <= 0:
        return []
    chunk_count = max(1, min(chunk_count, frame_count))
    overlap = max(0, overlap)
    window = math.ceil((frame_count + overlap * (chunk_count - 1)) / chunk_count)
    window = max(1, min(window, frame_count))
    if chunk_count == 1 or frame_count <= window:
        return [(0, frame_count - 1)]

    max_start = frame_count - window
    starts = [round(i * max_start / (chunk_count - 1)) for i in range(chunk_count)]
    ranges: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for start in starts:
        end = min(frame_count - 1, start + window - 1)
        item = (start, end)
        if item not in seen:
            ranges.append(item)
            seen.add(item)
    if ranges[-1][1] < frame_count - 1:
        ranges.append((max(0, frame_count - window), frame_count - 1))
    return ranges


def resolve_video_path(path_text: str, project_root: Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else project_root / path


def decode_video_frames(video_path: Path) -> list[Any]:
    try:
        import cv2
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("cv2 and pillow are required to build chunk sheets") from exc

    cap = cv2.VideoCapture(str(video_path))
    frames = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb))
    finally:
        cap.release()
    return frames


def write_contact_sheet(
    frames: list[Any],
    frame_ids: list[int],
    output_path: Path,
    *,
    thumb_width: int,
    thumb_height: int,
    columns: int,
) -> Path:
    from PIL import Image, ImageDraw

    if not frames:
        raise ValueError("cannot build a contact sheet with no frames")
    columns = max(1, columns)
    rows = math.ceil(len(frames) / columns)
    label_height = 18
    cell_height = thumb_height + label_height
    sheet = Image.new("RGB", (columns * thumb_width, rows * cell_height), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (frame, frame_id) in enumerate(zip(frames, frame_ids)):
        col = index % columns
        row = index // columns
        x = col * thumb_width
        y = row * cell_height
        draw.text((x + 4, y + 2), f"f{frame_id:03d}", fill=(40, 40, 40))
        image = frame.copy()
        image.thumbnail((thumb_width, thumb_height))
        canvas = Image.new("RGB", (thumb_width, thumb_height), "white")
        offset = ((thumb_width - image.width) // 2, (thumb_height - image.height) // 2)
        canvas.paste(image, offset)
        sheet.paste(canvas, (x, y + label_height))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)
    return output_path


def prompt_for_chunk(chunk: dict[str, str]) -> str:
    return "\n".join(
        [
            "You are labeling one temporal chunk from a generated video for clean-source causal validity.",
            "The contact sheet shows every decoded frame in this chunk, ordered left-to-right and top-to-bottom.",
            f"Chunk: {int(chunk['chunk_index']) + 1}/{chunk['chunk_total']}",
            f"Frame range: f{int(chunk['frame_start']):03d}-f{int(chunk['frame_end']):03d}",
            "Answer only from this chunk. The target or effect may appear in other chunks, so do not make a global-video judgment.",
            f"Target concept: {chunk['target_concept']}",
            f"Expected causal footprint/effect: {chunk['expected_effect']}",
            f"Full prompt: {chunk['prompt']}",
            "For frame id fields, use visible labels from the sheet, such as f003 or f014.",
            "local_temporal_order=yes only if this chunk visibly supports target-before-effect ordering.",
            "local_causal_transition=yes only if this chunk visibly supports the target causing or transitioning into the effect.",
            "Return only valid JSON with exactly these keys:",
            json.dumps(CHUNK_RESPONSE_SCHEMA, ensure_ascii=False),
        ]
    )


def image_data_url(path: Path) -> str:
    suffix = path.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def request_payload_for(chunk: dict[str, str], *, model: str, temperature: float, max_tokens: int) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_for_chunk(chunk)},
                    {"type": "image_url", "image_url": {"url": image_data_url(Path(chunk["sheet_path"]))}},
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


def call_api_with_retries(
    url: str,
    api_key: str,
    payload: dict[str, Any],
    *,
    timeout: int,
    retries: int,
    retry_sleep: float,
    transport=urllib_transport,
) -> dict[str, Any]:
    last_error: Exception | None = None
    attempts = max(1, retries)
    for attempt in range(attempts):
        try:
            return transport(url, api_key, payload, timeout)
        except Exception as exc:  # pragma: no cover - concrete errors vary by transport
            last_error = exc
            if attempt + 1 >= attempts:
                break
            time.sleep(max(0.0, retry_sleep) * (attempt + 1))
    assert last_error is not None
    raise last_error


def extract_content(response: dict[str, Any]) -> str:
    content = response["choices"][0]["message"]["content"]
    if isinstance(content, list):
        return "".join(str(part.get("text", "")) if isinstance(part, dict) else str(part) for part in content)
    return str(content)


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
    if isinstance(value, list):
        value = value[0] if value else ""
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
    return f"{max(0.0, min(1.0, confidence)):.4f}"


def normalize_frame_ids(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        items = re.findall(r"f?\d+", value)
    elif isinstance(value, list):
        items = []
        for item in value:
            items.extend(re.findall(r"f?\d+", str(item)))
    else:
        items = re.findall(r"f?\d+", str(value))
    normalized = []
    for item in items:
        digits = re.search(r"\d+", item)
        if digits:
            normalized.append(f"f{int(digits.group(0)):03d}")
    return ";".join(dict.fromkeys(normalized))


def normalize_chunk_prediction(parsed: dict[str, Any]) -> dict[str, str]:
    pred = {
        "target_visible": normalize_flag(parsed.get("target_visible", "")),
        "target_frame_ids": normalize_frame_ids(parsed.get("target_frame_ids", [])),
        "effect_visible": normalize_flag(parsed.get("effect_visible", "")),
        "effect_frame_ids": normalize_frame_ids(parsed.get("effect_frame_ids", [])),
        "local_temporal_order": normalize_flag(parsed.get("local_temporal_order", "")),
        "local_causal_transition": normalize_flag(parsed.get("local_causal_transition", "")),
        "video_quality": normalize_flag(parsed.get("video_quality", "")),
        "confidence": normalize_confidence(parsed.get("confidence", 0.0)),
        "reason": str(parsed.get("reason", "")),
    }
    for field in ["target_visible", "effect_visible", "local_temporal_order", "local_causal_transition"]:
        if pred[field] not in FLAG3:
            raise ValueError(f"invalid {field}: {pred[field]}")
    if pred["video_quality"] not in FLAG2:
        raise ValueError(f"invalid video_quality: {pred['video_quality']}")
    return pred


def fallback_chunk_prediction_for(chunk: dict[str, str], exc: Exception) -> dict[str, str]:
    return {
        "target_visible": "no",
        "target_frame_ids": "",
        "effect_visible": "no",
        "effect_frame_ids": "",
        "local_temporal_order": "no",
        "local_causal_transition": "no",
        "video_quality": "no",
        "confidence": "0.0000",
        "reason": f"{type(exc).__name__}: {exc}",
    }


def chunk_prediction_key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return (
        str(row.get("row_index", "")),
        str(row.get("chunk_index", "")),
        str(row.get("frame_start", "")),
        str(row.get("frame_end", "")),
    )


def pending_chunks(chunks: list[dict[str, str]], existing_predictions: list[dict[str, str]]) -> list[dict[str, str]]:
    completed = {chunk_prediction_key(row) for row in existing_predictions}
    return [chunk for chunk in chunks if chunk_prediction_key(chunk) not in completed]


def parse_frame_numbers(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, list):
        text = ";".join(str(item) for item in value)
    else:
        text = str(value)
    return [int(match) for match in re.findall(r"f?(\d+)", text)]


def present_flag(values: list[str]) -> str:
    if "yes" in values:
        return "yes"
    if "partial" in values:
        return "partial"
    return "no"


def support_flag(values: list[str], *, fallback: bool = False) -> str:
    if "yes" in values:
        return "yes"
    if "partial" in values or fallback:
        return "partial"
    return "no"


def aggregate_chunk_predictions(row: dict[str, str], chunks: list[dict[str, str]]) -> dict[str, str]:
    target_values = [chunk["target_visible"] for chunk in chunks]
    effect_values = [chunk["effect_visible"] for chunk in chunks]
    temporal_values = [chunk["local_temporal_order"] for chunk in chunks]
    transition_values = [chunk["local_causal_transition"] for chunk in chunks]
    quality_values = [chunk["video_quality"] for chunk in chunks]

    target_frames = [frame for chunk in chunks for frame in parse_frame_numbers(chunk.get("target_frame_ids", ""))]
    effect_frames = [frame for chunk in chunks for frame in parse_frame_numbers(chunk.get("effect_frame_ids", ""))]
    first_target = min(target_frames) if target_frames else None
    first_effect = min(effect_frames) if effect_frames else None
    temporal_by_ids = first_target is not None and first_effect is not None and first_target <= first_effect
    target_present = present_flag(target_values)
    effect_present = present_flag(effect_values)
    video_quality = "yes" if "yes" in quality_values else "no"
    temporal_support = support_flag(temporal_values, fallback=temporal_by_ids)
    transition_support = support_flag(transition_values, fallback=temporal_by_ids)
    clean_candidate = (
        target_present in {"yes", "partial"}
        and effect_present in {"yes", "partial"}
        and video_quality == "yes"
        and (temporal_support in {"yes", "partial"} or transition_support in {"yes", "partial"})
    )
    confidences = []
    for chunk in chunks:
        try:
            confidences.append(float(chunk["confidence"]))
        except (KeyError, ValueError):
            pass
    reason_bits = [
        f"c{idx}:{chunk.get('reason', '')}" for idx, chunk in enumerate(chunks) if chunk.get("reason")
    ]
    return {
        "sample_id": row.get("prompt_id", ""),
        "pair_id": row.get("pair_id", ""),
        "target_concept": row.get("target_concept", ""),
        "expected_effect": row.get("expected_effect", ""),
        "chunks_seen": str(len(chunks)),
        "target_present": target_present,
        "effect_present": effect_present,
        "temporal_order_support": temporal_support,
        "causal_transition_support": transition_support,
        "video_quality": video_quality,
        "first_target_frame": "" if first_target is None else str(first_target),
        "first_effect_frame": "" if first_effect is None else str(first_effect),
        "rule_clean_source_candidate": "yes" if clean_candidate else "no",
        "confidence": normalize_confidence(sum(confidences) / len(confidences) if confidences else 0.0),
        "reason": " | ".join(reason_bits)[:1000],
    }


def build_chunk_manifest(
    rows: list[tuple[int, dict[str, str]]],
    *,
    project_root: Path,
    output_dir: Path,
    chunk_count: int,
    chunk_overlap: int,
    thumb_width: int,
    thumb_height: int,
    sheet_columns: int,
) -> list[dict[str, str]]:
    chunks: list[dict[str, str]] = []
    sheet_dir = output_dir / "chunk_sheets"
    for row_index, row in rows:
        video_path = resolve_video_path(row.get("video_path", ""), project_root)
        if not video_path.exists():
            continue
        frames = decode_video_frames(video_path)
        ranges = chunk_ranges(len(frames), chunk_count=chunk_count, overlap=chunk_overlap)
        for chunk_index, (start, end) in enumerate(ranges):
            frame_ids = list(range(start, end + 1))
            sheet_path = sheet_dir / f"{row_index:03d}_{safe_stem(row.get('prompt_id', f'case_{row_index:03d}'))}_chunk{chunk_index:02d}_f{start:03d}_f{end:03d}.jpg"
            write_contact_sheet(
                [frames[frame_id] for frame_id in frame_ids],
                frame_ids,
                sheet_path,
                thumb_width=thumb_width,
                thumb_height=thumb_height,
                columns=sheet_columns,
            )
            chunks.append(
                {
                    "sample_id": row.get("prompt_id", f"case_{row_index:03d}"),
                    "pair_id": row.get("pair_id", ""),
                    "row_index": str(row_index),
                    "chunk_index": str(chunk_index),
                    "chunk_total": str(len(ranges)),
                    "frame_start": str(start),
                    "frame_end": str(end),
                    "frame_ids": ";".join(f"f{frame_id:03d}" for frame_id in frame_ids),
                    "sheet_path": str(sheet_path),
                    "video_path": str(video_path),
                    "target_concept": row.get("target_concept", ""),
                    "expected_effect": row.get("expected_effect", ""),
                    "prompt": row.get("prompt", ""),
                }
            )
    return chunks


def payload_for_chunk(chunk: dict[str, str], *, model: str, temperature: float, max_tokens: int) -> dict[str, Any]:
    payload = {
        "sample_id": chunk["sample_id"],
        "pair_id": chunk["pair_id"],
        "row_index": chunk["row_index"],
        "chunk_index": chunk["chunk_index"],
        "frame_start": chunk["frame_start"],
        "frame_end": chunk["frame_end"],
        "image_path": chunk["sheet_path"],
        "prompt": prompt_for_chunk(chunk),
        "response_schema": CHUNK_RESPONSE_SCHEMA,
    }
    payload["request"] = request_payload_for(chunk, model=model, temperature=temperature, max_tokens=max_tokens)
    return payload


def load_api_config(args: argparse.Namespace) -> tuple[str, str]:
    url = args.api_url or os.environ.get("OPENAI_BASE_URL", "").rstrip("/")
    key = args.api_key or os.environ.get("OPENAI_API_KEY", "")
    if not url:
        raise SystemExit("missing API URL; pass --api-url or set OPENAI_BASE_URL")
    if not key:
        raise SystemExit("missing API key; pass --api-key or set OPENAI_API_KEY")
    return url.rstrip("/") + "/chat/completions", key


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--review-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--chunk-count", type=int, default=5)
    parser.add_argument("--chunk-overlap", type=int, default=3)
    parser.add_argument("--thumb-width", type=int, default=160)
    parser.add_argument("--thumb-height", type=int, default=90)
    parser.add_argument("--sheet-columns", type=int, default=7)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--api-url")
    parser.add_argument("--api-key")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=600)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--api-retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument("--continue-on-error", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.chunk_count <= 0:
        parser.error("--chunk-count must be positive")
    if args.chunk_overlap < 0:
        parser.error("--chunk-overlap must be non-negative")
    if args.start_index < 0:
        parser.error("--start-index must be non-negative")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.api_retries <= 0:
        parser.error("--api-retries must be positive")
    if args.retry_sleep < 0:
        parser.error("--retry-sleep must be non-negative")

    rows = read_csv(args.review_csv)
    selected = selected_rows(rows, start_index=args.start_index, limit=args.limit)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    chunks = build_chunk_manifest(
        selected,
        project_root=args.project_root,
        output_dir=args.output_dir,
        chunk_count=args.chunk_count,
        chunk_overlap=args.chunk_overlap,
        thumb_width=args.thumb_width,
        thumb_height=args.thumb_height,
        sheet_columns=args.sheet_columns,
    )
    write_csv(args.output_dir / "chunk_manifest.csv", chunks, CHUNK_FIELDS)
    payloads = [
        payload_for_chunk(chunk, model=args.model, temperature=args.temperature, max_tokens=args.max_tokens)
        for chunk in chunks
    ]
    write_jsonl(args.output_dir / "chunk_payloads.jsonl", payloads)
    if args.dry_run:
        print(f"Wrote {len(chunks)} clean chunk payloads to {args.output_dir}")
        return 0

    api_url, api_key = load_api_config(args)
    chunk_predictions_path = args.output_dir / "chunk_predictions.csv"
    aggregate_predictions_path = args.output_dir / "aggregate_predictions.csv"
    raw_path = args.output_dir / "raw_responses.jsonl"
    chunk_predictions: list[dict[str, str]] = (
        read_csv(chunk_predictions_path) if chunk_predictions_path.exists() else []
    )
    chunks_to_eval = pending_chunks(chunks, chunk_predictions)
    raw_handle = raw_path.open("a", encoding="utf-8")
    try:
        for chunk in chunks_to_eval:
            try:
                request_payload = request_payload_for(
                    chunk,
                    model=args.model,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                )
                response = call_api_with_retries(
                    api_url,
                    api_key,
                    request_payload,
                    timeout=args.timeout,
                    retries=args.api_retries,
                    retry_sleep=args.retry_sleep,
                )
                parsed = parse_model_json(extract_content(response))
                pred = normalize_chunk_prediction(parsed)
                append_jsonl_row(raw_handle, {"chunk": chunk, "response": response, "parsed": parsed})
            except Exception as exc:
                if not args.continue_on_error:
                    raise
                pred = fallback_chunk_prediction_for(chunk, exc)
                append_jsonl_row(raw_handle, {"chunk": chunk, "error": f"{type(exc).__name__}: {exc}"})
            chunk_predictions.append(
                {
                    "sample_id": chunk["sample_id"],
                    "pair_id": chunk["pair_id"],
                    "row_index": chunk["row_index"],
                    "chunk_index": chunk["chunk_index"],
                    "frame_start": chunk["frame_start"],
                    "frame_end": chunk["frame_end"],
                    **pred,
                }
            )
            write_csv(chunk_predictions_path, chunk_predictions, CHUNK_PREDICTION_FIELDS)
    finally:
        raw_handle.close()

    by_row: dict[str, list[dict[str, str]]] = {}
    row_lookup = {str(index): row for index, row in selected}
    for pred in chunk_predictions:
        by_row.setdefault(pred["row_index"], []).append(pred)
    aggregate_rows = [
        aggregate_chunk_predictions(row_lookup[row_index], preds)
        for row_index, preds in sorted(by_row.items(), key=lambda item: int(item[0]))
    ]
    write_csv(aggregate_predictions_path, aggregate_rows, AGGREGATE_FIELDS)
    print(f"Wrote {len(chunk_predictions)} chunk predictions and {len(aggregate_rows)} aggregate rows to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
