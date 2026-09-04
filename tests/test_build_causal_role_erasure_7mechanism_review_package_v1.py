from __future__ import annotations

import csv
import hashlib
import json
import os
import stat
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from scripts import build_causal_role_erasure_7mechanism_review_package_v1 as review


IDENTIFICATION_LOCAL_INDICES = (
    0,
    3,
    5,
    6,
    9,
    10,
    12,
    15,
    16,
    19,
    21,
    22,
    25,
    26,
    27,
    29,
    30,
    32,
    33,
    34,
    36,
    37,
    40,
    41,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def make_fake_video(path: Path, width: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 8.0, (width, 480)
    )
    assert writer.isOpened()
    for index in range(49):
        frame = np.zeros((480, width, 3), dtype=np.uint8)
        frame[:, :, 0] = (index * 5) % 256
        frame[:, :, 1] = (index * 11) % 256
        frame[:, :, 2] = (index * 17) % 256
        cv2.putText(frame, f"frame {index:02d}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        writer.write(frame)
    writer.release()


def case_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for mechanism_index, mechanism in enumerate(review.MECHANISMS):
        for local_index in range(42):
            global_index = mechanism_index * 42 + local_index
            case_kind = "causal" if local_index < 24 else "specificity"
            rows.append(
                {
                    "protocol_version": review.PROTOCOL_VERSION,
                    "case_id": f"eval7m_m{mechanism_index:02d}_c{local_index:02d}",
                    "global_case_index": global_index,
                    "mechanism_index": mechanism_index,
                    "mechanism": mechanism,
                    "mechanism_name": mechanism.replace("_", " ").title(),
                    "case_kind": case_kind,
                    "source_object": f"protected source object {global_index}",
                    "receiver": f"receiver {global_index}",
                    "prompt": f"A neutral scene description for case {global_index}.",
                    "expected_footprint": f"visible footprint {global_index}",
                    "expected_counterfactual_state": f"counterfactual state {global_index}",
                    "protected_object": f"protected source object {global_index}" if case_kind == "specificity" else "",
                    "acceptable_alternative_cause": f"alternative cause {global_index}" if case_kind == "specificity" else "",
                    "seed": 1_000_000 + global_index,
                    "num_frames": 49,
                    "fps": 8,
                    "height": 480,
                    "width": 832,
                }
            )
    return rows


def identification_rows(cases: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for mechanism_index in (0, 2):
        for local_index in IDENTIFICATION_LOCAL_INDICES:
            case = cases[mechanism_index * 42 + local_index]
            rows.append(
                {
                    "protocol_version": review.PROTOCOL_VERSION,
                    "identification_case_index": len(rows),
                    "mechanism": case["mechanism"],
                    "case_id": case["case_id"],
                    "case_kind": case["case_kind"],
                    "seed": case["seed"],
                    "included_streams": "matched_control,V4",
                    "additional_streams": "generic_paraphrase,bystander_token",
                }
            )
    return rows


def media(width: int, *, wan: bool) -> dict[str, object]:
    value: dict[str, object] = {
        "decoded_frames": 49,
        "fps": "8/1",
        "height": 480,
        "width": width,
    }
    if wan:
        value.update({"video_streams": 1, "audio_streams": 0})
    return value


def item(
    *,
    root: Path,
    base_video: Path,
    case: dict[str, object],
    stream: str,
    source: str,
) -> dict[str, object]:
    target = root / "generated" / source / stream.lower() / f"{case['case_id']}.mp4"
    target.parent.mkdir(parents=True, exist_ok=True)
    os.link(base_video, target)
    width = 720 if stream in review.COGVIDEOX_STREAMS else 832
    result: dict[str, object] = {
        "case_id": case["case_id"],
        "job_id": f"{stream}__{case['mechanism']}",
        "mechanism": case["mechanism"],
        "seed": case["seed"],
        "video_path": str(target),
        "video_sha256": sha256(base_video),
        "size_bytes": base_video.stat().st_size,
        "media": media(width, wan=stream not in review.COGVIDEOX_STREAMS),
    }
    if source in ("wan_original", "trained_wan"):
        result["formal_global_index"] = case["global_case_index"]
    if source == "trained_wan":
        result["arm"] = stream
    else:
        result["stream"] = stream
    return result


def frozen_fixture(root: Path) -> dict[str, Path]:
    cases = case_rows()
    identification = identification_rows(cases)
    formal_path = root / "data" / "formal_cases.csv"
    identification_path = root / "data" / "identification_subset.csv"
    write_csv(formal_path, cases)
    write_csv(identification_path, identification)
    wan_video = root / "fixtures" / "wan.mp4"
    cog_video = root / "fixtures" / "cog.mp4"
    make_fake_video(wan_video, 832)
    make_fake_video(cog_video, 720)

    wan_original_items = [
        item(root=root, base_video=wan_video, case=case, stream="wan_original", source="wan_original")
        for case in cases
    ]
    trained_items: list[dict[str, object]] = []
    for stream in ("matched_control", "V4"):
        trained_items.extend(
            item(root=root, base_video=wan_video, case=case, stream=stream, source="trained_wan")
            for case in cases
        )
    identification_by_id = {str(row["case_id"]): row for row in identification}
    identification_cases = [case for case in cases if str(case["case_id"]) in identification_by_id]
    identification_cases.sort(key=lambda case: int(identification_by_id[str(case["case_id"])]["identification_case_index"]))
    for stream in review.IDENTIFICATION_STREAMS:
        trained_items.extend(
            item(root=root, base_video=wan_video, case=case, stream=stream, source="trained_wan")
            for case in identification_cases
        )
    core_items: list[dict[str, object]] = []
    for stream in review.COGVIDEOX_STREAMS[:4]:
        core_items.extend(
            item(root=root, base_video=cog_video, case=case, stream=stream, source="cog_core")
            for case in cases
        )
    safree_items = [
        item(root=root, base_video=cog_video, case=case, stream="safree_cogvideox", source="safree")
        for case in cases
    ]

    manifests = root / "manifests"
    paths = {
        "formal": formal_path,
        "identification": identification_path,
        "wan_original": manifests / "wan_original_generation_manifest.json",
        "trained_wan": manifests / "eval_generation_manifest.json",
        "cog_core": manifests / "core_baseline_generation_manifest.json",
        "safree": manifests / "safree_baseline_generation_manifest.json",
        "blind_key": root / "private_input" / "blind.key",
    }
    formal_binding = {
        "formal_cases": str(formal_path),
        "formal_cases_sha256": sha256(formal_path),
    }
    write_json(
        paths["wan_original"],
        {
            "schema_version": 1,
            "protocol_id": review.PROTOCOL_VERSION,
            "runner_id": "causal_role_erasure_7mechanism_wan_original_v2",
            "status": "frozen_after_exact_294_video_validation",
            "video_count": 294,
            "inputs": dict(formal_binding),
            "items": wan_original_items,
        },
    )
    write_json(
        paths["trained_wan"],
        {
            "schema_version": 1,
            "protocol_id": review.PROTOCOL_VERSION,
            "runner_id": "causal_role_erasure_7mechanism_trained_wan_eval_v2",
            "status": "frozen_after_exact_684_video_validation",
            "video_count": 684,
            "inputs": {
                **formal_binding,
                "identification_subset": str(identification_path),
                "identification_subset_sha256": sha256(identification_path),
            },
            "items": trained_items,
        },
    )
    write_json(
        paths["cog_core"],
        {
            "schema_version": 1,
            "protocol": "causal_role_erasure_7mechanism_baseline_queue_v2",
            "protocol_version": review.PROTOCOL_VERSION,
            "selected_streams": list(review.COGVIDEOX_STREAMS[:4]),
            "status": "frozen_after_full_media_validation",
            "video_count": 1176,
            "inputs": dict(formal_binding),
            "items": core_items,
        },
    )
    write_json(
        paths["safree"],
        {
            "schema_version": 1,
            "protocol": "causal_role_erasure_7mechanism_baseline_queue_v2",
            "protocol_version": review.PROTOCOL_VERSION,
            "selected_streams": ["safree_cogvideox"],
            "status": "frozen_after_full_media_validation",
            "video_count": 294,
            "inputs": dict(formal_binding),
            "items": safree_items,
        },
    )
    paths["blind_key"].parent.mkdir(parents=True)
    paths["blind_key"].write_bytes(b"formal-review-blind-key-material-01")
    return paths


@pytest.fixture(scope="module")
def built_package(tmp_path_factory):
    root = tmp_path_factory.mktemp("formal_review_v1")
    paths = frozen_fixture(root)
    output = root / "review_package"
    receipt = review.build_review_package(
        project_root=root,
        formal_cases_path=paths["formal"],
        identification_path=paths["identification"],
        wan_original_manifest_path=paths["wan_original"],
        trained_wan_manifest_path=paths["trained_wan"],
        cog_core_manifest_path=paths["cog_core"],
        safree_manifest_path=paths["safree"],
        blind_key_path=paths["blind_key"],
        output_dir=output,
        decoder=review.decode_video_opencv_for_test,
    )
    return root, paths, output, receipt


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_cpu_fake_video_covers_all_49_frames_in_one_five_panel_image(tmp_path):
    video = tmp_path / "fake.mp4"
    output = tmp_path / "panel.jpg"
    make_fake_video(video, 832)

    decoded = review.decode_video_opencv_for_test(video)
    artifact = review.render_composite(decoded.frames, output)

    assert decoded.media() == {
        "video_streams": 1,
        "audio_streams": 0,
        "decoded_frames": 49,
        "fps": "8/1",
        "width": 832,
        "height": 480,
    }
    assert artifact["used_frame_indices"] == [list(range(start, end + 1)) for start, end in review.PANEL_WINDOWS]
    assert set().union(*(set(row) for row in artifact["used_frame_indices"])) == set(range(49))
    assert output.stat().st_size <= 3_145_728
    with Image.open(output) as image:
        assert image.size == (review.COMPOSITE_WIDTH, review.COMPOSITE_HEIGHT)


def test_exact_2448_inventory_two_independent_blind_passes_and_private_modes(built_package):
    _, _, output, receipt = built_package
    pass_a = read_jsonl(output / "public" / "pass_a" / "assignments.jsonl")
    pass_b = read_jsonl(output / "public" / "pass_b" / "assignments.jsonl")
    scores_a = read_jsonl(output / "public" / "pass_a" / "scores.jsonl")
    ledger = read_jsonl(output / "private" / "generation_ledger.jsonl")
    full_key = read_jsonl(output / "private" / "full_key.jsonl")
    original_key = read_jsonl(output / "private" / "original_only_key.jsonl")

    assert receipt["status"] == "frozen_after_exact_2448_media_and_blinding_validation"
    assert len(ledger) == len(full_key) == len(pass_a) == len(pass_b) == len(scores_a) == 2448
    assert len(original_key) == 588
    assert sum(row["evaluation_partition"] == "main" for row in ledger) == 2352
    assert sum(row["evaluation_partition"] == "identification" for row in ledger) == 96
    assert {row["stream"] for row in ledger if row["evaluation_partition"] == "main"} == set(review.MAIN_STREAMS)
    assert {row["stream"] for row in ledger if row["evaluation_partition"] == "identification"} == set(review.IDENTIFICATION_STREAMS)
    order_a = [row["anonymous_review_id"] for row in pass_a]
    order_b = [row["anonymous_review_id"] for row in pass_b]
    assert order_a != order_b
    assert set(order_a) == set(order_b)
    assert len(set(order_a)) == 2448
    assert all(set(row) == set(review.PUBLIC_ASSIGNMENT_FIELDS) for row in pass_a)
    for row in scores_a:
        fields = review.CAUSAL_SCORE_FIELDS if row["case_kind"] == "causal" else review.SPECIFICITY_SCORE_FIELDS
        assert row["scores"] == {field: None for field in fields}
        assert row["confidence"] == {field: None for field in fields}
        assert row["evidence_frames"] == {field: [] for field in fields}
        assert row["evidence_observations"] == {field: [] for field in fields}
        assert row["status"] == "pending"
    for private_name in ("generation_ledger.jsonl", "full_key.jsonl", "original_only_key.jsonl", "package_receipt.json"):
        assert stat.S_IMODE((output / "private" / private_name).stat().st_mode) == 0o600
    assert stat.S_IMODE((output / "private").stat().st_mode) == 0o700


def test_public_contract_has_no_method_backbone_source_path_or_case_id_leak(built_package):
    root, _, output, _ = built_package
    public_json = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((output / "public").rglob("*.json*"))
    )
    for forbidden in (
        '"stream"',
        '"arm"',
        '"backbone"',
        '"video_path"',
        '"source_video_path"',
        "wan_original",
        "matched_control",
        "V4",
        "cogvideox_original",
        "negative_prompt",
        "videoeraser_official",
        "t2vunlearning_adapted",
        "safree_cogvideox",
        "generic_paraphrase",
        "bystander_token",
        str(root),
        "eval7m_m00_c00",
    ):
        assert forbidden not in public_json
    assert "mechanism_name" in public_json
    assert "media/rv_" in public_json


def test_assignment_hashes_and_pass_manifests_bind_exact_public_bytes(built_package):
    _, _, output, _ = built_package
    for pass_name, pass_id in (("pass_a", "A"), ("pass_b", "B")):
        root = output / "public" / pass_name
        manifest = json.loads((root / "pass_manifest.json").read_text(encoding="utf-8"))
        assignments = read_jsonl(root / "assignments.jsonl")
        assert manifest["protocol"] == review.PACKAGE_PROTOCOL
        assert manifest["schema_version"] == 1
        assert manifest["pass_id"] == pass_id
        assert manifest["item_count"] == 2448
        assert manifest["panel_windows"] == [list(window) for window in review.PANEL_WINDOWS]
        assert manifest["assignments"]["sha256"] == sha256(root / "assignments.jsonl")
        assert manifest["blank_scores"]["sha256"] == sha256(root / "scores.jsonl")
        assert manifest["composite_contract"]["path_base"] == "pass_root"
        for row in assignments:
            claimed = row.pop("assignment_sha256")
            assert claimed == hashlib.sha256(review.canonical_json_bytes(row)).hexdigest()
            image_path = root / row["composite_path"]
            assert image_path.is_file() and not image_path.is_symlink()
            assert sha256(image_path) == row["composite_sha256"]


def test_generation_mutation_fails_closed_without_partial_output(tmp_path):
    root = tmp_path / "case"
    paths = frozen_fixture(root)
    manifest = json.loads(paths["wan_original"].read_text(encoding="utf-8"))
    manifest["items"][0]["seed"] += 1
    write_json(paths["wan_original"], manifest)
    output = root / "must_not_exist"

    with pytest.raises(review.ReviewPackageError, match="seed mismatch"):
        review.build_review_package(
            project_root=root,
            formal_cases_path=paths["formal"],
            identification_path=paths["identification"],
            wan_original_manifest_path=paths["wan_original"],
            trained_wan_manifest_path=paths["trained_wan"],
            cog_core_manifest_path=paths["cog_core"],
            safree_manifest_path=paths["safree"],
            blind_key_path=paths["blind_key"],
            output_dir=output,
            decoder=review.decode_video_opencv_for_test,
        )
    assert not output.exists()
    assert not list(root.glob(".must_not_exist.tmp-*"))


def test_missing_wan_original_manifest_refuses_before_output(tmp_path):
    root = tmp_path / "case"
    root.mkdir()
    output = root / "review"
    with pytest.raises(review.ReviewPackageError, match="missing or symlinked"):
        review.build_review_package(
            project_root=root,
            formal_cases_path=Path("formal.csv"),
            identification_path=Path("identification.csv"),
            wan_original_manifest_path=Path("missing-wan-original.json"),
            trained_wan_manifest_path=Path("trained.json"),
            cog_core_manifest_path=Path("core.json"),
            safree_manifest_path=Path("safree.json"),
            blind_key_path=Path("blind.key"),
            output_dir=output,
        )
    assert not output.exists()
