from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_EVAL_SHA256 = "dca68f8632e10ef83cc5f3867679c9cba54f4cbce96426db5db8c5214ac1ec1a"
EXPECTED_PROMPTS_SHA256 = "06dae57a0202e2d53e32fc02f9b26fd694237755a18f85bdd67c728bf706681c"
EXPECTED_TRAIN_SHA256 = "3d4d8cbf9244b1575357f0ac74380cd7cb6265df4d1a85bf450de4cac120aee4"
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from score_water_impact_dynamic_v3_sampling import review_binding_sha256  # noqa: E402
FIELDS = (
    "target_visibility_0_absent_2_clear",
    "footprint_visibility_0_absent_2_clear",
    "receiver_preservation_0_bad_2_good",
    "video_quality_0_bad_2_good",
)


def write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    review_path = tmp_path / "review.csv"
    key_path = tmp_path / "key.csv"
    review_rows = []
    key_rows = []
    groups = ["unseen_source", "unseen_receiver", "both_unseen"]
    for index in range(12):
        for code, method in (("A", "balanced"), ("B", "exposure")):
            review_id = f"r{index:03d}_{code}"
            if method == "balanced":
                scores = (2, 1 if index < 8 else 2, 2, 2)
            elif index < 3:
                scores = (0, 0, 2, 2) if index == 0 else (1, 1, 2, 2)
            else:
                scores = (2, 1 if index < 8 else 2, 2, 2)
            review_rows.append(
                {
                    "review_id": review_id,
                    "sample_index": index,
                    "pair_id": f"pair_{index}",
                    "generalization_group": groups[index % 3],
                    "candidate_code": code,
                    "composite_path": str(tmp_path / f"r{index:03d}.jpg"),
                    "source_object": "object",
                    "receiver": "receiver",
                    **dict(zip(FIELDS, scores)),
                    "notes": "",
                }
            )
            key_rows.append(
                {
                    "review_id": review_id,
                    "sample_index": index,
                    "pair_id": f"pair_{index}",
                    "generalization_group": groups[index % 3],
                    "candidate_code": code,
                    "method": method,
                    "video_path": f"{method}_{index}.mp4",
                }
            )
    with review_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(review_rows[0]))
        writer.writeheader()
        writer.writerows(review_rows)
    with key_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(key_rows[0]))
        writer.writeheader()
        writer.writerows(key_rows)
    composite_hashes = {}
    for index in range(12):
        composite = tmp_path / f"r{index:03d}.jpg"
        composite.write_bytes(f"composite {index}".encode("utf-8"))
        composite_hashes[composite.name] = hashlib.sha256(composite.read_bytes()).hexdigest()
    generation_manifests = {}
    for label in ("original", "balanced", "exposure"):
        path = tmp_path / f"{label}_generation_manifest.json"
        path.write_text(json.dumps({"label": label}), encoding="utf-8")
        generation_manifests[label] = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    (tmp_path / "review_manifest.json").write_text(
        json.dumps(
            {
                "eval_csv_sha256": EXPECTED_EVAL_SHA256,
                "prompts_sha256": EXPECTED_PROMPTS_SHA256,
                "train_manifest_sha256": EXPECTED_TRAIN_SHA256,
                "sample_count": 12,
                "review_rows": 24,
                "answer_key_sha256": hashlib.sha256(key_path.read_bytes()).hexdigest(),
                "review_binding_sha256": review_binding_sha256(review_rows),
                "composite_sha256": composite_hashes,
                "generation_manifests": generation_manifests,
            }
        ),
        encoding="utf-8",
    )
    return review_path, key_path


def test_promotes_exposure_when_all_registered_checks_pass(tmp_path: Path) -> None:
    review, key = write_inputs(tmp_path)
    output = tmp_path / "scores"
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "score_water_impact_dynamic_v3_sampling.py"),
            "--review",
            str(review),
            "--answer-key",
            str(key),
            "--output-dir",
            str(output),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    gate = json.loads((output / "gate.json").read_text(encoding="utf-8"))
    assert gate["source_positive"] is True
    assert gate["promote_operating_point"] is True
    assert len(gate["source_improvements"]) == 3


def test_rejects_incomplete_scores(tmp_path: Path) -> None:
    review, key = write_inputs(tmp_path)
    rows = list(csv.DictReader(review.open(encoding="utf-8")))
    rows[0][FIELDS[0]] = ""
    with review.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "score_water_impact_dynamic_v3_sampling.py"),
            "--review",
            str(review),
            "--answer-key",
            str(key),
            "--output-dir",
            str(tmp_path / "bad_scores"),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert "invalid target_visibility" in result.stderr


def test_rejects_missing_eval12_pair_even_when_manifest_and_key_match(tmp_path: Path) -> None:
    review, key = write_inputs(tmp_path)
    review_rows = list(csv.DictReader(review.open(encoding="utf-8")))[:-2]
    key_rows = list(csv.DictReader(key.open(encoding="utf-8")))[:-2]
    for path, rows in ((review, review_rows), (key, key_rows)):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    manifest_path = tmp_path / "review_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["answer_key_sha256"] = hashlib.sha256(key.read_bytes()).hexdigest()
    manifest["review_binding_sha256"] = review_binding_sha256(review_rows)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "score_water_impact_dynamic_v3_sampling.py"),
            "--review",
            str(review),
            "--answer-key",
            str(key),
            "--output-dir",
            str(tmp_path / "missing_pair_scores"),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "exactly 24" in result.stderr


def test_rejects_review_group_tampering(tmp_path: Path) -> None:
    review, key = write_inputs(tmp_path)
    rows = list(csv.DictReader(review.open(encoding="utf-8")))
    rows[0]["generalization_group"] = "both_unseen"
    with review.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest_path = tmp_path / "review_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["review_binding_sha256"] = review_binding_sha256(rows)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "score_water_impact_dynamic_v3_sampling.py"),
            "--review",
            str(review),
            "--answer-key",
            str(key),
            "--output-dir",
            str(tmp_path / "tampered_group_scores"),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "answer-key mismatch for generalization_group" in result.stderr


def test_rejects_answer_key_method_tampering(tmp_path: Path) -> None:
    review, key = write_inputs(tmp_path)
    rows = list(csv.DictReader(key.open(encoding="utf-8")))
    rows[0]["method"] = "exposure"
    with key.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "score_water_impact_dynamic_v3_sampling.py"),
            "--review",
            str(review),
            "--answer-key",
            str(key),
            "--output-dir",
            str(tmp_path / "tampered_key_scores"),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "answer-key hash" in result.stderr


def test_rejects_immutable_review_binding_tampering(tmp_path: Path) -> None:
    review, key = write_inputs(tmp_path)
    rows = list(csv.DictReader(review.open(encoding="utf-8")))
    rows[0]["source_object"] = "different object"
    with review.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "score_water_impact_dynamic_v3_sampling.py"),
            "--review",
            str(review),
            "--answer-key",
            str(key),
            "--output-dir",
            str(tmp_path / "tampered_binding_scores"),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "immutable review binding" in result.stderr


def test_rejects_composite_changed_after_review_package(tmp_path: Path) -> None:
    review, key = write_inputs(tmp_path)
    (tmp_path / "r000.jpg").write_bytes(b"changed")

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "score_water_impact_dynamic_v3_sampling.py"),
            "--review",
            str(review),
            "--answer-key",
            str(key),
            "--output-dir",
            str(tmp_path / "changed_composite_scores"),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "composite hash mismatch" in result.stderr


def test_rejects_generation_manifest_changed_after_review_package(tmp_path: Path) -> None:
    review, key = write_inputs(tmp_path)
    (tmp_path / "balanced_generation_manifest.json").write_text(
        json.dumps({"label": "changed"}), encoding="utf-8"
    )

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "score_water_impact_dynamic_v3_sampling.py"),
            "--review",
            str(review),
            "--answer-key",
            str(key),
            "--output-dir",
            str(tmp_path / "changed_manifest_scores"),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "generation manifest hash mismatch" in result.stderr
