from collections import Counter
import json
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from build_protocol_v1_manifests import (  # noqa: E402
    GENERALIZATION_GROUPS,
    build_eval_rows,
    build_preserve_rows,
    build_train_rows,
    validate,
)


REGISTRY_PATH = PROJECT_ROOT / "data" / "protocol_v1" / "registry.json"


def load_registry():
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def test_protocol_v1_counts_and_disjoint_splits():
    registry = load_registry()
    train = build_train_rows(registry)
    preserve = build_preserve_rows(registry)
    evaluation = build_eval_rows(registry)

    validate(registry, train, preserve, evaluation)
    assert len(train) == 144
    assert len(preserve) == 36
    assert len(evaluation) == 80

    for mechanism, spec in registry["mechanisms"].items():
        assert {item["id"] for item in spec["train_sources"]}.isdisjoint(
            item["id"] for item in spec["test_sources"]
        )
        assert {item["id"] for item in spec["train_receivers"]}.isdisjoint(
            item["id"] for item in spec["test_receivers"]
        )
        groups = Counter(
            row["generalization_group"]
            for row in evaluation
            if row["mechanism"] == mechanism
        )
        assert groups == Counter({group: 5 for group in GENERALIZATION_GROUPS})


def test_protocol_v1_cli_writes_all_hashed_artifacts(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "build_protocol_v1_manifests.py"),
            "--registry",
            str(REGISTRY_PATH),
            "--output-dir",
            str(tmp_path),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads((tmp_path / "manifest_summary.json").read_text(encoding="utf-8"))
    assert summary["counts"] == {"train_erase": 144, "preserve": 36, "eval": 80}
    assert len(summary["sha256"]) == 6
    for filename in (
        "train_erase_manifest.csv",
        "preserve_manifest.csv",
        "eval_manifest.csv",
        "train_erase.prompts",
        "preserve.prompts",
        "eval.prompts",
    ):
        assert (tmp_path / filename).is_file()
