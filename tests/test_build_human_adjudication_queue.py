from pathlib import Path
import csv
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_build_human_adjudication_queue_copies_only_unresolved_rows(tmp_path):
    manifest = tmp_path / "manifest.csv"
    output = tmp_path / "queue.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["item_id", "baseline", "adjudicated_label", "human_notes", "vlm_label"])
        writer.writeheader()
        writer.writerows(
            [
                {"item_id": "i1", "baseline": "negative_prompt", "adjudicated_label": "strict_leakage", "human_notes": "", "vlm_label": "strict_leakage"},
                {"item_id": "i2", "baseline": "videoeraser", "adjudicated_label": "", "human_notes": "needs review", "vlm_label": "borderline"},
            ]
        )

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "build_human_adjudication_queue.py"),
            "--manifest",
            str(manifest),
            "--output",
            str(output),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "Wrote 1 human adjudication queue rows" in result.stdout
    rows = list(csv.DictReader(output.open(newline="", encoding="utf-8")))
    assert len(rows) == 1
    assert rows[0]["item_id"] == "i2"
    assert rows[0]["review_status"] == "needs_human_review"
