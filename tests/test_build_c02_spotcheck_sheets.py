import csv
import importlib.util
import json
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "build_c02_spotcheck_sheets.py"


def load_module():
    spec = importlib.util.spec_from_file_location("build_c02_spotcheck_sheets", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_answer_key_and_strips(tmp_path: Path) -> tuple[Path, Path]:
    variants = ["original", "remove_target", "footprint_only", "target_only"]
    key_path = tmp_path / "answer_key.csv"
    strip_dir = tmp_path / "frame_strips"
    strip_dir.mkdir()
    rows = []
    for item_index in ["3", "10"]:
        for seed_index in ["0", "1"]:
            for variant in variants:
                review_id = f"c01_{item_index}_s{seed_index}_{variant}"
                Image.new("RGB", (80, 32), color=(int(item_index) * 10, int(seed_index) * 50, 20)).save(
                    strip_dir / f"{review_id}.jpg"
                )
                rows.append(
                    {
                        "review_id": review_id,
                        "pair_id": f"pair_{item_index}",
                        "item_index": item_index,
                        "seed_index": seed_index,
                        "seed": f"52{item_index}{seed_index}",
                        "variant": variant,
                        "expected_target_visible": "yes" if variant in {"original", "target_only"} else "no",
                        "expected_footprint_visible": "yes" if variant in {"original", "footprint_only"} else "no",
                        "prompt": f"{variant} prompt",
                        "video_path": f"videos/{review_id}.mp4",
                    }
                )
    with key_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return key_path, strip_dir


def test_build_spotcheck_sheets_writes_one_sheet_per_item(tmp_path):
    module = load_module()
    answer_key, strip_dir = write_answer_key_and_strips(tmp_path)
    output_dir = tmp_path / "spotcheck"

    result = module.main(
        [
            "--answer-key",
            str(answer_key),
            "--frame-strip-dir",
            str(strip_dir),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert result == 0
    manifest = json.loads((output_dir / "spotcheck_manifest.json").read_text(encoding="utf-8"))
    assert manifest["item_count"] == 2
    assert manifest["sheet_count"] == 2
    assert manifest["missing_strip_count"] == 0
    assert {
        "item_3_all_seeds_four_cells.jpg",
        "item_10_all_seeds_four_cells.jpg",
    } == {Path(row["sheet_path"]).name for row in manifest["items"]}
    for row in manifest["items"]:
        assert Path(row["sheet_path"]).exists()
