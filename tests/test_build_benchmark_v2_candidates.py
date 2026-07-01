from pathlib import Path
import csv
import json
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def write_v1_items(tmp_path: Path) -> Path:
    path = tmp_path / "candidate_items.jsonl"
    rows = [
        {
            "index": 0,
            "item_id": "v1:fluid_pebble",
            "pair_id": "fluid_pebble",
            "source_collection": "test_v1",
            "source_item_id": "fluid_pebble",
            "pair_source": "fixture",
            "mechanism_type": "fluid_impact",
            "temporal_type": "delayed",
            "target_concept": "pebble",
            "expected_effect": "ripples spread outward",
            "causal_footprint": "ripples spread outward",
            "causal_chain": "pebble appears -> ripples spread outward",
            "source_prompt": "A pebble drops into clear still water, causing ripples to spread outward.",
            "counterfactual_prompt": "Clear still water with no pebble and no ripples.",
            "control_prompt": "Ripples spread across water with no pebble.",
            "candidate_status": "candidate_v1",
            "scores": {
                "exclusivity_score": 5,
                "counterfactual_clarity": 5,
                "generatability_score": 5,
                "erasure_targetability": 5,
            },
            "notes": "fixture row",
        }
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def test_build_benchmark_v2_candidates_exports_v1_plus_targeted_expansion(tmp_path):
    v1_items = write_v1_items(tmp_path)
    output_dir = tmp_path / "benchmark_v2"
    prompt_dir = tmp_path / "prompts"

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "build_benchmark_v2_candidates.py"),
            "--v1-items",
            str(v1_items),
            "--output-dir",
            str(output_dir),
            "--prompt-dir",
            str(prompt_dir),
            "--target-count",
            "8",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Wrote 8 candidate items" in result.stdout

    candidate_items = [
        json.loads(line)
        for line in (output_dir / "candidate_items.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(candidate_items) == 8
    assert candidate_items[0]["item_id"] == "v2:fluid_pebble"
    assert candidate_items[0]["source_collection"] == "causal_footprint_v1"
    assert candidate_items[0]["candidate_status"] == "candidate_v2"
    assert candidate_items[1]["source_collection"] == "causal_footprint_v2_targeted_expansion"
    assert candidate_items[1]["pair_source"] == "v2_targeted_expansion"
    assert candidate_items[1]["mechanism_type"] == "fracture_damage"
    assert "no pre-existing" in candidate_items[1]["source_prompt"]
    assert "effect begins only after contact" in candidate_items[1]["source_prompt"]

    with (output_dir / "candidate_pairs.tsv").open(newline="", encoding="utf-8") as handle:
        tsv_rows = list(csv.DictReader(handle, delimiter="\t"))
    assert len(tsv_rows) == 8
    assert all(row["status"] == "candidate_v2" for row in tsv_rows)
    assert len({row["pair_id"] for row in tsv_rows}) == 8

    controls = [
        json.loads(line)
        for line in (output_dir / "controls_specs.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(controls) == 24
    assert {row["control_type"] for row in controls} == {
        "no_cause",
        "effect_only",
        "alternative_cause",
    }
    assert controls[0]["source_name"] == "causal_footprint_v2"
    assert controls[0]["source_baseline"] == "candidate_v2_clean_screening"

    prompt_lines = [
        line
        for line in (prompt_dir / "causal_footprint_v2_candidates.txt").read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    ]
    assert len(prompt_lines) == 8
    assert prompt_lines[0].endswith(" | pebble | ripples spread outward")

    manifest = json.loads((output_dir / "export_candidate_manifest.json").read_text(encoding="utf-8"))
    assert manifest["slice_name"] == "causal_footprint_v2_candidates"
    assert manifest["count"] == 8
    assert manifest["items"][1]["pair_source"] == "v2_targeted_expansion"
