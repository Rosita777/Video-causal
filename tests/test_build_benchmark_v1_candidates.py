from pathlib import Path
import csv
import json
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def write_fixture_files(tmp_path: Path) -> tuple[Path, Path, Path]:
    v0_items = tmp_path / "items.jsonl"
    v0_items.write_text(
        json.dumps(
            {
                "item_id": "valid5:fluid_pebble",
                "source_name": "valid5",
                "pair_id": "fluid_pebble",
                "mechanism_type": "fluid_impact",
                "temporal_type": "delayed",
                "target_concept": "pebble",
                "expected_effect": "ripples spread outward",
                "source_prompt": "A pebble drops into still water, causing ripples to spread outward.",
                "counterfactual_prompt": "Still water with no pebble and no ripples.",
                "control_prompt": "Wind texture on water with no pebble.",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    round6 = tmp_path / "round6.tsv"
    round6.write_text(
        "round6_id\tmechanism_type\ttarget_concept\tcausal_footprint\tsource_prompt\tnotes\n"
        "fluid_pebble\tfluid_impact\tpebble\tripples spread outward\t"
        "Duplicate row that should be skipped.\tduplicate\n"
        "round6_surface_stamp\t surface_trace \tstamp\tink mark remains\t"
        "A stamp presses on paper, causing an ink mark to remain.\ttrim mechanism whitespace\n",
        encoding="utf-8",
    )

    legacy = tmp_path / "legacy.tsv"
    legacy.write_text(
        "pair_id\ttarget_concept\tcausal_footprint\tmechanism_type\ttemporal_type\t"
        "exclusivity_score\tcounterfactual_clarity\tgeneratability_score\t"
        "erasure_targetability\tstatus\tpair_source\tcausal_chain\tsource_prompt\t"
        "counterfactual_prompt\tcontrol_prompt\tnotes\n"
        "legacy_crack\trock\tcrack spreads\tfracture_damage\tpersistent\t"
        "5\t5\t5\t5\taccepted_v0_slice\tmanual\trock hits glass -> crack spreads\t"
        "A rock hits glass, causing cracks to spread.\tIntact glass with no rock.\t"
        "Old cracked glass with no rock.\tlegacy supplemental row\n",
        encoding="utf-8",
    )
    return v0_items, round6, legacy


def test_build_benchmark_v1_candidates_exports_candidates_prompts_and_controls(tmp_path):
    v0_items, round6, legacy = write_fixture_files(tmp_path)
    output_dir = tmp_path / "benchmark_v1"
    prompt_dir = tmp_path / "prompts"

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "build_benchmark_v1_candidates.py"),
            "--v0-items",
            str(v0_items),
            "--round6-candidates",
            str(round6),
            "--legacy-candidates",
            str(legacy),
            "--output-dir",
            str(output_dir),
            "--prompt-dir",
            str(prompt_dir),
            "--target-count",
            "3",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Wrote 3 candidate items" in result.stdout
    candidate_items = [
        json.loads(line)
        for line in (output_dir / "candidate_items.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [row["pair_id"] for row in candidate_items] == [
        "fluid_pebble",
        "round6_surface_stamp",
        "legacy_crack",
    ]
    assert candidate_items[0]["source_collection"] == "causal_footprint_v0"
    assert candidate_items[1]["mechanism_type"] == "surface_trace"
    assert candidate_items[2]["source_collection"] == "legacy_candidate_pairs"

    with (output_dir / "candidate_pairs.tsv").open(newline="", encoding="utf-8") as handle:
        tsv_rows = list(csv.DictReader(handle, delimiter="\t"))
    assert len(tsv_rows) == 3
    assert all(row["status"] == "candidate_v1" for row in tsv_rows)

    prompt_lines = [
        line
        for line in (prompt_dir / "causal_footprint_v1_candidates.txt").read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    ]
    assert prompt_lines[0] == (
        "A pebble drops into still water, causing ripples to spread outward. | "
        "pebble | ripples spread outward"
    )

    controls = [
        json.loads(line)
        for line in (output_dir / "controls_specs.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(controls) == 9
    assert {row["control_type"] for row in controls} == {
        "no_cause",
        "effect_only",
        "alternative_cause",
    }
    assert controls[0]["control_id"] == "fluid_pebble__no_cause"
    assert controls[0]["source_pair_id"] == "fluid_pebble"
    assert controls[0]["expected_target_presence"] == "no"
    assert controls[0]["expected_footprint_presence"] == "no"
    assert "no pebble" in controls[0]["prompt"]

    manifest = json.loads((output_dir / "export_candidate_manifest.json").read_text(encoding="utf-8"))
    assert manifest["count"] == 3
    assert manifest["items"][1]["pair_id"] == "round6_surface_stamp"


def test_build_benchmark_v1_candidates_fails_when_pool_is_too_small(tmp_path):
    v0_items, round6, legacy = write_fixture_files(tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "build_benchmark_v1_candidates.py"),
            "--v0-items",
            str(v0_items),
            "--round6-candidates",
            str(round6),
            "--legacy-candidates",
            str(legacy),
            "--output-dir",
            str(tmp_path / "out"),
            "--prompt-dir",
            str(tmp_path / "prompts"),
            "--target-count",
            "4",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "only 3 unique candidate rows available" in result.stderr
