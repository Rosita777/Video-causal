from pathlib import Path
import json
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_build_benchmark_items_merges_manifest_prompts_and_summary(tmp_path):
    manifest = tmp_path / "manifest.json"
    prompts = tmp_path / "prompts.txt"
    summary = tmp_path / "summary.csv"
    output = tmp_path / "items.jsonl"

    prompts.write_text(
        "# Format: <prompt> | <target> | <effect>\n"
        "\n"
        "A pebble falls into still water, causing ripples. | pebble | ripples spread outward\n",
        encoding="utf-8",
    )
    manifest.write_text(
        json.dumps(
            {
                "output_prompts": str(prompts),
                "items": [
                    {
                        "pair_id": "p1",
                        "target_concept": "pebble",
                        "causal_footprint": "ripples spread outward",
                        "mechanism_type": "fluid_impact",
                        "temporal_type": "delayed",
                        "counterfactual_prompt": "Still water. No pebble.",
                        "control_prompt": "Wind ripples water. No pebble.",
                        "clean_source_valid": "valid",
                        "clean_source_notes": "source chain is clear",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    summary.write_text(
        "run_id,prompt_index,pair_id,mechanism_type,baseline,seed,target_concept,expected_effect,"
        "video_path,video_exists,video_bytes,target_visible,causal_effect_visible,causeless_effect,"
        "video_quality,usable_for_claim,failure_mode,notes\n"
        "run,0,p1,fluid_impact,negative_prompt,7,pebble,ripples spread outward,"
        "outputs/np.mp4,True,123,no,yes,yes,good,yes,causal_footprint_leakage,"
        "ripples remain without pebble\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "build_benchmark_items.py"),
            "--source",
            f"valid5,{manifest},{summary}",
            "--output",
            str(output),
        ],
        cwd=PROJECT_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "Wrote 1 benchmark items" in result.stdout
    item = json.loads(output.read_text(encoding="utf-8"))
    assert item["item_id"] == "valid5:p1"
    assert item["source_name"] == "valid5"
    assert item["source_prompt"] == "A pebble falls into still water, causing ripples."
    assert item["expected_effect"] == "ripples spread outward"
    assert item["clean_reference"]["clean_source_valid"] == "valid"
    assert item["clean_reference"]["video_path"] == ""
    assert item["baseline_outputs"] == [
        {
            "run_id": "run",
            "prompt_index": 0,
            "baseline": "negative_prompt",
            "seed": 7,
            "video_path": "outputs/np.mp4",
            "video_exists": True,
            "video_bytes": 123,
            "target_visible": "no",
            "causal_effect_visible": "yes",
            "causeless_effect": "yes",
            "video_quality": "good",
            "usable_for_claim": "yes",
            "failure_mode": "causal_footprint_leakage",
            "notes": "ripples remain without pebble",
        }
    ]


def test_build_benchmark_items_uses_clean_reference_rows(tmp_path):
    manifest = tmp_path / "manifest.json"
    summary = tmp_path / "summary.csv"
    output = tmp_path / "items.jsonl"

    manifest.write_text(
        json.dumps(
            {
                "items": [
                    {
                        "pair_id": "p2",
                        "target_concept": "rock",
                        "causal_footprint": "crack spreads",
                        "mechanism_type": "fracture_damage",
                        "prompt": "A rock hits a windshield and cracks spread.",
                        "clean_source_valid": "yes",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    summary.write_text(
        "run_id,prompt_index,pair_id,mechanism_type,baseline,seed,target_concept,expected_effect,"
        "video_path,video_exists,video_bytes,target_visible,causal_effect_visible,causeless_effect,"
        "video_quality,usable_for_claim,failure_mode,notes\n"
        "run,0,p2,fracture_damage,clean_reference,11,rock,crack spreads,"
        "outputs/clean.mp4,True,456,yes,yes,no,good,no,clean_reference,reference only\n"
        "run,0,p2,fracture_damage,videoeraser,12,rock,crack spreads,"
        "outputs/ve.mp4,True,321,no,yes,yes,ok,yes,causal_footprint_leakage,crack remains\n",
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "build_benchmark_items.py"),
            "--source",
            f"round4,{manifest},{summary}",
            "--output",
            str(output),
        ],
        cwd=PROJECT_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    item = json.loads(output.read_text(encoding="utf-8"))
    assert item["clean_reference"]["video_path"] == "outputs/clean.mp4"
    assert item["clean_reference"]["seed"] == 11
    assert item["clean_reference"]["video_exists"] is True
    assert [row["baseline"] for row in item["baseline_outputs"]] == ["videoeraser"]
