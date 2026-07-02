import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "build_mvp0_phase_b_paraphrase_probe.py"


def test_phase_b_builder_expands_each_link_to_three_pairs(tmp_path):
    source = tmp_path / "probe_manifest.json"
    source.write_text(
        json.dumps(
            {
                "probe_name": "zeroscope_mvp0_causal_chain_probe",
                "items": [
                    {
                        "probe_index": 0,
                        "pair_id": "fluid_impact_pebble_pond_002",
                        "slice_index": 0,
                        "source_index": "0",
                        "mechanism_type": "fluid_impact",
                        "target_concept": "pebble",
                        "causal_footprint": "circular ripples spread outward",
                        "source_prompt": "long source",
                        "generation_prompt": "A pond. pebble causes circular ripples spread outward.",
                        "counterfactual_prompt": "A pond with no pebble.",
                        "control_prompt": "A pond.",
                        "minimal_pairs": {
                            "cause": {"positive": "with pebble", "negative": "without pebble"},
                            "mechanism": {"positive": "with pebble impact", "negative": "with no impact"},
                            "footprint": {"positive": "with ripples", "negative": "with no ripples"},
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "phase_b.json"

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--probe-manifest", str(source), "--output", str(out)],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    manifest = json.loads(out.read_text())
    item = manifest["items"][0]
    assert manifest["probe_name"] == "zeroscope_mvp0_causal_chain_probe_phase_b_paraphrase"
    assert len(item["minimal_pairs"]["cause"]) == 3
    assert len(item["minimal_pairs"]["mechanism"]) == 3
    assert len(item["minimal_pairs"]["footprint"]) == 3
    assert len(item["minimal_pairs"]["orthogonal_semantic"]) == 3
    assert item["minimal_pairs"]["cause"][0] == {"positive": "with pebble", "negative": "without pebble"}
    assert "pebble is visible" in item["minimal_pairs"]["cause"][1]["positive"]
    assert "no circular ripples spread outward" in item["minimal_pairs"]["footprint"][2]["negative"]
    assert item["minimal_pairs"]["orthogonal_semantic"][0] == {
        "positive": "A realistic video with birds flying across the sky.",
        "negative": "A realistic video with no birds in the sky.",
    }
    assert manifest["phase_b_control_method"] == "paraphrase_averaged_norm_matched_orthogonal"
