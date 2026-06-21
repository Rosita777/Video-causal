from pathlib import Path
import json
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_export_benchmark_prompts_filters_status_and_writes_manifest(tmp_path):
    candidates = tmp_path / "candidate_pairs.tsv"
    prompts = tmp_path / "prompts.txt"
    manifest = tmp_path / "manifest.json"
    candidates.write_text(
        "pair_id\ttarget_concept\tcausal_footprint\tmechanism_type\ttemporal_type\t"
        "exclusivity_score\tcounterfactual_clarity\tgeneratability_score\terasure_targetability\t"
        "status\tpair_source\tcausal_chain\tsource_prompt\tcounterfactual_prompt\tcontrol_prompt\tnotes\n"
        "p1\tpebble\tripples\tfluid_impact\tdelayed\t4\t5\t5\t5\taccepted_v0_slice\t"
        "taxonomy\tchain\tA pebble drops into still water, causing ripples.\tStill water.\tNatural waves.\tgood\n"
        "p2\twind\tmoving leaves\tfield_mediated\tsynchronous\t1\t2\t4\t2\trejected\t"
        "taxonomy\tchain\tWind moves leaves.\tStill leaves.\tFan moves leaves.\trejected\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "export_benchmark_prompts.py"),
            "--candidates",
            str(candidates),
            "--output-prompts",
            str(prompts),
            "--output-manifest",
            str(manifest),
            "--status",
            "accepted_v0_slice",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "Exported 1 prompts" in result.stdout
    assert prompts.read_text(encoding="utf-8").splitlines() == [
        "# Exported from candidate_pairs.tsv",
        "# Status filter: accepted_v0_slice",
        "# Format: <prompt> | <target> | <effect>",
        "",
        "A pebble drops into still water, causing ripples. | pebble | ripples",
    ]
    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert data["count"] == 1
    assert data["status"] == "accepted_v0_slice"
    assert data["items"][0]["pair_id"] == "p1"
    assert data["items"][0]["mechanism_type"] == "fluid_impact"


def test_export_benchmark_prompts_rejects_duplicate_pair_ids(tmp_path):
    candidates = tmp_path / "candidate_pairs.tsv"
    candidates.write_text(
        "pair_id\ttarget_concept\tcausal_footprint\tmechanism_type\ttemporal_type\t"
        "exclusivity_score\tcounterfactual_clarity\tgeneratability_score\terasure_targetability\t"
        "status\tpair_source\tcausal_chain\tsource_prompt\tcounterfactual_prompt\tcontrol_prompt\tnotes\n"
        "p1\tpebble\tripples\tfluid_impact\tdelayed\t4\t5\t5\t5\taccepted_v0_slice\t"
        "taxonomy\tchain\tPrompt one.\tCF.\tControl.\tgood\n"
        "p1\tstone\tripples\tfluid_impact\tdelayed\t4\t5\t5\t5\taccepted_v0_slice\t"
        "taxonomy\tchain\tPrompt two.\tCF.\tControl.\tduplicate\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "export_benchmark_prompts.py"),
            "--candidates",
            str(candidates),
            "--output-prompts",
            str(tmp_path / "prompts.txt"),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "duplicate pair_id: p1" in result.stderr


def test_export_benchmark_prompts_filters_clean_source_labels(tmp_path):
    candidates = tmp_path / "candidate_pairs.tsv"
    labels = tmp_path / "clean_labels.csv"
    prompts = tmp_path / "prompts.txt"
    manifest = tmp_path / "manifest.json"
    candidates.write_text(
        "pair_id\ttarget_concept\tcausal_footprint\tmechanism_type\ttemporal_type\t"
        "exclusivity_score\tcounterfactual_clarity\tgeneratability_score\terasure_targetability\t"
        "status\tpair_source\tcausal_chain\tsource_prompt\tcounterfactual_prompt\tcontrol_prompt\tnotes\n"
        "p_valid\tpebble\tripples\tfluid_impact\tdelayed\t4\t5\t5\t5\taccepted_v0_slice\t"
        "taxonomy\tchain\tA pebble drops into water.\tStill water.\tNatural waves.\tgood\n"
        "p_weak\tstone\tripples\tfluid_impact\tdelayed\t4\t5\t5\t5\taccepted_v0_slice\t"
        "taxonomy\tchain\tA stone drops into water.\tStill water.\tNatural waves.\tweak\n"
        "p_rejected\tball\tcracks\tfracture_damage\tdelayed\t4\t5\t5\t5\trejected\t"
        "taxonomy\tchain\tA ball hits glass.\tClean glass.\tOld cracks.\trejected\n",
        encoding="utf-8",
    )
    labels.write_text(
        "pair_id,clean_source_valid,notes\n"
        "p_valid,valid,usable source\n"
        "p_weak,weak,unclear ordering\n"
        "p_rejected,valid,would be excluded by candidate status\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "export_benchmark_prompts.py"),
            "--candidates",
            str(candidates),
            "--clean-labels",
            str(labels),
            "--clean-source-valid",
            "valid",
            "--output-prompts",
            str(prompts),
            "--output-manifest",
            str(manifest),
        ],
        cwd=PROJECT_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "Exported 1 prompts" in result.stdout
    assert prompts.read_text(encoding="utf-8").splitlines()[-1] == "A pebble drops into water. | pebble | ripples"
    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert data["count"] == 1
    assert data["clean_source_valid"] == ["valid"]
    assert data["items"][0]["pair_id"] == "p_valid"
    assert data["items"][0]["clean_source_valid"] == "valid"
    assert data["items"][0]["clean_source_notes"] == "usable source"
