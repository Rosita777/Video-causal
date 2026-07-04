import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "build_c03_scope_locked_candidates.py"


def test_candidate_builder_writes_pre_registered_manifest(tmp_path):
    output = tmp_path / "candidate_manifest.json"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--output", str(output)],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    manifest = json.loads(output.read_text(encoding="utf-8"))
    assert manifest["probe_name"] == "c03_scope_locked_surface_trace_candidates"
    assert manifest["count"] == 8
    assert manifest["candidate_scope"] == "low_entanglement_rigid_object_surface_trace"
    assert [item["probe_index"] for item in manifest["items"]] == list(range(8))
    assert all(item["prior_seen"] is False for item in manifest["items"])
    assert all(
        item["prompt_template_id"] == "c03_scope_locked"
        for item in manifest["items"]
    )
    assert all(item["surface_or_object"] for item in manifest["items"])
    assert all(item["causal_footprint"] for item in manifest["items"])
    assert all(item["causal_footprint_absence"] for item in manifest["items"])
    assert all("scope_predicates_met" in item for item in manifest["items"])


def test_candidate_manifest_is_runner_compatible(tmp_path):
    output = tmp_path / "candidate_manifest.json"
    subprocess.run(
        [sys.executable, str(SCRIPT), "--output", str(output)],
        cwd=PROJECT_ROOT,
        check=True,
    )
    manifest = json.loads(output.read_text(encoding="utf-8"))
    required = {
        "probe_index",
        "pair_id",
        "mechanism_type",
        "source_prompt",
        "generation_prompt",
        "counterfactual_prompt",
        "control_prompt",
        "target_concept",
        "causal_footprint",
        "causal_footprint_absence",
        "surface_or_object",
    }
    for item in manifest["items"]:
        assert required <= set(item)
