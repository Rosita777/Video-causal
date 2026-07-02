from pathlib import Path
import importlib.util
import json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SWEEP_PATH = PROJECT_ROOT / "scripts" / "adapters" / "run_mvp0_zeroscope_sweep.py"


def load_sweep_module():
    spec = importlib.util.spec_from_file_location("run_mvp0_zeroscope_sweep", SWEEP_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_sweep_cells_expands_alpha_window_grid(tmp_path):
    module = load_sweep_module()
    args = module.build_parser().parse_args(
        [
            "--probe-manifest",
            "probe.json",
            "--output-dir",
            str(tmp_path / "sweep"),
            "--alpha-grid",
            "0.15,0.25",
            "--timestep-window-grid",
            "2:5,3:6",
            "--condition",
            "target_negative",
            "--condition",
            "full_chain_steering",
            "--limit-items",
            "3",
            "--dry-run",
        ]
    )

    cells = module.build_sweep_cells(args)

    assert [cell["cell_id"] for cell in cells] == [
        "alpha_0p15_window_2_5",
        "alpha_0p15_window_3_6",
        "alpha_0p25_window_2_5",
        "alpha_0p25_window_3_6",
    ]
    first = cells[0]
    assert first["alpha"] == 0.15
    assert first["timestep_window"] == [2, 5]
    assert first["output_dir"].endswith("alpha_0p15_window_2_5")
    assert "--alpha" in first["runner_argv"]
    assert "0.15" in first["runner_argv"]
    assert "--timestep-window" in first["runner_argv"]
    assert "2:5" in first["runner_argv"]
    assert first["runner_argv"].count("--condition") == 2
    assert "--dry-run" in first["runner_argv"]


def test_default_phase_a_grid_is_three_by_three(tmp_path):
    module = load_sweep_module()
    args = module.build_parser().parse_args(
        [
            "--probe-manifest",
            "probe.json",
            "--output-dir",
            str(tmp_path / "sweep"),
            "--dry-run",
        ]
    )

    cells = module.build_sweep_cells(args)

    assert len(cells) == 9
    assert cells[0]["cell_id"] == "alpha_0p15_window_2_5"
    assert cells[-1]["cell_id"] == "alpha_0p35_window_4_7"
    assert cells[0]["conditions"] == [
        "target_negative",
        "target_footprint_negative",
        "full_chain_steering",
        "random_direction",
        "orthogonal_semantic",
    ]
    assert cells[0]["runner_argv"].count("--condition") == 5


def test_main_executes_limited_cells_and_writes_manifest(tmp_path, monkeypatch):
    module = load_sweep_module()
    calls = []

    def fake_run(cmd, cwd, text, capture_output):
        calls.append({"cmd": cmd, "cwd": cwd, "text": text, "capture_output": capture_output})
        return module.CompletedCell(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(module, "run_subprocess", fake_run)

    result = module.main(
        [
            "--probe-manifest",
            "experiments/probe_manifest.json",
            "--output-dir",
            str(tmp_path / "phase_a"),
            "--alpha-grid",
            "0.15,0.25",
            "--timestep-window-grid",
            "2:5",
            "--dry-run",
            "--max-cells",
            "1",
        ]
    )

    assert result == 0
    assert len(calls) == 1
    assert calls[0]["cmd"][0].endswith("python")
    assert calls[0]["cmd"][1].endswith("scripts/adapters/run_mvp0_zeroscope_probe.py")
    manifest = json.loads((tmp_path / "phase_a" / "sweep_manifest.json").read_text())
    assert manifest["executed_cells"] == 1
    assert manifest["total_cells"] == 2
    assert manifest["cells"][0]["status"] == "completed"
    assert manifest["cells"][1]["status"] == "planned"
