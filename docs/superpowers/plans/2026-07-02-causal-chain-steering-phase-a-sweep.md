# Causal Chain Steering Phase A Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a small, testable Phase A sweep orchestrator for ZeroScope causal-chain steering so alpha/window cells can be dry-run or executed consistently.

**Architecture:** Keep the existing diffusion runner unchanged. Add one orchestration script that expands alpha/window grids into runner invocations, writes a sweep manifest, and optionally executes a limited number of cells. Unit tests exercise planning and subprocess behavior without loading models.

**Tech Stack:** Python standard library, pytest, existing `scripts/adapters/run_mvp0_zeroscope_probe.py`.

---

### Task 1: Add Phase A Sweep Planner Tests

**Files:**
- Create: `tests/test_run_mvp0_zeroscope_sweep.py`
- Create later: `scripts/adapters/run_mvp0_zeroscope_sweep.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_run_mvp0_zeroscope_sweep.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python -m pytest tests/test_run_mvp0_zeroscope_sweep.py -q
```

Expected: FAIL because `scripts/adapters/run_mvp0_zeroscope_sweep.py` does not exist.

### Task 2: Implement Phase A Sweep Orchestrator

**Files:**
- Create: `scripts/adapters/run_mvp0_zeroscope_sweep.py`
- Test: `tests/test_run_mvp0_zeroscope_sweep.py`

- [ ] **Step 1: Write minimal implementation**

Create `scripts/adapters/run_mvp0_zeroscope_sweep.py` with:

```python
#!/usr/bin/env python3
"""Run or dry-run Phase A alpha/window sweeps for the MVP-0 ZeroScope probe."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNNER = PROJECT_ROOT / "scripts" / "adapters" / "run_mvp0_zeroscope_probe.py"
DEFAULT_CONDITIONS = [
    "target_negative",
    "target_footprint_negative",
    "full_chain_steering",
    "random_direction",
    "orthogonal_semantic",
]


@dataclass
class CompletedCell:
    returncode: int
    stdout: str
    stderr: str


def parse_alpha_grid(value: str) -> list[float]:
    alphas = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not alphas:
        raise argparse.ArgumentTypeError("alpha grid must not be empty")
    if any(alpha <= 0 for alpha in alphas):
        raise argparse.ArgumentTypeError("alpha values must be positive")
    return alphas


def parse_window(value: str) -> tuple[int, int]:
    left, sep, right = value.partition(":")
    if not sep:
        raise argparse.ArgumentTypeError("window must be START:END")
    start = int(left)
    end = int(right)
    if start < 0 or end < start:
        raise argparse.ArgumentTypeError("expected 0 <= START <= END")
    return start, end


def parse_window_grid(value: str) -> list[tuple[int, int]]:
    windows = [parse_window(part.strip()) for part in value.split(",") if part.strip()]
    if not windows:
        raise argparse.ArgumentTypeError("window grid must not be empty")
    return windows


def alpha_slug(alpha: float) -> str:
    return f"{alpha:.2f}".rstrip("0").rstrip(".").replace(".", "p")


def build_runner_argv(args: argparse.Namespace, alpha: float, window: tuple[int, int], cell_dir: Path) -> list[str]:
    argv = [
        sys.executable,
        str(RUNNER),
        "--probe-manifest",
        str(args.probe_manifest),
        "--output-dir",
        str(cell_dir),
        "--model",
        args.model,
        "--seed",
        str(args.seed),
        "--steps",
        str(args.steps),
        "--guidance-scale",
        str(args.guidance_scale),
        "--num-frames",
        str(args.num_frames),
        "--fps",
        str(args.fps),
        "--height",
        str(args.height),
        "--width",
        str(args.width),
        "--dtype",
        args.dtype,
        "--device",
        args.device,
        "--limit-items",
        str(args.limit_items),
        "--alpha",
        str(alpha),
        "--timestep-window",
        f"{window[0]}:{window[1]}",
        "--strict-prompt-length",
    ]
    for condition in args.condition:
        argv.extend(["--condition", condition])
    if args.enable_model_cpu_offload:
        argv.append("--enable-model-cpu-offload")
    if args.enable_sequential_cpu_offload:
        argv.append("--enable-sequential-cpu-offload")
    if args.vae_slicing:
        argv.append("--vae-slicing")
    if args.dry_run:
        argv.append("--dry-run")
    return argv


def build_sweep_cells(args: argparse.Namespace) -> list[dict[str, object]]:
    cells: list[dict[str, object]] = []
    for alpha in args.alpha_grid:
        for window in args.timestep_window_grid:
            cell_id = f"alpha_{alpha_slug(alpha)}_window_{window[0]}_{window[1]}"
            cell_dir = args.output_dir / cell_id
            cells.append(
                {
                    "cell_id": cell_id,
                    "alpha": alpha,
                    "timestep_window": [window[0], window[1]],
                    "conditions": list(args.condition),
                    "output_dir": str(cell_dir),
                    "runner_argv": build_runner_argv(args, alpha, window, cell_dir),
                    "status": "planned",
                }
            )
    return cells


def run_subprocess(cmd: list[str], cwd: Path, text: bool, capture_output: bool) -> CompletedCell:
    result = subprocess.run(cmd, cwd=cwd, text=text, capture_output=capture_output)
    return CompletedCell(result.returncode, result.stdout, result.stderr)


def write_sweep_manifest(args: argparse.Namespace, cells: list[dict[str, object]], executed_cells: int) -> Path:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "phase_a_conservative_parameter_sweep",
        "probe_manifest": str(args.probe_manifest),
        "model": args.model,
        "dry_run": args.dry_run,
        "total_cells": len(cells),
        "executed_cells": executed_cells,
        "cells": cells,
    }
    path = args.output_dir / "sweep_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--probe-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="models/zeroscope_v2_576w")
    parser.add_argument("--seed", type=int, default=15000)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--guidance-scale", type=float, default=9.0)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--width", type=int, default=432)
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit-items", type=int, default=3)
    parser.add_argument("--condition", action="append", choices=DEFAULT_CONDITIONS, default=[])
    parser.add_argument("--alpha-grid", type=parse_alpha_grid, default=parse_alpha_grid("0.15,0.25,0.35"))
    parser.add_argument("--timestep-window-grid", type=parse_window_grid, default=parse_window_grid("2:5,3:6,4:7"))
    parser.add_argument("--max-cells", type=int)
    parser.add_argument("--enable-model-cpu-offload", action="store_true")
    parser.add_argument("--enable-sequential-cpu-offload", action="store_true")
    parser.add_argument("--vae-slicing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.condition:
        args.condition = list(DEFAULT_CONDITIONS)
    if args.limit_items <= 0:
        parser.error("--limit-items must be positive")
    if args.max_cells is not None and args.max_cells < 0:
        parser.error("--max-cells must be non-negative")
    cells = build_sweep_cells(args)
    to_execute = cells if args.max_cells is None else cells[: args.max_cells]
    failures = 0
    for cell in to_execute:
        result = run_subprocess(
            list(cell["runner_argv"]),
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
        )
        cell["returncode"] = result.returncode
        cell["stdout"] = result.stdout
        cell["stderr"] = result.stderr
        cell["status"] = "completed" if result.returncode == 0 else "failed"
        if result.returncode != 0:
            failures += 1
    manifest = write_sweep_manifest(args, cells, len(to_execute))
    print(f"Wrote Phase A sweep manifest: {manifest}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run tests to verify they pass**

Run:

```bash
python -m pytest tests/test_run_mvp0_zeroscope_sweep.py -q
```

Expected: PASS.

### Task 3: Run Phase A Dry-Run Manifest Sweep

**Files:**
- Existing: `experiments/method_probe/zeroscope_mvp0_causal_chain_probe_20260702/probe_manifest.json`
- Output: `experiments/method_probe/zeroscope_mvp0_causal_chain_probe_20260702/phase_a_sweep_dry_run/sweep_manifest.json`

- [ ] **Step 1: Execute dry-run sweep**

Run:

```bash
PYTHONNOUSERSITE=1 /home/deepseek_VG/.conda/envs/vcecf/bin/python scripts/adapters/run_mvp0_zeroscope_sweep.py \
  --probe-manifest experiments/method_probe/zeroscope_mvp0_causal_chain_probe_20260702/probe_manifest.json \
  --output-dir experiments/method_probe/zeroscope_mvp0_causal_chain_probe_20260702/phase_a_sweep_dry_run \
  --model models/zeroscope_v2_576w \
  --seed 15000 \
  --dry-run \
  --limit-items 3
```

Expected: `sweep_manifest.json` reports `total_cells=9`, `executed_cells=9`,
and every cell status is `completed`.

- [ ] **Step 2: Inspect one generated runner manifest**

Run:

```bash
python - <<'PY'
import json
from pathlib import Path
p = Path("experiments/method_probe/zeroscope_mvp0_causal_chain_probe_20260702/phase_a_sweep_dry_run/alpha_0p15_window_2_5/generation_manifest.json")
m = json.loads(p.read_text())
print(m["dry_run"], len(m["items"]), m["generation"]["steering_alpha"], m["generation"]["timestep_window"])
PY
```

Expected output:

```text
True 15 0.15 [2, 5]
```

### Task 4: Run One Real Phase A Cell

**Files:**
- Output: `experiments/method_probe/zeroscope_mvp0_causal_chain_probe_20260702/phase_a_real_alpha_0p25_window_3_6/`

- [ ] **Step 1: Execute one conservative real cell**

Run on the emptiest GPU:

```bash
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=<GPU_ID> PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/deepseek_VG/.conda/envs/vcecf/bin/python scripts/adapters/run_mvp0_zeroscope_probe.py \
  --probe-manifest experiments/method_probe/zeroscope_mvp0_causal_chain_probe_20260702/probe_manifest.json \
  --output-dir experiments/method_probe/zeroscope_mvp0_causal_chain_probe_20260702/phase_a_real_alpha_0p25_window_3_6 \
  --model models/zeroscope_v2_576w \
  --seed 15000 \
  --condition target_negative \
  --condition target_footprint_negative \
  --condition full_chain_steering \
  --condition random_direction \
  --condition orthogonal_semantic \
  --limit-items 3 \
  --steps 10 \
  --num-frames 16 \
  --height 240 \
  --width 432 \
  --alpha 0.25 \
  --timestep-window 3:6 \
  --enable-model-cpu-offload \
  --vae-slicing \
  --strict-prompt-length
```

Expected: 15 videos and a `generation_manifest.json`.

- [ ] **Step 2: Verify video files open**

Run:

```bash
PYTHONNOUSERSITE=1 /home/deepseek_VG/.conda/envs/vcecf/bin/python - <<'PY'
from pathlib import Path
import cv2
root = Path("experiments/method_probe/zeroscope_mvp0_causal_chain_probe_20260702/phase_a_real_alpha_0p25_window_3_6/videos")
for path in sorted(root.glob("*.mp4")):
    cap = cv2.VideoCapture(str(path))
    print(path.name, cap.isOpened(), int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    cap.release()
PY
```

Expected: 15 lines, each with `True 16`.

### Task 5: Evaluate the Real Cell

**Files:**
- Output: `experiments/evaluation/mvp0_zeroscope_phase_a_alpha_0p25_window_3_6_fable_20260702/`

- [ ] **Step 1: Build review CSV and frame strips**

Use the same review-builder helper pattern from the previous pilot to produce:

```text
review.csv
frame_strips/*.jpg
```

Expected: 18 rows: 3 clean references plus 15 outputs.

- [ ] **Step 2: Run fable VLM evaluation**

Run `scripts/evaluate_v2_baseline_with_vlm.py` with the fable token loaded into
environment variables, not printed on the command line.

Expected: `vlm_predictions.csv` contains 15 rows.

- [ ] **Step 3: Compute low-level proxies**

Compute the same metrics as the previous pilot:

```text
mean_absdiff
mean_flow
edge_density
laplacian_var
brightness
contrast
```

Expected: `low_level_proxy.csv` and `low_level_proxy_summary.csv` exist.

- [ ] **Step 4: Apply Phase A gate**

Compare fable labels by condition:

```text
full_chain_steering strict_causal_footprint_leakage count
random_direction strict_causal_footprint_leakage count
orthogonal_semantic strict_causal_footprint_leakage count
target_footprint_negative target_leakage count
full_chain_steering target_leakage count
```

Expected: decide whether this cell deserves more grid cells. If full-chain does
not beat both controls, record it as another negative diagnostic cell and run a
different alpha/window before considering Phase B.

### Task 6: Update Experiment Log

**Files:**
- Modify: `docs/experiment_log.md`

- [ ] **Step 1: Append Phase A dry-run and real-cell results**

Append:

```markdown
## 2026-07-02: Phase A Conservative Sweep

...
```

Include the dry-run manifest path, real cell path, fable summary, low-level
proxy summary, and the Phase A gate decision.

- [ ] **Step 2: Run focused tests**

Run:

```bash
python -m pytest tests/test_run_mvp0_zeroscope_sweep.py tests/test_run_mvp0_zeroscope_probe.py tests/test_build_mvp0_causal_chain_probe.py -q
```

Expected: PASS.

### No-Git Note

This workspace directory is not currently a git repository. Replace commit
steps with explicit status notes in `docs/experiment_log.md`.
