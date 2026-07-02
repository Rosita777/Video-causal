# ZeroScope MVP-0 Causal Chain Probe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a small, testable ZeroScope mechanism probe that checks whether minimal-pair cause/mechanism/footprint denoising directions contain signal beyond prompt-only erasure.

**Architecture:** The first deliverable is a dry-runable probe manifest builder that selects 6-12 clean-valid ZeroScope v2 items and expands each item into prompt-only controls plus minimal-pair steering contracts. The second deliverable is a ZeroScope probe runner skeleton that records all steering conditions in manifests before real denoising steering is enabled. The actual heavy GPU run is gated behind passing dry-run tests and a one-prompt smoke.

**Tech Stack:** Python 3.10, existing project prompt/manifest utilities, pytest, diffusers `TextToVideoSDPipeline` for the later real runner.

---

## File Structure

- Create `scripts/build_mvp0_causal_chain_probe.py`
  - Reads `benchmarks/causal_footprint_v2/zeroscope_clean_valid_gpt54_96_manifest.json`.
  - Optionally reads the merged ZeroScope VLM labels to prioritize strict leakage cases.
  - Selects a small balanced probe slice.
  - Builds deterministic minimal-pair prompts and condition contracts.
  - Writes a probe manifest and prompt files.
- Create `tests/test_build_mvp0_causal_chain_probe.py`
  - Tests selection, minimal-pair construction, and manifest schema.
- Create `scripts/adapters/run_mvp0_zeroscope_probe.py`
  - Dry-run first: records prompt-only controls and steering-contract metadata.
  - Later real mode: loads ZeroScope and executes requested probe conditions.
- Create `tests/test_run_mvp0_zeroscope_probe.py`
  - Tests dry-run manifest contracts and validates guardrails for unsupported real mode.
- Modify `docs/method_candidate_causal_chain_steering.md`
  - Add a short "Implementation Probe Status" section after the first dry-run exists.
- Do not modify the existing ZeroScope baseline scripts unless the probe runner needs a small shared helper.

## Task 1: Probe Manifest Builder

**Files:**
- Create: `scripts/build_mvp0_causal_chain_probe.py`
- Test: `tests/test_build_mvp0_causal_chain_probe.py`

- [ ] **Step 1: Write the failing test for minimal-pair construction**

Create `tests/test_build_mvp0_causal_chain_probe.py` with:

```python
from pathlib import Path
import json
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_probe_builder_writes_minimal_pair_contracts(tmp_path):
    clean_manifest = tmp_path / "clean_valid.json"
    clean_manifest.write_text(
        json.dumps(
            {
                "items": [
                    {
                        "pair_id": "fluid_impact_pebble_pond_002",
                        "source_index": "0",
                        "slice_index": 0,
                        "mechanism_type": "fluid_impact",
                        "target_concept": "pebble",
                        "causal_footprint": "circular ripples spread outward",
                        "source_prompt": "A realistic close-up video of a small pebble dropping into a still pond, causing circular ripples to spread outward across the water.",
                        "counterfactual_prompt": "A realistic close-up video of a still pond with a calm, undisturbed surface. No pebble is present.",
                        "control_prompt": "A realistic close-up video of a still pond where gentle background ripples move across the surface with no impact point.",
                        "clean_video_path": "outputs/clean0.mp4",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "probe"

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "build_mvp0_causal_chain_probe.py"),
            "--clean-valid-manifest",
            str(clean_manifest),
            "--output-dir",
            str(output_dir),
            "--limit",
            "1",
            "--dry-run",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    manifest = json.loads((output_dir / "probe_manifest.json").read_text(encoding="utf-8"))
    assert manifest["probe_name"] == "zeroscope_mvp0_causal_chain_probe"
    assert manifest["dry_run"] is True
    assert manifest["items"][0]["pair_id"] == "fluid_impact_pebble_pond_002"
    pairs = manifest["items"][0]["minimal_pairs"]
    assert set(pairs) == {"cause", "mechanism", "footprint"}
    assert pairs["cause"]["positive"].startswith("A realistic close-up video")
    assert "pebble" in pairs["cause"]["positive"]
    assert "without pebble" in pairs["cause"]["negative"]
    assert "impact" in pairs["mechanism"]["positive"].lower()
    assert "no impact" in pairs["mechanism"]["negative"].lower()
    assert "circular ripples" in pairs["footprint"]["positive"]
    assert "no circular ripples" in pairs["footprint"]["negative"]
    assert (output_dir / "prompts" / "source_prompts.txt").exists()
    assert (output_dir / "prompts" / "counterfactual_prompts.txt").exists()
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
python -m pytest tests/test_build_mvp0_causal_chain_probe.py::test_probe_builder_writes_minimal_pair_contracts -q
```

Expected: FAIL because `scripts/build_mvp0_causal_chain_probe.py` does not exist.

- [ ] **Step 3: Implement the manifest builder**

Create `scripts/build_mvp0_causal_chain_probe.py`:

```python
#!/usr/bin/env python3
"""Build the ZeroScope MVP-0 causal-chain probe manifest."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROBE_NAME = "zeroscope_mvp0_causal_chain_probe"
PRIORITY_MECHANISMS = ["fluid_impact", "fracture_damage", "elastic_deformation", "particle_dispersion"]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: missing header")
        return list(reader)


def leakage_pair_ids(labels_path: Path | None) -> set[str]:
    if labels_path is None:
        return set()
    return {
        row["pair_id"]
        for row in read_csv(labels_path)
        if row.get("final_label") == "strict_causal_footprint_leakage"
    }


def scene_context(item: dict[str, Any]) -> str:
    control = str(item.get("control_prompt", "")).strip()
    counterfactual = str(item.get("counterfactual_prompt", "")).strip()
    source = str(item.get("source_prompt", "")).strip()
    return control or counterfactual or source


def mechanism_phrase(item: dict[str, Any]) -> str:
    mechanism = str(item.get("mechanism_type", ""))
    target = str(item.get("target_concept", "target"))
    if mechanism == "fluid_impact":
        return f"{target} impact with water"
    if mechanism == "fracture_damage":
        return f"{target} impact causing fracture"
    if mechanism == "surface_trace":
        return f"{target} contact leaving a surface trace"
    if mechanism == "elastic_deformation":
        return f"{target} collision causing elastic deformation"
    if mechanism == "field_mediated":
        return f"{target} field interaction"
    if mechanism == "particle_dispersion":
        return f"{target} collision dispersing particles"
    return f"{target} causal interaction"


def no_phrase(text: str) -> str:
    cleaned = text.strip().rstrip(".")
    return f"no {cleaned}"


def minimal_pairs_for(item: dict[str, Any]) -> dict[str, dict[str, str]]:
    context = scene_context(item).rstrip(".")
    target = str(item.get("target_concept", "")).strip()
    footprint = str(item.get("causal_footprint") or item.get("expected_effect") or "").strip()
    mechanism = mechanism_phrase(item)
    return {
        "cause": {
            "positive": f"{context}, with {target}.",
            "negative": f"{context}, without {target}.",
        },
        "mechanism": {
            "positive": f"{context}, with {mechanism}.",
            "negative": f"{context}, with no impact or causal disturbance.",
        },
        "footprint": {
            "positive": f"{context}, with {footprint}.",
            "negative": f"{context}, with {no_phrase(footprint)}.",
        },
    }


def priority_key(item: dict[str, Any], strict_pairs: set[str]) -> tuple[int, int, str]:
    pair_id = str(item.get("pair_id", ""))
    mechanism = str(item.get("mechanism_type", ""))
    strict_rank = 0 if pair_id in strict_pairs else 1
    mechanism_rank = PRIORITY_MECHANISMS.index(mechanism) if mechanism in PRIORITY_MECHANISMS else len(PRIORITY_MECHANISMS)
    return strict_rank, mechanism_rank, pair_id


def select_items(items: list[dict[str, Any]], strict_pairs: set[str], limit: int) -> list[dict[str, Any]]:
    return sorted(items, key=lambda item: priority_key(item, strict_pairs))[:limit]


def manifest_item(item: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        "probe_index": index,
        "pair_id": str(item.get("pair_id", "")),
        "source_index": str(item.get("source_index", "")),
        "slice_index": int(item.get("slice_index", index)),
        "mechanism_type": str(item.get("mechanism_type", "")),
        "target_concept": str(item.get("target_concept", "")),
        "causal_footprint": str(item.get("causal_footprint") or item.get("expected_effect") or ""),
        "source_prompt": str(item.get("source_prompt", "")),
        "counterfactual_prompt": str(item.get("counterfactual_prompt", "")),
        "control_prompt": str(item.get("control_prompt", "")),
        "clean_video_path": str(item.get("clean_video_path", "")),
        "minimal_pairs": minimal_pairs_for(item),
    }


def write_prompt_file(path: Path, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def write_probe_prompts(output_dir: Path, items: list[dict[str, Any]]) -> None:
    prompts_dir = output_dir / "prompts"
    write_prompt_file(
        prompts_dir / "source_prompts.txt",
        [
            "# source prompts",
            "# Format: <prompt> | <target> | <expected_effect>",
            "",
            *[
                f"{item['source_prompt']} | {item['target_concept']} | {item['causal_footprint']}"
                for item in items
            ],
        ],
    )
    write_prompt_file(
        prompts_dir / "counterfactual_prompts.txt",
        [
            "# counterfactual prompt-only controls",
            "# Format: <prompt> | <target> | <expected_effect>",
            "",
            *[
                f"{item['counterfactual_prompt']} | {item['target_concept']} | {item['causal_footprint']}"
                for item in items
            ],
        ],
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--clean-valid-manifest", type=Path, required=True)
    parser.add_argument("--baseline-labels", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.limit <= 0:
        parser.error("--limit must be positive")
    data = read_json(args.clean_valid_manifest)
    items = data.get("items")
    if not isinstance(items, list):
        parser.exit(2, f"{args.clean_valid_manifest}: missing list field 'items'\n")
    selected = [manifest_item(item, index) for index, item in enumerate(select_items(items, leakage_pair_ids(args.baseline_labels), args.limit))]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_probe_prompts(args.output_dir, selected)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "probe_name": PROBE_NAME,
        "dry_run": args.dry_run,
        "source_manifest": str(args.clean_valid_manifest),
        "baseline_labels": str(args.baseline_labels) if args.baseline_labels else "",
        "count": len(selected),
        "items": selected,
    }
    out = args.output_dir / "probe_manifest.json"
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(selected)} MVP-0 probe items to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the test to verify it passes**

Run:

```bash
python -m pytest tests/test_build_mvp0_causal_chain_probe.py::test_probe_builder_writes_minimal_pair_contracts -q
```

Expected: PASS.

- [ ] **Step 5: Add a selection-priority test**

Append this test to `tests/test_build_mvp0_causal_chain_probe.py`:

```python
def test_probe_builder_prioritizes_strict_leakage_and_priority_mechanisms(tmp_path):
    clean_manifest = tmp_path / "clean_valid.json"
    clean_manifest.write_text(
        json.dumps(
            {
                "items": [
                    {
                        "pair_id": "surface_trace_a",
                        "source_index": "0",
                        "slice_index": 0,
                        "mechanism_type": "surface_trace",
                        "target_concept": "shoe",
                        "causal_footprint": "footprint trace",
                        "source_prompt": "A shoe presses into mud leaving a footprint trace.",
                        "counterfactual_prompt": "A smooth mud surface with no shoe.",
                        "control_prompt": "A smooth mud surface.",
                    },
                    {
                        "pair_id": "fracture_damage_b",
                        "source_index": "1",
                        "slice_index": 1,
                        "mechanism_type": "fracture_damage",
                        "target_concept": "hammer",
                        "causal_footprint": "cracks in glass",
                        "source_prompt": "A hammer hits glass and cracks spread.",
                        "counterfactual_prompt": "An intact glass pane with no hammer.",
                        "control_prompt": "An intact glass pane.",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    labels = tmp_path / "labels.csv"
    labels.write_text(
        "pair_id,final_label\nsurface_trace_a,strict_causal_footprint_leakage\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "probe"

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "build_mvp0_causal_chain_probe.py"),
            "--clean-valid-manifest",
            str(clean_manifest),
            "--baseline-labels",
            str(labels),
            "--output-dir",
            str(output_dir),
            "--limit",
            "1",
            "--dry-run",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    manifest = json.loads((output_dir / "probe_manifest.json").read_text(encoding="utf-8"))
    assert [item["pair_id"] for item in manifest["items"]] == ["surface_trace_a"]
```

- [ ] **Step 6: Run the full builder test file**

Run:

```bash
python -m pytest tests/test_build_mvp0_causal_chain_probe.py -q
```

Expected: `2 passed`.

## Task 2: Probe Runner Dry-Run Contract

**Files:**
- Create: `scripts/adapters/run_mvp0_zeroscope_probe.py`
- Test: `tests/test_run_mvp0_zeroscope_probe.py`

- [ ] **Step 1: Write the failing dry-run runner test**

Create `tests/test_run_mvp0_zeroscope_probe.py` with:

```python
from pathlib import Path
import json
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def write_probe_manifest(tmp_path: Path) -> Path:
    path = tmp_path / "probe_manifest.json"
    path.write_text(
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
                        "source_prompt": "A pebble drops into a pond and causes circular ripples.",
                        "counterfactual_prompt": "A calm pond with no pebble.",
                        "control_prompt": "A calm pond surface.",
                        "minimal_pairs": {
                            "cause": {"positive": "A calm pond surface, with pebble.", "negative": "A calm pond surface, without pebble."},
                            "mechanism": {"positive": "A calm pond surface, with pebble impact with water.", "negative": "A calm pond surface, with no impact or causal disturbance."},
                            "footprint": {"positive": "A calm pond surface, with circular ripples spread outward.", "negative": "A calm pond surface, with no circular ripples spread outward."},
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_mvp0_zeroscope_probe_dry_run_records_conditions(tmp_path):
    probe_manifest = write_probe_manifest(tmp_path)
    output_dir = tmp_path / "probe_run"
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "adapters" / "run_mvp0_zeroscope_probe.py"),
            "--probe-manifest",
            str(probe_manifest),
            "--output-dir",
            str(output_dir),
            "--model",
            "models/zeroscope_v2_576w",
            "--seed",
            "15000",
            "--condition",
            "target_footprint_negative",
            "--condition",
            "monolithic_counterfactual",
            "--condition",
            "full_chain_steering",
            "--dry-run",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    manifest = json.loads((output_dir / "generation_manifest.json").read_text(encoding="utf-8"))
    assert manifest["baseline"] == "mvp0_causal_chain_probe"
    assert manifest["dry_run"] is True
    assert manifest["conditions"] == [
        "target_footprint_negative",
        "monolithic_counterfactual",
        "full_chain_steering",
    ]
    assert len(manifest["items"]) == 3
    full_chain = [item for item in manifest["items"] if item["condition"] == "full_chain_steering"][0]
    assert full_chain["steering"]["links"] == ["cause", "mechanism", "footprint"]
    assert full_chain["video_path"].endswith("_full_chain_steering_seed15000.mp4")
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
python -m pytest tests/test_run_mvp0_zeroscope_probe.py::test_mvp0_zeroscope_probe_dry_run_records_conditions -q
```

Expected: FAIL because `run_mvp0_zeroscope_probe.py` does not exist.

- [ ] **Step 3: Implement the dry-run runner**

Create `scripts/adapters/run_mvp0_zeroscope_probe.py`:

```python
#!/usr/bin/env python3
"""Run or plan ZeroScope MVP-0 causal-chain probe conditions."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from generate_cogvideox_clean import slugify  # noqa: E402


BASELINE = "mvp0_causal_chain_probe"
CONDITIONS = [
    "target_negative",
    "target_footprint_negative",
    "monolithic_counterfactual",
    "cause_steering",
    "mechanism_steering",
    "footprint_steering",
    "full_chain_steering",
    "random_direction",
]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def generation_config(args: argparse.Namespace) -> dict[str, object]:
    return {
        "seed": args.seed,
        "num_inference_steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "num_frames": args.num_frames,
        "fps": args.fps,
        "height": args.height,
        "width": args.width,
        "dtype": args.dtype,
    }


def condition_prompt(item: dict, condition: str) -> tuple[str, str]:
    target = str(item["target_concept"])
    footprint = str(item["causal_footprint"])
    if condition == "target_negative":
        return str(item["source_prompt"]), target
    if condition == "target_footprint_negative":
        return str(item["source_prompt"]), f"{target}, {footprint}"
    if condition == "monolithic_counterfactual":
        return str(item["counterfactual_prompt"]), ""
    return str(item["source_prompt"]), f"{target}, {footprint}"


def steering_contract(item: dict, condition: str) -> dict[str, object]:
    if condition == "cause_steering":
        links = ["cause"]
    elif condition == "mechanism_steering":
        links = ["mechanism"]
    elif condition == "footprint_steering":
        links = ["footprint"]
    elif condition == "full_chain_steering":
        links = ["cause", "mechanism", "footprint"]
    elif condition == "random_direction":
        links = ["random"]
    else:
        links = []
    return {
        "enabled": bool(links),
        "links": links,
        "minimal_pairs": item.get("minimal_pairs", {}),
        "alpha": "not_applied_in_dry_run",
        "timestep_window": "not_applied_in_dry_run",
    }


def build_items(args: argparse.Namespace, probe_items: list[dict]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for probe_item in probe_items:
        for condition in args.condition:
            prompt, negative_prompt = condition_prompt(probe_item, condition)
            seed = args.seed + int(probe_item["probe_index"])
            slug = slugify(str(probe_item["pair_id"]))
            rows.append(
                {
                    "probe_index": probe_item["probe_index"],
                    "pair_id": probe_item["pair_id"],
                    "slice_index": probe_item["slice_index"],
                    "mechanism_type": probe_item["mechanism_type"],
                    "condition": condition,
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "target_concept": probe_item["target_concept"],
                    "causal_footprint": probe_item["causal_footprint"],
                    "seed": seed,
                    "steering": steering_contract(probe_item, condition),
                    "video_path": str(args.output_dir / "videos" / f"{int(probe_item['probe_index']):03d}_{slug}_{condition}_seed{seed}.mp4"),
                }
            )
    return rows


def write_manifest(args: argparse.Namespace, items: list[dict[str, object]]) -> Path:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline": BASELINE,
        "dry_run": args.dry_run,
        "model": args.model,
        "probe_manifest": str(args.probe_manifest),
        "conditions": args.condition,
        "generation": generation_config(args),
        "items": items,
    }
    out = args.output_dir / "generation_manifest.json"
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--probe-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="models/zeroscope_v2_576w")
    parser.add_argument("--condition", action="append", choices=CONDITIONS)
    parser.add_argument("--seed", type=int, default=15000)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=12.5)
    parser.add_argument("--num-frames", type=int, default=24)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--width", type=int, default=576)
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if not args.condition:
        args.condition = ["target_negative", "target_footprint_negative", "monolithic_counterfactual"]
    if args.steps <= 0:
        parser.error("--steps must be positive")
    if args.num_frames <= 0:
        parser.error("--num-frames must be positive")
    if not args.dry_run:
        parser.exit(2, "real MVP-0 probe generation is not implemented yet; use --dry-run\n")
    probe = read_json(args.probe_manifest)
    probe_items = probe.get("items")
    if not isinstance(probe_items, list):
        parser.exit(2, f"{args.probe_manifest}: missing list field 'items'\n")
    out = write_manifest(args, build_items(args, probe_items))
    print(f"ZeroScope MVP-0 probe manifest written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the dry-run runner test**

Run:

```bash
python -m pytest tests/test_run_mvp0_zeroscope_probe.py::test_mvp0_zeroscope_probe_dry_run_records_conditions -q
```

Expected: PASS.

- [ ] **Step 5: Add unsupported-real-mode guard test**

Append to `tests/test_run_mvp0_zeroscope_probe.py`:

```python
def test_mvp0_zeroscope_probe_real_mode_is_guarded_until_implemented(tmp_path):
    probe_manifest = write_probe_manifest(tmp_path)
    output_dir = tmp_path / "probe_run"
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "adapters" / "run_mvp0_zeroscope_probe.py"),
            "--probe-manifest",
            str(probe_manifest),
            "--output-dir",
            str(output_dir),
            "--condition",
            "full_chain_steering",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 2
    assert "real MVP-0 probe generation is not implemented yet" in result.stderr
```

- [ ] **Step 6: Run the full runner test file**

Run:

```bash
python -m pytest tests/test_run_mvp0_zeroscope_probe.py -q
```

Expected: `2 passed`.

## Task 3: Generate the Real Probe Plan Artifacts

**Files:**
- Generate: `experiments/method_probe/zeroscope_mvp0_causal_chain_probe_20260702/probe_manifest.json`
- Generate: `experiments/method_probe/zeroscope_mvp0_causal_chain_probe_20260702/prompts/source_prompts.txt`
- Generate: `experiments/method_probe/zeroscope_mvp0_causal_chain_probe_20260702/prompts/counterfactual_prompts.txt`
- Generate: `experiments/method_probe/zeroscope_mvp0_causal_chain_probe_20260702/dryrun_probe_generation/generation_manifest.json`

- [ ] **Step 1: Run all new unit tests**

Run:

```bash
python -m pytest tests/test_build_mvp0_causal_chain_probe.py tests/test_run_mvp0_zeroscope_probe.py -q
```

Expected: `4 passed`.

- [ ] **Step 2: Build a 12-item probe manifest from real ZeroScope v2 data**

Run:

```bash
python scripts/build_mvp0_causal_chain_probe.py \
  --clean-valid-manifest benchmarks/causal_footprint_v2/zeroscope_clean_valid_gpt54_96_manifest.json \
  --baseline-labels experiments/evaluation/zeroscope_v2_clean_valid96_baselines_gpt54_sharded32_20260701/vlm_predictions_merged_retry1.csv \
  --output-dir experiments/method_probe/zeroscope_mvp0_causal_chain_probe_20260702 \
  --limit 12 \
  --dry-run
```

Expected:

```text
Wrote 12 MVP-0 probe items to experiments/method_probe/zeroscope_mvp0_causal_chain_probe_20260702/probe_manifest.json
```

- [ ] **Step 3: Inspect probe item balance**

Run:

```bash
python - <<'PY'
import json
from collections import Counter
data = json.load(open('experiments/method_probe/zeroscope_mvp0_causal_chain_probe_20260702/probe_manifest.json'))
print('count', data['count'])
print(Counter(item['mechanism_type'] for item in data['items']))
print([item['pair_id'] for item in data['items'][:5]])
PY
```

Expected: count is `12`; mechanisms include at least two of `fluid_impact`, `fracture_damage`, `elastic_deformation`, `particle_dispersion`.

- [ ] **Step 4: Build the dry-run generation contract**

Run:

```bash
python scripts/adapters/run_mvp0_zeroscope_probe.py \
  --probe-manifest experiments/method_probe/zeroscope_mvp0_causal_chain_probe_20260702/probe_manifest.json \
  --output-dir experiments/method_probe/zeroscope_mvp0_causal_chain_probe_20260702/dryrun_probe_generation \
  --condition target_negative \
  --condition target_footprint_negative \
  --condition monolithic_counterfactual \
  --condition cause_steering \
  --condition mechanism_steering \
  --condition footprint_steering \
  --condition full_chain_steering \
  --condition random_direction \
  --seed 15000 \
  --dry-run
```

Expected:

```text
ZeroScope MVP-0 probe manifest written: experiments/method_probe/zeroscope_mvp0_causal_chain_probe_20260702/dryrun_probe_generation/generation_manifest.json
```

- [ ] **Step 5: Inspect the condition matrix**

Run:

```bash
python - <<'PY'
import json
from collections import Counter
data = json.load(open('experiments/method_probe/zeroscope_mvp0_causal_chain_probe_20260702/dryrun_probe_generation/generation_manifest.json'))
print('rows', len(data['items']))
print(Counter(item['condition'] for item in data['items']))
print(data['items'][0]['steering'])
PY
```

Expected: `rows 96`, with 12 rows for each of 8 conditions.

## Task 4: Documentation Update

**Files:**
- Modify: `docs/method_candidate_causal_chain_steering.md`
- Modify: `docs/experiment_log.md`

- [ ] **Step 1: Update method hypothesis doc with probe artifact paths**

Add this section near the end of `docs/method_candidate_causal_chain_steering.md`:

```markdown
## MVP-0 Probe Artifact Plan

The first validation run is intentionally small and dry-runable before any
heavy denoising modification. The planned artifact root is:

```text
experiments/method_probe/zeroscope_mvp0_causal_chain_probe_20260702/
```

The initial probe manifest selects 12 clean-valid ZeroScope v2 items and
expands each into prompt-only controls and steering contracts. Real generation
is gated until the dry-run matrix is inspected.
```
```

- [ ] **Step 2: Add experiment log entry**

Append this entry to `docs/experiment_log.md`:

```markdown
## 2026-07-02: MVP-0 Causal Chain Probe Planning

**Goal:** Start validating the method hypothesis before committing to a full
causal-chain steering method.

**Decision:** The first validation is a mechanism probe, not a full method
claim. It uses ZeroScope v2 clean-valid items, deterministic
cause-mechanism-footprint minimal pairs, and prompt-only/random-direction
controls.

**Planned artifact root:**

```text
experiments/method_probe/zeroscope_mvp0_causal_chain_probe_20260702/
```

**Gate:** Do not scale to Wan or a full method run until the ZeroScope dry-run
matrix and one-prompt smoke are inspected.
```
```

- [ ] **Step 3: Run documentation grep checks**

Run:

```bash
rg -n "MVP-0|zeroscope_mvp0_causal_chain_probe_20260702|not a full method" docs/method_candidate_causal_chain_steering.md docs/experiment_log.md
```

Expected: lines appear in both documents.

## Task 5: Real Steering Runner Design Gate

**Files:**
- No code changes unless Task 1-4 pass.
- Read-only review of `diffusers` pipeline internals is allowed.

- [ ] **Step 1: Locate installed ZeroScope pipeline call path**

Run:

```bash
python - <<'PY'
import inspect
from diffusers import TextToVideoSDPipeline
print(inspect.getsourcefile(TextToVideoSDPipeline))
print(TextToVideoSDPipeline.__call__)
PY
```

Expected: prints the local diffusers pipeline source path and confirms `__call__` exists.

- [ ] **Step 2: Inspect denoising loop source**

Run:

```bash
python - <<'PY'
import inspect
from diffusers import TextToVideoSDPipeline
src = inspect.getsource(TextToVideoSDPipeline.__call__)
for i, line in enumerate(src.splitlines(), 1):
    if 'for' in line and 'timesteps' in line:
        print(i, line)
    if 'noise_pred' in line:
        print(i, line)
PY
```

Expected: prints loop and `noise_pred` lines, enough to decide whether to subclass, copy a small loop, or use a callback.

- [ ] **Step 3: Decide implementation path**

Record the decision in `docs/method_candidate_causal_chain_steering.md` under a new subsection:

```markdown
### Real Runner Implementation Decision

Chosen path: [subclass / copied pipeline loop / callback / defer].

Reason: [one paragraph].
```

Do not implement real steering until this decision is written.

---

## Self-Review Checklist

- Spec coverage: This plan covers MVP-0 manifest selection, minimal-pair prompt contracts, dry-run condition matrix, docs, and a real-runner inspection gate.
- Placeholder scan: No task uses "TBD" or "implement later"; real steering is explicitly guarded as not implemented until the dry-run and source inspection pass.
- Type consistency: Probe manifests use `items`, `minimal_pairs`, `condition`, and `steering` consistently across builder and runner tests.

