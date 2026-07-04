# C0.3 Scope-Locked Validity Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the C0.3 scope-locked validity gate infrastructure: diagnostic scoring for C0.2, a pre-registered C0.3 candidate manifest, a C0.3 prompt-template mode, and a dry-run/review package before any new GPU generation.

**Architecture:** Extend existing C0/C0.1 tooling instead of adding a parallel pipeline. `score_c01_factorial_gate.py` gains named threshold profiles; a new candidate-builder script emits a runner-compatible pre-registered manifest; `run_c0_counterfactual_grid.py` gains a `c03_scope_locked` prompt template that reads item-level surface/footprint fields.

**Tech Stack:** Python standard library, existing ZeroScope runner, existing C0.1 review builder, pytest.

---

## Files

- Modify: `scripts/score_c01_factorial_gate.py`
- Modify: `tests/test_score_c01_factorial_gate.py`
- Modify: `scripts/adapters/run_c0_counterfactual_grid.py`
- Modify: `tests/test_run_c0_counterfactual_grid.py`
- Create: `scripts/build_c03_scope_locked_candidates.py`
- Create: `tests/test_build_c03_scope_locked_candidates.py`
- Create: `experiments/method_probe/c03_scope_locked_candidates_20260704/candidate_manifest.json`
- Create after dry run: `experiments/method_probe/c03_scope_locked_gate_20260704_dryrun/generation_manifest.json`
- Create after dry run: `experiments/evaluation/c03_scope_locked_gate_20260704_dryrun/blind_review.csv`
- Create after dry run: `experiments/evaluation/c03_scope_locked_gate_20260704_dryrun/answer_key.csv`
- Create after dry run: `experiments/evaluation/c03_scope_locked_gate_20260704_dryrun/review_manifest.json`
- Modify: `docs/experiment_log.md`

---

### Task 1: Add Scoring Profiles For C0.2 And C0.3

**Files:**
- Modify: `scripts/score_c01_factorial_gate.py`
- Modify: `tests/test_score_c01_factorial_gate.py`

- [x] **Step 1: Add failing tests for diagnostic and strict profiles**

Append these tests to `tests/test_score_c01_factorial_gate.py`:

```python
def make_rows_with_seed_count(
    pair_id: str,
    seed_count: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    review_rows = []
    key_rows = []
    expected = {
        "original": ("yes", "yes"),
        "remove_target": ("no", "no"),
        "footprint_only": ("no", "yes"),
        "target_only": ("yes", "no"),
    }
    for seed_index in range(seed_count):
        for variant, (target_expected, footprint_expected) in expected.items():
            review_id = f"c03_000_s{seed_index:02d}_{variant}"
            review_rows.append(
                {
                    "review_id": review_id,
                    "target_visible": "present" if target_expected == "yes" else "absent",
                    "footprint_visible": "present"
                    if footprint_expected == "yes"
                    else "absent",
                    "scene_structure_preserved": "yes",
                    "cells_distinguishable": "yes",
                    "generation_failure": "no",
                    "mode_collapse": "no",
                    "notes": "",
                }
            )
            key_rows.append(
                {
                    "review_id": review_id,
                    "pair_id": pair_id,
                    "item_index": "0",
                    "seed_index": str(seed_index),
                    "variant": variant,
                    "expected_target_visible": target_expected,
                    "expected_footprint_visible": footprint_expected,
                }
            )
    return review_rows, key_rows


def test_c02_diagnostic_profile_uses_two_of_three_thresholds(tmp_path):
    review_rows, key_rows = make_rows_with_seed_count("c02_pair", 3)
    target_only_rows = [
        row for row in review_rows if row["review_id"].endswith("_target_only")
    ]
    target_only_rows[0]["footprint_visible"] = "present"
    review_csv = tmp_path / "blind_review.csv"
    key_csv = tmp_path / "answer_key.csv"
    write_csv(review_csv, review_rows)
    write_csv(key_csv, key_rows)

    module = load_module()
    output_dir = tmp_path / "scores"
    module.main(
        [
            "--review-csv",
            str(review_csv),
            "--answer-key",
            str(key_csv),
            "--output-dir",
            str(output_dir),
            "--profile",
            "c02_diagnostic",
        ]
    )

    item_rows = list(csv.DictReader((output_dir / "item_gate_summary.csv").open()))
    assert item_rows[0]["gate_status"] == "diagnostic_promising"
    assert item_rows[0]["target_only_successes"] == "2"
    assert item_rows[0]["target_only_threshold"] == "2"


def test_c03_profile_preserves_c01_five_seed_thresholds(tmp_path):
    review_rows, key_rows = make_rows_with_seed_count("c03_pair", 5)
    footprint_rows = [
        row for row in review_rows if row["review_id"].endswith("_footprint_only")
    ]
    footprint_rows[0]["footprint_visible"] = "absent"
    footprint_rows[1]["footprint_visible"] = "absent"
    footprint_rows[2]["footprint_visible"] = "absent"
    review_csv = tmp_path / "blind_review.csv"
    key_csv = tmp_path / "answer_key.csv"
    write_csv(review_csv, review_rows)
    write_csv(key_csv, key_rows)

    module = load_module()
    output_dir = tmp_path / "scores"
    module.main(
        [
            "--review-csv",
            str(review_csv),
            "--answer-key",
            str(key_csv),
            "--output-dir",
            str(output_dir),
            "--profile",
            "c03",
        ]
    )

    item_rows = list(csv.DictReader((output_dir / "item_gate_summary.csv").open()))
    assert item_rows[0]["gate_status"] == "fail"
    assert item_rows[0]["footprint_only_successes"] == "2"
    assert item_rows[0]["footprint_only_threshold"] == "3"
    assert "footprint_only_below_threshold" in item_rows[0]["rejection_reasons"]
    assert "footprint_only_incoherent" in item_rows[0]["rejection_reasons"]
```

- [x] **Step 2: Run tests and verify they fail**

Run:

```bash
PYTHONNOUSERSITE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/deepseek_VG/.conda/envs/vcecf/bin/python -m pytest \
  tests/test_score_c01_factorial_gate.py -q
```

Expected: fail because `--profile` is not implemented.

- [x] **Step 3: Add named threshold profiles**

In `scripts/score_c01_factorial_gate.py`, replace the single `THRESHOLDS`
constant with:

```python
THRESHOLD_PROFILES = {
    "c01": {
        "status_on_pass": "pass",
        "thresholds": {
            "original": 4,
            "remove_target": 4,
            "target_only": 4,
            "footprint_only": 3,
        },
    },
    "c02_diagnostic": {
        "status_on_pass": "diagnostic_promising",
        "thresholds": {
            "original": 2,
            "remove_target": 2,
            "target_only": 2,
            "footprint_only": 2,
        },
    },
    "c03": {
        "status_on_pass": "pass",
        "thresholds": {
            "original": 4,
            "remove_target": 4,
            "target_only": 4,
            "footprint_only": 3,
        },
    },
}
```

Change `aggregate_item_scores` to accept `thresholds` and `status_on_pass`:

```python
def aggregate_item_scores(
    cell_rows: Sequence[dict[str, Any]],
    *,
    thresholds: dict[str, int],
    status_on_pass: str = "pass",
) -> list[dict[str, Any]]:
```

Inside the variant loop, read `threshold = thresholds[variant]` instead of
`THRESHOLDS[variant]`. When all variants meet their thresholds, set:

```python
gate_status = status_on_pass if not rejection_reasons and not missing else "fail"
```

Add a parser argument:

```python
parser.add_argument(
    "--profile",
    choices=sorted(THRESHOLD_PROFILES),
    default="c01",
    help="threshold profile: c01, c02_diagnostic, or c03",
)
```

In `main`, select the profile and pass it into aggregation:

```python
profile = THRESHOLD_PROFILES[str(args.profile)]
item_rows = aggregate_item_scores(
    cell_rows,
    thresholds=profile["thresholds"],
    status_on_pass=str(profile["status_on_pass"]),
)
```

- [x] **Step 4: Run tests and verify they pass**

Run:

```bash
PYTHONNOUSERSITE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/deepseek_VG/.conda/envs/vcecf/bin/python -m pytest \
  tests/test_score_c01_factorial_gate.py -q
```

Expected: all tests pass.

- [x] **Step 5: Commit scorer profiles**

```bash
git add scripts/score_c01_factorial_gate.py tests/test_score_c01_factorial_gate.py
git commit -m "add factorial gate scoring profiles"
```

---

### Task 2: Build The Pre-Registered C0.3 Candidate Manifest

**Files:**
- Create: `scripts/build_c03_scope_locked_candidates.py`
- Create: `tests/test_build_c03_scope_locked_candidates.py`
- Create: `experiments/method_probe/c03_scope_locked_candidates_20260704/candidate_manifest.json`

- [x] **Step 1: Write failing tests for the candidate builder**

Create `tests/test_build_c03_scope_locked_candidates.py`:

```python
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
    assert all(item["prompt_template_id"] == "c03_scope_locked" for item in manifest["items"])
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
```

- [x] **Step 2: Run tests and verify they fail**

Run:

```bash
PYTHONNOUSERSITE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/deepseek_VG/.conda/envs/vcecf/bin/python -m pytest \
  tests/test_build_c03_scope_locked_candidates.py -q
```

Expected: fail because the script does not exist.

- [x] **Step 3: Create the candidate builder**

Create `scripts/build_c03_scope_locked_candidates.py`:

```python
#!/usr/bin/env python3
"""Build the pre-registered C0.3 scope-locked candidate manifest."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


SCOPE = "low_entanglement_rigid_object_surface_trace"
PREDICATES = [
    "rigid_or_tool_like_target",
    "simple_static_surface",
    "localized_persistent_footprint",
    "footprint_only_plausible",
    "target_footprint_text_separable",
    "stable_background_and_camera",
]

CANDIDATES = [
    ("metal comb", "smooth sand tray", "parallel grooves in the sand", "parallel grooves in the sand"),
    ("toy car", "soft clay slab", "two tire tracks in the soft clay", "tire tracks in the soft clay"),
    ("rubber stamp", "blank paper sheet", "a square ink stamp mark on the paper", "square ink stamp mark on the paper"),
    ("piece of chalk", "clean blackboard", "a white chalk line on the blackboard", "white chalk line on the blackboard"),
    ("wooden block", "smooth clay slab", "a square imprint in the clay", "square imprint in the clay"),
    ("hiking boot", "wet sand patch", "a single boot print in the wet sand", "boot print in the wet sand"),
    ("paint roller", "white paper strip", "a blue paint stripe on the paper", "blue paint stripe on the paper"),
    ("wooden stylus", "smooth wax tablet", "a thin carved line in the wax", "thin carved line in the wax"),
]


def pair_id_for(index: int, target: str, footprint: str) -> str:
    target_slug = target.replace(" ", "_").replace("-", "_")
    footprint_slug = footprint.replace(" ", "_").replace("-", "_")
    return f"c03_surface_trace_{index:02d}_{target_slug}_{footprint_slug}"


def item_row(
    index: int,
    target: str,
    surface: str,
    footprint: str,
    absence_footprint: str,
) -> dict[str, object]:
    pair_id = pair_id_for(index, target, footprint)
    source_prompt = (
        f"A realistic fixed-camera close-up video of a {target} contacting a {surface}, "
        f"leaving {footprint}."
    )
    return {
        "probe_index": index,
        "slice_index": index,
        "source_index": f"c03_{index:02d}",
        "pair_id": pair_id,
        "mechanism_type": "scope_locked_surface_trace",
        "target_concept": target,
        "surface_or_object": surface,
        "causal_footprint": footprint,
        "causal_footprint_absence": absence_footprint,
        "source_prompt": source_prompt,
        "generation_prompt": source_prompt,
        "counterfactual_prompt": (
            f"A realistic fixed-camera close-up video of a clean {surface}. "
            f"No {target} is present. The scene shows no {absence_footprint}."
        ),
        "control_prompt": (
            f"A realistic fixed-camera close-up video of a {surface} with {footprint}. "
            f"No {target} is present and no visible cause appears in the frame."
        ),
        "prompt_template_id": "c03_scope_locked",
        "prior_seen": False,
        "candidate_scope": SCOPE,
        "scope_predicates_met": list(PREDICATES),
    }


def build_manifest() -> dict[str, object]:
    items = [item_row(index, *candidate) for index, candidate in enumerate(CANDIDATES)]
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "probe_name": "c03_scope_locked_surface_trace_candidates",
        "dry_run": False,
        "candidate_scope": SCOPE,
        "count": len(items),
        "items": items,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = build_manifest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {manifest['count']} C0.3 candidates to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [x] **Step 4: Run tests and verify they pass**

Run:

```bash
PYTHONNOUSERSITE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/deepseek_VG/.conda/envs/vcecf/bin/python -m pytest \
  tests/test_build_c03_scope_locked_candidates.py -q
```

Expected: all tests pass.

- [x] **Step 5: Generate and commit the pre-registered candidate manifest**

Run:

```bash
PYTHONNOUSERSITE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/deepseek_VG/.conda/envs/vcecf/bin/python \
  scripts/build_c03_scope_locked_candidates.py \
  --output experiments/method_probe/c03_scope_locked_candidates_20260704/candidate_manifest.json
```

Then commit:

```bash
git add scripts/build_c03_scope_locked_candidates.py \
  tests/test_build_c03_scope_locked_candidates.py \
  experiments/method_probe/c03_scope_locked_candidates_20260704/candidate_manifest.json
git commit -m "add c03 scope locked candidate manifest"
```

---

### Task 3: Add The C0.3 Prompt Template To The Runner

**Files:**
- Modify: `scripts/adapters/run_c0_counterfactual_grid.py`
- Modify: `tests/test_run_c0_counterfactual_grid.py`

- [x] **Step 1: Add failing runner prompt-template test**

Append this test to `tests/test_run_c0_counterfactual_grid.py`:

```python
def test_c03_scope_locked_prompt_template_uses_manifest_surface_and_footprint():
    module = load_module()
    item = {
        "target_concept": "metal comb",
        "surface_or_object": "smooth sand tray",
        "causal_footprint": "parallel grooves in the sand",
        "causal_footprint_absence": "parallel grooves in the sand",
    }

    original, _ = module.variant_prompt(
        item, "original", prompt_template="c03_scope_locked"
    )
    remove_target, _ = module.variant_prompt(
        item, "remove_target", prompt_template="c03_scope_locked"
    )
    footprint_only, _ = module.variant_prompt(
        item, "footprint_only", prompt_template="c03_scope_locked"
    )
    target_only, _ = module.variant_prompt(
        item, "target_only", prompt_template="c03_scope_locked"
    )

    assert "metal comb" in original
    assert "smooth sand tray" in original
    assert "parallel grooves in the sand" in original
    assert "No metal comb is present" in remove_target
    assert "The scene shows no parallel grooves in the sand" in remove_target
    assert "No metal comb is present" in footprint_only
    assert "Parallel grooves in the sand is clearly visible" in footprint_only
    assert "does not touch, strike, mark, press, disturb, or change it" in target_only
    assert "The scene shows no parallel grooves in the sand" in target_only
```

- [x] **Step 2: Run tests and verify failure**

Run:

```bash
PYTHONNOUSERSITE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/deepseek_VG/.conda/envs/vcecf/bin/python -m pytest \
  tests/test_run_c0_counterfactual_grid.py -q
```

Expected: fail because `c03_scope_locked` is not an allowed prompt template.

- [x] **Step 3: Implement C0.3 prompt template**

In `scripts/adapters/run_c0_counterfactual_grid.py`, change:

```python
PROMPT_TEMPLATES = ["legacy", "c02_discrete"]
```

to:

```python
PROMPT_TEMPLATES = ["legacy", "c02_discrete", "c03_scope_locked"]
```

Add:

```python
def c03_surface_for(item: dict[str, object]) -> str:
    surface = normalize_space(str(item.get("surface_or_object", "")))
    return surface or c02_surface_for(item)


def c03_footprints_for(item: dict[str, object]) -> tuple[str, str]:
    footprint = normalize_space(str(item.get("causal_footprint", "causal footprint")))
    absence = normalize_space(str(item.get("causal_footprint_absence", "")))
    return footprint, absence or footprint


def c03_scope_locked_prompt(item: dict[str, object], variant: str) -> tuple[str, str]:
    target = normalize_space(str(item.get("target_concept", "target")))
    visible_footprint, absence_footprint = c03_footprints_for(item)
    surface = c03_surface_for(item)
    anchor = c02_scene_anchor()
    if variant == "original":
        return normalize_space(
            f"{anchor} The {target} is clearly visible and contacts the {surface}. "
            f"After contact, {visible_footprint} is clearly visible."
        ), ""
    if variant == "remove_target":
        return normalize_space(
            f"{anchor} No {target} is present. No visible cause is present. "
            f"The {surface} stays clean and unchanged. "
            f"The scene shows no {absence_footprint}."
        ), ""
    if variant == "footprint_only":
        return normalize_space(
            f"{anchor} No {target} is present and no visible cause appears in the frame. "
            f"{visible_footprint.capitalize()} is clearly visible on the {surface}. "
            "The scene otherwise stays the same."
        ), ""
    if variant == "target_only":
        return normalize_space(
            f"{anchor} The {target} is clearly visible, but it is separated from the "
            f"{surface} and does not touch, strike, mark, press, disturb, or change it. "
            f"The scene shows no {absence_footprint}."
        ), ""
    raise ValueError(f"unknown variant: {variant}")
```

Update `variant_prompt`:

```python
if prompt_template == "c03_scope_locked":
    return c03_scope_locked_prompt(item, variant)
```

- [x] **Step 4: Run tests and verify they pass**

Run:

```bash
PYTHONNOUSERSITE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/deepseek_VG/.conda/envs/vcecf/bin/python -m pytest \
  tests/test_run_c0_counterfactual_grid.py -q
```

Expected: all tests pass.

- [x] **Step 5: Commit C0.3 prompt template**

```bash
git add scripts/adapters/run_c0_counterfactual_grid.py tests/test_run_c0_counterfactual_grid.py
git commit -m "add c03 scope locked prompt template"
```

---

### Task 4: Run The C0.3 Dry Run And Build Review Artifacts

**Files:**
- Create: `experiments/method_probe/c03_scope_locked_gate_20260704_dryrun/generation_manifest.json`
- Create: `experiments/evaluation/c03_scope_locked_gate_20260704_dryrun/blind_review.csv`
- Create: `experiments/evaluation/c03_scope_locked_gate_20260704_dryrun/answer_key.csv`
- Create: `experiments/evaluation/c03_scope_locked_gate_20260704_dryrun/review_manifest.json`
- Modify: `docs/experiment_log.md`

- [ ] **Step 1: Run the 160-row dry run**

Run:

```bash
PYTHONNOUSERSITE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/deepseek_VG/.conda/envs/vcecf/bin/python \
  scripts/adapters/run_c0_counterfactual_grid.py \
  --probe-manifest experiments/method_probe/c03_scope_locked_candidates_20260704/candidate_manifest.json \
  --output-dir experiments/method_probe/c03_scope_locked_gate_20260704_dryrun \
  --prompt-template c03_scope_locked \
  --seed 53000 \
  --seeds-per-item 5 \
  --dry-run
```

Expected: `generation_manifest.json` with 160 rows.

- [ ] **Step 2: Build dry-run blinded review package**

Run:

```bash
PYTHONNOUSERSITE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/deepseek_VG/.conda/envs/vcecf/bin/python \
  scripts/build_c01_factorial_gate_review.py \
  --generation-manifest experiments/method_probe/c03_scope_locked_gate_20260704_dryrun/generation_manifest.json \
  --output-dir experiments/evaluation/c03_scope_locked_gate_20260704_dryrun \
  --frames-per-video 5 \
  --thumb-width 192 \
  --thumb-height 128 \
  --skip-frame-extraction \
  --shuffle-seed 37
```

Expected: review CSV and answer key with 160 rows. Frame strips are absent in
dry-run because videos are not generated; this is acceptable for metadata
validation and should be recorded as `frame_strip_count=0`.

- [ ] **Step 3: Run metadata integrity checks**

Run:

```bash
PYTHONNOUSERSITE=1 /home/deepseek_VG/.conda/envs/vcecf/bin/python - <<'PY'
import csv, json
from collections import Counter
from pathlib import Path

manifest_path = Path("experiments/method_probe/c03_scope_locked_gate_20260704_dryrun/generation_manifest.json")
eval_dir = Path("experiments/evaluation/c03_scope_locked_gate_20260704_dryrun")
manifest = json.loads(manifest_path.read_text())
items = manifest["items"]
blind = list(csv.DictReader((eval_dir / "blind_review.csv").open()))
key = list(csv.DictReader((eval_dir / "answer_key.csv").open()))
print("manifest_rows", len(items))
print("blind_rows", len(blind))
print("answer_key_rows", len(key))
print("variants", dict(sorted(Counter(row["variant"] for row in items).items())))
print("seed_indices", dict(sorted(Counter(row["seed_index"] for row in items).items())))
print("prompt_template", manifest["generation"]["prompt_template"])
print("dry_run", manifest["dry_run"])
review_manifest = json.loads((eval_dir / "review_manifest.json").read_text())
print("frame_strip_count", review_manifest["frame_strip_count"])
assert len(items) == 160
assert len(blind) == 160
assert len(key) == 160
assert manifest["generation"]["prompt_template"] == "c03_scope_locked"
assert manifest["dry_run"] is True
assert review_manifest["frame_strip_count"] == 0
PY
```

Expected: all assertions pass.

- [ ] **Step 4: Append dry-run summary to experiment log**

Append to `docs/experiment_log.md`:

```markdown
### 2026-07-04 C0.3 scope-locked validity gate dry run

Created the C0.3 scope-locked surface-trace candidate manifest and validated a
160-row dry-run grid before any new GPU generation.

Artifacts:

```text
experiments/method_probe/c03_scope_locked_candidates_20260704/candidate_manifest.json
experiments/method_probe/c03_scope_locked_gate_20260704_dryrun/generation_manifest.json
experiments/evaluation/c03_scope_locked_gate_20260704_dryrun/blind_review.csv
experiments/evaluation/c03_scope_locked_gate_20260704_dryrun/answer_key.csv
experiments/evaluation/c03_scope_locked_gate_20260704_dryrun/review_manifest.json
```

Integrity:

```text
candidate_items=8
denominator_items=8
prior_seen=false for all denominator items
rows=160
variants=40 each
seed_indices=0..4
prompt_template=c03_scope_locked
dry_run=true
frame_strip_count=0
```

No new real videos were generated in this step. The next gate is human approval
of the pre-registered candidate manifest before launching the 160-video real
run.
```

- [ ] **Step 5: Commit dry-run artifacts**

```bash
git add docs/experiment_log.md \
  experiments/method_probe/c03_scope_locked_candidates_20260704/candidate_manifest.json \
  experiments/method_probe/c03_scope_locked_gate_20260704_dryrun/generation_manifest.json \
  experiments/evaluation/c03_scope_locked_gate_20260704_dryrun/blind_review.csv \
  experiments/evaluation/c03_scope_locked_gate_20260704_dryrun/answer_key.csv \
  experiments/evaluation/c03_scope_locked_gate_20260704_dryrun/review_manifest.json
git commit -m "prepare c03 scope locked dry run"
```

---

### Task 5: Execution Gate Before Real Videos

**Files:**
- No code files.
- Read: `experiments/method_probe/c03_scope_locked_candidates_20260704/candidate_manifest.json`
- Read: `docs/experiment_log.md`

- [ ] **Step 1: Review the pre-registered candidate manifest**

Open:

```bash
PYTHONNOUSERSITE=1 /home/deepseek_VG/.conda/envs/vcecf/bin/python - <<'PY'
import json
from pathlib import Path
path = Path("experiments/method_probe/c03_scope_locked_candidates_20260704/candidate_manifest.json")
data = json.loads(path.read_text())
for item in data["items"]:
    print(item["probe_index"], "|", item["target_concept"], "|", item["surface_or_object"], "|", item["causal_footprint"])
PY
```

Expected: eight prior-unseen low-entanglement surface-trace candidates.

- [ ] **Step 2: Ask for approval before GPU generation**

Report the candidate list and ask whether to run the real 160-video C0.3 panel.
Do not launch GPU generation until the user approves the denominator.

- [ ] **Step 3: If approved, run the real C0.3 panel**

Use the least-loaded GPU from:

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
```

Then run:

```bash
CUDA_VISIBLE_DEVICES=<physical_gpu> PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  PYTHONNOUSERSITE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/deepseek_VG/.conda/envs/vcecf/bin/python \
  scripts/adapters/run_c0_counterfactual_grid.py \
  --probe-manifest experiments/method_probe/c03_scope_locked_candidates_20260704/candidate_manifest.json \
  --output-dir experiments/method_probe/c03_scope_locked_gate_20260704_real_s10_f16_240x432 \
  --prompt-template c03_scope_locked \
  --seed 53000 \
  --seeds-per-item 5 \
  --steps 10 \
  --num-frames 16 \
  --height 240 \
  --width 432 \
  --guidance-scale 9.0 \
  --dtype fp16 \
  --device cuda:0 \
  --enable-model-cpu-offload \
  --vae-slicing
```

Expected: 160 videos and a real generation manifest. If this OOMs, stop and
report the GPU failure instead of reducing the denominator after seeing partial
outputs.

---

## Self-Review Checklist

- Spec coverage: C0.2 audit, pre-registered candidate denominator, C0.3 prompt template, dry-run, and pre-GPU approval are covered.
- Placeholder scan: no `TBD`, no open-ended "add tests", and no missing paths.
- Scope check: real GPU generation is gated behind explicit candidate approval, so the implementation plan can be completed without spending GPU.
- Type consistency: candidate manifest fields match `run_c0_counterfactual_grid.py` expectations and add C0.3-specific fields without breaking legacy items.
