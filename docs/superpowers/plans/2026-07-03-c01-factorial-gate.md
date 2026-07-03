# C0.1 Factorial Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the C0.1 seed-matched factorial prompt gate: 3 items by 5 seeds by 4 cells, human-blinded review artifacts, and a deterministic gate scorer.

**Architecture:** Extend the existing C0 runner for multi-seed generation, add a dedicated C0.1 human-review builder that hides cell labels, and add a scorer that joins blinded labels to an answer key and applies the gate thresholds. Keep C0.1 separate from fable/VLM judging; fable remains a method-review helper only.

**Tech Stack:** Python standard library, existing ZeroScope adapter helpers, pytest, CSV/JSON manifests, existing frame-strip helper from `scripts/build_baseline_review.py`.

---

## File Structure

- Modify: `scripts/adapters/run_c0_counterfactual_grid.py`
  - Add multi-seed support with `--seeds-per-item`, `seed_index`, and stable per-item seed allocation.
  - Preserve existing single-seed behavior when `--seeds-per-item 1`.
- Modify: `tests/test_run_c0_counterfactual_grid.py`
  - Add TDD coverage for 3 items by N seeds by selected variants.
- Create: `scripts/build_c01_factorial_gate_review.py`
  - Build blinded human-review CSV, answer-key CSV, and optional frame strips from a C0/C0.1 generation manifest.
- Create: `tests/test_build_c01_factorial_gate_review.py`
  - Verify review rows hide variant labels while answer key preserves them.
- Create: `scripts/score_c01_factorial_gate.py`
  - Score human labels against the answer key and emit per-cell and per-item gate summaries.
- Create: `tests/test_score_c01_factorial_gate.py`
  - Verify threshold logic, uncertain handling, scene drift handling, and structured rejection reasons.
- Modify: `docs/experiment_log.md`
  - Add a short entry after dry-run validation only.

Before implementation, note that this worktree already contains uncommitted C0 scorer/base-validity changes. Do not revert them. If a task touches the same file, read the current file and preserve those changes.

---

### Task 1: Add Multi-Seed Rows To The C0 Runner

**Files:**
- Modify: `scripts/adapters/run_c0_counterfactual_grid.py`
- Modify: `tests/test_run_c0_counterfactual_grid.py`

- [ ] **Step 1: Add the failing multi-seed unit test**

Append this test to `tests/test_run_c0_counterfactual_grid.py`:

```python
def test_build_items_expands_each_probe_item_over_multiple_seeds(tmp_path):
    module = load_runner_module()
    probe_manifest = write_probe_manifest(tmp_path)
    args = module.build_parser().parse_args(
        [
            "--probe-manifest",
            str(probe_manifest),
            "--output-dir",
            str(tmp_path / "out"),
            "--seed",
            "40000",
            "--seeds-per-item",
            "3",
        ]
    )

    rows = module.build_items(args, json.loads(probe_manifest.read_text())["items"])

    assert len(rows) == 12
    assert [row["seed_index"] for row in rows[:4]] == [0, 0, 0, 0]
    assert [row["seed_index"] for row in rows[4:8]] == [1, 1, 1, 1]
    assert [row["seed_index"] for row in rows[8:12]] == [2, 2, 2, 2]
    assert {row["seed"] for row in rows[:4]} == {40002}
    assert {row["seed"] for row in rows[4:8]} == {40003}
    assert {row["seed"] for row in rows[8:12]} == {40004}
    assert rows[4]["video_path"].endswith("_seed01_original_seed40003.mp4")
```

- [ ] **Step 2: Run the focused test and verify it fails**

Run:

```bash
PYTHONNOUSERSITE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/deepseek_VG/.conda/envs/vcecf/bin/python -m pytest \
  tests/test_run_c0_counterfactual_grid.py::test_build_items_expands_each_probe_item_over_multiple_seeds -q
```

Expected: fail because `--seeds-per-item` and `seed_index` do not exist.

- [ ] **Step 3: Implement `--seeds-per-item`**

In `build_parser()`, add:

```python
parser.add_argument("--seeds-per-item", type=int, default=1)
```

In `main()`, after the `--limit-items` validation, add:

```python
if args.seeds_per_item <= 0:
    parser.error("--seeds-per-item must be positive")
```

In `generation_config(args)`, add:

```python
"seeds_per_item": args.seeds_per_item,
```

Replace the loop in `build_items()` with a nested seed loop:

```python
for item in probe_items:
    probe_index = int(item.get("probe_index", len(rows)))
    pair_id = str(item.get("pair_id", f"item_{probe_index}"))
    slug = slugify(pair_id)
    for seed_index in range(args.seeds_per_item):
        seed = args.seed + probe_index + seed_index
        for variant in variants:
            prompt, negative_prompt = variant_prompt(item, variant)
            expected_target, expected_footprint = EXPECTED_STATES[variant]
            video_path = (
                args.output_dir
                / "videos"
                / f"{probe_index:03d}_{slug}_seed{seed_index:02d}_{variant}_seed{seed}.mp4"
            )
            rows.append(
                {
                    "probe_index": probe_index,
                    "pair_id": pair_id,
                    "slice_index": item.get("slice_index", probe_index),
                    "source_index": str(item.get("source_index", "")),
                    "mechanism_type": str(item.get("mechanism_type", "")),
                    "seed_index": seed_index,
                    "variant": variant,
                    "variant_label": VARIANT_LABELS[variant],
                    "variant_role": variant,
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "source_prompt": str(item.get("source_prompt", "")),
                    "generation_prompt": str(
                        item.get("generation_prompt") or item.get("source_prompt", "")
                    ),
                    "counterfactual_prompt": str(item.get("counterfactual_prompt", "")),
                    "control_prompt": str(item.get("control_prompt", "")),
                    "target_concept": str(item.get("target_concept", "")),
                    "causal_footprint": str(item.get("causal_footprint", "")),
                    "expected_target_visible": expected_target,
                    "expected_footprint_visible": expected_footprint,
                    "seed": seed,
                    "video_path": str(video_path),
                    "clean_video_path": str(item.get("clean_video_path", "")),
                }
            )
```

- [ ] **Step 4: Run focused runner tests**

Run:

```bash
PYTHONNOUSERSITE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/deepseek_VG/.conda/envs/vcecf/bin/python -m pytest \
  tests/test_run_c0_counterfactual_grid.py -q
```

Expected: all tests in `tests/test_run_c0_counterfactual_grid.py` pass.

- [ ] **Step 5: Commit Task 1**

```bash
git add scripts/adapters/run_c0_counterfactual_grid.py tests/test_run_c0_counterfactual_grid.py
git commit -m "add multi-seed c0 grid rows"
```

---

### Task 2: Build Blinded C0.1 Human Review Artifacts

**Files:**
- Create: `scripts/build_c01_factorial_gate_review.py`
- Create: `tests/test_build_c01_factorial_gate_review.py`

- [ ] **Step 1: Add failing review-builder tests**

Create `tests/test_build_c01_factorial_gate_review.py`:

```python
import csv
import importlib.util
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "build_c01_factorial_gate_review.py"


def load_module():
    spec = importlib.util.spec_from_file_location("build_c01_factorial_gate_review", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_generation_manifest(tmp_path: Path) -> Path:
    items = []
    for seed_index, seed in enumerate([40000, 40001]):
        for variant, expected in [
            ("original", ("yes", "yes")),
            ("remove_target", ("no", "no")),
            ("footprint_only", ("no", "yes")),
            ("target_only", ("yes", "no")),
        ]:
            video_path = tmp_path / f"{seed_index}_{variant}.mp4"
            video_path.write_bytes(b"fake")
            items.append(
                {
                    "probe_index": 0,
                    "pair_id": "fluid_impact_pebble_pond_002",
                    "slice_index": 5,
                    "source_index": "12",
                    "mechanism_type": "fluid_impact",
                    "seed_index": seed_index,
                    "seed": seed,
                    "variant": variant,
                    "variant_label": variant,
                    "variant_role": variant,
                    "video_path": str(video_path),
                    "target_concept": "pebble",
                    "causal_footprint": "circular ripples",
                    "source_prompt": "A pebble drops into a pond.",
                    "prompt": f"{variant} prompt",
                    "expected_target_visible": expected[0],
                    "expected_footprint_visible": expected[1],
                }
            )
    path = tmp_path / "generation_manifest.json"
    path.write_text(json.dumps({"baseline": "c0_counterfactual_grid", "items": items}), encoding="utf-8")
    return path


def test_build_review_outputs_blind_rows_and_answer_key(tmp_path):
    module = load_module()
    manifest = write_generation_manifest(tmp_path)
    output_dir = tmp_path / "review"

    result = module.main(
        [
            "--generation-manifest",
            str(manifest),
            "--output-dir",
            str(output_dir),
            "--skip-frame-extraction",
            "--shuffle-seed",
            "7",
        ]
    )

    assert result == 0
    review_rows = list(csv.DictReader((output_dir / "blind_review.csv").open(encoding="utf-8")))
    key_rows = list(csv.DictReader((output_dir / "answer_key.csv").open(encoding="utf-8")))
    assert len(review_rows) == 8
    assert len(key_rows) == 8
    assert "variant" not in review_rows[0]
    assert "expected_target_visible" not in review_rows[0]
    assert {"target_visible", "footprint_visible", "scene_structure_preserved", "cells_distinguishable"}.issubset(review_rows[0])
    assert {"variant", "expected_target_visible", "expected_footprint_visible"}.issubset(key_rows[0])
    assert {row["review_id"] for row in review_rows} == {row["review_id"] for row in key_rows}
```

- [ ] **Step 2: Run the new test and verify it fails**

Run:

```bash
PYTHONNOUSERSITE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/deepseek_VG/.conda/envs/vcecf/bin/python -m pytest \
  tests/test_build_c01_factorial_gate_review.py -q
```

Expected: fail because `scripts/build_c01_factorial_gate_review.py` does not exist.

- [ ] **Step 3: Create the review builder**

Create `scripts/build_c01_factorial_gate_review.py` with these top-level fields:

```python
REVIEW_FIELDS = [
    "review_id",
    "item_index",
    "seed_index",
    "video_path",
    "video_exists",
    "strip_path",
    "strip_exists",
    "target_concept",
    "footprint_definition",
    "source_prompt",
    "target_visible",
    "footprint_visible",
    "scene_structure_preserved",
    "cells_distinguishable",
    "generation_failure",
    "mode_collapse",
    "reviewer_id",
    "notes",
]

KEY_FIELDS = [
    "review_id",
    "pair_id",
    "item_index",
    "seed_index",
    "seed",
    "variant",
    "expected_target_visible",
    "expected_footprint_visible",
    "prompt",
    "video_path",
]
```

Implement:

```python
def review_id_for(item: dict[str, object]) -> str:
    return (
        f"c01_{int(item.get('probe_index', 0)):03d}_"
        f"s{int(item.get('seed_index', 0)):02d}_"
        f"{str(item.get('variant', 'cell'))}"
    )
```

Use `attach_video_and_strip()` from `scripts/build_baseline_review.py` exactly as `scripts/build_c0_counterfactual_review.py` does. Shuffle review rows with `random.Random(args.shuffle_seed).shuffle(rows)`. Do not shuffle `answer_key.csv`; sort key rows by `review_id`.

- [ ] **Step 4: Run review-builder tests**

Run:

```bash
PYTHONNOUSERSITE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/deepseek_VG/.conda/envs/vcecf/bin/python -m pytest \
  tests/test_build_c01_factorial_gate_review.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit Task 2**

```bash
git add scripts/build_c01_factorial_gate_review.py tests/test_build_c01_factorial_gate_review.py
git commit -m "add c01 blinded review builder"
```

---

### Task 3: Implement The C0.1 Gate Scorer

**Files:**
- Create: `scripts/score_c01_factorial_gate.py`
- Create: `tests/test_score_c01_factorial_gate.py`

- [ ] **Step 1: Add failing scorer tests**

Create `tests/test_score_c01_factorial_gate.py`:

```python
import csv
import importlib.util
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "score_c01_factorial_gate.py"


def load_module():
    spec = importlib.util.spec_from_file_location("score_c01_factorial_gate", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_rows(pair_id: str = "pair_a") -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    review_rows = []
    key_rows = []
    expected = {
        "original": ("yes", "yes"),
        "remove_target": ("no", "no"),
        "footprint_only": ("no", "yes"),
        "target_only": ("yes", "no"),
    }
    for seed_index in range(5):
        for variant, (target_expected, footprint_expected) in expected.items():
            review_id = f"c01_000_s{seed_index:02d}_{variant}"
            target_label = "present" if target_expected == "yes" else "absent"
            footprint_label = "present" if footprint_expected == "yes" else "absent"
            review_rows.append(
                {
                    "review_id": review_id,
                    "target_visible": target_label,
                    "footprint_visible": footprint_label,
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


def test_score_gate_passes_clean_item(tmp_path):
    module = load_module()
    review_rows, key_rows = make_rows()
    review_csv = tmp_path / "blind_review.csv"
    key_csv = tmp_path / "answer_key.csv"
    write_csv(review_csv, review_rows)
    write_csv(key_csv, key_rows)

    output_dir = tmp_path / "scores"
    result = module.main(["--review-csv", str(review_csv), "--answer-key", str(key_csv), "--output-dir", str(output_dir)])

    assert result == 0
    item_rows = list(csv.DictReader((output_dir / "item_gate_summary.csv").open(encoding="utf-8")))
    assert item_rows[0]["gate_status"] == "pass"
    assert item_rows[0]["original_successes"] == "5"
    assert item_rows[0]["footprint_only_successes"] == "5"


def test_score_gate_fails_uncertain_and_scene_drift(tmp_path):
    review_rows, key_rows = make_rows(pair_id="pair_b")
    review_rows[0]["target_visible"] = "uncertain"
    review_rows[1]["scene_structure_preserved"] = "no"
    review_csv = tmp_path / "blind_review.csv"
    key_csv = tmp_path / "answer_key.csv"
    write_csv(review_csv, review_rows)
    write_csv(key_csv, key_rows)

    module = load_module()
    output_dir = tmp_path / "scores"
    module.main(["--review-csv", str(review_csv), "--answer-key", str(key_csv), "--output-dir", str(output_dir)])

    item_rows = list(csv.DictReader((output_dir / "item_gate_summary.csv").open(encoding="utf-8")))
    assert item_rows[0]["gate_status"] == "fail"
    assert "review_uncertain" in item_rows[0]["rejection_reasons"]
    assert "scene_drift" in item_rows[0]["rejection_reasons"]
```

- [ ] **Step 2: Run the new scorer tests and verify they fail**

Run:

```bash
PYTHONNOUSERSITE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/deepseek_VG/.conda/envs/vcecf/bin/python -m pytest \
  tests/test_score_c01_factorial_gate.py -q
```

Expected: fail because `scripts/score_c01_factorial_gate.py` does not exist.

- [ ] **Step 3: Create the scorer**

Create `scripts/score_c01_factorial_gate.py` with:

```python
THRESHOLDS = {
    "original": 4,
    "remove_target": 4,
    "target_only": 4,
    "footprint_only": 3,
}

EXPECTED_VARIANTS = ["original", "remove_target", "footprint_only", "target_only"]
```

Implement label normalization:

```python
def normalize_presence(value: str) -> str:
    value = str(value).strip().lower()
    aliases = {
        "yes": "present",
        "true": "present",
        "1": "present",
        "strong": "present",
        "present": "present",
        "no": "absent",
        "false": "absent",
        "0": "absent",
        "absent": "absent",
        "uncertain": "uncertain",
        "unknown": "uncertain",
        "": "uncertain",
    }
    if value not in aliases:
        raise ValueError(f"unknown presence label: {value}")
    return aliases[value]
```

Implement success logic:

```python
def expected_to_presence(value: str) -> str:
    return "present" if str(value).strip().lower() == "yes" else "absent"


def cell_success(review: dict[str, str], key: dict[str, str]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    target = normalize_presence(review.get("target_visible", ""))
    footprint = normalize_presence(review.get("footprint_visible", ""))
    if target == "uncertain" or footprint == "uncertain":
        reasons.append("review_uncertain")
    if str(review.get("scene_structure_preserved", "yes")).strip().lower() in {"no", "false", "0"}:
        reasons.append("scene_drift")
    if str(review.get("generation_failure", "no")).strip().lower() in {"yes", "true", "1"}:
        reasons.append("generation_failure")
    if str(review.get("mode_collapse", "no")).strip().lower() in {"yes", "true", "1"}:
        reasons.append("mode_collapse")
    if target != "uncertain" and target != expected_to_presence(key["expected_target_visible"]):
        if key["variant"] == "original":
            reasons.append("original_unreliable")
        elif key["variant"] == "remove_target":
            reasons.append("remove_target_failed")
        else:
            reasons.append(f"{key['variant']}_target_mismatch")
    if footprint != "uncertain" and footprint != expected_to_presence(key["expected_footprint_visible"]):
        if key["variant"] == "target_only":
            reasons.append("target_only_preserves_footprint")
        elif key["variant"] == "footprint_only":
            reasons.append("footprint_only_incoherent")
        elif key["variant"] == "original":
            reasons.append("original_unreliable")
        elif key["variant"] == "remove_target":
            reasons.append("remove_target_failed")
    return not reasons, sorted(set(reasons))
```

Write `cell_gate_summary.csv` with one row per review id and `item_gate_summary.csv` with one row per pair id. An item passes only when every `EXPECTED_VARIANTS` count meets `THRESHOLDS`.

- [ ] **Step 4: Run scorer tests**

Run:

```bash
PYTHONNOUSERSITE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/deepseek_VG/.conda/envs/vcecf/bin/python -m pytest \
  tests/test_score_c01_factorial_gate.py -q
```

Expected: all scorer tests pass.

- [ ] **Step 5: Commit Task 3**

```bash
git add scripts/score_c01_factorial_gate.py tests/test_score_c01_factorial_gate.py
git commit -m "add c01 gate scorer"
```

---

### Task 4: Validate End-To-End Dry Run

**Files:**
- Modify: `docs/experiment_log.md`

- [ ] **Step 1: Run the full focused test suite**

Run:

```bash
PYTHONNOUSERSITE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/deepseek_VG/.conda/envs/vcecf/bin/python -m pytest \
  tests/test_run_c0_counterfactual_grid.py \
  tests/test_build_c01_factorial_gate_review.py \
  tests/test_score_c01_factorial_gate.py \
  tests/test_build_c0_counterfactual_review.py \
  tests/test_run_zeroscope_attention_probe.py -q
```

Expected: all selected tests pass.

- [ ] **Step 2: Generate a C0.1 dry-run manifest**

Run:

```bash
PYTHONNOUSERSITE=1 /home/deepseek_VG/.conda/envs/vcecf/bin/python \
  scripts/adapters/run_c0_counterfactual_grid.py \
  --probe-manifest experiments/method_probe/zeroscope_mvp0_causal_chain_probe_20260702/probe_manifest.json \
  --output-dir experiments/method_probe/c01_factorial_gate_20260703_dryrun \
  --limit-items 3 \
  --seed 41000 \
  --seeds-per-item 5 \
  --dry-run
```

Expected: `generation_manifest.json` contains 60 items.

- [ ] **Step 3: Build blinded review dry-run artifacts**

Run:

```bash
PYTHONNOUSERSITE=1 /home/deepseek_VG/.conda/envs/vcecf/bin/python \
  scripts/build_c01_factorial_gate_review.py \
  --generation-manifest experiments/method_probe/c01_factorial_gate_20260703_dryrun/generation_manifest.json \
  --output-dir experiments/evaluation/c01_factorial_gate_20260703_dryrun \
  --skip-frame-extraction \
  --shuffle-seed 17
```

Expected:

```text
experiments/evaluation/c01_factorial_gate_20260703_dryrun/blind_review.csv
experiments/evaluation/c01_factorial_gate_20260703_dryrun/answer_key.csv
```

Both CSVs contain 60 data rows.

- [ ] **Step 4: Smoke-score a synthetic completed review**

Run this helper to create a synthetic review file from the dry-run review CSV:

```bash
PYTHONNOUSERSITE=1 /home/deepseek_VG/.conda/envs/vcecf/bin/python - <<'PY'
import csv
from pathlib import Path

root = Path("experiments/evaluation/c01_factorial_gate_20260703_dryrun")
review = list(csv.DictReader((root / "blind_review.csv").open(encoding="utf-8")))
key_by_id = {
    row["review_id"]: row
    for row in csv.DictReader((root / "answer_key.csv").open(encoding="utf-8"))
}
for row in review:
    key = key_by_id[row["review_id"]]
    row["target_visible"] = "present" if key["expected_target_visible"] == "yes" else "absent"
    row["footprint_visible"] = "present" if key["expected_footprint_visible"] == "yes" else "absent"
    row["scene_structure_preserved"] = "yes"
    row["cells_distinguishable"] = "yes"
    row["generation_failure"] = "no"
    row["mode_collapse"] = "no"
    row["reviewer_id"] = "synthetic"
out = root / "synthetic_completed_review.csv"
with out.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(review[0]))
    writer.writeheader()
    writer.writerows(review)
print(out)
PY
```

Then run:

```bash
PYTHONNOUSERSITE=1 /home/deepseek_VG/.conda/envs/vcecf/bin/python \
  scripts/score_c01_factorial_gate.py \
  --review-csv experiments/evaluation/c01_factorial_gate_20260703_dryrun/synthetic_completed_review.csv \
  --answer-key experiments/evaluation/c01_factorial_gate_20260703_dryrun/answer_key.csv \
  --output-dir experiments/evaluation/c01_factorial_gate_20260703_dryrun/synthetic_scores
```

Expected: `item_gate_summary.csv` marks all three items as `pass`.

- [ ] **Step 5: Update the experiment log**

Append a short entry to `docs/experiment_log.md`:

```markdown
## 2026-07-03: C0.1 Factorial Gate Dry Run

Implemented the C0.1 seed-matched factorial prompt gate infrastructure:
multi-seed C0 generation manifests, blinded human-review CSVs, answer keys,
and deterministic gate scoring. The dry run expanded 3 MVP-0 items into
60 planned rows (3 items x 5 seeds x 4 cells). A synthetic completed-review
smoke confirmed that the scorer applies the 4/5 and 3/5 thresholds as designed.

No GPU generation or C1 repair claim is made in this step.
```

- [ ] **Step 6: Commit Task 4**

```bash
git add docs/experiment_log.md experiments/method_probe/c01_factorial_gate_20260703_dryrun experiments/evaluation/c01_factorial_gate_20260703_dryrun
git commit -m "validate c01 gate dry run"
```

---

### Task 5: Decide Whether To Run The Real 60-Video Pilot

**Files:**
- No code files.

- [ ] **Step 1: Check GPU availability**

Run:

```bash
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
```

Expected: choose the GPU with the lowest memory and utilization.

- [ ] **Step 2: Run the real C0.1 pilot only after human approval**

The command below uses GPU 0. If Step 1 shows a different least-busy GPU,
replace the single `CUDA_VISIBLE_DEVICES=0` value with that GPU index before
running.

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 \
  /home/deepseek_VG/.conda/envs/vcecf/bin/python \
  scripts/adapters/run_c0_counterfactual_grid.py \
  --probe-manifest experiments/method_probe/zeroscope_mvp0_causal_chain_probe_20260702/probe_manifest.json \
  --output-dir experiments/method_probe/c01_factorial_gate_20260703_quality3 \
  --limit-items 3 \
  --seed 41000 \
  --seeds-per-item 5 \
  --steps 10 \
  --num-frames 16 \
  --height 240 \
  --width 432 \
  --dtype fp16 \
  --device cuda \
  --vae-slicing
```

Expected: 60 MP4 files and a `generation_manifest.json`.

- [ ] **Step 3: Build real blinded review artifacts**

Run:

```bash
PYTHONNOUSERSITE=1 /home/deepseek_VG/.conda/envs/vcecf/bin/python \
  scripts/build_c01_factorial_gate_review.py \
  --generation-manifest experiments/method_probe/c01_factorial_gate_20260703_quality3/generation_manifest.json \
  --output-dir experiments/evaluation/c01_factorial_gate_quality3_human_20260703 \
  --shuffle-seed 17
```

Expected: `blind_review.csv`, `answer_key.csv`, and frame strips for human review.

- [ ] **Step 4: Stop for human labeling**

Do not score or proceed to C1 until the human review CSV has been filled with
`present`, `absent`, or `uncertain` labels and scene/cell flags.
