# C0.2 Discrete Factorial Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add C0.2 support for selecting discrete candidate items, applying stricter four-cell prompt templates, generating a 48-row dry run, and producing contact sheets for human spot-check.

**Architecture:** Extend the existing C0 grid runner instead of adding a second generator. Add an item-index selector and a prompt-template mode that uses item-specific surface overrides. Add a small contact-sheet builder that consumes the C0.2 generation manifest and existing frame strips.

**Tech Stack:** Python standard library, pytest, existing ZeroScope C0 runner, existing C0.1 review builder, Pillow for contact-sheet assembly.

---

## File Structure

- Modify: `scripts/adapters/run_c0_counterfactual_grid.py`
  - Add `--item-indices`.
  - Add `--prompt-template {legacy,c02_discrete}`.
  - Store selected item indices and prompt template in generation config.
- Modify: `tests/test_run_c0_counterfactual_grid.py`
  - Add failing tests for item-index selection and C0.2 prompt templates.
- Create: `scripts/build_c02_spotcheck_sheets.py`
  - Build one contact sheet per item from C0.2 frame strips and answer key.
- Create: `tests/test_build_c02_spotcheck_sheets.py`
  - Verify one sheet per item, all seeds/cells represented, and variant labels visible only in the sheet, not in blind review.
- Modify: `docs/experiment_log.md`
  - Record dry-run paths and integrity checks after validation.

---

### Task 1: Add Item Selection And C0.2 Prompt Templates

**Files:**
- Modify: `scripts/adapters/run_c0_counterfactual_grid.py`
- Modify: `tests/test_run_c0_counterfactual_grid.py`

- [x] **Step 1: Add failing tests**

Append tests that assert:

```python
def test_item_indices_selects_non_contiguous_probe_items(tmp_path):
    module = load_runner_module()
    probe_manifest = write_multi_item_probe_manifest(tmp_path)
    args = module.build_parser().parse_args(
        [
            "--probe-manifest",
            str(probe_manifest),
            "--output-dir",
            str(tmp_path / "out"),
            "--seed",
            "50000",
            "--item-indices",
            "3,10",
            "--dry-run",
        ]
    )

    selected = module.select_probe_items(
        json.loads(probe_manifest.read_text())["items"],
        args,
    )

    assert [item["probe_index"] for item in selected] == [3, 10]
```

and:

```python
def test_c02_discrete_prompt_template_makes_all_factors_explicit(tmp_path):
    module = load_runner_module()
    probe_manifest = write_multi_item_probe_manifest(tmp_path)
    args = module.build_parser().parse_args(
        [
            "--probe-manifest",
            str(probe_manifest),
            "--output-dir",
            str(tmp_path / "out"),
            "--seed",
            "50000",
            "--item-indices",
            "10",
            "--prompt-template",
            "c02_discrete",
        ]
    )

    rows = module.build_items(args, module.select_probe_items(json.loads(probe_manifest.read_text())["items"], args))
    prompts = {row["variant"]: row["prompt"] for row in rows}

    assert "same simple scene" in prompts["original"]
    assert "marker pen is clearly visible and contacts the whiteboard surface" in prompts["original"]
    assert "No marker pen is present" in prompts["remove_target"]
    assert "There is no a black line remains on the whiteboard" in prompts["remove_target"]
    assert "No marker pen is present" in prompts["footprint_only"]
    assert "a black line remains on the whiteboard is clearly visible on the whiteboard surface" in prompts["footprint_only"]
    assert "marker pen is clearly visible, but it is separated from the whiteboard surface" in prompts["target_only"]
    assert "does not touch, strike, mark, press, disturb, or change it" in prompts["target_only"]
```

- [x] **Step 2: Verify RED**

Run:

```bash
PYTHONNOUSERSITE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/deepseek_VG/.conda/envs/vcecf/bin/python -m pytest \
  tests/test_run_c0_counterfactual_grid.py::test_item_indices_selects_non_contiguous_probe_items \
  tests/test_run_c0_counterfactual_grid.py::test_c02_discrete_prompt_template_makes_all_factors_explicit -q
```

Expected: fail because `write_multi_item_probe_manifest`, `--item-indices`,
`--prompt-template`, and `select_probe_items` do not exist.

- [x] **Step 3: Implement selection and prompt templates**

Add:

```python
C02_SURFACE_OVERRIDES = {
    "makeup brush": "compact of pink powder",
    "garden rake": "smooth soil bed",
    "hand": "pillow surface",
    "marker pen": "whiteboard surface",
}
PROMPT_TEMPLATES = ["legacy", "c02_discrete"]
```

Add parser args:

```python
parser.add_argument("--item-indices", default="")
parser.add_argument("--prompt-template", choices=PROMPT_TEMPLATES, default="legacy")
```

Add helper functions:

```python
def parse_item_indices(text: str) -> list[int]:
    ...

def select_probe_items(probe_items: Sequence[dict], args: argparse.Namespace) -> list[dict]:
    ...

def c02_surface_for(item: dict[str, object]) -> str:
    ...

def c02_discrete_prompt(item: dict[str, object], variant: str) -> tuple[str, str]:
    ...
```

Make `variant_prompt` dispatch to C0.2 when `args.prompt_template == "c02_discrete"` by changing its signature to accept `args`.

- [x] **Step 4: Verify GREEN**

Run:

```bash
PYTHONNOUSERSITE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/deepseek_VG/.conda/envs/vcecf/bin/python -m pytest \
  tests/test_run_c0_counterfactual_grid.py -q
```

Expected: all runner tests pass.

---

### Task 2: Validate C0.2 Dry Run

**Files:**
- Modify: `docs/experiment_log.md`

- [x] **Step 1: Run C0.2 dry run**

Run:

```bash
PYTHONNOUSERSITE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/deepseek_VG/.conda/envs/vcecf/bin/python \
  scripts/adapters/run_c0_counterfactual_grid.py \
  --probe-manifest experiments/method_probe/zeroscope_mvp0_causal_chain_probe_20260702/probe_manifest.json \
  --output-dir experiments/method_probe/c02_discrete_factorial_gate_20260704_dryrun \
  --item-indices 3,4,8,10 \
  --prompt-template c02_discrete \
  --seed 52000 \
  --seeds-per-item 3 \
  --dry-run
```

- [x] **Step 2: Check dry-run integrity**

Verify:

```text
rows=48
variants: 12 each
probe_index set: 3,4,8,10
seed_index set: 0,1,2
prompt_template=c02_discrete
```

- [x] **Step 3: Log dry-run result**

Append a concise C0.2 dry-run section to `docs/experiment_log.md`, including
the dry-run path, row counts, candidate indices, and the real-run command.

- [x] **Step 4: Run focused tests**

Run:

```bash
PYTHONNOUSERSITE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/deepseek_VG/.conda/envs/vcecf/bin/python -m pytest \
  tests/test_run_c0_counterfactual_grid.py \
  tests/test_build_c01_factorial_gate_review.py -q
```

---

### Task 3: Build Spot-Check Contact Sheets

**Files:**
- Create: `scripts/build_c02_spotcheck_sheets.py`
- Create: `tests/test_build_c02_spotcheck_sheets.py`

- [x] **Step 1: Add failing contact-sheet tests**

Create a test that builds a tiny answer key and fake frame strips for two items,
then asserts the script writes one contact sheet per item and a JSON summary.

- [x] **Step 2: Verify RED**

Run the focused test and confirm it fails because the script does not exist.

- [x] **Step 3: Implement sheet builder**

Read `answer_key.csv`, group by `item_index`, `seed_index`, and `variant`, find
each `review_id` strip in `frame_strips/`, and write:

```text
spotcheck_contact_sheets/item_<item_index>_all_seeds_four_cells.jpg
spotcheck_contact_sheets/spotcheck_manifest.json
```

- [x] **Step 4: Verify GREEN**

Run:

```bash
PYTHONNOUSERSITE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/deepseek_VG/.conda/envs/vcecf/bin/python -m pytest \
  tests/test_build_c02_spotcheck_sheets.py -q
```

---

### Task 4: Commit And Real Run Decision

**Files:**
- Modify: `docs/experiment_log.md`

- [ ] Run the combined focused test suite.
- [ ] Commit C0.2 implementation and dry-run artifacts.
- [ ] If dry-run checks pass and GPU is available, run the real 48-video C0.2 pilot.
- [ ] Build C0.1-style review artifacts and C0.2 contact sheets for the real pilot.
- [ ] Log real-run paths and mechanical integrity checks.
- [ ] Commit real-run CSV/JSON summaries, leaving generated videos/images local according to `.gitignore`.
