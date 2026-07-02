# Causal Chain Steering Phase B Paraphrase Averaging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make causal-chain steering directions less brittle by averaging multiple paraphrased positive/negative minimal pairs per causal link.

**Architecture:** Keep the existing runner API compatible with single-pair manifests. Extend the runner so each `minimal_pairs[link]` value may be either one `{positive, negative}` pair or a list of such pairs. At each denoising step, compute each pair's residual direction, average those directions per link, and apply the averaged direction through the existing steering path. Random controls must norm-match the averaged footprint direction.

**Tech Stack:** Python standard library, pytest, existing ZeroScope runner and probe manifest format.

---

### Task 1: Add Multi-Pair Normalization and Averaging Tests

**Files:**
- Modify: `tests/test_run_mvp0_zeroscope_probe.py`
- Modify later: `scripts/adapters/run_mvp0_zeroscope_probe.py`

- [ ] **Step 1: Write failing tests**

Append these tests to `tests/test_run_mvp0_zeroscope_probe.py`:

```python
def test_normalize_minimal_pair_value_accepts_single_and_list_pairs():
    module = load_runner_module()
    single = {"positive": "with pebble", "negative": "without pebble"}
    multi = [
        {"positive": "with pebble", "negative": "without pebble"},
        {"positive": "pebble present", "negative": "pebble absent"},
    ]

    assert module.normalize_minimal_pair_value(single) == [single]
    assert module.normalize_minimal_pair_value(multi) == multi


def test_average_pair_predictions_preserves_single_pair_direction():
    module = load_runner_module()
    predictions = [
        {"positive": [5.0, 7.0], "negative": [2.0, 3.0]},
    ]

    averaged = module.average_pair_predictions(predictions)

    assert averaged == {"positive": [3.0, 4.0], "negative": [0.0, 0.0]}


def test_average_pair_predictions_averages_multiple_directions():
    module = load_runner_module()
    predictions = [
        {"positive": [5.0, 7.0], "negative": [2.0, 3.0]},
        {"positive": [10.0, 4.0], "negative": [4.0, 2.0]},
        {"positive": [1.0, 9.0], "negative": [0.0, 3.0]},
    ]

    averaged = module.average_pair_predictions(predictions)

    assert averaged == {"positive": [10.0 / 3.0, 4.0], "negative": [0.0, 0.0]}
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python -m pytest tests/test_run_mvp0_zeroscope_probe.py::test_normalize_minimal_pair_value_accepts_single_and_list_pairs tests/test_run_mvp0_zeroscope_probe.py::test_average_pair_predictions_preserves_single_pair_direction tests/test_run_mvp0_zeroscope_probe.py::test_average_pair_predictions_averages_multiple_directions -q
```

Expected: FAIL because `normalize_minimal_pair_value` and `average_pair_predictions` do not exist.

### Task 2: Implement Pair Normalization and Direction Averaging

**Files:**
- Modify: `scripts/adapters/run_mvp0_zeroscope_probe.py`
- Test: `tests/test_run_mvp0_zeroscope_probe.py`

- [ ] **Step 1: Add helper implementation**

Add helper functions near `_diff` in `scripts/adapters/run_mvp0_zeroscope_probe.py`:

```python
def normalize_minimal_pair_value(value):
    if isinstance(value, list):
        return value
    return [value]


def _zero_like(value):
    try:
        return value * 0
    except TypeError:
        return [part * 0 for part in value]


def _add(left, right):
    try:
        return left + right
    except TypeError:
        return [a + b for a, b in zip(left, right)]


def _scale(value, factor: float):
    try:
        return value * factor
    except TypeError:
        return [part * factor for part in value]


def average_pair_predictions(predictions):
    directions = [_diff(pair["positive"], pair["negative"]) for pair in predictions]
    if not directions:
        return None
    total = directions[0]
    for direction in directions[1:]:
        total = _add(total, direction)
    averaged = _scale(total, 1.0 / len(directions))
    return {"positive": averaged, "negative": _zero_like(averaged)}
```

- [ ] **Step 2: Run tests to verify they pass**

Run:

```bash
python -m pytest tests/test_run_mvp0_zeroscope_probe.py::test_normalize_minimal_pair_value_accepts_single_and_list_pairs tests/test_run_mvp0_zeroscope_probe.py::test_average_pair_predictions_preserves_single_pair_direction tests/test_run_mvp0_zeroscope_probe.py::test_average_pair_predictions_averages_multiple_directions -q
```

Expected: PASS.

### Task 3: Add Encoding and Runtime Averaging Tests

**Files:**
- Modify: `tests/test_run_mvp0_zeroscope_probe.py`
- Modify later: `scripts/adapters/run_mvp0_zeroscope_probe.py`

- [ ] **Step 1: Write failing tests for multi-pair encoding**

Append:

```python
def test_encode_pair_embeds_encodes_multiple_pairs(monkeypatch):
    module = load_runner_module()
    calls = []

    def fake_encode_cfg(pipe, torch_module, *, prompt, negative_prompt, device):
        calls.append(prompt)
        return [f"pos:{prompt}"], [f"neg:{prompt}"]

    class FakeTorch:
        @staticmethod
        def cat(values):
            out = []
            for value in values:
                out.extend(value)
            return out

    monkeypatch.setattr(module, "encode_cfg", fake_encode_cfg)
    row = {
        "negative_prompt": "",
        "steering": {
            "links": ["cause"],
            "alpha": 0.25,
            "minimal_pairs": {
                "cause": [
                    {"positive": "cause p1", "negative": "cause n1"},
                    {"positive": "cause p2", "negative": "cause n2"},
                ]
            },
        },
    }

    embeds = module.encode_pair_embeds(None, FakeTorch, "cuda", row)

    assert calls == ["cause p1", "cause n1", "cause p2", "cause n2"]
    assert len(embeds["cause"]) == 2
    assert embeds["cause"][0]["positive"] == ["neg:cause p1", "pos:cause p1"]
    assert embeds["cause"][1]["negative"] == ["neg:cause n1", "pos:cause n1"]
```

- [ ] **Step 2: Write failing test for runtime averaging**

Append:

```python
def test_run_steered_pipeline_averages_multi_pair_predictions(monkeypatch):
    module = load_runner_module()
    captured = []

    def fake_guided_noise_pred(pipe, latent_model_input, timestep, embeds, cross_attention_kwargs=None):
        return embeds

    def fake_apply_cfg(noise_pred, guidance_scale):
        return noise_pred

    def fake_synthesize_random(torch_module, link_predictions, row, *, step_index):
        return None

    def fake_apply(main_residual, link_predictions, row, *, step_index, alpha, timestep_window):
        captured.append(link_predictions["cause"])
        return main_residual

    monkeypatch.setattr(module, "_guided_noise_pred", fake_guided_noise_pred)
    monkeypatch.setattr(module, "apply_cfg", fake_apply_cfg)
    monkeypatch.setattr(module, "synthesize_random_control_prediction", fake_synthesize_random)
    monkeypatch.setattr(module, "apply_steering_residual", fake_apply)

    class FakeTensor(list):
        def chunk(self, parts):
            return self[:1], self[1:]

        def permute(self, *args):
            return self

        def reshape(self, *args):
            return self

    class FakeTorch:
        @staticmethod
        def cat(values):
            out = FakeTensor()
            for value in values:
                out.extend(value)
            return out

        @staticmethod
        def no_grad():
            class Context:
                def __enter__(self): return None
                def __exit__(self, exc_type, exc, tb): return False
            return Context()

    class FakeScheduler:
        timesteps = [0]

        def set_timesteps(self, steps, device):
            self.timesteps = [0]

        def scale_model_input(self, latent_model_input, timestep):
            return latent_model_input

        def step(self, scheduler_noise_pred, timestep, scheduler_latents, **kwargs):
            return type("Step", (), {"prev_sample": FakeLatents()})()

    class FakeLatents(FakeTensor):
        shape = (1, 1, 1, 1, 1)

        def __getitem__(self, value):
            return self

    class FakePipe:
        _execution_device = "cpu"
        scheduler = FakeScheduler()
        unet = type("Unet", (), {"config": type("Config", (), {"in_channels": 1})()})()

        def prepare_latents(self, *args):
            return FakeLatents([0.0])

        def prepare_extra_step_kwargs(self, generator, eta):
            return {}

        def progress_bar(self, total):
            class Progress:
                def __enter__(self): return self
                def __exit__(self, exc_type, exc, tb): return False
                def update(self): return None
            return Progress()

        def maybe_free_model_hooks(self):
            return None

    module.run_steered_pipeline(
        FakePipe(),
        FakeTorch,
        row={"steering": {"links": ["cause"]}},
        prompt_embeds=FakeTensor([100.0]),
        negative_prompt_embeds=FakeTensor([0.0]),
        link_embeds={
            "cause": [
                {"positive": FakeTensor([5.0]), "negative": FakeTensor([2.0])},
                {"positive": FakeTensor([10.0]), "negative": FakeTensor([4.0])},
            ]
        },
        generator=None,
        steps=1,
        num_frames=1,
        guidance_scale=1.0,
        height=1,
        width=1,
        alpha=0.25,
        timestep_window=(0, 0),
        output_type="latent",
    )

    assert captured == [{"positive": [4.5], "negative": [0.0]}]
```

- [ ] **Step 3: Run tests to verify they fail**

Run:

```bash
python -m pytest tests/test_run_mvp0_zeroscope_probe.py::test_encode_pair_embeds_encodes_multiple_pairs tests/test_run_mvp0_zeroscope_probe.py::test_run_steered_pipeline_averages_multi_pair_predictions -q
```

Expected: FAIL because encoding/runtime still expect one pair per link.

### Task 4: Implement Multi-Pair Encoding and Runtime Averaging

**Files:**
- Modify: `scripts/adapters/run_mvp0_zeroscope_probe.py`
- Test: `tests/test_run_mvp0_zeroscope_probe.py`

- [ ] **Step 1: Update `encode_pair_embeds`**

Change the function so each link maps to a list of encoded pair dictionaries:

```python
for pair in normalize_minimal_pair_value(minimal_pairs.get(pair_key)):
    encoded_pair = {}
    for side in ["positive", "negative"]:
        ...
        encoded_pair[side] = ...
    pair_embeds.setdefault(embed_key, []).append(encoded_pair)
```

Skip missing pairs as before. Existing single-pair manifests should produce a
one-element list.

- [ ] **Step 2: Update `run_steered_pipeline` link prediction construction**

Change the per-step loop from one pair per link to many pairs:

```python
for link, encoded_pairs in link_embeds.items():
    pair_predictions = []
    for pair in encoded_pairs:
        prediction_pair = {}
        for side, embeds in pair.items():
            pred = _guided_noise_pred(...)
            ...
            prediction_pair[side] = pred
        pair_predictions.append(prediction_pair)
    averaged = average_pair_predictions(pair_predictions)
    if averaged is not None:
        link_predictions[link] = averaged
```

- [ ] **Step 3: Run focused tests**

Run:

```bash
python -m pytest tests/test_run_mvp0_zeroscope_probe.py::test_encode_pair_embeds_encodes_multiple_pairs tests/test_run_mvp0_zeroscope_probe.py::test_run_steered_pipeline_averages_multi_pair_predictions tests/test_run_mvp0_zeroscope_probe.py::test_encode_pair_embeds_encodes_random_reference tests/test_run_mvp0_zeroscope_probe.py::test_synthesize_random_control_prediction_matches_reference_norm -q
```

Expected: PASS.

### Task 5: Add Phase B Manifest Builder

**Files:**
- Create: `scripts/build_mvp0_phase_b_paraphrase_probe.py`
- Create: `tests/test_build_mvp0_phase_b_paraphrase_probe.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_build_mvp0_phase_b_paraphrase_probe.py`:

```python
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
    assert item["minimal_pairs"]["cause"][0] == {"positive": "with pebble", "negative": "without pebble"}
    assert "pebble is visible" in item["minimal_pairs"]["cause"][1]["positive"]
    assert "no circular ripples spread outward" in item["minimal_pairs"]["footprint"][2]["negative"]
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python -m pytest tests/test_build_mvp0_phase_b_paraphrase_probe.py -q
```

Expected: FAIL because the builder script does not exist.

- [ ] **Step 3: Implement builder**

Create `scripts/build_mvp0_phase_b_paraphrase_probe.py` with deterministic
rule-based paraphrases. Preserve the original first pair, then add two simple
variants per link using `target_concept`, `causal_footprint`, and
`mechanism_type`.

- [ ] **Step 4: Run builder test**

Run:

```bash
python -m pytest tests/test_build_mvp0_phase_b_paraphrase_probe.py -q
```

Expected: PASS.

### Task 6: Build Phase B Manifest and Dry-Run

**Files:**
- Input: `experiments/method_probe/zeroscope_mvp0_causal_chain_probe_20260702/probe_manifest.json`
- Output: `experiments/method_probe/zeroscope_mvp0_causal_chain_probe_20260702/phase_b_paraphrase_probe_manifest.json`
- Output: `experiments/method_probe/zeroscope_mvp0_causal_chain_probe_20260702/phase_b_dry_run/generation_manifest.json`

- [ ] **Step 1: Build Phase B manifest**

Run:

```bash
python scripts/build_mvp0_phase_b_paraphrase_probe.py \
  --probe-manifest experiments/method_probe/zeroscope_mvp0_causal_chain_probe_20260702/probe_manifest.json \
  --output experiments/method_probe/zeroscope_mvp0_causal_chain_probe_20260702/phase_b_paraphrase_probe_manifest.json
```

Expected: manifest has the same 3 pilot items and 3 pairs per causal link.

- [ ] **Step 2: Dry-run runner against Phase B manifest**

Run:

```bash
PYTHONNOUSERSITE=1 /home/deepseek_VG/.conda/envs/vcecf/bin/python scripts/adapters/run_mvp0_zeroscope_probe.py \
  --probe-manifest experiments/method_probe/zeroscope_mvp0_causal_chain_probe_20260702/phase_b_paraphrase_probe_manifest.json \
  --output-dir experiments/method_probe/zeroscope_mvp0_causal_chain_probe_20260702/phase_b_dry_run \
  --model models/zeroscope_v2_576w \
  --seed 15000 \
  --condition target_negative \
  --condition target_footprint_negative \
  --condition full_chain_steering \
  --condition random_direction \
  --condition orthogonal_semantic \
  --limit-items 3 \
  --dry-run \
  --strict-prompt-length
```

Expected: dry-run manifest contains 15 items. `full_chain_steering` has 3
pairs for cause, mechanism, and footprint; `random_direction` records
`control_reference=footprint`.

### Task 7: Run One Phase B Real Cell and Evaluate

**Files:**
- Output: `experiments/method_probe/zeroscope_mvp0_causal_chain_probe_20260702/phase_b_real_alpha_0p25_window_3_6/`
- Output: `experiments/evaluation/mvp0_zeroscope_phase_b_alpha_0p25_window_3_6_fable_20260702/`

- [ ] **Step 1: Run real cell**

Use the midpoint conservative cell first:

```bash
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=<GPU_ID> PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/deepseek_VG/.conda/envs/vcecf/bin/python scripts/adapters/run_mvp0_zeroscope_probe.py \
  --probe-manifest experiments/method_probe/zeroscope_mvp0_causal_chain_probe_20260702/phase_b_paraphrase_probe_manifest.json \
  --output-dir experiments/method_probe/zeroscope_mvp0_causal_chain_probe_20260702/phase_b_real_alpha_0p25_window_3_6 \
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

- [ ] **Step 2: Build review, run fable, compute low-level proxies**

Use the same review-building and fable evaluation pattern as Phase A. The gate
is identical: full-chain must beat random and orthogonal controls on strict
footprint leakage without increasing target leakage.

- [ ] **Step 3: Update experiment log**

Append Phase B implementation, dry-run, real-cell, fable, and proxy results to
`docs/experiment_log.md`.

### No-Git Note

This workspace directory is not currently a git repository. Replace commit
steps with explicit status notes in `docs/experiment_log.md`.
