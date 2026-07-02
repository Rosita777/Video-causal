from pathlib import Path
import importlib.util
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = PROJECT_ROOT / "scripts" / "adapters" / "run_mvp0_zeroscope_probe.py"


def load_runner_module():
    spec = importlib.util.spec_from_file_location("run_mvp0_zeroscope_probe", RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


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
                            "cause": {
                                "positive": "A calm pond surface, with pebble.",
                                "negative": "A calm pond surface, without pebble.",
                            },
                            "mechanism": {
                                "positive": "A calm pond surface, with pebble impact with water.",
                                "negative": "A calm pond surface, with no impact or causal disturbance.",
                            },
                            "footprint": {
                                "positive": "A calm pond surface, with circular ripples spread outward.",
                                "negative": "A calm pond surface, with no circular ripples spread outward.",
                            },
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_runner_uses_generation_prompt_for_source_based_conditions(tmp_path):
    module = load_runner_module()
    probe_manifest = tmp_path / "probe_manifest.json"
    probe_manifest.write_text(
        json.dumps(
            {
                "probe_name": "zeroscope_mvp0_causal_chain_probe",
                "items": [
                    {
                        "probe_index": 0,
                        "pair_id": "fracture_damage_puck_mirror",
                        "slice_index": 0,
                        "source_index": "47",
                        "mechanism_type": "fracture_damage",
                        "target_concept": "black hockey puck",
                        "causal_footprint": "a star-shaped crack spreads across the mirror",
                        "source_prompt": "long original source prompt that should not be used",
                        "generation_prompt": "compact source prompt with puck and crack",
                        "counterfactual_prompt": "compact counterfactual prompt",
                        "control_prompt": "compact control prompt",
                        "minimal_pairs": {},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    args = module.build_parser().parse_args(
        [
            "--probe-manifest",
            str(probe_manifest),
            "--output-dir",
            str(tmp_path / "out"),
            "--condition",
            "target_negative",
            "--condition",
            "full_chain_steering",
        ]
    )
    rows = module.build_items(args, json.loads(probe_manifest.read_text())["items"])

    assert [row["prompt"] for row in rows] == [
        "compact source prompt with puck and crack",
        "compact source prompt with puck and crack",
    ]
    assert all(row["source_prompt"] == "long original source prompt that should not be used" for row in rows)


def test_runner_strict_prompt_length_rejects_over_limit_prompts(tmp_path):
    module = load_runner_module()
    rows = [
        {
            "pair_id": "too_long",
            "condition": "target_negative",
            "prompt": "one two three four",
            "negative_prompt": "",
            "steering": {"minimal_pairs": {}},
        }
    ]

    class FakeTokenizer:
        model_max_length = 3

        def __call__(self, text, truncation=False):
            return SimpleNamespace(input_ids=text.split())

    try:
        module.audit_prompt_lengths(rows, FakeTokenizer(), limit=3, strict=True)
    except ValueError as exc:
        assert "too_long/target_negative prompt has 4 tokens > 3" in str(exc)
    else:
        raise AssertionError("strict prompt audit should reject over-limit prompts")


def test_prompt_length_audit_handles_multi_pair_prompts():
    module = load_runner_module()
    rows = [
        {
            "pair_id": "multi",
            "condition": "full_chain_steering",
            "prompt": "short",
            "negative_prompt": "",
            "steering": {
                "links": ["cause"],
                "minimal_pairs": {
                    "cause": [
                        {"positive": "one two", "negative": "three four"},
                        {"positive": "five six", "negative": "seven eight"},
                    ]
                },
            },
        }
    ]

    class FakeTokenizer:
        def __call__(self, text, truncation=False):
            return SimpleNamespace(input_ids=text.split())

    assert module.audit_prompt_lengths(rows, FakeTokenizer(), limit=2, strict=True) == []


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
            "--alpha",
            "0.25",
            "--timestep-window",
            "2:6",
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
    assert full_chain["steering"]["alpha"] == 0.25
    assert full_chain["steering"]["timestep_window"] == [2, 6]
    assert manifest["generation"]["steering_alpha"] == 0.25
    assert manifest["generation"]["timestep_window"] == [2, 6]
    assert full_chain["video_path"].endswith("_full_chain_steering_seed15000.mp4")


def test_mvp0_zeroscope_probe_real_mode_calls_generator_and_writes_manifest(tmp_path, monkeypatch):
    module = load_runner_module()
    probe_manifest = write_probe_manifest(tmp_path)
    output_dir = tmp_path / "probe_run"
    calls = []

    def fake_generate_probe_videos(args, rows):
        calls.append(
            {
                "conditions": list(args.condition),
                "rows": rows,
                "dry_run": args.dry_run,
            }
        )

    monkeypatch.setattr(module, "generate_probe_videos", fake_generate_probe_videos)

    result = module.main(
        [
            "--probe-manifest",
            str(probe_manifest),
            "--output-dir",
            str(output_dir),
            "--model",
            "models/zeroscope_v2_576w",
            "--seed",
            "15000",
            "--condition",
            "full_chain_steering",
            "--limit-items",
            "1",
            "--device",
            "cuda:0",
            "--enable-model-cpu-offload",
            "--vae-slicing",
        ]
    )

    assert result == 0
    assert len(calls) == 1
    assert calls[0]["conditions"] == ["full_chain_steering"]
    assert calls[0]["dry_run"] is False
    assert len(calls[0]["rows"]) == 1
    assert calls[0]["rows"][0]["steering"]["links"] == ["cause", "mechanism", "footprint"]
    manifest = json.loads((output_dir / "generation_manifest.json").read_text(encoding="utf-8"))
    assert manifest["dry_run"] is False
    assert manifest["items"][0]["condition"] == "full_chain_steering"
    assert manifest["generation"]["device"] == "cuda:0"
    assert manifest["generation"]["enable_model_cpu_offload"] is True
    assert manifest["generation"]["vae_slicing"] is True


def test_apply_steering_residual_combines_selected_links_inside_window():
    module = load_runner_module()
    main = [10.0, 20.0]
    link_predictions = {
        "cause": {"positive": [3.0, 5.0], "negative": [1.0, 2.0]},
        "footprint": {"positive": [8.0, 10.0], "negative": [5.0, 6.0]},
    }
    row = {
        "steering": {"links": ["cause", "footprint"]},
    }

    steered = module.apply_steering_residual(
        main,
        link_predictions,
        row,
        step_index=3,
        alpha=0.5,
        timestep_window=(2, 4),
    )

    assert steered == [7.5, 16.5]


def test_apply_steering_residual_leaves_controls_and_out_of_window_unchanged():
    module = load_runner_module()
    main = [10.0, 20.0]
    link_predictions = {
        "cause": {"positive": [3.0, 5.0], "negative": [1.0, 2.0]},
    }

    assert module.apply_steering_residual(
        main,
        link_predictions,
        {"steering": {"links": []}},
        step_index=3,
        alpha=0.5,
        timestep_window=(2, 4),
    ) == main
    assert module.apply_steering_residual(
        main,
        link_predictions,
        {"steering": {"links": ["cause"]}},
        step_index=5,
        alpha=0.5,
        timestep_window=(2, 4),
    ) == main


def test_apply_steering_residual_applies_random_direction_when_prediction_exists():
    module = load_runner_module()
    main = [10.0, 20.0]
    link_predictions = {
        "random": {"positive": [3.0, 5.0], "negative": [1.0, 2.0]},
    }

    steered = module.apply_steering_residual(
        main,
        link_predictions,
        {"steering": {"links": ["random"]}},
        step_index=3,
        alpha=0.5,
        timestep_window=(2, 4),
    )

    assert steered == [9.0, 18.5]


def test_steering_contract_defines_random_and_orthogonal_controls():
    module = load_runner_module()
    item = {
        "minimal_pairs": {
            "footprint": {
                "positive": "pond with ripples",
                "negative": "pond without ripples",
            }
        }
    }

    random_contract = module.steering_contract(item, "random_direction")
    assert random_contract["links"] == ["random"]
    assert random_contract["control_reference"] == "footprint"
    assert random_contract["control_type"] == "gaussian_norm_matched"

    orthogonal_contract = module.steering_contract(item, "orthogonal_semantic")
    assert orthogonal_contract["links"] == ["orthogonal_semantic"]
    assert orthogonal_contract["minimal_pairs"]["orthogonal_semantic"] == {
        "positive": "A realistic video with birds flying across the sky.",
        "negative": "A realistic video with no birds in the sky.",
    }


def test_steering_contract_preserves_manifest_orthogonal_pairs():
    module = load_runner_module()
    orthogonal_pairs = [
        {"positive": "birds in sky", "negative": "no birds in sky"},
        {"positive": "candle on table", "negative": "no candle on table"},
        {"positive": "red car on road", "negative": "no red car on road"},
    ]
    item = {
        "minimal_pairs": {
            "footprint": {
                "positive": "pond with ripples",
                "negative": "pond without ripples",
            },
            "orthogonal_semantic": orthogonal_pairs,
        }
    }

    contract = module.steering_contract(item, "orthogonal_semantic")

    assert contract["links"] == ["orthogonal_semantic"]
    assert contract["control_reference"] == "footprint"
    assert contract["control_type"] == "unrelated_semantic_direction_norm_matched"
    assert contract["minimal_pairs"]["orthogonal_semantic"] == orthogonal_pairs


def test_encode_pair_embeds_encodes_random_reference(monkeypatch):
    module = load_runner_module()
    calls = []

    def fake_encode_cfg(pipe, torch_module, *, prompt, negative_prompt, device):
        calls.append((prompt, negative_prompt, device))
        return f"pos:{prompt}", f"neg:{negative_prompt}"

    class FakeTorch:
        @staticmethod
        def cat(values):
            return "|".join(values)

    monkeypatch.setattr(module, "encode_cfg", fake_encode_cfg)
    row = {
        "negative_prompt": "pebble, ripples",
        "steering": {
            "links": ["random"],
            "control_reference": "footprint",
            "alpha": 0.5,
            "minimal_pairs": {
                "footprint": {
                    "positive": "pond with ripples",
                    "negative": "pond without ripples",
                }
            },
        },
    }

    embeds = module.encode_pair_embeds(None, FakeTorch, "cuda", row)

    assert set(embeds) == {"__random_reference__"}
    assert len(embeds["__random_reference__"]) == 1
    assert set(embeds["__random_reference__"][0]) == {"positive", "negative"}
    assert calls == [
        ("pond with ripples", "pebble, ripples", "cuda"),
        ("pond without ripples", "pebble, ripples", "cuda"),
    ]


def test_synthesize_random_control_prediction_matches_reference_norm():
    module = load_runner_module()
    link_predictions = {
        "__random_reference__": {
            "positive": [3.0, 4.0],
            "negative": [0.0, 0.0],
        }
    }
    row = {
        "seed": 123,
        "steering": {"links": ["random"]},
    }

    module.synthesize_random_control_prediction(None, link_predictions, row, step_index=2)
    direction = [
        link_predictions["random"]["positive"][0] - link_predictions["random"]["negative"][0],
        link_predictions["random"]["positive"][1] - link_predictions["random"]["negative"][1],
    ]
    norm = (direction[0] ** 2 + direction[1] ** 2) ** 0.5

    assert round(norm, 6) == 5.0


def test_synthesize_orthogonal_control_prediction_matches_reference_norm():
    module = load_runner_module()
    link_predictions = {
        "__orthogonal_reference__": {
            "positive": [3.0, 4.0],
            "negative": [0.0, 0.0],
        },
        "orthogonal_semantic": {
            "positive": [6.0, 8.0],
            "negative": [0.0, 0.0],
        },
    }
    row = {
        "seed": 123,
        "steering": {"links": ["orthogonal_semantic"]},
    }

    module.synthesize_orthogonal_control_prediction(None, link_predictions, row, step_index=2)
    direction = [
        link_predictions["orthogonal_semantic"]["positive"][0]
        - link_predictions["orthogonal_semantic"]["negative"][0],
        link_predictions["orthogonal_semantic"]["positive"][1]
        - link_predictions["orthogonal_semantic"]["negative"][1],
    ]
    norm = (direction[0] ** 2 + direction[1] ** 2) ** 0.5

    assert round(norm, 6) == 5.0
    assert direction == [3.0, 4.0]


def test_run_steered_pipeline_applies_residual_before_scheduler_step(monkeypatch):
    module = load_runner_module()
    calls = []

    class FakeTensor:
        dtype = "fake_dtype"
        shape = (1, 1, 1, 1, 1)

        def __init__(self, value):
            self.value = value

        def __sub__(self, other):
            return FakeTensor(self.value - other.value)

        def __mul__(self, scalar):
            return FakeTensor(self.value * scalar)

        def permute(self, *args):
            return self

        def reshape(self, *args):
            return self

        def __getitem__(self, key):
            return self

    class FakeTorch:
        @staticmethod
        def cat(values):
            return FakeTensor(sum(value.value for value in values))

        @staticmethod
        def no_grad():
            class Context:
                def __enter__(self):
                    return None

                def __exit__(self, exc_type, exc, tb):
                    return False

            return Context()

    class FakeUnet:
        config = SimpleNamespace(in_channels=1)

        def __call__(self, latent_model_input, timestep, encoder_hidden_states=None, **kwargs):
            if encoder_hidden_states == "main":
                return (FakeTensor(10),)
            if encoder_hidden_states == "cause_pos":
                return (FakeTensor(3),)
            if encoder_hidden_states == "cause_neg":
                return (FakeTensor(1),)
            raise AssertionError(f"unexpected embeds {encoder_hidden_states!r}")

    class FakeScheduler:
        order = 1
        timesteps = [100]

        def set_timesteps(self, steps, device=None):
            self.timesteps = [100]

        def scale_model_input(self, latent_model_input, timestep):
            return latent_model_input

        def step(self, noise_pred, timestep, latents, **kwargs):
            calls.append(("scheduler_step", noise_pred.value))
            return SimpleNamespace(prev_sample=FakeTensor(noise_pred.value))

    class FakePipe:
        unet = FakeUnet()
        scheduler = FakeScheduler()
        vae_scale_factor = 1
        _execution_device = "cpu"

        def check_inputs(self, *args, **kwargs):
            return None

        def encode_prompt(self, *args, **kwargs):
            return "unused", "unused"

        def prepare_latents(self, *args, **kwargs):
            return FakeTensor(1)

        def prepare_extra_step_kwargs(self, *args, **kwargs):
            return {}

        def progress_bar(self, total):
            class Progress:
                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc, tb):
                    return False

                def update(self):
                    return None

            return Progress()

        def decode_latents(self, latents):
            return latents

        def maybe_free_model_hooks(self):
            return None

    def fake_apply(main_residual, link_predictions, row, *, step_index, alpha, timestep_window):
        calls.append(("apply", main_residual.value, link_predictions["cause"]["positive"].value))
        return FakeTensor(7)

    monkeypatch.setattr(module, "apply_steering_residual", fake_apply)

    result = module.run_steered_pipeline(
        FakePipe(),
        FakeTorch,
        row={"steering": {"links": ["cause"]}},
        prompt_embeds="main",
        negative_prompt_embeds="negative",
        link_embeds={"cause": {"positive": "cause_pos", "negative": "cause_neg"}},
        generator=None,
        steps=1,
        num_frames=1,
        guidance_scale=1.0,
        height=1,
        width=1,
        alpha=0.5,
        timestep_window=(0, 0),
        output_type="latent",
    )

    assert result.value == 7
    assert calls == [("apply", 10, 2.0), ("scheduler_step", 7)]


def test_run_steered_pipeline_guides_minimal_pair_predictions(monkeypatch):
    module = load_runner_module()
    calls = []

    class FakeTensor:
        dtype = "fake_dtype"
        shape = (1, 1, 1, 1, 1)

        def __init__(self, value):
            self.value = value

        def chunk(self, count):
            assert count == 2
            return FakeTensor(self.value), FakeTensor(self.value + 10)

        def __sub__(self, other):
            return FakeTensor(self.value - other.value)

        def __add__(self, other):
            return FakeTensor(self.value + other.value)

        def __mul__(self, scalar):
            return FakeTensor(self.value * scalar)

        def __rmul__(self, scalar):
            return self.__mul__(scalar)

        def permute(self, *args):
            return self

        def reshape(self, *args):
            return self

        def __getitem__(self, key):
            return self

    class FakeTorch:
        @staticmethod
        def cat(values):
            return FakeTensor(sum(value.value for value in values))

        @staticmethod
        def no_grad():
            class Context:
                def __enter__(self):
                    return None

                def __exit__(self, exc_type, exc, tb):
                    return False

            return Context()

    class FakeUnet:
        config = SimpleNamespace(in_channels=1)

        def __call__(self, latent_model_input, timestep, encoder_hidden_states=None, **kwargs):
            values = {1: 10, 11: 3, 13: 1}
            return (FakeTensor(values[encoder_hidden_states.value]),)

    class FakeScheduler:
        order = 1
        timesteps = [100]

        def set_timesteps(self, steps, device=None):
            return None

        def scale_model_input(self, latent_model_input, timestep):
            return latent_model_input

        def step(self, noise_pred, timestep, latents, **kwargs):
            return SimpleNamespace(prev_sample=noise_pred)

    class FakePipe:
        unet = FakeUnet()
        scheduler = FakeScheduler()
        vae_scale_factor = 1
        _execution_device = "cpu"

        def prepare_latents(self, *args, **kwargs):
            return FakeTensor(1)

        def prepare_extra_step_kwargs(self, *args, **kwargs):
            return {}

        def progress_bar(self, total):
            class Progress:
                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc, tb):
                    return False

                def update(self):
                    return None

            return Progress()

        def maybe_free_model_hooks(self):
            return None

    def fake_apply(main_residual, link_predictions, row, *, step_index, alpha, timestep_window):
        calls.append(
            (
                main_residual.value,
                link_predictions["cause"]["positive"].value,
                link_predictions["cause"]["negative"].value,
            )
        )
        return main_residual

    monkeypatch.setattr(module, "apply_steering_residual", fake_apply)

    module.run_steered_pipeline(
        FakePipe(),
        FakeTorch,
        row={"steering": {"links": ["cause"]}},
        prompt_embeds=FakeTensor(1),
        negative_prompt_embeds=FakeTensor(0),
        link_embeds={
            "cause": {
                "positive": FakeTensor(11),
                "negative": FakeTensor(13),
            }
        },
        generator=None,
        steps=1,
        num_frames=1,
        guidance_scale=2.0,
        height=1,
        width=1,
        alpha=0.5,
        timestep_window=(0, 0),
        output_type="latent",
    )

    assert calls == [(30.0, 2.0, 0.0)]


def test_run_steered_pipeline_decodes_under_no_grad():
    module = load_runner_module()

    class FakeTensor:
        dtype = "fake_dtype"
        shape = (1, 1, 1, 1, 1)

        def __init__(self, value):
            self.value = value

        def permute(self, *args):
            return self

        def reshape(self, *args):
            return self

        def __getitem__(self, key):
            return self

    class FakeTorch:
        active = False

        @classmethod
        def no_grad(cls):
            class Context:
                def __enter__(self):
                    cls.active = True

                def __exit__(self, exc_type, exc, tb):
                    cls.active = False
                    return False

            return Context()

    class FakeUnet:
        config = SimpleNamespace(in_channels=1)

        def __call__(self, *args, **kwargs):
            return (FakeTensor(10),)

    class FakeScheduler:
        order = 1
        timesteps = [100]

        def set_timesteps(self, steps, device=None):
            return None

        def scale_model_input(self, latent_model_input, timestep):
            return latent_model_input

        def step(self, noise_pred, timestep, latents, **kwargs):
            return SimpleNamespace(prev_sample=noise_pred)

    class FakeVideoProcessor:
        def postprocess_video(self, video, output_type):
            assert FakeTorch.active is True
            return "video"

    class FakePipe:
        unet = FakeUnet()
        scheduler = FakeScheduler()
        vae_scale_factor = 1
        _execution_device = "cpu"
        video_processor = FakeVideoProcessor()

        def prepare_latents(self, *args, **kwargs):
            return FakeTensor(1)

        def prepare_extra_step_kwargs(self, *args, **kwargs):
            return {}

        def progress_bar(self, total):
            class Progress:
                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc, tb):
                    return False

                def update(self):
                    return None

            return Progress()

        def decode_latents(self, latents):
            assert FakeTorch.active is True
            return latents

        def maybe_free_model_hooks(self):
            return None

    assert module.run_steered_pipeline(
        FakePipe(),
        FakeTorch,
        row={"steering": {"links": []}},
        prompt_embeds=FakeTensor(1),
        negative_prompt_embeds=FakeTensor(0),
        link_embeds={},
        generator=None,
        steps=1,
        num_frames=1,
        guidance_scale=1.0,
        height=1,
        width=1,
        alpha=0.5,
        timestep_window=(0, 0),
        output_type="np",
    ) == "video"


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

    assert averaged["positive"] == pytest.approx([10.0 / 3.0, 4.0])
    assert averaged["negative"] == [0.0, 0.0]


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
    assert embeds["cause"][1]["negative"] == ["neg:cause n2", "pos:cause n2"]


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
        dtype = "fake_dtype"

        def chunk(self, parts):
            return self[:1], self[1:]

        def permute(self, *args):
            return self

        def reshape(self, *args):
            return self

        def __getitem__(self, value):
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
                def __enter__(self):
                    return None

                def __exit__(self, exc_type, exc, tb):
                    return False

            return Context()

    class FakeScheduler:
        timesteps = [0]

        def set_timesteps(self, steps, device):
            self.timesteps = [0]

        def scale_model_input(self, latent_model_input, timestep):
            return latent_model_input

        def step(self, scheduler_noise_pred, timestep, scheduler_latents, **kwargs):
            return SimpleNamespace(prev_sample=FakeLatents([0.0]))

    class FakeLatents(FakeTensor):
        shape = (1, 1, 1, 1, 1)

    class FakePipe:
        _execution_device = "cpu"
        scheduler = FakeScheduler()
        unet = SimpleNamespace(config=SimpleNamespace(in_channels=1))

        def prepare_latents(self, *args):
            return FakeLatents([0.0])

        def prepare_extra_step_kwargs(self, generator, eta):
            return {}

        def progress_bar(self, total):
            class Progress:
                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc, tb):
                    return False

                def update(self):
                    return None

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
