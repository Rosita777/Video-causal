from pathlib import Path
import importlib.util
import json

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROBE_PATH = PROJECT_ROOT / "scripts" / "adapters" / "run_zeroscope_attention_probe.py"


def load_probe_module():
    spec = importlib.util.spec_from_file_location("run_zeroscope_attention_probe", PROBE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_normalize_token_text_removes_clip_boundaries():
    module = load_probe_module()

    assert module.normalize_token_text("</w>") == ""
    assert module.normalize_token_text("pebble</w>") == "pebble"
    assert module.normalize_token_text("Ġsoccer") == "soccer"
    assert module.normalize_token_text("black</w>") == "black"


def test_find_token_indices_matches_multiword_target():
    module = load_probe_module()
    tokens = [
        "<|startoftext|>",
        "a</w>",
        "black</w>",
        "hockey</w>",
        "puck</w>",
        "hits</w>",
        "mirror</w>",
        "<|endoftext|>",
    ]

    assert module.find_token_indices(tokens, "black hockey puck") == [2, 3, 4]


def test_find_token_indices_matches_bpe_fragments_and_punctuation():
    module = load_probe_module()
    tokens = [
        "<|startoftext|>",
        "a</w>",
        "star</w>",
        "-</w>",
        "shaped</w>",
        "crack</w>",
        "ripp",
        "les</w>",
        "<|endoftext|>",
    ]

    assert module.find_token_indices(tokens, "star-shaped crack") == [2, 4, 5]
    assert module.find_token_indices(tokens, "ripples") == [6, 7]


def test_find_token_indices_raises_for_missing_target():
    module = load_probe_module()

    with pytest.raises(ValueError, match="could not find token span"):
        module.find_token_indices(["a</w>", "pond</w>"], "soccer ball")


def test_comparison_token_indices_excludes_special_and_selected_tokens():
    module = load_probe_module()
    tokens = ["<|startoftext|>", "a</w>", "small</w>", "pebble</w>", "falls</w>", "<|endoftext|>"]

    comparison = module.comparison_token_indices(
        tokens,
        selected_indices={3},
        count=2,
    )

    assert comparison == [1, 2]


def test_attention_summary_recorder_records_token_mass():
    module = load_probe_module()
    recorder = module.AttentionSummaryRecorder(
        target_indices=[1],
        footprint_indices=[3],
        comparison_indices=[2],
    )
    probs = [
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.2, 0.3, 0.1, 0.4],
        ],
        [
            [0.3, 0.1, 0.4, 0.2],
            [0.1, 0.2, 0.5, 0.2],
        ],
    ]

    recorder.record(
        module_name="mid_block.attn2",
        step_index=4,
        attention_probs=probs,
        query_tokens=2,
        key_tokens=4,
    )

    assert len(recorder.records) == 1
    record = recorder.records[0]
    assert record["module_name"] == "mid_block.attn2"
    assert record["step_index"] == 4
    assert record["target_mass"] == pytest.approx(0.2)
    assert record["footprint_mass"] == pytest.approx(0.3)
    assert record["comparison_mass"] == pytest.approx(0.325)
    assert record["all_text_mass"] == pytest.approx(0.25)


def test_attention_summary_recorder_uses_tensor_path_without_tolist(monkeypatch):
    torch = pytest.importorskip("torch")
    module = load_probe_module()
    recorder = module.AttentionSummaryRecorder(
        target_indices=[1],
        footprint_indices=[2],
        comparison_indices=[0],
    )
    probs = torch.tensor(
        [
            [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.1],
            ]
        ]
    )

    def fail_tolist(self):
        raise AssertionError("tolist should not be used for torch tensors")

    monkeypatch.setattr(torch.Tensor, "tolist", fail_tolist)

    recorder.record(
        module_name="attn2",
        step_index=0,
        attention_probs=probs,
        query_tokens=2,
        key_tokens=3,
    )

    assert recorder.records[0]["target_mass"] == pytest.approx(0.35)


def test_reweight_attention_columns_suppresses_and_renormalizes():
    torch = pytest.importorskip("torch")
    module = load_probe_module()
    probs = torch.tensor(
        [
            [
                [0.2, 0.3, 0.5],
                [0.1, 0.4, 0.5],
            ]
        ]
    )

    reweighted = module.reweight_attention_columns(
        probs,
        selected_indices=[1],
        scale=0.0,
    )

    assert reweighted[:, :, 1].max().item() == pytest.approx(0.0)
    assert torch.allclose(reweighted.sum(dim=-1), torch.ones(1, 2))
    assert reweighted[0, 0, 0].item() == pytest.approx(0.2 / 0.7)
    assert reweighted[0, 0, 2].item() == pytest.approx(0.5 / 0.7)


def test_recording_attention_processor_records_cross_attention():
    torch = pytest.importorskip("torch")
    module = load_probe_module()

    class FakeAttention:
        spatial_norm = None
        group_norm = None
        norm_cross = False
        residual_connection = False
        rescale_output_factor = 1.0
        heads = 1

        def __init__(self):
            self.to_q = torch.nn.Linear(2, 2, bias=False)
            self.to_k = torch.nn.Linear(2, 2, bias=False)
            self.to_v = torch.nn.Linear(2, 2, bias=False)
            self.to_out = torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Identity()])
            self._last_probs = None
            with torch.no_grad():
                self.to_q.weight.copy_(torch.eye(2))
                self.to_k.weight.copy_(torch.eye(2))
                self.to_v.weight.copy_(torch.eye(2))

        def prepare_attention_mask(self, attention_mask, sequence_length, batch_size):
            return attention_mask

        def norm_encoder_hidden_states(self, encoder_hidden_states):
            return encoder_hidden_states

        def head_to_batch_dim(self, tensor):
            return tensor

        def batch_to_head_dim(self, tensor):
            return tensor

        def get_attention_scores(self, query, key, attention_mask):
            scores = torch.bmm(query, key.transpose(1, 2))
            probs = torch.softmax(scores, dim=-1)
            self._last_probs = probs.detach()
            return probs

    attention = FakeAttention()
    recorder = module.AttentionSummaryRecorder(
        target_indices=[0],
        footprint_indices=[1],
        comparison_indices=[2],
    )
    processor = module.RecordingAttnProcessor(
        module_name="down_blocks.0.attentions.0.transformer_blocks.0.attn2.processor",
        recorder=recorder,
        step_getter=lambda: 7,
    )

    hidden_states = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    encoder_hidden_states = torch.tensor([[[2.0, 0.0], [0.0, 2.0], [1.0, 1.0]]])

    output = processor(
        attention,
        hidden_states,
        encoder_hidden_states=encoder_hidden_states,
    )

    assert output.shape == hidden_states.shape
    assert len(recorder.records) == 1
    record = recorder.records[0]
    assert record["step_index"] == 7
    assert record["query_tokens"] == 2
    assert record["key_tokens"] == 3
    assert record["target_mass"] == pytest.approx(
        attention._last_probs[:, :, 0].mean().item()
    )


def test_recording_attention_processor_records_cfg_text_conditioned_half_only():
    torch = pytest.importorskip("torch")
    module = load_probe_module()

    class FakeAttention:
        spatial_norm = None
        group_norm = None
        norm_cross = False
        residual_connection = False
        rescale_output_factor = 1.0
        heads = 1

        def __init__(self):
            self.to_q = torch.nn.Linear(2, 2, bias=False)
            self.to_k = torch.nn.Linear(2, 2, bias=False)
            self.to_v = torch.nn.Linear(2, 2, bias=False)
            self.to_out = torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Identity()])
            self._last_probs = None
            with torch.no_grad():
                self.to_q.weight.copy_(torch.eye(2))
                self.to_k.weight.copy_(torch.eye(2))
                self.to_v.weight.copy_(torch.eye(2))

        def prepare_attention_mask(self, attention_mask, sequence_length, batch_size):
            return attention_mask

        def norm_encoder_hidden_states(self, encoder_hidden_states):
            return encoder_hidden_states

        def head_to_batch_dim(self, tensor):
            return tensor

        def batch_to_head_dim(self, tensor):
            return tensor

        def get_attention_scores(self, query, key, attention_mask):
            scores = torch.bmm(query, key.transpose(1, 2))
            probs = torch.softmax(scores, dim=-1)
            self._last_probs = probs.detach()
            return probs

    attention = FakeAttention()
    recorder = module.AttentionSummaryRecorder(
        target_indices=[0],
        footprint_indices=[1],
        comparison_indices=[2],
    )
    processor = module.RecordingAttnProcessor(
        module_name="attn2.processor",
        recorder=recorder,
        step_getter=lambda: 0,
    )

    hidden_states = torch.tensor(
        [
            [[1.0, 0.0]],
            [[1.0, 0.0]],
        ]
    )
    encoder_hidden_states = torch.tensor(
        [
            [[0.0, 2.0], [2.0, 0.0], [1.0, 1.0]],
            [[2.0, 0.0], [0.0, 2.0], [1.0, 1.0]],
        ]
    )

    processor(attention, hidden_states, encoder_hidden_states=encoder_hidden_states)

    record = recorder.records[0]
    expected_text_conditioned = attention._last_probs[1:, :, 0].mean().item()
    all_batches = attention._last_probs[:, :, 0].mean().item()
    assert expected_text_conditioned != pytest.approx(all_batches)
    assert record["target_mass"] == pytest.approx(expected_text_conditioned)


def test_recording_attention_processor_reweights_text_conditioned_half_only():
    torch = pytest.importorskip("torch")
    module = load_probe_module()

    class FakeAttention:
        spatial_norm = None
        group_norm = None
        norm_cross = False
        residual_connection = False
        rescale_output_factor = 1.0
        heads = 1

        def __init__(self):
            self.to_q = torch.nn.Linear(2, 2, bias=False)
            self.to_k = torch.nn.Linear(2, 2, bias=False)
            self.to_v = torch.nn.Linear(2, 2, bias=False)
            self.to_out = torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Identity()])
            self._raw_probs = None
            with torch.no_grad():
                self.to_q.weight.copy_(torch.eye(2))
                self.to_k.weight.copy_(torch.eye(2))
                self.to_v.weight.copy_(torch.eye(2))

        def prepare_attention_mask(self, attention_mask, sequence_length, batch_size):
            return attention_mask

        def norm_encoder_hidden_states(self, encoder_hidden_states):
            return encoder_hidden_states

        def head_to_batch_dim(self, tensor):
            return tensor

        def batch_to_head_dim(self, tensor):
            return tensor

        def get_attention_scores(self, query, key, attention_mask):
            scores = torch.bmm(query, key.transpose(1, 2))
            probs = torch.softmax(scores, dim=-1)
            self._raw_probs = probs.detach()
            return probs

    attention = FakeAttention()
    recorder = module.AttentionSummaryRecorder(
        target_indices=[0],
        footprint_indices=[1],
        comparison_indices=[2],
    )
    processor = module.RecordingAttnProcessor(
        module_name="attn2.processor",
        recorder=recorder,
        step_getter=lambda: 0,
        intervention_indices=[0],
        intervention_scale=0.0,
    )
    hidden_states = torch.tensor(
        [
            [[1.0, 0.0]],
            [[1.0, 0.0]],
        ]
    )
    encoder_hidden_states = torch.tensor(
        [
            [[0.0, 2.0], [2.0, 0.0], [1.0, 1.0]],
            [[2.0, 0.0], [0.0, 2.0], [1.0, 1.0]],
        ]
    )

    output = processor(attention, hidden_states, encoder_hidden_states=encoder_hidden_states)

    uncond_expected = torch.bmm(attention._raw_probs[:1], encoder_hidden_states[:1])
    text_reweighted = module.reweight_attention_columns(
        attention._raw_probs[1:],
        selected_indices=[0],
        scale=0.0,
    )
    text_expected = torch.bmm(text_reweighted, encoder_hidden_states[1:])
    assert torch.allclose(output[:1], uncond_expected)
    assert torch.allclose(output[1:], text_expected)
    assert recorder.records[0]["target_mass"] == pytest.approx(0.0)


def test_dry_run_writes_attention_probe_manifest(tmp_path):
    module = load_probe_module()
    probe_manifest = tmp_path / "probe_manifest.json"
    probe_manifest.write_text(
        json.dumps(
            {
                "probe_name": "unit_probe",
                "items": [
                    {
                        "probe_index": 0,
                        "pair_id": "pond_case",
                        "target_concept": "pebble",
                        "causal_footprint": "circular ripples spread outward",
                        "generation_prompt": (
                            "A realistic close-up video of a still pond. "
                            "pebble causes circular ripples spread outward."
                        ),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "attention_probe"

    assert module.main(
        [
            "--probe-manifest",
            str(probe_manifest),
            "--output-dir",
            str(output_dir),
            "--dry-run",
            "--limit-items",
            "1",
        ]
    ) == 0

    manifest = json.loads((output_dir / "generation_manifest.json").read_text())
    assert manifest["probe_name"] == "zeroscope_attention_dependency_probe"
    assert manifest["dry_run"] is True
    item = manifest["items"][0]
    assert item["target_indices"]
    assert item["footprint_indices"]
    assert item["attention_trace_path"].endswith("attention_trace.jsonl")
    assert item["attention_summary_path"].endswith("attention_summary.csv")


def test_dry_run_expands_b2_mask_conditions(tmp_path):
    module = load_probe_module()
    probe_manifest = tmp_path / "probe_manifest.json"
    probe_manifest.write_text(
        json.dumps(
            {
                "probe_name": "unit_probe",
                "items": [
                    {
                        "probe_index": 2,
                        "pair_id": "soccer_case",
                        "target_concept": "soccer ball",
                        "causal_footprint": "net stretches backward",
                        "generation_prompt": (
                            "A close-up video. soccer ball causes net stretches backward."
                        ),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "b2_probe"

    assert module.main(
        [
            "--probe-manifest",
            str(probe_manifest),
            "--output-dir",
            str(output_dir),
            "--dry-run",
            "--condition",
            "baseline",
            "--condition",
            "chain_mask",
            "--condition",
            "random_token_mask",
            "--mask-scale",
            "0.0",
        ]
    ) == 0

    manifest = json.loads((output_dir / "generation_manifest.json").read_text())
    conditions = [item["condition"] for item in manifest["items"]]
    assert conditions == ["baseline", "chain_mask", "random_token_mask"]
    baseline, chain, random_row = manifest["items"]
    assert baseline["intervention_indices"] == []
    assert chain["intervention_indices"] == (
        chain["target_indices"] + chain["footprint_indices"]
    )
    assert len(random_row["intervention_indices"]) == len(chain["intervention_indices"])
    assert not set(random_row["intervention_indices"]) & set(chain["intervention_indices"])
    assert chain["intervention_scale"] == 0.0


def test_install_recording_processors_replaces_only_cross_attention():
    module = load_probe_module()

    class FakeUNet:
        def __init__(self):
            self.attn_processors = {
                "down_blocks.0.attentions.0.transformer_blocks.0.attn1.processor": object(),
                "down_blocks.0.attentions.0.transformer_blocks.0.attn2.processor": object(),
                "mid_block.attentions.0.transformer_blocks.0.attn2.processor": object(),
            }
            self.received = None

        def set_attn_processor(self, processors):
            self.received = processors
            self.attn_processors = processors

    unet = FakeUNet()
    recorder = module.AttentionSummaryRecorder(
        target_indices=[1],
        footprint_indices=[2],
        comparison_indices=[3],
    )

    installed = module.install_recording_processors(
        unet,
        recorder=recorder,
        step_getter=lambda: 0,
    )

    assert installed == 2
    assert isinstance(
        unet.received["down_blocks.0.attentions.0.transformer_blocks.0.attn2.processor"],
        module.RecordingAttnProcessor,
    )
    assert isinstance(
        unet.received["mid_block.attentions.0.transformer_blocks.0.attn2.processor"],
        module.RecordingAttnProcessor,
    )
    assert not isinstance(
        unet.received["down_blocks.0.attentions.0.transformer_blocks.0.attn1.processor"],
        module.RecordingAttnProcessor,
    )


def test_install_recording_processors_passes_intervention_config():
    module = load_probe_module()

    class FakeUNet:
        def __init__(self):
            self.attn_processors = {
                "mid_block.attentions.0.transformer_blocks.0.attn2.processor": object(),
            }

        def set_attn_processor(self, processors):
            self.attn_processors = processors

    recorder = module.AttentionSummaryRecorder(
        target_indices=[1],
        footprint_indices=[2],
        comparison_indices=[3],
    )
    unet = FakeUNet()

    module.install_recording_processors(
        unet,
        recorder=recorder,
        step_getter=lambda: 0,
        intervention_indices=[4, 5],
        intervention_scale=0.25,
    )

    processor = unet.attn_processors[
        "mid_block.attentions.0.transformer_blocks.0.attn2.processor"
    ]
    assert processor.intervention_indices == [4, 5]
    assert processor.intervention_scale == 0.25
