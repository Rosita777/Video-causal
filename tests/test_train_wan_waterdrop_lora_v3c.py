from __future__ import annotations

import ast
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

import numpy  # Keep NumPy resident while temporary dependency stubs are restored.


ROOT = Path(__file__).parents[1]
TRAINER = ROOT / "scripts/train_wan_waterdrop_lora_v3c.py"
LAUNCHER = ROOT / "scripts/run_water_impact_dynamic_sft_v3c_teacher.sh"
PROTOCOL_DOC = ROOT / "docs/water_impact_dynamic_v3c_sigma_weighted_teacher.md"


def fake_dependency_modules() -> dict[str, ModuleType]:
    modules = {name: ModuleType(name) for name in (
        "av",
        "torch",
        "diffusers",
        "diffusers.utils",
        "peft",
        "transformers",
        "causal_lora_activation_gate",
        "target_token_attention_suppression",
    )}
    for name in ("AutoencoderKLWan", "WanPipeline", "WanTransformer3DModel"):
        setattr(modules["diffusers"], name, object)
    modules["diffusers.utils"].convert_state_dict_to_diffusers = lambda value: value
    modules["peft"].LoraConfig = object
    modules["peft"].get_peft_model_state_dict = lambda value: value
    modules["transformers"].AutoTokenizer = object
    modules["torch"].no_grad = lambda: (lambda function: function)
    modules["causal_lora_activation_gate"].CausalLoRAActivationGate = object
    modules["causal_lora_activation_gate"].make_temporally_persistent_gate = (
        lambda value: value
    )
    modules["target_token_attention_suppression"].find_token_mask = lambda *args: None
    return modules


def load_module():
    scripts = str(TRAINER.parent)
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    spec = importlib.util.spec_from_file_location(
        "train_wan_waterdrop_lora_v3c_test", TRAINER
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    with mock.patch.dict(sys.modules, fake_dependency_modules()):
        spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SigmaWeightedTeacherTest(unittest.TestCase):
    def test_erase_combined_loss_is_flow_plus_eight_sigma_teacher(self) -> None:
        module = load_module()

        class Scalar:
            def __init__(self, value: float) -> None:
                self.value = value

            def __add__(self, other):
                return Scalar(self.value + other.value)

            def __mul__(self, other: float):
                return Scalar(self.value * other)

            __rmul__ = __mul__

            def float(self):
                return self

            def detach(self):
                return self

        teacher_loss = Scalar(3.0)
        module.torch.nn = SimpleNamespace(
            functional=SimpleNamespace(mse_loss=lambda prediction, teacher: teacher_loss)
        )
        loss, combined, effective_weight = (
            module.combine_sigma_weighted_target_prompt_teacher_loss(
                Scalar(5.0),
                Scalar(0.0),
                Scalar(0.0),
                sigma=0.25,
                base_weight=4.0,
            )
        )

        self.assertIs(loss, teacher_loss)
        self.assertAlmostEqual(effective_weight, 2.0)  # 8 * sigma
        self.assertAlmostEqual(combined.value, 11.0)  # 5 + (8 * .25) * 3

    def test_metrics_use_per_observation_eight_sigma_gradient_ratio(self) -> None:
        module = load_module()

        metrics = module.target_prompt_scale_metrics(
            [0.01, 0.04], [0.25, 0.75], 4.0
        )

        # Effective weights are [2, 6], so g_i values are [0.2, 1.2].
        self.assertAlmostEqual(metrics["mean_sigma"], 0.5)
        self.assertAlmostEqual(metrics["mean_effective_teacher_weight"], 4.0)
        self.assertAlmostEqual(metrics["mean_weighted_loss_ratio"], 0.13)
        self.assertAlmostEqual(metrics["mean_weighted_output_grad_ratio"], 0.7)
        self.assertAlmostEqual(metrics["median_weighted_output_grad_ratio"], 0.7)

    def test_scale_sanity_writes_registered_formula_and_enforces_max(self) -> None:
        module = load_module()
        passing = [
            {
                "global_step": 2 * index + 1,
                "scene_id": f"scene_{index}",
                "flow_loss": 1.0,
                "target_prompt_teacher_loss": 0.01,
                "raw_loss_ratio": 0.01,
                "sigma": 0.5,
                "effective_teacher_weight": 4.0,
                # Deliberately false: the writer must recompute from raw ratio + sigma.
                "weighted_loss_ratio": 999.0,
                "weighted_output_gradient_norm_ratio": 999.0,
            }
            for index in range(16)
        ]
        with tempfile.TemporaryDirectory() as directory:
            payload = module.write_target_prompt_scale_sanity(
                Path(directory),
                passing,
                base_weight=4.0,
                mean_min=0.2,
                mean_max=0.5,
                single_max=1.0,
                calibration_id="v3c_two_sigma_mean_one_preregistered_v1",
                run_registration_sha256="a" * 64,
            )

        self.assertTrue(payload["passed"])
        self.assertEqual(
            payload["protocol"],
            "water_impact_dynamic_v3c_sigma_weighted_scale_sanity_v1",
        )
        self.assertEqual(payload["weight_schedule"], "2*sigma")
        self.assertAlmostEqual(payload["mean_weighted_output_grad_ratio"], 0.4)
        self.assertAlmostEqual(payload["observations"][0]["weighted_loss_ratio"], 0.04)
        self.assertAlmostEqual(
            payload["observations"][0]["weighted_output_gradient_norm_ratio"],
            0.4,
        )

        failing = [dict(row) for row in passing]
        failing[0]["sigma"] = 1.0
        failing[0]["raw_loss_ratio"] = 0.04  # g_0 = 8 * 1 * 0.2 = 1.6
        with tempfile.TemporaryDirectory() as directory:
            failed = module.write_target_prompt_scale_sanity(
                Path(directory),
                failing,
                base_weight=4.0,
                mean_min=0.2,
                mean_max=0.5,
                single_max=1.0,
                calibration_id="v3c_two_sigma_mean_one_preregistered_v1",
                run_registration_sha256="a" * 64,
            )
        self.assertFalse(failed["passed"])
        self.assertAlmostEqual(
            failed["max_weighted_output_gradient_norm_ratio"], 1.6
        )

    def test_metrics_reject_misaligned_or_invalid_sigma(self) -> None:
        module = load_module()
        with self.assertRaisesRegex(ValueError, "one sigma per ratio"):
            module.target_prompt_scale_metrics([0.1], [], 4.0)
        with self.assertRaisesRegex(ValueError, "finite and valid"):
            module.target_prompt_scale_metrics([0.1], [1.1], 4.0)

    def test_cli_exposes_only_the_sigma_weighted_target_prompt_objective(self) -> None:
        module = load_module()
        common = [
            "trainer",
            "--manifest", "manifest.csv",
            "--model", "model",
            "--cache-dir", "cache",
            "--output-dir", "output",
            "--role", "all",
            "--balanced-roles",
            "--target-prompt-cache-dir", "teacher",
            "--target-prompt-cache-sha256", "a" * 64,
            "--target-prompt-calibration-id", "v3c_two_sigma_mean_one_preregistered_v1",
            "--target-prompt-teacher-weight", "4",
        ]
        with mock.patch.object(
            sys,
            "argv",
            common + ["--objective", "target_prompt_teacher_sigma_weighted"],
        ):
            args = module.parse_args()
        self.assertEqual(args.objective, "target_prompt_teacher_sigma_weighted")
        self.assertEqual(args.target_prompt_teacher_weight, 4.0)

        with mock.patch.object(
            sys, "argv", common + ["--objective", "target_prompt_teacher"]
        ), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                module.parse_args()

        for forbidden in ("--rebuild-cache", "--cache-only"):
            with self.subTest(forbidden=forbidden), mock.patch.object(
                sys,
                "argv",
                common
                + ["--objective", "target_prompt_teacher_sigma_weighted", forbidden],
            ), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    module.parse_args()

    def test_missing_v3c_cache_fails_without_entering_build_or_train(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as directory:
            cache_dir = Path(directory) / "frozen-cache"
            cache_dir.mkdir()
            args = SimpleNamespace(
                dry_run=False,
                role="all",
                objective="target_prompt_teacher_sigma_weighted",
                cache_dir=cache_dir,
            )
            rows = [{"scene_id": "missing_scene"}]
            with mock.patch.object(module, "parse_args", return_value=args), mock.patch.object(
                module, "load_rows", return_value=rows
            ), mock.patch.object(module, "build_cache") as build, mock.patch.object(
                module, "train"
            ) as train, contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(FileNotFoundError, "will not build"):
                    module.main()
            build.assert_not_called()
            train.assert_not_called()
            self.assertEqual(list(cache_dir.iterdir()), [])

    def test_transformer_inventory_rejects_byte_and_index_tampering(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model"
            transformer = model / "transformer"
            transformer.mkdir(parents=True)
            (transformer / "config.json").write_bytes(b"config-v1")
            first = transformer / "diffusion_pytorch_model-00001-of-00002.safetensors"
            second = transformer / "diffusion_pytorch_model-00002-of-00002.safetensors"
            first.write_bytes(b"first-shard")
            second.write_bytes(b"second-shard")
            index = transformer / "diffusion_pytorch_model.safetensors.index.json"
            index.write_text(
                json.dumps(
                    {
                        "weight_map": {
                            "layer.0": first.name,
                            "layer.1": second.name,
                        }
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            digest, records = module.compute_transformer_inventory(model)
            expected = tuple(
                (record["path"], record["size"], record["sha256"])
                for record in records
            )
            payload = module.validate_frozen_transformer_inventory(
                model,
                expected_inventory=expected,
                expected_sha256=digest,
            )
            self.assertEqual(payload["transformer_inventory_sha256"], digest)
            self.assertEqual(
                [record["path"] for record in records],
                sorted(record["path"] for record in records),
            )

            first.write_bytes(b"FIRST-shard")  # same byte count, different payload
            with self.assertRaisesRegex(ValueError, "file inventory mismatch"):
                module.validate_frozen_transformer_inventory(
                    model,
                    expected_inventory=expected,
                    expected_sha256=digest,
                )
            first.write_bytes(b"first-shard")
            index.write_text(
                json.dumps(
                    {"weight_map": {"layer.0": "unregistered.safetensors"}}
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "disagrees with its index"):
                module.compute_transformer_inventory(model)

    def test_launcher_independently_recomputes_and_rejects_model_tampering(self) -> None:
        module = load_module()
        launcher = LAUNCHER.read_text(encoding="utf-8")
        marker = (
            'verify_transformer_inventory() {\n  "$PYTHON" - "$MODEL" '
            '"$EXPECTED_TRANSFORMER_INVENTORY_SHA256" <<\'PY\'\n'
        )
        start = launcher.index(marker) + len(marker)
        verifier = launcher[start : launcher.index("\nPY\n}", start)]

        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model"
            transformer = model / "transformer"
            transformer.mkdir(parents=True)
            (transformer / "config.json").write_bytes(b"config")
            shard = transformer / "one.safetensors"
            shard.write_bytes(b"weights")
            index = transformer / "diffusion_pytorch_model.safetensors.index.json"
            index.write_text(
                json.dumps({"weight_map": {"layer": shard.name}}),
                encoding="utf-8",
            )
            digest, records = module.compute_transformer_inventory(model)
            records_start = verifier.index("expected_records = [")
            records_end = verifier.index("\ntransformer = model", records_start)
            fixture_verifier = (
                verifier[:records_start]
                + f"expected_records = {records!r}"
                + verifier[records_end:]
            )

            def run() -> None:
                with mock.patch.object(
                    sys, "argv", ["verify", str(model), digest]
                ), contextlib.redirect_stdout(io.StringIO()):
                    exec(compile(fixture_verifier, "<launcher-verifier>", "exec"), {})

            run()
            shard.write_bytes(b"WEIGHTS")
            with self.assertRaisesRegex(SystemExit, "file inventory mismatch"):
                run()
            shard.write_bytes(b"weights")
            index.write_text(
                json.dumps({"weight_map": {"layer": "missing.safetensors"}}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(SystemExit, "disagrees with index"):
                run()

    def test_checkpoint_write_is_guarded_by_completed_sanity(self) -> None:
        module = load_module()
        self.assertFalse(module.checkpoint_allowed(True, None))
        self.assertFalse(module.checkpoint_allowed(True, {"passed": False}))
        self.assertTrue(module.checkpoint_allowed(True, {"passed": True}))
        self.assertTrue(module.checkpoint_allowed(False, None))

        source = TRAINER.read_text(encoding="utf-8")
        train_source = source[source.index("def train(") :]
        barrier = train_source.index("v3c scale sanity has not passed")
        save = train_source.index("save_lora(")

        self.assertLess(barrier, save)
        self.assertIn("checkpoint_allowed(", train_source[:save])
        self.assertIn("continue", train_source[barrier:save])
        self.assertIn(
            '!= run_registration.get("expected_sample_order_sha256")',
            train_source[:save],
        )
        self.assertIn(
            '!= run_registration.get("expected_noise_sigma_rng_final_sha256")',
            train_source[:save],
        )

    def test_launcher_and_protocol_bind_the_same_frozen_interface(self) -> None:
        launcher = LAUNCHER.read_text(encoding="utf-8")
        protocol = PROTOCOL_DOC.read_text(encoding="utf-8")
        protocol_id = (
            "water_impact_dynamic_v3c_sigma_weighted_target_prompt_teacher_v1"
        )

        self.assertIn(protocol_id, launcher)
        self.assertIn(protocol_id, protocol)
        self.assertIn("scripts/train_wan_waterdrop_lora_v3c.py", launcher)
        self.assertIn("--objective target_prompt_teacher_sigma_weighted", launcher)
        self.assertIn('"save_every": 25', launcher)
        self.assertIn(
            "outputs/water_impact_dynamic_v3c/adapter_target_prompt_teacher_sigma2_scale4_v1",
            launcher,
        )
        self.assertIn('if ! mkdir "$OUTPUT_DIR"', launcher)
        self.assertIn("verify_transformer_inventory()", launcher)
        self.assertIn("verify_transformer_inventory\n", launcher)
        self.assertIn("sha256_ordered_name_nul_bytes_lf_v1", launcher)
        self.assertIn(
            "fe0c20ba99318c4b6d28d153b839888bbcf8a7a856796979566c305788dc18ac",
            launcher,
        )
        self.assertIn(
            '"expected_noise_sigma_rng_initial_sha256": "49b65850', launcher
        )
        self.assertIn(
            '"expected_noise_sigma_rng_final_sha256": "79ff6c9a', launcher
        )

    def test_preserve_branch_and_teacher_forward_are_identical_to_v3b(self) -> None:
        base = ROOT / "scripts/train_wan_waterdrop_lora.py"

        def function(path: Path, name: str) -> tuple[str, ast.FunctionDef]:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source)
            node = next(
                item
                for item in tree.body
                if isinstance(item, ast.FunctionDef) and item.name == name
            )
            return source, node

        base_source, base_forward = function(base, "forward_target_prompt_teacher_pair")
        v3c_source, v3c_forward = function(
            TRAINER, "forward_target_prompt_teacher_pair"
        )
        self.assertEqual(
            ast.get_source_segment(base_source, base_forward),
            ast.get_source_segment(v3c_source, v3c_forward),
        )

        _, base_train = function(base, "train")
        _, v3c_train = function(TRAINER, "train")

        def preserve_branch(node: ast.FunctionDef) -> ast.If:
            for item in ast.walk(node):
                if not isinstance(item, ast.If) or not isinstance(item.test, ast.Name):
                    continue
                if item.test.id != "is_preserve":
                    continue
                assigned = {
                    target.id
                    for statement in item.body
                    if isinstance(statement, ast.Assign)
                    for target in statement.targets
                    if isinstance(target, ast.Name)
                }
                if {"preserve_loss", "combined_loss"} <= assigned:
                    return item
            raise AssertionError("preserve loss branch not found")

        self.assertEqual(
            ast.dump(
                ast.Module(body=preserve_branch(base_train).body, type_ignores=[]),
                include_attributes=False,
            ),
            ast.dump(
                ast.Module(body=preserve_branch(v3c_train).body, type_ignores=[]),
                include_attributes=False,
            ),
        )

    def test_run_registration_happy_path_and_tamper_rejection(self) -> None:
        module = load_module()
        manifest_sha = (
            "3d4d8cbf9244b1575357f0ac74380cd7cb6265df4d1a85bf450de4cac120aee4"
        )
        cache_sha = (
            "4654ee67b8937d248963839b744e59f4f535f8be8b8eee421aef264fb3cd4d65"
        )
        teacher_sha = (
            "6cf7ba0112d8df0e0a5253a7a943411fcfd85c56fc6df580446776b49d1ac9a9"
        )
        output = Path(
            "outputs/water_impact_dynamic_v3c/"
            "adapter_target_prompt_teacher_sigma2_scale4_v1"
        )
        args = SimpleNamespace(
            manifest=Path(
                "data/water_impact_dynamic_v1/train_dynamic_sft_preserve_v2.csv"
            ),
            cache_dir=Path(
                "outputs/water_impact_dynamic_v1/cache_dynamic_sft_preserve_v2"
            ),
            target_prompt_cache_dir=Path(
                "outputs/water_impact_dynamic_v3b/teacher_prompt_cache_v1"
            ),
            target_prompt_cache_sha256=teacher_sha,
            output_dir=output,
            target_prompt_calibration_id=(
                "v3c_two_sigma_mean_one_preregistered_v1"
            ),
            target_prompt_teacher_weight=4.0,
            target_prompt_sanity_min_output_grad_ratio=0.2,
            target_prompt_sanity_max_output_grad_ratio=0.5,
            target_prompt_sanity_max_single_output_grad_ratio=1.0,
            model=Path("models/Wan2.1-T2V-1.3B-Diffusers"),
            height=480,
            width=832,
            num_frames=49,
            max_steps=200,
            learning_rate=5e-5,
            rank=16,
            alpha=16,
            grad_accum=1,
            save_every=25,
            seed=26000,
            device="cuda",
            role="all",
            objective="target_prompt_teacher_sigma_weighted",
            balanced_roles=True,
            preserve_weight=4.0,
        )
        reference_hashes = {
            "outputs/water_impact_dynamic_v3b/adapter_target_prompt_teacher_scale4_v1/"
            "checkpoint-000200/pytorch_lora_weights.safetensors": (
                "d3fecf26b7f1ca6c4a8f46c86850a47a7ec5a62762d0e0aa15c49363040875d3"
            ),
            "outputs/water_impact_dynamic_v3b/adapter_target_prompt_teacher_scale4_v1/"
            "checkpoint-000200/training_state.json": (
                "0f9aa26e825f4f6f497b1312c507b685c054bc319f2f9f538e45eeb7a7908bea"
            ),
            "outputs/water_impact_dynamic_v3b/adapter_target_prompt_teacher_scale4_v1/"
            "run_registration.json": (
                "53f0a7c472ba02a38b90b55651f378e5feda0bcd709f86786702de163b3a87f4"
            ),
            "outputs/water_impact_dynamic_v3b/adapter_target_prompt_teacher_scale4_v1/"
            "target_prompt_scale_sanity.json": (
                "26fb8b1ff9e0d446fd186765ba1ff9a9d1a085d75d230cd0a419509ea00bbb12"
            ),
        }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            code_paths = {
                "trainer": root / "scripts/train_wan_waterdrop_lora_v3c.py",
                "launcher": root / "scripts/run_water_impact_dynamic_sft_v3c_teacher.sh",
                "protocol_doc": (
                    root / "docs/water_impact_dynamic_v3c_sigma_weighted_teacher.md"
                ),
            }
            for path in code_paths.values():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(f"fixture:{path.name}\n", encoding="utf-8")
            for relative in reference_hashes:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"frozen-v3b-fixture")
            (root / output).mkdir(parents=True)

            registration = module.expected_target_prompt_run_registration(
                args,
                manifest_sha256=manifest_sha,
                base_cache_sha256=cache_sha,
            )
            for prefix, path in code_paths.items():
                registration[f"{prefix}_path"] = str(path.relative_to(root))
                registration[f"{prefix}_sha256"] = sha256(path)
            registration["v3b_reference_artifacts"] = [
                {"path": path, "sha256": digest}
                for path, digest in reference_hashes.items()
            ]
            registration_path = root / output / "run_registration.json"

            def fake_hash(path: Path) -> str:
                resolved = Path(path).resolve()
                try:
                    relative = str(resolved.relative_to(root))
                except ValueError:
                    relative = ""
                if relative in reference_hashes:
                    return reference_hashes[relative]
                return sha256(resolved)

            def write(payload: dict[str, object]) -> None:
                registration_path.write_text(
                    json.dumps(payload), encoding="utf-8"
                )

            previous = Path.cwd()
            original_module_file = module.__file__
            try:
                os.chdir(root)
                module.__file__ = str(code_paths["trainer"])
                with mock.patch.object(module, "file_sha256", side_effect=fake_hash):
                    write(registration)
                    path, _, payload = module.validate_target_prompt_run_registration(
                        args,
                        manifest_sha256=manifest_sha,
                        base_cache_sha256=cache_sha,
                    )
                    self.assertEqual(path, output / "run_registration.json")
                    self.assertEqual(payload["target_prompt_teacher_schedule"], "2*sigma")

                    mutations = {
                        "schedule": ("target_prompt_teacher_schedule", "constant"),
                        "base weight": ("target_prompt_teacher_base_weight", 3.0),
                        "noise RNG": (
                            "expected_noise_sigma_rng_final_sha256",
                            "0" * 64,
                        ),
                        "sample order": ("expected_sample_order_sha256", "0" * 64),
                        "transformer inventory": (
                            "transformer_inventory_sha256",
                            "0" * 64,
                        ),
                        "code artifact": ("trainer_sha256", "0" * 64),
                    }
                    for label, (field, value) in mutations.items():
                        with self.subTest(label=label):
                            tampered = json.loads(json.dumps(registration))
                            tampered[field] = value
                            write(tampered)
                            with self.assertRaises(ValueError):
                                module.validate_target_prompt_run_registration(
                                    args,
                                    manifest_sha256=manifest_sha,
                                    base_cache_sha256=cache_sha,
                                )

                    write(registration)
                    wrong_output_args = SimpleNamespace(**vars(args))
                    wrong_output_args.output_dir = Path("outputs/wrong")
                    with self.assertRaisesRegex(ValueError, "data/output inputs"):
                        module.validate_target_prompt_run_registration(
                            wrong_output_args,
                            manifest_sha256=manifest_sha,
                            base_cache_sha256=cache_sha,
                        )
            finally:
                module.__file__ = original_module_file
                os.chdir(previous)

    def test_v3b_registration_bound_files_are_byte_unchanged(self) -> None:
        expected = {
            ROOT / "scripts/train_wan_waterdrop_lora.py": (
                "6912d2a2adb4ed659ae2bc95b6882106366b707949ed56be71913f462cfec087"
            ),
            ROOT / "scripts/run_water_impact_dynamic_sft_v3b_teacher.sh": (
                "d7879c8885f401a3d9972a489d53b48ea68466b6ca86e8922c9b36f458bbc66f"
            ),
            ROOT / "docs/water_impact_dynamic_v3b_target_prompt_teacher.md": (
                "ac96d88327984f91d8c0d1b2075eaa544251382b01ec11075ea16b2d9022422a"
            ),
        }
        self.assertEqual({path: sha256(path) for path in expected}, expected)


if __name__ == "__main__":
    unittest.main()
