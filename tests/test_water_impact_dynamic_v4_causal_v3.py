#!/usr/bin/env python3
"""Tests for the public-only v4_dev72_v3 capacity reproducer."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "validate_water_impact_dynamic_v4_causal_capacity_v3.py"
SPEC = importlib.util.spec_from_file_location("v4_dev72_v3_capacity", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
capacity = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(capacity)


class RecordingRng:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[int, ...]]] = []

    def beta(self, alpha, beta, *, size):
        shape = tuple(size)
        self.calls.append(("beta", shape))
        alpha_array = np.asarray(alpha)
        if alpha_array.shape == () and float(alpha_array) == 9.0:
            return np.full(shape, 0.9, dtype=np.float64)
        return np.full(shape, 0.5, dtype=np.float64)

    def random(self, size):
        shape = tuple(size)
        self.calls.append(("random", shape))
        return np.full(shape, 0.75, dtype=np.float64)


class CapacityPlanningTests(unittest.TestCase):
    def test_analytic_models_match_preregistration(self) -> None:
        report = capacity.analytic_capacity_report()
        m0 = report["M0_uniform_Beta_1_1"]
        m1 = report["M1_Jeffreys_Beta_0p5_0p5"]
        expected_m0 = (
            0.0013772508,
            0.0213109327,
            0.0008142951,
            0.0000057043,
            0.0000659546,
            0.0134069232,
        )
        expected_m1 = (
            0.0020423675,
            0.0567120666,
            0.0007782908,
            0.0000022088,
            0.0001283477,
            0.0400662581,
        )
        self.assertEqual(
            tuple(round(value, 10) for value in m0["cell_shortage_probabilities"].values()),
            expected_m0,
        )
        self.assertEqual(
            tuple(round(value, 10) for value in m1["cell_shortage_probabilities"].values()),
            expected_m1,
        )
        self.assertEqual(round(m0["familywise_shortage_probability"], 10), 0.0366161849)
        self.assertEqual(round(m1["familywise_shortage_probability"], 10), 0.0971766186)
        self.assertTrue(m0["passes"] and m1["passes"])

    def test_seed_domains_and_wilson_values_match_preregistration(self) -> None:
        expected = {
            capacity.SEARCH_DOMAIN: (
                "a8e24792910d700ced6dff45d9817be05fe5370e2e102cf6e0363ed5e8244580",
                12169367837932875788,
            ),
            capacity.CONFIRM_RHO010_DOMAIN: (
                "8b6015284b8d55a49246904746fb115437b7a04e818a471942a41cc7f4c2fe8d",
                10043050431846634916,
            ),
            capacity.CONFIRM_RHO020_DOMAIN: (
                "8a85d1c114b988dddb75670fc322a0beee98403c9f61468b48205c0a581d6aad",
                9981614776343169245,
            ),
            capacity.CONFIRM_SHARED_FRAILTY_DOMAIN: (
                "3027d87c835f977849a031eb2a530438c33378d91eba80137aaa88709ec7f1ed",
                3469980067203880824,
            ),
        }
        for domain, (digest, seed) in expected.items():
            record = capacity.seed_record(domain)
            self.assertEqual(record["domain_sha256"], digest)
            self.assertEqual(record["uint64_first_8_bytes_big_endian"], seed)
        self.assertEqual(
            round(capacity.wilson_upper_one_sided_95(28_527, 200_000), 10),
            0.1439260337,
        )
        self.assertEqual(
            round(capacity.wilson_upper_one_sided_95(143_547, 1_000_000), 10),
            0.1441246991,
        )
        self.assertEqual(
            round(capacity.wilson_upper_one_sided_95(264_002, 1_000_000), 10),
            0.2647276898,
        )
        self.assertEqual(
            round(capacity.wilson_upper_one_sided_95(149_245, 1_000_000), 10),
            0.1498320593,
        )

    def test_graph_degrees_and_delete_up_to_two_robustness(self) -> None:
        report = capacity.graph_robustness_report()
        self.assertEqual(report["candidate_count"], 576)
        self.assertEqual(report["R1_delete_up_to_2"], {
            "checked": 301, "expected": 301, "failures": 0
        })
        self.assertEqual(
            report["R3_assignments_x_delete_up_to_2"],
            {
                "variant_assignments": 70,
                "deletion_sets": 529,
                "checked": 37_030,
                "expected": 37_030,
                "failures": 0,
            },
        )
        degrees = report["degrees"]
        self.assertEqual(
            (degrees["G1-D"]["source_degree_min"], degrees["G1-D"]["receiver_degree_min"]),
            (2, 2),
        )
        self.assertEqual(
            (degrees["G1-N"]["source_degree_min"], degrees["G1-N"]["receiver_degree_min"]),
            (7, 7),
        )
        self.assertEqual(degrees["G3-D"]["receiver_degree_histogram"], {3: 32})
        self.assertEqual(degrees["G3-N"]["receiver_degree_histogram"], {6: 8, 7: 24})

    def test_graph_degree_or_edge_tamper_is_rejected(self) -> None:
        tampered = dict(capacity.GRAPHS)
        rows = [list(row) for row in tampered["G1-D"]]
        rows[0].pop()
        tampered["G1-D"] = tuple(tuple(row) for row in rows)
        with self.assertRaisesRegex(ValueError, "shape/edge uniqueness"):
            capacity.graph_robustness_report(tampered)

        # A global receiver relabelling preserves every shape, degree, subset,
        # and deletion-robustness statistic.  It is still not the frozen graph.
        tampered = dict(capacity.GRAPHS)
        for name in ("G1-D", "G1-N"):
            tampered[name] = tuple(
                tuple((receiver + 1) % 24 for receiver in row)
                for row in tampered[name]
            )
        self.assertNotEqual(
            capacity.graph_specification(tampered)["graph_sha256"],
            capacity.graph_specification()["graph_sha256"],
        )
        with self.assertRaisesRegex(ValueError, "frozen exact graph"):
            capacity.graph_robustness_report(tampered)

        tampered = dict(capacity.GRAPHS)
        rows = [list(row) for row in tampered["G3-D"]]
        rows[0][0] = rows[0][1]
        tampered["G3-D"] = tuple(tuple(row) for row in rows)
        with self.assertRaisesRegex(ValueError, "shape/edge uniqueness"):
            capacity.graph_robustness_report(tampered)

    def test_exact_oracles_enforce_head_receiver_anchor_and_variant_constraints(self) -> None:
        g1d = capacity.receiver_masks(capacity.GRAPHS["G1-D"])
        g1n = capacity.receiver_masks(capacity.GRAPHS["G1-N"])
        self.assertTrue(capacity.g1_complete(g1d, g1n))
        self.assertFalse(capacity.g1_complete((1,) * 24, (1,) * 24))
        self.assertFalse(capacity.g1_complete((0,) * 24, g1n))

        self.assertTrue(capacity.g2_complete((True,) * 8, (True,) * 8))
        self.assertFalse(
            capacity.g2_complete((True,) * 3 + (False,) * 5, (True,) * 8)
        )
        self.assertFalse(
            capacity.g2_complete((True,) * 8, (True,) * 3 + (False,) * 5)
        )

        g3d = capacity.receiver_masks(capacity.GRAPHS["G3-D"])
        g3n = capacity.receiver_masks(capacity.GRAPHS["G3-N"])
        self.assertTrue(capacity.fixed_anchor_group_complete(g3d, g3n))
        self.assertFalse(capacity.fixed_anchor_group_complete((1,) * 8, (1,) * 8))

    def test_draw_order_is_exactly_cell_anchor_iteration_edge(self) -> None:
        rng = RecordingRng()
        capacity.draw_eligibility_batch(
            rng, count=3, rho=0.10, shared_frailty=False
        )
        beta_shapes = [shape for name, shape in rng.calls if name == "beta"]
        random_shapes = [shape for name, shape in rng.calls if name == "random"]
        self.assertEqual(
            beta_shapes,
            [(3, 6), (3, 24), (3, 24), (3, 8), (3, 8), (3, 8), (3, 8)],
        )
        expected_random = (
            [(3, 2)] * 24
            + [(3, 7)] * 24
            + [(3, 3)] * 8
            + [(3, 3)] * 8
            + [(3, 12)] * 8
            + [(3, 27)] * 8
        )
        self.assertEqual(random_shapes, expected_random)

    def test_shared_frailty_draws_follow_all_theta_and_precede_uniforms(self) -> None:
        rng = RecordingRng()
        capacity.draw_eligibility_batch(
            rng, count=2, rho=0.10, shared_frailty=True
        )
        first_random = next(index for index, call in enumerate(rng.calls) if call[0] == "random")
        before_random = rng.calls[:first_random]
        self.assertEqual(
            [shape for name, shape in before_random if name == "beta"],
            [
                (2, 6),
                (2, 24), (2, 24), (2, 8), (2, 8), (2, 8), (2, 8),
                (2, 24), (2, 8), (2, 8),
            ],
        )

    def test_compiled_oracle_matches_python_reference(self) -> None:
        with capacity.compiled_oracle() as oracle:
            self.assertTrue(capacity.compiled_oracle_self_test(oracle))

    def test_smoke_monte_carlo_is_deterministic(self) -> None:
        first = capacity.smoke_report(
            iterations=120,
            batch_size=60,
            rho=0.10,
            shared_frailty=False,
            engine="python",
        )
        second = capacity.smoke_report(
            iterations=120,
            batch_size=60,
            rho=0.10,
            shared_frailty=False,
            engine="python",
        )
        self.assertEqual(first, second)
        self.assertEqual(first["status"], "non_authorizing_smoke")
        self.assertEqual(first["result"]["iterations"], 120)

    def test_default_cli_is_bounded_smoke(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(capacity.main([]), 0)
        payload = json.loads(output.getvalue())
        self.assertEqual(payload["status"], "non_authorizing_smoke")
        self.assertEqual(payload["result"]["iterations"], capacity.DEFAULT_SMOKE_ITERATIONS)
        self.assertLess(payload["result"]["iterations"], capacity.CONFIRM_ITERATIONS)

    def test_smoke_iteration_cap_blocks_accidental_large_run(self) -> None:
        with self.assertRaisesRegex(ValueError, "smoke iterations"):
            capacity.smoke_report(
                iterations=capacity.MAX_SMOKE_ITERATIONS + 1,
                batch_size=100,
                rho=0.10,
                shared_frailty=False,
                engine="python",
            )

    def test_exact_run_rejects_wrong_numpy_before_simulation(self) -> None:
        if np.__version__ == capacity.REQUIRED_NUMPY_VERSION:
            self.skipTest("current NumPy is the exact frozen runtime")
        with self.assertRaisesRegex(RuntimeError, "require NumPy"):
            capacity.build_exact_search_artifact()

    def test_combined_confirmation_has_one_fixed_three_stream_order(self) -> None:
        observed: list[str] = []

        def fake_profile(profile, oracle):
            del oracle
            observed.append(profile)
            return {
                "seed": {"domain": profile},
                "result": {"global_wilson_upper_one_sided_95": 0.14},
                "reference_match": True,
            }

        fake_context = contextlib.nullcontext(object())
        with (
            mock.patch.object(capacity, "_require_exact_environment"),
            mock.patch.object(capacity, "graph_robustness_report", return_value={}),
            mock.patch.object(capacity, "analytic_capacity_report", return_value={}),
            mock.patch.object(capacity, "compiled_oracle", return_value=fake_context),
            mock.patch.object(capacity, "compiled_oracle_self_test", return_value=True),
            mock.patch.object(capacity, "_run_exact_profile", side_effect=fake_profile),
        ):
            artifact = capacity.build_combined_confirmation_artifact()
        self.assertEqual(observed, ["rho010", "rho020", "shared-frailty"])
        self.assertEqual(artifact["scenario_order"], observed)
        self.assertEqual(set(artifact["scenarios"]), set(observed))
        self.assertEqual(artifact["profile"], "combined_confirmation")
        self.assertTrue(artifact["decision"]["passes"])

    def test_formal_outputs_are_standard_safe_paths_only(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO) as directory:
            root = Path(directory)
            (root / "data/water_impact_dynamic_v4").mkdir(parents=True)
            self.assertEqual(
                capacity.standard_formal_output(root, "search"),
                root.resolve()
                / "data/water_impact_dynamic_v4/v4_causal_capacity_search_v3.json",
            )
            self.assertEqual(
                capacity.standard_formal_output(root, "confirm"),
                root.resolve()
                / "data/water_impact_dynamic_v4/v4_causal_capacity_confirm_v3.json",
            )

        with tempfile.TemporaryDirectory(dir=REPO, prefix="sealed_") as directory:
            root = Path(directory)
            (root / "data/water_impact_dynamic_v4").mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "sealed/final36"):
                capacity.standard_formal_output(root, "search")

        with tempfile.TemporaryDirectory(dir=REPO) as directory:
            real = Path(directory) / "real"
            real.mkdir()
            (real / "data/water_impact_dynamic_v4").mkdir(parents=True)
            alias = Path(directory) / "alias"
            alias.symlink_to(real, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symlink component"):
                capacity.standard_formal_output(alias, "confirm")

    def test_exclusive_atomic_output_never_overwrites(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO) as directory:
            path = Path(directory) / "capacity.json"
            digest = capacity.write_json_exclusive_atomic(path, {"a": 1})
            original = path.read_bytes()
            self.assertEqual(digest, capacity.sha256_bytes(original))
            with self.assertRaises(FileExistsError):
                capacity.write_json_exclusive_atomic(path, {"a": 2})
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(
                [item.name for item in Path(directory).iterdir()], ["capacity.json"]
            )

    def test_reference_validation_rejects_any_count_change(self) -> None:
        result = {
            "iterations": 200_000,
            "rho": 0.10,
            "shared_frailty": False,
            "cell_shortage_failures": {name: 0 for name in capacity.CELL_NAMES},
            "exact_failure_counts": {
                "G1": 0, "G2": 0, "G3": 0, "global": 28_526
            },
            "global_failure_rate": 28_526 / 200_000,
            "global_wilson_upper_one_sided_95": capacity.wilson_upper_one_sided_95(
                28_526, 200_000
            ),
        }
        with self.assertRaisesRegex(RuntimeError, "global failure count"):
            capacity.validate_reference_result("search", result)


if __name__ == "__main__":
    unittest.main()
