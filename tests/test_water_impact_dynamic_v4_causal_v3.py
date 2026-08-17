#!/usr/bin/env python3
"""Tests for v4_dev72_v3 capacity and fail-closed causal core interfaces."""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import math
import os
import shutil
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


def _load_v3_module(name: str, relative: str):
    module_spec = importlib.util.spec_from_file_location(name, REPO / relative)
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


v3_protocol = _load_v3_module(
    "water_impact_dynamic_v4_eval_protocol_v3",
    "scripts/water_impact_dynamic_v4_eval_protocol_v3.py",
)
v3_builder = _load_v3_module(
    "build_water_impact_dynamic_v4_causal_candidates_v3",
    "scripts/build_water_impact_dynamic_v4_causal_candidates_v3.py",
)
v3_selector = _load_v3_module(
    "select_water_impact_dynamic_v4_causal_v3",
    "scripts/select_water_impact_dynamic_v4_causal_v3.py",
)
v3_validator = _load_v3_module(
    "validate_water_impact_dynamic_v4_causal_v3",
    "scripts/validate_water_impact_dynamic_v4_causal_v3.py",
)


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


def _impact(head: str) -> dict[str, object]:
    return {
        "verdict": "pass",
        "compact_and_rigid": True,
        "natural_drop_entry": True,
        "visible_brief_splash_or_ripple_plausible": True,
        "predominantly_buoyant_or_windborne": False,
        "flexible_or_film_like": False,
        "fragile": False,
        "powder": False,
        "loose_aggregate": False,
        "porous": False,
        "food_or_produce": False,
        "negative_buoyancy": True,
        "visually_recognizable": True,
        "entity_state": "solid_one_piece",
        "material": "dense alloy",
        "density_g_cm3": 7.8,
        "mass_g": 350,
        "dimensions_cm": [8.0, 5.0, 3.0],
        "size_class": "palm_sized_explicit",
        "source_specific_feature": f"machined face on {head}",
        "curator_note": f"the {head} is compact and recognizable",
    }


def _holdout_fixture() -> dict[str, object]:
    rows = []
    for pool in ("G1", "G2"):
        for ordinal in range(24):
            head = f"head{pool.casefold()}{ordinal:02d}"
            phrase = f"dense palm sized alloy {head}"
            rows.append(
                {
                    "source_id": f"v4v3_{pool.casefold()}_source_{ordinal:02d}",
                    "source_phrase": phrase,
                    "normalized_phrase": phrase,
                    "head_lemma": head,
                    "origin": "manufactured",
                    "food_status": "non_food",
                    "shape_class": "compact_irregular",
                    "color_family": "gray",
                    "material_family": "dense_alloy",
                    "texture_class": "smooth",
                    "impact_plausibility": _impact(head),
                    "physical_audit_status": "strict_physical_pass_v3",
                    "curator": "synthetic_test_curator",
                    "curation_stratum": "dense_alloy",
                    "group_pool": pool,
                    "head_ordinal": ordinal,
                }
            )
    return {
        "protocol": v3_builder.HOLDOUT_PROTOCOL,
        "dataset_version": v3_protocol.DATASET_VERSION,
        "source_count": 48,
        "sources": rows,
        "curation_audit": {"status": "synthetic_strict_pass"},
        "disjointness_commitment": "1" * 64,
    }


def _receiver_fixture() -> dict[str, object]:
    rows = []
    for pool, count in (("R1", 24), ("R3", 32)):
        for ordinal in range(count):
            head = f"receiver{pool.casefold()}{ordinal:02d}"
            phrase = (
                f"a clearly bounded stone rim with still water and an "
                f"unobstructed landing {head}"
            )
            rows.append(
                {
                    "receiver_id": f"v4v3_{pool.casefold()}_{ordinal:02d}",
                    "receiver_phrase": phrase,
                    "normalized_phrase": phrase,
                    "head_lemma": head,
                    "receiver_type": f"landscape_feature_{pool.casefold()}_{ordinal:02d}",
                    "pool": pool,
                    "receiver_ordinal": ordinal,
                    "curator_note": f"distinct receiver {head}",
                    "curator": "synthetic_test_curator",
                }
            )
    return {
        "protocol": v3_builder.RECEIVER_PROTOCOL,
        "dataset_version": v3_protocol.DATASET_VERSION,
        "receiver_count": 56,
        "pools": {"R1": 24, "R3": 32},
        "receivers": rows,
        "curation_audit": {"status": "synthetic_strict_pass"},
        "disjointness_commitment": "2" * 64,
    }


def _historical_fixture() -> dict[str, object]:
    rows = []
    for anchor in range(8):
        head = f"historical{anchor}"
        phrase = f"a historical water receiver {head}"
        rows.append(
            {
                "anchor_id": f"g2a{anchor}",
                "receiver_id": f"historical_receiver_{anchor}",
                "receiver_phrase": phrase,
                "normalized_phrase": phrase,
                "head_lemma": head,
                "historical_training_binding_sha256": f"{anchor + 1:x}" * 64,
            }
        )
    return {
        "protocol": v3_builder.HISTORICAL_PROTOCOL,
        "dataset_version": v3_protocol.DATASET_VERSION,
        "anchor_count": 8,
        "training_receiver_inventory_sha256": "a" * 64,
        "v2_disjointness_commitment": "b" * 64,
        "anchors": rows,
    }


def _eligible_rows(candidates) -> tuple[dict[str, str], ...]:
    return tuple(
        {
            "candidate_id": row["case_id"],
            "semantic_case_id": row["case_id"],
            "group": row["group"],
            "prompt_variant": row["prompt_variant"],
            "source_visibility": "2",
            "footprint_visibility": "2",
            "receiver": "2",
            "quality": "2",
            "causal_link": "2",
            "eligible": "yes",
        }
        for row in candidates
    )


def _artifact_records(expected, overrides=None):
    overrides = overrides or {}
    records = {}
    dispute_count = 7
    for index, (name, row_rule) in enumerate(expected.items(), start=1):
        if row_rule is None:
            rows = None
        elif row_rule == "positive":
            rows = 1
        elif row_rule == "disputes":
            rows = dispute_count
        else:
            rows = row_rule
        records[name] = {
            "sha256": f"{index % 16:x}" * 64,
            "size_bytes": index,
            "row_count": rows,
        }
    for name, digest in overrides.items():
        records[name]["sha256"] = digest
    return records


class V3CoreProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bank = json.loads(
            (REPO / v3_protocol.V2_BANK).read_text(encoding="utf-8")
        )
        cls.graph, cls.manifest = v3_builder.build_candidate_graph(
            holdout_payload=_holdout_fixture(),
            receiver_payload=_receiver_fixture(),
            historical_payload=_historical_fixture(),
            source_bank_payload=cls.bank,
            graph_assignment_salt="c" * 64,
        )
        cls.raw_eligibility = _eligible_rows(cls.manifest["candidates"])
        cls.eligibility = v3_selector.validate_eligibility_rows(
            cls.raw_eligibility, cls.manifest["candidates"]
        )
        cls.selected, cls.decisions = v3_selector.greedy_select(
            cls.manifest["candidates"], cls.eligibility, "d" * 64
        )

    def test_graph_exact_counts_incidence_and_projection(self) -> None:
        self.assertEqual(len(self.graph["edges"]), 576)
        self.assertEqual(self.graph["cell_counts"], {
            "holdout_source_new_receiver:direct": 48,
            "holdout_source_new_receiver:natural": 168,
            "holdout_source_seen_receiver:direct": 24,
            "holdout_source_seen_receiver:natural": 24,
            "seen_source_new_receiver:direct": 96,
            "seen_source_new_receiver:natural": 216,
        })
        self.assertEqual(self.manifest["candidates"], self.graph["edges"])
        v3_builder.validate_candidate_projection(self.graph, self.manifest)

    def test_graph_structural_tamper_rejected_even_when_rehashed(self) -> None:
        tampered = json.loads(json.dumps(self.graph))
        row = tampered["edges"][0]
        row["receiver_id"] = tampered["r1"]["receiver_ids"][5]
        base = dict(row)
        base.pop("canonical_record_sha256")
        row["canonical_record_sha256"] = v3_protocol.sha256_bytes(
            v3_protocol.canonical_json_bytes(base)
        )
        graph_base = dict(tampered)
        graph_base.pop("graph_sha256")
        tampered["graph_sha256"] = v3_protocol.sha256_bytes(
            v3_protocol.canonical_json_bytes(graph_base)
        )
        with self.assertRaisesRegex(ValueError, "G1 graph incidence"):
            v3_builder.validate_graph_payload(tampered)

    def test_rank_and_seed_domains_are_exact(self) -> None:
        row = self.manifest["candidates"][0]
        expected_rank = v3_protocol.sha256_bytes(
            b"causal-selector-v3\x00"
            + ("e" * 64).encode("ascii")
            + b"\x00"
            + v3_protocol.candidate_record_bytes(row)
        )
        self.assertEqual(v3_protocol.selection_rank(row, "e" * 64), expected_rank)
        expected_seed = int.from_bytes(
            hashlib.sha256(
                b"causal-eval-seed-v3\x00"
                + ("f" * 64).encode("ascii")
                + b"\x00"
                + row["case_id"].encode("utf-8")
                + b"\x001"
            ).digest()[:4],
            "big",
        )
        self.assertEqual(
            v3_protocol.derive_evaluation_seed("f" * 64, row["case_id"], 1),
            expected_seed,
        )

    def test_greedy_selector_enforces_all_registered_constraints(self) -> None:
        self.assertEqual(len(self.selected), 24)
        v3_selector.validate_selected_rows(self.selected)
        self.assertEqual(
            [row["selection_rank_sha256"] for row in self.selected],
            sorted(row["selection_rank_sha256"] for row in self.selected),
        )
        self.assertTrue(any(row["decision"] == "excluded_no_completion" for row in self.decisions))

    def test_g2_forced_and_excluded_candidate_semantics(self) -> None:
        ranked = []
        for row in self.manifest["candidates"]:
            item = dict(row)
            item["eligible"] = True
            item["selection_rank_sha256"] = v3_protocol.selection_rank(row, "d" * 64)
            ranked.append(item)
        g2 = [row for row in ranked if row["group"] == v3_protocol.GROUPS[1]]
        same_anchor = [row for row in g2 if row["physical_anchor_id"] == "g2a0"]
        self.assertIsNone(
            v3_selector.group_completion(
                v3_protocol.GROUPS[1], ranked,
                frozenset({same_anchor[0]["case_id"], same_anchor[1]["case_id"]}),
                frozenset(),
            )
        )
        forced = same_anchor[0]
        completion = v3_selector.group_completion(
            v3_protocol.GROUPS[1], ranked,
            frozenset({forced["case_id"]}), frozenset(),
        )
        self.assertIsNotNone(completion)
        self.assertIn(forced["case_id"], completion)
        self.assertIsNone(
            v3_selector.group_completion(
                v3_protocol.GROUPS[1], ranked,
                frozenset({forced["case_id"]}), frozenset({forced["case_id"]}),
            )
        )
        excluded_anchor = frozenset(
            row["case_id"] for row in same_anchor
        )
        self.assertIsNone(
            v3_selector.group_completion(
                v3_protocol.GROUPS[1], ranked, frozenset(), excluded_anchor
            )
        )

    def test_rank_tie_is_terminal(self) -> None:
        original = v3_selector.protocol.selection_rank
        try:
            v3_selector.protocol.selection_rank = lambda row, salt: "0" * 64
            with self.assertRaisesRegex(v3_selector.PreflightDatasetInvalid, "rank tie"):
                v3_selector.greedy_select(
                    self.manifest["candidates"], self.eligibility, "d" * 64
                )
        finally:
            v3_selector.protocol.selection_rank = original

    def test_seed_collision_fails_closed(self) -> None:
        first_seed = v3_protocol.derive_evaluation_seed(
            "f" * 64, self.selected[0]["case_id"], 0
        )
        with self.assertRaisesRegex(ValueError, "seed collision"):
            v3_selector.build_private_outputs(
                self.selected,
                evaluation_salt="f" * 64,
                screening_seed=1,
                forbidden_seeds={first_seed},
            )

    def test_stage_registry_exact_inventory_and_dispute_cross_count(self) -> None:
        stage0 = {
            "protocol": v3_protocol.COMMITMENT_PROTOCOL,
            "dataset": "causal",
            "dataset_version": v3_protocol.DATASET_VERSION,
            "stage": 0,
            "status": "committed",
            "sealed_final36_status": "unopened",
            "artifacts": _artifact_records(v3_protocol.STAGE0_ARTIFACT_ROWS),
        }
        v3_protocol.validate_commitment_registry(stage0, stage=0)
        stage1 = {
            "protocol": v3_protocol.COMMITMENT_PROTOCOL,
            "dataset": "causal",
            "dataset_version": v3_protocol.DATASET_VERSION,
            "stage": 1,
            "status": "committed",
            "sealed_final36_status": "unopened",
            "stage0_registry_sha256": "1" * 64,
            "artifacts": _artifact_records(v3_protocol.STAGE1_ARTIFACT_ROWS),
        }
        v3_protocol.validate_commitment_registry(
            stage1, stage=1, expected_stage0_sha256="1" * 64
        )
        stage1["artifacts"]["screening_adjudication"]["row_count"] += 1
        with self.assertRaisesRegex(ValueError, "counts differ"):
            v3_protocol.validate_commitment_registry(
                stage1, stage=1, expected_stage0_sha256="1" * 64
            )

    def test_reports_and_invalid_outcome_are_exact_and_aggregate_only(self) -> None:
        identity = {
            "protocol": v3_protocol.IDENTITY_REPORT_PROTOCOL,
            "status": "passed",
            "dataset_version": v3_protocol.DATASET_VERSION,
            "v2_stage0_registry_sha256": v3_protocol.V2_STAGE0_SHA256,
            "v2_candidate_manifest_sha256": "1" * 64,
            "v3_candidate_graph_sha256": "2" * 64,
            "v3_ontology_bundle_sha256": "3" * 64,
            "compared_counts": {"v2": 48, "v3": 576},
            "allowed_identity_exceptions": {
                "original_source_nodes": 8,
                "historical_receiver_nodes": 8,
            },
            "intersection_counts": {
                "case_id": 0,
                "canonical_record": 0,
                "fresh_source_id": 0,
                "fresh_receiver_id": 0,
                "source_receiver_pair": 0,
                "source_receiver_variant_triple": 0,
            },
        }
        v3_protocol.validate_identity_disjointness_report(identity)
        construct = {
            "protocol": v3_protocol.CONSTRUCT_REPORT_PROTOCOL,
            "status": "passed",
            "dataset_version": v3_protocol.DATASET_VERSION,
            "v2_stage0_registry_sha256": v3_protocol.V2_STAGE0_SHA256,
            "v2_file_sha256": {
                "templates": v3_protocol.V2_TEMPLATE_SHA256,
                "field_rules": v3_protocol.V2_FIELD_RULES_SHA256,
                "selection_rules": v3_protocol.V2_SELECTION_RULES_SHA256,
            },
            "v3_file_sha256": {
                "templates": v3_protocol.V2_TEMPLATE_SHA256,
                "field_rules": v3_protocol.V2_FIELD_RULES_SHA256,
                "selection_rules": "4" * 64,
            },
            "qualification_sha256": {"v2": "5" * 64, "v3": "5" * 64},
            "cell_quota_sha256": {"v2": "6" * 64, "v3": "6" * 64},
            "exact_equal": {
                "templates": True,
                "field_rules": True,
                "qualification": True,
                "cell_quota": True,
            },
        }
        v3_protocol.validate_construct_equivalence_report(construct)
        invalid = {
            "protocol": v3_protocol.INVALID_OUTCOME_PROTOCOL,
            "dataset": "causal",
            "dataset_version": v3_protocol.DATASET_VERSION,
            "status": "preflight_dataset_invalid",
            "failure_phase": "stage0_authorization",
            "reason_code": "stage0_authorization_integrity_failure",
            "stage0_registry_sha256": None,
            "candidate_count": 576,
            "eligible_count": None,
            "cell_eligible_counts": None,
            "selector_output_created": False,
            "unit_manifest_created": False,
            "stage1_registry_created": False,
            "sealed_final36_status": "unopened",
            "bound_artifacts": {
                "stage0_registry": None,
                "screening_generation_manifest": None,
                "screening_package_commitment": None,
                "screening_freeze_manifest": None,
                "canonical_eligibility": None,
                "selector_stderr": None,
            },
        }
        v3_protocol.validate_invalid_outcome(invalid)
        invalid["free_text"] = "forbidden"
        with self.assertRaisesRegex(ValueError, "fields are not exact"):
            v3_protocol.validate_invalid_outcome(invalid)

    def test_runtime_v2_read_gate_and_static_import_gate(self) -> None:
        allowed = REPO / v3_protocol.V2_BANK
        self.assertEqual(
            v3_protocol.validate_runtime_read_path(REPO, allowed, allow_v2=True),
            v3_protocol.V2_BANK.as_posix(),
        )
        with self.assertRaisesRegex(ValueError, "nonallowlisted v2 read"):
            v3_protocol.validate_runtime_read_path(
                REPO,
                REPO / "data/water_impact_dynamic_v4/causal_stage0_commitment_v2.json",
                allow_v2=True,
            )
        with tempfile.TemporaryDirectory(dir=REPO) as directory:
            root = Path(directory)
            report = root / v3_protocol.CONSTRUCT_REPORT
            report.parent.mkdir(parents=True)
            report.write_text("{}\n", encoding="utf-8")
            self.assertEqual(
                v3_protocol.validate_runtime_read_path(
                    root, report, allow_v2=False
                ),
                v3_protocol.CONSTRUCT_REPORT.as_posix(),
            )
        with tempfile.TemporaryDirectory(dir=REPO) as directory:
            bad = Path(directory) / "bad_v3.py"
            bad.write_text("import water_impact_dynamic_v4_eval_protocol_v2\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "v2 import forbidden"):
                v3_protocol.validate_no_v2_imports([bad], REPO)

    def test_parent_directory_alias_is_rejected_by_protocol_and_builder(self) -> None:
        alias = (
            REPO
            / "data/water_impact_dynamic_v4/../water_impact_dynamic_v4/source_bank_public64_registry_v2.json"
        )
        with self.assertRaisesRegex(
            ValueError, "noncanonical lexical alias|parent-directory alias"
        ):
            v3_protocol.validate_runtime_read_path(REPO, alias, allow_v2=True)

        with tempfile.TemporaryDirectory(dir=REPO) as directory:
            private = Path(directory)
            private.chmod(0o700)
            arguments = [
                "--project-root", str(REPO),
                "--private-root", str(private),
                "--holdout-ontology", str(private / "eval_holdout_source_ontology_private48_v3.json"),
                "--receiver-ontology", str(private / "receiver_ontology_private56_v3.json"),
                "--historical-anchors", str(private / "historical_receiver_anchors_private8_v3.json"),
                "--templates", str(private / "causal_stage0_templates_private_v3.json"),
                "--field-rules", str(private / "causal_stage0_field_rules_private_v3.json"),
                "--graph-salt", str(private / "causal_graph_assignment_salt_v3.txt"),
                "--source-bank", str(alias),
                "--source-mapping", str(REPO / v3_protocol.V2_MAPPING),
                "--graph-output", str(private / "causal_stage0_candidate_graph_private576_v3.json"),
                "--candidate-output", str(private / "causal_stage0_candidates_private576_v3.json"),
            ]
            with self.assertRaisesRegex(
                ValueError, "noncanonical lexical alias|parent-directory alias"
            ):
                v3_builder.main(arguments)
            self.assertEqual(list(private.iterdir()), [])

    def test_transitive_v2_import_cycle_and_missing_local_import_fail(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO) as directory:
            root = Path(directory)
            entry = root / "entry_v3.py"
            middle = root / "middle_v3.py"
            entry.write_text("import middle_v3\n", encoding="utf-8")
            middle.write_text(
                "import entry_v3\nimport water_impact_dynamic_v4_eval_protocol_v2\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "v2 import forbidden"):
                v3_protocol.validate_no_v2_imports([entry], root)

            middle.write_text("import entry_v3\n", encoding="utf-8")
            v3_protocol.validate_no_v2_imports([entry], root)
            entry.write_text(
                "import water_impact_dynamic_v4_missing_v3\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(FileNotFoundError, "repo-local import is missing"):
                v3_protocol.validate_no_v2_imports([entry], root)

    def test_each_allowed_v2_public_file_tamper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO) as directory:
            root = Path(directory)
            for relative in v3_protocol.V2_RUNTIME_READ_ALLOWLIST:
                destination = root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(REPO / relative, destination)
            baseline = v3_protocol.validate_v2_public_inputs(root)
            self.assertEqual(baseline, v3_protocol.V2_RUNTIME_READ_ALLOWLIST)
            for relative in v3_protocol.V2_RUNTIME_READ_ALLOWLIST:
                with self.subTest(relative=relative):
                    path = root / relative
                    original = path.read_bytes()
                    path.write_bytes(original + b"\n")
                    with self.assertRaisesRegex(ValueError, "hash mismatch"):
                        v3_builder._require_v2_hashes_unchanged(root, baseline)
                    path.write_bytes(original)

    def test_v2_fallback_and_code_registry_drift_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO) as directory:
            root = Path(directory)
            artifact = root / "artifact_v3.py"
            artifact.write_text("VALUE = 1\n", encoding="utf-8")
            with mock.patch.object(
                v3_protocol,
                "CODE_ARTIFACT_PATHS",
                {"stage0_authorizer": "artifact_v3.py"},
            ):
                payload = {
                    "protocol": v3_protocol.CODE_REGISTRY_PROTOCOL,
                    "status": "frozen",
                    "dataset_version": v3_protocol.DATASET_VERSION,
                    "v2_read_allowlist": dict(v3_protocol.V2_RUNTIME_READ_ALLOWLIST),
                    "artifacts": {
                        "stage0_authorizer": {
                            "path": "artifact_v3.py",
                            "sha256": v3_protocol.sha256_file(artifact),
                        }
                    },
                }
                v3_protocol.validate_code_registry(payload, root)
                artifact.write_text("VALUE = 2\n", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "byte drift"):
                    v3_protocol.validate_code_registry(payload, root)
                payload["artifacts"]["stage0_authorizer"]["path"] = (
                    "artifact_v2.py"
                )
                with self.assertRaisesRegex(ValueError, "mismatch"):
                    v3_protocol.validate_code_registry(payload, root)

    def test_builder_rechecks_v2_inputs_before_and_after_publication(self) -> None:
        baseline = dict(v3_protocol.V2_RUNTIME_READ_ALLOWLIST)
        changed = dict(baseline)
        first_key = next(iter(changed))
        changed[first_key] = "0" * 64
        with tempfile.TemporaryDirectory(dir=REPO) as directory:
            private = Path(directory)
            private.chmod(0o700)
            arguments = [
                "--project-root", str(REPO),
                "--private-root", str(private),
                "--holdout-ontology", str(private / "eval_holdout_source_ontology_private48_v3.json"),
                "--receiver-ontology", str(private / "receiver_ontology_private56_v3.json"),
                "--historical-anchors", str(private / "historical_receiver_anchors_private8_v3.json"),
                "--templates", str(private / "causal_stage0_templates_private_v3.json"),
                "--field-rules", str(private / "causal_stage0_field_rules_private_v3.json"),
                "--graph-salt", str(private / "causal_graph_assignment_salt_v3.txt"),
                "--source-bank", str(REPO / v3_protocol.V2_BANK),
                "--source-mapping", str(REPO / v3_protocol.V2_MAPPING),
                "--graph-output", str(private / "causal_stage0_candidate_graph_private576_v3.json"),
                "--candidate-output", str(private / "causal_stage0_candidates_private576_v3.json"),
            ]
            bank = json.loads((REPO / v3_protocol.V2_BANK).read_text())
            side_effects = [baseline, baseline, changed]
            def fake_load(path, **kwargs):
                if Path(path) == REPO / v3_protocol.V2_BANK:
                    return bank
                if Path(path) == REPO / v3_protocol.V2_MAPPING:
                    return {}
                return {}
            with mock.patch.object(
                v3_builder.protocol,
                "validate_v2_public_inputs",
                side_effect=side_effects,
            ), mock.patch.object(
                v3_builder.protocol,
                "validate_runtime_read_path",
                side_effect=lambda root, path, allow_v2=False: Path(path).resolve().relative_to(REPO).as_posix(),
            ), mock.patch.object(
                v3_builder.protocol, "load_json", side_effect=fake_load
            ), mock.patch.object(
                v3_builder,
                "validate_templates_and_fields",
                return_value=({}, {}),
            ), mock.patch.object(
                v3_builder, "_load_salt", return_value="c" * 64
            ), mock.patch.object(
                v3_builder.protocol, "validate_private_output_path"
            ), mock.patch.object(
                v3_builder,
                "build_candidate_graph",
                return_value=(self.graph, self.manifest),
            ), mock.patch.object(
                v3_builder.protocol, "write_json_exclusive_atomic"
            ) as writer:
                with self.assertRaisesRegex(ValueError, "changed during"):
                    v3_builder.main(arguments)
                self.assertEqual(writer.call_count, 2)
            with mock.patch.object(
                v3_builder.protocol,
                "validate_v2_public_inputs",
                side_effect=[baseline, changed],
            ), mock.patch.object(
                v3_builder.protocol,
                "validate_runtime_read_path",
                side_effect=lambda root, path, allow_v2=False: Path(path).resolve().relative_to(REPO).as_posix(),
            ), mock.patch.object(
                v3_builder.protocol, "load_json", side_effect=fake_load
            ), mock.patch.object(
                v3_builder,
                "validate_templates_and_fields",
                return_value=({}, {}),
            ), mock.patch.object(
                v3_builder, "_load_salt", return_value="c" * 64
            ), mock.patch.object(
                v3_builder.protocol, "validate_private_output_path"
            ), mock.patch.object(
                v3_builder,
                "build_candidate_graph",
                return_value=(self.graph, self.manifest),
            ), mock.patch.object(
                v3_builder.protocol, "write_json_exclusive_atomic"
            ) as writer:
                with self.assertRaisesRegex(ValueError, "changed during"):
                    v3_builder.main(arguments)
                writer.assert_not_called()

    def test_private_permissions_and_hardlinks_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO) as directory:
            root = Path(directory)
            root.chmod(0o700)
            original = root / "value.json"
            original.write_text("{}\n", encoding="utf-8")
            original.chmod(0o600)
            v3_protocol.validate_private_path(root, original)
            alias = root / "alias.json"
            os.link(original, alias)
            with self.assertRaisesRegex(PermissionError, "hardlinks"):
                v3_protocol.validate_private_path(root, original)

    def test_selector_math_outputs_validate_but_formal_stages_remain_closed(self) -> None:
        selected_payload, unit_payload = v3_selector.build_private_outputs(
            self.selected,
            evaluation_salt="f" * 64,
            screening_seed=7,
            forbidden_seeds={8, 9},
        )
        stage0 = {
            "protocol": v3_protocol.COMMITMENT_PROTOCOL,
            "dataset": "causal",
            "dataset_version": v3_protocol.DATASET_VERSION,
            "stage": 0,
            "status": "committed",
            "sealed_final36_status": "unopened",
            "artifacts": _artifact_records(v3_protocol.STAGE0_ARTIFACT_ROWS),
        }
        stage0_sha256 = v3_protocol.sha256_bytes(
            v3_protocol.canonical_json_bytes(stage0)
        )
        summary = v3_selector.build_selector_summary(
            eligibility=self.eligibility,
            selected_payload=selected_payload,
            unit_payload=unit_payload,
            stage0_registry_sha256=stage0_sha256,
            screening_freeze_sha256="b" * 64,
            eligibility_table_sha256="c" * 64,
        )
        v3_validator.validate_selected_payload(selected_payload)
        v3_validator.validate_unit_payload(
            unit_payload,
            selected_payload["selected"],
            evaluation_salt="f" * 64,
            screening_seed=7,
            forbidden_seeds={8, 9},
        )
        v3_protocol.validate_selector_summary(summary)
        self.assertFalse(
            hasattr(v3_validator, "_validate_stage1_selection_fixture")
        )
        with self.assertRaisesRegex(
            RuntimeError, "formal Stage1 provenance validation not implemented"
        ):
            v3_validator.validate_stage1_core(selector_summary=summary)
        with self.assertRaisesRegex(
            RuntimeError, "formal Stage0 provenance validation not implemented"
        ):
            v3_validator.validate_stage0_core(
                registry=stage0,
                graph=self.graph,
                candidate_manifest=self.manifest,
                holdout_ontology={},
                receiver_ontology={},
                historical_anchors={},
                source_bank={},
                graph_assignment_salt="c" * 64,
                identity_report={},
                construct_report={},
                graph_file_sha256="1" * 64,
                template_file_sha256="2" * 64,
                field_rules_file_sha256="3" * 64,
                selection_rules_file_sha256="4" * 64,
            )
        unit_payload["units"][0]["seed"] += 1
        with self.assertRaisesRegex(ValueError, "seed derivation"):
            v3_validator.validate_unit_payload(
                unit_payload,
                selected_payload["selected"],
                evaluation_salt="f" * 64,
                screening_seed=7,
                forbidden_seeds={8, 9},
            )

    def test_formal_stage0_and_stage1_cli_write_nothing(self) -> None:
        rebound_graph = json.loads(json.dumps(self.graph))
        rebound_graph["edges"][0]["receiver_phrase"] = "rebound private row"
        with self.assertRaisesRegex(
            RuntimeError, "formal Stage1 provenance validation not implemented"
        ):
            v3_validator.validate_stage1_core(graph=rebound_graph)

        with tempfile.TemporaryDirectory(dir=REPO) as directory:
            root = Path(directory)
            paths = {
                0: root / v3_protocol.STAGE0_REGISTRY,
                1: root / v3_protocol.STAGE1_REGISTRY,
            }
            for path in paths.values():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}\n", encoding="utf-8")
            before = sorted(
                item.relative_to(root).as_posix() for item in root.rglob("*")
            )
            for stage, path in paths.items():
                with self.assertRaisesRegex(
                    RuntimeError,
                    f"formal Stage{stage} provenance validation not implemented",
                ):
                    v3_validator.main([
                        "registry",
                        "--project-root", str(root),
                        "--path", str(path),
                        "--stage", str(stage),
                        "--stage0-sha256", "a" * 64,
                    ])
                with self.assertRaisesRegex(
                    RuntimeError,
                    f"formal Stage{stage} provenance validation not implemented",
                ):
                    v3_validator.validate_registry_file(
                        path,
                        stage=stage,
                        expected_stage0_sha256="a" * 64,
                    )
            self.assertEqual(
                before,
                sorted(item.relative_to(root).as_posix() for item in root.rglob("*")),
            )

    def test_preregistered_code_paths_and_missing_artifacts_fail_closed(self) -> None:
        self.assertEqual(
            v3_protocol.CODE_ARTIFACT_PATHS["candidate_builder"],
            "scripts/build_water_impact_dynamic_v4_causal_candidates_v3.py",
        )
        self.assertEqual(
            v3_protocol.CODE_ARTIFACT_PATHS["selector"],
            "scripts/select_water_impact_dynamic_v4_causal_v3.py",
        )
        with tempfile.TemporaryDirectory(dir=REPO) as directory:
            root = Path(directory)
            with self.assertRaisesRegex(FileNotFoundError, "required v3 code"):
                v3_validator.validate_static_code_boundary(root)
            for relative in v3_protocol.CODE_ARTIFACT_PATHS.values():
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("FROZEN_V3_TEST_STUB = True\n", encoding="utf-8")
            v3_validator.validate_static_code_boundary(root)

    def test_receiver_head_type_and_eligibility_booleans_are_strict(self) -> None:
        duplicate_head = _receiver_fixture()
        duplicate_head["receivers"][1]["head_lemma"] = duplicate_head["receivers"][0]["head_lemma"]
        duplicate_head["receivers"][1]["receiver_phrase"] = (
            "a clearly bounded metal rim with still water and an unobstructed "
            f"landing {duplicate_head['receivers'][0]['head_lemma']}"
        )
        duplicate_head["receivers"][1]["normalized_phrase"] = duplicate_head["receivers"][1]["receiver_phrase"]
        with self.assertRaisesRegex(ValueError, "head/type"):
            v3_builder.validate_receiver_ontology(duplicate_head)
        duplicate_type = _receiver_fixture()
        duplicate_type["receivers"][1]["receiver_type"] = duplicate_type["receivers"][0]["receiver_type"]
        with self.assertRaisesRegex(ValueError, "head/type"):
            v3_builder.validate_receiver_ontology(duplicate_type)

        nonboolean = [dict(row) for row in self.eligibility]
        nonboolean[0]["eligible"] = "yes"
        with self.assertRaisesRegex(ValueError, "strict booleans"):
            v3_selector.greedy_select(
                self.manifest["candidates"], nonboolean, "d" * 64
            )

    def test_private_path_rejects_symlink_ancestors_and_resolved_forbidden(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO) as directory:
            outer = Path(directory)
            real = outer / "real"
            real.mkdir(mode=0o700)
            alias = outer / "alias"
            alias.symlink_to(real, target_is_directory=True)
            private_root = alias / "private"
            private_root.mkdir(mode=0o700)
            value = private_root / "value.json"
            value.write_text("{}\n", encoding="utf-8")
            value.chmod(0o600)
            with self.assertRaisesRegex(ValueError, "symlink component"):
                v3_protocol.validate_private_path(private_root, value)

        with tempfile.TemporaryDirectory(dir=REPO) as directory:
            outer = Path(directory)
            forbidden = outer / "quarantine_area"
            forbidden.mkdir(mode=0o700)
            alias = outer / "benign_alias"
            alias.symlink_to(forbidden, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "quarantine"):
                v3_protocol.reject_forbidden_path(alias / "value.json")

    def test_validator_cli_rejects_v2_sealed_and_symlink_inputs_without_writes(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO) as directory:
            root = Path(directory)
            wrong_v2 = root / "data/water_impact_dynamic_v4/causal_stage0_commitment_v2.json"
            wrong_v2.parent.mkdir(parents=True)
            wrong_v2.write_text("{}\n", encoding="utf-8")
            before = sorted(item.relative_to(root).as_posix() for item in root.rglob("*"))
            with self.assertRaisesRegex(ValueError, "must be exactly"):
                v3_validator.main([
                    "invalid-outcome",
                    "--project-root", str(root),
                    "--path", str(wrong_v2),
                ])
            after = sorted(item.relative_to(root).as_posix() for item in root.rglob("*"))
            self.assertEqual(before, after)

        with tempfile.TemporaryDirectory(dir=REPO) as directory:
            root = Path(directory)
            sealed = root / "sealed/final36/causal_preflight_dataset_invalid_v3.json"
            sealed.parent.mkdir(parents=True)
            sealed.write_text("{}\n", encoding="utf-8")
            before = sorted(item.relative_to(root).as_posix() for item in root.rglob("*"))
            with self.assertRaisesRegex(ValueError, "sealed/final36"):
                v3_validator.main([
                    "invalid-outcome",
                    "--project-root", str(root),
                    "--path", str(sealed),
                ])
            self.assertEqual(
                before,
                sorted(item.relative_to(root).as_posix() for item in root.rglob("*")),
            )

    def test_public_and_code_hardlink_or_root_symlink_aliases_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO) as directory:
            root = Path(directory) / "real"
            root.mkdir()
            for relative in v3_protocol.V2_RUNTIME_READ_ALLOWLIST:
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                os.link(REPO / relative, target)
            with self.assertRaisesRegex(PermissionError, "hardlinks"):
                v3_protocol.validate_v2_public_inputs(root)
            alias = Path(directory) / "root_alias"
            alias.symlink_to(root, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symlink component"):
                v3_protocol.validate_project_root(alias)

        with tempfile.TemporaryDirectory(dir=REPO) as directory:
            root = Path(directory)
            source = root / "stub.py"
            source.write_text("FROZEN_V3_TEST_STUB = True\n", encoding="utf-8")
            for relative in v3_protocol.CODE_ARTIFACT_PATHS.values():
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                os.link(source, target)
            with self.assertRaisesRegex(PermissionError, "hardlinked"):
                v3_validator.validate_static_code_boundary(root)

    def test_commitment_stage_is_exactly_integer_zero_or_one(self) -> None:
        for invalid_stage in (True, 2, -1):
            payload = {
                "protocol": v3_protocol.COMMITMENT_PROTOCOL,
                "dataset": "causal",
                "dataset_version": v3_protocol.DATASET_VERSION,
                "stage": invalid_stage,
                "status": "committed",
                "sealed_final36_status": "unopened",
                "artifacts": _artifact_records(v3_protocol.STAGE1_ARTIFACT_ROWS),
            }
            with self.assertRaisesRegex(ValueError, "exactly 0 or 1"):
                v3_protocol.validate_commitment_registry(
                    payload, stage=invalid_stage
                )
        bool_payload = {
            "protocol": v3_protocol.COMMITMENT_PROTOCOL,
            "dataset": "causal",
            "dataset_version": v3_protocol.DATASET_VERSION,
            "stage": True,
            "status": "committed",
            "sealed_final36_status": "unopened",
            "stage0_registry_sha256": "1" * 64,
            "artifacts": _artifact_records(v3_protocol.STAGE1_ARTIFACT_ROWS),
        }
        with self.assertRaisesRegex(ValueError, "stage/status"):
            v3_protocol.validate_commitment_registry(
                bool_payload, stage=1, expected_stage0_sha256="1" * 64
            )

    def test_invalid_outcome_phase_matrix_and_stage0_rehash_are_exact(self) -> None:
        cells = {f"{g}:{v}": 4 for g, v in v3_protocol.CELL_ORDER}
        payload = {
            "protocol": v3_protocol.INVALID_OUTCOME_PROTOCOL,
            "dataset": "causal",
            "dataset_version": v3_protocol.DATASET_VERSION,
            "status": "preflight_dataset_invalid",
            "failure_phase": "selection",
            "reason_code": "global_subset_infeasible",
            "stage0_registry_sha256": "1" * 64,
            "candidate_count": 576,
            "eligible_count": 24,
            "cell_eligible_counts": cells,
            "selector_output_created": False,
            "unit_manifest_created": False,
            "stage1_registry_created": False,
            "sealed_final36_status": "unopened",
            "bound_artifacts": {
                name: ("1" * 64 if name == "stage0_registry" else "2" * 64)
                for name in v3_protocol.INVALID_BOUND_ARTIFACT_KEYS
            },
        }
        v3_protocol.validate_invalid_outcome(
            payload, expected_stage0_sha256="1" * 64
        )
        with self.assertRaisesRegex(ValueError, "exact standard Stage-0"):
            v3_protocol.validate_invalid_outcome(
                payload, expected_stage0_sha256="3" * 64
            )
        malformed = json.loads(json.dumps(payload))
        malformed["failure_phase"] = "original_generation"
        malformed["reason_code"] = "screening_generation_incomplete"
        with self.assertRaisesRegex(ValueError, "phase matrix|failure phase"):
            v3_protocol.validate_invalid_outcome(
                malformed, expected_stage0_sha256="1" * 64
            )

    def test_selector_cli_is_unconditionally_closed_and_writes_nothing(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO) as directory:
            root = Path(directory)
            output = root / "selector"
            arguments = [
                "--private-root", str(root),
                "--candidate-graph", str(root / "graph.json"),
                "--candidate-manifest", str(root / "manifest.json"),
                "--eligibility", str(root / "eligibility.csv"),
                "--selector-salt", str(root / "selector.txt"),
                "--evaluation-salt", str(root / "evaluation.txt"),
                "--screening-seed", str(root / "screening.txt"),
                "--forbidden-seeds", str(root / "forbidden.json"),
                "--stage0-registry-sha256", "1" * 64,
                "--screening-freeze-sha256", "2" * 64,
                "--output-dir", str(output),
            ]
            before = list(root.iterdir())
            with self.assertRaisesRegex(RuntimeError, "formal selector execution"):
                v3_selector.main(arguments)
            self.assertEqual(before, list(root.iterdir()))
            self.assertFalse(output.exists())

    def test_source_and_receiver_scientific_fields_fail_closed(self) -> None:
        holdout = _holdout_fixture()
        holdout["sources"][0]["source_phrase"] = "object headg100"
        holdout["sources"][0]["normalized_phrase"] = "object headg100"
        holdout["sources"][0]["head_lemma"] = "headg100"
        with self.assertRaisesRegex(ValueError, "palm-sized dense"):
            v3_builder.validate_holdout_ontology(holdout)
        holdout = _holdout_fixture()
        holdout["sources"][0]["impact_plausibility"]["dimensions_cm"] = [4.0, 4.0, 4.0]
        with self.assertRaisesRegex(ValueError, "palm-sized extent"):
            v3_builder.validate_holdout_ontology(holdout)

        receivers = _receiver_fixture()
        head = receivers["receivers"][0]["head_lemma"]
        receivers["receivers"][0]["receiver_phrase"] = f"object {head}"
        receivers["receivers"][0]["normalized_phrase"] = f"object {head}"
        with self.assertRaisesRegex(ValueError, "still water"):
            v3_builder.validate_receiver_ontology(receivers)

    def test_graph_salt_permutation_hashes_and_tie_are_exact(self) -> None:
        self.assertEqual(
            self.graph["graph_assignment_salt_sha256"],
            v3_protocol.sha256_bytes((("c" * 64) + "\n").encode("ascii")),
        )
        for name in ("r1", "r3"):
            self.assertEqual(
                self.graph[name]["permutation_sha256"],
                v3_protocol.sha256_bytes(
                    v3_protocol.canonical_json_bytes(
                        self.graph[name]["receiver_ids"]
                    )
                ),
            )

        class ConstantDigest:
            def hexdigest(self):
                return "0" * 64

        with mock.patch.object(
            v3_builder.hashlib, "sha256", return_value=ConstantDigest()
        ):
            with self.assertRaisesRegex(ValueError, "rank tie"):
                v3_builder._permuted_receivers(
                    _receiver_fixture()["receivers"], "R1", "c" * 64
                )

    def test_secret_separation_and_forbidden_seed_order_are_strict(self) -> None:
        with self.assertRaisesRegex(ValueError, "pairwise distinct"):
            v3_protocol.validate_secret_separation(
                graph_assignment_salt="1" * 64,
                selector_salt="1" * 64,
                evaluation_salt="2" * 64,
                screening_seed=7,
            )
        inventory = {
            "protocol": v3_selector.FORBIDDEN_PROTOCOL,
            "dataset": "causal",
            "status": "frozen_by_independent_seed_auditor",
            "seed_encoding": "nonnegative JSON integer below 2^63",
            "source_commitments": [
                {"name": "z", "sha256": "1" * 64, "seed_count": 1},
                {"name": "a", "sha256": "2" * 64, "seed_count": 1},
            ],
            "seeds": [2, 1],
        }
        with self.assertRaisesRegex(ValueError, "unique and sorted"):
            v3_selector.validate_forbidden_seed_inventory(inventory)

    def test_candidate_builder_requires_exact_private_basenames(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO) as directory:
            root = Path(directory)
            root.chmod(0o700)
            arguments = [
                "--project-root", str(REPO),
                "--private-root", str(root),
                "--holdout-ontology", str(root / "wrong.json"),
                "--receiver-ontology", str(root / "receiver_ontology_private56_v3.json"),
                "--historical-anchors", str(root / "historical_receiver_anchors_private8_v3.json"),
                "--templates", str(root / "causal_stage0_templates_private_v3.json"),
                "--field-rules", str(root / "causal_stage0_field_rules_private_v3.json"),
                "--graph-salt", str(root / "causal_graph_assignment_salt_v3.txt"),
                "--source-bank", str(REPO / v3_protocol.V2_BANK),
                "--source-mapping", str(REPO / v3_protocol.V2_MAPPING),
                "--graph-output", str(root / "causal_stage0_candidate_graph_private576_v3.json"),
                "--candidate-output", str(root / "causal_stage0_candidates_private576_v3.json"),
            ]
            with self.assertRaisesRegex(ValueError, "basename must be exactly"):
                v3_builder.main(arguments)
            self.assertEqual(list(root.iterdir()), [])

    def test_invalid_outcome_cli_rehashes_standard_stage0_bytes(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO) as directory:
            root = Path(directory)
            stage0_path = root / v3_protocol.STAGE0_REGISTRY
            stage0_path.parent.mkdir(parents=True)
            stage0_path.write_text('{"frozen":"bytes"}\n', encoding="utf-8")
            stage0_sha = v3_protocol.sha256_file(stage0_path)
            cells = {f"{g}:{v}": 4 for g, v in v3_protocol.CELL_ORDER}
            payload = {
                "protocol": v3_protocol.INVALID_OUTCOME_PROTOCOL,
                "dataset": "causal",
                "dataset_version": v3_protocol.DATASET_VERSION,
                "status": "preflight_dataset_invalid",
                "failure_phase": "selection",
                "reason_code": "global_subset_infeasible",
                "stage0_registry_sha256": stage0_sha,
                "candidate_count": 576,
                "eligible_count": 24,
                "cell_eligible_counts": cells,
                "selector_output_created": False,
                "unit_manifest_created": False,
                "stage1_registry_created": False,
                "sealed_final36_status": "unopened",
                "bound_artifacts": {
                    name: (stage0_sha if name == "stage0_registry" else "2" * 64)
                    for name in v3_protocol.INVALID_BOUND_ARTIFACT_KEYS
                },
            }
            invalid_path = root / v3_protocol.INVALID_OUTCOME
            invalid_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    v3_validator.main([
                        "invalid-outcome",
                        "--project-root", str(root),
                        "--path", str(invalid_path),
                    ]),
                    0,
                )
            payload["stage0_registry_sha256"] = "3" * 64
            payload["bound_artifacts"]["stage0_registry"] = "3" * 64
            invalid_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exact standard Stage-0"):
                v3_validator.main([
                    "invalid-outcome",
                    "--project-root", str(root),
                    "--path", str(invalid_path),
                ])

        with tempfile.TemporaryDirectory(dir=REPO) as directory:
            root = Path(directory)
            expected = root / v3_protocol.INVALID_OUTCOME
            expected.parent.mkdir(parents=True)
            target = root / "target.json"
            target.write_text("{}\n", encoding="utf-8")
            expected.symlink_to(target)
            before = sorted(item.relative_to(root).as_posix() for item in root.rglob("*"))
            with self.assertRaisesRegex(ValueError, "symlink component"):
                v3_validator.main([
                    "invalid-outcome",
                    "--project-root", str(root),
                    "--path", str(expected),
                ])
            self.assertEqual(
                before,
                sorted(item.relative_to(root).as_posix() for item in root.rglob("*")),
            )


if __name__ == "__main__":
    unittest.main()
