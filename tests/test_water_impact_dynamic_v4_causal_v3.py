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
import stat
import sys
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
    sys.modules[name] = module
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
identity_auditor = _load_v3_module(
    "audit_water_impact_dynamic_v4_v3_v2_disjointness",
    "scripts/audit_water_impact_dynamic_v4_v3_v2_disjointness.py",
)
construct_auditor = _load_v3_module(
    "audit_water_impact_dynamic_v4_v3_v2_construct_equivalence",
    "scripts/audit_water_impact_dynamic_v4_v3_v2_construct_equivalence.py",
)
forbidden_seed_auditor = _load_v3_module(
    "audit_water_impact_dynamic_v4_v3_forbidden_seeds",
    "scripts/audit_water_impact_dynamic_v4_v3_forbidden_seeds.py",
)
stage0_authorizer = _load_v3_module(
    "authorize_water_impact_dynamic_v4_causal_stage0_v3",
    "scripts/authorize_water_impact_dynamic_v4_causal_stage0_v3.py",
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
            "compared_counts": {
                "v2_candidates": 48,
                "v3_graph_edges": 576,
                "v3_fresh_sources": 48,
                "v3_fresh_receivers": 56,
                "v3_historical_receivers": 8,
                "v3_original_source_nodes": 8,
            },
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

        with tempfile.TemporaryDirectory(dir=REPO.parent) as directory:
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


class Stage0AuthorizerTests(unittest.TestCase):
    def test_missing_required_code_fails_before_binding_or_wrapper(self) -> None:
        wrapper = REPO / v3_protocol.STAGE0_REGISTRY
        self.assertFalse(wrapper.exists())
        with tempfile.TemporaryDirectory(dir=REPO.parent) as directory:
            private = Path(directory)
            private.chmod(0o700)
            binding = private / "causal_selection_binding_v3.json"
            missing_map = dict(stage0_authorizer.protocol.CODE_ARTIFACT_PATHS)
            missing_map["screening_runner"] = "scripts/synthetic_missing_runner_v3.py"
            with (
                mock.patch.object(
                    stage0_authorizer.protocol,
                    "CODE_ARTIFACT_PATHS",
                    missing_map,
                ),
                self.assertRaisesRegex(
                    FileNotFoundError, "required v3 code artifact is missing"
                ),
            ):
                stage0_authorizer.authorize(
                    project_root=REPO,
                    private_root=private,
                    pending_path=REPO / v3_protocol.STAGE0_PUBLIC,
                    binding_path=binding,
                    wrapper_path=wrapper,
                )
            self.assertFalse(binding.exists())
            self.assertFalse(wrapper.exists())

    def test_code_registry_has_exact_13_records_and_rejects_drift(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO) as directory:
            root = Path(directory)
            root.chmod(0o700)
            for name, relative in v3_protocol.CODE_ARTIFACT_PATHS.items():
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                content = f"# {name}\nVALUE = 1\n"
                if name == "generator":
                    content += (
                        "import generate_cogvideox_clean\n"
                        "import run_pilot\n"
                        "import causal_lora_activation_gate\n"
                        "import target_token_attention_suppression\n"
                    )
                path.write_text(content, encoding="utf-8")
            for relative in stage0_authorizer.GENERATOR_DEPENDENCY_PATHS:
                path = root / relative
                if not path.exists():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text("VALUE = 1\n", encoding="utf-8")
            payload = stage0_authorizer.build_code_registry_payload(root)
            self.assertEqual(len(payload["artifacts"]), 13)
            stage0_authorizer.validate_code_registry_full(payload, root)
            _, closure_before = stage0_authorizer.generator_dependency_closure(
                root
            )
            helper = root / "scripts/causal_lora_activation_gate.py"
            helper_before = helper.read_bytes()
            helper.write_bytes(helper_before + b"# dependency drift\n")
            _, closure_after = stage0_authorizer.generator_dependency_closure(
                root
            )
            self.assertNotEqual(closure_before, closure_after)
            helper.write_bytes(helper_before)
            for shadow in (
                root / "scripts/torch.py",
                root / "scripts/av.py",
                root / "scripts/__pycache__/run_pilot.cpython-311.pyc",
            ):
                shadow.parent.mkdir(parents=True, exist_ok=True)
                shadow.write_bytes(b"shadow")
                with self.assertRaisesRegex(ValueError, "shadow"):
                    stage0_authorizer.generator_dependency_closure(root)
                shadow.unlink()
            drift = root / v3_protocol.CODE_ARTIFACT_PATHS["selector"]
            drift.write_text("# drift\n", encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError, "current exact 13-file inventory"
            ):
                stage0_authorizer.validate_code_registry_full(payload, root)

    def test_pending_commitment_has_exact_30_openings(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO) as directory:
            root = Path(directory)
            root.chmod(0o700)
            for relative in (
                stage0_authorizer.PREREG_PATH,
                v3_protocol.V2_TERMINATION,
            ):
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(REPO / relative, target)
            components = {}
            for index, (name, rule) in enumerate(
                stage0_authorizer.PHYSICAL_ROW_COUNTS.items(), start=1
            ):
                rows = 1 if rule == "positive" else rule
                components[name] = {
                    "sha256": f"{index % 16:x}" * 64,
                    "size_bytes": index,
                    "row_count": rows,
                }
            self.assertEqual(len(components), 30)
            payload = {
                "protocol": stage0_authorizer.PENDING_PROTOCOL,
                "schema": stage0_authorizer.PENDING_SCHEMA,
                "registry": stage0_authorizer.PENDING_REGISTRY,
                "dataset_version": v3_protocol.DATASET_VERSION,
                "stage": 0,
                "status": "frozen_components_pending_authorization",
                "authorization_status": "not_authorized",
                "candidate_count": 576,
                "cell_counts": {
                    f"{group}:{variant}": v3_protocol.CELL_COUNTS[(group, variant)]
                    for group, variant in v3_protocol.CELL_ORDER
                },
                "sizing_rule": stage0_authorizer._expected_sizing_rule(components),
                "design_input": {
                    "preregistration": {
                        "path": stage0_authorizer.PREREG_PATH.as_posix(),
                        "sha256": stage0_authorizer.EXPECTED_PREREG_SHA256,
                    },
                    "v2_termination": {
                        "path": v3_protocol.V2_TERMINATION.as_posix(),
                        "sha256": v3_protocol.V2_RUNTIME_READ_ALLOWLIST[
                            v3_protocol.V2_TERMINATION.as_posix()
                        ],
                    },
                },
                "curation_audit": stage0_authorizer._expected_curation_audit(
                    components
                ),
                "public_metadata": stage0_authorizer._expected_public_metadata(
                    components
                ),
                "component_commitments": components,
                "remaining_blockers": [],
            }
            pending = root / v3_protocol.STAGE0_PUBLIC
            pending.parent.mkdir(parents=True, exist_ok=True)
            pending.write_text(json.dumps(payload), encoding="utf-8")
            stage0_authorizer.validate_pending(
                payload, project_root=root, pending_path=pending
            )
            payload["component_commitments"].pop(next(iter(components)))
            with self.assertRaisesRegex(ValueError, "exact 30 openings"):
                stage0_authorizer.validate_pending(
                    payload, project_root=root, pending_path=pending
                )

    def test_private_opening_inventory_is_exact_19_mode_600(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO) as directory:
            root = Path(directory)
            root.chmod(0o700)
            for name in stage0_authorizer.PRIVATE_INPUTS.values():
                path = root / name
                path.write_text("{}\n", encoding="utf-8")
                path.chmod(0o600)
            stage0_authorizer._validate_private_inventory(root)
            extra = root / "unexpected.json"
            extra.write_text("{}\n", encoding="utf-8")
            extra.chmod(0o600)
            with self.assertRaisesRegex(ValueError, "exactly the 19"):
                stage0_authorizer._validate_private_inventory(root)

    def test_secret_and_1728_seed_audit_are_fully_recomputed(self) -> None:
        secrets = {
            "protocol": stage0_authorizer.SECRETS_PROTOCOL,
            "dataset_version": v3_protocol.DATASET_VERSION,
            "status": "frozen",
            "screening_seed_namespace": v3_protocol.SCREENING_NAMESPACE,
            "screening_seed": 42,
            "graph_assignment_salt": "1" * 64,
            "selector_salt": "2" * 64,
            "evaluation_seed_namespace": v3_protocol.EVALUATION_NAMESPACE,
            "evaluation_seed_salt": "3" * 64,
        }
        commitments = stage0_authorizer._validate_secrets(
            secrets,
            graph_salt="1" * 64,
            selector_salt="2" * 64,
            evaluation_salt="3" * 64,
            screening_seed=42,
        )
        self.assertEqual(set(commitments), {
            "screening_seed",
            "graph_assignment_salt",
            "selector_salt",
            "evaluation_seed_salt",
        })
        candidates = [{"case_id": f"v4v3c{index:03d}"} for index in range(576)]
        records = [
            {
                "case_id": row["case_id"],
                "replicate": replicate,
                "seed": v3_protocol.derive_evaluation_seed(
                    "3" * 64, row["case_id"], replicate
                ),
            }
            for row in candidates
            for replicate in v3_protocol.REPLICATES
        ]
        payload = {
            "protocol": stage0_authorizer.SEED_AUDIT_PROTOCOL,
            "dataset_version": v3_protocol.DATASET_VERSION,
            "status": "passed",
            "candidate_manifest_sha256": "4" * 64,
            "evaluation_seed_salt_sha256": "5" * 64,
            "screening_seed_sha256": "6" * 64,
            "forbidden_seed_inventory_sha256": "7" * 64,
            "seed_count": 1728,
            "unique_seed_count": 1728,
            "screening_collision_count": 0,
            "forbidden_collision_count": 0,
            "ordered_seed_records_sha256": v3_protocol.sha256_bytes(
                v3_protocol.canonical_json_bytes(records)
            ),
            "records": records,
        }
        stage0_authorizer._validate_seed_audit(
            payload,
            candidates=candidates,
            evaluation_salt="3" * 64,
            screening_seed=42,
            forbidden={7},
            candidate_sha="4" * 64,
            evaluation_salt_sha="5" * 64,
            screening_seed_sha="6" * 64,
            forbidden_sha="7" * 64,
        )
        payload["records"][0]["seed"] += 1
        with self.assertRaisesRegex(ValueError, "seed audit mismatch"):
            stage0_authorizer._validate_seed_audit(
                payload,
                candidates=candidates,
                evaluation_salt="3" * 64,
                screening_seed=42,
                forbidden={7},
                candidate_sha="4" * 64,
                evaluation_salt_sha="5" * 64,
                screening_seed_sha="6" * 64,
                forbidden_sha="7" * 64,
            )

    def test_model_inventory_and_wrapper_publication_are_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO) as directory:
            root = Path(directory)
            model_root = root / "models/Wan2.1-T2V-1.3B-Diffusers"
            model_root.mkdir(parents=True)
            files = []
            for index in range(2):
                path = model_root / f"part{index}.bin"
                path.write_bytes(bytes([index + 1]))
                files.append(
                    {
                        "path": path.relative_to(root).as_posix(),
                        "sha256": v3_protocol.sha256_file(path),
                        "size_bytes": path.stat().st_size,
                    }
                )
            payload = {
                "protocol": stage0_authorizer.MODEL_INVENTORY_PROTOCOL,
                "status": "frozen",
                "dataset_version": v3_protocol.DATASET_VERSION,
                "model_root": "models/Wan2.1-T2V-1.3B-Diffusers",
                "file_count": 2,
                "files": files,
                "inventory_sha256": v3_protocol.sha256_bytes(
                    v3_protocol.canonical_json_bytes(files)
                ),
            }
            stage0_authorizer._validate_model_inventory(payload, root)
            extra = model_root / "unregistered.bin"
            extra.write_bytes(b"extra")
            with self.assertRaisesRegex(ValueError, "exact on-disk inventory"):
                stage0_authorizer._validate_model_inventory(payload, root)
            extra.unlink()
            (model_root / "part0.bin").write_bytes(b"drift")
            with self.assertRaisesRegex(ValueError, "model file byte mismatch"):
                stage0_authorizer._validate_model_inventory(payload, root)

        with tempfile.TemporaryDirectory(dir=REPO) as directory:
            path = Path(directory) / "wrapper.json"
            wrapper = {"protocol": "fixture", "status": "committed"}
            stage0_authorizer._write_public_wrapper_exclusive(path, wrapper)
            self.assertEqual(path.stat().st_mode & 0o777, 0o644)
            with self.assertRaises(FileExistsError):
                stage0_authorizer._write_public_wrapper_exclusive(path, wrapper)

        with tempfile.TemporaryDirectory(dir=REPO) as directory:
            path = Path(directory) / "wrapper.json"
            wrapper = {"protocol": "fixture", "status": "committed"}

            def partial_failure(target, value, mode):
                target.write_bytes(v3_protocol.canonical_json_bytes(value))
                target.chmod(mode)
                raise OSError("post-link failure")

            with mock.patch.object(
                stage0_authorizer.protocol,
                "write_json_exclusive_atomic",
                side_effect=partial_failure,
            ):
                with self.assertRaises(OSError):
                    stage0_authorizer._write_public_wrapper_exclusive(path, wrapper)
            self.assertTrue(path.exists())
            self.assertEqual(
                path.read_bytes(), v3_protocol.canonical_json_bytes(wrapper)
            )

    def test_capacity_and_cost_artifacts_are_exact_and_tamper_rejected(self) -> None:
        common = {
            "protocol": capacity.PROTOCOL,
            "dataset_version": capacity.DATASET_VERSION,
            "status": "frozen_capacity_model_spec",
            "numpy_version": capacity.REQUIRED_NUMPY_VERSION,
            "bit_generator": capacity.BIT_GENERATOR,
            "posterior": "M0 Beta(x+1,9-x)",
            "anchor_model": "Beta(p*kappa,(1-p)*kappa), kappa=(1-rho)/rho",
            "draw_order": (
                "per 5000-row batch: one (B,6) posterior-p call; six cell-ordered "
                "(B,A_c) theta calls; optional G1/G2/G3 frailty calls; then separate "
                "uniform calls in cell, anchor, iteration, edge order"
            ),
            "graph": stage0_authorizer._json_normalized(
                capacity.graph_specification()
            ),
            "graph_robustness": stage0_authorizer._json_normalized(
                capacity.graph_robustness_report()
            ),
            "analytic_models": stage0_authorizer._json_normalized(
                capacity.analytic_capacity_report()
            ),
            "oracle": {
                "engine": capacity.CompiledOracle.name,
                "embedded_c_source_sha256": capacity.ORACLE_C_SOURCE_SHA256,
                "self_test_against_python_reference": True,
            },
        }

        def result(profile):
            reference = capacity.REFERENCE_RESULTS[profile]
            iterations = reference["iterations"]
            failures = reference["global_failures"]
            groups = reference.get("group_failures", (0, 0, 0, failures))
            cells = reference.get("cell_shortage_failures", (0,) * 6)
            return {
                "iterations": iterations,
                "batch_size": capacity.FROZEN_BATCH_SIZE,
                "rho": reference["rho"],
                "shared_frailty": reference["shared_frailty"],
                "cell_shortage_failures": dict(zip(capacity.CELL_NAMES, cells)),
                "readiness_failure_counts": dict(
                    zip(capacity.GROUP_NAMES, (0, 0, 0, 0))
                ),
                "exact_failure_counts": dict(zip(capacity.GROUP_NAMES, groups)),
                "global_failure_rate": failures / iterations,
                "global_wilson_upper_one_sided_95": (
                    capacity.wilson_upper_one_sided_95(failures, iterations)
                ),
            }

        model = dict(common)
        search = {
            **common,
            "status": "exact_frozen_capacity_search_result",
            "profile": "search",
            "seed": capacity.seed_record(capacity.SEARCH_DOMAIN),
            "result": result("search"),
            "reference_match": True,
            "decision": {
                "search_ceiling": capacity.SEARCH_WILSON_CEILING,
                "passes": True,
                "first_lattice_point": True,
                "larger_lattice_points_inspected": 0,
            },
        }
        scenarios = {
            profile: {
                "seed": capacity.seed_record(
                    capacity.REFERENCE_RESULTS[profile]["domain"]
                ),
                "result": result(profile),
                "reference_match": True,
            }
            for profile in capacity.CONFIRMATION_PROFILE_ORDER
        }
        confirm = {
            **common,
            "status": "exact_frozen_capacity_confirmation_result",
            "profile": "combined_confirmation",
            "scenario_order": list(capacity.CONFIRMATION_PROFILE_ORDER),
            "scenarios": scenarios,
            "reference_match": True,
            "decision": {
                "authorization_scenario": "rho010",
                "confirmation_ceiling": capacity.CONFIRM_WILSON_CEILING,
                "passes": True,
                "rho020_report_only": True,
                "shared_frailty_report_only": True,
            },
        }
        graph_report = stage0_authorizer._json_normalized(
            capacity.graph_robustness_report()
        )
        stage0_authorizer._validate_capacity_artifacts(
            model, search, confirm, graph_report
        )
        confirm["decision"]["passes"] = False
        with self.assertRaisesRegex(ValueError, "decision mismatch"):
            stage0_authorizer._validate_capacity_artifacts(
                model, search, confirm, graph_report
            )

        calibration = {
            "protocol": stage0_authorizer.COST_PROTOCOL,
            "status": "passed",
            "dataset_version": v3_protocol.DATASET_VERSION,
            "hardware": {"accelerator": "A100"},
            "model_content_inventory_sha256": "1" * 64,
            "runtime_registry_sha256": "2" * 64,
            "render_configuration_sha256": "3" * 64,
            "public_prompt_sha256": [f"{index:x}" * 64 for index in range(5)],
            "wall_time_seconds": [100, 110, 120, 130, 140],
            "maximum_wall_time_seconds": 140,
            "maximum_allowed_seconds": 600,
            "candidate_count": 576,
            "gpu_hour_cap": 100,
            "passes": True,
        }
        stage0_authorizer._validate_cost_calibration(
            calibration,
            model_sha="1" * 64,
            runtime_sha="2" * 64,
            render_sha="3" * 64,
            live_hardware={"accelerator": "A100"},
        )
        calibration["maximum_wall_time_seconds"] = 141
        with self.assertRaisesRegex(ValueError, "calibration failed"):
            stage0_authorizer._validate_cost_calibration(
                calibration,
                model_sha="1" * 64,
                runtime_sha="2" * 64,
                render_sha="3" * 64,
                live_hardware={"accelerator": "A100"},
            )


def _synthetic_source_bank() -> dict[str, object]:
    entries: list[dict[str, object]] = []
    for index in range(64):
        if index < 8:
            entries.append(
                {
                    "bank_index": index,
                    "source_id": f"synthetic_original_{index}",
                    "source_phrase": f"synthetic original source original{index}",
                    "normalized_phrase": f"synthetic original source original{index}",
                    "head_lemma": f"original{index}",
                    "membership": "original_training_source",
                    "physical_audit_status": "legacy_original_source_exempt",
                }
            )
        else:
            entries.append(
                {
                    "bank_index": index,
                    "source_id": f"unused_{index}",
                    "membership": "new_bank_source",
                }
            )
    return {"entries": entries}


def _private_write(root: Path, name: str, raw: bytes) -> Path:
    path = root / name
    path.write_bytes(raw)
    path.chmod(0o600)
    return path


def _record(raw: bytes, row_count: int | None) -> dict[str, object]:
    return {
        "sha256": identity_auditor.sha256_bytes(raw),
        "size_bytes": len(raw),
        "row_count": row_count,
    }


def _synthetic_v2_candidates() -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for index in range(48):
        row: dict[str, object] = {
            "case_id": f"synthetic_v2_case_{index:03d}",
            "group": "seen_source_new_receiver",
            "prompt_variant": "direct" if index % 2 == 0 else "natural",
            "source_membership": "original_source",
            "source_id": f"synthetic_original_{index % 8}",
            "source_phrase": f"synthetic original source original{index % 8}",
            "source_head_lemma": f"original{index % 8}",
            "source_physical_audit_status": "legacy_original_source_exempt",
            "receiver_membership": "new_receiver",
            "receiver_id": f"synthetic_v2_receiver_{index:03d}",
            "receiver_phrase": f"synthetic old receiver oldreceiver{index:03d}",
            "canonical_prompt": f"synthetic old prompt {index:03d}",
        }
        row["canonical_record_sha256"] = identity_auditor.sha256_bytes(
            identity_auditor.canonical_json_bytes(row)
        )
        rows.append(row)
    return {
        "schema": "water_impact_dynamic_v4_source_slot_registry_v2",
        "protocol": "water_impact_dynamic_v4_source_slot_registry_v2",
        "dataset_version": "v4_dev72_v2",
        "stage": 0,
        "candidate_count": 48,
        "candidates": rows,
    }


def _make_isolated_audit_fixture(base: Path) -> dict[str, object]:
    project = base / "project"
    v2_root = base / "private_v2"
    v3_root = base / "private_v3"
    project.mkdir(mode=0o700)
    v2_root.mkdir(mode=0o700)
    v3_root.mkdir(mode=0o700)
    output_parent = project / "data/water_impact_dynamic_v4"
    output_parent.mkdir(parents=True)

    holdout = _holdout_fixture()
    receivers = _receiver_fixture()
    historical = _historical_fixture()
    graph, _ = v3_builder.build_candidate_graph(
        holdout_payload=holdout,
        receiver_payload=receivers,
        historical_payload=historical,
        source_bank_payload=_synthetic_source_bank(),
        graph_assignment_salt="c" * 64,
    )
    v3_payloads = {
        identity_auditor.V3_SOURCE_BASENAME: holdout,
        identity_auditor.V3_RECEIVER_BASENAME: receivers,
        identity_auditor.V3_HISTORICAL_BASENAME: historical,
        identity_auditor.V3_GRAPH_BASENAME: graph,
    }
    for name, payload in v3_payloads.items():
        _private_write(
            v3_root, name, identity_auditor.canonical_json_bytes(payload)
        )

    v2_candidates_raw = identity_auditor.canonical_json_bytes(
        _synthetic_v2_candidates()
    )
    _private_write(
        v2_root, identity_auditor.V2_CANDIDATE_BASENAME, v2_candidates_raw
    )

    templates = {"prompt_templates": {"direct": "fixed direct", "natural": "fixed natural"}}
    field_rules = {"normalization": "fixed canonical normalization"}
    v2_rules = {
        "qualification": {
            "source_visibility": 2,
            "footprint_visibility_min": 1,
            "receiver_min": 1,
            "quality_min": 1,
            "causal_link": 2,
        },
        "cell_quota": "exactly four qualified cases from every cell",
        "legacy_rule": "fixed legacy subset rule",
    }
    v3_rules = {
        **v2_rules,
        "candidate_graph": "frozen 576-edge v3 graph",
    }
    construct_raw = {
        construct_auditor.V2_TEMPLATE_BASENAME: identity_auditor.canonical_json_bytes(templates),
        construct_auditor.V2_FIELD_BASENAME: identity_auditor.canonical_json_bytes(field_rules),
        construct_auditor.V2_SELECTION_BASENAME: identity_auditor.canonical_json_bytes(v2_rules),
        construct_auditor.V3_TEMPLATE_BASENAME: identity_auditor.canonical_json_bytes(templates),
        construct_auditor.V3_FIELD_BASENAME: identity_auditor.canonical_json_bytes(field_rules),
        construct_auditor.V3_SELECTION_BASENAME: identity_auditor.canonical_json_bytes(v3_rules),
    }
    for name in construct_auditor.V2_PRIVATE_ALLOWLIST:
        _private_write(v2_root, name, construct_raw[name])
    for name in construct_auditor.V3_PRIVATE_ALLOWLIST:
        _private_write(v3_root, name, construct_raw[name])

    selection_record = _record(
        construct_raw[construct_auditor.V2_SELECTION_BASENAME], None
    )
    wrapper = {
        "protocol": "water_impact_dynamic_v4_eval_commitment_registry_v2",
        "dataset": "causal",
        "dataset_version": "v4_dev72_v2",
        "stage": 0,
        "status": "committed",
        "sealed_final36_status": "unopened",
        "artifacts": {
            "candidate_manifest_48": _record(v2_candidates_raw, 48),
            "canonical_templates": _record(
                construct_raw[construct_auditor.V2_TEMPLATE_BASENAME], None
            ),
            "field_normalization": _record(
                construct_raw[construct_auditor.V2_FIELD_BASENAME], None
            ),
            "ranking_formula": selection_record,
            "constrained_subset_algorithm": dict(selection_record),
        },
    }
    wrapper_raw = identity_auditor.canonical_json_bytes(wrapper)
    wrapper_path = project / identity_auditor.V2_STAGE0_RELATIVE
    wrapper_path.parent.mkdir(parents=True, exist_ok=True)
    wrapper_path.write_bytes(wrapper_raw)
    wrapper_path.chmod(0o644)
    stage0_sha = identity_auditor.sha256_bytes(wrapper_raw)
    identity_contract = identity_auditor.IdentityAuditContract(
        v2_stage0_sha256=stage0_sha
    )
    construct_contract = construct_auditor.ConstructAuditContract(
        v2_stage0_sha256=stage0_sha,
        v2_template_sha256=identity_auditor.sha256_bytes(
            construct_raw[construct_auditor.V2_TEMPLATE_BASENAME]
        ),
        v2_field_rules_sha256=identity_auditor.sha256_bytes(
            construct_raw[construct_auditor.V2_FIELD_BASENAME]
        ),
        v2_selection_rules_sha256=identity_auditor.sha256_bytes(
            construct_raw[construct_auditor.V2_SELECTION_BASENAME]
        ),
    )
    return {
        "project": project,
        "v2_root": v2_root,
        "v3_root": v3_root,
        "wrapper": wrapper,
        "wrapper_path": wrapper_path,
        "identity_contract": identity_contract,
        "construct_contract": construct_contract,
    }


class IsolatedAuditorTests(unittest.TestCase):
    def test_allowed_identity_and_construct_audits_publish_aggregate_only(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO) as directory:
            fixture = _make_isolated_audit_fixture(Path(directory))
            identity, identity_sha = identity_auditor.run_identity_audit(
                project_root=fixture["project"],
                private_v2_root=fixture["v2_root"],
                private_v3_root=fixture["v3_root"],
                contract=fixture["identity_contract"],
            )
            construct, construct_sha = construct_auditor.run_construct_audit(
                project_root=fixture["project"],
                private_v2_root=fixture["v2_root"],
                private_v3_root=fixture["v3_root"],
                contract=fixture["construct_contract"],
            )
            self.assertTrue(identity_sha and construct_sha)
            self.assertEqual(set(identity["intersection_counts"].values()), {0})
            self.assertEqual(set(construct["exact_equal"].values()), {True})
            identity_auditor.validate_identity_report(
                identity, fixture["identity_contract"]
            )
            construct_auditor._validate_construct_report(
                construct, fixture["construct_contract"]
            )
            encoded = json.dumps({"identity": identity, "construct": construct})
            for forbidden in (
                "source_phrase",
                "receiver_phrase",
                "canonical_prompt",
                '"seed"',
                '"score"',
            ):
                self.assertNotIn(forbidden, encoded)

    def test_private_allowlists_reject_every_unregistered_category(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO) as directory:
            fixture = _make_isolated_audit_fixture(Path(directory))
            with identity_auditor.SecurePrivateRoot(
                fixture["v2_root"], identity_auditor.V2_PRIVATE_ALLOWLIST
            ) as root:
                for forbidden in (
                    "media.mp4",
                    "screening_review.csv",
                    "eligibility.csv",
                    "screening_seed.txt",
                    "selector_salt.txt",
                    "sealed_payload.json",
                ):
                    with self.assertRaisesRegex(PermissionError, "nonallowlisted"):
                        root.read_exact(forbidden)
            with identity_auditor.SecurePrivateRoot(
                fixture["v3_root"], identity_auditor.V3_PRIVATE_ALLOWLIST
            ) as root:
                with self.assertRaisesRegex(PermissionError, "nonallowlisted"):
                    root.read_exact("causal_stage0_secrets_private_v3.json")

    def test_wrapper_hash_commitment_row_mix_and_rebind_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO) as directory:
            fixture = _make_isolated_audit_fixture(Path(directory))
            wrong_contract = identity_auditor.IdentityAuditContract(
                v2_stage0_sha256="0" * 64
            )
            with self.assertRaisesRegex(ValueError, "wrapper hash"):
                identity_auditor.run_identity_audit(
                    project_root=fixture["project"],
                    private_v2_root=fixture["v2_root"],
                    private_v3_root=fixture["v3_root"],
                    contract=wrong_contract,
                    publish=False,
                )

        with tempfile.TemporaryDirectory(dir=REPO) as directory:
            fixture = _make_isolated_audit_fixture(Path(directory))
            candidate = fixture["v2_root"] / identity_auditor.V2_CANDIDATE_BASENAME
            candidate.write_bytes(candidate.read_bytes() + b"\n")
            candidate.chmod(0o600)
            with self.assertRaisesRegex(ValueError, "committed size"):
                identity_auditor.run_identity_audit(
                    project_root=fixture["project"],
                    private_v2_root=fixture["v2_root"],
                    private_v3_root=fixture["v3_root"],
                    contract=fixture["identity_contract"],
                    publish=False,
                )

        with tempfile.TemporaryDirectory(dir=REPO) as directory:
            fixture = _make_isolated_audit_fixture(Path(directory))
            wrapper = json.loads(
                fixture["wrapper_path"].read_text(encoding="utf-8")
            )
            wrapper["artifacts"]["candidate_manifest_48"]["row_count"] = 47
            raw = identity_auditor.canonical_json_bytes(wrapper)
            fixture["wrapper_path"].write_bytes(raw)
            contract = identity_auditor.IdentityAuditContract(
                v2_stage0_sha256=identity_auditor.sha256_bytes(raw)
            )
            with self.assertRaisesRegex(ValueError, "row count"):
                identity_auditor.run_identity_audit(
                    project_root=fixture["project"],
                    private_v2_root=fixture["v2_root"],
                    private_v3_root=fixture["v3_root"],
                    contract=contract,
                    publish=False,
                )

        with tempfile.TemporaryDirectory(dir=REPO) as directory:
            fixture = _make_isolated_audit_fixture(Path(directory))
            graph_path = (
                fixture["v3_root"] / identity_auditor.V3_GRAPH_BASENAME
            )
            graph = json.loads(graph_path.read_text(encoding="utf-8"))
            graph["edges"][0]["source_phrase"] = "synthetic rebound source"
            edge = dict(graph["edges"][0])
            edge.pop("canonical_record_sha256")
            graph["edges"][0]["canonical_record_sha256"] = (
                identity_auditor.sha256_bytes(
                    identity_auditor.canonical_json_bytes(edge)
                )
            )
            graph_base = dict(graph)
            graph_base.pop("graph_sha256")
            graph["graph_sha256"] = identity_auditor.sha256_bytes(
                identity_auditor.canonical_json_bytes(graph_base)
            )
            graph_path.write_bytes(identity_auditor.canonical_json_bytes(graph))
            graph_path.chmod(0o600)
            with self.assertRaisesRegex(ValueError, "identity is rebound"):
                identity_auditor.run_identity_audit(
                    project_root=fixture["project"],
                    private_v2_root=fixture["v2_root"],
                    private_v3_root=fixture["v3_root"],
                    contract=fixture["identity_contract"],
                    publish=False,
                )

    def test_symlink_hardlink_nested_roots_and_forbidden_outputs_fail(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO) as directory:
            fixture = _make_isolated_audit_fixture(Path(directory))
            source = fixture["v3_root"] / identity_auditor.V3_SOURCE_BASENAME
            target = fixture["v3_root"] / "source_target.json"
            source.rename(target)
            source.symlink_to(target)
            with self.assertRaisesRegex((ValueError, PermissionError), "symlink|regular"):
                identity_auditor.run_identity_audit(
                    project_root=fixture["project"],
                    private_v2_root=fixture["v2_root"],
                    private_v3_root=fixture["v3_root"],
                    contract=fixture["identity_contract"],
                    publish=False,
                )

        with tempfile.TemporaryDirectory(dir=REPO) as directory:
            fixture = _make_isolated_audit_fixture(Path(directory))
            candidate = fixture["v2_root"] / identity_auditor.V2_CANDIDATE_BASENAME
            os.link(candidate, fixture["v2_root"] / "candidate_hardlink.json")
            with self.assertRaisesRegex(PermissionError, "nlink-1"):
                identity_auditor.run_identity_audit(
                    project_root=fixture["project"],
                    private_v2_root=fixture["v2_root"],
                    private_v3_root=fixture["v3_root"],
                    contract=fixture["identity_contract"],
                    publish=False,
                )

        with tempfile.TemporaryDirectory(dir=REPO) as directory:
            outer = Path(directory)
            project = outer / "project"
            project.mkdir(mode=0o700)
            nested = project / "private_v2"
            nested.mkdir(mode=0o700)
            separate = outer / "private_v3"
            separate.mkdir(mode=0o700)
            with self.assertRaisesRegex(ValueError, "nested"):
                identity_auditor.validate_distinct_roots(project, nested, separate)

        with tempfile.TemporaryDirectory(dir=REPO) as directory:
            project = Path(directory)
            (project / "data/water_impact_dynamic_v4").mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "private-content"):
                identity_auditor.write_standard_report(
                    project, {"source_id": "forbidden"}
                )
            with self.assertRaisesRegex(ValueError, "forbidden"):
                identity_auditor.write_report_to_relative(
                    project,
                    Path("data/final36/report.json"),
                    {"status": "passed"},
                )

    def test_construct_mismatch_placeholder_and_output_overwrite_fail(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO) as directory:
            fixture = _make_isolated_audit_fixture(Path(directory))
            v3_rules = fixture["v3_root"] / construct_auditor.V3_SELECTION_BASENAME
            payload = json.loads(v3_rules.read_text(encoding="utf-8"))
            payload["qualification"]["quality_min"] = 2
            v3_rules.write_bytes(identity_auditor.canonical_json_bytes(payload))
            v3_rules.chmod(0o600)
            with self.assertRaisesRegex(ValueError, "qualification"):
                construct_auditor.run_construct_audit(
                    project_root=fixture["project"],
                    private_v2_root=fixture["v2_root"],
                    private_v3_root=fixture["v3_root"],
                    contract=fixture["construct_contract"],
                    publish=False,
                )

        with tempfile.TemporaryDirectory(dir=REPO) as directory:
            fixture = _make_isolated_audit_fixture(Path(directory))
            construct_auditor.run_construct_audit(
                project_root=fixture["project"],
                private_v2_root=fixture["v2_root"],
                private_v3_root=fixture["v3_root"],
                contract=fixture["construct_contract"],
            )
            with self.assertRaisesRegex(FileExistsError, "overwrite"):
                construct_auditor.run_construct_audit(
                    project_root=fixture["project"],
                    private_v2_root=fixture["v2_root"],
                    private_v3_root=fixture["v3_root"],
                    contract=fixture["construct_contract"],
                )

        with tempfile.TemporaryDirectory(dir=REPO) as directory:
            fixture = _make_isolated_audit_fixture(Path(directory))
            v2_rules = fixture["v2_root"] / construct_auditor.V2_SELECTION_BASENAME
            v3_rules = fixture["v3_root"] / construct_auditor.V3_SELECTION_BASENAME
            payload = json.loads(v2_rules.read_text(encoding="utf-8"))
            payload["qualification"]["label"] = "TODO"
            raw = identity_auditor.canonical_json_bytes(payload)
            v2_rules.write_bytes(raw)
            v2_rules.chmod(0o600)
            v3_payload = dict(payload)
            v3_payload["candidate_graph"] = "frozen 576-edge v3 graph"
            v3_rules.write_bytes(identity_auditor.canonical_json_bytes(v3_payload))
            v3_rules.chmod(0o600)
            wrapper = json.loads(
                fixture["wrapper_path"].read_text(encoding="utf-8")
            )
            record = _record(raw, None)
            wrapper["artifacts"]["ranking_formula"] = record
            wrapper["artifacts"]["constrained_subset_algorithm"] = dict(record)
            wrapper_raw = identity_auditor.canonical_json_bytes(wrapper)
            fixture["wrapper_path"].write_bytes(wrapper_raw)
            contract = construct_auditor.ConstructAuditContract(
                v2_stage0_sha256=identity_auditor.sha256_bytes(wrapper_raw),
                v2_template_sha256=fixture["construct_contract"].v2_template_sha256,
                v2_field_rules_sha256=fixture["construct_contract"].v2_field_rules_sha256,
                v2_selection_rules_sha256=identity_auditor.sha256_bytes(raw),
            )
            with self.assertRaisesRegex(ValueError, "placeholder"):
                construct_auditor.run_construct_audit(
                    project_root=fixture["project"],
                    private_v2_root=fixture["v2_root"],
                    private_v3_root=fixture["v3_root"],
                    contract=contract,
                    publish=False,
                )

class V3CoreContinuationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        V3CoreProtocolTests.setUpClass()
        for name in (
            "bank",
            "graph",
            "manifest",
            "raw_eligibility",
            "eligibility",
            "selected",
            "decisions",
        ):
            setattr(cls, name, getattr(V3CoreProtocolTests, name))

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


def _write_canonical_json(path: Path, payload: dict[str, object], mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(v3_protocol.canonical_json_bytes(payload))
    path.chmod(mode)


def _historical_from_frozen_mapping(mapping: dict[str, object]) -> dict[str, object]:
    pairs = sorted(
        {
            (str(row["receiver_id"]), str(row["receiver"]))
            for row in mapping["mapping"]
        }
    )
    options = [v3_builder.normalize_phrase(phrase).split() for _, phrase in pairs]

    def choose(index: int, used: frozenset[str], output: tuple[tuple[int, str], ...]):
        if len(output) == 8:
            return output
        if len(pairs) - index < 8 - len(output):
            return None
        for token in options[index]:
            if token not in used and options[index].count(token) == 1:
                result = choose(
                    index + 1,
                    used | {token},
                    output + ((index, token),),
                )
                if result is not None:
                    return result
        return choose(index + 1, used, output)

    selected = choose(0, frozenset(), ())
    assert selected is not None and len(selected) == 8
    inventory = [
        {"receiver_id": receiver_id, "receiver_phrase": receiver_phrase}
        for receiver_id, receiver_phrase in pairs
    ]
    anchors = []
    for anchor_index, (pair_index, head) in enumerate(selected):
        receiver_id, receiver_phrase = pairs[pair_index]
        anchors.append(
            {
                "anchor_id": f"g2a{anchor_index}",
                "receiver_id": receiver_id,
                "receiver_phrase": receiver_phrase,
                "normalized_phrase": v3_builder.normalize_phrase(receiver_phrase),
                "head_lemma": head,
                "historical_training_binding_sha256": v3_protocol.sha256_bytes(
                    v3_protocol.canonical_json_bytes(
                        {
                            "receiver_id": receiver_id,
                            "receiver_phrase": receiver_phrase,
                        }
                    )
                ),
            }
        )
    payload = {
        "protocol": v3_builder.HISTORICAL_PROTOCOL,
        "dataset_version": v3_protocol.DATASET_VERSION,
        "anchor_count": 8,
        "training_receiver_inventory_sha256": v3_protocol.sha256_bytes(
            v3_protocol.canonical_json_bytes(inventory)
        ),
        "v2_disjointness_commitment": "b" * 64,
        "anchors": anchors,
    }
    v3_builder.validate_historical_anchors(payload)
    return payload


def _selection_rules_fixture() -> dict[str, object]:
    return {
        "protocol": stage0_authorizer.RULES_PROTOCOL,
        "dataset_version": v3_protocol.DATASET_VERSION,
        "status": "frozen",
        "qualification": dict(stage0_authorizer.QUALIFICATION),
        "cell_quota": {
            "per_group_prompt_variant": 4,
            "selected_per_group": 8,
            "selected_total": 24,
        },
        "graph_permutation_domain": v3_protocol.GRAPH_ASSIGNMENT_DOMAIN,
        "graph_permutation_formula": stage0_authorizer.GRAPH_FORMULA,
        "ranking_domain": v3_protocol.RANK_DOMAIN,
        "ranking_formula": stage0_authorizer.RANK_FORMULA,
        "subset_algorithm": {
            "algorithm": "rank_order_greedy_include_if_exact_completion_exists",
            "groups": v3_builder.graph_topology(),
            "rank_tie_policy": "invalidate_data_version",
        },
        "evaluation_seed_domain": v3_protocol.SEED_DOMAIN,
        "evaluation_seed_formula": stage0_authorizer.SEED_FORMULA,
        "replicates": [0, 1, 2],
        "required_selected_cases": 24,
        "required_evaluation_units": 72,
    }


def _make_stage0_authorize_fixture(base: Path) -> dict[str, object]:
    project = base / "project"
    private = base / "private_v3"
    project.mkdir(mode=0o700)
    private.mkdir(mode=0o700)

    for relative in (
        stage0_authorizer.PREREG_PATH,
        v3_protocol.V2_TERMINATION,
        v3_protocol.V2_BANK,
        v3_protocol.V2_MAPPING,
    ):
        target = project / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / relative, target)
        target.chmod(0o644)

    for name, relative in v3_protocol.CODE_ARTIFACT_PATHS.items():
        target = project / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        content = f"FROZEN_V3_{name.upper()} = True\n"
        if name == "generator":
            content += (
                "import generate_cogvideox_clean\n"
                "import run_pilot\n"
                "import causal_lora_activation_gate\n"
                "import target_token_attention_suppression\n"
            )
        target.write_text(content, encoding="utf-8")
        target.chmod(0o644)
    for relative in stage0_authorizer.GENERATOR_DEPENDENCY_PATHS:
        target = project / relative
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("FROZEN_GENERATOR_DEPENDENCY = True\n", encoding="utf-8")
            target.chmod(0o644)
    code_registry = stage0_authorizer.build_code_registry_payload(project)
    _write_canonical_json(
        project / v3_protocol.CODE_REGISTRY, code_registry, 0o644
    )

    source_bank = json.loads(
        (project / v3_protocol.V2_BANK).read_text(encoding="utf-8")
    )
    mapping = json.loads(
        (project / v3_protocol.V2_MAPPING).read_text(encoding="utf-8")
    )
    holdout = _holdout_fixture()
    receiver = _receiver_fixture()
    historical = _historical_from_frozen_mapping(mapping)
    graph_salt = "1" * 64
    selector_salt = "2" * 64
    evaluation_salt = "3" * 64
    graph, candidate = v3_builder.build_candidate_graph(
        holdout_payload=holdout,
        receiver_payload=receiver,
        historical_payload=historical,
        source_bank_payload=source_bank,
        graph_assignment_salt=graph_salt,
    )

    model_root = project / "models/Wan2.1-T2V-1.3B-Diffusers"
    model_root.mkdir(parents=True)
    model_part = model_root / "synthetic-model-part.bin"
    model_part.write_bytes(b"synthetic frozen model bytes")
    model_files = [
        {
            "path": model_part.relative_to(project).as_posix(),
            "sha256": v3_protocol.sha256_file(model_part),
            "size_bytes": model_part.stat().st_size,
        }
    ]
    model = {
        "protocol": stage0_authorizer.MODEL_INVENTORY_PROTOCOL,
        "status": "frozen",
        "dataset_version": v3_protocol.DATASET_VERSION,
        "model_root": "models/Wan2.1-T2V-1.3B-Diffusers",
        "file_count": 1,
        "files": model_files,
        "inventory_sha256": v3_protocol.sha256_bytes(
            v3_protocol.canonical_json_bytes(model_files)
        ),
    }
    _write_canonical_json(
        project / stage0_authorizer.MODEL_INVENTORY_PATH, model, 0o644
    )
    model_sha = model["inventory_sha256"]

    runtime = {"fixture": "frozen runtime registry bytes"}
    _write_canonical_json(
        project / stage0_authorizer.RUNTIME_REGISTRY_PATH, runtime, 0o644
    )
    runtime_sha = v3_protocol.sha256_file(
        project / stage0_authorizer.RUNTIME_REGISTRY_PATH
    )
    hardware = {
        "accelerator_type": "CUDA",
        "device_count": 1,
        "device_models": ["Synthetic NVIDIA A100"],
    }
    media_runtime_packages = {"av": "synthetic-av", "Pillow": "synthetic-pillow"}
    _, generator_dependency_closure_sha256 = (
        stage0_authorizer.generator_dependency_closure(project)
    )

    templates = {
        "prompt_templates": {
            variant: v3_builder.factual_prompt(
                "{source_phrase}", "{receiver_phrase}", variant
            )
            for variant in v3_protocol.PROMPT_VARIANTS
        },
        "template_fill_rules": {
            "direct": {
                "source_phrase": "python_str_capitalize",
                "receiver_phrase": "identity",
            },
            "natural": {
                "source_phrase": "identity",
                "receiver_phrase": "identity",
            },
        },
    }
    field_rules = {"normalization": "canonical ascii fixture rules"}
    rules = _selection_rules_fixture()
    render = {
        "protocol": stage0_authorizer.RENDER_PROTOCOL,
        "dataset_version": v3_protocol.DATASET_VERSION,
        "status": "frozen",
        "arm": "Original_only",
        "model_family": "Wan 2.1 T2V 1.3B",
        "model_content_inventory_sha256": model_sha,
        "steps": 25,
        "cfg": 5,
        "frames": 49,
        "width": 832,
        "height": 480,
        "fps": 8,
        "dtype": "bf16",
        "adapter": None,
        "screening_scope": "all 49 frames for every candidate",
    }

    private_json = {
        "candidate_manifest_576": candidate,
        "eval_holdout_source_ontology_48": holdout,
        "holdout_registry_48": {
            "protocol": "water_impact_dynamic_v4_holdout_registry_v3",
            "dataset_version": v3_protocol.DATASET_VERSION,
            "holdout_count": 48,
            "status": "frozen",
            "ordered_entries_sha256": v3_protocol.sha256_bytes(
                v3_protocol.canonical_json_bytes(holdout["sources"])
            ),
            "entries": holdout["sources"],
        },
        "receiver_ontology_56": receiver,
        "historical_receiver_anchors_8": historical,
        "candidate_graph_576": graph,
        "canonical_templates": templates,
        "field_normalization": field_rules,
        "raw_render_configuration": render,
        "stage0_secrets": {
            "protocol": stage0_authorizer.SECRETS_PROTOCOL,
            "dataset_version": v3_protocol.DATASET_VERSION,
            "status": "frozen",
            "screening_seed_namespace": v3_protocol.SCREENING_NAMESPACE,
            "screening_seed": 4_000_000_000,
            "graph_assignment_salt": graph_salt,
            "selector_salt": selector_salt,
            "evaluation_seed_namespace": v3_protocol.EVALUATION_NAMESPACE,
            "evaluation_seed_salt": evaluation_salt,
        },
        "selection_rules": rules,
    }
    for name, payload in private_json.items():
        _write_canonical_json(
            private / stage0_authorizer.PRIVATE_INPUTS[name], payload, 0o600
        )
    for name, value in (
        ("screening_seed", "4000000000\n"),
        ("graph_assignment_salt", graph_salt + "\n"),
        ("selector_salt", selector_salt + "\n"),
        ("evaluation_seed_salt", evaluation_salt + "\n"),
    ):
        path = private / stage0_authorizer.PRIVATE_INPUTS[name]
        path.write_text(value, encoding="ascii")
        path.chmod(0o600)

    forbidden = {
        "protocol": v3_selector.FORBIDDEN_PROTOCOL,
        "dataset": v3_protocol.DATASET,
        "status": "frozen_by_independent_seed_auditor",
        "seed_encoding": "nonnegative JSON integer below 2^63",
        "source_commitments": [
            {
                "name": "historical_registered_seed_union",
                "sha256": "4" * 64,
                "seed_count": 1,
            }
        ],
        "seeds": [2**62],
    }
    forbidden_path = private / stage0_authorizer.PRIVATE_INPUTS[
        "forbidden_seed_inventory"
    ]
    _write_canonical_json(forbidden_path, forbidden, 0o600)
    forbidden_audit = {
        "protocol": v3_protocol.FORBIDDEN_SEED_SOURCE_AUDIT_PROTOCOL,
        "status": "passed",
        "dataset_version": v3_protocol.DATASET_VERSION,
        "v2_stage0_registry_sha256": v3_protocol.V2_STAGE0_SHA256,
        "v2_forbidden_seed_inventory_sha256": (
            stage0_authorizer.V2_FORBIDDEN_SEED_INVENTORY_SHA256
        ),
        "v3_forbidden_seed_inventory_sha256": v3_protocol.sha256_file(
            forbidden_path
        ),
        "v2_seed_count": 1,
        "v3_seed_count": 1,
        "intersection_seed_count": 1,
        "v2_missing_from_v3_count": 0,
        "v3_additional_seed_count": 0,
        "set_relation": "equal",
    }
    _write_canonical_json(
        project / v3_protocol.FORBIDDEN_SEED_SOURCE_AUDIT,
        forbidden_audit,
        0o644,
    )

    seed_records = [
        {
            "case_id": row["case_id"],
            "replicate": replicate,
            "seed": v3_protocol.derive_evaluation_seed(
                evaluation_salt, row["case_id"], replicate
            ),
        }
        for row in candidate["candidates"]
        for replicate in v3_protocol.REPLICATES
    ]
    assert 4_000_000_000 not in {row["seed"] for row in seed_records}
    seed_audit = {
        "protocol": stage0_authorizer.SEED_AUDIT_PROTOCOL,
        "dataset_version": v3_protocol.DATASET_VERSION,
        "status": "passed",
        "candidate_manifest_sha256": v3_protocol.sha256_file(
            private / stage0_authorizer.PRIVATE_INPUTS["candidate_manifest_576"]
        ),
        "evaluation_seed_salt_sha256": v3_protocol.sha256_file(
            private / stage0_authorizer.PRIVATE_INPUTS["evaluation_seed_salt"]
        ),
        "screening_seed_sha256": v3_protocol.sha256_file(
            private / stage0_authorizer.PRIVATE_INPUTS["screening_seed"]
        ),
        "forbidden_seed_inventory_sha256": v3_protocol.sha256_file(
            forbidden_path
        ),
        "seed_count": 1728,
        "unique_seed_count": 1728,
        "screening_collision_count": 0,
        "forbidden_collision_count": 0,
        "ordered_seed_records_sha256": v3_protocol.sha256_bytes(
            v3_protocol.canonical_json_bytes(seed_records)
        ),
        "records": seed_records,
    }
    _write_canonical_json(
        private
        / stage0_authorizer.PRIVATE_INPUTS["preselection_seed_audit_1728"],
        seed_audit,
        0o600,
    )

    render_path = private / stage0_authorizer.PRIVATE_INPUTS[
        "raw_render_configuration"
    ]
    generation_spec = {
        "protocol": stage0_authorizer.GENERATION_SPEC_PROTOCOL,
        "dataset_version": v3_protocol.DATASET_VERSION,
        "status": "frozen_before_original_screening",
        "candidate_manifest_sha256": seed_audit["candidate_manifest_sha256"],
        "candidate_graph_sha256": v3_protocol.sha256_file(
            private / stage0_authorizer.PRIVATE_INPUTS["candidate_graph_576"]
        ),
        "render_configuration_sha256": v3_protocol.sha256_file(render_path),
        "screening_seed_sha256": seed_audit["screening_seed_sha256"],
        "graph_assignment_salt_sha256": v3_protocol.sha256_file(
            private / stage0_authorizer.PRIVATE_INPUTS["graph_assignment_salt"]
        ),
        "model_content_inventory_sha256": model_sha,
        "runtime_registry_sha256": runtime_sha,
        "generator_dependency_closure_sha256": (
            generator_dependency_closure_sha256
        ),
        "media_runtime_packages": media_runtime_packages,
        "candidate_count": 576,
        "generation": {
            "steps": 25,
            "cfg": 5,
            "frames": 49,
            "width": 832,
            "height": 480,
            "fps": 8,
            "dtype": "bf16",
            "adapter": None,
            "skip_existing": False,
            "resume": False,
            "worker_count": 1,
        },
    }
    _write_canonical_json(
        private / stage0_authorizer.PRIVATE_INPUTS["screening_generation_spec"],
        generation_spec,
        0o600,
    )

    private_records = {
        name: stage0_authorizer._file_record(
            private / basename,
            None,
        )
        for name, basename in stage0_authorizer.PRIVATE_INPUTS.items()
        if name != "raw_root_bundle"
    }
    bundle = {
        "protocol": stage0_authorizer.BUNDLE_PROTOCOL,
        "dataset_version": v3_protocol.DATASET_VERSION,
        "status": "frozen_before_pending_commitment",
        "components": {
            name: record["sha256"] for name, record in private_records.items()
        },
    }
    _write_canonical_json(
        private / stage0_authorizer.PRIVATE_INPUTS["raw_root_bundle"],
        bundle,
        0o600,
    )

    identity_bundle_sha = v3_protocol.sha256_bytes(
        v3_protocol.canonical_json_bytes(
            {
                stage0_authorizer.PRIVATE_INPUTS[
                    "eval_holdout_source_ontology_48"
                ]: v3_protocol.sha256_file(
                    private
                    / stage0_authorizer.PRIVATE_INPUTS[
                        "eval_holdout_source_ontology_48"
                    ]
                ),
                stage0_authorizer.PRIVATE_INPUTS[
                    "receiver_ontology_56"
                ]: v3_protocol.sha256_file(
                    private
                    / stage0_authorizer.PRIVATE_INPUTS["receiver_ontology_56"]
                ),
                stage0_authorizer.PRIVATE_INPUTS[
                    "historical_receiver_anchors_8"
                ]: v3_protocol.sha256_file(
                    private
                    / stage0_authorizer.PRIVATE_INPUTS[
                        "historical_receiver_anchors_8"
                    ]
                ),
            }
        )
    )
    identity = {
        "protocol": v3_protocol.IDENTITY_REPORT_PROTOCOL,
        "status": "passed",
        "dataset_version": v3_protocol.DATASET_VERSION,
        "v2_stage0_registry_sha256": v3_protocol.V2_STAGE0_SHA256,
        "v2_candidate_manifest_sha256": "5" * 64,
        "v3_candidate_graph_sha256": v3_protocol.sha256_file(
            private / stage0_authorizer.PRIVATE_INPUTS["candidate_graph_576"]
        ),
        "v3_ontology_bundle_sha256": identity_bundle_sha,
        "compared_counts": {
            "v2_candidates": 48,
            "v3_graph_edges": 576,
            "v3_fresh_sources": 48,
            "v3_fresh_receivers": 56,
            "v3_historical_receivers": 8,
            "v3_original_source_nodes": 8,
        },
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
    _write_canonical_json(project / v3_protocol.IDENTITY_REPORT, identity, 0o644)

    template_sha = v3_protocol.sha256_file(
        private / stage0_authorizer.PRIVATE_INPUTS["canonical_templates"]
    )
    field_sha = v3_protocol.sha256_file(
        private / stage0_authorizer.PRIVATE_INPUTS["field_normalization"]
    )
    rules_sha = v3_protocol.sha256_file(
        private / stage0_authorizer.PRIVATE_INPUTS["selection_rules"]
    )
    qualification_sha = v3_protocol.sha256_bytes(
        v3_protocol.canonical_json_bytes(rules["qualification"])
    )
    quota_sha = v3_protocol.sha256_bytes(
        v3_protocol.canonical_json_bytes(rules["cell_quota"])
    )
    construct = {
        "protocol": v3_protocol.CONSTRUCT_REPORT_PROTOCOL,
        "status": "passed",
        "dataset_version": v3_protocol.DATASET_VERSION,
        "v2_stage0_registry_sha256": v3_protocol.V2_STAGE0_SHA256,
        "v2_file_sha256": {
            "templates": template_sha,
            "field_rules": field_sha,
            "selection_rules": rules_sha,
        },
        "v3_file_sha256": {
            "templates": template_sha,
            "field_rules": field_sha,
            "selection_rules": rules_sha,
        },
        "qualification_sha256": {
            "v2": qualification_sha,
            "v3": qualification_sha,
        },
        "cell_quota_sha256": {"v2": quota_sha, "v3": quota_sha},
        "exact_equal": {
            "templates": True,
            "field_rules": True,
            "qualification": True,
            "cell_quota": True,
        },
    }
    _write_canonical_json(project / v3_protocol.CONSTRUCT_REPORT, construct, 0o644)

    for path in (
        stage0_authorizer.CAPACITY_MODEL_PATH,
        stage0_authorizer.CAPACITY_SEARCH_PATH,
        stage0_authorizer.CAPACITY_CONFIRM_PATH,
        stage0_authorizer.STATIC_GRAPH_PATH,
    ):
        _write_canonical_json(project / path, {"fixture": path.name}, 0o644)
    cost = {
        "protocol": stage0_authorizer.COST_PROTOCOL,
        "status": "passed",
        "dataset_version": v3_protocol.DATASET_VERSION,
        "hardware": hardware,
        "model_content_inventory_sha256": model_sha,
        "runtime_registry_sha256": runtime_sha,
        "render_configuration_sha256": v3_protocol.sha256_file(render_path),
        "public_prompt_sha256": [f"{index + 6:x}" * 64 for index in range(5)],
        "wall_time_seconds": [100, 110, 120, 130, 140],
        "maximum_wall_time_seconds": 140,
        "maximum_allowed_seconds": 600,
        "candidate_count": 576,
        "gpu_hour_cap": 100,
        "passes": True,
    }
    _write_canonical_json(
        project / stage0_authorizer.COST_CALIBRATION_PATH, cost, 0o644
    )

    opening_paths = stage0_authorizer._opening_paths(project, private)
    opening_records = stage0_authorizer._records_for_openings(opening_paths)
    secret_commitments = stage0_authorizer._validate_secrets(
        private_json["stage0_secrets"],
        graph_salt=graph_salt,
        selector_salt=selector_salt,
        evaluation_salt=evaluation_salt,
        screening_seed=4_000_000_000,
    )
    pending = {
        "protocol": stage0_authorizer.PENDING_PROTOCOL,
        "schema": stage0_authorizer.PENDING_SCHEMA,
        "registry": stage0_authorizer.PENDING_REGISTRY,
        "dataset_version": v3_protocol.DATASET_VERSION,
        "stage": 0,
        "status": "frozen_components_pending_authorization",
        "authorization_status": "not_authorized",
        "candidate_count": 576,
        "cell_counts": {
            f"{group}:{variant}": v3_protocol.CELL_COUNTS[(group, variant)]
            for group, variant in v3_protocol.CELL_ORDER
        },
        "sizing_rule": stage0_authorizer._expected_sizing_rule(opening_records),
        "design_input": {
            "preregistration": {
                "path": stage0_authorizer.PREREG_PATH.as_posix(),
                "sha256": stage0_authorizer.EXPECTED_PREREG_SHA256,
            },
            "v2_termination": {
                "path": v3_protocol.V2_TERMINATION.as_posix(),
                "sha256": v3_protocol.V2_RUNTIME_READ_ALLOWLIST[
                    v3_protocol.V2_TERMINATION.as_posix()
                ],
            },
        },
        "curation_audit": stage0_authorizer._expected_curation_audit(
            opening_records
        ),
        "public_metadata": stage0_authorizer._expected_public_metadata(
            opening_records, secret_commitments
        ),
        "component_commitments": opening_records,
        "remaining_blockers": [],
    }
    pending_path = project / v3_protocol.STAGE0_PUBLIC
    _write_canonical_json(pending_path, pending, 0o644)
    return {
        "project": project,
        "private": private,
        "pending": pending_path,
        "binding": private / "causal_selection_binding_v3.json",
        "wrapper": project / v3_protocol.STAGE0_REGISTRY,
        "candidate": private
        / stage0_authorizer.PRIVATE_INPUTS["candidate_manifest_576"],
        "hardware": hardware,
        "template_sha": template_sha,
        "field_sha": field_sha,
        "rules_sha": rules_sha,
        "generator_dependency_closure_sha256": (
            generator_dependency_closure_sha256
        ),
        "media_runtime_packages": media_runtime_packages,
    }


class Stage0AuthorizerA2Tests(unittest.TestCase):
    def _patches(self, fixture: dict[str, object]):
        return (
            mock.patch.object(
                stage0_authorizer,
                "_validate_runtime_registry",
                return_value=fixture["hardware"],
            ),
            mock.patch.object(stage0_authorizer, "_validate_capacity_artifacts"),
            mock.patch.object(
                stage0_authorizer,
                "probe_media_runtime_packages",
                return_value=fixture["media_runtime_packages"],
            ),
            mock.patch.multiple(
                v3_protocol,
                V2_TEMPLATE_SHA256=fixture["template_sha"],
                V2_FIELD_RULES_SHA256=fixture["field_sha"],
                V2_SELECTION_RULES_SHA256=fixture["rules_sha"],
            ),
            mock.patch.multiple(
                stage0_authorizer.protocol,
                V2_TEMPLATE_SHA256=fixture["template_sha"],
                V2_FIELD_RULES_SHA256=fixture["field_sha"],
                V2_SELECTION_RULES_SHA256=fixture["rules_sha"],
            ),
            mock.patch.multiple(
                stage0_authorizer.builder.protocol,
                V2_TEMPLATE_SHA256=fixture["template_sha"],
                V2_FIELD_RULES_SHA256=fixture["field_sha"],
                V2_SELECTION_RULES_SHA256=fixture["rules_sha"],
            ),
        )

    def _authorize(self, fixture: dict[str, object]):
        patches = self._patches(fixture)
        with (
            patches[0] as runtime_mock,
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patches[5],
        ):
            wrapper = stage0_authorizer.authorize(
                project_root=fixture["project"],
                private_root=fixture["private"],
                pending_path=fixture["pending"],
                binding_path=fixture["binding"],
                wrapper_path=fixture["wrapper"],
            )
        return wrapper, runtime_mock

    def test_full_synthetic_authorization_reopens_mapping_reports_and_37_artifacts(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO) as directory:
            fixture = _make_stage0_authorize_fixture(Path(directory))
            wrapper, runtime_mock = self._authorize(fixture)
            self.assertEqual(runtime_mock.call_count, 4)
            self.assertEqual(len(wrapper["artifacts"]), 37)
            v3_protocol.validate_commitment_registry(wrapper, stage=0)
            self.assertEqual(
                json.loads(fixture["wrapper"].read_text(encoding="utf-8")),
                wrapper,
            )
            self.assertEqual(fixture["binding"].stat().st_mode & 0o777, 0o600)
            self.assertEqual(fixture["wrapper"].stat().st_mode & 0o777, 0o644)

    def test_after_binding_opening_drift_rolls_back_both_outputs(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO) as directory:
            fixture = _make_stage0_authorize_fixture(Path(directory))
            real_write = stage0_authorizer.protocol.write_json_exclusive_atomic

            def write_then_drift(path, payload, *, mode=0o600):
                result = real_write(path, payload, mode=mode)
                if path == fixture["binding"]:
                    fixture["candidate"].write_bytes(
                        fixture["candidate"].read_bytes() + b" "
                    )
                return result

            patches = self._patches(fixture)
            with (
                patches[0],
                patches[1],
                patches[2],
                patches[3],
                patches[4],
                patches[5],
                mock.patch.object(
                    stage0_authorizer.protocol,
                    "write_json_exclusive_atomic",
                    side_effect=write_then_drift,
                ),
                self.assertRaisesRegex(ValueError, "opening bytes changed"),
            ):
                stage0_authorizer.authorize(
                    project_root=fixture["project"],
                    private_root=fixture["private"],
                    pending_path=fixture["pending"],
                    binding_path=fixture["binding"],
                    wrapper_path=fixture["wrapper"],
                )
            self.assertFalse(fixture["binding"].exists())
            self.assertFalse(fixture["wrapper"].exists())
            invalid = json.loads(
                (fixture["project"] / v3_protocol.INVALID_OUTCOME).read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(invalid["failure_phase"], "stage0_authorization")
            self.assertIsNone(invalid["stage0_registry_sha256"])

    def test_wrapper_post_link_failure_rolls_back_binding_and_wrapper(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO) as directory:
            fixture = _make_stage0_authorize_fixture(Path(directory))
            real_write = stage0_authorizer.protocol.write_json_exclusive_atomic

            def fail_wrapper_after_link(path, payload, *, mode=0o600):
                result = real_write(path, payload, mode=mode)
                if path == fixture["wrapper"]:
                    raise OSError("synthetic post-link publication failure")
                return result

            patches = self._patches(fixture)
            with (
                patches[0],
                patches[1],
                patches[2],
                patches[3],
                patches[4],
                patches[5],
                mock.patch.object(
                    stage0_authorizer.protocol,
                    "write_json_exclusive_atomic",
                    side_effect=fail_wrapper_after_link,
                ),
                self.assertRaisesRegex(OSError, "post-link publication"),
            ):
                stage0_authorizer.authorize(
                    project_root=fixture["project"],
                    private_root=fixture["private"],
                    pending_path=fixture["pending"],
                    binding_path=fixture["binding"],
                    wrapper_path=fixture["wrapper"],
                )
            self.assertTrue(fixture["binding"].exists())
            self.assertTrue(fixture["wrapper"].exists())
            invalid = json.loads(
                (fixture["project"] / v3_protocol.INVALID_OUTCOME).read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(invalid["failure_phase"], "original_generation")
            self.assertEqual(
                invalid["stage0_registry_sha256"],
                v3_protocol.sha256_file(fixture["wrapper"]),
            )

    def test_wrapper_prelink_static_failure_is_nonterminal_and_retryable(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO) as directory:
            fixture = _make_stage0_authorize_fixture(Path(directory))
            patches = self._patches(fixture)
            with (
                patches[0],
                patches[1],
                patches[2],
                patches[3],
                patches[4],
                patches[5],
                mock.patch.object(
                    stage0_authorizer,
                    "_write_public_wrapper_exclusive",
                    side_effect=OSError("synthetic pre-link filesystem failure"),
                ),
                self.assertRaisesRegex(
                    stage0_authorizer.StaticAuthorizationFailure,
                    "before boundary",
                ),
            ):
                stage0_authorizer.authorize(
                    project_root=fixture["project"],
                    private_root=fixture["private"],
                    pending_path=fixture["pending"],
                    binding_path=fixture["binding"],
                    wrapper_path=fixture["wrapper"],
                )
            self.assertFalse(fixture["binding"].exists())
            self.assertFalse(fixture["wrapper"].exists())
            self.assertFalse(
                (fixture["project"] / v3_protocol.INVALID_OUTCOME).exists()
            )

    def test_pending_and_prereg_are_rehashed_after_binding(self) -> None:
        for drift_name in ("pending", "prereg"):
            with self.subTest(drift_name=drift_name), tempfile.TemporaryDirectory(
                dir=REPO
            ) as directory:
                fixture = _make_stage0_authorize_fixture(Path(directory))
                target = (
                    fixture["pending"]
                    if drift_name == "pending"
                    else fixture["project"] / stage0_authorizer.PREREG_PATH
                )
                real_write = stage0_authorizer.protocol.write_json_exclusive_atomic

                def write_then_drift(path, payload, *, mode=0o600):
                    result = real_write(path, payload, mode=mode)
                    if path == fixture["binding"]:
                        target.write_bytes(target.read_bytes() + b" ")
                    return result

                patches = self._patches(fixture)
                with (
                    patches[0],
                    patches[1],
                    patches[2],
                    patches[3],
                    patches[4],
                    patches[5],
                    mock.patch.object(
                        stage0_authorizer.protocol,
                        "write_json_exclusive_atomic",
                        side_effect=write_then_drift,
                    ),
                    self.assertRaisesRegex(
                        ValueError,
                        "pending Stage-0 bytes changed|preregistration bytes changed",
                    ),
                ):
                    stage0_authorizer.authorize(
                        project_root=fixture["project"],
                        private_root=fixture["private"],
                        pending_path=fixture["pending"],
                        binding_path=fixture["binding"],
                        wrapper_path=fixture["wrapper"],
                    )
                self.assertFalse(fixture["binding"].exists())
                self.assertFalse(fixture["wrapper"].exists())

    def test_mapping_derived_historical_inventory_rejects_every_rebind(self) -> None:
        mapping = json.loads(
            (REPO / v3_protocol.V2_MAPPING).read_text(encoding="utf-8")
        )
        historical = _historical_from_frozen_mapping(mapping)
        inventory, digest = stage0_authorizer._historical_receiver_inventory(
            mapping, historical
        )
        self.assertEqual(len(inventory), 12)
        self.assertEqual(digest, historical["training_receiver_inventory_sha256"])

        wrong_top = json.loads(json.dumps(historical))
        wrong_top["training_receiver_inventory_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "inventory hash"):
            stage0_authorizer._historical_receiver_inventory(mapping, wrong_top)
        wrong_pair = json.loads(json.dumps(historical))
        wrong_pair["anchors"][0]["receiver_phrase"] += " rebound"
        with self.assertRaisesRegex(ValueError, "absent from mapping"):
            stage0_authorizer._historical_receiver_inventory(mapping, wrong_pair)
        wrong_binding = json.loads(json.dumps(historical))
        wrong_binding["anchors"][0]["historical_training_binding_sha256"] = (
            "0" * 64
        )
        with self.assertRaisesRegex(ValueError, "row binding"):
            stage0_authorizer._historical_receiver_inventory(
                mapping, wrong_binding
            )

    def test_pending_sizing_curation_and_public_metadata_are_exact(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO) as directory:
            fixture = _make_stage0_authorize_fixture(Path(directory))
            original = json.loads(fixture["pending"].read_text(encoding="utf-8"))
            cases = (
                ("sizing_rule", "candidate_count", 575, "sizing_rule"),
                ("curation_audit", "receiver_count", 55, "curation_audit"),
                (
                    "public_metadata",
                    "forbidden_seed_source_audit_sha256",
                    "0" * 64,
                    "public_metadata",
                ),
            )
            for section, key, value, message in cases:
                with self.subTest(section=section):
                    payload = json.loads(json.dumps(original))
                    payload[section][key] = value
                    with self.assertRaisesRegex(ValueError, message):
                        stage0_authorizer.validate_pending(
                            payload,
                            project_root=fixture["project"],
                            pending_path=fixture["pending"],
                        )

    def test_forbidden_source_report_and_screening_collision_are_rejected(self) -> None:
        report = {
            "protocol": v3_protocol.FORBIDDEN_SEED_SOURCE_AUDIT_PROTOCOL,
            "status": "passed",
            "dataset_version": v3_protocol.DATASET_VERSION,
            "v2_stage0_registry_sha256": v3_protocol.V2_STAGE0_SHA256,
            "v2_forbidden_seed_inventory_sha256": (
                stage0_authorizer.V2_FORBIDDEN_SEED_INVENTORY_SHA256
            ),
            "v3_forbidden_seed_inventory_sha256": "1" * 64,
            "v2_seed_count": 2,
            "v3_seed_count": 3,
            "intersection_seed_count": 2,
            "v2_missing_from_v3_count": 0,
            "v3_additional_seed_count": 1,
            "set_relation": "strict_superset",
        }
        stage0_authorizer._validate_forbidden_seed_source_audit(
            report, v3_inventory_sha256="1" * 64, v3_seed_count=3
        )
        report["v2_missing_from_v3_count"] = 1
        with self.assertRaisesRegex(ValueError, "does not prove v2 coverage"):
            stage0_authorizer._validate_forbidden_seed_source_audit(
                report, v3_inventory_sha256="1" * 64, v3_seed_count=3
            )

        candidates = [{"case_id": f"v4v3c{index:03d}"} for index in range(576)]
        records = [
            {
                "case_id": row["case_id"],
                "replicate": replicate,
                "seed": v3_protocol.derive_evaluation_seed(
                    "3" * 64, row["case_id"], replicate
                ),
            }
            for row in candidates
            for replicate in v3_protocol.REPLICATES
        ]
        audit = {
            "protocol": stage0_authorizer.SEED_AUDIT_PROTOCOL,
            "dataset_version": v3_protocol.DATASET_VERSION,
            "status": "passed",
            "candidate_manifest_sha256": "4" * 64,
            "evaluation_seed_salt_sha256": "5" * 64,
            "screening_seed_sha256": "6" * 64,
            "forbidden_seed_inventory_sha256": "7" * 64,
            "seed_count": 1728,
            "unique_seed_count": 1728,
            "screening_collision_count": 0,
            "forbidden_collision_count": 0,
            "ordered_seed_records_sha256": v3_protocol.sha256_bytes(
                v3_protocol.canonical_json_bytes(records)
            ),
            "records": records,
        }
        with self.assertRaisesRegex(ValueError, "seed audit mismatch"):
            stage0_authorizer._validate_seed_audit(
                audit,
                candidates=candidates,
                evaluation_salt="3" * 64,
                screening_seed=42,
                forbidden={42},
                candidate_sha="4" * 64,
                evaluation_salt_sha="5" * 64,
                screening_seed_sha="6" * 64,
                forbidden_sha="7" * 64,
            )

    def test_runtime_child_probe_and_cost_hardware_are_exact(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO) as directory:
            root = Path(directory)
            runtime_root = root / "models/.wan-runtime"
            executable = runtime_root / "bin/python"
            executable.parent.mkdir(parents=True)
            executable.write_bytes(b"synthetic executable")
            executable.chmod(0o755)
            packages = dict(stage0_authorizer.EXPECTED_RUNTIME_PACKAGE_VERSIONS)
            cuda = {
                "available_required": True,
                "torch_cuda_version": "12.4",
                "cudnn_version": 90100,
                "device_count": 1,
                "device_models": ["Synthetic NVIDIA A100"],
            }
            payload = {
                "protocol": stage0_authorizer.RUNTIME_REGISTRY_PROTOCOL,
                "status": "frozen",
                "dataset_version": v3_protocol.DATASET_VERSION,
                "runtime_root": "models/.wan-runtime",
                "python_executable": "models/.wan-runtime/bin/python",
                "sys_prefix_policy": "realpath(sys.prefix)==realpath(runtime_root)",
                "python": {"implementation": "CPython", "version": "3.11.15"},
                "torch": {
                    "distribution_version": "2.6.0",
                    "module_version": "2.6.0+cu124",
                },
                "cuda": cuda,
                "packages": packages,
            }
            observed = {
                "executable_realpath": os.path.realpath(executable),
                "prefix_realpath": os.path.realpath(runtime_root),
                "python": payload["python"],
                "torch": payload["torch"],
                "cuda": cuda,
                "packages": packages,
            }
            completed = mock.Mock(
                returncode=0,
                stdout=json.dumps(observed),
                stderr="",
            )
            with mock.patch.object(
                stage0_authorizer.subprocess, "run", return_value=completed
            ):
                hardware = stage0_authorizer._validate_runtime_registry(
                    payload, root
                )
            self.assertEqual(
                hardware,
                {
                    "accelerator_type": "CUDA",
                    "device_count": 1,
                    "device_models": ["Synthetic NVIDIA A100"],
                },
            )
            observed["executable_realpath"] = os.path.realpath(root / "wrong")
            completed.stdout = json.dumps(observed)
            with (
                mock.patch.object(
                    stage0_authorizer.subprocess, "run", return_value=completed
                ),
                self.assertRaisesRegex(ValueError, "interpreter/prefix"),
            ):
                stage0_authorizer._validate_runtime_registry(payload, root)
            tampered = json.loads(json.dumps(payload))
            tampered["packages"]["torch"] = "2.6.1"
            with self.assertRaisesRegex(ValueError, "package inventory"):
                stage0_authorizer._validate_runtime_registry(tampered, root)

    def test_receiver_and_historical_head_must_be_one_whole_token(self) -> None:
        receiver = _receiver_fixture()
        row = receiver["receivers"][0]
        head = row["head_lemma"]
        row["receiver_phrase"] = f"bounded {head} with still water unobstructed rim"
        row["normalized_phrase"] = row["receiver_phrase"]
        row["curator_note"] = f"distinct receiver {head} identity"
        v3_builder.validate_receiver_ontology(receiver)

        absent = json.loads(json.dumps(receiver))
        absent["receivers"][0]["head_lemma"] = "absenthead"
        with self.assertRaisesRegex(ValueError, "exactly once"):
            v3_builder.validate_receiver_ontology(absent)
        duplicate = json.loads(json.dumps(receiver))
        duplicate["receivers"][0]["receiver_phrase"] += f" {head}"
        duplicate["receivers"][0]["normalized_phrase"] += f" {head}"
        with self.assertRaisesRegex(ValueError, "exactly once"):
            v3_builder.validate_receiver_ontology(duplicate)


def _seed_inventory(protocol_name: str, seeds: list[int]) -> dict[str, object]:
    return {
        "protocol": protocol_name,
        "dataset": "causal",
        "status": "frozen_by_independent_seed_auditor",
        "seed_encoding": "nonnegative JSON integer below 2^63",
        "source_commitments": [
            {
                "name": "synthetic_registered_history",
                "sha256": "1" * 64,
                "seed_count": len(seeds),
            }
        ],
        "seeds": seeds,
    }


def _make_forbidden_seed_audit_fixture(
    base: Path, *, v2_seeds: list[int], v3_seeds: list[int]
) -> dict[str, object]:
    project = base / "project"
    v2_root = base / "private_v2"
    v3_root = base / "private_v3"
    project.mkdir(mode=0o700)
    v2_root.mkdir(mode=0o700)
    v3_root.mkdir(mode=0o700)
    (project / "data/water_impact_dynamic_v4").mkdir(parents=True)
    v2_raw = identity_auditor.canonical_json_bytes(
        _seed_inventory(forbidden_seed_auditor.V2_INVENTORY_PROTOCOL, v2_seeds)
    )
    v3_raw = identity_auditor.canonical_json_bytes(
        _seed_inventory(forbidden_seed_auditor.V3_INVENTORY_PROTOCOL, v3_seeds)
    )
    _private_write(
        v2_root, forbidden_seed_auditor.V2_INVENTORY_BASENAME, v2_raw
    )
    _private_write(
        v3_root, forbidden_seed_auditor.V3_INVENTORY_BASENAME, v3_raw
    )
    wrapper = {
        "protocol": "water_impact_dynamic_v4_eval_commitment_registry_v2",
        "dataset": "causal",
        "dataset_version": "v4_dev72_v2",
        "stage": 0,
        "status": "committed",
        "sealed_final36_status": "unopened",
        "artifacts": {
            "forbidden_seed_inventory": _record(v2_raw, None),
        },
    }
    wrapper_raw = identity_auditor.canonical_json_bytes(wrapper)
    wrapper_path = project / identity_auditor.V2_STAGE0_RELATIVE
    wrapper_path.parent.mkdir(parents=True, exist_ok=True)
    wrapper_path.write_bytes(wrapper_raw)
    wrapper_path.chmod(0o644)
    contract = forbidden_seed_auditor.ForbiddenSeedAuditContract(
        v2_stage0_sha256=identity_auditor.sha256_bytes(wrapper_raw),
        v2_inventory_sha256=identity_auditor.sha256_bytes(v2_raw),
    )
    return {
        "project": project,
        "v2_root": v2_root,
        "v3_root": v3_root,
        "wrapper_path": wrapper_path,
        "contract": contract,
    }


class ForbiddenSeedSourceAuditorTests(unittest.TestCase):
    def test_equal_and_strict_superset_publish_only_aggregate_scalars(self) -> None:
        for relation, v3_seeds in (
            ("equal", [11, 22]),
            ("strict_superset", [11, 22, 33]),
        ):
            with self.subTest(relation=relation), tempfile.TemporaryDirectory(
                dir=REPO
            ) as directory:
                fixture = _make_forbidden_seed_audit_fixture(
                    Path(directory), v2_seeds=[11, 22], v3_seeds=v3_seeds
                )
                report, digest = forbidden_seed_auditor.run_audit(
                    project_root=fixture["project"],
                    private_v2_root=fixture["v2_root"],
                    private_v3_root=fixture["v3_root"],
                    contract=fixture["contract"],
                )
                self.assertTrue(digest)
                self.assertEqual(set(report), forbidden_seed_auditor.REPORT_KEYS)
                self.assertEqual(report["set_relation"], relation)
                self.assertEqual(report["v2_missing_from_v3_count"], 0)
                self.assertTrue(
                    all(not isinstance(value, (list, dict)) for value in report.values())
                )
                encoded = json.dumps(report)
                self.assertNotIn('"seeds"', encoded)
                self.assertNotIn("synthetic_registered_history", encoded)

    def test_missing_duplicate_unsorted_and_signed63_seed_inputs_fail(self) -> None:
        cases = (
            ([11], "omits a v2 seed"),
            ([11, 11, 22], "sorted and unique"),
            ([22, 11], "sorted and unique"),
            ([11, 2**63], "signed63"),
            ([-1, 11], "signed63"),
        )
        for seeds, message in cases:
            with self.subTest(seeds=seeds), tempfile.TemporaryDirectory(
                dir=REPO
            ) as directory:
                fixture = _make_forbidden_seed_audit_fixture(
                    Path(directory), v2_seeds=[11, 22], v3_seeds=seeds
                )
                with self.assertRaisesRegex(ValueError, message):
                    forbidden_seed_auditor.run_audit(
                        project_root=fixture["project"],
                        private_v2_root=fixture["v2_root"],
                        private_v3_root=fixture["v3_root"],
                        contract=fixture["contract"],
                        publish=False,
                    )

        with tempfile.TemporaryDirectory(dir=REPO) as directory:
            fixture = _make_forbidden_seed_audit_fixture(
                Path(directory), v2_seeds=[11, 22], v3_seeds=[11, 22]
            )
            path = (
                fixture["v3_root"]
                / forbidden_seed_auditor.V3_INVENTORY_BASENAME
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["source_commitments"] = [
                {"name": "z", "sha256": "1" * 64, "seed_count": 1},
                {"name": "a", "sha256": "2" * 64, "seed_count": 1},
            ]
            path.write_bytes(identity_auditor.canonical_json_bytes(payload))
            path.chmod(0o600)
            with self.assertRaisesRegex(ValueError, "sorted and unique"):
                forbidden_seed_auditor.run_audit(
                    project_root=fixture["project"],
                    private_v2_root=fixture["v2_root"],
                    private_v3_root=fixture["v3_root"],
                    contract=fixture["contract"],
                    publish=False,
                )

    def test_wrapper_mix_hash_and_null_row_count_are_enforced(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO) as directory:
            fixture = _make_forbidden_seed_audit_fixture(
                Path(directory), v2_seeds=[11, 22], v3_seeds=[11, 22]
            )
            wrong = forbidden_seed_auditor.ForbiddenSeedAuditContract(
                v2_stage0_sha256="0" * 64,
                v2_inventory_sha256=fixture["contract"].v2_inventory_sha256,
            )
            with self.assertRaisesRegex(ValueError, "wrapper hash"):
                forbidden_seed_auditor.run_audit(
                    project_root=fixture["project"],
                    private_v2_root=fixture["v2_root"],
                    private_v3_root=fixture["v3_root"],
                    contract=wrong,
                    publish=False,
                )

        with tempfile.TemporaryDirectory(dir=REPO) as directory:
            fixture = _make_forbidden_seed_audit_fixture(
                Path(directory), v2_seeds=[11, 22], v3_seeds=[11, 22]
            )
            path = (
                fixture["v2_root"]
                / forbidden_seed_auditor.V2_INVENTORY_BASENAME
            )
            path.write_bytes(path.read_bytes() + b"\n")
            path.chmod(0o600)
            with self.assertRaisesRegex(ValueError, "committed size"):
                forbidden_seed_auditor.run_audit(
                    project_root=fixture["project"],
                    private_v2_root=fixture["v2_root"],
                    private_v3_root=fixture["v3_root"],
                    contract=fixture["contract"],
                    publish=False,
                )

        with tempfile.TemporaryDirectory(dir=REPO) as directory:
            fixture = _make_forbidden_seed_audit_fixture(
                Path(directory), v2_seeds=[11, 22], v3_seeds=[11, 22]
            )
            wrapper = json.loads(
                fixture["wrapper_path"].read_text(encoding="utf-8")
            )
            wrapper["artifacts"]["forbidden_seed_inventory"]["row_count"] = 2
            raw = identity_auditor.canonical_json_bytes(wrapper)
            fixture["wrapper_path"].write_bytes(raw)
            contract = forbidden_seed_auditor.ForbiddenSeedAuditContract(
                v2_stage0_sha256=identity_auditor.sha256_bytes(raw),
                v2_inventory_sha256=fixture["contract"].v2_inventory_sha256,
            )
            with self.assertRaisesRegex(ValueError, "row count"):
                forbidden_seed_auditor.run_audit(
                    project_root=fixture["project"],
                    private_v2_root=fixture["v2_root"],
                    private_v3_root=fixture["v3_root"],
                    contract=contract,
                    publish=False,
                )

        with tempfile.TemporaryDirectory(dir=REPO) as directory:
            fixture = _make_forbidden_seed_audit_fixture(
                Path(directory), v2_seeds=[11, 22], v3_seeds=[11, 22]
            )
            target = Path(directory) / "wrapper_target.json"
            fixture["wrapper_path"].rename(target)
            fixture["wrapper_path"].symlink_to(target)
            with self.assertRaises(OSError):
                forbidden_seed_auditor.run_audit(
                    project_root=fixture["project"],
                    private_v2_root=fixture["v2_root"],
                    private_v3_root=fixture["v3_root"],
                    contract=fixture["contract"],
                    publish=False,
                )

    def test_exact_roots_symlink_hardlink_alias_and_extra_files_fail(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO) as directory:
            fixture = _make_forbidden_seed_audit_fixture(
                Path(directory), v2_seeds=[11, 22], v3_seeds=[11, 22]
            )
            extra = fixture["v2_root"] / "extra.json"
            extra.write_text("{}\n", encoding="utf-8")
            extra.chmod(0o600)
            with self.assertRaisesRegex(ValueError, "inventory is not exact"):
                forbidden_seed_auditor.run_audit(
                    project_root=fixture["project"],
                    private_v2_root=fixture["v2_root"],
                    private_v3_root=fixture["v3_root"],
                    contract=fixture["contract"],
                    publish=False,
                )

        with tempfile.TemporaryDirectory(dir=REPO) as directory:
            fixture = _make_forbidden_seed_audit_fixture(
                Path(directory), v2_seeds=[11, 22], v3_seeds=[11, 22]
            )
            path = (
                fixture["v3_root"]
                / forbidden_seed_auditor.V3_INVENTORY_BASENAME
            )
            target = Path(directory) / "v3_target.json"
            path.rename(target)
            path.symlink_to(target)
            with self.assertRaisesRegex((ValueError, PermissionError), "symlink|regular"):
                forbidden_seed_auditor.run_audit(
                    project_root=fixture["project"],
                    private_v2_root=fixture["v2_root"],
                    private_v3_root=fixture["v3_root"],
                    contract=fixture["contract"],
                    publish=False,
                )

        with tempfile.TemporaryDirectory(dir=REPO) as directory:
            fixture = _make_forbidden_seed_audit_fixture(
                Path(directory), v2_seeds=[11, 22], v3_seeds=[11, 22]
            )
            path = (
                fixture["v2_root"]
                / forbidden_seed_auditor.V2_INVENTORY_BASENAME
            )
            os.link(path, Path(directory) / "v2_hardlink.json")
            with self.assertRaisesRegex((ValueError, PermissionError), "inventory|nlink"):
                forbidden_seed_auditor.run_audit(
                    project_root=fixture["project"],
                    private_v2_root=fixture["v2_root"],
                    private_v3_root=fixture["v3_root"],
                    contract=fixture["contract"],
                    publish=False,
                )

        with tempfile.TemporaryDirectory(dir=REPO) as directory:
            fixture = _make_forbidden_seed_audit_fixture(
                Path(directory), v2_seeds=[11, 22], v3_seeds=[11, 22]
            )
            alias = Path(directory) / "v3_alias"
            alias.symlink_to(fixture["v3_root"], target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symlink component"):
                forbidden_seed_auditor.run_audit(
                    project_root=fixture["project"],
                    private_v2_root=fixture["v2_root"],
                    private_v3_root=alias,
                    contract=fixture["contract"],
                    publish=False,
                )

    def test_report_leak_forbidden_path_and_overwrite_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO) as directory:
            fixture = _make_forbidden_seed_audit_fixture(
                Path(directory), v2_seeds=[11, 22], v3_seeds=[11, 22]
            )
            report, _ = forbidden_seed_auditor.run_audit(
                project_root=fixture["project"],
                private_v2_root=fixture["v2_root"],
                private_v3_root=fixture["v3_root"],
                contract=fixture["contract"],
            )
            with self.assertRaisesRegex(FileExistsError, "overwrite"):
                forbidden_seed_auditor.run_audit(
                    project_root=fixture["project"],
                    private_v2_root=fixture["v2_root"],
                    private_v3_root=fixture["v3_root"],
                    contract=fixture["contract"],
                )
            leaked = dict(report)
            leaked["seeds"] = [11]
            with self.assertRaisesRegex(ValueError, "fields are not exact"):
                forbidden_seed_auditor.validate_report(
                    leaked, fixture["contract"]
                )
            for token in ("final36", "sealed", "quarantine"):
                with self.subTest(token=token), self.assertRaisesRegex(
                    ValueError, "forbidden"
                ):
                    identity_auditor.write_report_to_relative(
                        fixture["project"],
                        Path(f"data/{token}/forbidden.json"),
                        report,
                    )


screening_freezer = _load_v3_module(
    "freeze_water_impact_dynamic_v4_causal_screening_v3",
    "scripts/freeze_water_impact_dynamic_v4_causal_screening_v3.py",
)


def _write_freezer_csv(
    path: Path,
    rows: list[dict[str, object]],
    header: tuple[str, ...],
) -> None:
    path.write_bytes(screening_freezer._csv_bytes(rows, header))
    path.chmod(0o600)


def _authorize_freezer_stage0(base: Path) -> tuple[dict[str, object], dict[str, object]]:
    fixture = _make_stage0_authorize_fixture(base)
    with (
        mock.patch.object(
            stage0_authorizer,
            "_validate_runtime_registry",
            return_value=fixture["hardware"],
        ),
        mock.patch.object(stage0_authorizer, "_validate_capacity_artifacts"),
        mock.patch.object(
            stage0_authorizer,
            "probe_media_runtime_packages",
            return_value=fixture["media_runtime_packages"],
        ),
        mock.patch.multiple(
            v3_protocol,
            V2_TEMPLATE_SHA256=fixture["template_sha"],
            V2_FIELD_RULES_SHA256=fixture["field_sha"],
            V2_SELECTION_RULES_SHA256=fixture["rules_sha"],
        ),
        mock.patch.multiple(
            stage0_authorizer.protocol,
            V2_TEMPLATE_SHA256=fixture["template_sha"],
            V2_FIELD_RULES_SHA256=fixture["field_sha"],
            V2_SELECTION_RULES_SHA256=fixture["rules_sha"],
        ),
        mock.patch.multiple(
            stage0_authorizer.builder.protocol,
            V2_TEMPLATE_SHA256=fixture["template_sha"],
            V2_FIELD_RULES_SHA256=fixture["field_sha"],
            V2_SELECTION_RULES_SHA256=fixture["rules_sha"],
        ),
    ):
        wrapper = stage0_authorizer.authorize(
            project_root=fixture["project"],
            private_root=fixture["private"],
            pending_path=fixture["pending"],
            binding_path=fixture["binding"],
            wrapper_path=fixture["wrapper"],
        )
    return fixture, wrapper


def _make_screening_freeze_fixture(base: Path) -> dict[str, object]:
    fixture, stage0 = _authorize_freezer_stage0(base)
    project = fixture["project"]
    private = fixture["private"]
    candidate_path = fixture["candidate"]
    candidate_payload = json.loads(candidate_path.read_text(encoding="utf-8"))
    candidates = candidate_payload["candidates"]
    graph_path = private / "causal_stage0_candidate_graph_private576_v3.json"
    stage0_sha = v3_protocol.sha256_file(fixture["wrapper"])

    lock_path = private / screening_freezer.screening_runner.CUDA_LOCK_BASENAME
    _write_canonical_json(
        lock_path,
        {
            "protocol": "synthetic_cuda_lock_v3",
            "dataset_version": v3_protocol.DATASET_VERSION,
            "status": "released_after_success",
        },
        0o600,
    )
    generation = private / screening_freezer.GENERATION_DIR
    generation.mkdir(mode=0o700)
    videos = generation / "videos"
    videos.mkdir(mode=0o700)
    public = private / screening_freezer.PUBLIC_PACKAGE_DIR
    public.mkdir(mode=0o700)
    media = public / "media"
    media.mkdir(mode=0o700)
    composites = public / "composites"
    composites.mkdir(mode=0o700)
    package_private = private / screening_freezer.PRIVATE_PACKAGE_DIR
    package_private.mkdir(mode=0o700)

    template_rows: list[dict[str, object]] = []
    answer_rows: list[dict[str, object]] = []
    binding_rows: list[dict[str, object]] = []
    raw_rows: list[dict[str, object]] = []
    anonymous_records: list[dict[str, object]] = []
    composite_records: list[dict[str, object]] = []
    raw_map: dict[str, dict[str, object]] = {}
    anonymous_map: dict[str, dict[str, object]] = {}
    composite_map: dict[str, dict[str, object]] = {}
    screening_seed_sha = stage0["artifacts"]["screening_seed"]["sha256"]
    for index, candidate in enumerate(candidates):
        candidate_id = str(candidate["case_id"])
        review_id = f"s{index:03d}"
        video_name = f"screening_{index:03d}.mp4"
        source_relative = (
            f"{screening_freezer.GENERATION_DIR}/videos/{video_name}"
        )
        anonymous_relative = (
            f"{screening_freezer.PUBLIC_PACKAGE_DIR}/media/{review_id}.mp4"
        )
        composite_relative = (
            f"{screening_freezer.PUBLIC_PACKAGE_DIR}/composites/{review_id}.jpg"
        )
        source = private / source_relative
        anonymous = private / anonymous_relative
        composite = private / composite_relative
        video_raw = b"synthetic-v3-screening-video\x00" + index.to_bytes(4, "big")
        composite_raw = b"synthetic-v3-composite\x00" + index.to_bytes(4, "big")
        for path, raw in (
            (source, video_raw),
            (anonymous, video_raw),
            (composite, composite_raw),
        ):
            path.write_bytes(raw)
            path.chmod(0o600)
        video_sha = hashlib.sha256(video_raw).hexdigest()
        composite_sha = hashlib.sha256(composite_raw).hexdigest()
        raw_record = {"sha256": video_sha, "size_bytes": len(video_raw)}
        anonymous_record = {
            "sha256": video_sha,
            "size_bytes": len(video_raw),
        }
        composite_record = {
            "sha256": composite_sha,
            "size_bytes": len(composite_raw),
        }
        raw_map[review_id] = raw_record
        anonymous_map[review_id] = anonymous_record
        composite_map[review_id] = composite_record
        template_rows.append(
            {
                "review_id": review_id,
                "candidate_video_path": f"media/{review_id}.mp4",
                "candidate_video_sha256": video_sha,
                "composite_path": f"composites/{review_id}.jpg",
                "composite_sha256": composite_sha,
                **{field: "" for field in screening_freezer.SCORE_FIELDS},
                "notes": "",
            }
        )
        answer_rows.append(
            {
                "review_id": review_id,
                "candidate_index": index,
                "case_id": candidate_id,
                "raw_video_sha256": video_sha,
                "anonymous_video_sha256": video_sha,
                "composite_sha256": composite_sha,
            }
        )
        binding_rows.append(
            {
                "review_id": review_id,
                "candidate": candidate,
                "raw_video_sha256": video_sha,
                "anonymous_video_sha256": video_sha,
                "composite_sha256": composite_sha,
            }
        )
        raw_rows.append(
            {
                "index": index,
                "case_id": candidate_id,
                "video_name": video_name,
                "size_bytes": len(video_raw),
                "sha256": video_sha,
                "prompt_sha256": hashlib.sha256(
                    str(candidate["canonical_prompt"]).encode("utf-8")
                ).hexdigest(),
                "screening_seed_sha256": screening_seed_sha,
                **screening_freezer.MEDIA_EXPECTED,
            }
        )
        anonymous_records.append(
            {
                "review_id": review_id,
                "sha256": video_sha,
                "size_bytes": len(video_raw),
                **screening_freezer.MEDIA_EXPECTED,
            }
        )
        composite_records.append(
            {
                "review_id": review_id,
                "sha256": composite_sha,
                "size_bytes": len(composite_raw),
            }
        )

    answer_path = package_private / screening_freezer.ANSWER_KEY
    template_path = public / screening_freezer.TEMPLATE
    _write_freezer_csv(answer_path, answer_rows, screening_freezer.ANSWER_KEY_HEADER)
    _write_freezer_csv(
        template_path, template_rows, screening_freezer.PUBLIC_REVIEW_HEADER
    )

    review_ids = [f"s{index:03d}" for index in range(v3_protocol.CANDIDATE_COUNT)]
    review_order_sha = v3_protocol.sha256_bytes(
        v3_protocol.canonical_json_bytes(review_ids)
    )
    anonymous_inventory = {
        "protocol": screening_freezer.ANONYMOUS_INVENTORY_PROTOCOL,
        "dataset_version": v3_protocol.DATASET_VERSION,
        "status": "frozen_before_screening_review",
        "candidate_count": v3_protocol.CANDIDATE_COUNT,
        "review_order_sha256": review_order_sha,
        "videos": anonymous_records,
    }
    anonymous_inventory_path = public / screening_freezer.ANONYMOUS_INVENTORY
    _write_canonical_json(anonymous_inventory_path, anonymous_inventory, 0o600)
    composite_inventory = {
        "protocol": screening_freezer.COMPOSITE_INVENTORY_PROTOCOL,
        "dataset_version": v3_protocol.DATASET_VERSION,
        "status": "frozen_before_screening_review",
        "candidate_count": v3_protocol.CANDIDATE_COUNT,
        "review_order_sha256": review_order_sha,
        "frame_indices": list(
            screening_freezer.screening_runner.FRAME_INDICES
        ),
        "composites": composite_records,
    }
    composite_inventory_path = public / screening_freezer.COMPOSITE_INVENTORY
    _write_canonical_json(composite_inventory_path, composite_inventory, 0o600)

    support_payloads = {
        ".run_reservation_v3.json": b"{\"synthetic\":\"reservation\"}\n",
        "execution_started_v3.json": b"{\"synthetic\":\"started\"}\n",
        "generator_output_v3.log": b"synthetic generator log\n",
        "prompts.txt": b"synthetic committed prompts\n",
        screening_freezer.screening_runner.GENERIC_MANIFEST_BASENAME: (
            b"{\"synthetic\":\"generic manifest\"}\n"
        ),
    }
    support_paths: dict[str, Path] = {}
    for name, raw in support_payloads.items():
        path = generation / name
        path.write_bytes(raw)
        path.chmod(0o600)
        support_paths[name] = path

    raw_payload = {
        "protocol": screening_freezer.RAW_INVENTORY_PROTOCOL,
        "dataset_version": v3_protocol.DATASET_VERSION,
        "status": "complete",
        "candidate_count": v3_protocol.CANDIDATE_COUNT,
        "stage0_registry_sha256": stage0_sha,
        "candidate_manifest_sha256": v3_protocol.sha256_file(candidate_path),
        "generation_spec_sha256": stage0["artifacts"][
            "screening_generation_spec"
        ]["sha256"],
        "videos": raw_rows,
    }
    raw_path = generation / screening_freezer.RAW_INVENTORY
    _write_canonical_json(raw_path, raw_payload, 0o600)
    code_registry = json.loads(
        (project / v3_protocol.CODE_REGISTRY).read_text(encoding="utf-8")
    )
    model_inventory = json.loads(
        (
            project
            / screening_freezer.screening_runner.authorizer.MODEL_INVENTORY_PATH
        ).read_text(encoding="utf-8")
    )

    generation_payload = {
        "protocol": screening_freezer.GENERATION_PROTOCOL,
        "dataset_version": v3_protocol.DATASET_VERSION,
        "status": "complete_original_screening_generation",
        "candidate_count": v3_protocol.CANDIDATE_COUNT,
        "worker_count": 1,
        "stage0_registry_sha256": stage0_sha,
        "selection_binding_sha256": stage0["artifacts"]["selection_binding"][
            "sha256"
        ],
        "candidate_manifest_sha256": v3_protocol.sha256_file(candidate_path),
        "candidate_graph_sha256": v3_protocol.sha256_file(graph_path),
        "generation_spec_sha256": stage0["artifacts"][
            "screening_generation_spec"
        ]["sha256"],
        "screening_seed_sha256": screening_seed_sha,
        "model_content_inventory_sha256": model_inventory[
            "inventory_sha256"
        ],
        "runtime_registry_sha256": stage0["artifacts"]["runtime_registry"][
            "sha256"
        ],
        "code_registry_sha256": v3_protocol.sha256_file(
            project / v3_protocol.CODE_REGISTRY
        ),
        "generator_sha256": code_registry["artifacts"]["generator"]["sha256"],
        "generator_dependency_closure_sha256": fixture[
            "generator_dependency_closure_sha256"
        ],
        "media_runtime_packages": fixture["media_runtime_packages"],
        "cuda_lock_sha256": v3_protocol.sha256_file(lock_path),
        "run_reservation_sha256": v3_protocol.sha256_file(
            support_paths[".run_reservation_v3.json"]
        ),
        "execution_started_sha256": v3_protocol.sha256_file(
            support_paths["execution_started_v3.json"]
        ),
        "generator_log_sha256": v3_protocol.sha256_file(
            support_paths["generator_output_v3.log"]
        ),
        "prompt_file_sha256": v3_protocol.sha256_file(
            support_paths["prompts.txt"]
        ),
        "generic_generation_manifest_sha256": v3_protocol.sha256_file(
            support_paths[
                screening_freezer.screening_runner.GENERIC_MANIFEST_BASENAME
            ]
        ),
        "raw_video_inventory_sha256": v3_protocol.sha256_file(raw_path),
        "videos": raw_rows,
    }
    generation_path = generation / screening_freezer.GENERATION_MANIFEST
    _write_canonical_json(generation_path, generation_payload, 0o600)

    public_manifest = {
        "protocol": screening_freezer.PUBLIC_PACKAGE_PROTOCOL,
        "dataset_version": v3_protocol.DATASET_VERSION,
        "status": "frozen_before_screening_review",
        "candidate_count": v3_protocol.CANDIDATE_COUNT,
        "stage0_registry_sha256": stage0_sha,
        "generation_manifest_sha256": v3_protocol.sha256_file(generation_path),
        "review_order_sha256": review_order_sha,
        "review_template_sha256": v3_protocol.sha256_file(template_path),
        "anonymous_video_inventory_sha256": v3_protocol.sha256_file(
            anonymous_inventory_path
        ),
        "composite_inventory_sha256": v3_protocol.sha256_file(
            composite_inventory_path
        ),
    }
    public_manifest_path = public / screening_freezer.PUBLIC_MANIFEST
    _write_canonical_json(public_manifest_path, public_manifest, 0o600)

    candidate_binding = {
        "protocol": screening_freezer.CANDIDATE_BINDING_PROTOCOL,
        "dataset_version": v3_protocol.DATASET_VERSION,
        "status": "frozen_candidate_binding",
        "candidate_count": v3_protocol.CANDIDATE_COUNT,
        "rows": binding_rows,
    }
    candidate_binding_path = package_private / screening_freezer.CANDIDATE_BINDING
    _write_canonical_json(candidate_binding_path, candidate_binding, 0o600)

    private_manifest = {
        "protocol": screening_freezer.PRIVATE_PACKAGE_PROTOCOL,
        "dataset_version": v3_protocol.DATASET_VERSION,
        "status": "frozen_before_screening_review",
        "candidate_count": v3_protocol.CANDIDATE_COUNT,
        "stage0_registry_sha256": stage0_sha,
        "selection_binding_sha256": stage0["artifacts"]["selection_binding"][
            "sha256"
        ],
        "candidate_graph_sha256": v3_protocol.sha256_file(graph_path),
        "candidate_manifest_sha256": v3_protocol.sha256_file(candidate_path),
        "generation_spec_sha256": stage0["artifacts"][
            "screening_generation_spec"
        ]["sha256"],
        "generation_manifest_sha256": v3_protocol.sha256_file(generation_path),
        "raw_video_inventory_sha256": v3_protocol.sha256_file(raw_path),
        "review_order_sha256": review_order_sha,
        "review_template_sha256": v3_protocol.sha256_file(template_path),
        "answer_key_sha256": v3_protocol.sha256_file(answer_path),
        "candidate_binding_sha256": v3_protocol.sha256_file(
            candidate_binding_path
        ),
        "anonymous_video_inventory_sha256": v3_protocol.sha256_file(
            anonymous_inventory_path
        ),
        "composite_inventory_sha256": v3_protocol.sha256_file(
            composite_inventory_path
        ),
        "public_manifest_sha256": v3_protocol.sha256_file(public_manifest_path),
        "generator_dependency_closure_sha256": fixture[
            "generator_dependency_closure_sha256"
        ],
        "media_runtime_packages": fixture["media_runtime_packages"],
        "raw_media": raw_map,
        "anonymous_media": anonymous_map,
        "composites": composite_map,
    }
    private_manifest_path = package_private / screening_freezer.PRIVATE_MANIFEST
    _write_canonical_json(private_manifest_path, private_manifest, 0o600)

    commitment = {
        "protocol": screening_freezer.PACKAGE_PROTOCOL,
        "dataset_version": v3_protocol.DATASET_VERSION,
        "status": "committed_before_any_screening_review",
        "candidate_count": v3_protocol.CANDIDATE_COUNT,
        "stage0_registry_sha256": stage0_sha,
        "pending_commitment_sha256": v3_protocol.sha256_file(
            fixture["pending"]
        ),
        "selection_binding_sha256": private_manifest[
            "selection_binding_sha256"
        ],
        "candidate_manifest_sha256": private_manifest[
            "candidate_manifest_sha256"
        ],
        "candidate_graph_sha256": private_manifest["candidate_graph_sha256"],
        "generation_spec_sha256": private_manifest[
            "generation_spec_sha256"
        ],
        "generation_manifest_sha256": private_manifest[
            "generation_manifest_sha256"
        ],
        "raw_video_inventory_sha256": private_manifest[
            "raw_video_inventory_sha256"
        ],
        "model_content_inventory_sha256": model_inventory[
            "inventory_sha256"
        ],
        "runtime_registry_sha256": stage0["artifacts"]["runtime_registry"][
            "sha256"
        ],
        "code_registry_sha256": stage0["artifacts"]["eval_code_registry"][
            "sha256"
        ],
        "generator_sha256": code_registry["artifacts"]["generator"]["sha256"],
        "generator_dependency_closure_sha256": fixture[
            "generator_dependency_closure_sha256"
        ],
        "media_runtime_packages": fixture["media_runtime_packages"],
        "cuda_lock_sha256": generation_payload["cuda_lock_sha256"],
        "run_reservation_sha256": generation_payload[
            "run_reservation_sha256"
        ],
        "execution_started_sha256": generation_payload[
            "execution_started_sha256"
        ],
        "generator_log_sha256": generation_payload["generator_log_sha256"],
        "prompt_file_sha256": generation_payload["prompt_file_sha256"],
        "generic_generation_manifest_sha256": generation_payload[
            "generic_generation_manifest_sha256"
        ],
        "review_order_sha256": private_manifest["review_order_sha256"],
        "review_template_sha256": private_manifest[
            "review_template_sha256"
        ],
        "answer_key_sha256": private_manifest["answer_key_sha256"],
        "candidate_binding_sha256": private_manifest[
            "candidate_binding_sha256"
        ],
        "anonymous_video_inventory_sha256": private_manifest[
            "anonymous_video_inventory_sha256"
        ],
        "composite_inventory_sha256": private_manifest[
            "composite_inventory_sha256"
        ],
        "public_manifest_sha256": private_manifest["public_manifest_sha256"],
        "private_manifest_sha256": v3_protocol.sha256_file(
            private_manifest_path
        ),
        "raw_media": raw_map,
        "anonymous_media": anonymous_map,
        "composites": composite_map,
    }
    commitment_path = package_private / screening_freezer.PACKAGE_COMMITMENT
    _write_canonical_json(commitment_path, commitment, 0o600)
    success_status = {
        "protocol": screening_freezer.screening_runner.STATUS_PROTOCOL,
        "dataset_version": v3_protocol.DATASET_VERSION,
        "status": "succeeded",
        "stage0_registry_sha256": stage0_sha,
        "failure_phase": None,
        "reason_code": None,
        "generation_manifest_sha256": v3_protocol.sha256_file(generation_path),
        "package_commitment_sha256": v3_protocol.sha256_file(commitment_path),
    }
    _write_canonical_json(
        generation / "execution_succeeded_v3.json", success_status, 0o600
    )

    coordination = base / "public-screening-coordination"
    shutil.copytree(public, coordination)
    review_a: list[dict[str, object]] = []
    review_b: list[dict[str, object]] = []
    for index, template_row in enumerate(template_rows):
        scores_a = {
            "source_visibility": 2,
            "footprint_visibility": 1,
            "receiver": 1,
            "quality": 1,
            "causal_link": 2,
        }
        scores_b = dict(scores_a)
        if index == 0:
            scores_b["source_visibility"] = 0
        review_a.append({**template_row, **scores_a, "notes": "review A complete"})
        review_b.append({**template_row, **scores_b, "notes": "review B complete"})
    review_a_path = coordination / screening_freezer.REVIEW_A
    review_b_path = coordination / screening_freezer.REVIEW_B
    _write_freezer_csv(
        review_a_path, review_a, screening_freezer.PUBLIC_REVIEW_HEADER
    )
    _write_freezer_csv(
        review_b_path, review_b, screening_freezer.PUBLIC_REVIEW_HEADER
    )

    blind = private / screening_freezer.BLIND_INPUT_DIR
    blind.mkdir(mode=0o700)
    execution = private / screening_freezer.FREEZE_PARENT
    execution.mkdir(mode=0o700)
    return {
        **fixture,
        "stage0": stage0,
        "public": public,
        "coordination": coordination,
        "review_a_path": review_a_path,
        "review_b_path": review_b_path,
        "blind": blind,
        "execution": execution,
        "template_rows": template_rows,
        "candidates": candidates,
    }


class ScreeningFreezerV3Tests(unittest.TestCase):
    def test_merge_is_exact_every_only_median_and_eligibility(self) -> None:
        candidates = [
            {
                "case_id": f"v4v3c{index:03d}",
                "group": "G1",
                "prompt_variant": "direct",
            }
            for index in range(v3_protocol.CANDIDATE_COUNT)
        ]
        template = [
            {
                "review_id": f"s{index:03d}",
                "candidate_video_path": f"media/s{index:03d}.mp4",
                "candidate_video_sha256": f"{index + 1:064x}",
                "composite_path": f"composites/s{index:03d}.jpg",
                "composite_sha256": f"{index + 577:064x}",
                **{field: "" for field in screening_freezer.SCORE_FIELDS},
                "notes": "",
            }
            for index in range(v3_protocol.CANDIDATE_COUNT)
        ]
        left = [
            {
                **row,
                "source_visibility": "2",
                "footprint_visibility": "1",
                "receiver": "1",
                "quality": "1",
                "causal_link": "2",
                "notes": "complete",
            }
            for row in template
        ]
        right = [dict(row) for row in left]
        right[0]["source_visibility"] = "0"
        disputes = [{"review_id": "s000", "field": "source_visibility"}]
        answer = {
            f"s{index:03d}": {"case_id": f"v4v3c{index:03d}"}
            for index in range(v3_protocol.CANDIDATE_COUNT)
        }
        with self.assertRaisesRegex(ValueError, "every-only disagreement"):
            screening_freezer._merge_reviews(
                candidates, template, left, right, disputes, [], answer
            )
        with self.assertRaisesRegex(ValueError, "unexpected/duplicate"):
            screening_freezer._merge_reviews(
                candidates,
                template,
                left,
                right,
                disputes,
                [
                    {
                        "review_id": "s001",
                        "field": "quality",
                        "score": "1",
                        "brief_reason": "not disputed",
                    }
                ],
                answer,
            )
        eligibility, audit = screening_freezer._merge_reviews(
            candidates,
            template,
            left,
            right,
            disputes,
            [
                {
                    "review_id": "s000",
                    "field": "source_visibility",
                    "score": "1",
                    "brief_reason": "median adjudication",
                }
            ],
            answer,
        )
        self.assertEqual((eligibility[0]["source_visibility"], eligibility[0]["eligible"]), (1, "no"))
        self.assertEqual(sum(row["eligible"] == "yes" for row in eligibility), 575)
        self.assertEqual(audit[0]["canonical"], 1)

    def test_public_disputes_and_transactional_full_freeze(self) -> None:
        def prepare_blind(fixture: dict[str, object], *, adjudicate: bool) -> Path:
            dispute_path = fixture["coordination"] / screening_freezer.DISPUTES
            if not dispute_path.exists():
                screening_freezer.derive_disputes(
                    fixture["project"], fixture["coordination"]
                )
            for name, source in (
                (screening_freezer.REVIEW_A, fixture["review_a_path"]),
                (screening_freezer.REVIEW_B, fixture["review_b_path"]),
                (screening_freezer.DISPUTES, dispute_path),
            ):
                target = fixture["blind"] / name
                shutil.copyfile(source, target)
                target.chmod(0o600)
            if adjudicate:
                _write_freezer_csv(
                    fixture["blind"] / screening_freezer.ADJUDICATION,
                    [
                        {
                            "review_id": "s000",
                            "field": "source_visibility",
                            "score": 1,
                            "brief_reason": "independent atomic adjudication",
                        }
                    ],
                    screening_freezer.ADJUDICATION_HEADER,
                )
            return dispute_path

        with tempfile.TemporaryDirectory(dir=REPO.parent) as directory:
            fixture = _make_screening_freeze_fixture(Path(directory))
            coordination = fixture["coordination"]
            dispute_path = coordination / screening_freezer.DISPUTES
            outside_link = Path(directory) / "review-a-hardlink.csv"
            os.link(fixture["review_a_path"], outside_link)
            with self.assertRaisesRegex(ValueError, "single-link regular"):
                screening_freezer.derive_disputes(
                    fixture["project"], coordination
                )
            self.assertFalse(dispute_path.exists())
            outside_link.unlink()
            dispute_count, dispute_sha = screening_freezer.derive_disputes(
                fixture["project"], coordination
            )
            self.assertEqual(dispute_count, 1)
            self.assertEqual(v3_protocol.sha256_file(dispute_path), dispute_sha)
            with self.assertRaisesRegex(FileExistsError, "overwrite"):
                screening_freezer.derive_disputes(
                    fixture["project"], coordination
                )
            self.assertEqual(v3_protocol.sha256_file(dispute_path), dispute_sha)
            prepare_blind(fixture, adjudicate=True)
            freeze = screening_freezer.freeze_screening(
                fixture["project"],
                fixture["private"],
                decode=lambda _: screening_freezer.MEDIA_EXPECTED,
            )
            self.assertEqual(
                (freeze["candidate_count"], freeze["eligible_count"], freeze["dispute_count"]),
                (576, 575, 1),
            )
            frozen_dir = fixture["execution"] / screening_freezer.FREEZE_DIR
            self.assertEqual(
                {path.name for path in frozen_dir.iterdir()},
                {
                    screening_freezer.ELIGIBILITY_OUT,
                    screening_freezer.AUDIT_OUT,
                    screening_freezer.FREEZE_MANIFEST_OUT,
                },
            )
            eligibility = screening_freezer._read_csv_bytes(
                (frozen_dir / screening_freezer.ELIGIBILITY_OUT).read_bytes(),
                screening_freezer.ELIGIBILITY_HEADER,
                "frozen eligibility",
            )
            self.assertEqual(
                (eligibility[0]["source_visibility"], eligibility[0]["eligible"]),
                ("1", "no"),
            )
            self.assertEqual(
                freeze["artifacts"]["screening_private_package_manifest_576"][
                    "row_count"
                ],
                576,
            )
            self.assertFalse((fixture["execution"] / "selector").exists())
            self.assertFalse(
                (fixture["project"] / v3_protocol.STAGE1_REGISTRY).exists()
            )

        with tempfile.TemporaryDirectory(dir=REPO.parent) as directory:
            fixture = _make_screening_freeze_fixture(Path(directory))
            prepare_blind(fixture, adjudicate=False)
            with self.assertRaisesRegex(
                screening_freezer.ScreeningTerminalFailure,
                "screening_adjudication_integrity_failure",
            ):
                screening_freezer.freeze_screening(
                    fixture["project"],
                    fixture["private"],
                    decode=lambda _: screening_freezer.MEDIA_EXPECTED,
                )
            invalid = json.loads(
                (fixture["project"] / v3_protocol.INVALID_OUTCOME).read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                (invalid["failure_phase"], invalid["reason_code"]),
                ("screening_freeze", "screening_adjudication_integrity_failure"),
            )
            self.assertEqual(list(fixture["execution"].iterdir()), [])
            _write_freezer_csv(
                fixture["blind"] / screening_freezer.ADJUDICATION,
                [
                    {
                        "review_id": "s000",
                        "field": "source_visibility",
                        "score": 1,
                        "brief_reason": "late repair is forbidden",
                    }
                ],
                screening_freezer.ADJUDICATION_HEADER,
            )
            with self.assertRaisesRegex(FileExistsError, "may not retry"):
                screening_freezer.freeze_screening(
                    fixture["project"],
                    fixture["private"],
                    decode=lambda _: screening_freezer.MEDIA_EXPECTED,
                )

        with tempfile.TemporaryDirectory(dir=REPO.parent) as directory:
            fixture = _make_screening_freeze_fixture(Path(directory))
            extra = fixture["public"] / "unregistered.txt"
            extra.write_bytes(b"not committed")
            extra.chmod(0o600)
            with self.assertRaisesRegex(
                screening_freezer.ScreeningTerminalFailure,
                "screening_package_integrity_failure",
            ):
                screening_freezer.freeze_screening(
                    fixture["project"],
                    fixture["private"],
                    decode=lambda _: screening_freezer.MEDIA_EXPECTED,
                )
            invalid = json.loads(
                (fixture["project"] / v3_protocol.INVALID_OUTCOME).read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(invalid["failure_phase"], "screening_package")
            self.assertEqual(list(fixture["execution"].iterdir()), [])

        with tempfile.TemporaryDirectory(dir=REPO.parent) as directory:
            fixture = _make_screening_freeze_fixture(Path(directory))
            prepare_blind(fixture, adjudicate=True)
            real_write = screening_freezer._write_private_file
            call_count = 0

            def fail_second_write(path: Path, raw: bytes) -> None:
                nonlocal call_count
                call_count += 1
                if call_count == 2:
                    raise OSError("synthetic transaction failure")
                real_write(path, raw)

            with (
                mock.patch.object(
                    screening_freezer,
                    "_write_private_file",
                    side_effect=fail_second_write,
                ),
                self.assertRaisesRegex(
                    screening_freezer.ScreeningTerminalFailure,
                    "screening_adjudication_integrity_failure",
                ),
            ):
                screening_freezer.freeze_screening(
                    fixture["project"],
                    fixture["private"],
                    decode=lambda _: screening_freezer.MEDIA_EXPECTED,
                )
            self.assertEqual(list(fixture["execution"].iterdir()), [])
            self.assertTrue(
                (fixture["project"] / v3_protocol.INVALID_OUTCOME).exists()
            )

    def test_cli_exposes_no_free_artifact_or_stage1_path(self) -> None:
        parser = screening_freezer.build_parser()
        derive = parser.parse_args(
            [
                "derive-disputes",
                "--project-root",
                "/tmp/project-v3",
                "--public-root",
                "/tmp/public-v3",
            ]
        )
        freeze = parser.parse_args(
            [
                "freeze",
                "--project-root",
                "/tmp/project-v3",
                "--private-root",
                "/tmp/private-v3",
            ]
        )
        self.assertEqual(
            set(vars(derive)), {"command", "project_root", "public_root"}
        )
        self.assertEqual(
            set(vars(freeze)), {"command", "project_root", "private_root"}
        )

    def test_shared_mutex_serializes_success_terminal_and_crash_release(self) -> None:
        def prepare_blind(fixture: dict[str, object], *, adjudicate: bool) -> None:
            _, _ = screening_freezer.derive_disputes(
                fixture["project"], fixture["coordination"]
            )
            for name, source in (
                (screening_freezer.REVIEW_A, fixture["review_a_path"]),
                (screening_freezer.REVIEW_B, fixture["review_b_path"]),
                (
                    screening_freezer.DISPUTES,
                    fixture["coordination"] / screening_freezer.DISPUTES,
                ),
            ):
                target = fixture["blind"] / name
                shutil.copyfile(source, target)
                target.chmod(0o600)
            if adjudicate:
                _write_freezer_csv(
                    fixture["blind"] / screening_freezer.ADJUDICATION,
                    [
                        {
                            "review_id": "s000",
                            "field": "source_visibility",
                            "score": 1,
                            "brief_reason": "mutex owner adjudication",
                        }
                    ],
                    screening_freezer.ADJUDICATION_HEADER,
                )

        def snapshot(*roots: Path) -> tuple[tuple[object, ...], ...]:
            rows: list[tuple[object, ...]] = []
            for root_index, root in enumerate(roots):
                for path in sorted(root.rglob("*")):
                    info = os.lstat(path)
                    relative = path.relative_to(root).as_posix()
                    digest = (
                        hashlib.sha256(path.read_bytes()).hexdigest()
                        if path.is_file() and not path.is_symlink()
                        else None
                    )
                    rows.append(
                        (
                            root_index,
                            relative,
                            stat.S_IFMT(info.st_mode),
                            stat.S_IMODE(info.st_mode),
                            info.st_nlink,
                            info.st_size,
                            digest,
                        )
                    )
            return tuple(rows)

        for terminal_owner in (False, True):
            with self.subTest(terminal_owner=terminal_owner), tempfile.TemporaryDirectory(
                dir=REPO.parent
            ) as directory:
                fixture = _make_screening_freeze_fixture(Path(directory))
                prepare_blind(fixture, adjudicate=not terminal_owner)
                ready_read, ready_write = os.pipe()
                release_read, release_write = os.pipe()
                child = os.fork()
                if child == 0:
                    try:
                        os.close(ready_read)
                        os.close(release_write)
                        with screening_runner._screening_mutex(
                            fixture["private"]
                        ):
                            os.write(ready_write, b"1")
                            if os.read(release_read, 1) != b"1":
                                os._exit(91)
                            try:
                                screening_freezer._freeze_screening_locked(
                                    fixture["project"],
                                    fixture["private"],
                                    decode=lambda _: screening_freezer.MEDIA_EXPECTED,
                                )
                            except screening_freezer.ScreeningTerminalFailure:
                                if not terminal_owner:
                                    os._exit(92)
                            else:
                                if terminal_owner:
                                    os._exit(93)
                        os._exit(0)
                    except BaseException:
                        os._exit(94)
                os.close(ready_write)
                os.close(release_read)
                try:
                    self.assertEqual(os.read(ready_read, 1), b"1")
                    before = snapshot(fixture["project"], fixture["private"])
                    with self.assertRaisesRegex(FileExistsError, "mutex"):
                        screening_freezer.freeze_screening(
                            fixture["project"],
                            fixture["private"],
                            decode=lambda _: screening_freezer.MEDIA_EXPECTED,
                        )
                    self.assertEqual(
                        before, snapshot(fixture["project"], fixture["private"])
                    )
                    os.write(release_write, b"1")
                finally:
                    os.close(ready_read)
                    os.close(release_write)
                _, status = os.waitpid(child, 0)
                self.assertTrue(os.WIFEXITED(status))
                self.assertEqual(os.WEXITSTATUS(status), 0)
                if terminal_owner:
                    self.assertTrue(
                        (fixture["project"] / v3_protocol.INVALID_OUTCOME).exists()
                    )
                    self.assertEqual(list(fixture["execution"].iterdir()), [])
                else:
                    self.assertTrue(
                        (
                            fixture["execution"]
                            / screening_freezer.FREEZE_DIR
                            / screening_freezer.FREEZE_MANIFEST_OUT
                        ).exists()
                    )
                    self.assertFalse(
                        (fixture["project"] / v3_protocol.INVALID_OUTCOME).exists()
                    )

        with tempfile.TemporaryDirectory(dir=REPO.parent) as directory:
            private_root = Path(directory)
            private_root.chmod(0o700)
            ready_read, ready_write = os.pipe()
            child = os.fork()
            if child == 0:
                os.close(ready_read)
                with screening_runner._screening_mutex(private_root):
                    os.write(ready_write, b"1")
                    os._exit(17)
            os.close(ready_write)
            self.assertEqual(os.read(ready_read, 1), b"1")
            os.close(ready_read)
            _, status = os.waitpid(child, 0)
            self.assertEqual(os.WEXITSTATUS(status), 17)
            with screening_runner._screening_mutex(private_root):
                pass


screening_runner = screening_freezer.screening_runner


class ScreeningRunnerV3Tests(unittest.TestCase):
    def _runner_validation_patches(self, fixture: dict[str, object]):
        return (
            mock.patch.object(
                screening_runner.authorizer,
                "_validate_runtime_registry",
                return_value=fixture["hardware"],
            ),
            mock.patch.object(
                screening_runner.authorizer, "_validate_capacity_artifacts"
            ),
            mock.patch.multiple(
                v3_protocol,
                V2_TEMPLATE_SHA256=fixture["template_sha"],
                V2_FIELD_RULES_SHA256=fixture["field_sha"],
                V2_SELECTION_RULES_SHA256=fixture["rules_sha"],
            ),
            mock.patch.multiple(
                screening_runner.authorizer.protocol,
                V2_TEMPLATE_SHA256=fixture["template_sha"],
                V2_FIELD_RULES_SHA256=fixture["field_sha"],
                V2_SELECTION_RULES_SHA256=fixture["rules_sha"],
            ),
            mock.patch.multiple(
                screening_runner.builder.protocol,
                V2_TEMPLATE_SHA256=fixture["template_sha"],
                V2_FIELD_RULES_SHA256=fixture["field_sha"],
                V2_SELECTION_RULES_SHA256=fixture["rules_sha"],
            ),
            mock.patch.object(
                screening_runner.authorizer,
                "probe_media_runtime_packages",
                return_value=fixture["media_runtime_packages"],
            ),
            mock.patch.object(
                screening_runner, "validate_runner_process_environment"
            ),
        )

    @staticmethod
    def _fake_decode(path: Path, *, collect_composite_frames: bool):
        metadata = {
            "frame_count": 49,
            "width": 832,
            "height": 480,
            "fps_numerator": 8,
            "fps_denominator": 1,
        }
        return metadata, ([object()] * 7 if collect_composite_frames else [])

    @staticmethod
    def _fake_composite(path: Path, frames) -> None:
        if len(frames) != 7:
            raise AssertionError("composite must receive seven frozen frames")
        path.write_bytes(b"synthetic-v3-composite")
        path.chmod(0o600)

    def test_single_worker_command_and_all_exact_payload_validators(self) -> None:
        command = screening_runner.generation_command(
            python_executable="models/.wan-runtime/bin/python",
            generator_relative="scripts/generate_wan_clean.py",
            prompt_path=Path("/private/prompts.txt"),
            generation_dir=Path("/private/generation"),
            screening_seed=77,
        )
        self.assertNotIn("--skip-existing", command)
        self.assertNotIn("--dry-run", command)
        self.assertNotIn("--limit", command)
        self.assertNotIn("--lora-path", command)
        self.assertNotIn("--activation-gate-dir", command)
        self.assertNotIn("--attention-gate-dir", command)
        self.assertEqual(command[1:4], ["-I", "-c", screening_runner.GENERATOR_BOOTSTRAP])
        self.assertEqual(command.count("--seeds"), 1)
        self.assertEqual(
            len(command[command.index("--seeds") + 1].split(",")), 576
        )
        environment = screening_runner.sanitized_worker_environment(
            {
                "PATH": "/usr/bin",
                "PYTHONPATH": "/attacker",
                "PYTHONHOME": "/attacker-home",
                "CUDA_VISIBLE_DEVICES": "0",
            }
        )
        self.assertNotIn("PYTHONPATH", environment)
        self.assertNotIn("PYTHONHOME", environment)
        self.assertEqual(environment["PYTHONNOUSERSITE"], "1")
        self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "0")
        with tempfile.TemporaryDirectory(dir=REPO.parent) as directory:
            private_root = Path(directory)
            private_root.chmod(0o700)
            with screening_runner._screening_mutex(private_root):
                with self.assertRaisesRegex(FileExistsError, "owns"):
                    with screening_runner._screening_mutex(private_root):
                        pass

        generation_spec = {
            "protocol": stage0_authorizer.GENERATION_SPEC_PROTOCOL,
            "dataset_version": v3_protocol.DATASET_VERSION,
            "status": "frozen_before_original_screening",
            "candidate_manifest_sha256": "1" * 64,
            "candidate_graph_sha256": "2" * 64,
            "render_configuration_sha256": "3" * 64,
            "screening_seed_sha256": "4" * 64,
            "graph_assignment_salt_sha256": "5" * 64,
            "model_content_inventory_sha256": "6" * 64,
            "runtime_registry_sha256": "7" * 64,
            "generator_dependency_closure_sha256": "8" * 64,
            "media_runtime_packages": {
                "av": "synthetic-av",
                "Pillow": "synthetic-pillow",
            },
            "candidate_count": 576,
            "generation": dict(screening_runner.EXPECTED_GENERATION),
        }
        stage0_authorizer._validate_generation_spec(
            generation_spec,
            candidate_sha="1" * 64,
            graph_sha="2" * 64,
            render_sha="3" * 64,
            screening_seed_sha="4" * 64,
            graph_salt_sha="5" * 64,
            model_sha="6" * 64,
            runtime_sha="7" * 64,
            generator_dependency_closure_sha256="8" * 64,
            media_runtime_packages={
                "av": "synthetic-av",
                "Pillow": "synthetic-pillow",
            },
        )
        generation_spec["generation"]["worker_count"] = 2
        with self.assertRaisesRegex(ValueError, "generation spec mismatch"):
            stage0_authorizer._validate_generation_spec(
                generation_spec,
                candidate_sha="1" * 64,
                graph_sha="2" * 64,
                render_sha="3" * 64,
                screening_seed_sha="4" * 64,
                graph_salt_sha="5" * 64,
                model_sha="6" * 64,
                runtime_sha="7" * 64,
                generator_dependency_closure_sha256="8" * 64,
                media_runtime_packages={
                    "av": "synthetic-av",
                    "Pillow": "synthetic-pillow",
                },
            )
        generation_spec["generation"]["worker_count"] = 1
        generation_spec["generator_dependency_closure_sha256"] = "9" * 64
        with self.assertRaisesRegex(ValueError, "generation spec mismatch"):
            stage0_authorizer._validate_generation_spec(
                generation_spec,
                candidate_sha="1" * 64,
                graph_sha="2" * 64,
                render_sha="3" * 64,
                screening_seed_sha="4" * 64,
                graph_salt_sha="5" * 64,
                model_sha="6" * 64,
                runtime_sha="7" * 64,
                generator_dependency_closure_sha256="8" * 64,
                media_runtime_packages={
                    "av": "synthetic-av",
                    "Pillow": "synthetic-pillow",
                },
            )
        generation_spec["generator_dependency_closure_sha256"] = "8" * 64
        generation_spec["media_runtime_packages"]["av"] = "drifted-av"
        with self.assertRaisesRegex(ValueError, "generation spec mismatch"):
            stage0_authorizer._validate_generation_spec(
                generation_spec,
                candidate_sha="1" * 64,
                graph_sha="2" * 64,
                render_sha="3" * 64,
                screening_seed_sha="4" * 64,
                graph_salt_sha="5" * 64,
                model_sha="6" * 64,
                runtime_sha="7" * 64,
                generator_dependency_closure_sha256="8" * 64,
                media_runtime_packages={
                    "av": "synthetic-av",
                    "Pillow": "synthetic-pillow",
                },
            )

        with tempfile.TemporaryDirectory(dir=REPO.parent) as directory:
            fixture = _make_screening_freeze_fixture(Path(directory))
            generation = (
                fixture["private"] / screening_runner.GENERATION_DIRNAME
            )
            public = fixture["public"]
            private_package = (
                fixture["private"] / screening_runner.PRIVATE_PACKAGE_DIRNAME
            )
            raw = json.loads(
                (generation / screening_runner.RAW_INVENTORY_BASENAME).read_text(
                    encoding="utf-8"
                )
            )
            manifest = json.loads(
                (
                    generation / screening_runner.GENERATION_MANIFEST_BASENAME
                ).read_text(encoding="utf-8")
            )
            candidate_binding = json.loads(
                (
                    private_package
                    / screening_runner.CANDIDATE_BINDING_BASENAME
                ).read_text(encoding="utf-8")
            )
            anonymous = json.loads(
                (
                    public / screening_runner.ANONYMOUS_INVENTORY_BASENAME
                ).read_text(encoding="utf-8")
            )
            composite = json.loads(
                (
                    public / screening_runner.COMPOSITE_INVENTORY_BASENAME
                ).read_text(encoding="utf-8")
            )
            public_manifest = json.loads(
                (public / screening_runner.PUBLIC_MANIFEST_BASENAME).read_text(
                    encoding="utf-8"
                )
            )
            private_manifest = json.loads(
                (
                    private_package / screening_runner.PRIVATE_MANIFEST_BASENAME
                ).read_text(encoding="utf-8")
            )
            commitment = json.loads(
                (
                    private_package
                    / screening_runner.PACKAGE_COMMITMENT_BASENAME
                ).read_text(encoding="utf-8")
            )
            screening_runner.validate_raw_video_inventory(raw)
            screening_runner.validate_generation_manifest(manifest)
            screening_runner.validate_candidate_binding(candidate_binding)
            screening_runner.validate_anonymous_inventory(anonymous)
            screening_runner.validate_composite_inventory(composite)
            screening_runner._validate_public_package_payload(public_manifest)
            screening_runner.validate_private_package_manifest(private_manifest)
            screening_runner.validate_package_commitment(commitment)

            tamper_cases = []
            value = json.loads(json.dumps(raw))
            value["videos"][0]["index"] = 1
            tamper_cases.append(
                (screening_runner.validate_raw_video_inventory, value)
            )
            value = json.loads(json.dumps(manifest))
            value["worker_count"] = 2
            tamper_cases.append(
                (screening_runner.validate_generation_manifest, value)
            )
            value = json.loads(json.dumps(candidate_binding))
            value["rows"][0]["review_id"] = "s001"
            tamper_cases.append(
                (screening_runner.validate_candidate_binding, value)
            )
            value = json.loads(json.dumps(anonymous))
            value["videos"][0]["size_bytes"] = 0
            tamper_cases.append(
                (screening_runner.validate_anonymous_inventory, value)
            )
            value = json.loads(json.dumps(composite))
            value["frame_indices"][-1] = 47
            tamper_cases.append(
                (screening_runner.validate_composite_inventory, value)
            )
            value = json.loads(json.dumps(public_manifest))
            value["path"] = "/private/leak"
            tamper_cases.append(
                (screening_runner._validate_public_package_payload, value)
            )
            value = json.loads(json.dumps(private_manifest))
            value["raw_media"].pop("s000")
            tamper_cases.append(
                (screening_runner.validate_private_package_manifest, value)
            )
            value = json.loads(json.dumps(commitment))
            value["anonymous_media"].pop("s000")
            tamper_cases.append(
                (screening_runner.validate_package_commitment, value)
            )
            for validator, payload in tamper_cases:
                with self.subTest(validator=validator.__name__), self.assertRaises(
                    ValueError
                ):
                    validator(payload)

            header = (
                public / screening_runner.REVIEW_TEMPLATE_BASENAME
            ).read_text(encoding="utf-8").splitlines()[0]
            self.assertEqual(header, ",".join(screening_runner.REVIEW_HEADER))
            self.assertFalse(
                (public / screening_runner.PACKAGE_COMMITMENT_BASENAME).exists()
            )

    def test_full_576_mock_run_publishes_exact_package_and_terminal_status(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO.parent) as directory:
            fixture, _ = _authorize_freezer_stage0(Path(directory))
            project = fixture["project"]
            private = fixture["private"]

            def execute(command, **kwargs):
                self.assertFalse(kwargs["shell"])
                self.assertEqual(
                    kwargs["timeout"],
                    screening_runner.MAX_SCREENING_GENERATION_SECONDS,
                )
                self.assertNotIn("PYTHONPATH", kwargs["env"])
                generation = private / screening_runner.GENERATION_DIRNAME
                videos = generation / "videos"
                videos.mkdir(mode=0o700)
                for index in range(v3_protocol.CANDIDATE_COUNT):
                    (videos / f"{index:03d}.mp4").write_bytes(
                        b"synthetic-screening-video" + index.to_bytes(4, "big")
                    )
                (
                    generation / screening_runner.GENERIC_MANIFEST_BASENAME
                ).write_bytes(b"{}\n")
                return mock.Mock(returncode=0)

            def generic_manifest(*, context, generation_dir, prompt_path):
                del context, prompt_path
                return {}, [
                    generation_dir / "videos" / f"{index:03d}.mp4"
                    for index in range(v3_protocol.CANDIDATE_COUNT)
                ]

            patches = self._runner_validation_patches(fixture)
            with (
                patches[0],
                patches[1],
                patches[2],
                patches[3],
                patches[4],
                patches[5],
                patches[6],
                mock.patch.object(
                    screening_runner,
                    "_validate_generic_manifest",
                    side_effect=generic_manifest,
                ),
                mock.patch.object(
                    screening_runner,
                    "_decode_video",
                    side_effect=self._fake_decode,
                ),
                mock.patch.object(
                    screening_runner,
                    "_build_composite",
                    side_effect=self._fake_composite,
                ),
            ):
                result = screening_runner.run_screening(
                    project_root=project,
                    private_root=private,
                    python_executable="models/.wan-runtime/bin/python",
                    worker_count=1,
                    run=execute,
                )
            self.assertEqual(result["status"], "succeeded")
            self.assertEqual(result["candidate_count"], 576)
            generation = private / screening_runner.GENERATION_DIRNAME
            public = private / screening_runner.PUBLIC_PACKAGE_DIRNAME
            private_package = private / screening_runner.PRIVATE_PACKAGE_DIRNAME
            self.assertTrue((generation / "execution_succeeded_v3.json").exists())
            self.assertTrue(
                (private / screening_runner.CUDA_LOCK_BASENAME).exists()
            )
            self.assertFalse((project / v3_protocol.INVALID_OUTCOME).exists())
            self.assertEqual(
                {entry.name for entry in public.iterdir()},
                {
                    "media",
                    "composites",
                    screening_runner.REVIEW_TEMPLATE_BASENAME,
                    screening_runner.ANONYMOUS_INVENTORY_BASENAME,
                    screening_runner.COMPOSITE_INVENTORY_BASENAME,
                    screening_runner.PUBLIC_MANIFEST_BASENAME,
                },
            )
            self.assertEqual(
                {entry.name for entry in private_package.iterdir()},
                {
                    screening_runner.ANSWER_KEY_BASENAME,
                    screening_runner.CANDIDATE_BINDING_BASENAME,
                    screening_runner.PRIVATE_MANIFEST_BASENAME,
                    screening_runner.PACKAGE_COMMITMENT_BASENAME,
                },
            )
            first = (
                public / screening_runner.REVIEW_TEMPLATE_BASENAME
            ).read_text(encoding="utf-8").splitlines()[1].split(",")
            self.assertEqual(first[1], "media/s000.mp4")
            self.assertEqual(first[3], "composites/s000.jpg")
            self.assertTrue(all(value == "" for value in first[5:]))

    def test_generation_failure_is_terminal_and_same_version_cannot_retry(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO.parent) as directory:
            fixture, _ = _authorize_freezer_stage0(Path(directory))
            patches = self._runner_validation_patches(fixture)
            with (
                patches[0],
                patches[1],
                patches[2],
                patches[3],
                patches[4],
                patches[5],
                patches[6],
                self.assertRaisesRegex(
                    screening_runner.TerminalScreeningFailure,
                    "original_generation_failure",
                ),
            ):
                screening_runner.run_screening(
                    project_root=fixture["project"],
                    private_root=fixture["private"],
                    python_executable="models/.wan-runtime/bin/python",
                    worker_count=1,
                    run=lambda *args, **kwargs: mock.Mock(returncode=9),
                )
            invalid = json.loads(
                (
                    fixture["project"] / v3_protocol.INVALID_OUTCOME
                ).read_text(encoding="utf-8")
            )
            v3_protocol.validate_invalid_outcome(
                invalid,
                expected_stage0_sha256=v3_protocol.sha256_file(
                    fixture["wrapper"]
                ),
            )
            self.assertEqual(invalid["failure_phase"], "original_generation")
            self.assertTrue(
                (
                    fixture["private"]
                    / screening_runner.GENERATION_DIRNAME
                    / "execution_failed_v3.json"
                ).exists()
            )
            self.assertTrue(
                (
                    fixture["private"] / screening_runner.CUDA_LOCK_BASENAME
                ).exists()
            )
            before = sorted(
                entry.relative_to(fixture["private"]).as_posix()
                for entry in fixture["private"].rglob("*")
            )
            with self.assertRaises(FileExistsError):
                screening_runner.run_screening(
                    project_root=fixture["project"],
                    private_root=fixture["private"],
                    python_executable="models/.wan-runtime/bin/python",
                    worker_count=1,
                    run=lambda *args, **kwargs: mock.Mock(returncode=0),
                )
            self.assertEqual(
                before,
                sorted(
                    entry.relative_to(fixture["private"]).as_posix()
                    for entry in fixture["private"].rglob("*")
                ),
            )

    def test_package_failure_binds_generation_and_publishes_no_package(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO.parent) as directory:
            fixture, _ = _authorize_freezer_stage0(Path(directory))
            private = fixture["private"]

            def execute(command, **kwargs):
                del command, kwargs
                generation = private / screening_runner.GENERATION_DIRNAME
                videos = generation / "videos"
                videos.mkdir(mode=0o700)
                for index in range(v3_protocol.CANDIDATE_COUNT):
                    (videos / f"{index:03d}.mp4").write_bytes(
                        b"synthetic-screening-video" + index.to_bytes(4, "big")
                    )
                (
                    generation / screening_runner.GENERIC_MANIFEST_BASENAME
                ).write_bytes(b"{}\n")
                return mock.Mock(returncode=0)

            def generic_manifest(*, context, generation_dir, prompt_path):
                del context, prompt_path
                return {}, [
                    generation_dir / "videos" / f"{index:03d}.mp4"
                    for index in range(v3_protocol.CANDIDATE_COUNT)
                ]

            patches = self._runner_validation_patches(fixture)
            with (
                patches[0],
                patches[1],
                patches[2],
                patches[3],
                patches[4],
                patches[5],
                patches[6],
                mock.patch.object(
                    screening_runner,
                    "_validate_generic_manifest",
                    side_effect=generic_manifest,
                ),
                mock.patch.object(
                    screening_runner,
                    "_decode_video",
                    side_effect=self._fake_decode,
                ),
                mock.patch.object(
                    screening_runner,
                    "build_screening_package",
                    side_effect=ValueError("synthetic package integrity failure"),
                ),
                self.assertRaisesRegex(
                    screening_runner.TerminalScreeningFailure,
                    "screening_package_failure",
                ),
            ):
                screening_runner.run_screening(
                    project_root=fixture["project"],
                    private_root=private,
                    python_executable="models/.wan-runtime/bin/python",
                    worker_count=1,
                    run=execute,
                )
            invalid = json.loads(
                (
                    fixture["project"] / v3_protocol.INVALID_OUTCOME
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(invalid["failure_phase"], "screening_package")
            self.assertTrue(
                invalid["bound_artifacts"]["screening_generation_manifest"]
            )
            self.assertFalse(
                (private / screening_runner.PUBLIC_PACKAGE_DIRNAME).exists()
            )
            self.assertFalse(
                (private / screening_runner.PRIVATE_PACKAGE_DIRNAME).exists()
            )
            self.assertTrue(
                (
                    private
                    / screening_runner.GENERATION_DIRNAME
                    / screening_runner.GENERATION_MANIFEST_BASENAME
                ).exists()
            )

    def test_keyboard_interrupt_after_lock_is_terminalized(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO.parent) as directory:
            fixture, _ = _authorize_freezer_stage0(Path(directory))
            patches = self._runner_validation_patches(fixture)

            def interrupt(*args, **kwargs):
                del args, kwargs
                raise KeyboardInterrupt()

            with (
                patches[0],
                patches[1],
                patches[2],
                patches[3],
                patches[4],
                patches[5],
                patches[6],
                self.assertRaisesRegex(
                    screening_runner.TerminalScreeningFailure,
                    "original_generation_failure",
                ),
            ):
                screening_runner.run_screening(
                    project_root=fixture["project"],
                    private_root=fixture["private"],
                    python_executable="models/.wan-runtime/bin/python",
                    worker_count=1,
                    run=interrupt,
                )
            self.assertTrue(
                (fixture["project"] / v3_protocol.INVALID_OUTCOME).exists()
            )
            self.assertTrue(
                (
                    fixture["private"]
                    / screening_runner.GENERATION_DIRNAME
                    / "execution_failed_v3.json"
                ).exists()
            )

    def test_post_lock_reservation_failure_always_publishes_invalid_outcome(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO.parent) as directory:
            fixture, _ = _authorize_freezer_stage0(Path(directory))
            private = fixture["private"]

            def consume_lock_then_fail(context, worker_count):
                self.assertEqual(worker_count, 1)
                lock_path = private / screening_runner.CUDA_LOCK_BASENAME
                screening_runner._write_bytes_exclusive(
                    lock_path,
                    screening_runner._json_bytes(
                        {
                            "protocol": screening_runner.LOCK_PROTOCOL,
                            "dataset_version": v3_protocol.DATASET_VERSION,
                            "status": "consumed_for_one_shot_screening",
                            "stage0_registry_sha256": context.stage0_sha256,
                            "worker_count": 1,
                        }
                    ),
                    0o600,
                )
                raise screening_runner.ConsumedReservationFailure(
                    "synthetic mkdir failure after lock"
                )

            patches = self._runner_validation_patches(fixture)
            with (
                patches[0],
                patches[1],
                patches[2],
                patches[3],
                patches[4],
                patches[5],
                patches[6],
                mock.patch.object(
                    screening_runner,
                    "_reserve_execution",
                    side_effect=consume_lock_then_fail,
                ),
                self.assertRaisesRegex(
                    screening_runner.TerminalScreeningFailure,
                    "original_generation_reservation_failure",
                ),
            ):
                screening_runner.run_screening(
                    project_root=fixture["project"],
                    private_root=private,
                    python_executable="models/.wan-runtime/bin/python",
                    worker_count=1,
                )
            self.assertTrue(
                (private / screening_runner.CUDA_LOCK_BASENAME).exists()
            )
            self.assertFalse(
                (private / screening_runner.GENERATION_DIRNAME).exists()
            )
            invalid = json.loads(
                (
                    fixture["project"] / v3_protocol.INVALID_OUTCOME
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(invalid["failure_phase"], "original_generation")
            self.assertEqual(
                invalid["bound_artifacts"],
                {
                    "stage0_registry": v3_protocol.sha256_file(
                        fixture["wrapper"]
                    ),
                    "screening_generation_manifest": None,
                    "screening_package_commitment": None,
                    "screening_freeze_manifest": None,
                    "canonical_eligibility": None,
                    "selector_stderr": None,
                },
            )


if __name__ == "__main__":
    unittest.main()
