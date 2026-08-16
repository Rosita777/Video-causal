from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import build_water_impact_dynamic_v3c_eval_split as builder  # noqa: E402
import water_impact_dynamic_v3c_eval_protocol as protocol  # noqa: E402


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class V3CSplitTests(unittest.TestCase):
    def test_partition_is_deterministic_stratified_disjoint_and_exhaustive(self) -> None:
        test_rows = rows(PROJECT_ROOT / protocol.TEST_PAIRS)
        eval12_rows = rows(PROJECT_ROOT / protocol.EXHAUSTED_EVAL12)
        fresh_a, final_a = builder.derive_partition(test_rows, eval12_rows)
        fresh_b, final_b = builder.derive_partition(test_rows, eval12_rows)
        self.assertEqual(fresh_a, fresh_b)
        self.assertEqual(final_a, final_b)
        self.assertEqual(len(fresh_a), 24)
        self.assertEqual(len(final_a), 36)
        counts = Counter(
            (row["generalization_group"], row["prompt_variant"])
            for _, row in fresh_a
        )
        self.assertEqual(
            counts,
            Counter(
                {
                    (group, variant): 4
                    for group in protocol.GENERALIZATION_GROUPS
                    for variant in protocol.PROMPT_VARIANTS
                }
            ),
        )
        exhausted = {row["pair_id"] for row in eval12_rows}
        fresh = {row["pair_id"] for _, row in fresh_a}
        final = {row["pair_id"] for _, row in final_a}
        self.assertFalse(exhausted & fresh)
        self.assertFalse(exhausted & final)
        self.assertFalse(fresh & final)
        self.assertEqual(exhausted | fresh | final, {row["pair_id"] for row in test_rows})

    def test_committed_stage1_registry_is_byte_exact(self) -> None:
        payload = protocol.validate_split_registration(PROJECT_ROOT)
        self.assertEqual(payload["gate_spec"], protocol.GATE_SPEC)
        self.assertEqual(payload["status"], "frozen_before_v3c_generation")
        self.assertEqual(len(protocol.SPLIT_REGISTRY_SHA256), 64)

    def test_another_validly_stratified_selection_cannot_impersonate_frozen_rank(self) -> None:
        test_rows = rows(PROJECT_ROOT / protocol.TEST_PAIRS)
        eval12_rows = rows(PROJECT_ROOT / protocol.EXHAUSTED_EVAL12)
        fresh, final = builder.derive_partition(test_rows, eval12_rows)
        actual = [
            {"pair_id": row["pair_id"]}
            for _, row in fresh
        ]
        first_group = fresh[0][1]["generalization_group"]
        first_variant = fresh[0][1]["prompt_variant"]
        replacement = next(
            row
            for _, row in final
            if row["generalization_group"] == first_group
            and row["prompt_variant"] == first_variant
        )
        actual[0] = {"pair_id": replacement["pair_id"]}
        with self.assertRaisesRegex(ValueError, "registered SHA-rank selection"):
            protocol.validate_exact_selection(actual, fresh, "fresh-dev")

    def test_split_builder_refuses_to_overwrite_frozen_outputs(self) -> None:
        with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
            builder.build(PROJECT_ROOT)


if __name__ == "__main__":
    unittest.main()
