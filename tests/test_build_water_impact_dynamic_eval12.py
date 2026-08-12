from collections import Counter

from scripts.build_water_impact_dynamic_eval12 import SELECTED_INDICES
from scripts.build_water_impact_dynamic_pairs_v1 import build_rows


def test_eval12_balances_generalization_groups() -> None:
    _, test_rows = build_rows()
    rows = [test_rows[index] for index in SELECTED_INDICES]
    assert len(rows) == 12
    assert Counter(row["generalization_group"] for row in rows) == {
        "unseen_source": 4,
        "unseen_receiver": 4,
        "unseen_source_and_receiver": 4,
    }
    assert {row["prompt_variant"] for row in rows} == {"direct", "natural"}
