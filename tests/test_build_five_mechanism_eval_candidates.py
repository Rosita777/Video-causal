from collections import Counter, defaultdict

from scripts.build_five_mechanism_eval_candidates import MECHANISMS, build_rows


def test_builds_balanced_candidate_pool():
    rows = build_rows()

    assert len(rows) == 150
    assert Counter(row["mechanism"] for row in rows) == {name: 30 for name in MECHANISMS}
    assert len({row["candidate_id"] for row in rows}) == 150
    assert all(row["generation_repetitions"] == "1" for row in rows)
    assert all(row["intended_split"] == "evaluation_candidate_only" for row in rows)


def test_each_mechanism_has_one_target_and_unique_receivers():
    grouped = defaultdict(list)
    for row in build_rows():
        grouped[row["mechanism"]].append(row)

    for rows in grouped.values():
        assert len({row["target_concept"] for row in rows}) == 1
        assert len({row["receiver"] for row in rows}) == 30


def test_prompts_state_temporal_causal_order():
    for row in build_rows():
        assert row["target_concept"] in row["prompt"]
        assert row["receiver"] in row["prompt"]
        assert row["expected_footprint"] in row["prompt"]
        assert "Only" in row["prompt"]
