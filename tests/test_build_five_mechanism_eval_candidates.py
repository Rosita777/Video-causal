from collections import Counter, defaultdict

from scripts.build_five_mechanism_eval_candidates import (
    MECHANISMS,
    build_rows,
    ink_stain_smoke_rows,
    mixed_smoke_rows_v2,
    simple_smoke_rows,
    smoke_rows,
)


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


def test_smoke_rows_cover_every_mechanism():
    rows = smoke_rows(build_rows())

    assert len(rows) == 10
    assert Counter(row["mechanism"] for row in rows) == {name: 2 for name in MECHANISMS}


def test_simple_smoke_uses_short_prompts_and_high_contrast_surfaces():
    rows = simple_smoke_rows(build_rows())

    assert len(rows) == 10
    assert Counter(row["mechanism"] for row in rows) == {name: 2 for name in MECHANISMS}
    assert max(len(row["prompt"].split()) for row in rows) < 35
    particle_receivers = [row["receiver"] for row in rows if row["mechanism"] == "blue_ball_particles"]
    assert all("white sand" not in receiver for receiver in particle_receivers)


def test_mixed_v2_replaces_trace_with_ink_stain():
    rows = mixed_smoke_rows_v2(build_rows())

    assert len(rows) == 10
    assert Counter(row["mechanism"] for row in rows) == {
        "waterdrop_impact": 2,
        "red_ball_collision": 2,
        "steel_ball_fracture": 2,
        "blue_ball_particles": 2,
        "ink_droplet_stain": 2,
    }
    assert all(row["mechanism"] != "toy_car_trace" for row in rows)
    assert all("Only after contact" in row["prompt"] for row in ink_stain_smoke_rows())
