from scripts.build_water_impact_dynamic_pairs_v1 import build_rows


def test_dynamic_pair_counts_and_generalization_splits() -> None:
    train_rows, test_rows = build_rows()

    assert len(train_rows) == 192
    assert len(test_rows) == 72
    assert {row["generalization_group"] for row in test_rows} == {
        "unseen_source",
        "unseen_receiver",
        "unseen_source_and_receiver",
    }
    assert all(
        sum(row["generalization_group"] == group for row in test_rows) == 24
        for group in {
            "unseen_source",
            "unseen_receiver",
            "unseen_source_and_receiver",
        }
    )


def test_unseen_entities_do_not_leak_into_training() -> None:
    train_rows, test_rows = build_rows()
    train_sources = {row["source_id"] for row in train_rows}
    train_receivers = {row["receiver_id"] for row in train_rows}
    unseen_sources = {
        row["source_id"] for row in test_rows if row["source_seen"] == "no"
    }
    unseen_receivers = {
        row["receiver_id"] for row in test_rows if row["receiver_seen"] == "no"
    }

    assert train_sources.isdisjoint(unseen_sources)
    assert train_receivers.isdisjoint(unseen_receivers)


def test_training_prompts_and_seeds_are_unique() -> None:
    train_rows, _ = build_rows()

    assert len({row["pair_id"] for row in train_rows}) == len(train_rows)
    assert len({row["training_prompt"] for row in train_rows}) == len(train_rows)
    assert len({row["seed"] for row in train_rows}) == len(train_rows)
    assert all("first two seconds" not in row["training_prompt"] for row in train_rows)
    assert all(
        "first two seconds" not in row["target_generation_prompt"] for row in train_rows
    )
    banned_target_words = {"falls", "impact", "splash", "cavity", "ripple", "ring"}
    assert all(
        not any(word in row["target_generation_prompt"].lower() for word in banned_target_words)
        for row in train_rows
    )
    assert all("camera" not in row["target_generation_prompt"].lower() for row in train_rows)
