from scripts.finalize_dynamic_counterfactual_screen import REJECTIONS


def test_rejection_list_is_explicit_and_unique() -> None:
    assert len(REJECTIONS) == 14
    assert all(0 <= index < 192 for index in REJECTIONS)
    assert all(reason.strip() for reason in REJECTIONS.values())
