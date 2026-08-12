import numpy as np

from scripts.screen_dynamic_counterfactual_targets import technical_status, temporal_metrics


def test_rejects_repeated_static_frames() -> None:
    frames = np.zeros((49, 16, 16, 3), dtype=np.uint8)
    metrics = temporal_metrics(frames)
    assert technical_status(49, 8.0, metrics) == "reject_nearly_static"


def test_accepts_smooth_dynamic_frames() -> None:
    frames = np.stack(
        [np.full((16, 16, 3), index, dtype=np.uint8) for index in range(49)]
    )
    metrics = temporal_metrics(frames)
    assert technical_status(49, 8.0, metrics) == "candidate"


def test_rejects_wrong_video_shape_metadata() -> None:
    metrics = {
        "mean_adjacent_mae": 1.0,
        "first_last_mae": 2.0,
        "max_adjacent_mae": 1.5,
        "max_to_median_ratio": 1.5,
    }
    assert technical_status(48, 8.0, metrics) == "reject_invalid_video"
