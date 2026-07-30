from pathlib import Path
import sys

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from build_static_counterfactual_pair import build_reference_frame  # noqa: E402


def test_build_reference_frame_uses_temporal_median():
    frames = np.array(
        [
            [[[10, 20, 30]]],
            [[[12, 22, 32]]],
            [[[200, 200, 200]]],
        ],
        dtype=np.uint8,
    )

    reference = build_reference_frame(frames, 0, 2)

    assert reference.tolist() == [[[11, 21, 31]]]


def test_build_reference_frame_rejects_invalid_range():
    frames = np.zeros((2, 4, 4, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="reference range"):
        build_reference_frame(frames, 0, 3)
