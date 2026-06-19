from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


def test_annotation_fields_include_causal_footprint_labels():
    from build_baseline_comparison import ANNOTATION_FIELDS

    assert ANNOTATION_FIELDS == [
        "prompt_id",
        "baseline",
        "video_path",
        "prompt",
        "target_concept",
        "expected_effect",
        "target_visible",
        "causal_effect_visible",
        "causeless_effect",
        "video_quality",
        "usable_for_claim",
        "notes",
    ]


def test_evenly_spaced_indices_cover_requested_frame_count():
    from build_baseline_comparison import evenly_spaced_indices

    assert evenly_spaced_indices(total=49, count=7) == [0, 8, 16, 24, 32, 40, 48]
    assert evenly_spaced_indices(total=3, count=7) == [0, 1, 2]
