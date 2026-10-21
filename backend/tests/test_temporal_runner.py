"""11단계의 표본 선정·test 차단·특징 비교 의미를 검증한다."""
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from diagnose_temporal_use import select_cohorts, verify_cache, feature_distance, feature_stats, length_bin


def test_selection_is_order_independent_and_train_only():
    rows = [{"id": f"train{i:02d}", "split": "train", "speaker_id": f"S{i % 2}",
             "frames": 20 + i * 3} for i in range(20)]
    rows += [{"id": "heldout", "split": "test", "speaker_id": "T", "frames": 1}]
    rows += [{"id": "development", "split": "val", "speaker_id": "V", "frames": 4}]
    first, second = select_cohorts(rows), select_cohorts(list(reversed(rows)))
    for key in ("short16", "train_stratified", "train_controls_union", "strata"):
        assert first[key] == second[key]
    assert first["short16"] == [f"train{i:02d}" for i in range(16)]
    assert "heldout" not in first["train_controls_union"]
    assert "development" not in first["train_controls_union"]
    assert len(first["train_controls_union"]) == len(set(first["train_stratified"] + first["short16"]))


def test_test_cache_is_rejected_before_any_filesystem_access():
    with pytest.raises(ValueError, match="test"):
        verify_cache({"split": "test"})


def test_constant_feature_has_no_temporal_variation():
    stats = feature_stats(np.full((4, 3), 2.0))
    assert stats["rms"] == 2
    assert stats["temporal_centered_rms"] == 0
    assert stats["adjacent_rms"] == 0


def test_source_frame_correspondence_is_different_from_position_comparison():
    original = np.asarray([[1., 0.], [0., 2.], [3., 0.]])
    permutation = np.asarray([2, 0, 1])
    shuffled = original[permutation]
    assert feature_distance(original, shuffled)["rms_difference"] > 0
    assert feature_distance(original[permutation], shuffled)["rms_difference"] == 0


@pytest.mark.parametrize("frames,expected", [(1, "001-050"), (50, "001-050"),
    (51, "051-100"), (100, "051-100"), (101, "101-150"), (150, "101-150"),
    (151, "151-200"), (200, "151-200")])
def test_length_bins_cover_boundaries(frames, expected):
    assert length_bin(frames) == expected


@pytest.mark.parametrize("frames", [0, 201])
def test_out_of_contract_lengths_fail(frames):
    with pytest.raises(ValueError):
        length_bin(frames)
