"""Boundary and geometric diagnostic tests for the independent evaluation."""
import pytest

from prepare_step03_inputs import bbox_diagnostics, native_bounds


@pytest.mark.parametrize("start,end,fps", [
    (0.0,1.0,30.0),(246.997333,249.216,30000/1001),
    (10.069333333333333,11.477333333333334,30000/1001),
    (1/30,3/30,30.0),
])
def test_frame_selection_matches_half_open_timestamp_interval(start,end,fps):
    first,stop=native_bounds(start,end,fps)
    assert first/fps >= start-1e-9
    assert (first-1)/fps < start
    assert (stop-1)/fps < end
    assert stop/fps >= end-1e-9
    assert first/fps-start < 1/fps+1e-9
    assert stop/fps-end < 1/fps+1e-9


def test_long_recording_avoids_nominal30_drift():
    start,end,fps=246.997333,249.216,30000/1001
    first,stop=native_bounds(start,end,fps)
    assert first < int(start*30)
    assert stop < int(end*30)
    assert int(end*30)/fps-end > .2
    assert stop/fps-end < 1/fps


def test_comparison_accounts_for_margin_without_treating_it_as_failure():
    metrics=bbox_diagnostics([10,10,20,20],[5,5,25,25])
    assert metrics["iou"]==.25
    assert metrics["area_ratio_auto_to_annotation"]==4
    assert bbox_diagnostics([10,10,20,20],None)["iou"] is None


@pytest.mark.parametrize("start,end,fps",[(1,0,30),(0,1,0),(0,0,30),(-1,1,30)])
def test_invalid_intervals_rejected(start,end,fps):
    with pytest.raises(ValueError):
        native_bounds(start,end,fps)
