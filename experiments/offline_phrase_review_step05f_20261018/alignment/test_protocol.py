"""Boundary arithmetic and safety-scope checks with synthetic, non-media inputs."""
from fractions import Fraction
import itertools
import math

import numpy as np
import pytest

from protocol import acoustic_oov, fraction, load_ctc, selected_phrase, span_geometry
from run_alignment import validate_plan


def test_exact_source_origin_and_offset_once():
    origin, offset = Fraction(1001, 48000), Fraction(1024, 48000)
    value = span_geometry(1, 3, 16000, origin, offset)
    assert value["local_audio_start_sample_16k"] == 360
    assert value["local_audio_end_sample_16k"] == 1000
    assert fraction(value["wav_start_seconds_exact"]) == Fraction(2081, 48000)
    assert fraction(value["video_start_seconds_exact"]) == Fraction(3105, 48000)
    assert value["video_offset_applied_count"] == 1
    assert value["frame_count"] is None and value["ctc_feasible"] is None


@pytest.mark.parametrize("indices", [(-1, 2, 16000), (3, 3, 16000), (1, 5, 1000), (True, 2, 16000)])
def test_outside_or_invalid_occupancy_rejected(indices):
    with pytest.raises(ValueError):
        span_geometry(*indices, Fraction(0), Fraction(0))


def test_acoustic_oov_does_not_delete_or_substitute_text():
    text = "탔어"
    assert acoustic_oov(text, {"어": 1, "|": 2}) == ["탔"]
    assert text == "탔어"


def test_whole_parent_phrase_keeps_separate_boundary_review():
    spans = [{"start_frame": i * 2, "end_frame": i * 2 + 1} for i in range(len("결혼해서 너무 좋아"))]
    row = selected_phrase("결혼해서 너무 좋아", "결혼해서 너무 좋아", 0, 3,
                          {"token_spans": spans}, 16000, Fraction(1, 48000), Fraction(1024, 48000))
    assert row["phrase_equals_whole_parent_text"]
    assert row["parent_review_is_separate_from_phrase_boundary_review"]
    assert not row["human_phrase_audio_boundary_approved"]
    assert row["local_audio_start_sample_16k"] == 40
    assert row["local_audio_end_sample_16k"] != 16000


def test_same_phrase_text_uses_selected_word_occurrence():
    text = "너무 좋아 너무 좋아"
    spans = [{"start_frame": i, "end_frame": i + 1} for i in range(len(text))]
    row = selected_phrase(text, "너무 좋아", 2, 4, {"token_spans": spans}, 16000, Fraction(0), Fraction(0))
    assert row["char_start"] == 6
    assert row["local_audio_start_sample_16k"] == 1960


def test_changed_parent_requires_explicit_phrase_remapping():
    with pytest.raises(ValueError):
        selected_phrase("정말 잘 돼 있어", "정말 잘 되어 있어", 0, 4,
                        {"token_spans": []}, 16000, Fraction(0), Fraction(0))


@pytest.mark.parametrize("value", [{"numerator": 1., "denominator": 3},
                                    {"numerator": 1, "denominator": 0},
                                    {"numerator": True, "denominator": 3}])
def test_rational_rejects_float_bool_and_zero(value):
    with pytest.raises(ValueError):
        fraction(value)


def test_frozen_dp_matches_exhaustive_repeat_target():
    ctc = load_ctc()
    p = np.array([[.2, .8], [.7, .3], [.4, .6], [.8, .2]], dtype=np.float64)
    target = [1, 1]
    scores = [sum(math.log(p[t, token]) for t, token in enumerate(path))
              for path in itertools.product(range(2), repeat=4)
              if ctc.collapse_ctc(list(path), 0) == target]
    result = ctc.forced_align(np.log(p), target, 0)
    assert result["best_log_score"] == pytest.approx(max(scores), abs=1e-12)


def test_prepared_document_is_not_execution_authorization():
    with pytest.raises(ValueError):
        validate_plan({"schema": "offline-stage5f-alignment-preparation-v1"})
    with pytest.raises(ValueError):
        validate_plan({"schema": "offline-stage5f-authorized-acoustic-plan-v1", "neural_execution_authorized": False})
