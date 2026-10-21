"""Regression checks for actual Q02 origin and precise parent-answer scope."""
from copy import deepcopy
from fractions import Fraction

import pytest

from protocol import STAGE, read
from build_bound_plan import checked_answer, global_audio_origin


def inputs():
    """Read small bound metadata only; no PCM or model arrays are loaded."""
    rows = read(STAGE / "effective_parent_inputs.json")["records"]
    answers = {r["evidence_id"]: r for r in read(STAGE / "user_evidence/parent_answers.json")["records"]}
    return deepcopy(rows), deepcopy(answers)


def test_q02_repair_retains_absolute_recording_origin():
    rows, _ = inputs()
    row = rows[1]
    assert row["audio"]["source_first_sample"] == 16800
    assert global_audio_origin(row) == Fraction(13559808, 48000)
    assert global_audio_origin(row) != Fraction(16800, 48000)


def test_context_relative_index_cannot_replace_global_origin():
    rows, _ = inputs()
    row = rows[1]
    row["original_recording_first_sample_48k"] = 16800
    row["audio_origin_seconds_rational"] = {"numerator": 16800, "denominator": 48000}
    with pytest.raises(ValueError):
        global_audio_origin(row)


def test_all_eight_actual_parent_answers_bind_exact_wavs():
    rows, answers = inputs()
    assert len(rows) == 8 and len(answers) == 9
    for row in rows:
        answer = checked_answer(row, answers)
        assert answer["question_audio"]["sha256"] == row["audio"]["sha256"]
        assert not answer["phrase_audio_boundary_approved"]


def test_q02_original_clipped_answer_cannot_approve_repair():
    rows, answers = inputs()
    row = rows[1]
    row["actual_answer_evidence_id"] = "parent_answer_02"
    row["actual_answer_question_item_id"] = answers["parent_answer_02"]["question_item_id"]
    with pytest.raises(ValueError):
        checked_answer(row, answers)


def test_phrase_scope_cannot_be_promoted_by_parent_record():
    rows, answers = inputs()
    rows[0]["phrase_audio_boundary_approved"] = True
    with pytest.raises(ValueError):
        checked_answer(rows[0], answers)


def test_same_transcript_different_audio_does_not_reuse_answer():
    rows, answers = inputs()
    rows[0]["audio"]["sha256"] = "0" * 64
    with pytest.raises(ValueError):
        checked_answer(rows[0], answers)
