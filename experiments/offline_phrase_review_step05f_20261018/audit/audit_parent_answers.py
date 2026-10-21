"""Verify trusted caller's actual answers against exact questions/audio variants."""
from __future__ import annotations

from fractions import Fraction
import json
from pathlib import Path
from typing import Any
import wave

from audit_preflight import HERE, bind, check_binding, read_json


def inspect() -> dict[str, Any]:
    base = HERE.parent
    answers_path = base / "user_evidence/parent_answers.json"
    effective_path = base / "effective_parent_inputs.json"
    answers, effective = read_json(answers_path), read_json(effective_path)
    for source in answers["sources"] + effective["sources"]:
        check_binding(source)
    original = {r["review_id"]: r for r in read_json(base / "parent_review_manifest.json")["records"]}
    revision_path = base / "media_revisions/Q02_end_plus120ms/record.json"
    revision = read_json(revision_path)
    rows = answers["records"]
    assert len(rows) == answers["actual_answer_count"] == 9
    assert len({r["question_item_id"] for r in rows}) == 9
    assert [r["review_id"] for r in rows] == ["Q01", "Q02", "Q03", "Q04", "Q05", "Q06", "Q02", "Q07", "Q08"]
    assert rows[1]["answer"] == '끝부분 "뻐"가 거의 안들리고 잘림'
    assert rows[6]["answer"] == "문장 일치·마지막까지 온전함·혼입 없음"
    assert all(r["answer"] == "문장 일치·잘림·혼입 없음" for i, r in enumerate(rows) if i not in [1, 6])
    expected_calls = [
        ("call_d217164aaf344e25a5eafcc0faa6650e", i) for i in range(3)
    ] + [("call_5ad016f14413435f980d05b5c843df71", i) for i in range(3)] + [
        ("call_3fee2d64d5d14e7db14289988c2454b8", i) for i in range(3)
    ]
    for index, (row, call) in enumerate(zip(rows, expected_calls, strict=True)):
        assert json.loads(row["question_item_id"]) == ["request_user_input_async", *call]
        q = original[row["review_id"]]
        assert row["parent_id"] == q["parent_id"]
        revised = index == 6
        assert row["revised_parent_audio"] is revised
        expected_audio = revision["sentence_audio"] if revised else q["media"]["sentence_audio"]
        assert row["question_audio"] == bind(Path(expected_audio["path"]))
        if not revised:
            assert row["question"] == q["question"]
            assert row["question_media_record"] == q["media_record"]
        else:
            assert "끝을 0.12초 늘린 새 음성" in row["question"]
            assert q["parent_text"] in row["question"]
            assert row["question_media_record"] == bind(revision_path)
        check_binding(row["question_media_record"])
        passed = index != 1
        assert row["parent_transcript_confirmed"] is passed
        assert row["parent_audio_qualitatively_confirmed"] is passed
        assert row["original_parent_end_clipping_reported"] is (not passed)
        assert all(row[k] is False for k in ["phrase_audio_boundary_approved", "phrase_visual_boundary_approved", "precision_articulation_boundary_certified", "training_ready"])
    assert sum(r["parent_transcript_confirmed"] for r in rows) == 8
    assert sum(r["original_parent_end_clipping_reported"] for r in rows) == 1
    assert len(effective["records"]) == 8
    selected_samples = 0
    for number, row in enumerate(effective["records"], 1):
        assert row["review_id"] == f"Q{number:02}"
        positive = [r for r in rows if r["review_id"] == row["review_id"] and r["parent_transcript_confirmed"]]
        assert len(positive) == 1
        evidence = positive[0]
        assert row["actual_answer_evidence_id"] == evidence["evidence_id"]
        assert row["actual_answer_question_item_id"] == evidence["question_item_id"]
        assert bind(Path(row["audio"]["path"])) == evidence["question_audio"]
        old = read_json(Path(row["media_record"]["path"]))
        assert row["parent_text"] == old["parent_text"]
        assert row["phrase_text"] == old["proposed_phrase"]
        assert row["word_start"] == old["word_start"] and row["word_end"] == old["word_end"]
        assert row["video_offset_native_samples"] == old["offset_native_samples"]
        first = row["original_recording_first_sample_48k"]
        end = row["original_recording_end_sample_48k"]
        assert first == old["parent_native_start_sample"]
        if number == 2:
            assert row["effective_revision_record"] == bind(revision_path)
            assert row["audio"] == revision["sentence_audio"]
            assert end == old["parent_native_end_sample"] + 5760
            assert first == 13559808 and Fraction(first, 48000) == Fraction(35312, 125)
        else:
            assert row["effective_revision_record"] is None
            assert row["audio"] == old["sentence_audio"]
            assert end == old["parent_native_end_sample"]
        assert Fraction(**row["audio_origin_seconds_rational"]) == Fraction(first, 48000)
        with wave.open(row["audio"]["path"], "rb") as stream:
            assert (stream.getframerate(), stream.getnchannels(), stream.getsampwidth(), stream.getnframes()) == (16000, 1, 2, row["audio"]["samples"])
        selected_samples += row["audio"]["samples"]
        assert row["parent_transcript_confirmed"] and row["parent_audio_qualitatively_confirmed"]
        assert not row["phrase_audio_boundary_approved"] and not row["phrase_visual_boundary_approved"]
        assert row["split"] is None and row["training_ready"] is False and row["offset_applied_to_audio"] is False
    assert selected_samples == effective["selected_audio_samples_16k"] == 446549
    assert effective["selected_audio_duration_seconds"] == selected_samples / 16000
    return {"schema": "offline-stage5f-independent-parent-answer-audit-v1", "status": "passed",
            "answers": bind(answers_path), "effective_inputs": bind(effective_path),
            "actual_answer_bindings_verified": 9, "effective_parents": 8,
            "preserved_original_end_clipping_reports": 1,
            "selected_native_revisions": 1, "parent_text_corrections": 0,
            "effective_audio_samples_16k": selected_samples,
            "effective_audio_seconds": selected_samples / 16000,
            "Q02_global_source_origin_seconds": 282.496,
            "phrase_audio_approvals": 0, "phrase_visual_approvals": 0,
            "source_authority_limit": "Actual-answer provenance is supplied by the trusted root chat receiving the user responses; this independent audit checks literal records, IDs, question/audio bindings and scoped consequences, not authentication from a self-asserted JSON source label.",
            "training_ready": False}


if __name__ == "__main__":
    result = inspect()
    with (HERE / "parent_answer_audit.json").open("x", encoding="utf8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "answers": 9, "effective_parents": 8, "audio_samples": result["effective_audio_samples_16k"]}))
