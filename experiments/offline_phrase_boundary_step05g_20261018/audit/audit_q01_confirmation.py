"""Bind the later positive Q01 answer without changing earlier snapshots."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from source_preflight import HERE, PREVIOUS, bind, check, read_json


def inspect() -> dict[str, Any]:
    """Check the revised-audio-only approval and current progress handoff."""
    stage = HERE.parent
    path = stage / "user_evidence/002_Q01_revised_audio_confirmed.json"
    evidence = read_json(path)
    question = read_json(Path(evidence["question_manifest"]["path"]))
    revision = read_json(Path(evidence["revision_record"]["path"]))
    initial = read_json(HERE / "Q01_revision_audit.json")
    summary_path = stage / "results/stage5g_summary_r1.json"
    handoff_path = stage / "results/next_step_handoff_r1.json"
    summary = read_json(summary_path)
    handoff = read_json(handoff_path)
    old_rows = {r["review_id"]: r for r in read_json(PREVIOUS / "results/stage5f_summary.json")["phrase_proposals"]}
    for row in (evidence["question_manifest"], evidence["question_audio"], evidence["revision_record"],
                initial["evidence"], initial["revision"], initial["review_question"], handoff["summary"],
                handoff["Q01_approved_audio"], *summary["evidence"]):
        check(row)
    assert evidence["question"] == question["question"]
    assert evidence["answer"] == "온전히 들림·잘림·혼입 없음"
    assert json.loads(evidence["question_item_id"]) == ["request_user_input_async", "call_1b490b8408464cb3af9030edd01a27ef", 0]
    assert evidence["review_id"] == revision["review_id"] == "Q01"
    assert evidence["revision_id"] == revision["revision_id"] == "Q01_end_parent_r1"
    assert evidence["question_audio"] == handoff["Q01_approved_audio"] == bind(Path(question["question_audio"]["path"]))
    assert evidence["phrase_audio_boundary_approved"] is True
    assert evidence["phrase_visual_boundary_approved"] is evidence["original_candidate_approved"] is evidence["training_ready"] is False
    assert initial["new_candidate_approval"] is False
    assert question["actual_revision_answer"] is None
    assert question["phrase_audio_boundary_approved"] is revision["phrase_audio_boundary_approved"] is False
    assert len(list((stage / "user_evidence").glob("*.json"))) == summary["actual_phrase_answers"] == 2
    assert summary["original_end_clipping_reports"] == 1
    assert summary["selected_phrase_audio_confirmed"] == 1
    assert summary["selected_phrase_visual_confirmed"] == 0
    assert summary["stage5g_complete"] is False
    assert summary["status"] == "in_progress_Q01_revised_audio_confirmed_remaining_phrase_review_pending"
    assert [r["review_id"] for r in summary["records"]] == [f"Q{i:02}" for i in range(1, 9)]
    for row in summary["records"]:
        prior = old_rows[row["review_id"]]
        for key in ("review_id", "parent_id", "candidate_id", "text"):
            assert row[key] == prior[key]
        if row["review_id"] == "Q01":
            assert row["selected_audio"] == evidence["question_audio"]
            assert row["audio_status"] == "confirmed_revised_audio"
            assert row["human_phrase_audio_boundary_approved"] is True
        else:
            assert row["selected_audio"] == prior["audio"]
            assert row["human_phrase_audio_boundary_approved"] is False
            expected_status = "question_answer_pending" if row["review_id"] in ("Q02", "Q03") else "question_not_yet_asked"
            assert row["audio_status"] == expected_status
        assert row["human_phrase_visual_boundary_approved"] is row["training_ready"] is False
        assert all(row[k] is None for k in ("actual_split", "frame_count", "ctc_feasible"))
    assert summary["Q01_end_extension_samples_16k"] == initial["extension_samples_16k"] == 3288
    assert summary["Q01_end_extension_seconds"] == initial["extension_seconds"] == 0.2055
    assert summary["Q01_new_duration_seconds"] == initial["new_candidate_seconds"] == 1.4855
    assert summary["new_neural_calls"] == summary["new_tar_payload_bytes"] == summary["training_updates"] == summary["new_training_cache_files"] == 0
    assert summary["new_video_created"] is summary["training_ready"] is False
    assert handoff["current_audio_confirmed"] == ["Q01"] and handoff["current_visual_confirmed"] == []
    assert handoff["not_yet_asked"] == ["Q04", "Q05", "Q06", "Q07", "Q08"]
    for idx, row in enumerate(handoff["pending_existing_questions"], 1):
        assert row["review_id"] == f"Q{idx + 1:02}"
        assert json.loads(row["question_item_id"]) == ["request_user_input_async", "call_c6b912f45a194f748b5c9e8dc2e486de", idx]
    assert handoff["Q01_old_video_is_not_revised_boundary_preview"] is True
    assert handoff["no_automatic_training_or_cache_or_Q09_Q20_extraction"] is True
    return {
        "schema": "offline-stage5g-independent-Q01-confirmation-audit-v1", "status": "passed",
        "actual_confirmation": bind(path), "summary": bind(summary_path), "handoff": bind(handoff_path),
        "earlier_revision_audit": bind(HERE / "Q01_revision_audit.json"),
        "actual_phrase_answers": 2, "confirmed_revised_phrase_audio_ids": ["Q01"],
        "original_Q01_candidate_approved": False, "visual_approved_ids": [],
        "pending_existing_question_ids": ["Q02", "Q03"], "not_yet_asked_ids": handoff["not_yet_asked"],
        "historical_pending_snapshots_preserved": True, "stage5g_complete": False,
        "checks_repeated": "Small evidence and question bindings only; no repeated PCM/HTTP, historical 247 hashes, TAR or model execution.",
        "training_ready": False,
    }


if __name__ == "__main__":
    result = inspect()
    with (HERE / "Q01_confirmation_audit.json").open("x", encoding="utf8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "actual_answers": 2, "audio_confirmed": ["Q01"], "visual_confirmed": []}))
