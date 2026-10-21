"""Audit the final Q06 reply and current r3 review state with bounded checks."""
from __future__ import annotations

import json
from pathlib import Path
import urllib.error
import urllib.request
from typing import Any

from source_preflight import HERE, bind, check, read_json


def inspect() -> dict[str, Any]:
    """Check new evidence, current selections, and only the small r2 snapshot."""
    stage = HERE.parent
    summary_path = stage / "results/stage5g_summary_r3.json"
    handoff_path = stage / "results/next_step_handoff_r3.json"
    ui_path = stage / "ui/r3/manifest.json"
    summary, handoff, ui = (read_json(p) for p in (summary_path, handoff_path, ui_path))
    confirmed = ["Q01", "Q04", "Q05", "Q06"]
    evidence_path = stage / "user_evidence/008_Q06_plus40ms_audio_confirmed.json"
    evidence = read_json(evidence_path)
    for key in ("question_manifest", "question_audio", "revision_record"):
        check(evidence[key])
    question = read_json(Path(evidence["question_manifest"]["path"]))["records"][0]
    assert evidence["question"] == question["question"]
    assert evidence["question_audio"] == question["question_audio"]
    assert evidence["revision_record"] == question["revision"]
    assert json.loads(evidence["question_item_id"]) == ["request_user_input_async", "call_8767739030ba4feeb8192d672083921e", 0]
    assert evidence["answer"] == "온전히 들림·잘림·혼입 없음"
    assert evidence["phrase_audio_boundary_approved"] is True
    assert evidence["phrase_visual_boundary_approved"] is evidence["original_candidate_approved"] is evidence["training_ready"] is False
    assert question["actual_answer"] is None
    assert question["phrase_audio_boundary_approved"] is False
    trial = read_json(Path(evidence["revision_record"]["path"]))
    assert trial["phrase_audio_boundary_approved"] is False
    assert trial["revision_id"] == evidence["revision_id"] == "Q06_end_plus40ms_r1"
    previous_audit = read_json(HERE / "Q05_confirmation_Q06_midpoint_audit.json")
    assert previous_audit["new_trial"] == evidence["revision_record"]
    for row in previous_audit["actual_answers"]:
        check(row)
    assert len(summary["evidence"]) == summary["actual_phrase_answers"] == 8
    for row in summary["evidence"]:
        check(row)
    assert summary["original_end_clipping_reports"] == 3
    assert summary["revised_other_speech_mixing_reports"] == 1
    assert summary["selected_phrase_audio_confirmed"] == 4
    assert summary["selected_phrase_visual_confirmed"] == 0
    assert summary["stage5g_complete"] is summary["training_ready"] is False
    assert "free-text" in summary["actual_phrase_answer_units_note"]
    assert summary["new_review_videos_this_revision"] == 0
    assert summary["new_neural_calls"] == summary["new_tar_payload_bytes"] == summary["training_updates"] == summary["new_training_cache_files"] == 0
    assert handoff["summary"] == ui["summary"] == bind(summary_path)
    assert handoff["current_audio_confirmed"] == confirmed
    assert handoff["current_visual_confirmed"] == []
    assert [r["review_id"] for r in handoff["pending_existing_questions"]] == ["Q02", "Q03"]
    assert handoff["not_yet_asked"] == ["Q07", "Q08"]
    assert handoff["matching_visual_preview_not_yet_prepared"] == ["Q05", "Q06"]
    assert handoff["Q05_Q06_old_videos_must_not_represent_revised_boundaries"] is True
    assert handoff["no_automatic_training_or_cache_or_Q09_Q20_extraction"] is True
    pointer = read_json(stage / "ui/current.json")
    assert pointer["manifest"] == bind(ui_path)
    assert len(ui["records"]) == 8 and len(ui["routes"]) == 29
    by_id = {r["review_id"]: r for r in ui["records"]}
    for row in summary["records"]:
        actual = by_id[row["review_id"]]
        assert actual["text"] == row["text"]
        assert actual["phrase_audio_confirmed"] == row["human_phrase_audio_boundary_approved"] == (row["review_id"] in confirmed)
        assert actual["phrase_visual_confirmed"] is row["human_phrase_visual_boundary_approved"] is False
        assert actual["media"]["phrase_audio"] == row["selected_audio"]
        assert row["training_ready"] is False
        assert all(row[k] is None for k in ("actual_split", "frame_count", "ctc_feasible"))
        for role, media in actual["media"].items():
            check(media)
            assert ui["routes"][actual["urls"][role]] == media
    for qid, directory in [("Q05", "Q05_end_confirmed_parent"), ("Q06", "Q06_end_plus40ms")]:
        row = by_id[qid]
        assert set(row["media"]) == set(row["urls"]) == {"phrase_audio", "context_audio"}
        revision = read_json(stage / "revisions" / directory / "record.json")
        assert row["media"]["phrase_audio"] == bind(Path(revision["media"]["phrase"]["path"]))
        assert row["media"]["context_audio"] == bind(Path(revision["media"]["context"]["path"]))
    check(ui["routes"]["/"])
    page = Path(ui["routes"]["/"]["path"]).read_text(encoding="utf8")
    assert page.count("음성 확인 완료 · 입술 검수 대기") == 4
    assert "<form" not in page
    base_url = ui["current_url"]
    assert base_url == summary["current_review_url"] == handoff["current_review_url"] == "http://127.0.0.1:8775/"
    for qid in ("Q05", "Q06"):
        row = by_id[qid]
        req = urllib.request.Request(base_url.rstrip("/") + row["urls"]["phrase_audio"], headers={"Range": "bytes=0-127"})
        with urllib.request.urlopen(req, timeout=5) as response:
            assert response.status == 206
            assert response.read() == Path(row["media"]["phrase_audio"]["path"]).read_bytes()[:128]
    old_ui = read_json(stage / "ui/r2/manifest.json")
    old_q05 = next(r for r in old_ui["records"] if r["review_id"] == "Q05")
    stale_url = base_url.rstrip("/") + old_q05["urls"]["phrase_video"]
    try:
        urllib.request.urlopen(stale_url, timeout=5)
    except urllib.error.HTTPError as exc:
        assert exc.code == 404
    else:
        raise AssertionError("Old Q05 boundary video is still served by current UI")
    snapshot_path = stage / "results/review_snapshot_manifest_r2.json"
    snapshot = read_json(snapshot_path)
    assert len(snapshot["artifacts"]) == snapshot["artifact_count"] == 42
    for row in snapshot["artifacts"]:
        check(row)
    return {"schema": "offline-stage5g-independent-progress-ui-r3-audit-v1", "status": "passed",
            "summary": bind(summary_path), "handoff": bind(handoff_path), "ui_manifest": bind(ui_path),
            "latest_actual_answer": bind(evidence_path), "r2_preservation_manifest": bind(snapshot_path),
            "r2_entries_preserved": 42, "older_frozen_sets_repeated": False,
            "actual_item_answers": 8, "audio_confirmed_ids": confirmed, "visual_confirmed_ids": [],
            "pending_existing_ids": ["Q02", "Q03"], "not_yet_asked_ids": ["Q07", "Q08"],
            "Q05_Q06_old_boundary_videos_hidden": True, "current_media_routes": 28,
            "new_question_HTTP206_audio_verified": ["Q05", "Q06"], "stale_Q05_video_HTTP404": True,
            "TAR_reads": 0, "neural_calls": 0, "stage5g_complete": False, "training_ready": False}


if __name__ == "__main__":
    result = inspect()
    with (HERE / "progress_ui_r3_audit.json").open("x", encoding="utf8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "item_answers": 8, "audio_confirmed": 4, "r2_preserved": 42}))
