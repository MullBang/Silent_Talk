"""Bind r4's all-eight visual question to current confirmed audio and previews."""
from __future__ import annotations

import json
from pathlib import Path
import urllib.request
from typing import Any

from source_preflight import HERE, PREVIOUS, bind, check, read_json


def inspect() -> dict[str, Any]:
    """Inspect current selection metadata and three new video HTTP bindings."""
    stage = HERE.parent
    summary_path = stage / "results/stage5g_summary_r4.json"
    handoff_path = stage / "results/next_step_handoff_r4.json"
    question_path = stage / "question_batches/008_all8_visual.json"
    ui_path = stage / "ui/r4/manifest.json"
    summary, handoff, question, ui = (read_json(p) for p in (summary_path, handoff_path, question_path, ui_path))
    ids = [f"Q{i:02}" for i in range(1, 9)]
    assert handoff["summary"] == ui["summary"] == bind(summary_path)
    assert summary["next_question_batch"] == handoff["pending_visual_question"] == bind(question_path)
    assert summary["actual_phrase_answers"] == len(summary["evidence"]) == 13
    assert summary["original_end_clipping_reports"] == 4 and summary["revised_other_speech_mixing_reports"] == 1
    assert summary["selected_phrase_audio_confirmed"] == 8 and summary["selected_phrase_visual_confirmed"] == 0
    assert summary["stage5g_complete"] is summary["training_ready"] is False
    assert handoff["current_audio_confirmed"] == ids and handoff["current_visual_confirmed"] == []
    assert handoff["pending_existing_questions"] == handoff["not_yet_asked"] == handoff["matching_visual_preview_not_yet_prepared"] == []
    assert question["actual_answer"] is None and question["training_ready"] is False
    assert [r["review_id"] for r in question["records"]] == ids
    assert "뚜렷하게 잘리는" in question["question"] and "구분이 어려운" in question["question"]
    revised_ids = {"Q01", "Q05", "Q06", "Q08"}
    total_new_frames = 0
    for row, state, visible in zip(question["records"], summary["records"], ui["records"], strict=True):
        qid = row["review_id"]
        assert qid == state["review_id"] == visible["review_id"]
        assert row["candidate_id"] == state["candidate_id"]
        assert row["text"] == state["text"] == visible["text"]
        assert row["audio"] == state["selected_audio"] == visible["media"]["phrase_audio"]
        check(row["audio_confirmation"])
        answer = read_json(Path(row["audio_confirmation"]["path"]))
        assert answer["review_id"] == qid and answer["phrase_audio_boundary_approved"] is True
        assert answer["question_audio"] == row["audio"]
        assert state["human_phrase_audio_boundary_approved"] is visible["phrase_audio_confirmed"] is True
        assert state["human_phrase_visual_boundary_approved"] is visible["phrase_visual_confirmed"] is False
        if qid in revised_ids:
            record = read_json(stage / f"visual_{qid}_r1/record.json")
            expected_video = record["videos"]["phrase"]
            expected_context = record["videos"]["context"]
            total_new_frames += expected_video["frames"] + expected_context["frames"]
        else:
            record = read_json(PREVIOUS / "phrase_media_r2" / qid / "record.json")
            expected_video = record["muted_video"]["phrase"]
            expected_context = record["muted_video"]["phrase_context"]
        assert row["phrase_video"] == visible["media"]["phrase_video"] == bind(Path(expected_video["path"]))
        assert row["context_video"] == visible["media"]["phrase_context_video"] == bind(Path(expected_context["path"]))
        assert row["human_visual_answer"] is None and row["training_ready"] is False
        for role, binding in visible["media"].items():
            check(binding)
            assert ui["routes"][visible["urls"][role]] == binding
    assert total_new_frames == summary["new_review_display_frames"] == 372
    assert summary["new_review_videos"] == 8 and summary["new_review_videos_this_revision"] == 6
    assert len(ui["routes"]) == 33
    pointer = read_json(stage / "ui/current.json")
    assert pointer["manifest"] == bind(ui_path)
    check(ui["routes"]["/"])
    page = Path(ui["routes"]["/"]["path"]).read_text(encoding="utf8")
    assert page.count("音声") == 0
    assert page.count("음성 확인 완료 · 입술 검수 대기") == 8
    assert "<form" not in page
    for visible in ui["records"]:
        if visible["review_id"] not in ("Q05", "Q06", "Q08"):
            continue
        role = "phrase_video"
        req = urllib.request.Request(ui["current_url"].rstrip("/") + visible["urls"][role], headers={"Range": "bytes=0-127"})
        with urllib.request.urlopen(req, timeout=5) as response:
            assert response.status == 206
            assert response.read() == Path(visible["media"][role]["path"]).read_bytes()[:128]
    return {"schema": "offline-stage5g-independent-all8-visual-question-r4-audit-v1", "status": "passed",
            "summary": bind(summary_path), "handoff": bind(handoff_path), "question": bind(question_path),
            "ui_manifest": bind(ui_path), "question_bound_audios": 8, "question_bound_videos": 16,
            "audio_confirmed": 8, "visual_confirmed_in_snapshot": 0,
            "cumulative_new_preview_videos": 8, "cumulative_new_preview_display_frames": 372,
            "new_video_HTTP206_sample": ["Q05", "Q06", "Q08"], "media_routes_bound": 32,
            "new_decoding_or_historical_hash_sweep": False, "TAR_reads": 0, "neural_calls": 0,
            "scope": "This is the pre-answer qualitative visual question snapshot; it does not infer human visual approval.",
            "training_ready": False}


if __name__ == "__main__":
    result = inspect()
    with (HERE / "all8_visual_question_r4_audit.json").open("x", encoding="utf8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "bound_audios": 8, "bound_videos": 16}))
