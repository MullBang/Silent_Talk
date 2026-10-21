"""Audit the current r2 progress snapshot and a bounded HTTP route sample."""
from __future__ import annotations

import json
from pathlib import Path
import urllib.error
import urllib.request
from typing import Any

from source_preflight import HERE, PREVIOUS, bind, check, read_json


def inspect() -> dict[str, Any]:
    """Check current reporting without repeating source-video decoding."""
    stage = HERE.parent
    summary_path = stage / "results/stage5g_summary_r2.json"
    handoff_path = stage / "results/next_step_handoff_r2.json"
    ui_path = stage / "ui/r2/manifest.json"
    summary, handoff, ui = (read_json(p) for p in (summary_path, handoff_path, ui_path))
    pointer = read_json(stage / "ui/current.json")
    check(pointer["manifest"])
    assert pointer["manifest"] == bind(ui_path)
    assert handoff["summary"] == ui["summary"] == bind(summary_path)
    for row in [*summary["evidence"], summary["Q01_visual_preview"], summary["next_question_batch"],
                ui["builder"], ui["range_handler"]]:
        check(row)
    assert summary["actual_phrase_answers"] == len(summary["evidence"]) == 3
    assert summary["selected_phrase_audio_confirmed"] == 2 and summary["selected_phrase_visual_confirmed"] == 0
    assert summary["new_review_videos"] == 2 and summary["new_review_display_frames"] == 86
    assert summary["stage5g_complete"] is summary["training_ready"] is False
    assert handoff["current_audio_confirmed"] == ["Q01", "Q04"] and handoff["current_visual_confirmed"] == []
    assert [r["review_id"] for r in handoff["pending_existing_questions"]] == ["Q02", "Q03", "Q05", "Q06"]
    assert handoff["not_yet_asked"] == ["Q07", "Q08"]
    assert handoff["Q01_revised_visual_approved"] is False
    assert summary["new_neural_calls"] == summary["new_tar_payload_bytes"] == summary["training_updates"] == summary["new_training_cache_files"] == 0
    assert handoff["no_automatic_training_or_cache_or_Q09_Q20_extraction"] is True
    assert len(ui["records"]) == 8 and len(ui["routes"]) == 33
    previous = {r["review_id"]: r for r in read_json(PREVIOUS / "phrase_review_manifest.json")["records"]}
    visual = read_json(Path(summary["Q01_visual_preview"]["path"]))
    revision = read_json(Path(visual["revision_record"]["path"]))
    expected_roles = {"phrase_audio", "context_audio", "phrase_video", "phrase_context_video"}
    for actual, state in zip(ui["records"], summary["records"], strict=True):
        qid = actual["review_id"]
        assert qid == state["review_id"] and actual["text"] == state["text"]
        assert actual["phrase_audio_confirmed"] == state["human_phrase_audio_boundary_approved"] == (qid in ("Q01", "Q04"))
        assert actual["phrase_visual_confirmed"] is state["human_phrase_visual_boundary_approved"] is False
        assert state["training_ready"] is False
        assert all(state[k] is None for k in ("actual_split", "frame_count", "ctc_feasible"))
        assert set(actual["media"]) == set(actual["urls"]) == expected_roles
        assert actual["media"]["phrase_audio"] == state["selected_audio"]
        if qid == "Q01":
            selected = {"phrase_audio": revision["media"]["phrase"], "context_audio": revision["media"]["context"],
                        "phrase_video": visual["videos"]["phrase"], "phrase_context_video": visual["videos"]["context"]}
            expected = {role: bind(Path(item["path"])) for role, item in selected.items()}
        else:
            expected = previous[qid]["media"]
        for role, item in actual["media"].items():
            check(item)
            assert item == {k: expected[role][k] for k in ("path", "sha256", "bytes")}
            assert ui["routes"][actual["urls"][role]] == item
    check(ui["routes"]["/"])
    page = Path(ui["routes"]["/"]["path"]).read_text(encoding="utf8")
    assert page.count("음성 확인 완료 · 입술 검수 대기") == 2
    assert "<form" not in page and "입술 검수" in page
    url = ui["current_url"]
    assert url == summary["current_review_url"] == handoff["current_review_url"] == "http://127.0.0.1:8775/"
    selected_routes = [("Q01", "phrase_audio"), ("Q01", "phrase_video"), ("Q05", "phrase_audio"), ("Q06", "phrase_audio")]
    rows = {r["review_id"]: r for r in ui["records"]}
    for qid, role in selected_routes:
        row = rows[qid]
        item = row["media"][role]
        req = urllib.request.Request(url.rstrip("/") + row["urls"][role], headers={"Range": "bytes=0-127"})
        with urllib.request.urlopen(req, timeout=5) as response:
            assert response.status == 206
            assert response.headers["Content-Range"] == f'bytes 0-127/{item["bytes"]}'
            assert response.read() == Path(item["path"]).read_bytes()[:128]
    try:
        urllib.request.urlopen(url + "unlisted.mp4", timeout=5)
    except urllib.error.HTTPError as exc:
        assert exc.code == 404
    else:
        raise AssertionError("Unlisted URL was not rejected")
    return {"schema": "offline-stage5g-independent-progress-ui-r2-audit-v1", "status": "passed",
            "summary": bind(summary_path), "handoff": bind(handoff_path), "ui_manifest": bind(ui_path),
            "explicit_pointer_target_observed": pointer["manifest"], "pointer_itself_is_mutable": True,
            "actual_phrase_answers": 3, "audio_confirmed_ids": ["Q01", "Q04"], "visual_confirmed_ids": [],
            "pending_existing_ids": ["Q02", "Q03", "Q05", "Q06"], "not_yet_asked_ids": ["Q07", "Q08"],
            "media_routes_bound": 32, "http206_sample_routes_verified": selected_routes, "unlisted404": True,
            "source_video_decode_repeated": False, "TAR_reads": 0, "neural_calls": 0, "training_ready": False}


if __name__ == "__main__":
    result = inspect()
    with (HERE / "progress_ui_r2_audit.json").open("x", encoding="utf8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "routes": 32, "http_sample": 4}))
