"""Verify the Q01-only qualitative judgement and unchanged r6 viewing media."""
from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

from source_preflight import HERE, bind, check, read_json


def inspect() -> dict[str, Any]:
    """Check a single human-state transition without revisiting media decoding."""
    stage = HERE.parent
    evidence_path = stage / "user_evidence/015_Q01_visual_no_obvious_clipping.json"
    summary_path = stage / "results/stage5g_summary_r6.json"
    ui_path = stage / "ui/r6/manifest.json"
    evidence, summary, ui = (read_json(p) for p in (evidence_path, summary_path, ui_path))
    old_ui = read_json(stage / "ui/r5/manifest.json")
    old_summary = read_json(stage / "results/stage5g_summary_r5.json")
    for row in (evidence["question_manifest"], evidence["viewing_aid"], evidence["previous_inconclusive_answer"],
                ui["summary"], ui["previous_ui"], ui["actual_visual_evidence"], ui["builder"], ui["template_builder"]):
        check(row)
    question = read_json(Path(evidence["question_manifest"]["path"]))
    assert evidence["question"] == question["question"]
    assert evidence["record"] == question["record"]
    assert evidence["viewing_aid"] == question["viewing_aid"] == bind(stage / "ui/r5/manifest.json")
    assert evidence["record"]["review_id"] == evidence["review_id"] == "Q01"
    assert json.loads(evidence["question_item_id"]) == ["request_user_input_async", "call_a581c61051f347fbb98c5d635ab3638c", 0]
    assert evidence["answer"] == "뚜렷한 잘림은 보이지 않음"
    assert evidence["qualitative_visual_no_obvious_clipping"] is evidence["phrase_visual_boundary_approved"] is True
    assert evidence["legacy_approval_field_scope"] == "qualitative preview inspection only"
    assert all(evidence[k] is False for k in ("precise_articulatory_boundary_approved", "original_training_pixels_approved", "other_seven_items_approved", "training_ready"))
    assert summary["actual_visual_answers"] == 2 and summary["actual_phrase_answers"] == 13
    assert summary["selected_phrase_audio_confirmed"] == 8
    assert summary["selected_phrase_visual_confirmed"] == 1 and summary["visual_inconclusive_items"] == 7
    assert summary["stage5g_complete"] is summary["training_ready"] is False
    assert ui["summary"] == bind(summary_path) and ui["actual_visual_evidence"] == bind(evidence_path)
    assert ui["default_review_id"] == "Q02"
    assert ui["qualitative_visual_confirmations_total"] == 1 and ui["visual_inconclusive_total"] == 7
    assert {k: v for k, v in ui["routes"].items() if k != "/"} == {k: v for k, v in old_ui["routes"].items() if k != "/"}
    changed_ids = []
    for row, old, state, old_state in zip(ui["records"], old_ui["records"], summary["records"], old_summary["records"], strict=True):
        qid = row["review_id"]
        assert qid == old["review_id"] == state["review_id"] == old_state["review_id"]
        for key in ("media", "urls", "markers", "context_frames", "phrase_frames", "fps", "visual_record"):
            assert row[key] == old[key]
        assert state["selected_audio"] == old_state["selected_audio"]
        assert row["phrase_audio_confirmed"] is state["human_phrase_audio_boundary_approved"] is True
        expected = qid == "Q01"
        assert row["phrase_visual_confirmed"] == state["human_phrase_visual_boundary_approved"] == expected
        assert row["precise_articulatory_boundary_approved"] is row["training_ready"] is False
        if expected:
            assert state["visual_status"] == row["human_visual_status"] == "qualitative_no_obvious_clipping"
            assert state["visual_answer_evidence"] == row["visual_answer_evidence"] == bind(evidence_path)
            changed_ids.append(qid)
        else:
            assert state["visual_status"] == old_state["visual_status"] == "inconclusive_group_review"
            assert state["visual_answer_evidence"] == old_state["visual_answer_evidence"]
            assert row["human_visual_answer"] == old["human_visual_answer"] == "구분하기 어려움"
    assert changed_ids == ["Q01"]
    check(ui["routes"]["/"])
    page = Path(ui["routes"]["/"]["path"]).read_text(encoding="utf8")
    old_page = Path(old_ui["routes"]["/"]["path"]).read_text(encoding="utf8")
    seek_pattern = r'async function seekFrame\(index\).*?(?=async function showPart)'
    assert re.search(seek_pattern, page, flags=re.S).group() == re.search(seek_pattern, old_page, flags=re.S).group()
    assert "seekFrame(lastRequestedFrame-1)" in page and "seekFrame(lastRequestedFrame+1)" in page
    question2_path = stage / "question_batches/010_Q02_visual_aid.json"
    question2 = read_json(question2_path)
    assert question2["record"]["review_id"] == "Q02" and question2["viewing_aid"] == bind(ui_path)
    q02 = ui["records"][1]
    for key, role in [("audio", "phrase_audio"), ("phrase_video", "phrase_video"), ("context_video", "phrase_context_video")]:
        assert question2["record"][key] == q02["media"][role]
    assert question2["actual_answer"] is None and question2["training_ready"] is False
    assert ui["new_media_files"] == ui["new_raw_payload_bytes"] == ui["new_neural_calls"] == ui["new_training_updates"] == 0
    note = {
        "field": "summary.new_review_videos_this_revision", "recorded_value": summary["new_review_videos_this_revision"],
        "actual_new_media_in_r6": 0,
        "interpretation": "The summary retains the earlier six-preview work-batch counter. The r6 UI/media comparison proves no new media for this UI-state revision; preserve the question-bound snapshot and clarify the counter separately.",
    }
    return {"schema": "offline-stage5g-independent-Q01-visual-transition-r6-audit-v1", "status": "passed_with_counter_note",
            "evidence": bind(evidence_path), "summary": bind(summary_path), "ui_manifest": bind(ui_path),
            "next_Q02_question": bind(question2_path), "changed_visual_ids": changed_ids,
            "audio_confirmed": 8, "qualitative_no_obvious_clipping": 1, "inconclusive": 7,
            "precise_articulation_or_training_pixel_approvals": 0, "markers_media_seek_function_unchanged": True,
            "counter_note": note, "new_decodes": 0, "past_hash_sweeps": 0, "training_ready": False}


if __name__ == "__main__":
    result = inspect()
    with (HERE / "Q01_visual_transition_r6_audit.json").open("x", encoding="utf8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "changed_visual_ids": ["Q01"], "inconclusive": 7}))
