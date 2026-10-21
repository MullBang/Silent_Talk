"""Inspect exact r5 display markers and bounded frame-seek code without decoding."""
from __future__ import annotations

from fractions import Fraction
import json
import math
from pathlib import Path
import re
from typing import Any

from source_preflight import HERE, bind, check, read_json


def ratio(value: dict[str, Any]) -> Fraction:
    """Read rational marker metadata and require its displayed numeric value."""
    result = Fraction(value["numerator"], value["denominator"])
    if "seconds" in value:
        assert value["seconds"] == float(result)
    return result


def inspect() -> dict[str, Any]:
    """Check same-media viewing aids, scoped uncertainty and boundary seeking."""
    stage = HERE.parent
    path = stage / "ui/r5/manifest.json"
    manifest = read_json(path)
    previous = read_json(Path(manifest["previous_ui"]["path"]))
    summary = read_json(Path(manifest["summary"]["path"]))
    evidence = read_json(Path(manifest["actual_visual_evidence"]["path"]))
    question = read_json(Path(evidence["question_manifest"]["path"]))
    root_check_path = HERE / "visual_inconclusive_evidence_r5_check.json"
    root_check = read_json(root_check_path)
    assert root_check["status"] == "passed" and root_check["prior_r4_files_preserved"] == 121
    assert root_check["evidence"] == manifest["actual_visual_evidence"]
    assert root_check["summary"] == manifest["summary"]
    for row in [manifest["summary"], manifest["builder"], manifest["actual_visual_evidence"],
                manifest["previous_ui"], evidence["question_manifest"], *manifest["source_bindings"]]:
        check(row)
    assert evidence["answer"] == "구분하기 어려움"
    assert evidence["question"] == question["question"]
    assert json.loads(evidence["question_item_id"]) == ["request_user_input_async", "call_eb638ba038b14dfda288c214a2441a40", 0]
    assert evidence["records"] == question["records"]
    assert evidence["answer_unit"] == "one_group_answer_for_the_eight_bound_items"
    assert evidence["phrase_visual_boundary_approved"] is evidence["precise_articulatory_boundary_approved"] is False
    assert evidence["audio_confirmations_preserved"] is True
    assert summary["actual_visual_answers"] == 1 and summary["visual_inconclusive_items"] == 8
    assert summary["selected_phrase_audio_confirmed"] == 8 and summary["selected_phrase_visual_confirmed"] == 0
    assert summary["training_ready"] is False
    assert len(manifest["records"]) == 8
    old_routes = {k: v for k, v in previous["routes"].items() if k != "/"}
    new_routes = {k: v for k, v in manifest["routes"].items() if k != "/"}
    assert old_routes == new_routes and len(new_routes) == 32
    markers = []
    for row, old, state in zip(manifest["records"], previous["records"], summary["records"], strict=True):
        assert row["review_id"] == old["review_id"] == state["review_id"]
        assert row["media"] == old["media"] and row["urls"] == old["urls"]
        assert row["human_visual_status"] == state["visual_status"] == "inconclusive_group_review"
        assert row["human_visual_answer"] == "구분하기 어려움"
        assert row["phrase_audio_confirmed"] is state["human_phrase_audio_boundary_approved"] is True
        assert row["phrase_visual_confirmed"] is row["training_ready"] is False
        check(row["visual_record"])
        record = read_json(Path(row["visual_record"]["path"]))
        videos = record.get("videos", record.get("muted_video"))
        phrase = videos["phrase"]
        context = videos.get("context", videos.get("phrase_context"))
        assert row["media"]["phrase_video"] == {k: phrase[k] for k in ("path", "sha256", "bytes")}
        assert row["media"]["phrase_context_video"] == {k: context[k] for k in ("path", "sha256", "bytes")}
        pw, cw = phrase["clock_window"], context["clock_window"]
        start, end = ratio(pw["semantic_video_start"]), ratio(pw["semantic_video_end"])
        origin, stop = ratio(cw["preview_display_video_start"]), ratio(cw["preview_display_video_end"])
        m = row["markers"]
        assert ratio(m["phrase_start_in_context_video"]) == start - origin
        assert ratio(m["phrase_end_in_context_video"]) == end - origin
        assert ratio(m["context_display_duration"]) == stop - origin == Fraction(row["context_frames"], 25)
        assert ratio(m["context_source_video_origin"]) == origin
        assert ratio(m["confirmed_audio_mapped_video_start"]) == start
        assert ratio(m["confirmed_audio_mapped_video_end"]) == end
        assert ratio(m["phrase_display_before_audio"]) == start - ratio(pw["preview_display_video_start"])
        assert ratio(m["phrase_display_after_audio"]) == ratio(pw["preview_display_video_end"]) - end
        assert 0 < start - origin < end - origin < stop - origin
        assert row["fps"] == 25 and row["context_frames"] == context["frames"]
        count = row["context_frames"]
        # Boundary arithmetic for the actual formula read below, not browser-decoding certification.
        cases = [-1, 0, 1, count - 2, count - 1, count]
        targets = [(min(count - 1, max(0, index)) + .5) / 25 for index in cases]
        assert all(0 < value < count / 25 for value in targets)
        assert targets[0] == targets[1] == .02
        assert targets[-1] == targets[-2] == (count - .5) / 25
        assert min(count - 1, max(0, math.floor((count / 25 + 1e-8) * 25))) == count - 1
        markers.append({"review_id": row["review_id"], "audio_start_in_context": float(start - origin),
                        "audio_end_in_context": float(end - origin), "context_frames": count,
                        "last_frame_seek_center": targets[-1], "seek_stays_inside_duration": True})
    page_path = Path(manifest["routes"]["/"]["path"])
    check(manifest["routes"]["/"])
    page = page_path.read_text(encoding="utf8")
    embedded = re.search(r'<script id="reviewData" type="application/json">(.*?)</script>', page, flags=re.S)
    assert embedded and json.loads(embedded.group(1)) == manifest["records"]
    required_code = [
        "function frameFor(t){return Math.max(0,Math.min(current.context_frames-1,Math.floor((t+1e-8)*25)));}",
        "lastRequestedFrame=Math.max(0,Math.min(current.context_frames-1,index));const target=(lastRequestedFrame+0.5)/25;",
        "seekFrame(lastRequestedFrame-1)", "seekFrame(lastRequestedFrame+1)",
        "seekFrame(frameFor(startTime()))", "seekFrame(frameFor(endTime()))",
        "slider.max=String(current.context_frames-1)",
        "if(generation!==seekSerial||active!==current.review_id)return false;",
    ]
    assert all(code in page for code in required_code)
    assert "0.25배" in page and "0.5배" in page and "1배" in page
    assert "확인한 음성 범위" in page and "정답 경계를 뜻하지 않습니다" in page
    assert "판정을 저장하지 않습니다" in page and "구분 어려움 · 확인 처리 0" in page
    assert "<form" not in page and "requestVideoFrameCallback" not in page
    assert manifest["new_media_files"] == manifest["new_raw_payload_bytes"] == manifest["new_neural_calls"] == manifest["new_training_updates"] == manifest["new_visual_approvals"] == 0
    assert manifest["approval_persistence"] is manifest["training_ready"] is False
    return {"schema": "offline-stage5g-independent-boundary-aid-r5-audit-v1", "status": "passed",
            "manifest": bind(path), "page": bind(page_path), "builder": manifest["builder"],
            "evidence": manifest["actual_visual_evidence"], "summary": manifest["summary"],
            "root_r4_preservation_check_referenced": bind(root_check_path), "r4_hash_sweep_repeated": False,
            "same_media_routes": 32, "same_bound_items": 8, "actual_group_visual_answers": 1,
            "audio_confirmed": 8, "visual_confirmed": 0, "visual_inconclusive": 8,
            "markers_and_seek_bounds": markers, "code_review": {
                "context_origin_is_display_video_start": True, "offset_not_added_again": True,
                "seek_target_uses_frame_center": True, "first_last_frame_clamping": True,
                "ended_duration_maps_to_last_frame": True, "generation_guard_present": True,
                "browser_precision_guaranteed": False, "actual_browser_test_by_root": True},
            "new_video_decodes": 0, "TAR_reads": 0, "neural_calls": 0,
            "limit": "Marker lines represent confirmed audio times. Mathematical seek-center checks and code review do not certify browser pixel-accurate seeking or human articulatory boundaries.",
            "training_ready": False}


if __name__ == "__main__":
    result = inspect()
    with (HERE / "boundary_aid_r5_audit.json").open("x", encoding="utf8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "items": 8, "same_media_routes": 32, "new_video_decodes": 0}))
