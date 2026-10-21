"""Verify r2 phrase questions and loopback routes without recording answers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import urllib.error
import urllib.request

from audit_preflight import HERE, bind, check_binding, read_json
from audit_review_page import PageParser


def inspect() -> dict[str, Any]:
    base = HERE.parent
    path = base / "phrase_review_manifest.json"
    manifest = read_json(path)
    selection = read_json(base / "phrase_media_selected.json")
    check_binding(selection["manifest"])
    check_binding(selection["recovery"])
    candidates = read_json(Path(selection["manifest"]["path"]))["records"]
    assert Path(selection["manifest"]["path"]).parent == base / "phrase_media_r2"
    assert Path(manifest["media_root"]) == base / "phrase_media_r2"
    for source in manifest["sources"]:
        check_binding(source)
    check_binding(manifest["page"])
    server = read_json(base / "phrase_server_plan.json")
    for name in ["range_handler", "server", "manifest", "page"]:
        check_binding(server[name])
    assert server["read_only"] is True and server["listen"] == "127.0.0.1:8773"
    assert server["manifest"] == bind(path)
    assert manifest["actual_phrase_answers"] == manifest["actual_visual_answers"] == 0
    assert len(manifest["records"]) == len(candidates) == 8
    page_path = Path(manifest["page"]["path"])
    parsed = PageParser()
    parsed.feed(page_path.read_text("utf8"))
    assert [attrs["id"] for tag, attrs in parsed.tags if tag == "section"] == [f"Q{i:02}" for i in range(1, 9)]
    assert not any(tag in {"form", "input", "button"} for tag, _ in parsed.tags)
    audio_sources = [attrs["src"] for tag, attrs in parsed.tags if tag == "audio"]
    video_sources = [attrs["src"] for tag, attrs in parsed.tags if tag == "video"]
    assert len(audio_sources) == len(video_sources) == 16
    assert all("muted" in attrs for tag, attrs in parsed.tags if tag == "video")
    text = " ".join(parsed.text)
    assert "Q02 전체 음성은 끝을 0.12초 늘린 수정본으로 확인했습니다" in text
    assert "자동으로 추정한 구간" in text
    assert "각 음성과 영상은 따로 재생합니다" in text
    assert "영상의 시작·끝은 음성과 조금 다를 수 있습니다" in text
    checks = []
    for row, candidate in zip(manifest["records"], candidates, strict=True):
        for name in ["review_id", "parent_id", "candidate_id", "text"]:
            assert row[name] == candidate[name]
        assert row["question"] == f'{row["review_id"]} 짧게 자른 구절 음성에서 “{row["text"]}”가 시작·끝 잘림이나 다른 말의 혼입 없이 온전히 들리나요?'
        assert row["candidate_record"] == bind(base / "phrase_media_r2" / row["review_id"] / "record.json")
        expected_media = {"phrase_audio": candidate["phrase_audio"], "context_audio": candidate["phrase_context_audio"],
                          "phrase_video": candidate["muted_video"]["phrase"], "phrase_context_video": candidate["muted_video"]["phrase_context"]}
        assert row["media"] == expected_media
        assert row["parent_transcript_confirmed"] is True
        assert row["phrase_audio_boundary_status"] == row["phrase_visual_boundary_status"] == "pending"
        assert row["training_ready"] is False
        for kind, spec in row["media"].items():
            check_binding(spec)
            assert Path(spec["path"]).is_relative_to(base / "phrase_media_r2")
            route = row["urls"][kind]
            assert route in (video_sources if kind.endswith("video") else audio_sources)
            request = urllib.request.Request("http://127.0.0.1:8773" + route, headers={"Range": "bytes=0-63"})
            with urllib.request.urlopen(request, timeout=10) as response:
                assert response.status == 206
                assert response.headers["Content-Range"] == f'bytes 0-63/{spec["bytes"]}'
                actual_bytes = response.read()
            with Path(spec["path"]).open("rb") as stream:
                assert actual_bytes == stream.read(64)
            checks.append({"review_id": row["review_id"], "kind": kind, "status": 206, "served_bytes_exact": True})
    with urllib.request.urlopen("http://127.0.0.1:8773/", timeout=10) as response:
        assert response.status == 200 and response.read() == page_path.read_bytes()
    try:
        urllib.request.urlopen("http://127.0.0.1:8773/not-an-allowed-file", timeout=10)
    except urllib.error.HTTPError as exc:
        assert exc.code == 404
    else:
        raise AssertionError("Unknown route was not rejected")
    return {"schema": "offline-stage5f-independent-phrase-page-audit-v1", "status": "passed",
            "manifest": bind(path), "page": bind(page_path), "server_plan": bind(base / "phrase_server_plan.json"),
            "fixed_questions_verified": 8, "served_range_checks": checks,
            "selected_r2_media_only": True, "page_bytes_exact": True, "unknown_route_rejected": True,
            "snapshot_phrase_answers": 0, "snapshot_visual_answers": 0,
            "phrase_approvals_added": 0, "training_ready": False,
            "scope": "The page's original pending snapshot binds exact phrase audio questions separately from muted visual previews and parent approvals. Later replies require separate evidence."}


if __name__ == "__main__":
    result = inspect()
    with (HERE / "phrase_page_audit.json").open("x", encoding="utf8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "questions": 8, "range_checks": len(result["served_range_checks"])}))
