"""Check original eight exact parent questions, media bindings and read-only UI."""
from __future__ import annotations

from html.parser import HTMLParser
import json
from pathlib import Path
from typing import Any
import urllib.error
import urllib.request

from audit_preflight import HERE, bind, check_binding, read_json


class PageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tags: list[tuple[str, dict[str, str | None]]] = []
        self.text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.tags.append((tag, dict(attrs)))

    def handle_data(self, data: str) -> None:
        self.text.append(data)


def inspect() -> dict[str, Any]:
    base = HERE.parent
    manifest_path = base / "parent_review_manifest.json"
    manifest = read_json(manifest_path)
    expected = read_json(base / "selected_inputs.json")["records"]
    for binding in manifest["sources"]:
        check_binding(binding)
    assert manifest["actual_answers"] == 0 and not manifest["phrase_approval_implied"]
    assert len(manifest["records"]) == 8
    page_path = base / "index.html"
    page = page_path.read_text("utf8")
    parsed = PageParser()
    parsed.feed(page)
    assert not any(tag in {"form", "input", "button"} for tag, _ in parsed.tags)
    sections = [attrs["id"] for tag, attrs in parsed.tags if tag == "section"]
    assert sections == [f"Q{i:02}" for i in range(1, 9)]
    page_text = " ".join(parsed.text)
    tests = []
    for row, pair in zip(manifest["records"], expected, strict=True):
        q = pair["review"]
        for name in ["review_id", "parent_id", "candidate_id", "speaker_id", "parent_text"]:
            assert row[name] == q[name]
        assert row["original_source_text"] == q["parent_source_text"]
        assert row["candidate_phrase_text"] == q["text"]
        question = (f'{q["review_id"]} {q["speaker_id"]} 전체 음성이 “{q["parent_text"]}”와 일치하고, '
                    '첫·마지막 소리가 잘리거나 다른 말이 섞이지 않고 온전히 들리나요?')
        assert row["question"] == question
        check_binding(row["media_record"])
        record = read_json(Path(row["media_record"]["path"]))
        assert record["parent_id"] == q["parent_id"]
        for name in ["sentence_audio", "context_audio"]:
            assert row["media"][name]["sha256"] == record[name]["sha256"]
        assert row["media"]["context_lips"]["sha256"] == record["video"]["roi_preview"]["sha256"]
        assert q["parent_text"] in page_text and q["text"] in page_text
        main_audio = [a for tag, a in parsed.tags if tag == "audio" and a.get("id") == "audio-" + q["review_id"]]
        assert len(main_audio) == 1 and main_audio[0]["src"] == row["urls"]["sentence_audio"]
        assert all(row[k] == "pending" for k in ["parent_transcript_status", "parent_audio_boundary_status", "phrase_audio_boundary_status", "phrase_visual_boundary_status"])
        assert row["training_ready"] is False and row["new_human_approval_added"] is False
        for name, media in row["media"].items():
            check_binding(media)
            request = urllib.request.Request("http://127.0.0.1:8772" + row["urls"][name], headers={"Range": "bytes=0-63"})
            with urllib.request.urlopen(request, timeout=10) as response:
                assert response.status == 206
                payload = response.read()
                assert response.headers["Content-Range"] == f'bytes 0-63/{media["bytes"]}'
            with Path(media["path"]).open("rb") as stream:
                assert payload == stream.read(64)
            tests.append({"review_id": q["review_id"], "media_kind": name, "range_status": 206, "first_64_bytes_exact": True})
    with urllib.request.urlopen("http://127.0.0.1:8772/", timeout=10) as response:
        assert response.status == 200 and response.read() == page_path.read_bytes()
    try:
        urllib.request.urlopen("http://127.0.0.1:8772/not-an-allowed-source", timeout=10)
    except urllib.error.HTTPError as exc:
        assert exc.code == 404
    else:
        raise AssertionError("Unknown route must be rejected")
    assert "영상과 문맥 음성은 시작 시각이 서로 조금 다를 수 있어 각각 재생합니다" in page_text
    assert "페이지 열기나 재생은 확인 답변으로 기록되지 않습니다" in page_text
    return {"schema": "offline-stage5f-independent-parent-page-audit-v1", "status": "passed",
            "manifest": bind(manifest_path), "page": bind(page_path),
            "original_parent_questions_verified": 8, "http_range_checks": tests,
            "unknown_route_rejected": True, "page_bytes_exact": True,
            "original_snapshot_answers": 0,
            "scope_note": "This original page remains a pending snapshot. Later actual chat responses and corrected candidates belong to separate evidence and are not overwritten into the original page.",
            "phrase_approvals_implied": False, "new_human_approvals_added": 0}


if __name__ == "__main__":
    result = inspect()
    with (HERE / "parent_page_audit.json").open("x", encoding="utf8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "questions": 8, "range_checks": len(result["http_range_checks"])}))
