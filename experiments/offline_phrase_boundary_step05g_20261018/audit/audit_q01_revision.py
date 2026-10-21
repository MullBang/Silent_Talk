"""Independently bind Q01's actual answer and exact stored-PCM revision."""
from __future__ import annotations

import json
from pathlib import Path
import urllib.error
import urllib.request
import wave
from typing import Any

from source_preflight import HERE, PREVIOUS, bind, check, read_json


def pcm(path: Path) -> tuple[bytes, int]:
    """Read bounded local mono 16-bit 16k PCM without resampling."""
    with wave.open(str(path), "rb") as wav:
        if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getcomptype()) != (1, 2, 16000, "NONE"):
            raise AssertionError(f"Unexpected review PCM format: {path}")
        count = wav.getnframes()
        raw = wav.readframes(count)
    if len(raw) != 2 * count:
        raise AssertionError(f"Truncated review PCM: {path}")
    return raw, count


def inspect() -> dict[str, Any]:
    """Verify provenance, revised slice, pending scope and exact HTTP routes."""
    stage = HERE.parent
    evidence_path = stage / "user_evidence/001_Q01_original_end_clipping.json"
    revision_path = stage / "revisions/Q01_end_parent/record.json"
    review_path = stage / "review_manifest_r1.json"
    evidence = read_json(evidence_path)
    revision = read_json(revision_path)
    review = read_json(review_path)
    plan = read_json(stage / "server_plan_r1.json")
    old_questions = read_json(PREVIOUS / "phrase_review_manifest.json")
    old_question = next(r for r in old_questions["records"] if r["review_id"] == "Q01")
    old = read_json(PREVIOUS / "phrase_media_r2/Q01/record.json")
    parent = next(r for r in read_json(PREVIOUS / "effective_parent_inputs.json")["records"] if r["review_id"] == "Q01")
    parent_record = read_json(Path(parent["media_record"]["path"]))
    context_meta = parent_record["context_audio"]
    for row in (evidence["question_manifest"], evidence["question_audio"], revision["trigger"],
                revision["previous_candidate"], revision["effective_parent_inputs"], revision["original_question_audio"],
                revision["code"], review["revision"], review["page"], parent["media_record"], context_meta,
                plan["manifest"], plan["range_handler"], plan["server"]):
        check(row)
    assert evidence["question"] == old_question["question"] == 'Q01 짧게 자른 구절 음성에서 “너무 속상해”가 시작·끝 잘림이나 다른 말의 혼입 없이 온전히 들리나요?'
    assert evidence["answer"] == '마지막 "해" 부분이 거의 들리지 않을정도로 끝이 잘림'
    assert json.loads(evidence["question_item_id"]) == ["request_user_input_async", "call_c6b912f45a194f748b5c9e8dc2e486de", 0]
    assert evidence["question_audio"] == bind(Path(old["phrase_audio"]["path"]))
    for key in ("parent_id", "candidate_id", "review_id"):
        assert evidence[key] == revision[key] == old[key] == parent[key]
    assert revision["text"] == old["text"] == "너무 속상해"
    assert revision["parent_confirmation_evidence_id"] == parent["actual_answer_evidence_id"] == "parent_answer_01"
    assert parent["parent_transcript_confirmed"] is parent["parent_audio_qualitatively_confirmed"] is True
    source, source_n = pcm(Path(context_meta["path"]))
    old_pcm, old_n = pcm(Path(old["phrase_audio"]["path"]))
    start = old["phrase_audio"]["source_context_first_sample_16k"]
    old_end = old["phrase_audio"]["source_context_end_sample_16k"]
    native_origin = context_meta["source_first_sample"]
    native_end = parent["original_recording_end_sample_48k"]
    assert (native_end - native_origin) % 3 == 0
    end = (native_end - native_origin) // 3
    assert (start, old_end, end, old_n) == (76040, 96520, 99808, 20480)
    assert old_pcm == source[start * 2:old_end * 2]
    assert revision["extension_samples_16k"] == end - old_end == 3288
    assert revision["extension_seconds"] == (end - old_end) / 16000 == 0.2055
    observed = []
    for role, first, stop in (("phrase", start, end), ("context", start - 3200, end + 3200)):
        row = revision["media"][role]
        check(row)
        check(row["source_context"])
        assert row["source_context"] == bind(Path(context_meta["path"]))
        raw, count = pcm(Path(row["path"]))
        assert 0 <= first < stop <= source_n
        assert raw == source[first * 2:stop * 2]
        assert row["samples"] == count == stop - first
        assert row["duration_seconds"] == count / 16000
        assert row["source_context_first_sample_16k"] == first
        assert row["source_context_end_sample_16k"] == stop
        assert row["source_original_wav_first_sample_48k"] == native_origin + 3 * first
        assert row["source_original_wav_end_sample_48k"] == native_origin + 3 * stop
        assert row["new_resampling"] is row["padding"] is row["fade"] is False
        if role == "phrase":
            assert raw[:len(old_pcm)] == old_pcm
            assert row["source_original_wav_end_sample_48k"] == native_end == 12539904
        observed.append({"role": role, "samples": count, "exact_source_PCM": True})
    assert revision["start_unchanged"] is revision["text_unchanged"] is revision["extension_is_not_measured_boundary_error"] is True
    assert revision["new_video_created"] is False
    assert revision["new_neural_calls"] == revision["new_tar_payload_bytes"] == revision["new_training_updates"] == 0
    for obj in (evidence, revision, review):
        assert obj["phrase_audio_boundary_approved"] is obj["phrase_visual_boundary_approved"] is obj["training_ready"] is False
    assert evidence["new_revision_approved"] is False
    assert review["actual_revision_answer"] is None
    assert review["Q02_Q03_original_answers"] == "pending"
    assert review["question_audio"] == review["media"]["phrase"] == revision["media"]["phrase"]
    assert len(list((stage / "user_evidence").glob("*.json"))) == 1
    page = Path(review["page"]["path"]).read_text(encoding="utf8")
    assert "아직 확인 전" in page and "Q02·Q03" in page and "답변을 기다리고" in page
    assert page.count("<audio ") == 2 and "<form" not in page
    for route, binding in plan["routes"].items():
        check(binding)
        if route == "/":
            assert binding == review["page"]
            continue
        role = "phrase" if route == "/q01/phrase.wav" else "context"
        assert binding == bind(Path(revision["media"][role]["path"]))
        request = urllib.request.Request(review["review_url"].rstrip("/") + route, headers={"Range": "bytes=0-127"})
        with urllib.request.urlopen(request, timeout=5) as response:
            assert response.status == 206
            assert response.headers["Content-Range"] == f'bytes 0-127/{binding["bytes"]}'
            assert response.read() == Path(binding["path"]).read_bytes()[:128]
    try:
        urllib.request.urlopen(review["review_url"] + "unlisted.wav", timeout=5)
    except urllib.error.HTTPError as exc:
        assert exc.code == 404
    else:
        raise AssertionError("Unlisted route did not return 404")
    return {
        "schema": "offline-stage5g-independent-Q01-revision-audit-v1", "status": "passed",
        "evidence": bind(evidence_path), "revision": bind(revision_path), "review_question": bind(review_path),
        "actual_answer_count": 1, "original_candidate_end_clipping_reports": 1,
        "new_candidate_start_context16k": start, "new_candidate_end_context16k": end,
        "new_candidate_samples_16k": end - start, "new_candidate_seconds": (end - start) / 16000,
        "extension_samples_16k": end - old_end, "extension_seconds": (end - old_end) / 16000,
        "confirmed_parent_end_native48k": native_end, "exact_old_PCM_prefix": True,
        "media": observed, "http206_exact_routes": 2, "unlisted_route_404": True,
        "new_candidate_approval": False, "Q02_Q03_answers": "pending", "new_visual_approval": False,
        "TAR_payload_reads": 0, "neural_calls": 0, "training_updates": 0,
        "limit": "User-answer authority comes from the root chat, not the JSON source label. Exact PCM and confirmed parent end do not approve this revised phrase's boundary.",
        "training_ready": False,
    }


if __name__ == "__main__":
    result = inspect()
    with (HERE / "Q01_revision_audit.json").open("x", encoding="utf8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "samples": result["new_candidate_samples_16k"], "http206_routes": 2}))
