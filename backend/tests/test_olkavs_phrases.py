"""지정 문장 보조 모드(7단계) 검사: 후보 점수, 서비스 전달·판정, 목록 읽기, 응답 스키마 (GPU·실제 모델 불필요)."""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
for _path in (_TESTS_DIR, os.path.dirname(_TESTS_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from schemas.models import ResultItem  # noqa: E402
from test_olkavs_service import _StubWorker, _roi_result, svc  # noqa: E402
from test_olkavs_transcribe import _FakeModel, _FakeVocab, ot  # noqa: E402

_PHRASES = [{"id": "A01", "text": "첫째 문장"}, {"id": "N01", "text": "둘째 문장"}]


def _ok(value: float) -> dict:
    """정렬 가능한 후보 점수."""
    return {"status": "ok", "per_token": value}


def test_rank_phrases_reports_margin_and_refuses_partial_candidates() -> None:
    """모든 후보를 정렬할 수 있을 때만 고른다. 프레임이 모자란 후보가 있으면 짧은 문장끼리 겨루게 되므로 고르지 않는다."""
    phrases = [{"id": "A", "text": "가"}, {"id": "B", "text": "나"}, {"id": "C", "text": "다"}]
    out = ot.rank_phrases(phrases, [_ok(-3.0), _ok(-4.0), _ok(-2.5)])
    assert out == {"id": "C", "text": "다", "score": -2.5, "margin": 0.5, "runner_up_id": "A"}
    assert ot.rank_phrases(phrases, [_ok(-3.0), {"status": "infeasible", "per_token": None}, _ok(-2.5)]) is None
    assert ot.rank_phrases(phrases[:1], [_ok(-3.0)]) is None


def _frames_for(monkeypatch, path: list[int]) -> np.ndarray:
    """정해 둔 프레임별 토큰을 내는 가짜 모델을 등록하고 입력 프레임을 만든다."""
    logits = torch.full((1, len(path), 72), -8.0)
    for t, k in enumerate(path):
        logits[0, t, k] = 4.0
    bundle = {"model": _FakeModel(logits), "vocab": _FakeVocab(), "grp2char": lambda g: g, "search_cls": None,
              "contract": {"strict": True}, "device": "cpu"}
    monkeypatch.setitem(ot._MODEL_CACHE, "cpu", bundle)
    return (np.random.RandomState(0).rand(len(path), 96, 96, 3) * 255).astype(np.uint8)


def test_recognize_picks_phrase_matching_frame_probabilities(monkeypatch) -> None:
    """프레임 확률과 맞는 후보가 1등이다. 후보를 주지 않으면 phrase 키가 없다."""
    frames = _frames_for(monkeypatch, [5, 3, 6, 3, 7, 3])
    table = {"맞는": [5, 6, 7], "다른": [8, 9, 10]}
    monkeypatch.setattr(ot, "phrase_tokens", lambda text, _bundle: table[text])
    out = ot.recognize(frames, "cpu", "ctc", [{"id": "X", "text": "다른"}, {"id": "Y", "text": "맞는"}])
    assert out["phrase"]["id"] == "Y" and out["phrase"]["runner_up_id"] == "X" and out["phrase"]["margin"] > 0
    assert "phrase" not in ot.recognize(frames, "cpu", "ctc")


def test_recognize_keeps_text_when_phrase_scoring_fails(monkeypatch) -> None:
    """후보 토큰 변환이 실패해도 자유 인식 결과는 돌려준다."""
    frames = _frames_for(monkeypatch, [5, 3, 6, 3])

    def boom(_text, _bundle):
        raise ValueError("사전에 없는 토큰")

    monkeypatch.setattr(ot, "phrase_tokens", boom)
    out = ot.recognize(frames, "cpu", "ctc", [{"id": "X", "text": "가"}, {"id": "Y", "text": "나"}])
    assert out["phrase"] is None and out["token_ids"] == [5, 6]


def _one_window(status: str = "ok") -> list[dict]:
    return [{"index": 0, "start_ms": 0, "end_ms": 3000, "status": status, "npz": "a.npz" if status == "ok" else None}]


def _service(monkeypatch, segments: list[dict], output: dict):
    """후보 목록을 고정한 서비스와 인식 대체 워커."""
    monkeypatch.setattr(svc, "load_phrases", lambda _path: list(_PHRASES))
    monkeypatch.setattr(svc.config, "OLKAVS_MIN_CONFIDENCE", 0.75)
    monkeypatch.setattr(svc.config, "OLKAVS_PHRASE_MIN_MARGIN", 0.4)
    monkeypatch.setattr(svc.config, "OLKAVS_PHRASE_MIN_SCORE", -3.45)
    infer = _StubWorker(lambda payload, _progress: {"ok": True, **output})
    return svc.OlkavsService(_StubWorker(_roi_result(segments)), infer), infer


def test_phrase_is_sent_and_accepted_for_single_window_recording(tmp_path, monkeypatch) -> None:
    """녹화 한 창이면 후보를 보내고, 차이가 기준 이상이면 accepted다. 자유 인식 결과는 그대로다."""
    best = {"id": "N01", "text": "둘째 문장", "score": -2.0, "margin": 0.4, "runner_up_id": "A01"}
    service, infer = _service(monkeypatch, _one_window(), {"text": "모델 문장", "confidence": 0.9, "raw_score": -3.0, "phrase": best})
    item = service.transcribe_video("v.webm", str(tmp_path), utterance=True)[0]
    assert infer.calls[0]["phrases"] == _PHRASES
    assert item["text"] == "모델 문장" and item["phrase"] == {**best, "accepted": True}
    assert ResultItem(**item).phrase.accepted is True


def test_phrase_below_margin_is_kept_but_not_accepted(tmp_path, monkeypatch) -> None:
    """차이가 기준보다 작으면 accepted=False다. 자유 인식이 '[인식 불가]'여도 phrase는 남는다."""
    best = {"id": "A01", "text": "첫째 문장", "score": -3.0, "margin": 0.39, "runner_up_id": "N01"}
    service, _infer = _service(monkeypatch, _one_window(), {"text": "모델 문장", "confidence": 0.5, "raw_score": -3.0, "phrase": best})
    item = service.transcribe_video("v.webm", str(tmp_path), utterance=True)[0]
    assert item["text"] == svc.UNREC_TEXT and item["candidate"] == "모델 문장"
    assert item["phrase"]["accepted"] is False and item["phrase"]["text"] == "첫째 문장"


def test_phrase_below_score_floor_is_not_accepted(tmp_path, monkeypatch) -> None:
    """차이가 커도 1등 점수가 하한보다 낮으면 어느 후보와도 맞지 않는 말로 보고 고르지 않는다."""
    best = {"id": "A01", "text": "첫째 문장", "score": -3.46, "margin": 0.9, "runner_up_id": "N01"}
    service, _infer = _service(monkeypatch, _one_window(), {"text": "모델 문장", "confidence": 0.9, "raw_score": -3.0, "phrase": best})
    assert service.transcribe_video("v.webm", str(tmp_path), utterance=True)[0]["phrase"]["accepted"] is False


def test_phrase_not_requested_when_part_of_recording_failed(tmp_path, monkeypatch) -> None:
    """검출 실패 구간이 함께 있으면 발화 일부만 본 것이므로 후보를 보내지 않는다."""
    segments = _one_window() + [{"index": 1, "start_ms": 3000, "end_ms": 4000, "status": "failed", "npz": None, "reason": "no_face"}]
    service, infer = _service(monkeypatch, segments, {"text": "모델 문장", "confidence": 0.9, "raw_score": -3.0})
    service.transcribe_video("v.webm", str(tmp_path), utterance=True)
    assert len(infer.calls) == 1 and "phrases" not in infer.calls[0]


def test_phrase_not_requested_for_uploads_or_multiple_windows(tmp_path, monkeypatch) -> None:
    """업로드 경로와 여러 창으로 나뉜 긴 녹화에는 후보를 보내지 않는다."""
    output = {"text": "모델 문장", "confidence": 0.9, "raw_score": -3.0}
    service, infer = _service(monkeypatch, _one_window(), output)
    assert "phrase" not in service.transcribe_video("v.mp4", str(tmp_path))[0]
    assert "phrases" not in infer.calls[0]
    two = [{"index": i, "start_ms": i * 1000, "end_ms": (i + 1) * 1000, "status": "ok", "npz": f"{i}.npz"} for i in range(2)]
    service, infer = _service(monkeypatch, two, output)
    service.transcribe_video("v.webm", str(tmp_path), utterance=True)
    assert len(infer.calls) == 2 and all("phrases" not in c for c in infer.calls)


def test_phrase_absent_when_worker_returns_none_or_no_speech(tmp_path, monkeypatch) -> None:
    """후보 계산이 실패해 phrase=None이면 자유 인식만 남고, 발화 없음은 인식 자체를 하지 않는다."""
    service, _infer = _service(monkeypatch, _one_window(), {"text": "모델 문장", "confidence": 0.9, "raw_score": -3.0, "phrase": None})
    item = service.transcribe_video("v.webm", str(tmp_path), utterance=True)[0]
    assert "phrase" not in item and ResultItem(**item).phrase is None
    service, infer = _service(monkeypatch, _one_window("no_speech"), {})
    assert service.transcribe_video("v.webm", str(tmp_path), utterance=True)[0]["text"] == svc.NO_SPEECH_TEXT
    assert infer.calls == []


def test_load_phrases_reads_prompts_and_turns_off_on_bad_input(tmp_path) -> None:
    """실제 목록 20문장을 읽고, 파일 없음·형식 오류·후보 1개·빈 경로면 빈 목록이다."""
    real = svc.load_phrases(svc.config.OLKAVS_PHRASE_FILE)
    assert len(real) == 20 and set(real[0]) == {"id", "text"} and real[0]["id"] == "A01"
    assert svc.load_phrases(str(tmp_path / "none.json")) == []
    assert svc.load_phrases("") == []
    bad = tmp_path / "bad.json"
    bad.write_text("{", encoding="utf-8")
    assert svc.load_phrases(str(bad)) == []
    bad.write_text('{"fixed": [{"id": "A"}]}', encoding="utf-8")
    assert svc.load_phrases(str(bad)) == []
    bad.write_text('{"fixed": [{"id": "A", "text": "하나"}]}', encoding="utf-8")
    assert svc.load_phrases(str(bad)) == []
