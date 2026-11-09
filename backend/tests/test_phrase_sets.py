"""지정 문장 묶음(phrase_set) 검사 — 병실 무음 도우미 서버 쪽 (GPU·실제 모델 불필요).

- config: 묶음 ID 화이트리스트 → (목록 파일, 차이 하한, 점수 하한, 디코더), ward 임시 기준값과 환경변수 검사.
- 서비스: 기본 묶음의 인식 요청이 묶음 기능 이전과 바이트 단위로 같은지, ward는 명령 목록·ward 기준·ctc를 쓰는지.
- API: POST /api/record의 phrase_set 선택 필드(기본·ward·거부, 빈 값·중복 필드도 거부), Job 모델 정보의 디코더.
- 상태: /health의 병실 목록 상태(ward_phrases)와 목록을 읽지 못할 때의 기동 ERROR 로그.
- 스키마: PhraseMatch.set.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import typing

import pytest

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_ROOT = os.path.dirname(_TESTS_DIR)
for _path in (_TESTS_DIR, _BACKEND_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import config  # noqa: E402
from schemas.models import PhraseMatch, ResultItem  # noqa: E402
from test_olkavs_service import _StubWorker, _roi_result, svc  # noqa: E402
from test_record_api import client, env, jm, make_video, record_module  # noqa: E402,F401 - env는 fixture

_SENTENCES = [{"id": "A01", "text": "첫째 문장"}, {"id": "N01", "text": "둘째 문장"}]
_WARD = [{"id": "W01", "text": "간호사 선생님 좀 불러 주세요"}, {"id": "W05", "text": "목이 말라 물 한 모금만"}]


# ---------------------------------------------------------------------------
# config: 화이트리스트와 기준값
# ---------------------------------------------------------------------------

def test_phrase_set_whitelist_maps_ids_to_settings() -> None:
    """기본은 sentences(기존 고정 설정, 디코더 없음), ward는 병실 목록·ward 기준·ctc다."""
    assert config.PHRASE_SET_IDS == ("sentences", "ward")
    assert config.PHRASE_SET_DEFAULT == "sentences"
    assert config.phrase_set_config("sentences") == config.PhraseSetConfig(
        config.OLKAVS_PHRASE_FILE, config.OLKAVS_PHRASE_MIN_MARGIN, config.OLKAVS_PHRASE_MIN_SCORE, None)
    assert config.phrase_set_config("ward") == config.PhraseSetConfig(
        config.OLKAVS_WARD_PHRASE_FILE, config.OLKAVS_WARD_MIN_MARGIN, config.OLKAVS_WARD_MIN_SCORE, "ctc")


@pytest.mark.parametrize("value", ["Ward", "WARD", " ward", "", "prompts", "../frontend/src/capture/prompts.json",
                                   config.OLKAVS_WARD_PHRASE_FILE, None, ["ward"], 1])
def test_phrase_set_config_rejects_unknown_ids(value) -> None:
    """화이트리스트에 정확히 있는 ID만 받는다. 경로·대소문자 변형·해시할 수 없는 값은 ValueError."""
    with pytest.raises(ValueError):
        config.phrase_set_config(value)


def test_phrase_set_config_reads_current_values(monkeypatch) -> None:
    """부를 때마다 현재 값을 읽는다(환경변수·테스트에서 바꾼 기준이 반영된다). 두 묶음은 서로 섞이지 않는다."""
    monkeypatch.setattr(config, "OLKAVS_WARD_MIN_MARGIN", 0.9)
    monkeypatch.setattr(config, "OLKAVS_WARD_MIN_SCORE", -2.0)
    monkeypatch.setattr(config, "OLKAVS_PHRASE_MIN_MARGIN", 0.1)
    ward = config.phrase_set_config("ward")
    assert (ward.min_margin, ward.min_score) == (0.9, -2.0)
    assert config.phrase_set_config("sentences").min_margin == 0.1


def test_ward_thresholds_default_to_provisional_values() -> None:
    """환경 변수가 없으면 ward 기준은 임시값(차이 0.5, 점수 -3.40)이고 목록은 저장소의 ward/commands.json이다."""
    if any(os.environ.get(name) for name in ("LIPNET_OLKAVS_WARD_MIN_MARGIN", "LIPNET_OLKAVS_WARD_MIN_SCORE",
                                             "LIPNET_OLKAVS_WARD_PHRASE_FILE")):
        pytest.skip("환경 변수로 ward 설정을 덮어쓴 실행")
    assert config.OLKAVS_WARD_MIN_MARGIN == 0.5
    assert config.OLKAVS_WARD_MIN_SCORE == -3.40
    assert config.OLKAVS_WARD_PHRASE_FILE == os.path.join(config.PROJECT_ROOT, "frontend", "src", "ward",
                                                          "commands.json")
    assert os.path.isfile(config.OLKAVS_WARD_PHRASE_FILE)


def _import_config(env_overrides: dict[str, str]) -> subprocess.CompletedProcess:
    """새 프로세스에서 config를 import한다 (기동 시 환경변수 검사를 실제로 거친다)."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("LIPNET_OLKAVS_WARD_")}
    env.update(env_overrides, PYTHONIOENCODING="utf-8")
    code = ("import config; print(config.OLKAVS_WARD_MIN_MARGIN, config.OLKAVS_WARD_MIN_SCORE, "
            "config.OLKAVS_WARD_PHRASE_FILE == '', config.OLKAVS_PHRASE_MIN_MARGIN, config.OLKAVS_PHRASE_MIN_SCORE)")
    return subprocess.run([sys.executable, "-c", code], cwd=_BACKEND_ROOT, env=env, capture_output=True,
                          text=True, encoding="utf-8", timeout=60)


def test_ward_thresholds_follow_environment_without_touching_sentence_thresholds() -> None:
    """LIPNET_OLKAVS_WARD_*로 ward 기준만 바꾼다. 문장 묶음 고정값은 그대로다."""
    out = _import_config({"LIPNET_OLKAVS_WARD_MIN_MARGIN": "0.7", "LIPNET_OLKAVS_WARD_MIN_SCORE": "-3.2",
                          "LIPNET_OLKAVS_WARD_PHRASE_FILE": ""})
    assert out.returncode == 0, out.stderr
    margin, score, file_off, sentence_margin, sentence_score = out.stdout.split()
    assert (float(margin), float(score), file_off) == (0.7, -3.2, "True")
    if not (os.environ.get("LIPNET_OLKAVS_PHRASE_MIN_MARGIN") or os.environ.get("LIPNET_OLKAVS_PHRASE_MIN_SCORE")):
        assert (float(sentence_margin), float(sentence_score)) == (0.5, -3.40)


@pytest.mark.parametrize("name,value", [
    ("LIPNET_OLKAVS_WARD_MIN_MARGIN", "-0.1"),
    ("LIPNET_OLKAVS_WARD_MIN_MARGIN", "nan"),
    ("LIPNET_OLKAVS_WARD_MIN_MARGIN", "inf"),
    ("LIPNET_OLKAVS_WARD_MIN_MARGIN", "abc"),
    ("LIPNET_OLKAVS_WARD_MIN_SCORE", "0.1"),
    ("LIPNET_OLKAVS_WARD_MIN_SCORE", "-inf"),
    ("LIPNET_OLKAVS_WARD_MIN_SCORE", "nan"),
])
def test_invalid_ward_thresholds_stop_startup(name, value) -> None:
    """범위 밖·NaN·무한대·숫자 아님은 서버 기동(config import) 자체를 막는다."""
    out = _import_config({name: value})
    assert out.returncode != 0
    assert "ValueError" in out.stderr


def test_ward_commands_load_as_twenty_candidates() -> None:
    """실제 병실 목록: W01~W20 20개, 서버는 id와 인식용 text만 쓴다(display 아님). 문장 묶음 ID와 겹치지 않는다."""
    ward = svc.load_phrases(config.OLKAVS_WARD_PHRASE_FILE)
    with open(config.OLKAVS_WARD_PHRASE_FILE, encoding="utf-8") as handle:
        fixed = json.load(handle)["fixed"]
    assert [p["id"] for p in ward] == [f"W{i:02d}" for i in range(1, 21)]
    assert all(set(p) == {"id", "text"} for p in ward)
    assert [p["text"] for p in ward] == [p["text"] for p in fixed]
    sentences = svc.load_phrases(config.OLKAVS_PHRASE_FILE)
    assert not {p["id"] for p in ward} & {p["id"] for p in sentences}


# ---------------------------------------------------------------------------
# 서비스: 인식 요청과 판정
# ---------------------------------------------------------------------------

def _one_window() -> list[dict]:
    return [{"index": 0, "start_ms": 0, "end_ms": 3000, "status": "ok", "npz": "a.npz"}]


def _service(monkeypatch, output: dict, segments: list[dict] | None = None):
    """목록 파일 경로별로 고정 후보를 돌려주고, 인식 요청을 기록하는 서비스."""
    lists = {config.OLKAVS_PHRASE_FILE: _SENTENCES, config.OLKAVS_WARD_PHRASE_FILE: _WARD}
    monkeypatch.setattr(svc, "load_phrases", lambda path: [dict(p) for p in lists.get(path, [])])
    monkeypatch.setattr(svc.config, "OLKAVS_MIN_CONFIDENCE", 0.75)
    roi = _StubWorker(_roi_result(segments or _one_window()))
    infer = _StubWorker(lambda payload, _progress: {"ok": True, **json.loads(json.dumps(output))})
    return svc.OlkavsService(roi, infer), roi, infer


def _wire(payload: dict) -> str:
    """WorkerProcess.request가 보내는 줄과 같은 직렬화(id 제외)."""
    return json.dumps(payload, ensure_ascii=True)


def test_default_set_sends_byte_identical_infer_request(tmp_path, monkeypatch) -> None:
    """phrase_set을 주지 않거나 'sentences'를 주면 인식 요청은 묶음 기능 이전과 바이트 단위로 같다(decoder 키 없음)."""
    output = {"text": "모델 문장", "confidence": 0.9, "raw_score": -3.0,
              "phrase": {"id": "N01", "text": "둘째 문장", "score": -2.0, "margin": 0.8, "runner_up_id": "A01"}}
    before = _wire({"cmd": "infer", "npz": "a.npz", "phrases": _SENTENCES})
    results = []
    for kwargs in ({}, {"phrase_set": "sentences"}):
        service, roi, infer = _service(monkeypatch, output)
        results.append(service.transcribe_video("v.webm", str(tmp_path), utterance=True, **kwargs))
        assert [_wire(c) for c in infer.calls] == [before]
        assert "decoder" not in infer.calls[0]
    assert results[0] == results[1]
    assert results[0][0]["phrase"] == {**output["phrase"], "accepted": True, "set": "sentences"}


def test_roi_request_does_not_depend_on_phrase_set(tmp_path, monkeypatch) -> None:
    """입술 추출 요청은 묶음과 관계없이 같다(work_dir만 같으면 바이트 단위로 같다)."""
    wires = []
    for phrase_set in ("sentences", "ward"):
        service, roi, _infer = _service(monkeypatch, {"text": "가", "confidence": 0.9, "raw_score": -1.0})
        service.transcribe_video("v.webm", str(tmp_path), utterance=True, video_seconds=3.0, phrase_set=phrase_set)
        wires.append(_wire(roi.calls[0]))
    assert wires[0] == wires[1]


def test_ward_set_uses_ward_list_thresholds_and_ctc(tmp_path, monkeypatch) -> None:
    """ward: 병실 목록·ward 기준으로 판정하고 decoder='ctc'를 보낸다. 문장 묶음 기준은 쓰지 않는다."""
    monkeypatch.setattr(svc.config, "OLKAVS_PHRASE_MIN_MARGIN", 0.0)   # 문장 기준이면 모두 통과할 값
    monkeypatch.setattr(svc.config, "OLKAVS_PHRASE_MIN_SCORE", -10.0)
    monkeypatch.setattr(svc.config, "OLKAVS_WARD_MIN_MARGIN", 0.6)
    monkeypatch.setattr(svc.config, "OLKAVS_WARD_MIN_SCORE", -3.0)
    cases = [((0.55, -2.0), False), ((0.65, -3.1), False), ((0.65, -2.9), True), ((0.6, -3.0), True)]
    for (margin, score), expected in cases:
        best = {"id": "W05", "text": "목이 말라 물 한 모금만", "score": score, "margin": margin, "runner_up_id": "W01"}
        output = {"text": "목이 말라", "confidence": 0.9, "raw_score": -3.0, "decoder": "ctc", "phrase": best}
        service, _roi, infer = _service(monkeypatch, output)
        item = service.transcribe_video("v.webm", str(tmp_path), utterance=True, phrase_set="ward")[0]
        assert _wire(infer.calls[0]) == _wire({"cmd": "infer", "npz": "a.npz", "phrases": _WARD, "decoder": "ctc"})
        assert item["phrase"] == {**best, "accepted": expected, "set": "ward"}, (margin, score)
        assert item["text"] == "목이 말라" and item["confidence"] == 0.9  # 자유 인식 결과 처리는 같다
        sentences_item = _service(monkeypatch, output)[0].transcribe_video("v.webm", str(tmp_path), utterance=True)[0]
        assert sentences_item["phrase"]["accepted"] is True  # 같은 점수라도 문장 묶음은 자기 기준으로 판정한다


def test_ward_set_uses_ctc_for_every_window_but_phrases_only_for_single_window(tmp_path, monkeypatch) -> None:
    """ward 긴 녹화(여러 창)는 후보를 보내지 않지만 모든 인식 요청을 ctc로 한다. 업로드 경로도 같다."""
    two = [{"index": i, "start_ms": i * 1000, "end_ms": (i + 1) * 1000, "status": "ok", "npz": f"{i}.npz"}
           for i in range(2)]
    output = {"text": "가", "confidence": 0.9, "raw_score": -1.0}
    service, _roi, infer = _service(monkeypatch, output, two)
    items = service.transcribe_video("v.webm", str(tmp_path), utterance=True, phrase_set="ward")
    assert infer.calls == [{"cmd": "infer", "npz": "0.npz", "decoder": "ctc"},
                                        {"cmd": "infer", "npz": "1.npz", "decoder": "ctc"}]
    assert all("phrase" not in item for item in items)
    service, _roi, infer = _service(monkeypatch, output)
    service.transcribe_video("v.mp4", str(tmp_path), phrase_set="ward")
    assert infer.calls == [{"cmd": "infer", "npz": "a.npz", "decoder": "ctc"}]


def test_ward_set_without_list_file_keeps_free_recognition(tmp_path, monkeypatch) -> None:
    """ward 목록 파일을 끄면(빈 경로) 후보 없이 ctc 자유 인식만 남는다."""
    service, _roi, infer = _service(monkeypatch, {"text": "가", "confidence": 0.9, "raw_score": -1.0})
    monkeypatch.setattr(svc.config, "OLKAVS_WARD_PHRASE_FILE", "")
    assert svc.load_phrases("") == []
    item = service.transcribe_video("v.webm", str(tmp_path), utterance=True, phrase_set="ward")[0]
    assert infer.calls == [{"cmd": "infer", "npz": "a.npz", "decoder": "ctc"}] and "phrase" not in item


def test_unknown_phrase_set_rejected_before_workers(tmp_path, monkeypatch) -> None:
    """화이트리스트 밖의 묶음은 워커를 쓰기 전에 ValueError. 대기열 차례도 잡지 않는다."""
    service, roi, infer = _service(monkeypatch, {"text": "가", "confidence": 0.9, "raw_score": -1.0})
    with pytest.raises(ValueError):
        service.transcribe_video("v.webm", str(tmp_path), utterance=True, phrase_set="../prompts.json")
    assert roi.calls == [] and infer.calls == [] and service.gate.waiting() == 0
    service.transcribe_video("v.webm", str(tmp_path), utterance=True)  # 다음 작업은 그대로 처리된다
    assert len(infer.calls) == 1


# ---------------------------------------------------------------------------
# API: POST /api/record phrase_set
# ---------------------------------------------------------------------------

def _post_with(env, phrase_set: str | list[str] | None):
    """2초 WebM을 phrase_set과 함께 보낸다 (None이면 필드를 보내지 않고, 목록이면 같은 필드를 여러 번 보낸다)."""
    video = make_video(str(env["tmp"] / "clip.webm"), 2.0)
    with open(video, "rb") as handle:
        content = handle.read()
    data = {} if phrase_set is None else {"phrase_set": phrase_set}
    return client.post("/api/record", files={"file": ("recording.webm", content, "video/webm")}, data=data)


@pytest.mark.parametrize("sent,expected,decoder", [
    (None, "sentences", "hybrid"),
    ("sentences", "sentences", "hybrid"),
    ("ward", "ward", "ctc"),
])
def test_record_phrase_set_reaches_service(env, sent, expected, decoder) -> None:
    """phrase_set이 서비스까지 전달된다. 응답 형식은 그대로이고, ward Job의 모델 정보는 실제로 요청한 ctc를 알린다."""
    resp = _post_with(env, sent)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert set(body) == {"job_id", "session_id", "status", "duration_sec"} and body["status"] == "processing"
    try:
        call = env["service"].calls[-1]
        assert call["phrase_set"] == expected and call["utterance"] is True
        result = client.get(f"/api/result/{body['session_id']}").json()
        assert result["status"] == "done", result
        assert result["model"] == {**env["service"].model_info(), "decoder": decoder}
        assert os.listdir(env["upload_dir"]) == []
    finally:
        jm.clear_job(body["session_id"])


@pytest.mark.parametrize("value", ["Ward", "WARD", " ward", "ward ", "prompts", "commands.json",
                                   "../frontend/src/capture/prompts.json", "C:/secret/list.json", "sentences,ward",
                                   "0", "a" * 500,
                                   "",                        # 빈 값: FastAPI는 None으로 바꾸지만 원래 폼으로 거부한다
                                   ["ward", "ward"],          # 같은 필드 여러 번
                                   ["sentences", "ward"]])
def test_record_rejects_unknown_phrase_set(env, value) -> None:
    """화이트리스트 밖의 값·빈 값·중복 필드는 400 + 고정 한국어 문구. 보낸 값을 되돌려 보여 주지 않고 파일도 남기지 않는다."""
    resp = _post_with(env, value)
    assert resp.status_code == 400, resp.text
    detail = resp.json()["detail"]
    assert detail == record_module.MSG_BAD_PHRASE_SET
    assert all(phrase_set in detail for phrase_set in config.PHRASE_SET_IDS)
    assert os.listdir(env["upload_dir"]) == [] and env["service"].calls == []


class _FilePart:
    """폼에 문자열 대신 파일 파트로 들어온 값 흉내."""


@pytest.mark.parametrize("values,expected", [(None, "sentences"), ([], "sentences"), (["sentences"], "sentences"),
                                             (["ward"], "ward")])
def test_resolve_phrase_set_accepts_absent_or_exact_ids(values, expected) -> None:
    """필드가 없을 때만 기본값이고, 정확히 한 번 보낸 화이트리스트 ID는 그대로 받는다."""
    assert record_module.resolve_phrase_set(values) == expected


@pytest.mark.parametrize("values", [[""], ["ward", "ward"], ["ward", "sentences"], [_FilePart()], [None], ["ward "]])
def test_resolve_phrase_set_rejects_empty_duplicate_and_non_text(values) -> None:
    """빈 값·중복·문자열 아닌 파트는 기본값으로 바꾸지 않고 고정 문구로 거부한다."""
    with pytest.raises(record_module.RecordingRejected) as caught:
        record_module.resolve_phrase_set(values)
    assert caught.value.user_message == record_module.MSG_BAD_PHRASE_SET


def test_start_background_job_rejects_unknown_phrase_set(env) -> None:
    """(방어) 공용 Job 시작 함수도 화이트리스트 밖의 묶음을 ValueError로 거부하고 세션을 남기지 않는다."""
    from fastapi import BackgroundTasks

    from api import infer as infer_module

    session_id = "4b9f7a3e-1c2d-4e5f-8a9b-0c1d2e3f4a5b"
    with pytest.raises(ValueError):
        infer_module.start_background_job(session_id, "x", BackgroundTasks(), mode="record", phrase_set="WARD")
    assert session_id not in infer_module._running_sessions and jm.get_job(session_id) is None


# ---------------------------------------------------------------------------
# 상태: 병실 목록을 읽을 수 있는지 (/health, 기동 로그)
# ---------------------------------------------------------------------------

def test_ward_phrase_status_reports_real_list() -> None:
    """저장소의 병실 목록: 켜짐, 후보 20개, 버전 문자열. 경로는 상태에 넣지 않는다."""
    if os.environ.get("LIPNET_OLKAVS_WARD_PHRASE_FILE") is not None:
        pytest.skip("환경 변수로 ward 목록 파일을 덮어쓴 실행")
    assert svc.ward_phrase_status() == {"enabled": True, "count": 20, "version": "ward-commands-v1"}


@pytest.mark.parametrize("content,expected", [
    (None, {"enabled": True, "count": 0, "version": None}),                           # 파일 없음
    ("{not json", {"enabled": True, "count": 0, "version": None}),                    # 깨진 JSON
    ('{"fixed": [{"id": "W01"}]}', {"enabled": True, "count": 0, "version": None}),   # text 없음
    ('{"version": "v9", "fixed": [{"id": "W01", "text": "가"}]}', {"enabled": True, "count": 0, "version": "v9"}),
    ("[1, 2]", {"enabled": True, "count": 0, "version": None}),
    ('{"version": 3, "fixed": [{"id": "A", "text": "가"}, {"id": "B", "text": "나"}]}',
     {"enabled": True, "count": 2, "version": None}),
])
def test_ward_phrase_status_counts_only_usable_lists(tmp_path, monkeypatch, content, expected) -> None:
    """읽지 못했거나 2개 미만이면 count 0이다(실제로 후보를 보내지 않는 load_phrases와 같은 규칙)."""
    path = tmp_path / "commands.json"
    if content is not None:
        path.write_text(content, encoding="utf-8")
    monkeypatch.setattr(svc.config, "OLKAVS_WARD_PHRASE_FILE", str(path))
    assert svc.ward_phrase_status() == expected
    assert len(svc.load_phrases(str(path))) == expected["count"]


def test_ward_phrase_status_disabled_by_empty_path(monkeypatch, caplog) -> None:
    """빈 경로(일부러 끔)는 enabled=False이고 ERROR 로그를 남기지 않는다."""
    monkeypatch.setattr(svc.config, "OLKAVS_WARD_PHRASE_FILE", "")
    with caplog.at_level(logging.ERROR, logger=svc.logger.name):
        assert svc.check_ward_phrases() == {"enabled": False, "count": 0, "version": None}
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]


def test_status_includes_ward_phrases_without_paths(tmp_path, monkeypatch) -> None:
    """/health(service.status())에 병실 목록 상태가 있고, 목록 파일 경로는 보이지 않는다."""
    monkeypatch.setattr(svc.config, "OLKAVS_WARD_PHRASE_FILE", str(tmp_path / "secret_dir" / "commands.json"))
    status = svc.OlkavsService(_StubWorker(lambda p, _p: {}), _StubWorker(lambda p, _p: {})).status()
    assert status["ward_phrases"] == {"enabled": True, "count": 0, "version": None}
    assert "secret_dir" not in json.dumps(status, ensure_ascii=False, default=str)


def test_get_service_logs_error_once_when_ward_list_unreadable(tmp_path, monkeypatch, caplog) -> None:
    """서비스를 처음 만들 때만 목록을 점검하고, 설정된 목록을 읽지 못하면 ERROR를 한 번 남긴다."""
    monkeypatch.setattr(svc, "_service", None)
    monkeypatch.setattr(svc, "OlkavsService", lambda: object())
    monkeypatch.setattr(svc.config, "OLKAVS_WARD_PHRASE_FILE", str(tmp_path / "missing.json"))
    with caplog.at_level(logging.ERROR, logger=svc.logger.name):
        first = svc.get_service()
        assert svc.get_service() is first
    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert len(errors) == 1 and "LIPNET_OLKAVS_WARD_PHRASE_FILE" in errors[0].getMessage()


def test_get_service_is_quiet_when_ward_list_loads(monkeypatch, caplog) -> None:
    """저장소 목록을 읽을 수 있으면 ERROR가 없다."""
    monkeypatch.setattr(svc, "_service", None)
    monkeypatch.setattr(svc, "OlkavsService", lambda: object())
    monkeypatch.setattr(svc.config, "OLKAVS_WARD_PHRASE_FILE",
                        os.path.join(config.PROJECT_ROOT, "frontend", "src", "ward", "commands.json"))
    with caplog.at_level(logging.ERROR, logger=svc.logger.name):
        svc.get_service()
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]


# ---------------------------------------------------------------------------
# 스키마
# ---------------------------------------------------------------------------

def test_phrase_match_schema_set_field() -> None:
    """set은 화이트리스트 ID만 받고, 없던 예전 결과는 None이다. 필드 순서는 계약 순서다."""
    phrase = {"id": "W05", "text": "목이 말라 물 한 모금만", "score": -2.1, "margin": 0.7, "runner_up_id": "W01",
              "accepted": True}
    item = {"start_ms": 0, "end_ms": 3000, "text": "가", "confidence": 0.9, "raw_score": -1.0}
    assert ResultItem(**item, phrase={**phrase, "set": "ward"}).phrase.set == "ward"
    assert ResultItem(**item, phrase=phrase).phrase.set is None
    with pytest.raises(ValueError):  # pydantic ValidationError는 ValueError의 하위 클래스다
        ResultItem(**item, phrase={**phrase, "set": "prompts"})
    assert list(PhraseMatch.model_fields) == ["id", "text", "score", "margin", "runner_up_id", "accepted", "set"]
    literal = typing.get_args(PhraseMatch.model_fields["set"].annotation)[0]
    assert typing.get_args(literal) == config.PHRASE_SET_IDS
    dumped = ResultItem(**item, phrase={**phrase, "set": "sentences"}).model_dump()["phrase"]
    assert dumped == {**phrase, "set": "sentences"}
