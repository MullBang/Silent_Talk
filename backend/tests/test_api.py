"""API 통합 테스트 (FastAPI TestClient).

설계문서 6절 테스트 케이스 TC-01~TC-05를 명시적으로 매핑하고, 추가로 MIME
불일치/404/409/결과 폴링/CORS 화이트리스트/평가 검증을 포함한다.

- TC-01 정상 업로드 → 200 + session_id
- TC-02 미지원 형식(TXT) → 400
- TC-03 파일 크기 초과(mock) → 400
- TC-04 영상 길이 초과(mock) → 400
- TC-05 추론 Job 시작 → {job_id, session_id, status:'processing'}, results 미포함
"""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

# backend 패키지 루트를 import 경로에 추가
_BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

from api import infer as infer_module  # noqa: E402
from api import upload as upload_module  # noqa: E402
from config import CORS_ORIGINS  # noqa: E402
from main import app  # noqa: E402
from services import job_manager as jm  # noqa: E402

client = TestClient(app)


def _make_mp4(path: str, fps: float = 25.0, num_frames: int = 50) -> None:
    """테스트용 소형 mp4를 생성한다 (기본 2초, 64×64).

    설계 TC는 50MB MP4를 명시하나, 검증 로직은 내용/길이/형식 기반이므로
    동일 경로를 타는 소형 유효 mp4로 대체한다.
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (64, 64))
    assert writer.isOpened()
    for i in range(num_frames):
        writer.write(np.full((64, 64, 3), (i * 5) % 256, dtype=np.uint8))
    writer.release()


@pytest.fixture()
def tmp_upload_dir(tmp_path, monkeypatch):
    """업로드/추론 모듈의 임시 디렉토리를 테스트 격리 폴더로 교체한다.

    이 파일의 기존 시나리오는 compact 전처리 경로를 검증하므로 백엔드를 compact로 고정한다
    (기본값 olkavs는 상주 워커를 띄운다 — 해당 경로는 test_olkavs_service.py에서 대체물로 검증).
    """
    d = str(tmp_path / "uploads")
    os.makedirs(d, exist_ok=True)
    monkeypatch.setattr(upload_module, "TMP_UPLOAD_DIR", d)
    monkeypatch.setattr(infer_module, "TMP_UPLOAD_DIR", d)
    monkeypatch.setattr(infer_module, "TMP_ROI_DIR", str(tmp_path / "roi"))
    monkeypatch.setattr(infer_module, "MODEL_BACKEND", "compact")
    return d


def _upload_sample(tmp_path) -> str:
    """소형 mp4를 업로드하고 session_id를 반환한다."""
    mp4 = str(tmp_path / "sample.mp4")
    _make_mp4(mp4)
    with open(mp4, "rb") as f:
        resp = client.post(
            "/api/upload", files={"file": ("sample.mp4", f.read(), "video/mp4")}
        )
    assert resp.status_code == 200, resp.text
    return resp.json()["session_id"]


# ---------------------------------------------------------------------------
# 설계문서 테스트 케이스 TC-01 ~ TC-05
# ---------------------------------------------------------------------------

def test_tc01_normal_upload(tmp_upload_dir, tmp_path) -> None:
    """TC-01: 정상 영상 업로드 → 200 + session_id 포함."""
    mp4 = str(tmp_path / "v.mp4")
    _make_mp4(mp4)
    with open(mp4, "rb") as f:
        resp = client.post(
            "/api/upload", files={"file": ("v.mp4", f.read(), "video/mp4")}
        )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["session_id"]
    assert body["file_name"] == "v.mp4"
    assert body["duration_sec"] == pytest.approx(2.0, abs=0.3)


def test_tc02_unsupported_format(tmp_upload_dir) -> None:
    """TC-02: 미지원 파일 형식(TXT) → 400 + 에러 메시지."""
    resp = client.post(
        "/api/upload",
        files={"file": ("note.txt", b"hello world", "text/plain")},
    )
    assert resp.status_code == 400
    assert "detail" in resp.json()


def test_tc03_file_size_exceeded(tmp_upload_dir, tmp_path, monkeypatch) -> None:
    """TC-03: 파일 크기 초과(mock) → 400.

    실제 500MB 파일 대신 허용 크기를 0MB로 낮춰 초과 상황을 시뮬레이션한다.
    """
    monkeypatch.setattr(upload_module, "MAX_UPLOAD_SIZE_MB", 0)
    mp4 = str(tmp_path / "big.mp4")
    _make_mp4(mp4)
    with open(mp4, "rb") as f:
        resp = client.post(
            "/api/upload", files={"file": ("big.mp4", f.read(), "video/mp4")}
        )
    assert resp.status_code == 400
    assert "이하" in resp.json()["detail"]


def test_tc04_duration_exceeded(tmp_upload_dir, tmp_path, monkeypatch) -> None:
    """TC-04: 영상 길이 초과(mock) → 400.

    get_video_duration_sec가 4분(240초) 초과로 ValueError를 던지는 상황을
    시뮬레이션한다.
    """

    def _fake_duration(_path: str) -> float:
        raise ValueError("영상 길이(240.00s)가 허용 최대치(180s)를 초과했습니다.")

    monkeypatch.setattr(upload_module, "get_video_duration_sec", _fake_duration)
    mp4 = str(tmp_path / "long.mp4")
    _make_mp4(mp4)
    with open(mp4, "rb") as f:
        resp = client.post(
            "/api/upload", files={"file": ("long.mp4", f.read(), "video/mp4")}
        )
    assert resp.status_code == 400
    assert "초과" in resp.json()["detail"]
    # 길이 초과 시 임시 파일이 정리되어야 한다
    assert os.listdir(tmp_upload_dir) == []


def test_tc05_infer_job_start(tmp_upload_dir, tmp_path) -> None:
    """TC-05: 업로드 후 추론 시작 → {job_id, session_id, status:'processing'}.

    설계 규칙: 응답에 results 필드를 절대 포함하지 않는다.
    """
    session_id = _upload_sample(tmp_path)
    resp = client.post("/api/infer", json={"session_id": session_id})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["job_id"]
    assert body["session_id"] == session_id
    assert body["status"] == "processing"
    assert "results" not in body  # 즉시 결과 반환 금지


# ---------------------------------------------------------------------------
# 추가 검증 (설계 규칙/에러 처리)
# ---------------------------------------------------------------------------

def test_upload_invalid_mime_mismatch(tmp_upload_dir) -> None:
    """확장자는 mp4지만 내용이 영상이 아니면 MIME 불일치로 400."""
    resp = client.post(
        "/api/upload",
        files={"file": ("fake.mp4", b"this is plain text, not a video", "video/mp4")},
    )
    assert resp.status_code == 400


def test_infer_missing_session_404(tmp_upload_dir) -> None:
    """업로드되지 않은 세션 추론 요청은 404."""
    resp = client.post("/api/infer", json={"session_id": "ghost-session"})
    assert resp.status_code == 404


def test_infer_conflict_409(tmp_upload_dir, tmp_path) -> None:
    """이미 처리 중인 세션은 409."""
    session_id = _upload_sample(tmp_path)
    # 처리 중 상태를 인위적으로 만든 뒤 재요청
    jm.create_job(session_id)
    jm.update_progress(session_id, 10.0)  # → processing
    try:
        resp = client.post("/api/infer", json={"session_id": session_id})
        assert resp.status_code == 409
    finally:
        jm.clear_job(session_id)


def test_result_polling_flow(tmp_upload_dir, tmp_path) -> None:
    """업로드 → 추론 → 결과 폴링 (TestClient는 백그라운드 동기 실행)."""
    session_id = _upload_sample(tmp_path)
    client.post("/api/infer", json={"session_id": session_id})

    resp = client.get(f"/api/result/{session_id}")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["session_id"] == session_id
    assert body["status"] == "done"
    assert body["results"] is not None
    # 합성 영상(얼굴 없음)은 전부 검출 실패 구간
    assert all(item["text"] == "[검출 실패 구간]" for item in body["results"])


def test_result_missing_session_404() -> None:
    """존재하지 않는 세션 결과 조회는 404."""
    assert client.get("/api/result/no-such-session").status_code == 404
    assert client.get("/api/result/0b9f7a3e-1c2d-4e5f-8a9b-0c1d2e3f4a5b").status_code == 404


@pytest.mark.parametrize("bad_id", ["*", "../victim", "..\\victim", "a*", "C:\\victim", "", "0B9F7A3E-1C2D-4E5F-8A9B-0C1D2E3F4A5B"])
def test_infer_rejects_non_uuid_session_ids(tmp_upload_dir, tmp_path, bad_id) -> None:
    """UUID4가 아닌 session_id는 다른 세션·업로드 폴더 밖 파일을 집거나 지우지 못한다 (404)."""
    session_id = _upload_sample(tmp_path)  # 다른 사용자의 업로드
    outside = tmp_path / "victim_keep.txt"
    outside.write_text("keep", encoding="utf-8")
    resp = client.post("/api/infer", json={"session_id": bad_id})
    assert resp.status_code == 404
    assert outside.exists()
    assert any(name.startswith(session_id) for name in os.listdir(tmp_upload_dir))
    assert jm.get_job(bad_id) is None


def test_result_rejects_glob_session_id(tmp_upload_dir, tmp_path) -> None:
    """GET /api/result/* 로 다른 세션 결과를 읽을 수 없다."""
    session_id = _upload_sample(tmp_path)
    client.post("/api/infer", json={"session_id": session_id})
    assert client.get(f"/api/result/{session_id}").status_code == 200
    assert client.get("/api/result/*").status_code == 404
    jm.clear_job(session_id)


def test_compact_missing_weights_fails_job_instead_of_dummy_text(tmp_upload_dir, tmp_path, monkeypatch) -> None:
    """가중치가 없으면 결과처럼 보이는 더미 문구 대신 Job을 error로 끝낸다."""
    monkeypatch.setattr(infer_module, "_model", None)
    monkeypatch.setattr(infer_module, "_model_loaded", True)  # 가중치 없음 상태
    monkeypatch.setattr(infer_module, "run_preprocessing_pipeline",
                        lambda _p: [{"tensor": object(), "start_ms": 0, "end_ms": 3000, "skipped": False}])
    session_id = _upload_sample(tmp_path)
    client.post("/api/infer", json={"session_id": session_id})
    body = client.get(f"/api/result/{session_id}").json()
    assert body["status"] == "error"
    assert "가중치" in body["error_message"]
    assert body["results"] is None
    assert os.listdir(tmp_upload_dir) == []  # 실패해도 업로드 파일은 즉시 삭제
    jm.clear_job(session_id)


def test_olkavs_backend_job_uses_service(tmp_upload_dir, tmp_path, monkeypatch) -> None:
    """olkavs 백엔드: 서비스 결과(raw_score 포함)를 저장하고 업로드·입술 임시 폴더를 지운다."""
    from services import olkavs_service

    calls = {}

    class FakeService:
        def transcribe_video(self, video_path, work_dir, on_progress=None, should_abort=None, **kwargs):
            calls["video"], calls["work_dir"] = video_path, work_dir
            calls["kwargs"] = kwargs
            remaining = kwargs["remaining_seconds"]()
            assert 0 < remaining <= infer_module.JOB_TIMEOUT_SEC
            kwargs["warnings"].append("낮은 프레임률 경고")
            os.makedirs(work_dir, exist_ok=True)
            with open(os.path.join(work_dir, "seg_000.npz"), "wb") as handle:
                handle.write(b"x")
            on_progress(40.0)
            assert should_abort() is False
            on_progress(100.0)
            return [{"start_ms": 0, "end_ms": 2000, "text": "안녕하세요", "confidence": 0.8, "raw_score": -3.5},
                    {"start_ms": 2000, "end_ms": 2600, "text": olkavs_service.FAIL_TEXT, "confidence": None, "raw_score": None},
                    {"start_ms": 2600, "end_ms": 3000, "text": olkavs_service.UNREC_TEXT, "confidence": 0.1,
                     "raw_score": -40.0, "candidate": "낮은 확신 문장"}]

        def model_info(self):
            return {"backend": "olkavs", "decoder": "hybrid", "checkpoint_sha256": "f41d7423",
                    "upstream_commit": "abc", "roi_profile": "p"}

    monkeypatch.setattr(infer_module, "MODEL_BACKEND", "olkavs")
    monkeypatch.setattr(olkavs_service, "get_service", lambda: FakeService())
    session_id = _upload_sample(tmp_path)
    client.post("/api/infer", json={"session_id": session_id})
    body = client.get(f"/api/result/{session_id}").json()
    assert body["status"] == "done", body
    assert body["results"][0] == {"start_ms": 0, "end_ms": 2000, "text": "안녕하세요", "confidence": 0.8,
                                  "raw_score": -3.5, "candidate": None}
    assert body["results"][1]["text"] == "[검출 실패 구간]"
    assert body["results"][2] == {"start_ms": 2600, "end_ms": 3000, "text": "[인식 불가]", "confidence": 0.1,
                                  "raw_score": -40.0, "candidate": "낮은 확신 문장"}
    assert body["warnings"] == ["낮은 프레임률 경고"]
    assert body["model"] == {"backend": "olkavs", "decoder": "hybrid", "checkpoint_sha256": "f41d7423",
                             "upstream_commit": "abc", "roi_profile": "p"}
    assert "video_seconds" in calls["kwargs"]
    assert calls["kwargs"]["utterance"] is False  # 업로드 경로는 발화 단위 잘라 내기를 하지 않는다
    assert calls["work_dir"] == os.path.join(str(tmp_path / "roi"), session_id)
    assert not os.path.exists(calls["work_dir"])
    assert os.listdir(tmp_upload_dir) == []
    jm.clear_job(session_id)


def test_probe_video_seconds_uses_metadata_only(tmp_path) -> None:
    """영상 길이 어림은 메타데이터만 쓰고, 읽을 수 없거나 정보가 없으면 None."""
    import cv2
    import numpy as np

    path = str(tmp_path / "probe.avi")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (32, 32))
    for _ in range(45):
        writer.write(np.zeros((32, 32, 3), dtype=np.uint8))
    writer.release()
    assert abs(infer_module._probe_video_seconds(path) - 1.5) < 0.05
    assert infer_module._probe_video_seconds(str(tmp_path / "missing.mp4")) is None
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(b"not a video")
    assert infer_module._probe_video_seconds(str(junk)) is None


def test_result_without_warnings_is_null(tmp_upload_dir) -> None:
    """경고·모델 정보·candidate가 없으면 응답의 해당 필드는 null이다 (기존 클라이언트 호환)."""
    session_id = "1b9f7a3e-1c2d-4e5f-8a9b-0c1d2e3f4a5b"
    jm.create_job(session_id)
    try:
        assert client.get(f"/api/result/{session_id}").json()["model"] is None
        jm.complete_job(session_id, [{"start_ms": 0, "end_ms": 10, "text": "가", "confidence": 0.5}], warnings=[])
        body = client.get(f"/api/result/{session_id}").json()
        assert body["status"] == "done" and body["warnings"] is None
        assert body["model"] is None and body["results"][0]["candidate"] is None
    finally:
        jm.clear_job(session_id)


def test_compact_result_has_no_model_info(tmp_upload_dir, tmp_path) -> None:
    """compact 백엔드 결과에는 모델 정보(model)가 null이다."""
    session_id = _upload_sample(tmp_path)
    client.post("/api/infer", json={"session_id": session_id})
    body = client.get(f"/api/result/{session_id}").json()
    try:
        assert body["status"] == "done" and body["model"] is None
    finally:
        jm.clear_job(session_id)


def _assert_no_path(detail: str, tmp_path) -> None:
    """사용자 문구에 서버 경로·예외 원문이 없는지."""
    for bad in (str(tmp_path), ":\\", ":/", "\\", "Error", "Traceback", "uploads", ".mp4"):
        assert bad not in detail, (bad, detail)


@pytest.mark.parametrize("exc_factory, expected", [
    (lambda p: FileNotFoundError(f"영상을 열 수 없습니다: {p}"), upload_module.MSG_VIDEO_UNREADABLE),
    (lambda p: ValueError(f"유효하지 않은 원본 FPS(0.0)입니다: {p}"), upload_module.MSG_VIDEO_UNREADABLE),
    (lambda p: ValueError(f"프레임이 존재하지 않습니다: {p}"), upload_module.MSG_VIDEO_UNREADABLE),
    (lambda p: RuntimeError(f"cv2 crashed reading {p}"), upload_module.MSG_VIDEO_UNREADABLE),
    (lambda p: ValueError("영상 길이(240.00s)가 허용 최대치(180s)를 초과했습니다."), upload_module.MSG_DURATION_EXCEEDED),
])
def test_upload_errors_use_fixed_messages_without_paths(tmp_upload_dir, tmp_path, monkeypatch, exc_factory,
                                                        expected) -> None:
    """길이 검사 오류 문구의 서버 경로·예외 원문을 사용자에게 보내지 않고, 임시 파일을 지운다."""
    def fake_duration(path: str) -> float:
        raise exc_factory(path)

    monkeypatch.setattr(upload_module, "get_video_duration_sec", fake_duration)
    mp4 = str(tmp_path / "v.mp4")
    _make_mp4(mp4)
    with open(mp4, "rb") as f:
        resp = client.post("/api/upload", files={"file": ("v.mp4", f.read(), "video/mp4")})
    assert resp.status_code == 400
    assert resp.json()["detail"] == expected
    _assert_no_path(resp.json()["detail"], tmp_path)
    assert os.listdir(tmp_upload_dir) == []


def test_upload_mime_mismatch_detail_is_fixed(tmp_upload_dir, tmp_path) -> None:
    """MIME 불일치 안내는 감지된 내부 값을 섞지 않은 고정 문구다."""
    resp = client.post("/api/upload", files={"file": ("fake.mp4", b"plain text, not a video", "video/mp4")})
    assert resp.status_code == 400
    assert resp.json()["detail"] == upload_module.MSG_MIME_MISMATCH
    _assert_no_path(resp.json()["detail"], tmp_path)
    assert "text/plain" not in resp.json()["detail"] and "octet-stream" not in resp.json()["detail"]


def test_upload_save_failure_detail_is_fixed(tmp_upload_dir, tmp_path, monkeypatch) -> None:
    """저장 실패(500)도 경로·예외 원문 없이 고정 문구다."""
    import builtins

    real_open = builtins.open

    def failing_open(file, mode="r", *args, **kwargs):
        if "b" in mode and ("w" in mode or "x" in mode) and str(file).startswith(tmp_upload_dir):
            raise OSError(f"disk full: {file}")
        return real_open(file, mode, *args, **kwargs)

    mp4 = str(tmp_path / "v.mp4")
    _make_mp4(mp4)
    with real_open(mp4, "rb") as f:
        content = f.read()
    monkeypatch.setattr(builtins, "open", failing_open)
    resp = client.post("/api/upload", files={"file": ("v.mp4", content, "video/mp4")})
    monkeypatch.setattr(builtins, "open", real_open)
    assert resp.status_code == 500
    assert resp.json()["detail"] == upload_module.MSG_SAVE_FAILED
    _assert_no_path(resp.json()["detail"], tmp_path)
    assert os.listdir(tmp_upload_dir) == []


_ORIGIN = "http://localhost:5173"
_BOUNDARY = "lipnetUploadBoundary"


def _asgi_upload(first: bytes, total_bytes: int, content_length: int | None, chunk: int = 256 * 1024) -> dict:
    """앱을 ASGI로 직접 불러 /api/upload 본문을 조각으로 흘려 보낸다. 서버가 receive()로 가져간 바이트를 센다."""
    import asyncio
    import json as _json

    headers = [(b"host", b"testserver"), (b"origin", _ORIGIN.encode()),
               (b"content-type", f"multipart/form-data; boundary={_BOUNDARY}".encode())]
    if content_length is not None:
        headers.append((b"content-length", str(content_length).encode()))
    scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "POST",
             "scheme": "http", "path": "/api/upload", "raw_path": b"/api/upload", "root_path": "",
             "query_string": b"", "headers": headers, "client": ("127.0.0.1", 50001), "server": ("testserver", 80)}
    state = {"sent": 0, "first": True}
    messages: list[dict] = []

    async def receive() -> dict:
        if state["sent"] >= total_bytes:
            return {"type": "http.disconnect"}
        body = first if state["first"] else b"\x00" * min(chunk, total_bytes - state["sent"])
        state["first"] = False
        state["sent"] += len(body)
        return {"type": "http.request", "body": body, "more_body": state["sent"] < total_bytes}

    async def send(message: dict) -> None:
        messages.append(message)

    asyncio.run(app(scope, receive, send))
    start = next(m for m in messages if m["type"] == "http.response.start")
    body = b"".join(m.get("body", b"") for m in messages if m["type"] == "http.response.body")
    return {"status": start["status"], "headers": {k.decode().lower(): v.decode() for k, v in start["headers"]},
            "json": _json.loads(body.decode("utf-8")), "pulled": state["sent"]}


def _upload_prefix() -> bytes:
    return (f"--{_BOUNDARY}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"big.mp4\"\r\n"
            "Content-Type: video/mp4\r\n\r\n").encode("ascii") + b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00"


@pytest.mark.parametrize("declared", ["over", None, 1024])
def test_upload_oversize_body_rejected_with_413_while_receiving(tmp_upload_dir, monkeypatch, declared) -> None:
    """MAX_UPLOAD_SIZE_MB+여유분을 넘는 본문은 413(+CORS). 큰 Content-Length는 본문을 받지 않고, 길이 없음/속인
    길이는 받은 바이트가 상한을 넘는 즉시 끊는다 (50MB를 끝까지 받지 않는다)."""
    from config import REQUEST_BODY_OVERHEAD_BYTES

    monkeypatch.setattr(upload_module, "MAX_UPLOAD_SIZE_MB", 1)
    limit = 1024 * 1024 + REQUEST_BODY_OVERHEAD_BYTES
    content_length = limit + 1 if declared == "over" else declared
    chunk = 256 * 1024
    out = _asgi_upload(_upload_prefix(), 50 * 1024 * 1024, content_length, chunk=chunk)
    assert out["status"] == 413, out
    assert out["json"] == {"detail": upload_module.MSG_TOO_LARGE.format(mb=1)}
    assert out["headers"].get("access-control-allow-origin") == _ORIGIN
    if declared == "over":
        assert out["pulled"] == 0
    else:
        assert limit < out["pulled"] <= limit + chunk
    assert os.listdir(tmp_upload_dir) == []


def test_upload_streams_to_disk_and_probes_duration_off_event_loop(tmp_upload_dir, tmp_path, monkeypatch) -> None:
    """업로드는 전체 읽기(read(-1)) 없이 앞부분만 읽고 나머지는 스레드에서 1MB 단위로 저장한다.
    길이 검사도 이벤트 루프 밖(스레드)에서 돈다. 파일 상한(여유분 안의 초과)은 상한+1바이트까지만 옮기고 400."""
    import asyncio

    from starlette.datastructures import UploadFile as StarletteUploadFile

    from api import limits as limits_module

    reads: list[int] = []
    original_read = StarletteUploadFile.read

    async def spy_read(self, size: int = -1):
        assert size != -1, "파일 전체를 메모리로 읽으면 안 된다"
        data = await original_read(self, size)
        reads.append(len(data))
        return data

    monkeypatch.setattr(StarletteUploadFile, "read", spy_read)
    probe_threads: list[bool] = []
    real_duration = upload_module.get_video_duration_sec

    def duration_off_loop(path: str) -> float:
        try:
            asyncio.get_running_loop()
            probe_threads.append(False)  # 이벤트 루프 위에서 불렸다
        except RuntimeError:
            probe_threads.append(True)
        return real_duration(path)

    monkeypatch.setattr(upload_module, "get_video_duration_sec", duration_off_loop)
    mp4 = str(tmp_path / "v.mp4")
    _make_mp4(mp4)
    with open(mp4, "rb") as f:
        content = f.read()
    resp = client.post("/api/upload", files={"file": ("v.mp4", content, "video/mp4")})
    assert resp.status_code == 200, resp.text
    assert probe_threads == [True]
    assert sum(reads) <= 4096  # 앞부분(형식 판정용)만 UploadFile.read로 읽는다
    saved = os.path.join(tmp_upload_dir, f"{resp.json()['session_id']}_v.mp4")
    with open(saved, "rb") as f:
        assert f.read() == content
    os.remove(saved)

    copied: list[int] = []
    original_copy = limits_module._copy_capped

    def spy_copy(source, head, path, max_bytes):
        total = original_copy(source, head, path, max_bytes)
        copied.append(total)
        return total

    monkeypatch.setattr(limits_module, "_copy_capped", spy_copy)
    monkeypatch.setattr(upload_module, "MAX_UPLOAD_SIZE_MB", 1)
    big = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * (1024 * 1024 + 512 * 1024)
    resp = client.post("/api/upload", files={"file": ("big.mp4", big, "video/mp4")})
    assert resp.status_code == 400
    assert resp.json()["detail"] == upload_module.MSG_TOO_LARGE.format(mb=1)
    assert copied == [1024 * 1024 + 1]
    assert os.listdir(tmp_upload_dir) == []


def test_job_timeout_is_applied_and_not_overwritten(tmp_upload_dir) -> None:
    """시작 후 JOB_TIMEOUT_SEC를 넘긴 Job은 폴링 시 timeout이 되고, 늦은 완료가 덮어쓰지 않는다."""
    from datetime import datetime, timedelta

    session_id = "0b9f7a3e-1c2d-4e5f-8a9b-0c1d2e3f4a5b"
    jm.create_job(session_id)
    jm.update_progress(session_id, 30.0)
    jm.get_job(session_id).started_at = datetime.now() - timedelta(seconds=infer_module.JOB_TIMEOUT_SEC + 5)
    try:
        assert client.get(f"/api/result/{session_id}").json()["status"] == "timeout"
        jm.complete_job(session_id, [{"text": "늦은 결과"}])
        jm.fail_job(session_id, "늦은 실패")
        jm.update_progress(session_id, 90.0)
        record = jm.get_job(session_id)
        assert record.status == "timeout" and record.results is None and record.progress == 30.0
    finally:
        jm.clear_job(session_id)


def test_same_session_rejected_while_previous_job_thread_runs(tmp_upload_dir, tmp_path, monkeypatch) -> None:
    """타임아웃으로 응답했어도 이전 작업 스레드가 돌고 있으면 같은 세션 재요청은 409."""
    session_id = _upload_sample(tmp_path)
    monkeypatch.setattr(infer_module, "_running_sessions", {session_id})
    jm.create_job(session_id)
    jm.timeout_job(session_id)
    try:
        assert client.post("/api/infer", json={"session_id": session_id}).status_code == 409
    finally:
        jm.clear_job(session_id)


def test_olkavs_job_aborts_on_server_side_timeout_without_polling(tmp_upload_dir, tmp_path, monkeypatch) -> None:
    """폴링이 없어도 작업 스스로 Job 시간 상한을 판정해 중단하고, 고정 문구로 timeout을 남긴다."""
    from datetime import datetime, timedelta

    from services import olkavs_service

    class SlowService:
        def transcribe_video(self, video_path, work_dir, on_progress=None, should_abort=None, **kwargs):
            jm.get_job(session_id).started_at = datetime.now() - timedelta(seconds=infer_module.JOB_TIMEOUT_SEC + 1)
            if should_abort():
                raise olkavs_service.JobAborted()
            raise AssertionError("시간 상한을 넘겼는데 중단하지 않음")

    monkeypatch.setattr(infer_module, "MODEL_BACKEND", "olkavs")
    monkeypatch.setattr(olkavs_service, "get_service", lambda: SlowService())
    session_id = _upload_sample(tmp_path)
    jm.create_job(session_id)
    infer_module._run_infer_job(session_id, infer_module._find_upload_file(session_id))
    record = jm.get_job(session_id)
    assert record.status == "timeout"
    assert os.listdir(tmp_upload_dir) == [] and session_id not in infer_module._running_sessions
    jm.clear_job(session_id)


def test_user_error_message_hides_paths() -> None:
    """고정 문구가 있으면 그것을, 없으면 경로를 지운 짧은 문구를 사용자에게 보낸다."""
    from services.olkavs_service import WorkerError

    assert infer_module.user_error_message(WorkerError("고정 문구", "C:/secret/log | tail")) == "고정 문구"
    text = infer_module.user_error_message(ValueError(r"영상을 열 수 없습니다: C:\Users\me\tmp\a.mp4"))
    assert "C:\\Users" not in text and "<경로>" in text
    assert infer_module.user_error_message(ValueError("")) == "영상을 처리하지 못했습니다."


def test_health_reports_model_backend() -> None:
    """/health는 인식 백엔드와 모델 상태를 알려 주고 워커를 새로 띄우지 않는다."""
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_backend"] in ("olkavs", "compact")
    assert isinstance(body["model"], dict)


def test_evaluation_requires_registered_test_set() -> None:
    """등록되지 않은 test_set_id는 400 (경로 직접 전달 방지)."""
    resp = client.post(
        "/api/evaluation/run",
        json={"test_set_id": "unregistered", "eval_unit": "word"},
    )
    assert resp.status_code == 400


def test_cors_allows_whitelisted_origin_only() -> None:
    """화이트리스트 Origin만 허용하고 미등록 Origin/와일드카드는 거부."""
    allowed = CORS_ORIGINS[0]
    r_ok = client.get("/health", headers={"Origin": allowed})
    assert r_ok.headers.get("access-control-allow-origin") == allowed

    r_bad = client.get("/health", headers={"Origin": "http://evil.example.com"})
    assert r_bad.headers.get("access-control-allow-origin") != "http://evil.example.com"
    assert r_bad.headers.get("access-control-allow-origin") != "*"
