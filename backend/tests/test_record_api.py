"""POST /api/record (웹캠 녹화 접수, 6단계) 검사.

테스트 영상은 실행 중에 만든다: cv2.VideoWriter(VP80 → .webm, mp4v → .mp4)가 안 되면 프로젝트의 ffmpeg
바이너리로, 그것도 없으면 해당 검사를 건너뛴다. 인식 서비스는 대체물로 바꿔 record 모드 Job의 인자·결과·
임시 파일 삭제를 확인한다. 실제 OLKAVS 워커는 띄우지 않는다.
"""
from __future__ import annotations

import os
import subprocess
import sys

import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")
pytest.importorskip("fastapi")
pytest.importorskip("multipart")

from fastapi.testclient import TestClient  # noqa: E402

_BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

import config  # noqa: E402
from api import infer as infer_module  # noqa: E402
from api import record as record_module  # noqa: E402
from main import app  # noqa: E402
from services import job_manager as jm  # noqa: E402
from services import olkavs_service  # noqa: E402

client = TestClient(app)

_FFMPEG = os.path.join(config.PROJECT_ROOT, "experiments", "olkavs_official", "audio_tools", "imageio_ffmpeg",
                       "binaries", "ffmpeg-win-x86_64-v7.1.exe")

_FIXED_400 = {
    record_module.MSG_EMPTY,
    record_module.MSG_UNSUPPORTED,
    record_module.MSG_TOO_LARGE.format(mb=config.RECORD_MAX_SIZE_MB),
    record_module.MSG_UNREADABLE,
    record_module.MSG_TOO_SHORT.format(sec=config.RECORD_MIN_SECONDS),
    record_module.MSG_TOO_LONG.format(sec=config.RECORD_MAX_SECONDS),
}


def _write_with_cv2(path: str, frames: int, fps: float, fourcc: str) -> bool:
    """cv2로 작은 영상을 쓴다. 성공하면 True."""
    try:
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*fourcc), fps, (32, 32))
        if not writer.isOpened():
            return False
        for i in range(frames):
            writer.write(np.full((32, 32, 3), (i * 7) % 256, dtype=np.uint8))
        writer.release()
        return os.path.exists(path) and os.path.getsize(path) > 0
    except Exception:  # noqa: BLE001 - 코덱 미지원이면 다음 방법으로
        return False


def _write_with_ffmpeg(path: str, frames: int, fps: float, codec: str) -> bool:
    """ffmpeg 바이너리로 작은 영상을 쓴다. 성공하면 True."""
    if not os.path.exists(_FFMPEG):
        return False
    try:
        result = subprocess.run(
            [_FFMPEG, "-y", "-loglevel", "error", "-f", "lavfi", "-i", f"testsrc=size=32x32:rate={fps}",
             "-frames:v", str(frames), "-c:v", codec, path],
            capture_output=True, timeout=120)
        return result.returncode == 0 and os.path.exists(path) and os.path.getsize(path) > 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def make_video(path: str, seconds: float, fps: float = 10.0) -> str:
    """길이 seconds(= 프레임 수 / fps)의 WebM 또는 MP4를 만든다. 만들 수 없으면 건너뛴다."""
    frames = max(1, int(round(seconds * fps)))
    webm = path.endswith(".webm")
    if _write_with_cv2(path, frames, fps, "VP80" if webm else "mp4v"):
        return path
    if _write_with_ffmpeg(path, frames, fps, "libvpx" if webm else "mpeg4"):
        return path
    pytest.skip("테스트용 WebM/MP4를 만들 수 없음 (cv2 VP80/mp4v·ffmpeg 모두 불가)")
    return path


def _ebml_header(doctype: bytes) -> bytes:
    """최소 EBML 헤더 (EBMLVersion + DocType)."""
    body = b"\x42\x86\x81\x01" + b"\x42\x82" + bytes([0x80 | len(doctype)]) + doctype
    return b"\x1a\x45\xdf\xa3" + bytes([0x80 | len(body)]) + body


class FakeService:
    """transcribe_video 인자를 기록하는 대체 인식 서비스."""

    def __init__(self, results=None, error: Exception | None = None):
        self.calls: list[dict] = []
        self.results = results if results is not None else [
            {"start_ms": 300, "end_ms": 1700, "text": "안녕하세요", "confidence": 0.8, "raw_score": -2.5}]
        self.error = error
        self.seen_file: dict = {}

    def transcribe_video(self, video_path, work_dir, on_progress=None, should_abort=None, **kwargs):
        self.calls.append({"video": video_path, "work_dir": work_dir, **kwargs})
        self.seen_file = {"exists": os.path.exists(video_path), "name": os.path.basename(video_path),
                          "head": open(video_path, "rb").read(4)}
        os.makedirs(work_dir, exist_ok=True)
        with open(os.path.join(work_dir, "seg_000.npz"), "wb") as handle:
            handle.write(b"x")
        if on_progress:
            on_progress(50.0)
        if self.error is not None:
            raise self.error
        return [dict(r) for r in self.results]

    def model_info(self):
        return {"backend": "olkavs", "decoder": "hybrid", "checkpoint_sha256": "f41d", "upstream_commit": "abc",
                "roi_profile": "p1"}


@pytest.fixture()
def env(tmp_path, monkeypatch):
    """업로드·입술 임시 폴더 격리, olkavs 백엔드, 대체 서비스."""
    upload_dir = str(tmp_path / "uploads")
    roi_dir = str(tmp_path / "roi")
    os.makedirs(upload_dir, exist_ok=True)
    monkeypatch.setattr(infer_module, "TMP_UPLOAD_DIR", upload_dir)
    monkeypatch.setattr(infer_module, "TMP_ROI_DIR", roi_dir)
    monkeypatch.setattr(infer_module, "MODEL_BACKEND", "olkavs")
    service = FakeService()
    monkeypatch.setattr(olkavs_service, "get_service", lambda: service)
    return {"upload_dir": upload_dir, "roi_dir": roi_dir, "service": service, "tmp": tmp_path}


def _post(content: bytes, filename: str = "recording.webm", mime: str = "video/webm"):
    return client.post("/api/record", files={"file": (filename, content, mime)})


def _assert_fixed_400(resp, expected: str, tmp_path) -> None:
    """400 + 고정 문구, 경로·예외 원문 없음."""
    assert resp.status_code == 400, resp.text
    detail = resp.json()["detail"]
    assert detail == expected
    assert detail in _FIXED_400
    for bad in (str(tmp_path), ":\\", ":/", "\\", "Error", "Traceback", ".webm", ".mp4", "uploads"):
        assert bad not in detail


# ---------------------------------------------------------------------------
# 접수 성공 → record 모드 Job
# ---------------------------------------------------------------------------

def test_record_webm_starts_record_mode_job_and_cleans_up(env) -> None:
    """WebM 녹화: 200 {job_id, session_id, processing, duration_sec}, record 모드 인자, 결과·모델 정보, 임시 파일 0."""
    video = make_video(str(env["tmp"] / "clip.webm"), 2.0)
    with open(video, "rb") as handle:
        resp = _post(handle.read(), "../../etc/evil.txt", "text/plain")  # 클라이언트 파일명·MIME은 무시
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert set(body) == {"job_id", "session_id", "status", "duration_sec"}
    assert body["status"] == "processing" and "results" not in body
    assert body["duration_sec"] == pytest.approx(2.0, abs=0.05)
    session_id = body["session_id"]
    assert infer_module.is_valid_session_id(session_id)
    call = env["service"].calls[0]
    assert call["utterance"] is True
    assert call["video_seconds"] == pytest.approx(2.0, abs=0.05)
    assert env["service"].seen_file == {"exists": True, "name": f"{session_id}_record.webm",
                                        "head": b"\x1a\x45\xdf\xa3"}
    assert os.path.dirname(call["video"]) == env["upload_dir"]
    assert call["work_dir"] == os.path.join(env["roi_dir"], session_id)
    # Job 뒤에는 녹화 파일과 입술 임시 폴더가 남지 않는다.
    assert os.listdir(env["upload_dir"]) == []
    assert not os.path.exists(call["work_dir"])
    result = client.get(f"/api/result/{session_id}").json()
    try:
        assert result["status"] == "done", result
        assert result["results"] == [{"start_ms": 300, "end_ms": 1700, "text": "안녕하세요", "confidence": 0.8,
                                      "raw_score": -2.5, "candidate": None}]
        assert result["model"] == env["service"].model_info()
        assert session_id not in infer_module._running_sessions
    finally:
        jm.clear_job(session_id)


def test_record_mp4_is_accepted_and_saved_as_mp4(env) -> None:
    """Safari용 MP4(ftyp)도 접수하고 서버가 정한 .mp4 이름으로 저장한다."""
    video = make_video(str(env["tmp"] / "clip.mp4"), 1.5)
    with open(video, "rb") as handle:
        resp = _post(handle.read(), "recording.mp4", "video/mp4")
    assert resp.status_code == 200, resp.text
    session_id = resp.json()["session_id"]
    try:
        assert env["service"].seen_file["name"] == f"{session_id}_record.mp4"
        assert resp.json()["duration_sec"] == pytest.approx(1.5, abs=0.05)
        assert os.listdir(env["upload_dir"]) == []
    finally:
        jm.clear_job(session_id)


def test_record_result_markers_and_candidate_pass_through(env) -> None:
    """서비스의 '[발화 없음]'·'[인식 불가]'+candidate 결과가 그대로 응답에 실린다."""
    env["service"].results = [
        {"start_ms": 0, "end_ms": 800, "text": "[발화 없음]", "confidence": None, "raw_score": None},
        {"start_ms": 800, "end_ms": 2000, "text": "[인식 불가]", "confidence": 0.1, "raw_score": -30.0,
         "candidate": "아무 말"},
    ]
    video = make_video(str(env["tmp"] / "clip.webm"), 2.0)
    with open(video, "rb") as handle:
        session_id = _post(handle.read()).json()["session_id"]
    try:
        result = client.get(f"/api/result/{session_id}").json()
        assert [r["text"] for r in result["results"]] == ["[발화 없음]", "[인식 불가]"]
        assert result["results"][0]["candidate"] is None and result["results"][1]["candidate"] == "아무 말"
        assert result["results"][1]["confidence"] == 0.1
    finally:
        jm.clear_job(session_id)


def test_record_job_failure_uses_fixed_message_and_deletes_file(env) -> None:
    """인식 실패도 고정 문구로 끝나고 녹화 파일·입술 폴더를 지운다."""
    env["service"].error = olkavs_service.WorkerError(olkavs_service.MSG_CRASHED, "C:/secret/log tail")
    video = make_video(str(env["tmp"] / "clip.webm"), 1.0)
    with open(video, "rb") as handle:
        session_id = _post(handle.read()).json()["session_id"]
    try:
        result = client.get(f"/api/result/{session_id}").json()
        assert result["status"] == "error" and result["error_message"] == olkavs_service.MSG_CRASHED
        assert result["model"] is None and result["results"] is None
        assert os.listdir(env["upload_dir"]) == []
        assert not os.path.exists(env["service"].calls[0]["work_dir"])
    finally:
        jm.clear_job(session_id)


def test_record_just_under_limit_is_accepted(env) -> None:
    """29.5초 녹화는 접수한다 (상한 30초 미만)."""
    video = make_video(str(env["tmp"] / "long_ok.webm"), 29.5)
    with open(video, "rb") as handle:
        resp = _post(handle.read())
    assert resp.status_code == 200, resp.text
    try:
        assert resp.json()["duration_sec"] == pytest.approx(29.5, abs=0.05)
    finally:
        jm.clear_job(resp.json()["session_id"])


# ---------------------------------------------------------------------------
# 접수 거부 (400 고정 문구, 파일 삭제)
# ---------------------------------------------------------------------------

def test_record_30_seconds_or_longer_rejected(env) -> None:
    """서버가 잰 길이가 30초 이상이면 거부하고 파일을 지운다."""
    for seconds in (30.0, 45.0):
        video = make_video(str(env["tmp"] / f"too_long_{seconds:.0f}.webm"), seconds)
        with open(video, "rb") as handle:
            resp = _post(handle.read())
        _assert_fixed_400(resp, record_module.MSG_TOO_LONG.format(sec=config.RECORD_MAX_SECONDS), env["tmp"])
        assert os.listdir(env["upload_dir"]) == []
    assert env["service"].calls == []


def test_record_shorter_than_minimum_rejected(env) -> None:
    """0.5초 미만은 거부한다."""
    video = make_video(str(env["tmp"] / "short.webm"), 0.3)
    with open(video, "rb") as handle:
        resp = _post(handle.read())
    _assert_fixed_400(resp, record_module.MSG_TOO_SHORT.format(sec=config.RECORD_MIN_SECONDS), env["tmp"])
    assert os.listdir(env["upload_dir"]) == [] and env["service"].calls == []


@pytest.mark.parametrize("content", [
    b"RIFF\x00\x00\x00\x00AVI LIST" + b"\x00" * 64,           # AVI
    b"ID3\x04\x00\x00\x00\x00\x00\x00" + b"\x00" * 64,          # MP3
    b"just some text, definitely not a video file at all...",  # 텍스트
    _ebml_header(b"foo") + b"\x00" * 64,                        # EBML이지만 webm/matroska 아님
    b"\x1a\x45\xdf\xa3\xff\xff" + b"\x00" * 64,                 # EBML 헤더 깨짐
])
def test_record_wrong_signature_rejected(env, content) -> None:
    """내용 시그니처가 WebM/MP4가 아니면 확장자·MIME과 관계없이 거부한다."""
    resp = _post(content, "recording.webm", "video/webm")
    _assert_fixed_400(resp, record_module.MSG_UNSUPPORTED, env["tmp"])
    assert os.listdir(env["upload_dir"]) == []


def test_record_garbage_webm_rejected_as_unreadable(env) -> None:
    """시그니처만 맞고 내용이 깨진 WebM은 '읽을 수 없음' 고정 문구로 거부한다."""
    content = _ebml_header(b"webm") + os.urandom(4096)
    resp = _post(content)
    _assert_fixed_400(resp, record_module.MSG_UNREADABLE, env["tmp"])
    assert os.listdir(env["upload_dir"]) == []
    truncated = make_video(str(env["tmp"] / "clip.webm"), 2.0)
    with open(truncated, "rb") as handle:
        head = handle.read()[:200]  # 헤더만 남기고 잘린 녹화
    resp = _post(head)
    _assert_fixed_400(resp, record_module.MSG_UNREADABLE, env["tmp"])
    assert os.listdir(env["upload_dir"]) == []


def test_record_empty_or_tiny_file_rejected(env) -> None:
    """빈 파일·너무 작은 파일은 거부한다."""
    for content in (b"", b"\x1a\x45\xdf\xa3"):
        _assert_fixed_400(_post(content), record_module.MSG_EMPTY, env["tmp"])
    assert os.listdir(env["upload_dir"]) == []


_ORIGIN = "http://localhost:5173"
_BOUNDARY = "lipnetBoundary7d1f"


def _multipart_prefix(filename: str = "r.webm", mime: str = "video/webm") -> bytes:
    """멀티파트 'file' 부분의 머리 (본문은 뒤에 이어 붙인다)."""
    return (f"--{_BOUNDARY}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\n"
            f"Content-Type: {mime}\r\n\r\n").encode("ascii")


def asgi_post(path: str, first: bytes, total_bytes: int, content_length: int | None,
              chunk: int = 256 * 1024) -> dict:
    """앱을 ASGI로 직접 불러 POST 본문을 조각으로 흘려 보낸다. 서버가 receive()로 실제 가져간 바이트를 센다.

    Args:
        path: 요청 경로.
        first: 첫 조각(멀티파트 머리 + 파일 앞부분).
        total_bytes: 보낼 수 있는 전체 바이트(나머지는 0으로 채움). 서버가 덜 가져가면 거기서 멈춘다.
        content_length: Content-Length 헤더 값 (None이면 헤더 없이 chunked처럼 보낸다).
        chunk: 조각 크기.

    Returns:
        {status, headers(dict), json, pulled(서버가 가져간 바이트 수)}.
    """
    import asyncio
    import json as _json

    headers = [(b"host", b"testserver"), (b"origin", _ORIGIN.encode()),
               (b"content-type", f"multipart/form-data; boundary={_BOUNDARY}".encode())]
    if content_length is not None:
        headers.append((b"content-length", str(content_length).encode()))
    scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "POST",
             "scheme": "http", "path": path, "raw_path": path.encode(), "root_path": "", "query_string": b"",
             "headers": headers, "client": ("127.0.0.1", 50000), "server": ("testserver", 80)}
    state = {"sent": 0, "first": True}
    messages: list[dict] = []

    async def receive() -> dict:
        if state["sent"] >= total_bytes:
            return {"type": "http.disconnect"}
        if state["first"]:
            state["first"] = False
            body = first[:total_bytes]
        else:
            body = b"\x00" * min(chunk, total_bytes - state["sent"])
        state["sent"] += len(body)
        return {"type": "http.request", "body": body, "more_body": state["sent"] < total_bytes}

    async def send(message: dict) -> None:
        messages.append(message)

    asyncio.run(app(scope, receive, send))
    start = next(m for m in messages if m["type"] == "http.response.start")
    body = b"".join(m.get("body", b"") for m in messages if m["type"] == "http.response.body")
    return {"status": start["status"], "headers": {k.decode().lower(): v.decode() for k, v in start["headers"]},
            "json": _json.loads(body.decode("utf-8")), "pulled": state["sent"]}


def test_record_oversize_content_length_rejected_before_receiving_body(env, monkeypatch) -> None:
    """Content-Length가 RECORD_MAX_SIZE_MB+여유분을 넘으면 본문을 한 바이트도 받지 않고 413(+CORS 헤더)."""
    monkeypatch.setattr(config, "RECORD_MAX_SIZE_MB", 1)
    saved: list = []
    monkeypatch.setattr(record_module, "_save_capped", lambda *a, **k: saved.append(a))
    declared = 1024 * 1024 + config.REQUEST_BODY_OVERHEAD_BYTES + 1
    out = asgi_post("/api/record", _multipart_prefix() + _ebml_header(b"webm"), 50 * 1024 * 1024, declared)
    assert out["status"] == 413
    assert out["json"] == {"detail": record_module.MSG_TOO_LARGE.format(mb=1)}
    assert out["pulled"] == 0  # 서버가 본문을 가져가지 않았다
    assert out["headers"].get("access-control-allow-origin") == _ORIGIN
    assert saved == [] and os.listdir(env["upload_dir"]) == [] and env["service"].calls == []


@pytest.mark.parametrize("declared", [None, 1024])  # 길이 없음(chunked) / 실제보다 작게 적은 길이
def test_record_oversize_streamed_body_stops_at_limit(env, monkeypatch, declared) -> None:
    """길이를 모르거나 속인 본문은 받은 바이트를 세어 상한을 넘는 즉시 끊고 413 — 50MB를 끝까지 받지 않는다."""
    monkeypatch.setattr(config, "RECORD_MAX_SIZE_MB", 1)
    limit = 1024 * 1024 + config.REQUEST_BODY_OVERHEAD_BYTES
    chunk = 256 * 1024
    out = asgi_post("/api/record", _multipart_prefix() + _ebml_header(b"webm"), 50 * 1024 * 1024, declared,
                    chunk=chunk)
    assert out["status"] == 413, out
    assert out["json"] == {"detail": record_module.MSG_TOO_LARGE.format(mb=1)}
    assert limit < out["pulled"] <= limit + chunk
    assert out["headers"].get("access-control-allow-origin") == _ORIGIN
    assert os.listdir(env["upload_dir"]) == [] and env["service"].calls == []


def test_record_over_file_cap_within_overhead_rejected_and_copy_capped(env, monkeypatch) -> None:
    """본문은 여유분 안이지만 파일이 RECORD_MAX_SIZE_MB를 넘으면 400. 사본은 상한+1바이트까지만 옮기고 지운다."""
    monkeypatch.setattr(config, "RECORD_MAX_SIZE_MB", 1)
    from api import limits as limits_module

    copied: list[int] = []
    original = limits_module._copy_capped

    def spy(source, head, path, max_bytes):
        total = original(source, head, path, max_bytes)
        copied.append(total)
        return total

    monkeypatch.setattr(limits_module, "_copy_capped", spy)
    content = _ebml_header(b"webm") + b"\x00" * (1024 * 1024 + 512 * 1024)
    resp = client.post("/api/record", files={"file": ("recording.webm", content, "video/webm")},
                       headers={"Origin": _ORIGIN})
    assert resp.status_code == 400
    assert resp.json()["detail"] == record_module.MSG_TOO_LARGE.format(mb=1)
    assert copied == [1024 * 1024 + 1]
    assert os.listdir(env["upload_dir"]) == []


def test_record_frame_count_guard(env, monkeypatch) -> None:
    """프레임 수 상한을 넘는 파일은 끝까지 읽지 않고 거부한다."""
    monkeypatch.setattr(config, "RECORD_MAX_FRAMES", 10)
    video = make_video(str(env["tmp"] / "many.webm"), 2.0)
    with open(video, "rb") as handle:
        resp = _post(handle.read())
    _assert_fixed_400(resp, record_module.MSG_UNREADABLE, env["tmp"])
    assert os.listdir(env["upload_dir"]) == []


def test_record_unexpected_measure_error_hides_details(env, monkeypatch) -> None:
    """측정 중 예상 밖 예외가 나도 경로·원문 없이 고정 문구로 거부한다."""
    def boom(path, *args, **kwargs):
        raise RuntimeError(f"decoder exploded at {path}")

    monkeypatch.setattr(record_module, "measure_recording_seconds", boom)
    video = make_video(str(env["tmp"] / "clip.webm"), 1.0)
    with open(video, "rb") as handle:
        resp = _post(handle.read())
    _assert_fixed_400(resp, record_module.MSG_UNREADABLE, env["tmp"])
    assert os.listdir(env["upload_dir"]) == []


def test_record_requires_olkavs_backend(env, monkeypatch) -> None:
    """compact 백엔드에서는 503이고 아무것도 저장하지 않는다."""
    monkeypatch.setattr(infer_module, "MODEL_BACKEND", "compact")
    video = make_video(str(env["tmp"] / "clip.webm"), 1.0)
    with open(video, "rb") as handle:
        resp = _post(handle.read())
    assert resp.status_code == 503
    assert resp.json()["detail"] == record_module.MSG_BACKEND_UNSUPPORTED
    assert os.listdir(env["upload_dir"]) == [] and env["service"].calls == []


def test_record_mode_job_on_compact_fails_with_fixed_message(env, monkeypatch) -> None:
    """(방어) record 모드 Job이 compact에서 실행되면 고정 문구로 실패하고 파일을 지운다."""
    monkeypatch.setattr(infer_module, "MODEL_BACKEND", "compact")
    session_id = "2b9f7a3e-1c2d-4e5f-8a9b-0c1d2e3f4a5b"
    path = os.path.join(env["upload_dir"], f"{session_id}_record.webm")
    with open(path, "wb") as handle:
        handle.write(b"x")
    jm.create_job(session_id)
    try:
        infer_module._run_infer_job(session_id, path, mode="record", video_seconds=1.0)
        record = jm.get_job(session_id)
        assert record.status == "error" and record.error_message == infer_module.MSG_RECORD_BACKEND
        assert not os.path.exists(path)
    finally:
        jm.clear_job(session_id)


def test_start_background_job_rejects_running_session(env) -> None:
    """공용 Job 시작 함수: 같은 세션이 처리 중이면 409, 알 수 없는 모드는 ValueError."""
    from fastapi import BackgroundTasks, HTTPException

    session_id = "3b9f7a3e-1c2d-4e5f-8a9b-0c1d2e3f4a5b"
    jm.create_job(session_id)
    jm.update_progress(session_id, 5.0)
    try:
        with pytest.raises(HTTPException) as info:
            infer_module.start_background_job(session_id, "x", BackgroundTasks(), mode="record")
        assert info.value.status_code == 409
        with pytest.raises(ValueError):
            infer_module.start_background_job(session_id, "x", BackgroundTasks(), mode="stream")
        assert session_id not in infer_module._running_sessions
    finally:
        jm.clear_job(session_id)


# ---------------------------------------------------------------------------
# 형식 판정·길이 측정 단위 검사
# ---------------------------------------------------------------------------

def test_detect_recording_container() -> None:
    """EBML + DocType webm/matroska → webm, ftyp → mp4, 그 밖은 None."""
    assert record_module.detect_recording_container(_ebml_header(b"webm")) == "webm"
    assert record_module.detect_recording_container(_ebml_header(b"matroska")) == "webm"
    assert record_module.detect_recording_container(_ebml_header(b"foo")) is None
    assert record_module.detect_recording_container(b"\x1a\x45\xdf\xa3") is None
    assert record_module.detect_recording_container(b"\x00\x00\x00\x1cftypisom\x00\x00\x02\x00") == "mp4"
    assert record_module.detect_recording_container(b"\x00\x00\x00\x1cftypqt  \x00\x00\x02\x00") == "mp4"
    assert record_module.detect_recording_container(b"RIFF\x00\x00\x00\x00AVI ") is None
    assert record_module.detect_recording_container(b"") is None


def test_measure_recording_seconds_uses_real_pts_and_stops_early(tmp_path) -> None:
    """실제 PTS로 길이를 재고, 상한을 1초 넘게 지나면 끝까지 읽지 않는다."""
    path = make_video(str(tmp_path / "m.webm"), 3.0)
    assert record_module.measure_recording_seconds(path) == pytest.approx(3.0, abs=0.05)
    long_path = make_video(str(tmp_path / "long.webm"), 60.0, fps=5.0)
    measured = record_module.measure_recording_seconds(long_path, max_seconds=10.0)
    assert 11.0 <= measured < 13.0  # 10초 + 1초 넘은 지점에서 멈춤 (60초 전체를 읽지 않음)
    with pytest.raises(record_module.RecordingRejected) as info:
        record_module.measure_recording_seconds(str(tmp_path / "missing.webm"))
    assert info.value.user_message == record_module.MSG_UNREADABLE
    with pytest.raises(record_module.RecordingRejected):
        record_module.measure_recording_seconds(path, time_budget=-1.0)


def _make_mediarecorder_like_webm(path: str, frames: int = 45, gap_ms: int = 67) -> str:
    """MediaRecorder처럼 길이 메타데이터가 없는 WebM을 만든다 (ffmpeg -live 1, 1000fps 시간 기준).

    cv2가 fps=1000·프레임 수 음수(또는 0)를 돌려줘야 한다. 그렇게 만들 수 없으면 건너뛴다.
    프레임 간격 gap_ms인 frames개 → 실제 길이 frames*gap_ms ms.
    """
    if not os.path.exists(_FFMPEG):
        pytest.skip("ffmpeg 바이너리 없음 — MediaRecorder 형태 WebM을 만들 수 없음")
    try:
        result = subprocess.run(
            [_FFMPEG, "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=size=32x32:rate=1000",
             "-vf", f"select=not(mod(n\\,{gap_ms}))", "-fps_mode", "passthrough", "-frames:v", str(frames),
             "-c:v", "libvpx", "-f", "webm", "-live", "1", path],
            capture_output=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired):
        pytest.skip("ffmpeg 실행 실패")
    if result.returncode != 0 or not os.path.exists(path):
        pytest.skip("ffmpeg로 libvpx WebM을 만들 수 없음")
    cap = cv2.VideoCapture(path)
    try:
        opened = cap.isOpened()
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    if not opened:
        pytest.skip("이 cv2가 VP8 WebM을 열지 못함")
    if count > 0 and 1.0 <= fps <= 240.0:
        pytest.skip(f"길이 메타데이터가 남은 WebM이 만들어짐 (fps={fps}, count={count})")
    return path


def test_measure_recording_seconds_on_generated_webm_without_duration_metadata(tmp_path) -> None:
    """MediaRecorder 형태(fps 1000·프레임 수 음수) WebM: 메타데이터 없이 실제 PTS로 길이를 잰다."""
    path = _make_mediarecorder_like_webm(str(tmp_path / "mr_like.webm"), frames=45, gap_ms=67)
    assert infer_module._probe_video_seconds(path) is None  # 메타데이터로는 길이를 알 수 없다
    assert record_module.measure_recording_seconds(path) == pytest.approx(45 * 0.067, abs=0.02)


def test_record_route_accepts_webm_without_duration_metadata(env, tmp_path) -> None:
    """MediaRecorder 형태 WebM 녹화도 접수되고, 서버가 잰 길이가 Job의 video_seconds로 간다."""
    path = _make_mediarecorder_like_webm(str(tmp_path / "mr_like.webm"), frames=45, gap_ms=67)
    with open(path, "rb") as handle:
        resp = _post(handle.read())
    assert resp.status_code == 200, resp.text
    assert resp.json()["duration_sec"] == pytest.approx(3.015, abs=0.02)
    call = env["service"].calls[-1]
    assert call["utterance"] is True and call["video_seconds"] == pytest.approx(3.015, abs=0.02)
    assert os.listdir(env["upload_dir"]) == []


def test_measure_recording_seconds_on_browser_webm_without_duration_metadata() -> None:
    """(선택) MediaRecorder WebM 표본: 길이 메타데이터 없이 실제 PTS로 약 8초를 잰다."""
    sample = os.environ.get("LIPNET_RECORD_SAMPLE_WEBM")
    if not sample or not os.path.exists(sample):
        pytest.skip("LIPNET_RECORD_SAMPLE_WEBM 미지정")
    expected = float(os.environ.get("LIPNET_RECORD_SAMPLE_SECONDS", "8"))
    assert record_module.measure_recording_seconds(sample) == pytest.approx(expected, abs=0.2)
