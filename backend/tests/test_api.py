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
    """업로드/추론 모듈의 임시 디렉토리를 테스트 격리 폴더로 교체한다."""
    d = str(tmp_path / "uploads")
    os.makedirs(d, exist_ok=True)
    monkeypatch.setattr(upload_module, "TMP_UPLOAD_DIR", d)
    monkeypatch.setattr(infer_module, "TMP_UPLOAD_DIR", d)
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
