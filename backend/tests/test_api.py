"""API 통합 테스트 (FastAPI TestClient).

설계문서 5절 — 업로드 검증, 추론 즉시 결과 반환 금지, 결과 폴링, CORS
화이트리스트, 평가 test_set_id 검증을 확인한다.
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

client = TestClient(app)


def _make_mp4(path: str, fps: float = 25.0, num_frames: int = 50) -> None:
    """테스트용 소형 mp4를 생성한다 (2초, 64×64)."""
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


def test_upload_rejects_bad_extension() -> None:
    """미지원 확장자는 400."""
    resp = client.post(
        "/api/upload",
        files={"file": ("note.txt", b"hello", "text/plain")},
    )
    assert resp.status_code == 400


def test_upload_rejects_invalid_mime(tmp_upload_dir) -> None:
    """확장자는 mp4지만 내용이 영상이 아니면 MIME 불일치로 400."""
    resp = client.post(
        "/api/upload",
        files={"file": ("fake.mp4", b"this is not a video", "video/mp4")},
    )
    assert resp.status_code == 400


def test_upload_success_returns_session(tmp_upload_dir, tmp_path) -> None:
    """정상 mp4 업로드 시 session_id와 duration_sec를 반환한다."""
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
    assert body["duration_sec"] == pytest.approx(2.0, abs=0.2)


def test_infer_does_not_return_results_immediately(tmp_upload_dir, tmp_path) -> None:
    """POST /api/infer는 results 없이 {job_id, session_id, status:'processing'}만 반환."""
    mp4 = str(tmp_path / "v.mp4")
    _make_mp4(mp4)
    with open(mp4, "rb") as f:
        up = client.post(
            "/api/upload", files={"file": ("v.mp4", f.read(), "video/mp4")}
        )
    session_id = up.json()["session_id"]

    resp = client.post("/api/infer", json={"session_id": session_id})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "processing"
    assert body["session_id"] == session_id
    assert "job_id" in body
    assert "results" not in body  # 절대 즉시 반환 금지


def test_infer_missing_session_404(tmp_upload_dir) -> None:
    """업로드되지 않은 세션 추론 요청은 404."""
    resp = client.post("/api/infer", json={"session_id": "ghost-session"})
    assert resp.status_code == 404


def test_result_polling_flow(tmp_upload_dir, tmp_path) -> None:
    """업로드 → 추론 → 결과 폴링까지 흐름 검증 (TestClient는 백그라운드 동기 실행)."""
    mp4 = str(tmp_path / "v.mp4")
    _make_mp4(mp4)
    with open(mp4, "rb") as f:
        up = client.post(
            "/api/upload", files={"file": ("v.mp4", f.read(), "video/mp4")}
        )
    session_id = up.json()["session_id"]
    client.post("/api/infer", json={"session_id": session_id})

    resp = client.get(f"/api/result/{session_id}")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["session_id"] == session_id
    # 백그라운드 추론 완료 → done, 합성 영상(얼굴 없음)은 전부 검출 실패 구간
    assert body["status"] == "done"
    assert body["results"] is not None
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
    """화이트리스트 Origin만 허용하고 미등록 Origin은 허용 헤더가 없다."""
    allowed = CORS_ORIGINS[0]
    r_ok = client.get("/health", headers={"Origin": allowed})
    assert r_ok.headers.get("access-control-allow-origin") == allowed

    r_bad = client.get("/health", headers={"Origin": "http://evil.example.com"})
    assert r_bad.headers.get("access-control-allow-origin") != "http://evil.example.com"
    assert r_bad.headers.get("access-control-allow-origin") != "*"
