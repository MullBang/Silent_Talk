"""실제 OLKAVS 워커 두 개(.venv-olkavs-roi, .venv-olkavs)를 띄우는 통합 검사 — 선택 실행.

얼굴이 나오는 영상이 필요하므로 저장소에 넣지 않고 환경변수로 지정한다.
  LIPNET_OLKAVS_TEST_VIDEO=<얼굴 정면 영상 경로>  [LIPNET_OLKAVS_TEST_EXPECT=<기대 문장 일부>]
지정하지 않았거나 가상환경이 없으면 건너뛴다. 모델 로드 때문에 수십 초 걸린다.

업로드 모드(창 분할)와 녹화 모드(utterance=True: 앞뒤 무발화 제거 → NPZ 다시 저장 → 인식)를 모두 실제 워커로 돈다.
입 움직임이 없는 창은 '[발화 없음]'(confidence None)일 수 있으므로 점수는 None이 아닐 때만 범위를 검사한다.
"""
from __future__ import annotations

import os
import sys

import pytest

_BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

import config  # noqa: E402
from services import olkavs_service as svc  # noqa: E402

VIDEO = os.environ.get("LIPNET_OLKAVS_TEST_VIDEO")
pytestmark = pytest.mark.skipif(
    not VIDEO or not os.path.exists(VIDEO or "") or not os.path.exists(config.OLKAVS_ROI_PYTHON)
    or not os.path.exists(config.OLKAVS_INFER_PYTHON),
    reason="LIPNET_OLKAVS_TEST_VIDEO 또는 OLKAVS 가상환경이 없음",
)

_MARKERS = (svc.FAIL_TEXT, svc.UNREC_TEXT, svc.NO_SPEECH_TEXT)


def _check_results(results: list[dict], progress: list[float]) -> list[dict]:
    """공통 결과 검사: 시간 순서·구간·점수 범위(None 제외)·진행률 단조. 표시 문구가 아닌 인식 결과를 돌려준다."""
    assert results and all(r["end_ms"] > r["start_ms"] for r in results)
    assert results == sorted(results, key=lambda r: r["start_ms"])
    for r in results:
        if r.get("confidence") is not None:
            assert 0.0 <= r["confidence"] <= 1.0, r
        if r["text"] == svc.NO_SPEECH_TEXT:
            assert r["confidence"] is None and r["raw_score"] is None
    assert progress and progress[0] == 0.0 and progress == sorted(progress) and progress[-1] == 100.0
    return [r for r in results if r["text"] not in _MARKERS]


def _metadata_seconds(path: str) -> float | None:
    """서비스가 쓰는 메타데이터 길이 어림 (서버 환경 cv2가 없으면 None)."""
    try:
        from api import infer as infer_module

        return infer_module._probe_video_seconds(path)
    except Exception:  # noqa: BLE001 - 길이 힌트는 선택 항목이다
        return None


@pytest.fixture(scope="module")
def service():
    """실제 워커 두 개를 띄운 서비스 (모듈 안에서 재사용, 끝나면 종료)."""
    instance = svc.OlkavsService()
    try:
        info = instance.start()
        assert info["roi"]["mediapipe"] == "0.10.21"
        assert info["infer"]["checkpoint_sha256"].startswith("f41d7423")
        yield instance
    finally:
        instance.shutdown()
    assert instance.status()["infer_worker"] == "stopped"


def test_real_workers_transcribe_video(service, tmp_path) -> None:
    """업로드 모드: 실제 워커로 영상 하나를 구간별 텍스트로 바꾸고, 워커를 재사용한다."""
    progress: list[float] = []
    results = service.transcribe_video(VIDEO, str(tmp_path / "roi"), on_progress=progress.append,
                                       video_seconds=_metadata_seconds(VIDEO))
    recognized = _check_results(results, progress)
    if not recognized and all(r["text"] in (svc.NO_SPEECH_TEXT, svc.UNREC_TEXT, svc.FAIL_TEXT) for r in results):
        pytest.skip("이 영상에서는 인식할 발화를 찾지 못함 (표시 문구만 나옴) — 인식 결과 검사는 건너뜀")
    assert recognized, results
    expect = os.environ.get("LIPNET_OLKAVS_TEST_EXPECT")
    if expect:
        assert expect in " ".join(r["text"] for r in recognized)
    pid_before = service.infer._proc.pid
    service.transcribe_video(VIDEO, str(tmp_path / "roi2"))
    assert service.infer._proc.pid == pid_before  # 상주 워커 재사용
    status = service.status()
    assert status["roi_worker"] == "ready" and status["infer_worker"] == "ready"


def test_real_workers_record_mode_utterance(service, tmp_path) -> None:
    """녹화 모드(utterance=True): 앞뒤 무발화를 잘라 NPZ를 다시 저장하고 인식한다. 길이 힌트도 보낸다."""
    progress: list[float] = []
    seconds = _metadata_seconds(VIDEO)
    if seconds is None:  # MediaRecorder WebM처럼 메타데이터 길이가 없으면 녹화 접수와 같은 실측 길이를 쓴다
        from api.record import RecordingRejected, measure_recording_seconds

        try:
            seconds = measure_recording_seconds(VIDEO)
        except RecordingRejected:  # 30초 이상 등 녹화로 접수되지 않는 영상이면 힌트 없이 돈다
            seconds = None
    results = service.transcribe_video(VIDEO, str(tmp_path / "roi_rec"), on_progress=progress.append,
                                       video_seconds=seconds, utterance=True)
    _check_results(results, progress)
    if seconds:  # 길이(메타데이터 또는 힌트)가 있으면 입술 추출 단계(0~40%) 진행률이 실제로 중간값을 낸다.
        assert any(0.0 < p < 40.0 for p in progress), progress
    if all(r["text"] in (svc.NO_SPEECH_TEXT, svc.FAIL_TEXT) for r in results):
        pytest.skip("이 영상에서는 발화 구간이나 얼굴을 찾지 못함 ('[발화 없음]'/'[검출 실패 구간]') — 인식 결과 검사는 건너뜀")
    # 발화 창은 실제로 인식 워커를 거친다: 인식 문장이 있거나, 낮은 확신으로 숨긴 '[인식 불가]'(candidate 포함).
    inferred = [r for r in results if r["text"] not in (svc.FAIL_TEXT, svc.NO_SPEECH_TEXT)]
    assert inferred, results
    assert all(r["text"] != svc.UNREC_TEXT or r.get("candidate") or r["confidence"] is None for r in inferred)
    if seconds:
        assert results[-1]["end_ms"] <= seconds * 1000.0 + 100.0
    assert service.status()["roi_worker"] == "ready"
