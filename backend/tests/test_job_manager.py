"""job_manager 단위 테스트.

설계문서 F-MVP-03 — Job 상태 전이(pending→processing→done/error/timeout),
진행률 갱신, 타임아웃 판정을 검증한다.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta

# backend 패키지 루트를 import 경로에 추가
_BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

from services import job_manager as jm  # noqa: E402


def _sid() -> str:
    """테스트마다 고유한 session_id를 생성한다."""
    return "sess-" + os.urandom(4).hex()


def test_create_job_returns_uuid_and_pending() -> None:
    """create_job이 job_id를 반환하고 초기 상태가 pending인지 검증한다."""
    sid = _sid()
    job_id = jm.create_job(sid)
    assert isinstance(job_id, str) and len(job_id) > 0

    rec = jm.get_job(sid)
    assert rec is not None
    assert rec.job_id == job_id
    assert rec.status == jm.STATUS_PENDING
    assert rec.progress == 0.0
    assert rec.started_at is not None


def test_get_job_missing_returns_none() -> None:
    """미존재 세션은 None을 반환한다."""
    assert jm.get_job("no-such-session") is None


def test_update_progress_transitions_to_processing() -> None:
    """update_progress가 진행률을 갱신하고 processing으로 전이하는지 검증한다."""
    sid = _sid()
    jm.create_job(sid)
    jm.update_progress(sid, 42.5)

    rec = jm.get_job(sid)
    assert rec.status == jm.STATUS_PROCESSING
    assert rec.progress == 42.5


def test_update_progress_clamped() -> None:
    """진행률이 0~100 범위로 클램프되는지 검증한다."""
    sid = _sid()
    jm.create_job(sid)
    jm.update_progress(sid, 250.0)
    assert jm.get_job(sid).progress == 100.0
    jm.update_progress(sid, -10.0)
    assert jm.get_job(sid).progress == 0.0


def test_complete_job_sets_done_and_results() -> None:
    """complete_job이 done/진행률 100/results를 설정하는지 검증한다."""
    sid = _sid()
    jm.create_job(sid)
    results = [{"start_ms": 0, "end_ms": 3000, "text": "안녕", "confidence": 0.9}]
    jm.complete_job(sid, results)

    rec = jm.get_job(sid)
    assert rec.status == jm.STATUS_DONE
    assert rec.progress == 100.0
    assert rec.results == results
    assert rec.finished_at is not None


def test_fail_job_sets_error() -> None:
    """fail_job이 error 상태와 메시지를 설정하는지 검증한다."""
    sid = _sid()
    jm.create_job(sid)
    jm.fail_job(sid, "전처리 실패")

    rec = jm.get_job(sid)
    assert rec.status == jm.STATUS_ERROR
    assert rec.error_message == "전처리 실패"
    assert rec.finished_at is not None


def test_timeout_job_sets_timeout() -> None:
    """timeout_job이 timeout 상태를 설정하는지 검증한다."""
    sid = _sid()
    jm.create_job(sid)
    jm.timeout_job(sid)
    assert jm.get_job(sid).status == jm.STATUS_TIMEOUT


def test_is_timeout_true_when_elapsed() -> None:
    """시작 후 timeout_sec 초과 시 is_timeout이 True인지 검증한다."""
    sid = _sid()
    jm.create_job(sid)
    # 시작 시각을 6분 전으로 조작
    rec = jm.get_job(sid)
    rec.started_at = datetime.now() - timedelta(seconds=360)
    assert jm.is_timeout(sid, timeout_sec=300) is True


def test_is_timeout_false_when_recent_or_terminal() -> None:
    """방금 시작했거나 종료된 Job은 is_timeout이 False인지 검증한다."""
    sid = _sid()
    jm.create_job(sid)
    assert jm.is_timeout(sid, timeout_sec=300) is False  # 방금 시작

    # 종료 상태면 경과와 무관하게 False
    rec = jm.get_job(sid)
    rec.started_at = datetime.now() - timedelta(seconds=360)
    jm.complete_job(sid, [])
    assert jm.is_timeout(sid, timeout_sec=300) is False


def test_is_timeout_missing_session() -> None:
    """미존재 세션은 False를 반환한다."""
    assert jm.is_timeout("no-such-session") is False
