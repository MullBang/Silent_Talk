"""비동기 Job 상태 관리 (메모리 dict 기반).

설계문서 F-MVP-03 기준. 추론 Job의 상태를 인메모리로 관리한다
(session_id ↔ Job 매핑). 인식 결과는 개인정보 보호상 영구 저장하지 않고
서버 메모리에만 유지한다.

상태 전이: pending → processing → (done | error | timeout)
"""

from __future__ import annotations

import os
import sys
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime

# backend 루트를 import 경로에 부트스트랩 (uvicorn `backend.*` / pytest 양쪽 대응)
_BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

from config import JOB_TIMEOUT_SEC  # noqa: E402

# 상태 상수 (설계문서 recognition_session.status CHECK 제약과 일치)
STATUS_PENDING = "pending"
STATUS_PROCESSING = "processing"
STATUS_DONE = "done"
STATUS_ERROR = "error"
STATUS_TIMEOUT = "timeout"

# 종료(terminal) 상태 — 더 이상 타임아웃 판정 대상이 아님
_TERMINAL_STATES = frozenset({STATUS_DONE, STATUS_ERROR, STATUS_TIMEOUT})


@dataclass
class JobRecord:
    """단일 추론 Job의 상태 레코드.

    설계문서의 Job 상태 dict
    ({session_id, job_id, status, progress, error_message, started_at,
    finished_at, results})를 dataclass로 표현한다.

    Note:
        schemas.models.JobStatus(Enum, API 응답용)와 구분하기 위해 이 데이터
        구조는 JobRecord로 명명한다.
    """

    session_id: str
    job_id: str
    status: str = STATUS_PENDING
    progress: float = 0.0  # 0.0 ~ 100.0
    error_message: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    results: list | None = None  # [{start_ms, end_ms, text, confidence}]


# session_id -> JobRecord. 다중 스레드(백그라운드 추론) 접근을 Lock으로 보호한다.
_jobs: dict[str, JobRecord] = {}
_lock = threading.Lock()


def create_job(session_id: str) -> str:
    """새 Job을 생성하고 status='pending'으로 등록한다.

    동일 session_id의 기존 Job이 있으면 새 Job으로 대체한다.

    Args:
        session_id: 업로드 시 발급된 세션 식별자.

    Returns:
        발급된 job_id (UUID4 문자열).
    """
    job_id = str(uuid.uuid4())
    record = JobRecord(
        session_id=session_id,
        job_id=job_id,
        status=STATUS_PENDING,
        progress=0.0,
        started_at=datetime.now(),
    )
    with _lock:
        _jobs[session_id] = record
    return job_id


def get_job(session_id: str) -> JobRecord | None:
    """세션 식별자로 Job 레코드를 조회한다.

    Args:
        session_id: 세션 식별자.

    Returns:
        JobRecord 또는 미존재 시 None.
    """
    with _lock:
        return _jobs.get(session_id)


def update_progress(session_id: str, progress: float) -> None:
    """Job 진행률을 갱신한다 (status를 processing으로 전이).

    Args:
        session_id: 세션 식별자.
        progress: 진행률 (0.0 ~ 100.0, 범위 밖은 클램프).
    """
    with _lock:
        record = _jobs.get(session_id)
        if record is None:
            return
        record.progress = max(0.0, min(100.0, float(progress)))
        if record.status == STATUS_PENDING:
            record.status = STATUS_PROCESSING


def complete_job(session_id: str, results: list) -> None:
    """Job을 완료 처리한다 (status='done').

    Args:
        session_id: 세션 식별자.
        results: [{start_ms, end_ms, text, confidence}] 결과 리스트.
    """
    with _lock:
        record = _jobs.get(session_id)
        if record is None:
            return
        record.status = STATUS_DONE
        record.progress = 100.0
        record.results = results
        record.error_message = None
        record.finished_at = datetime.now()


def fail_job(session_id: str, error_message: str) -> None:
    """Job을 실패 처리한다 (status='error').

    Args:
        session_id: 세션 식별자.
        error_message: 실패 사유.
    """
    with _lock:
        record = _jobs.get(session_id)
        if record is None:
            return
        record.status = STATUS_ERROR
        record.error_message = error_message
        record.finished_at = datetime.now()


def timeout_job(session_id: str) -> None:
    """Job을 타임아웃 처리한다 (status='timeout').

    Args:
        session_id: 세션 식별자.
    """
    with _lock:
        record = _jobs.get(session_id)
        if record is None:
            return
        record.status = STATUS_TIMEOUT
        record.error_message = "Job timed out"
        record.finished_at = datetime.now()


def is_timeout(session_id: str, timeout_sec: int = JOB_TIMEOUT_SEC) -> bool:
    """Job이 시작 후 timeout_sec(기본 5분)를 초과했는지 판정한다.

    이미 종료(done/error/timeout)된 Job은 항상 False.

    Args:
        session_id: 세션 식별자.
        timeout_sec: 타임아웃 기준 초 (기본 JOB_TIMEOUT_SEC=300).

    Returns:
        타임아웃 초과 시 True.
    """
    with _lock:
        record = _jobs.get(session_id)
        if record is None or record.started_at is None:
            return False
        if record.status in _TERMINAL_STATES:
            return False
        elapsed = (datetime.now() - record.started_at).total_seconds()
        return elapsed > timeout_sec


def clear_job(session_id: str) -> None:
    """Job 레코드를 메모리에서 제거한다 (세션 종료/정리용).

    Args:
        session_id: 세션 식별자.
    """
    with _lock:
        _jobs.pop(session_id, None)
