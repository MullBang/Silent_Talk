"""임시 파일 자동 삭제 서비스 (APScheduler 기반).

설계문서 13절 데이터 생명주기 기준.

추론 완료 시점에는 delete_upload_file()로 즉시 삭제하고, BackgroundScheduler가
1시간 주기로 TMP_UPLOAD_DIR의 1시간 이상 경과 파일을 안전망으로 회수한다.
삭제 실패는 /logs/cleanup_error.log에 기록하며, 연속 3회 실패 시 CRITICAL로
수동 처리 알림을 남긴다.
"""

from __future__ import annotations

import logging
import os
import sys
import time

from apscheduler.schedulers.background import BackgroundScheduler

# backend 루트를 import 경로에 부트스트랩
_BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

from config import LOG_DIR, TMP_UPLOAD_DIR  # noqa: E402

# 정리 기준: 1시간(3600초) 이상 경과한 임시 파일
_MAX_AGE_SEC = 3600
# 연속 실패 누적이 이 값에 도달하면 CRITICAL 로그
_CRITICAL_FAILURE_THRESHOLD = 3


def _build_logger() -> logging.Logger:
    """/logs/cleanup_error.log에 기록하는 전용 로거를 구성한다."""
    os.makedirs(LOG_DIR, exist_ok=True)
    logger = logging.getLogger("file_cleaner")
    logger.setLevel(logging.INFO)
    log_path = os.path.join(LOG_DIR, "cleanup_error.log")
    # 중복 핸들러 등록 방지
    if not any(
        isinstance(h, logging.FileHandler)
        and getattr(h, "baseFilename", None) == os.path.abspath(log_path)
        for h in logger.handlers
    ):
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        )
        logger.addHandler(handler)
    return logger


_logger = _build_logger()
_scheduler: BackgroundScheduler | None = None
_consecutive_failures = 0


def delete_upload_file(file_path: str) -> bool:
    """단일 임시 파일을 즉시 삭제한다 (추론 완료 직후 호출).

    Args:
        file_path: 삭제할 임시 파일 경로.

    Returns:
        삭제 성공 시 True, 실패 시 False (실패는 cleanup_error.log에 기록).
    """
    try:
        os.remove(file_path)
        return True
    except OSError as exc:
        _logger.error("임시 파일 삭제 실패: %s (%s)", file_path, exc)
        return False


def _cleanup_expired_files(max_age_sec: int = _MAX_AGE_SEC) -> tuple[int, int]:
    """TMP_UPLOAD_DIR에서 max_age_sec 이상 경과한 파일을 일괄 삭제한다.

    Args:
        max_age_sec: 삭제 기준 경과 시간(초).

    Returns:
        (삭제 성공 수, 삭제 실패 수).
    """
    deleted = 0
    failed = 0
    if not os.path.isdir(TMP_UPLOAD_DIR):
        return deleted, failed

    now = time.time()
    for name in os.listdir(TMP_UPLOAD_DIR):
        path = os.path.join(TMP_UPLOAD_DIR, name)
        if not os.path.isfile(path):
            continue
        try:
            age = now - os.path.getmtime(path)
        except OSError as exc:
            _logger.error("파일 상태 확인 실패: %s (%s)", path, exc)
            failed += 1
            continue
        if age >= max_age_sec:
            if delete_upload_file(path):
                deleted += 1
            else:
                failed += 1
    return deleted, failed


def _run_cleanup() -> None:
    """스케줄러가 주기적으로 호출하는 정리 작업.

    실패가 발생하면 연속 실패 카운터를 증가시키고, 임계치(3회) 도달 시
    CRITICAL 로그로 수동 처리 알림을 남긴다. 한 번이라도 성공적으로 끝나면
    카운터를 초기화한다.
    """
    global _consecutive_failures
    try:
        deleted, failed = _cleanup_expired_files()
        if failed > 0:
            _consecutive_failures += 1
            _logger.error(
                "정리 작업 실패 %d건 (삭제 %d건). 연속 실패 %d회",
                failed,
                deleted,
                _consecutive_failures,
            )
            if _consecutive_failures >= _CRITICAL_FAILURE_THRESHOLD:
                _logger.critical(
                    "임시 파일 정리 연속 %d회 실패 — 수동 처리 필요",
                    _consecutive_failures,
                )
        else:
            _consecutive_failures = 0
    except Exception as exc:  # noqa: BLE001 - 스케줄러 스레드 보호
        _consecutive_failures += 1
        _logger.error("정리 작업 예외: %s (연속 %d회)", exc, _consecutive_failures)
        if _consecutive_failures >= _CRITICAL_FAILURE_THRESHOLD:
            _logger.critical(
                "임시 파일 정리 연속 %d회 실패 — 수동 처리 필요",
                _consecutive_failures,
            )


def start_cleanup_scheduler() -> BackgroundScheduler:
    """BackgroundScheduler를 시작하고 1시간 주기 정리 작업을 등록한다.

    이미 실행 중이면 기존 스케줄러를 반환한다.

    Returns:
        실행 중인 BackgroundScheduler 인스턴스.
    """
    global _scheduler
    if _scheduler is not None and _scheduler.running:
        return _scheduler

    _scheduler = BackgroundScheduler()
    _scheduler.add_job(
        _run_cleanup,
        trigger="interval",
        hours=1,
        id="tmp_upload_cleanup",
        replace_existing=True,
    )
    _scheduler.start()
    return _scheduler


def shutdown_cleanup_scheduler() -> None:
    """스케줄러를 안전하게 종료한다 (애플리케이션 shutdown 시 호출)."""
    global _scheduler
    if _scheduler is not None and _scheduler.running:
        _scheduler.shutdown(wait=False)
    _scheduler = None
