"""임시 파일 자동 삭제 서비스.

APScheduler로 TMP_UPLOAD_DIR의 만료 임시 파일을 주기적으로 정리한다.
추론 완료 시점에는 즉시 os.remove()로 삭제하고, 본 스케줄러는 누락분
(중단/실패로 남은 파일)을 안전망으로 회수한다.
"""

from __future__ import annotations


def remove_temp_file(path: str) -> None:
    """단일 임시 파일을 즉시 삭제한다 (추론 완료 직후 호출).

    Args:
        path: 삭제할 임시 파일 경로.
    """
    pass


def cleanup_expired() -> int:
    """TMP_UPLOAD_DIR에서 만료된 임시 파일을 일괄 삭제한다.

    Returns:
        삭제된 파일 수.
    """
    pass


def start_scheduler() -> None:
    """APScheduler 백그라운드 스케줄러를 시작하고 cleanup_expired를 등록한다."""
    pass


def shutdown_scheduler() -> None:
    """스케줄러를 안전하게 종료한다 (애플리케이션 shutdown 시 호출)."""
    pass
