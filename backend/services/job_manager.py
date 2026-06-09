"""비동기 Job 상태 관리.

추론/평가 Job 상태를 인메모리 dict로 관리한다 (MVP). session_id ↔ job 매핑,
status 전이(processing → done/failed), JOB_TIMEOUT_SEC 초과 감지를 담당한다.
"""

from __future__ import annotations

from typing import Any


class JobManager:
    """인메모리 Job 레지스트리."""

    def __init__(self) -> None:
        """Job 저장 딕셔너리를 초기화한다."""
        self._jobs: dict[str, dict[str, Any]] = {}

    def create_job(self, session_id: str) -> str:
        """새 Job을 생성하고 status='processing'으로 등록한다.

        Args:
            session_id: 연결할 세션 식별자.

        Returns:
            발급된 job_id.
        """
        pass

    def update_status(
        self,
        job_id: str,
        status: str,
        result: Any | None = None,
        error: str | None = None,
    ) -> None:
        """Job 상태를 갱신한다.

        Args:
            job_id: 대상 Job 식별자.
            status: 'processing' | 'done' | 'failed'.
            result: 완료 시 결과 페이로드.
            error: 실패 시 에러 메시지.
        """
        pass

    def get_by_session(self, session_id: str) -> dict[str, Any] | None:
        """세션 식별자로 Job 상태를 조회한다.

        Args:
            session_id: 세션 식별자.

        Returns:
            Job 레코드 또는 미존재 시 None.
        """
        pass

    def is_timed_out(self, job_id: str) -> bool:
        """Job이 JOB_TIMEOUT_SEC를 초과했는지 확인한다.

        Args:
            job_id: 대상 Job 식별자.

        Returns:
            타임아웃되었으면 True.
        """
        pass


job_manager: JobManager = JobManager()
"""전역 단일 Job 매니저 인스턴스."""
