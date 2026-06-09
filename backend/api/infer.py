"""추론 라우트.

POST /api/infer  — 추론 Job 비동기 등록. results 즉시 반환 절대 금지.
                   반환: {job_id, session_id, status:'processing'}
GET  /api/result/{session_id} — 결과 폴링 전용 엔드포인트.
"""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks

router = APIRouter()


async def run_inference(session_id: str, background_tasks: BackgroundTasks) -> dict:
    """추론 Job을 비동기로 등록한다.

    절대 results를 즉시 반환하지 않는다. Job을 job_manager에 등록하고
    백그라운드 태스크로 전처리→모델→디코딩 파이프라인을 예약한다.

    Args:
        session_id: 업로드 시 발급된 세션 식별자.
        background_tasks: FastAPI 백그라운드 태스크 큐.

    Returns:
        {job_id, session_id, status:'processing'}.
    """
    pass


async def get_result(session_id: str) -> dict:
    """추론 결과 폴링 핸들러.

    Args:
        session_id: 세션 식별자.

    Returns:
        status가 'processing'이면 진행 상태, 'done'이면 결과(transcript,
        confidence, raw_score 포함), 'failed'이면 에러 사유.
    """
    pass
