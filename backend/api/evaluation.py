"""평가 라우트.

POST /api/evaluation/run            — 테스트셋 평가 Job 등록 (test_set_id 사용).
GET  /api/evaluation/status/{id}    — 평가 진행 상태/결과 조회.

보안: test_set_path 직접 전달 금지 → 등록된 test_set_id로만 접근한다.
"""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()


async def run_evaluation(test_set_id: str) -> dict:
    """테스트셋 평가 Job을 등록한다.

    test_set_path를 직접 받지 않고 사전 등록된 test_set_id로만 데이터셋을
    참조한다 (경로 주입 방지).

    Args:
        test_set_id: 등록된 테스트셋 식별자.

    Returns:
        {job_id, test_set_id, status:'processing'}.
    """
    pass


async def get_evaluation_status(job_id: str) -> dict:
    """평가 Job 상태/결과를 조회한다.

    Args:
        job_id: 평가 Job 식별자.

    Returns:
        진행률 또는 최종 지표(CER/WER 등)를 담은 딕셔너리.
    """
    pass
