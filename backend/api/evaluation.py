"""평가 라우트.

POST /api/evaluation/run            — 테스트셋 평가 Job 등록 (test_set_id 사용).
GET  /api/evaluation/status/{id}    — 평가 진행 상태/결과 조회.

보안: test_set_path 직접 전달 금지 → config.TEST_SETS에 등록된 test_set_id로만 접근.
"""

from __future__ import annotations

import os
import sys
import threading
import uuid

from fastapi import APIRouter, BackgroundTasks, HTTPException

# backend 루트를 import 경로에 부트스트랩
_BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

from config import TEST_SETS  # noqa: E402
from schemas.models import (  # noqa: E402
    EvalRequest,
    EvalResult,
    EvalStartResponse,
    EvalStatusResponse,
)

router = APIRouter()

# eval_job_id -> {status, progress, result}
_eval_jobs: dict[str, dict] = {}
_lock = threading.Lock()


@router.post("/evaluation/run", response_model=EvalStartResponse)
async def run_evaluation(
    req: EvalRequest, background_tasks: BackgroundTasks
) -> EvalStartResponse:
    """테스트셋 평가 Job을 등록한다.

    test_set_path를 직접 받지 않고 config.TEST_SETS에 등록된 test_set_id로만
    데이터셋을 참조한다 (경로 주입 방지).

    Args:
        req: {test_set_id, eval_unit}.
        background_tasks: FastAPI 백그라운드 태스크 큐.

    Returns:
        {eval_job_id, status:'queued'}.

    Raises:
        HTTPException: 등록되지 않은 test_set_id이면 400.
    """
    test_set_path = TEST_SETS.get(req.test_set_id)
    if test_set_path is None:
        raise HTTPException(
            status_code=400,
            detail=f"등록되지 않은 test_set_id입니다: {req.test_set_id}",
        )

    eval_job_id = str(uuid.uuid4())
    with _lock:
        _eval_jobs[eval_job_id] = {"status": "queued", "progress": 0.0, "result": None}

    background_tasks.add_task(
        _run_eval_job, eval_job_id, test_set_path, req.eval_unit
    )

    return EvalStartResponse(eval_job_id=eval_job_id, status="queued")


def _run_eval_job(eval_job_id: str, test_set_path: str, eval_unit: str) -> None:
    """백그라운드 평가 작업 (3단계는 더미 지표).

    Args:
        eval_job_id: 평가 Job 식별자.
        test_set_path: 등록된 테스트셋 경로.
        eval_unit: 'word' | 'sentence'.
    """
    try:
        with _lock:
            _eval_jobs[eval_job_id]["status"] = "processing"

        # TODO(3단계 모델): test_set 배치 추론 → CER/WER/지연/검출률 계산 → CSV 저장
        result = {
            "cer": 0.0,
            "wer": 0.0,
            "avg_latency_ms": 0.0,
            "detection_rate": 0.0,
        }

        with _lock:
            _eval_jobs[eval_job_id]["progress"] = 100.0
            _eval_jobs[eval_job_id]["result"] = result
            _eval_jobs[eval_job_id]["status"] = "done"
    except Exception as exc:  # noqa: BLE001
        with _lock:
            _eval_jobs[eval_job_id]["status"] = "error"
            _eval_jobs[eval_job_id]["result"] = None


@router.get("/evaluation/status/{eval_job_id}", response_model=EvalStatusResponse)
async def get_evaluation_status(eval_job_id: str) -> EvalStatusResponse:
    """평가 Job 상태/결과를 조회한다.

    Args:
        eval_job_id: 평가 Job 식별자.

    Returns:
        EvalStatusResponse {status, progress, result}.

    Raises:
        HTTPException: Job 미존재 시 404.
    """
    with _lock:
        job = _eval_jobs.get(eval_job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="평가 Job을 찾을 수 없습니다.")
        snapshot = dict(job)

    result = EvalResult(**snapshot["result"]) if snapshot["result"] else None
    return EvalStatusResponse(
        status=snapshot["status"], progress=snapshot["progress"], result=result
    )
