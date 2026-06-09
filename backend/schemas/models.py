"""Pydantic 요청/응답 스키마 정의.

설계문서 5절 API 명세 기준. API 계층의 입출력 계약을 정의한다.
confidence는 0~1 정규화값(또는 null), raw_score는 CTC log-prob 원본.
status 문자열은 services.job_manager의 상태값(pending/processing/done/error/timeout)을
따른다.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    """POST /api/upload 응답."""

    session_id: str
    file_name: str
    duration_sec: float


class InferRequest(BaseModel):
    """POST /api/infer 요청."""

    session_id: str


class InferStartResponse(BaseModel):
    """POST /api/infer 응답. results를 절대 포함하지 않는다 (폴링 전용 분리)."""

    job_id: str
    session_id: str
    status: str = "processing"


class ResultItem(BaseModel):
    """추론 결과 단일 세그먼트."""

    start_ms: Optional[int] = None
    end_ms: Optional[int] = None
    text: str
    confidence: Optional[float] = None  # 0.0~1.0 정규화 신뢰도 또는 null


class ResultResponse(BaseModel):
    """GET /api/result/{session_id} 응답."""

    session_id: str
    status: str
    progress: float = 0.0
    error_message: Optional[str] = None
    results: Optional[list[ResultItem]] = None


class EvalRequest(BaseModel):
    """POST /api/evaluation/run 요청. 경로 대신 등록된 식별자만 받는다."""

    test_set_id: str
    eval_unit: Literal["word", "sentence"] = "word"


class EvalStartResponse(BaseModel):
    """POST /api/evaluation/run 응답."""

    eval_job_id: str
    status: str = "queued"


class EvalResult(BaseModel):
    """평가 지표 결과."""

    cer: float
    wer: float
    avg_latency_ms: float
    detection_rate: float


class EvalStatusResponse(BaseModel):
    """GET /api/evaluation/status/{eval_job_id} 응답."""

    status: str
    progress: float = 0.0
    result: Optional[EvalResult] = None


class ErrorResponse(BaseModel):
    """공통 에러 응답."""

    detail: str
    code: Optional[str] = None
