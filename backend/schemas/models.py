"""Pydantic 요청/응답 스키마 정의.

API 계층의 입출력 계약을 정의한다. confidence는 0~1 정규화값,
raw_score는 CTC log-prob 원본임을 스키마 수준에서 구분한다.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel


class JobStatus(str, Enum):
    """Job 진행 상태 열거형."""

    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"


class UploadResponse(BaseModel):
    """POST /api/upload 응답."""

    session_id: str
    filename: str
    status: str


class InferRequest(BaseModel):
    """POST /api/infer 요청."""

    session_id: str


class InferResponse(BaseModel):
    """POST /api/infer 응답. results를 포함하지 않는다 (폴링 전용 분리)."""

    job_id: str
    session_id: str
    status: JobStatus


class InferResultItem(BaseModel):
    """추론 결과 단일 항목."""

    transcript: str
    confidence: float  # 0~1 정규화값
    raw_score: float   # CTC log-prob 원본


class ResultResponse(BaseModel):
    """GET /api/result/{session_id} 응답."""

    session_id: str
    status: JobStatus
    results: list[InferResultItem] | None = None
    error: str | None = None


class EvaluationRequest(BaseModel):
    """POST /api/evaluation/run 요청. 경로 대신 식별자만 받는다."""

    test_set_id: str


class EvaluationResponse(BaseModel):
    """POST /api/evaluation/run 응답."""

    job_id: str
    test_set_id: str
    status: JobStatus


class EvaluationStatusResponse(BaseModel):
    """GET /api/evaluation/status/{id} 응답."""

    job_id: str
    status: JobStatus
    progress: float | None = None
    metrics: dict | None = None


class ErrorResponse(BaseModel):
    """공통 에러 응답."""

    detail: str
    code: str | None = None
