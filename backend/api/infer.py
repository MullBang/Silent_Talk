"""추론 라우트.

POST /api/infer  — 추론 Job 비동기 등록. results 즉시 반환 절대 금지.
                   반환: {job_id, session_id, status:'processing'}
GET  /api/result/{session_id} — 결과 폴링 전용 엔드포인트.
"""

from __future__ import annotations

import glob
import os
import sys

from fastapi import APIRouter, BackgroundTasks, HTTPException

# backend 루트를 import 경로에 부트스트랩
_BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

from config import MODEL_WEIGHTS_PATH, TMP_UPLOAD_DIR  # noqa: E402
from preprocessing.pipeline import run_preprocessing_pipeline  # noqa: E402
from schemas.models import (  # noqa: E402
    InferRequest,
    InferStartResponse,
    ResultItem,
    ResultResponse,
)
from services import job_manager as jm  # noqa: E402
from services.file_cleaner import delete_upload_file  # noqa: E402

router = APIRouter()

_FAIL_TEXT = "[검출 실패 구간]"
_UNREC_TEXT = "[인식 불가]"
# 가중치가 없을 때의 폴백 텍스트.
_DUMMY_TEXT = "[모델 미학습 — 가중치 없음]"

# 지연 로딩 모델 캐시
_model = None
_model_loaded = False


def _get_model():
    """학습된 가중치가 있으면 모델을 1회 로드한다 (없으면 None)."""
    global _model, _model_loaded
    if not _model_loaded:
        _model_loaded = True
        try:
            from models.baseline import LipNetBaseline

            if os.path.exists(MODEL_WEIGHTS_PATH):
                _model = LipNetBaseline.load_from_checkpoint(MODEL_WEIGHTS_PATH)
        except Exception:  # noqa: BLE001 - 로드 실패 시 더미 폴백
            _model = None
    return _model


def _infer_segment(tensor) -> tuple[str, float | None]:
    """세그먼트 텐서를 모델로 추론한다 (가중치 없으면 더미)."""
    model = _get_model()
    if model is None:
        return _DUMMY_TEXT, None
    import torch

    from models.decoder import decode

    with torch.no_grad():
        log_probs = model(tensor)
    text, confidence, _raw = decode(log_probs)
    return (text or _UNREC_TEXT), confidence


def _find_upload_file(session_id: str) -> str | None:
    """session_id 접두사로 임시 업로드 파일 경로를 찾는다.

    Args:
        session_id: 업로드 시 발급된 세션 식별자.

    Returns:
        파일 경로 또는 미존재 시 None.
    """
    matches = glob.glob(os.path.join(TMP_UPLOAD_DIR, f"{session_id}_*"))
    return matches[0] if matches else None


@router.post("/infer", response_model=InferStartResponse)
async def start_inference(
    req: InferRequest, background_tasks: BackgroundTasks
) -> InferStartResponse:
    """추론 Job을 비동기로 등록한다 (results 즉시 반환 금지).

    Args:
        req: {session_id}.
        background_tasks: FastAPI 백그라운드 태스크 큐.

    Returns:
        {job_id, session_id, status:'processing'}.

    Raises:
        HTTPException: 업로드 파일 없음(404), 이미 처리 중(409).
    """
    file_path = _find_upload_file(req.session_id)
    if file_path is None:
        raise HTTPException(status_code=404, detail="업로드 파일을 찾을 수 없습니다.")

    existing = jm.get_job(req.session_id)
    if existing is not None and existing.status in (
        jm.STATUS_PENDING,
        jm.STATUS_PROCESSING,
    ):
        raise HTTPException(status_code=409, detail="이미 처리 중인 세션입니다.")

    job_id = jm.create_job(req.session_id)
    background_tasks.add_task(_run_infer_job, req.session_id, file_path)

    return InferStartResponse(
        job_id=job_id, session_id=req.session_id, status=jm.STATUS_PROCESSING
    )


def _run_infer_job(session_id: str, file_path: str) -> None:
    """백그라운드 추론 작업.

    전처리 파이프라인 실행 → 세그먼트별 (더미) 추론 → 진행률 갱신 →
    완료 처리 → 임시 파일 즉시 삭제. 예외 시 fail_job.

    Args:
        session_id: 세션 식별자.
        file_path: 임시 업로드 파일 경로.
    """
    try:
        segments = run_preprocessing_pipeline(file_path)
        results: list[dict] = []
        total = len(segments) or 1

        for i, seg in enumerate(segments):
            if seg.get("skipped"):
                results.append(
                    {
                        "start_ms": seg["start_ms"],
                        "end_ms": seg["end_ms"],
                        "text": _FAIL_TEXT,
                        "confidence": None,
                    }
                )
            else:
                text, confidence = _infer_segment(seg["tensor"])
                results.append(
                    {
                        "start_ms": seg["start_ms"],
                        "end_ms": seg["end_ms"],
                        "text": text,
                        "confidence": confidence,
                    }
                )
            jm.update_progress(session_id, (i + 1) / total * 100.0)

        jm.complete_job(session_id, results)
    except Exception as exc:  # noqa: BLE001 - 백그라운드 작업 보호
        jm.fail_job(session_id, str(exc))
    finally:
        # 추론 완료/실패와 무관하게 임시 파일 즉시 삭제 (개인정보 보호)
        delete_upload_file(file_path)


@router.get("/result/{session_id}", response_model=ResultResponse)
async def get_result(session_id: str) -> ResultResponse:
    """추론 결과 폴링 핸들러.

    Args:
        session_id: 세션 식별자.

    Returns:
        ResultResponse {session_id, status, progress, error_message, results}.

    Raises:
        HTTPException: Job 미존재 시 404.
    """
    record = jm.get_job(session_id)
    if record is None:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")

    items: list[ResultItem] | None = None
    if record.results is not None:
        items = [ResultItem(**r) for r in record.results]

    return ResultResponse(
        session_id=session_id,
        status=record.status,
        progress=record.progress,
        error_message=record.error_message,
        results=items,
    )
