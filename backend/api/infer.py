"""추론 라우트.

POST /api/infer  — 추론 Job 비동기 등록. results 즉시 반환 절대 금지.
                   반환: {job_id, session_id, status:'processing'}
GET  /api/result/{session_id} — 결과 폴링 전용 엔드포인트.

인식 모델은 config.MODEL_BACKEND로 고른다.
  olkavs  : 공식 모델 상주 워커(services.olkavs_service). 영상을 최대 12초 창으로 나눠 인식한다.
  compact : 자체 경량 모델(3초 세그먼트 전처리 파이프라인).

Job 모드
  upload : POST /api/infer (업로드 영상). 창 분할 인식, 앞뒤 무발화 구간을 잘라 내지 않는다.
  record : POST /api/record (웹캠 녹화, api/record.py). 앞뒤 무발화를 잘라 내고 12초 이하 녹화는 창 하나로 인식한다.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import sys
import threading
from datetime import datetime
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException

# backend 루트를 import 경로에 부트스트랩
_BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

from config import (  # noqa: E402
    JOB_TIMEOUT_SEC,
    MODEL_BACKEND,
    MODEL_WEIGHTS_PATH,
    TMP_ROI_DIR,
    TMP_UPLOAD_DIR,
)
from models.baseline import ModelCompatibilityError  # noqa: E402
from models.tokenizer import TokenizerMismatchError  # noqa: E402
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
logger = logging.getLogger(__name__)

_FAIL_TEXT = "[검출 실패 구간]"
_UNREC_TEXT = "[인식 불가]"

MODE_UPLOAD = "upload"
MODE_RECORD = "record"
_MODES = (MODE_UPLOAD, MODE_RECORD)

MSG_RECORD_BACKEND = "녹화 인식은 공식 인식 모델(olkavs) 서버에서만 사용할 수 있습니다."

# upload.py가 발급하는 str(uuid.uuid4()) 형식만 허용한다 (glob/경로 문자 차단).
_SESSION_ID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}")


class ModelUnavailableError(RuntimeError):
    """서비스 모델 가중치가 없어 인식할 수 없음 (결과처럼 보이는 더미 문구를 내지 않는다)."""


# 지연 로딩 모델 캐시
_model = None
_model_loaded = False

# 백그라운드 작업 스레드가 아직 도는 세션 (타임아웃 응답 뒤 같은 세션 재등록 차단)
_running_sessions: set[str] = set()
_running_lock = threading.Lock()


def is_valid_session_id(session_id: str) -> bool:
    """업로드가 발급한 UUID4 형식인지 확인한다."""
    try:
        return isinstance(session_id, str) and _SESSION_ID_RE.fullmatch(session_id) is not None
    except TypeError:
        return False


def _get_model():
    """학습된 가중치가 있으면 모델을 1회 로드한다 (없으면 None)."""
    global _model, _model_loaded
    if not _model_loaded:
        try:
            from models.baseline import LipNetBaseline

            if os.path.exists(MODEL_WEIGHTS_PATH):
                _model = LipNetBaseline.load_from_checkpoint(MODEL_WEIGHTS_PATH)
        except (TokenizerMismatchError, ModelCompatibilityError):
            raise  # 구형 가중치를 새 사전으로 해석하거나 미학습 상태로 숨기지 않는다.
        except Exception:  # noqa: BLE001 - 로드 실패 시 더미 폴백
            _model = None
        _model_loaded = True
    return _model


def _infer_segment(tensor) -> tuple[str, float | None]:
    """세그먼트 텐서를 compact 모델로 추론한다.

    Raises:
        ModelUnavailableError: 가중치가 없을 때. 예전처럼 더미 문구를 결과로 돌려주지 않는다.
    """
    model = _get_model()
    if model is None:
        raise ModelUnavailableError(
            "compact 서비스 모델 가중치가 없습니다. LIPNET_WEIGHTS로 경로를 지정하거나 "
            "LIPNET_MODEL_BACKEND=olkavs(공식 모델)로 실행하세요."
        )
    import torch

    from models.decoder import decode

    with torch.no_grad():
        log_probs = model(tensor)
    text, confidence, _raw = decode(log_probs)
    return (text or _UNREC_TEXT), confidence


def _find_upload_file(session_id: str) -> str | None:
    """session_id 접두사로 임시 업로드 파일 경로를 찾는다.

    UUID4 형식이 아니면 찾지 않는다. glob 대신 디렉토리 목록의 접두사 비교만 하므로
    '*'·'..'·절대경로 같은 값으로 업로드 폴더 밖이나 다른 세션의 파일을 집을 수 없다.

    Args:
        session_id: 업로드 시 발급된 세션 식별자.

    Returns:
        파일 경로 또는 미존재 시 None.
    """
    if not is_valid_session_id(session_id):
        return None
    try:
        names = sorted(os.listdir(TMP_UPLOAD_DIR))
    except OSError:
        return None
    prefix = f"{session_id}_"
    for name in names:
        path = os.path.join(TMP_UPLOAD_DIR, name)
        if name.startswith(prefix) and os.path.isfile(path):
            return path
    return None


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
    return start_background_job(req.session_id, file_path, background_tasks, mode=MODE_UPLOAD)


def start_background_job(session_id: str, file_path: str, background_tasks: BackgroundTasks,
                         mode: str = MODE_UPLOAD, video_seconds: float | None = None) -> InferStartResponse:
    """같은 세션 중복 실행을 막고 Job을 만든 뒤 백그라운드 인식을 예약한다 (upload·record 공용).

    Args:
        session_id: 세션 식별자 (UUID4).
        file_path: 임시 영상 파일 경로 (Job이 끝나면 지운다).
        background_tasks: FastAPI 백그라운드 태스크 큐.
        mode: 'upload'(창 분할) 또는 'record'(발화 전체·앞뒤 무발화 제거).
        video_seconds: 서버가 잰 영상 길이(초). 없으면 upload 모드에서 메타데이터로 어림한다.

    Returns:
        {job_id, session_id, status:'processing'}.

    Raises:
        HTTPException: 이미 처리 중인 세션(409).
        ValueError: 알 수 없는 mode.
    """
    if mode not in _MODES:
        raise ValueError(f"알 수 없는 Job 모드: {mode!r}")
    with _running_lock:
        existing = jm.get_job(session_id)
        # 타임아웃으로 응답한 Job이라도 이전 작업 스레드가 아직 돌면 같은 세션을 다시 받지 않는다
        # (이전 작업이 새 Job의 레코드·업로드 파일·입술 임시 폴더를 덮어쓰거나 지우는 경합 방지).
        if session_id in _running_sessions or (
            existing is not None and existing.status in (jm.STATUS_PENDING, jm.STATUS_PROCESSING)
        ):
            raise HTTPException(status_code=409, detail="이미 처리 중인 세션입니다.")
        _running_sessions.add(session_id)
    try:
        job_id = jm.create_job(session_id)
        background_tasks.add_task(_run_infer_job, session_id, file_path, mode, video_seconds)
    except Exception:
        with _running_lock:
            _running_sessions.discard(session_id)
        raise
    return InferStartResponse(job_id=job_id, session_id=session_id, status=jm.STATUS_PROCESSING)


def _run_compact_job(session_id: str, file_path: str) -> list[dict]:
    """compact 경로: 25fps·3초 세그먼트 전처리 → 세그먼트별 추론."""
    segments = run_preprocessing_pipeline(file_path)
    results: list[dict] = []
    total = len(segments) or 1
    for i, seg in enumerate(segments):
        if seg.get("skipped"):
            results.append({"start_ms": seg["start_ms"], "end_ms": seg["end_ms"],
                            "text": _FAIL_TEXT, "confidence": None})
        else:
            text, confidence = _infer_segment(seg["tensor"])
            results.append({"start_ms": seg["start_ms"], "end_ms": seg["end_ms"],
                            "text": text, "confidence": confidence})
        jm.update_progress(session_id, (i + 1) / total * 100.0)
    return results


def _probe_video_seconds(file_path: str) -> float | None:
    """메타데이터만으로 영상 길이(초)를 어림한다. 알 수 없으면 None (프레임을 디코딩하지 않는다).

    입장 제어(남은 Job 시간 안에 끝낼 수 있는지)에만 쓰며, 실제 프레임 선택은 ROI 워커가 한다.
    """
    try:
        import cv2  # 서버 환경의 cv2 (compact 전처리와 같은 모듈)

        cap = cv2.VideoCapture(file_path)
        try:
            if not cap.isOpened():
                return None
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            cap.release()
        if not (1.0 <= fps <= 240.0) or not (count > 0):
            return None  # 브라우저 WebM 등 메타데이터가 없거나 믿기 어려운 경우
        return count / fps
    except Exception as exc:  # noqa: BLE001 - 어림 실패는 입장 제어만 건너뛴다
        logger.warning("영상 길이 어림 실패: %s", exc)
        return None


def _service_model_info(service: Any) -> dict | None:
    """서비스가 알려 주는 모델 정보 (없거나 실패하면 None — 결과 표시를 막지 않는다)."""
    try:
        getter = getattr(service, "model_info", None)
        info = getter() if callable(getter) else None
        return dict(info) if isinstance(info, dict) else None
    except Exception as exc:  # noqa: BLE001 - 모델 정보는 부가 정보다
        logger.warning("모델 정보 조회 실패: %s", exc)
        return None


def _run_olkavs_job(session_id: str, file_path: str, warnings: list[str], mode: str = MODE_UPLOAD,
                    video_seconds: float | None = None) -> tuple[list[dict], dict | None]:
    """olkavs 경로: 상주 워커로 입술 추출(구간별 NPZ) → 구간별 인식. NPZ 폴더는 바로 지운다.

    record 모드(utterance=True)는 녹화 전체를 한 발화로 보고 앞뒤 무발화 구간을 잘라 낸다.
    12초 모델 창은 그대로라 12초를 넘는 녹화(30초 실험 모드)는 여러 창으로 나뉠 수 있다.
    warnings에는 결과와 함께 보여 줄 품질 경고(낮은 프레임률 등)를 덧붙인다.

    Returns:
        (결과 목록, 완료 시점의 모델 정보 또는 None).
    """
    from services.olkavs_service import get_service

    work_dir = os.path.join(TMP_ROI_DIR, session_id)
    try:
        def aborted() -> bool:
            # 폴링과 관계없이 서버가 직접 Job 시간 상한을 판정한다.
            if jm.is_timeout(session_id, JOB_TIMEOUT_SEC):
                jm.timeout_job(session_id)
            record = jm.get_job(session_id)
            return record is None or record.status == jm.STATUS_TIMEOUT

        def remaining() -> float | None:
            # 대기열에서 기다린 시간까지 빼고 남은 Job 시간.
            record = jm.get_job(session_id)
            if record is None or record.started_at is None:
                return None
            return JOB_TIMEOUT_SEC - (datetime.now() - record.started_at).total_seconds()

        if video_seconds is None:
            video_seconds = _probe_video_seconds(file_path)
        service = get_service()
        results = service.transcribe_video(
            file_path, work_dir,
            on_progress=lambda value: jm.update_progress(session_id, value),
            should_abort=aborted,
            remaining_seconds=remaining,
            video_seconds=video_seconds,
            warnings=warnings,
            utterance=(mode == MODE_RECORD),
        )
        return results, _service_model_info(service)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
        if os.path.exists(work_dir):
            logger.error("입술 프레임 임시 폴더 삭제 실패: %s", work_dir)


def _run_infer_job(session_id: str, file_path: str, mode: str = MODE_UPLOAD,
                   video_seconds: float | None = None) -> None:
    """백그라운드 추론 작업.

    선택된 모델 경로로 인식 → 진행률 갱신 → 완료 처리 → 임시 파일 즉시 삭제. 예외 시 fail_job.
    타임아웃으로 이미 종료 상태가 된 Job은 job_manager가 덮어쓰지 않는다.

    Args:
        session_id: 세션 식별자.
        file_path: 임시 업로드(또는 녹화) 파일 경로.
        mode: 'upload' 또는 'record'.
        video_seconds: 서버가 잰 영상 길이(초, record 모드). 없으면 메타데이터로 어림한다.
    """
    try:
        warnings: list[str] = []
        model_info: dict | None = None
        if MODEL_BACKEND == "olkavs":
            results, model_info = _run_olkavs_job(session_id, file_path, warnings, mode=mode,
                                                  video_seconds=video_seconds)
        elif mode == MODE_RECORD:
            raise _UserFacingError(MSG_RECORD_BACKEND)
        else:
            results = _run_compact_job(session_id, file_path)
        jm.complete_job(session_id, results, warnings=warnings or None, model_info=model_info)
    except Exception as exc:  # noqa: BLE001 - 백그라운드 작업 보호
        logger.warning("추론 작업 실패 (%s): %s %s", session_id, exc, getattr(exc, "detail", ""))
        jm.fail_job(session_id, user_error_message(exc))
    finally:
        # 추론 완료/실패와 무관하게 임시 파일 즉시 삭제 (개인정보 보호)
        delete_upload_file(file_path)
        with _running_lock:
            _running_sessions.discard(session_id)


class _UserFacingError(RuntimeError):
    """사용자에게 그대로 보여도 되는 고정 문구의 실패."""

    def __init__(self, message: str):
        super().__init__(message)
        self.user_message = message


_PATH_RE = re.compile(r"([A-Za-z]:[\\/]|\\\\|/(?:Users|home|tmp|mnt)/)[^\s'\"]*")


def user_error_message(exc: BaseException) -> str:
    """사용자에게 보낼 오류 문구. 고정 문구가 있으면 그것을, 없으면 경로를 지운 짧은 문구를 쓴다."""
    message = getattr(exc, "user_message", None)
    if message:
        return str(message)
    text = _PATH_RE.sub("<경로>", str(exc)).strip()
    return text[:300] or "영상을 처리하지 못했습니다."


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
    if not is_valid_session_id(session_id):
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")
    if jm.is_timeout(session_id, JOB_TIMEOUT_SEC):
        jm.timeout_job(session_id)  # 서버 쪽 Job 타임아웃(기본 300초) 적용
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
        warnings=record.warnings,
        model=record.model_info if record.status == jm.STATUS_DONE else None,
    )
