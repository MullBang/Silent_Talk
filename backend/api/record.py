"""웹캠 녹화 접수 라우트 (6단계).

POST /api/record — 브라우저 MediaRecorder 녹화(WebM, Safari는 MP4)를 받아 녹화 모드로 인식하는
백그라운드 Job을 등록한다. 녹화 모드는 앞뒤 무발화를 잘라 내고, 12초 이하 녹화는 창 하나로 인식한다
(30초 실험 모드처럼 12초를 넘으면 모델 창 길이 때문에 여러 창으로 나뉠 수 있다). 결과는 즉시 반환하지 않고 GET /api/result/{session_id} 폴링으로만 준다.

- 요청 본문은 limits.RequestBodyLimitMiddleware가 RECORD_MAX_SIZE_MB(+멀티파트 여유 1MB)를 넘으면 받는 즉시
  413으로 끊는다. 받아 둔 사본은 RECORD_MAX_SIZE_MB+1바이트까지만 옮긴다(넘으면 400). 내용 시그니처로만 형식을 판정한다
  (WebM: EBML 1A45DFA3 + DocType webm/matroska, MP4: ftyp). 클라이언트 파일명·확장자·MIME은 믿지 않는다.
- MediaRecorder WebM에는 길이 메타데이터가 없고 cv2가 fps=1000·프레임 수 음수를 돌려준다. 그래서 서버가
  실제 프레임 시각(CAP_PROP_POS_MSEC)을 읽어 길이를 잰다. RECORD_MIN_SECONDS 이상, RECORD_MAX_SECONDS
  미만만 접수한다. 측정은 asyncio.to_thread로 돌려 이벤트 루프를 막지 않는다.
- 거부 사유는 고정 한국어 문구로만 알린다(서버 경로·예외 원문 금지). 거부하면 저장한 파일을 바로 지운다.
- 공식 모델(olkavs) 백엔드에서만 지원한다. compact면 503.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import statistics
import sys
import time
import uuid

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile

_BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

import config  # noqa: E402
from api import infer as infer_api  # noqa: E402
from api import limits  # noqa: E402
from schemas.models import RecordStartResponse  # noqa: E402

router = APIRouter()
logger = logging.getLogger(__name__)

# 사용자에게 보내는 고정 문구 (경로·예외 원문을 넣지 않는다)
MSG_BACKEND_UNSUPPORTED = "녹화 인식은 공식 인식 모델(olkavs) 서버에서만 사용할 수 있습니다."
MSG_EMPTY = "녹화 파일이 비어 있거나 너무 작습니다. 다시 녹화해 주세요."
MSG_UNSUPPORTED = "지원하지 않는 녹화 형식입니다 (WebM 또는 MP4만 가능)."
MSG_TOO_LARGE = "녹화 파일이 너무 큽니다 ({mb}MB 이하만 접수합니다)."
MSG_UNREADABLE = "녹화 영상을 읽을 수 없습니다. 다시 녹화해 주세요."
MSG_TOO_SHORT = "녹화가 너무 짧습니다 ({sec:g}초 이상 녹화해 주세요)."
MSG_TOO_LONG = "녹화 길이는 {sec:g}초 미만이어야 합니다."
MSG_SAVE_FAILED = "녹화 파일을 저장하지 못했습니다. 잠시 후 다시 시도해 주세요."

_MIN_FILE_BYTES = 32
_EBML_MAGIC = b"\x1a\x45\xdf\xa3"
_EBML_DOCTYPE_ID = 0x4282
_HEADER_BYTES = 4096


class RecordingRejected(ValueError):
    """녹화를 접수하지 않는 사유. user_message는 사용자에게 보내도 되는 고정 문구다."""

    def __init__(self, user_message: str, detail: str = ""):
        super().__init__(user_message)
        self.user_message = user_message
        self.detail = detail


# ---------------------------------------------------------------------------
# 형식 판정 (내용 시그니처)
# ---------------------------------------------------------------------------

def _read_vint(data: bytes, pos: int, keep_marker: bool) -> tuple[int, int] | None:
    """EBML 가변 길이 정수를 읽는다.

    Args:
        data: 바이트열.
        pos: 시작 위치.
        keep_marker: True면 요소 ID처럼 길이 표시 비트를 남기고, False면 크기 값처럼 지운다.

    Returns:
        (값, 다음 위치) 또는 형식이 잘못되면 None.
    """
    if pos >= len(data):
        return None
    first = data[pos]
    length = 1
    mask = 0x80
    while length <= 8 and not (first & mask):
        length += 1
        mask >>= 1
    if length > 8 or pos + length > len(data):
        return None
    value = first if keep_marker else (first & (mask - 1))
    for byte in data[pos + 1:pos + length]:
        value = (value << 8) | byte
    return value, pos + length


def _ebml_doctype(head: bytes) -> str | None:
    """EBML 헤더의 DocType 문자열을 찾는다 (없거나 형식이 깨졌으면 None)."""
    try:
        if not head.startswith(_EBML_MAGIC):
            return None
        size = _read_vint(head, 4, keep_marker=False)
        if size is None:
            return None
        header_size, pos = size
        end = min(len(head), pos + header_size)
        while pos < end:
            element = _read_vint(head, pos, keep_marker=True)
            if element is None:
                return None
            element_id, pos = element
            length = _read_vint(head, pos, keep_marker=False)
            if length is None:
                return None
            element_size, pos = length
            if pos + element_size > len(head):
                return None
            if element_id == _EBML_DOCTYPE_ID:
                return head[pos:pos + element_size].rstrip(b"\x00").decode("ascii", "strict")
            pos += element_size
        return None
    except (UnicodeDecodeError, ValueError, IndexError):
        return None


def detect_recording_container(head: bytes) -> str | None:
    """녹화 파일 선두 바이트로 컨테이너를 판정한다.

    Args:
        head: 파일 앞부분(수 KB).

    Returns:
        'webm'(EBML + DocType webm/matroska), 'mp4'(ISO BMFF ftyp) 또는 지원하지 않으면 None.
    """
    if head.startswith(_EBML_MAGIC):
        doctype = _ebml_doctype(head)
        return "webm" if doctype in ("webm", "matroska") else None
    if len(head) >= 12 and head[4:8] == b"ftyp":
        return "mp4"
    return None


# ---------------------------------------------------------------------------
# 실제 길이 측정
# ---------------------------------------------------------------------------

def measure_recording_seconds(path: str, max_seconds: float = config.RECORD_MAX_SECONDS,
                              max_frames: int = config.RECORD_MAX_FRAMES,
                              time_budget: float = config.RECORD_PROBE_TIMEOUT_SEC) -> float:
    """프레임 시각(PTS)을 실제로 읽어 녹화 길이(초)를 잰다.

    길이 = 마지막 PTS − 첫 PTS + 대표 프레임 간격(PTS 차이의 중앙값). 메타데이터(fps·프레임 수)는
    MediaRecorder WebM에서 믿을 수 없으므로 쓰지 않는다(간격을 모를 때의 보조값으로만 사용).
    PTS가 상한을 1초 넘게 지나면 거기서 멈춰 그때까지의 길이(상한 이상)를 돌려준다.

    Args:
        path: 저장된 녹화 파일 경로.
        max_seconds: 길이 상한(초). 조기 종료 판정에만 쓴다.
        max_frames: 읽을 최대 프레임 수. 넘으면 비정상 파일로 거부한다.
        time_budget: 측정에 쓸 최대 시간(초). 넘으면 거부한다.

    Returns:
        녹화 길이(초).

    Raises:
        RecordingRejected: 열 수 없음·첫 프레임 디코딩 실패·PTS 비정상·프레임 수/시간 초과.
    """
    import cv2  # 서버 환경의 cv2 (지연 import)

    started = time.monotonic()
    cap = None
    try:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RecordingRejected(MSG_UNREADABLE, "VideoCapture를 열 수 없음")
        ok, frame = cap.read()  # 첫 프레임은 실제로 디코딩해 본다
        if not ok or frame is None or getattr(frame, "size", 0) == 0:
            raise RecordingRejected(MSG_UNREADABLE, "첫 프레임 디코딩 실패")
        first = float(cap.get(cv2.CAP_PROP_POS_MSEC))
        if not math.isfinite(first):
            raise RecordingRejected(MSG_UNREADABLE, "첫 프레임 PTS 비정상")
        last = first
        prev = first
        gaps: list[float] = []
        frames = 1
        limit_ms = max_seconds * 1000.0 + 1000.0
        while True:
            if last - first > limit_ms:
                break  # 상한을 충분히 넘었다: 더 읽지 않는다
            if frames >= max_frames:
                raise RecordingRejected(MSG_UNREADABLE, f"프레임 수 상한({max_frames}) 초과")
            if time.monotonic() - started > time_budget:
                raise RecordingRejected(MSG_UNREADABLE, f"길이 측정 시간 상한({time_budget:.0f}초) 초과")
            if not cap.grab():
                break
            pts = float(cap.get(cv2.CAP_PROP_POS_MSEC))
            if not math.isfinite(pts):
                raise RecordingRejected(MSG_UNREADABLE, "PTS 비정상")
            frames += 1
            if pts > prev:
                gaps.append(pts - prev)
            prev = pts
            last = max(last, pts)
        if gaps:
            gap = statistics.median(gaps)
        else:
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            gap = 1000.0 / fps if 1.0 <= fps <= 240.0 else 0.0
        return max(0.0, (last - first + gap) / 1000.0)
    except RecordingRejected:
        raise
    except Exception as exc:  # noqa: BLE001 - 디코더 오류는 모두 '읽을 수 없음'으로 본다
        raise RecordingRejected(MSG_UNREADABLE, f"길이 측정 오류: {type(exc).__name__}: {exc}") from exc
    finally:
        if cap is not None:
            try:
                cap.release()
            except Exception:  # noqa: BLE001 - 해제 실패는 무시
                pass


# ---------------------------------------------------------------------------
# 업로드 수신
# ---------------------------------------------------------------------------

def _safe_remove(path: str | None) -> None:
    """파일을 조용히 지운다 (없으면 무시, 실패는 서버 로그)."""
    limits.safe_remove(path)


_REJECT_MESSAGES = {
    limits.REJECT_EMPTY: lambda: MSG_EMPTY,
    limits.REJECT_UNSUPPORTED: lambda: MSG_UNSUPPORTED,
    limits.REJECT_TOO_LARGE: lambda: MSG_TOO_LARGE.format(mb=config.RECORD_MAX_SIZE_MB),
    limits.REJECT_SAVE_FAILED: lambda: MSG_SAVE_FAILED,
}


def record_body_limit() -> tuple[float, str]:
    """요청 본문 크기 제한 미들웨어 규칙: (RECORD_MAX_SIZE_MB, 413 고정 문구). 요청마다 설정을 다시 읽는다."""
    return float(config.RECORD_MAX_SIZE_MB), MSG_TOO_LARGE.format(mb=config.RECORD_MAX_SIZE_MB)


async def _save_capped(file: UploadFile, session_id: str) -> tuple[str, int]:
    """이미 받아 둔 업로드 사본을 RECORD_MAX_SIZE_MB+1바이트까지만 서버가 정한 이름으로 옮긴다.

    앞 4KB로 컨테이너를 판정하고 나머지는 1MB 단위로 스레드에서 복사한다(limits.save_upload_capped).
    서버가 '받는' 양의 제한은 이 함수가 아니라 limits.RequestBodyLimitMiddleware(413)가 맡는다.

    Returns:
        (저장 경로, 바이트 수).

    Raises:
        RecordingRejected: 비었음·형식 불일치·크기 초과·저장 실패. 이때 파일은 이미 지워져 있다.
    """
    max_bytes = int(config.RECORD_MAX_SIZE_MB) * 1024 * 1024
    upload_dir = infer_api.TMP_UPLOAD_DIR
    try:
        saved = await limits.save_upload_capped(
            file, max_bytes, detect_recording_container,
            lambda container: os.path.join(upload_dir, f"{session_id}_record.{container}"),
            min_bytes=_MIN_FILE_BYTES, head_bytes=_HEADER_BYTES)
    except limits.UploadRejected as exc:
        message = _REJECT_MESSAGES.get(exc.reason, lambda: MSG_UNREADABLE)()
        raise RecordingRejected(message, exc.detail) from None
    return saved.path, saved.size


@router.post("/record", response_model=RecordStartResponse)
async def start_record(background_tasks: BackgroundTasks,
                       file: UploadFile = File(...)) -> RecordStartResponse:
    """웹캠 녹화를 받아 발화 전체 인식 Job을 등록한다 (결과 즉시 반환 금지).

    Args:
        background_tasks: FastAPI 백그라운드 태스크 큐.
        file: 멀티파트 'file' 필드의 녹화 파일.

    Returns:
        {job_id, session_id, status:'processing', duration_sec}.

    Raises:
        HTTPException: 접수 거부 400(고정 문구), compact 백엔드 503, 같은 세션 처리 중 409.
    """
    if infer_api.MODEL_BACKEND != "olkavs":
        raise HTTPException(status_code=503, detail=MSG_BACKEND_UNSUPPORTED)
    session_id = str(uuid.uuid4())
    path: str | None = None
    try:
        path, size = await _save_capped(file, session_id)
        duration = await asyncio.to_thread(measure_recording_seconds, path, config.RECORD_MAX_SECONDS,
                                           config.RECORD_MAX_FRAMES, config.RECORD_PROBE_TIMEOUT_SEC)
        if duration < config.RECORD_MIN_SECONDS:
            raise RecordingRejected(MSG_TOO_SHORT.format(sec=config.RECORD_MIN_SECONDS), f"{duration:.3f}초")
        if not (duration < config.RECORD_MAX_SECONDS):
            raise RecordingRejected(MSG_TOO_LONG.format(sec=config.RECORD_MAX_SECONDS), f"{duration:.3f}초")
    except RecordingRejected as exc:
        _safe_remove(path)
        logger.info("녹화 접수 거부: %s (%s)", exc.user_message, exc.detail)
        raise HTTPException(status_code=400, detail=exc.user_message) from None
    except Exception as exc:  # noqa: BLE001 - 예상 밖 오류도 경로·원문 없이 400
        _safe_remove(path)
        logger.exception("녹화 접수 중 예기치 않은 오류: %s", type(exc).__name__)
        raise HTTPException(status_code=400, detail=MSG_UNREADABLE) from None
    finally:
        try:
            await file.close()
        except Exception:  # noqa: BLE001 - 업로드 임시 파일 정리 실패는 무시
            pass
    logger.info("녹화 접수: %s %d바이트 %.3f초", os.path.basename(path), size, duration)
    try:
        start = infer_api.start_background_job(session_id, path, background_tasks,
                                               mode="record", video_seconds=duration)
    except Exception:
        _safe_remove(path)
        raise
    return RecordStartResponse(job_id=start.job_id, session_id=start.session_id,
                               status=start.status, duration_sec=round(float(duration), 3))
