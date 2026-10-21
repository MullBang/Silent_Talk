"""업로드 라우트.

POST /api/upload — 영상 업로드, python-magic MIME 2중 검증(확장자 + 매직넘버),
크기/길이 제약 확인, 임시 디렉토리 저장 후 session_id 발급.

- 요청 본문이 MAX_UPLOAD_SIZE_MB(+멀티파트 여유 1MB)를 넘으면 limits.RequestBodyLimitMiddleware가 받는 즉시 413.
- 파일 전체를 메모리에 올리지 않는다: 앞 4KB로 MIME을 판정하고 1MB 단위로 스레드에서 저장, 상한+1바이트에서 멈춘다.
- 파일 쓰기·삭제와 길이 검사(get_video_duration_sec)는 asyncio.to_thread로 이벤트 루프 밖에서 한다.
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import sys
import uuid

from fastapi import APIRouter, File, HTTPException, UploadFile

# backend 루트를 import 경로에 부트스트랩
_BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

from config import MAX_DURATION_SEC, MAX_UPLOAD_SIZE_MB, TMP_UPLOAD_DIR  # noqa: E402
from api import limits  # noqa: E402
from preprocessing.resampler import get_video_duration_sec  # noqa: E402
from schemas.models import UploadResponse  # noqa: E402

router = APIRouter()
logger = logging.getLogger(__name__)

# 사용자에게 보내는 고정 문구. 서버 경로·예외 원문·감지된 내부 값은 넣지 않는다(상세는 서버 로그에만).
MSG_UNSUPPORTED_EXT = "지원하지 않는 파일 형식입니다 (MP4/AVI/MOV)."
MSG_MIME_MISMATCH = "파일 내용이 MP4/AVI/MOV 영상 형식과 일치하지 않습니다."
MSG_SAVE_FAILED = "파일을 저장하지 못했습니다. 잠시 후 다시 시도해 주세요."
MSG_DURATION_EXCEEDED = f"영상 길이가 허용 최대치({MAX_DURATION_SEC}초)를 초과했습니다."
MSG_VIDEO_UNREADABLE = "영상을 분석할 수 없습니다. 다른 파일로 다시 시도해 주세요."
MSG_TOO_LARGE = "{mb}MB 이하만 업로드 가능합니다."

# 허용 MIME 타입 (설계문서 F-MVP-01: MP4/AVI/MOV)
_ALLOWED_MIME = {"video/mp4", "video/x-msvideo", "video/quicktime"}
# 허용 확장자 (확장자 1차 검증)
_ALLOWED_EXT = {".mp4", ".avi", ".mov"}

# python-magic을 우선 사용하되, libmagic 네이티브 라이브러리 부재(주로 Windows)
# 시에는 파일 시그니처(매직바이트) 기반 폴백으로 동일한 2중 검증을 수행한다.
#
# 주의: libmagic DLL 부재 시 `import magic`이 native access violation을 일으키며,
# cv2(OpenCV)가 이미 로드된 서버 프로세스에서는 이 크래시가 try/except로 잡히지
# 않는 하드 segfault로 번져 프로세스를 죽인다. 따라서 본 프로세스에서 바로
# import하지 않고, 격리된 서브프로세스에서 import 가능 여부를 먼저 검증한 뒤
# 안전이 확인된 경우에만 in-process로 import한다.
def _probe_magic_safe() -> bool:
    """별도 프로세스에서 python-magic import 가능 여부를 안전하게 검증한다.

    Returns:
        서브프로세스에서 magic import + from_buffer 호출이 정상 종료하면 True.
        libmagic 부재로 인한 비정상 종료(OSError/segfault)면 False.
    """
    try:
        result = subprocess.run(
            [sys.executable, "-c",
             "import magic; magic.from_buffer(b'\\x00' * 16, mime=True)"],
            capture_output=True,
            timeout=15,
        )
        return result.returncode == 0
    except Exception:  # noqa: BLE001 - 프로브 실패는 곧 폴백
        return False


if _probe_magic_safe():
    try:
        import magic as _magic

        _magic.from_buffer(b"\x00" * 16, mime=True)
        _MAGIC_AVAILABLE = True
    except Exception:  # noqa: BLE001 - 프로브 통과 후에도 방어적 폴백
        _magic = None
        _MAGIC_AVAILABLE = False
else:
    _magic = None
    _MAGIC_AVAILABLE = False


def _sniff_video_mime(content: bytes) -> str:
    """파일 시그니처(매직바이트)로 영상 MIME을 추정한다 (libmagic 폴백).

    - MP4/MOV: 4~8바이트가 'ftyp' (ISO BMFF). 브랜드가 'qt'이면 quicktime.
    - AVI: 'RIFF'....'AVI '.

    Args:
        content: 파일 선두 바이트.

    Returns:
        추정 MIME 문자열 (미상이면 'application/octet-stream').
    """
    if len(content) >= 12 and content[4:8] == b"ftyp":
        brand = content[8:12]
        if brand[:2] == b"qt":
            return "video/quicktime"
        return "video/mp4"
    if len(content) >= 12 and content[0:4] == b"RIFF" and content[8:12] == b"AVI ":
        return "video/x-msvideo"
    return "application/octet-stream"


def _detect_mime(content: bytes) -> str:
    """업로드 내용의 MIME 타입을 감지한다 (python-magic 우선, 시그니처 폴백).

    Args:
        content: 파일 선두 바이트.

    Returns:
        감지된 MIME 문자열.
    """
    if _MAGIC_AVAILABLE:
        return _magic.from_buffer(content[:4096], mime=True)
    return _sniff_video_mime(content)


def _allowed_mime_or_none(head: bytes) -> str | None:
    """앞부분 바이트의 MIME이 허용 목록(MP4/AVI/MOV)이면 그 값을, 아니면 None을 돌려준다."""
    try:
        mime = _detect_mime(head)
    except Exception as exc:  # noqa: BLE001 - 판정 실패는 형식 불일치로 본다
        logger.info("업로드 MIME 판정 실패: %s", type(exc).__name__)
        return None
    if mime not in _ALLOWED_MIME:
        logger.info("업로드 MIME 불일치: %s", mime)
        return None
    return mime


def upload_body_limit() -> tuple[float, str]:
    """요청 본문 크기 제한 미들웨어 규칙: (MAX_UPLOAD_SIZE_MB, 413 고정 문구). 요청마다 다시 읽는다."""
    return float(MAX_UPLOAD_SIZE_MB), MSG_TOO_LARGE.format(mb=MAX_UPLOAD_SIZE_MB)


@router.post("/upload", response_model=UploadResponse)
async def upload_video(file: UploadFile = File(...)) -> UploadResponse:
    """영상 업로드 핸들러.

    처리: 확장자 + python-magic MIME 2중 검증(앞 4KB) → 상한까지만 1MB 단위 임시 저장(크기 검증) →
    길이 검증 → session_id 발급.

    Args:
        file: 멀티파트 업로드 파일.

    Returns:
        UploadResponse {session_id, file_name, duration_sec}.

    Raises:
        HTTPException: 미지원 포맷/MIME 불일치/크기 초과/길이 초과 시 400, 저장 실패 500.
    """
    file_name = os.path.basename(file.filename or "")
    ext = os.path.splitext(file_name)[1].lower()

    # ① 확장자 1차 검증
    if ext not in _ALLOWED_EXT:
        raise HTTPException(status_code=400, detail=MSG_UNSUPPORTED_EXT)

    # ②③④ MIME 2중 검증(앞 4KB, python-magic 우선·시그니처 폴백) → 1MB 단위 저장(상한+1바이트에서 멈춤).
    # 파일 전체를 메모리에 올리지 않고, 파일 쓰기는 스레드에서 한다. 저장 경로: {TMP_UPLOAD_DIR}/{session_id}_{filename}
    session_id = str(uuid.uuid4())
    upload_dir = TMP_UPLOAD_DIR
    max_bytes = int(MAX_UPLOAD_SIZE_MB * 1024 * 1024)
    try:
        saved = await limits.save_upload_capped(
            file, max_bytes, _allowed_mime_or_none,
            lambda _mime: os.path.join(upload_dir, f"{session_id}_{file_name}"))
    except limits.UploadRejected as exc:
        if exc.reason == limits.REJECT_TOO_LARGE:
            raise HTTPException(status_code=400, detail=MSG_TOO_LARGE.format(mb=MAX_UPLOAD_SIZE_MB)) from None
        if exc.reason == limits.REJECT_SAVE_FAILED:
            logger.error("업로드 파일 저장 실패: %s", exc.detail)
            raise HTTPException(status_code=500, detail=MSG_SAVE_FAILED) from None
        raise HTTPException(status_code=400, detail=MSG_MIME_MISMATCH) from None  # 판정 값은 위에서 로그
    save_path = saved.path

    # ⑤ 길이 검증 (MAX_DURATION_SEC). 초과/오류 시 임시 파일 제거 후 400.
    try:
        duration_sec = await asyncio.to_thread(get_video_duration_sec, save_path)
    except ValueError as exc:
        # get_video_duration_sec 문구에는 서버 경로가 들어갈 수 있다 → 길이 초과만 구분해 고정 문구로 바꾼다.
        await asyncio.to_thread(_safe_remove, save_path)
        logger.info("업로드 영상 길이 검사 거부: %s", exc)
        detail = MSG_DURATION_EXCEEDED if "초과" in str(exc) else MSG_VIDEO_UNREADABLE
        raise HTTPException(status_code=400, detail=detail) from None
    except Exception as exc:  # noqa: BLE001
        await asyncio.to_thread(_safe_remove, save_path)
        logger.info("업로드 영상 분석 실패: %s: %s", type(exc).__name__, exc)
        raise HTTPException(status_code=400, detail=MSG_VIDEO_UNREADABLE) from None

    # ⑥ 응답
    return UploadResponse(
        session_id=session_id, file_name=file_name, duration_sec=duration_sec
    )


def _safe_remove(path: str) -> None:
    """임시 파일을 조용히 제거한다 (없으면 무시, 실패는 서버 로그)."""
    limits.safe_remove(path)
