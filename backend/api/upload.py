"""업로드 라우트.

POST /api/upload — 영상 업로드, python-magic MIME 2중 검증(확장자 + 매직넘버),
크기/길이 제약 확인, 임시 디렉토리 저장 후 session_id 발급.
"""

from __future__ import annotations

import os
import sys
import uuid

from fastapi import APIRouter, File, HTTPException, UploadFile

# backend 루트를 import 경로에 부트스트랩
_BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

from config import MAX_UPLOAD_SIZE_MB, TMP_UPLOAD_DIR  # noqa: E402
from preprocessing.resampler import get_video_duration_sec  # noqa: E402
from schemas.models import UploadResponse  # noqa: E402

router = APIRouter()

# 허용 MIME 타입 (설계문서 F-MVP-01: MP4/AVI/MOV)
_ALLOWED_MIME = {"video/mp4", "video/x-msvideo", "video/quicktime"}
# 허용 확장자 (확장자 1차 검증)
_ALLOWED_EXT = {".mp4", ".avi", ".mov"}

# python-magic을 우선 사용하되, libmagic 네이티브 라이브러리 부재(주로 Windows)
# 시에는 파일 시그니처(매직바이트) 기반 폴백으로 동일한 2중 검증을 수행한다.
try:
    import magic as _magic

    _magic.from_buffer(b"\x00" * 16, mime=True)  # 실제 로드 가능 여부 확인
    _MAGIC_AVAILABLE = True
except Exception:  # noqa: BLE001 - libmagic 미설치/로드 실패 모두 폴백
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


@router.post("/upload", response_model=UploadResponse)
async def upload_video(file: UploadFile = File(...)) -> UploadResponse:
    """영상 업로드 핸들러.

    처리: 확장자 + python-magic MIME 2중 검증 → 크기 검증 → 임시 저장 →
    길이 검증 → session_id 발급.

    Args:
        file: 멀티파트 업로드 파일.

    Returns:
        UploadResponse {session_id, file_name, duration_sec}.

    Raises:
        HTTPException: 미지원 포맷/MIME 불일치/크기 초과/길이 초과 시 400.
    """
    file_name = os.path.basename(file.filename or "")
    ext = os.path.splitext(file_name)[1].lower()

    # ① 확장자 1차 검증
    if ext not in _ALLOWED_EXT:
        raise HTTPException(
            status_code=400, detail="지원하지 않는 파일 형식입니다 (MP4/AVI/MOV)."
        )

    # 파일 내용 수신
    content = await file.read()

    # ② MIME 2중 검증 (매직넘버 기반; python-magic 우선, 시그니처 폴백)
    mime = _detect_mime(content)
    if mime not in _ALLOWED_MIME:
        raise HTTPException(
            status_code=400,
            detail=f"파일 형식이 일치하지 않습니다 (감지된 형식: {mime}).",
        )

    # ③ 크기 검증
    if len(content) > MAX_UPLOAD_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=400, detail=f"{MAX_UPLOAD_SIZE_MB}MB 이하만 업로드 가능합니다."
        )

    # ④ 임시 저장: /tmp/uploads/{session_id}_{filename}
    session_id = str(uuid.uuid4())
    os.makedirs(TMP_UPLOAD_DIR, exist_ok=True)
    save_path = os.path.join(TMP_UPLOAD_DIR, f"{session_id}_{file_name}")
    try:
        with open(save_path, "wb") as f:
            f.write(content)
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"파일 저장 실패: {exc}")

    # ⑤ 길이 검증 (MAX_DURATION_SEC). 초과/오류 시 임시 파일 제거 후 400.
    try:
        duration_sec = get_video_duration_sec(save_path)
    except ValueError as exc:
        _safe_remove(save_path)
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        _safe_remove(save_path)
        raise HTTPException(status_code=400, detail=f"영상 분석 실패: {exc}")

    # ⑥ 응답
    return UploadResponse(
        session_id=session_id, file_name=file_name, duration_sec=duration_sec
    )


def _safe_remove(path: str) -> None:
    """임시 파일을 조용히 제거한다 (실패 무시)."""
    try:
        os.remove(path)
    except OSError:
        pass
