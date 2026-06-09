"""업로드 라우트.

POST /api/upload — 영상 파일 업로드, python-magic MIME 2중 검증, 크기/길이 제약 확인,
임시 디렉토리 저장 후 session_id 발급.
"""

from __future__ import annotations

from fastapi import APIRouter, UploadFile

router = APIRouter()


def validate_mime(file_bytes: bytes, filename: str) -> bool:
    """python-magic으로 MIME 타입을 2중 검증한다.

    확장자만 신뢰하지 말고 실제 매직 넘버로 video/* 여부를 확인한다.

    Args:
        file_bytes: 업로드 파일의 선두 바이트.
        filename: 원본 파일명 (확장자 교차 검증용).

    Returns:
        허용된 영상 MIME이면 True.
    """
    pass


async def upload_video(file: UploadFile) -> dict:
    """영상 업로드 핸들러.

    MIME/크기/길이 검증 후 TMP_UPLOAD_DIR에 저장하고 session_id를 발급한다.

    Args:
        file: 멀티파트 업로드 파일.

    Returns:
        {session_id, filename, status} 형태의 응답 딕셔너리.
    """
    pass
