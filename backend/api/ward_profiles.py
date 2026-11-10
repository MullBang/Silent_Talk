"""병실 '내 입 모양 등록' 관리 라우트.

GET    /api/ward/profiles            → {profiles:[{profile, count, complete, status, updated}]}
GET    /api/ward/profiles/{profile}  → {profile, count, complete, status, templates:{명령ID:{created, text_top, text_top_matches}}}
DELETE /api/ward/profiles/{profile}  → {deleted:true}

- 등록 이름은 영문·숫자·_·- 1~32자만 받는다(그 밖은 400 고정 문구). 경로는 받지도 내보내지도 않는다.
- 없는 등록의 GET은 status 'missing'(빈 templates)이고, DELETE는 404다.
- 템플릿 저장은 POST /api/record(phrase_set=ward + enroll_profile + enroll_command)로만 한다.
- stale 판정에 쓰는 가중치 SHA-256은 인식 워커가 떠 있을 때만 비교한다(기동 전에는 명령 목록만 비교).
"""
from __future__ import annotations

import logging
import os
import sys

from fastapi import APIRouter, HTTPException

_BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

import config  # noqa: E402
from schemas.models import WardProfileDeleted, WardProfileDetail, WardProfileList  # noqa: E402
from services import ward_profiles as wp  # noqa: E402

router = APIRouter()
logger = logging.getLogger(__name__)


def _current_checkpoint() -> str | None:
    """떠 있는 인식 워커의 가중치 SHA-256 (워커를 새로 띄우지 않는다). 모르면 None."""
    try:
        if config.MODEL_BACKEND != "olkavs":
            return None
        from services.olkavs_service import peek_service

        service = peek_service()
        info = getattr(getattr(service, "infer", None), "info", None) or {}
        value = info.get("checkpoint_sha256") if isinstance(info, dict) else None
        return value if isinstance(value, str) else None
    except Exception as exc:  # noqa: BLE001 - 상태 조회가 목록을 막지 않는다
        logger.warning("인식 가중치 SHA 조회 실패: %s", exc)
        return None


def _http_error(exc: wp.ProfileError) -> HTTPException:
    """저장소 오류 → HTTP 오류 (고정 문구만, 상세는 서버 로그)."""
    if isinstance(exc, wp.InvalidProfileId):
        return HTTPException(status_code=400, detail=exc.user_message)
    if isinstance(exc, wp.ProfileNotFound):
        return HTTPException(status_code=404, detail=exc.user_message)
    if isinstance(exc, wp.ProfileConflict):
        return HTTPException(status_code=409, detail=exc.user_message)
    logger.error("등록 저장소 오류: %s", exc.detail or exc)
    return HTTPException(status_code=500, detail=exc.user_message)


@router.get("/ward/profiles", response_model=WardProfileList)
async def list_profiles() -> WardProfileList:
    """저장된 등록 목록 (이름순).

    Raises:
        HTTPException: 저장 폴더를 읽지 못하면 500.
    """
    try:
        return WardProfileList(profiles=wp.get_store().list_profiles(_current_checkpoint()))
    except wp.ProfileError as exc:
        raise _http_error(exc) from None


@router.get("/ward/profiles/{profile}", response_model=WardProfileDetail)
async def get_profile(profile: str) -> WardProfileDetail:
    """등록 하나의 상태와 명령별 템플릿 요약.

    Raises:
        HTTPException: 이름 규칙 위반 400, 대소문자만 다른 등록 409, 등록 파일 손상 500.
    """
    try:
        return WardProfileDetail(**wp.get_store().detail(profile, _current_checkpoint()))
    except wp.ProfileError as exc:
        raise _http_error(exc) from None


@router.delete("/ward/profiles/{profile}", response_model=WardProfileDeleted)
async def delete_profile(profile: str) -> WardProfileDeleted:
    """등록 폴더(템플릿 전체)를 지운다.

    Raises:
        HTTPException: 이름 규칙 위반 400, 없음 404, 대소문자만 다른 등록 409, 삭제 실패 500.
    """
    try:
        wp.get_store().delete(profile)
    except wp.ProfileError as exc:
        raise _http_error(exc) from None
    return WardProfileDeleted(deleted=True)
