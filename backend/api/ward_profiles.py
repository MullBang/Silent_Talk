"""병실 '내 입 모양 등록' 관리 라우트.

GET    /api/ward/profiles            → {profiles:[{profile, count, complete, status, updated}]}
GET    /api/ward/profiles/{profile}  → {profile, count, complete, status, templates:{명령ID:{created, text_top, text_top_matches}}}
DELETE /api/ward/profiles/{profile}  → {deleted:true}

- 등록 이름은 영문·숫자·_·- 1~32자만 받는다(그 밖은 400 고정 문구). 경로는 받지도 내보내지도 않는다.
- 없는 등록의 GET은 status 'missing'(빈 templates)이고, DELETE는 404다.
- 템플릿 저장은 POST /api/record(phrase_set=ward + enroll_profile + enroll_command)로만 한다.
- stale 판정에 쓰는 가중치 SHA-256은 인식 워커가 떠 있을 때만 비교한다(기동 전에는 명령 목록만 비교).
- 세 경로 모두 ?list=<내 목록 ID>를 붙이면 그 내 목록의 등록 이름공간(services/ward_profiles.ListProfileStore)을 쓴다.
  없으면(보내지 않으면) 기본 목록 등록 그대로다. 목록 ID 규칙 위반·빈 값은 400, GET은 그 목록이 없으면 404다.
  DELETE는 목록이 없어도 남은 등록을 지울 수 있다(목록을 지우면 등록도 함께 지워지므로 보통은 404).
"""
from __future__ import annotations

import logging
import os
import sys

from fastapi import APIRouter, HTTPException, Query

_BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

import config  # noqa: E402
from schemas.models import WardProfileDeleted, WardProfileDetail, WardProfileList  # noqa: E402
from services import ward_lists as wl  # noqa: E402
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


def _list_store(list_id: str | None, need_list: bool = True) -> wp.ProfileStore:
    """?list가 없으면 기본 등록 저장소, 있으면 그 내 목록의 등록 이름공간.

    Args:
        list_id: ?list 값 (보내지 않았으면 None).
        need_list: True면 그 목록을 읽어 명령을 붙인다(없으면 404). False면 이름공간만 쓴다(지우기).

    Raises:
        HTTPException: 목록 ID 규칙 위반·빈 값 400, 목록 없음 404, 대소문자만 다른 목록 409, 목록 파일 손상 500.
    """
    if list_id is None:
        return wp.get_store()
    try:
        wl.validate_list_id(list_id)
        if not need_list:
            return wp.ListProfileStore(list_id)
        doc = wl.get_store().get(list_id)
        return wp.ListProfileStore(list_id, wp.list_commands(doc))
    except wl.ListError as exc:
        if isinstance(exc, (wl.InvalidListId, wl.ListLimit)):
            raise HTTPException(status_code=400, detail=exc.user_message) from None
        if isinstance(exc, wl.ListNotFound):
            raise HTTPException(status_code=404, detail=exc.user_message) from None
        if isinstance(exc, wl.ListConflict):
            raise HTTPException(status_code=409, detail=exc.user_message) from None
        logger.error("내 목록 읽기 오류: %s", exc.detail or exc)
        raise HTTPException(status_code=500, detail=exc.user_message) from None
    except (wp.ProfileError, ValueError) as exc:
        logger.error("내 목록 등록 저장소 오류: %s", getattr(exc, "detail", "") or exc)
        raise HTTPException(status_code=500, detail=wl.MSG_UNREADABLE) from None


@router.get("/ward/profiles", response_model=WardProfileList)
async def list_profiles(list_id: str | None = Query(None, alias="list")) -> WardProfileList:
    """저장된 등록 목록 (이름순). ?list=<내 목록 ID>면 그 목록의 등록.

    Raises:
        HTTPException: 저장 폴더를 읽지 못하면 500, 목록 ID 규칙 위반 400, 없는 목록 404.
    """
    store = _list_store(list_id)
    try:
        return WardProfileList(profiles=store.list_profiles(_current_checkpoint()))
    except wp.ProfileError as exc:
        raise _http_error(exc) from None


@router.get("/ward/profiles/{profile}", response_model=WardProfileDetail)
async def get_profile(profile: str, list_id: str | None = Query(None, alias="list")) -> WardProfileDetail:
    """등록 하나의 상태와 명령별 템플릿 요약. ?list=<내 목록 ID>면 그 목록의 등록.

    Raises:
        HTTPException: 이름·목록 ID 규칙 위반 400, 없는 목록 404, 대소문자만 다른 등록·목록 409, 파일 손상 500.
    """
    store = _list_store(list_id)
    try:
        return WardProfileDetail(**store.detail(profile, _current_checkpoint()))
    except wp.ProfileError as exc:
        raise _http_error(exc) from None


@router.delete("/ward/profiles/{profile}", response_model=WardProfileDeleted)
async def delete_profile(profile: str, list_id: str | None = Query(None, alias="list")) -> WardProfileDeleted:
    """등록 폴더(템플릿 전체)를 지운다. ?list=<내 목록 ID>면 그 목록의 등록.

    Raises:
        HTTPException: 이름·목록 ID 규칙 위반 400, 없음 404, 대소문자만 다른 등록 409, 삭제 실패 500.
    """
    store = _list_store(list_id, need_list=False)
    try:
        store.delete(profile)
    except wp.ProfileError as exc:
        raise _http_error(exc) from None
    return WardProfileDeleted(deleted=True)
