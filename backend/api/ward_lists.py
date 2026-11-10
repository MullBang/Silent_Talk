"""병실 '내 목록'(사용자가 직접 만든 명령 목록) 관리 라우트 (services/ward_lists.py).

GET    /api/ward/lists          → {lists:[{id, title, count, emergency_count, updated, sha256}]}
GET    /api/ward/lists/{id}     → 저장된 목록 {schema, id, title, created, updated, commands, sha256}
                                  + checks{errors, warnings}
PUT    /api/ward/lists/{id}     → 본문 {title, commands:[{text, display, category, emergency, cancel}]}을 검사해 저장.
                                  오류가 있으면 400 {detail, errors, warnings}. 저장하면 저장한 목록 + checks + warnings.
POST   /api/ward/lists/check    → 같은 본문을 검사만 한다 {errors, warnings, normalized, stats} (저장하지 않음).
DELETE /api/ward/lists/{id}     → {deleted:true}. 그 목록의 '내 입 모양 등록'도 함께 지운다.

- 목록 ID는 등록 이름과 같은 규칙(영문·숫자·_·- 1~32자, Windows 장치 이름 제외). 그 밖은 400 고정 문구.
  대소문자만 다른 목록이 있으면 409, 없는 목록은 404, 목록이 이미 20개면 새 목록은 400이다.
- PUT 저장 조건(선택 헤더): If-None-Match: * 이면 그 ID의 목록이 없을 때만 만들고, If-Match: <sha256>(따옴표 있어도 됨)이면
  지금 저장된 목록의 sha256이 같을 때만 바꾼다. 맞지 않으면 409(고정 문구)이고 아무것도 쓰지 않는다. 두 헤더를 함께 보내거나
  형식이 틀리면 400이다. 헤더가 없으면 조건 없이 저장한다(없으면 만들고 있으면 바꿈). 화면은 새 목록·고치기에 늘 보낸다.
- 상태를 바꾸는 요청(PUT·DELETE)은 Origin 헤더가 CORS 화이트리스트 밖이면 403이다(녹화 등록 필드와 같은 규칙,
  api/record.origin_allowed). 검사(check)는 아무것도 바꾸지 않으므로 Origin을 보지 않는다.
- PUT·check 본문은 JSON 객체이고 64KB(ward_lists.MAX_BODY_BYTES)를 넘으면 413이다. 받는 즉시 세어서 끊는다.
- 사용자에게 보내는 문구는 고정 한국어 문구와 검사기 문구뿐이다. 경로·예외 원문은 서버 로그에만 남긴다.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

_BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

from api.record import origin_allowed  # noqa: E402
from schemas.models import (  # noqa: E402
    WardListCheckResult,
    WardListDeleted,
    WardListDetail,
    WardListIndex,
    WardListSaved,
)
from services import ward_lists as wl  # noqa: E402

router = APIRouter()
logger = logging.getLogger(__name__)

MSG_ORIGIN = "허용하지 않은 화면 주소에서 보낸 요청이에요. 이 프로그램의 화면에서 다시 해 주세요."
MSG_TOO_LARGE = f"요청이 너무 커요 ({wl.MAX_BODY_BYTES // 1024}KB 이하만 받아요)."
MSG_BAD_JSON = "요청 내용을 읽지 못했어요. JSON 형식으로 보내 주세요."
MSG_BAD_PRECONDITION = "저장 조건(If-Match·If-None-Match) 형식이 올바르지 않아요."

_SHA_RE = re.compile(r"^[0-9a-f]{64}$")


def _http_error(exc: wl.ListError) -> HTTPException:
    """저장소 오류 → HTTP 오류 (고정 문구만, 상세는 서버 로그)."""
    if isinstance(exc, (wl.InvalidListId, wl.ListLimit)):
        return HTTPException(status_code=400, detail=exc.user_message)
    if isinstance(exc, wl.ListNotFound):
        return HTTPException(status_code=404, detail=exc.user_message)
    if isinstance(exc, (wl.ListConflict, wl.ListPrecondition)):
        if isinstance(exc, wl.ListPrecondition):
            logger.info("내 목록 저장 조건 불일치: %s", exc.detail or exc)
        return HTTPException(status_code=409, detail=exc.user_message)
    logger.error("내 목록 저장소 오류: %s", exc.detail or exc)
    return HTTPException(status_code=500, detail=exc.user_message)


def _check_origin(request: Request) -> None:
    """상태를 바꾸는 요청의 Origin 확인.

    Raises:
        HTTPException: 화이트리스트 밖 Origin(403).
    """
    origin = request.headers.get("origin")
    if not origin_allowed(origin):
        logger.warning("내 목록 요청 거부: 허용하지 않은 Origin %r", origin)
        raise HTTPException(status_code=403, detail=MSG_ORIGIN)


def _save_expect(request: Request) -> str | None:
    """PUT의 저장 조건 헤더를 읽는다.

    Returns:
        None(조건 없음), wl.EXPECT_NEW(If-None-Match: *), 또는 SHA-256 소문자 문자열(If-Match).

    Raises:
        HTTPException: 두 헤더를 함께 보냈거나 형식이 틀리면 400.
    """
    if_none = request.headers.get("if-none-match")
    if_match = request.headers.get("if-match")
    if if_none is not None and if_match is not None:
        raise HTTPException(status_code=400, detail=MSG_BAD_PRECONDITION)
    if if_none is not None:
        if if_none.strip() != "*":
            raise HTTPException(status_code=400, detail=MSG_BAD_PRECONDITION)
        return wl.EXPECT_NEW
    if if_match is not None:
        value = if_match.strip()
        if value.startswith("W/"):
            value = value[2:]
        value = value.strip().strip('"').lower()
        if not _SHA_RE.match(value):
            raise HTTPException(status_code=400, detail=MSG_BAD_PRECONDITION)
        return value
    return None


async def _read_json_body(request: Request) -> Any:
    """요청 본문을 상한(MAX_BODY_BYTES)까지만 받아 JSON으로 읽는다.

    Content-Length가 상한을 넘으면 읽지 않고, 적혀 있지 않거나 작게 적힌 본문은 받는 바이트를 세다가 넘는 순간 멈춘다.

    Raises:
        HTTPException: 상한 초과 413, JSON이 아님 400.
    """
    limit = wl.MAX_BODY_BYTES
    declared = request.headers.get("content-length")
    try:
        if declared is not None and int(declared.strip()) > limit:
            raise HTTPException(status_code=413, detail=MSG_TOO_LARGE)
    except ValueError:
        pass  # 숫자가 아니면 받는 바이트로만 판정한다
    chunks: list[bytes] = []
    total = 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > limit:
            raise HTTPException(status_code=413, detail=MSG_TOO_LARGE)
        chunks.append(chunk)
    try:
        return json.loads(b"".join(chunks).decode("utf-8"))
    except (UnicodeDecodeError, ValueError, RecursionError):
        raise HTTPException(status_code=400, detail=MSG_BAD_JSON) from None


@router.get("/ward/lists", response_model=WardListIndex)
async def list_lists() -> WardListIndex:
    """저장된 내 목록 요약 (ID순).

    Raises:
        HTTPException: 저장 폴더를 읽지 못하면 500.
    """
    try:
        return WardListIndex(lists=await asyncio.to_thread(wl.get_store().summaries))
    except wl.ListError as exc:
        raise _http_error(exc) from None


@router.post("/ward/lists/check", response_model=WardListCheckResult)
async def check_list(request: Request) -> WardListCheckResult:
    """목록 본문을 검사만 한다(저장하지 않는다). 오류가 있어도 200이고 errors에 담는다.

    Raises:
        HTTPException: 본문 상한 초과 413, JSON이 아님 400.
    """
    body = await _read_json_body(request)
    return WardListCheckResult(**wl.check_list(body))


@router.get("/ward/lists/{list_id}", response_model=WardListDetail)
async def get_list(list_id: str) -> WardListDetail:
    """저장된 목록 하나와 지금 검사기로 본 checks.

    Raises:
        HTTPException: ID 규칙 위반 400, 없음 404, 대소문자만 다른 목록 409, 파일 손상 500.
    """
    try:
        return WardListDetail.model_validate(await asyncio.to_thread(wl.get_store().detail, list_id))
    except wl.ListError as exc:
        raise _http_error(exc) from None


@router.put("/ward/lists/{list_id}", response_model=WardListSaved)
async def put_list(list_id: str, request: Request) -> Any:
    """목록을 검사하고 저장한다(있으면 바꾼다). 명령 ID는 서버가 U01부터 순서대로 붙인다.

    Returns:
        저장한 목록 + checks{errors:[], warnings} + warnings. 검사 오류면 400 {detail, errors, warnings}.

    Raises:
        HTTPException: 허용하지 않은 Origin 403, ID 규칙 위반·목록 수 상한·조건 헤더 형식 400, 본문 413/400,
            대소문자 충돌·저장 조건 불일치(If-None-Match: * 인데 이미 있음, If-Match sha256이 다름·목록 없음) 409, 저장 실패 500.
    """
    _check_origin(request)
    try:
        wl.validate_list_id(list_id)
    except wl.ListError as exc:
        raise _http_error(exc) from None
    expect = _save_expect(request)
    body = await _read_json_body(request)
    try:
        doc, warnings = await asyncio.to_thread(wl.get_store().save, list_id, body, expect)
    except wl.ListInvalid as exc:
        return JSONResponse(status_code=400, content={"detail": exc.user_message, "errors": exc.errors,
                                                      "warnings": exc.warnings})
    except wl.ListError as exc:
        raise _http_error(exc) from None
    return WardListSaved.model_validate({**doc, "checks": {"errors": [], "warnings": warnings}, "warnings": warnings})


@router.delete("/ward/lists/{list_id}", response_model=WardListDeleted)
async def delete_list(list_id: str, request: Request) -> WardListDeleted:
    """목록과 그 목록의 '내 입 모양 등록'을 지운다.

    Raises:
        HTTPException: 허용하지 않은 Origin 403, ID 규칙 위반 400, 없음 404, 대소문자만 다른 목록 409, 삭제 실패 500.
    """
    _check_origin(request)
    try:
        await asyncio.to_thread(wl.get_store().delete, list_id)
    except wl.ListError as exc:
        raise _http_error(exc) from None
    return WardListDeleted(deleted=True)
