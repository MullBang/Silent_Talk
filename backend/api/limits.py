"""업로드 크기 제한 공용 모듈 (6단계 보완).

1. RequestBodyLimitMiddleware — 순수 ASGI 미들웨어. 지정한 POST 경로(/api/record, /api/upload)의 요청 본문이
   상한(파일 상한 + 멀티파트 여유분)을 넘으면 413과 고정 한국어 {"detail": ...}로 거절한다.
   - Content-Length가 상한을 넘으면 본문을 한 바이트도 읽지 않고 바로 거절한다.
   - Content-Length가 없거나(chunked) 실제보다 작게 적힌 본문은 receive()를 감싸 받은 바이트를 세고,
     상한을 넘는 순간 멈춘다(Starlette 멀티파트 파서가 디스크에 임시 저장하기 전). 응답이 아직 시작되지
     않았으면 413을 보낸다.
   - CORS 미들웨어보다 안쪽에 등록해야 413 응답에도 화이트리스트 Origin의 CORS 헤더가 붙는다.
2. save_upload_capped — 업로드 파일을 앞부분(형식 판정용)만 먼저 읽고, 나머지는 1MB 단위로 디스크에 옮긴다.
   상한+1바이트에서 멈추고, 거절하면 부분 파일을 지운다. 파일 쓰기는 asyncio.to_thread로 이벤트 루프 밖에서 한다.
   (Starlette가 이미 받아 둔 임시 사본을 옮기는 단계다. 받는 양 자체의 제한은 1번 미들웨어가 맡는다.)
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Mapping, MutableMapping, Tuple, TypeVar

from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)

Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]

#: 경로별 규칙: () -> (파일 상한 MB, 413 사용자 문구). 요청마다 다시 읽어 설정·테스트 변경을 따른다.
LimitRule = Callable[[], Tuple[float, str]]

MSG_BODY_TOO_LARGE = "요청이 너무 큽니다."
"""규칙에서 문구를 얻지 못했을 때 쓰는 기본 413 문구."""

HEAD_BYTES = 4096
"""형식 판정(시그니처·python-magic)에 쓰는 파일 앞부분 크기."""

COPY_CHUNK_BYTES = 1024 * 1024
"""업로드를 디스크로 옮길 때 한 번에 읽고 쓰는 크기."""

T = TypeVar("T")


# ---------------------------------------------------------------------------
# 1. 요청 본문 크기 제한 미들웨어
# ---------------------------------------------------------------------------

class RequestBodyTooLarge(StarletteHTTPException):
    """받은 본문이 상한을 넘었음. HTTPException이라 FastAPI 본문 파싱이 400으로 바꾸지 않고 413으로 내보낸다."""

    def __init__(self, detail: str) -> None:
        super().__init__(status_code=413, detail=detail)


def _route_path(scope: Scope) -> str:
    """root_path를 뺀 요청 경로 (끝의 '/'는 지운다)."""
    try:
        path = str(scope.get("path") or "")
        root = str(scope.get("root_path") or "")
        if root and path.startswith(root):
            path = path[len(root):]
        return path.rstrip("/") or "/"
    except Exception:  # noqa: BLE001 - 경로를 못 읽으면 제한 대상이 아니다
        return ""


def _content_length(scope: Scope) -> int | None:
    """Content-Length 헤더 값 (없거나 잘못됐으면 None — 이때는 받은 바이트 수로만 판정한다)."""
    try:
        values = [v for k, v in scope.get("headers") or [] if k.lower() == b"content-length"]
        if len(values) != 1:
            return None
        text = values[0].decode("latin-1").strip()
        if not text.isdigit():
            return None
        return int(text)
    except Exception:  # noqa: BLE001 - 헤더 해석 실패는 본문 계수로 대신한다
        return None


class RequestBodyLimitMiddleware:
    """지정 경로의 POST 요청 본문 크기를 제한하는 순수 ASGI 미들웨어."""

    def __init__(self, app: ASGIApp, rules: Mapping[str, LimitRule], overhead_bytes: int = 1024 * 1024) -> None:
        """
        Args:
            app: 안쪽 ASGI 앱.
            rules: {경로: 규칙}. 경로는 '/api/record'처럼 root_path·끝 '/' 없이 쓴다.
            overhead_bytes: 멀티파트 경계·헤더 등 파일 밖 바이트 여유분.
        """
        self.app = app
        self.rules = {path.rstrip("/") or "/": rule for path, rule in rules.items()}
        self.overhead_bytes = max(0, int(overhead_bytes))

    def _limit_for(self, scope: Scope) -> tuple[int, str] | None:
        """이 요청에 적용할 (본문 상한 바이트, 413 문구). 대상이 아니면 None."""
        if scope.get("type") != "http" or str(scope.get("method", "")).upper() != "POST":
            return None
        rule = self.rules.get(_route_path(scope))
        if rule is None:
            return None
        try:
            limit_mb, message = rule()
            limit_mb = float(limit_mb)
            if not math.isfinite(limit_mb) or limit_mb < 0:
                raise ValueError(f"잘못된 상한 {limit_mb!r}")
            return int(limit_mb * 1024 * 1024) + self.overhead_bytes, str(message or MSG_BODY_TOO_LARGE)
        except Exception as exc:  # noqa: BLE001 - 규칙 오류로 요청을 막지 않는다(경로별 검사가 뒤에 또 있다)
            logger.error("요청 크기 규칙 오류(%s): %s", _route_path(scope), exc)
            return None

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """ASGI 진입점."""
        limit = self._limit_for(scope)
        if limit is None:
            await self.app(scope, receive, send)
            return
        max_body, message = limit
        declared = _content_length(scope)
        if declared is not None and declared > max_body:
            logger.info("요청 본문 거절(Content-Length %d > %d): %s", declared, max_body, _route_path(scope))
            await JSONResponse({"detail": message}, status_code=413)(scope, receive, send)
            return

        received = 0
        response_started = False

        async def limited_receive() -> Message:
            nonlocal received
            msg = await receive()
            if msg.get("type") == "http.request":
                received += len(msg.get("body", b"") or b"")
                if received > max_body:
                    logger.info("요청 본문 거절(받은 %d바이트 > %d): %s", received, max_body, _route_path(scope))
                    raise RequestBodyTooLarge(message)
            return msg

        async def tracking_send(msg: Message) -> None:
            nonlocal response_started
            if msg.get("type") == "http.response.start":
                response_started = True
            await send(msg)

        try:
            await self.app(scope, limited_receive, tracking_send)
        except RequestBodyTooLarge:
            # 보통은 안쪽 예외 처리기가 413으로 바꾼다. 거기까지 오지 못하고 새어 나온 경우의 대비.
            if response_started:
                raise
            await JSONResponse({"detail": message}, status_code=413)(scope, receive, send)


# ---------------------------------------------------------------------------
# 2. 업로드 파일 저장 (앞부분 판정 → 1MB 단위 복사, 상한+1에서 멈춤)
# ---------------------------------------------------------------------------

REJECT_EMPTY = "empty"
REJECT_UNSUPPORTED = "unsupported"
REJECT_TOO_LARGE = "too_large"
REJECT_SAVE_FAILED = "save_failed"


class UploadRejected(ValueError):
    """업로드를 저장하지 않은 사유. reason은 REJECT_* 중 하나, detail은 서버 로그용 상세."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(reason)
        self.reason = reason
        self.detail = detail


@dataclass(frozen=True)
class SavedUpload:
    """저장된 업로드."""

    path: str
    size: int
    kind: Any


def safe_remove(path: str | None) -> None:
    """파일을 조용히 지운다 (없으면 무시, 실패는 서버 로그)."""
    if not path:
        return
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
    except OSError as exc:
        logger.error("업로드 임시 파일 삭제 실패: %s (%s)", path, exc)


def _copy_capped(source: Any, head: bytes, path: str, max_bytes: int) -> int:
    """(스레드에서 실행) head를 쓰고 source의 나머지를 상한+1바이트까지 path로 옮긴다.

    Returns:
        쓴 바이트 수 (max_bytes를 넘으면 그 시점에서 멈춘 값).

    Raises:
        OSError: 폴더 생성·열기·쓰기 실패.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    total = len(head)
    with open(path, "xb") as handle:
        handle.write(head)
        while total <= max_bytes:
            chunk = source.read(min(COPY_CHUNK_BYTES, max_bytes + 1 - total))
            if not chunk:
                break
            handle.write(chunk)
            total += len(chunk)
    return total


async def save_upload_capped(file: Any, max_bytes: int, detect: Callable[[bytes], T | None],
                             build_path: Callable[[T], str], min_bytes: int = 0,
                             head_bytes: int = HEAD_BYTES) -> SavedUpload:
    """업로드를 크기 상한까지만 옮겨 서버가 정한 경로에 저장한다.

    앞 head_bytes만 먼저 읽어 detect로 형식을 정하고(None이면 거절), 나머지는 1MB 단위로 스레드에서 복사한다.
    상한+1바이트까지만 옮기므로 상한 초과 파일을 끝까지 복사하지 않는다. 거절하면 부분 파일을 지운다.
    본문 수신 자체의 상한은 RequestBodyLimitMiddleware가 맡는다(이 함수는 이미 받아 둔 임시 사본을 옮긴다).

    Args:
        file: Starlette/FastAPI UploadFile (read(size) 코루틴과 .file 동기 파일 객체).
        max_bytes: 허용 최대 바이트.
        detect: 앞부분 바이트 → 형식 값(예: 'webm', MIME) 또는 지원하지 않으면 None.
        build_path: 형식 값 → 저장 경로.
        min_bytes: 앞부분이 이보다 짧으면 REJECT_EMPTY.
        head_bytes: 형식 판정에 읽을 앞부분 크기.

    Returns:
        SavedUpload(path, size, kind).

    Raises:
        UploadRejected: 비었음·형식 불일치·크기 초과·저장 실패 (부분 파일은 이미 지워져 있다).
    """
    try:
        head = await file.read(head_bytes)
    except Exception as exc:  # noqa: BLE001 - 업로드 읽기 실패는 저장 실패로 본다
        raise UploadRejected(REJECT_SAVE_FAILED, f"업로드 읽기 실패: {type(exc).__name__}: {exc}") from exc
    if len(head) < min_bytes:
        raise UploadRejected(REJECT_EMPTY, f"{len(head)}바이트")
    kind = detect(head)
    if kind is None:
        raise UploadRejected(REJECT_UNSUPPORTED, f"시그니처 {head[:12].hex()}")
    path = build_path(kind)
    try:
        total = await asyncio.to_thread(_copy_capped, file.file, head, path, int(max_bytes))
    except OSError as exc:
        if not isinstance(exc, FileExistsError):
            await asyncio.to_thread(safe_remove, path)
        raise UploadRejected(REJECT_SAVE_FAILED, f"저장 실패: {exc}") from exc
    except Exception as exc:  # noqa: BLE001 - 읽기 오류 등도 저장 실패로 보고 부분 파일을 지운다
        await asyncio.to_thread(safe_remove, path)
        raise UploadRejected(REJECT_SAVE_FAILED, f"저장 실패: {type(exc).__name__}: {exc}") from exc
    if total > max_bytes:
        await asyncio.to_thread(safe_remove, path)
        raise UploadRejected(REJECT_TOO_LARGE, f"{total}바이트 > {max_bytes}")
    return SavedUpload(path=path, size=total, kind=kind)
