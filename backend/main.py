"""FastAPI 애플리케이션 진입점.

CORS 미들웨어(화이트리스트), 라우터 등록, 라이프사이클(startup/shutdown) 훅,
WebSocket placeholder를 구성한다. 실제 라우트 로직은 api/ 패키지에 위임한다.

실행: uvicorn backend.main:app --reload  (또는 backend/ 에서 uvicorn main:app)
"""

from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

# backend 루트를 import 경로에 부트스트랩
_BACKEND_ROOT = os.path.abspath(os.path.dirname(__file__))
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

from api.routes import get_api_router  # noqa: E402
from config import CORS_ORIGINS  # noqa: E402
from services.file_cleaner import (  # noqa: E402
    shutdown_cleanup_scheduler,
    start_cleanup_scheduler,
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """애플리케이션 수명주기 관리.

    startup: 임시 파일 정리 스케줄러 시작.
    shutdown: 스케줄러 종료.
    """
    start_cleanup_scheduler()
    try:
        yield
    finally:
        shutdown_cleanup_scheduler()


def create_app() -> FastAPI:
    """FastAPI 앱 인스턴스를 생성하고 미들웨어/라우터를 등록한다.

    CORS는 config.CORS_ORIGINS 화이트리스트만 허용한다 (와일드카드 금지).

    Returns:
        구성이 완료된 FastAPI 애플리케이션.
    """
    app = FastAPI(title="Silent Talk Lip Reading API", lifespan=lifespan)

    # CORS: 명시적 화이트리스트만 허용 (와일드카드 '*' 금지)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # /api 라우터 등록
    app.include_router(get_api_router())

    @app.get("/health", tags=["health"])
    async def health() -> dict:
        """헬스 체크 엔드포인트."""
        return {"status": "ok"}

    @app.websocket("/ws/stream")
    async def ws_stream(websocket: WebSocket) -> None:
        """WebSocket 실시간 스트리밍 (설계문서 4절, 2차 목표).

        Origin 검사·제어/바이너리 처리·추론은 services.ws_handler에 위임한다.
        """
        from services.ws_handler import handle_connection

        await handle_connection(websocket)

    return app


app: FastAPI = create_app()
"""ASGI 서버(uvicorn)가 참조하는 애플리케이션 인스턴스."""
