"""FastAPI 애플리케이션 진입점.

CORS 미들웨어(화이트리스트), 라우터 등록, 라이프사이클(startup/shutdown) 훅,
WebSocket placeholder를 구성한다. 실제 라우트 로직은 api/ 패키지에 위임한다.

실행: uvicorn backend.main:app --reload  (또는 backend/ 에서 uvicorn main:app)
"""

from __future__ import annotations

import logging
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

from api import record as record_api  # noqa: E402
from api import upload as upload_api  # noqa: E402
from api.limits import RequestBodyLimitMiddleware  # noqa: E402
from api.routes import get_api_router  # noqa: E402
from config import (  # noqa: E402
    CORS_ORIGINS,
    MODEL_BACKEND,
    MODEL_WEIGHTS_PATH,
    OLKAVS_PRELOAD,
    REQUEST_BODY_OVERHEAD_BYTES,
)
from services.file_cleaner import (  # noqa: E402
    purge_roi_dirs_on_startup,
    shutdown_cleanup_scheduler,
    start_cleanup_scheduler,
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """애플리케이션 수명주기 관리.

    startup: 이전 실행이 남긴 입술 crop 폴더 삭제, 임시 파일 정리 스케줄러 시작(즉시 1회 실행),
             (olkavs) 공식 모델 워커 백그라운드 사전 기동.
    shutdown: 스케줄러 종료, 워커 종료.
    """
    purge_roi_dirs_on_startup()
    start_cleanup_scheduler()
    if MODEL_BACKEND == "olkavs" and OLKAVS_PRELOAD:
        from services.olkavs_service import get_service

        get_service().warm_start_async()
    try:
        yield
    finally:
        shutdown_cleanup_scheduler()
        if MODEL_BACKEND == "olkavs":
            from services.olkavs_service import shutdown_service

            shutdown_service()


def model_status() -> dict:
    """/health 용 모델 상태. 워커를 새로 띄우거나 가중치를 읽지 않는다."""
    try:
        if MODEL_BACKEND == "olkavs":
            from services.olkavs_service import peek_service

            service = peek_service()
            return service.status() if service is not None else {"roi_worker": "stopped", "infer_worker": "stopped"}
        return {"weights_path": MODEL_WEIGHTS_PATH, "weights_present": os.path.exists(MODEL_WEIGHTS_PATH)}
    except Exception as exc:  # noqa: BLE001 - 헬스 체크는 실패하지 않는다
        logger.exception("모델 상태 조회 실패")
        return {"error": str(exc)}


def create_app() -> FastAPI:
    """FastAPI 앱 인스턴스를 생성하고 미들웨어/라우터를 등록한다.

    CORS는 config.CORS_ORIGINS 화이트리스트만 허용한다 (와일드카드 금지).

    Returns:
        구성이 완료된 FastAPI 애플리케이션.
    """
    app = FastAPI(title="Silent Talk Lip Reading API", lifespan=lifespan)

    # 요청 본문 크기 제한 (/api/record·/api/upload). Starlette는 나중에 추가한 미들웨어가 바깥에 오므로
    # CORS보다 먼저 추가해 CORS 안쪽에 둔다 → 413 응답에도 화이트리스트 Origin의 CORS 헤더가 붙는다.
    app.add_middleware(
        RequestBodyLimitMiddleware,
        rules={"/api/record": record_api.record_body_limit, "/api/upload": upload_api.upload_body_limit},
        overhead_bytes=REQUEST_BODY_OVERHEAD_BYTES,
    )

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
        """헬스 체크 엔드포인트. 서버 상태와 현재 인식 모델(백엔드·워커·가중치 SHA)을 알려 준다."""
        return {"status": "ok", "model_backend": MODEL_BACKEND, "model": model_status()}

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
