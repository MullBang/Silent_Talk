"""FastAPI 애플리케이션 진입점.

CORS 미들웨어(화이트리스트), 라우터 등록, 라이프사이클(startup/shutdown) 훅을
구성한다. 실제 라우트 로직은 api/ 패키지에 위임한다.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import CORS_ORIGINS


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """애플리케이션 수명주기 관리.

    startup: 임시/로그 디렉토리 생성, 모델 가중치 로드, 파일 클리너 스케줄러 시작.
    shutdown: 스케줄러 종료, 리소스 정리.
    """
    pass


def create_app() -> FastAPI:
    """FastAPI 앱 인스턴스를 생성하고 미들웨어/라우터를 등록한다.

    CORS는 config.CORS_ORIGINS 화이트리스트만 허용한다 (와일드카드 금지).

    Returns:
        구성이 완료된 FastAPI 애플리케이션.
    """
    pass


app: FastAPI | None = None
"""ASGI 서버(uvicorn)가 참조하는 애플리케이션 인스턴스."""
