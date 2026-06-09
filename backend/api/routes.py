"""라우터 통합 등록 모듈.

upload / infer / evaluation 서브 라우터를 '/api' prefix 아래로 묶어 main.py에
노출한다.
"""

from __future__ import annotations

import os
import sys

from fastapi import APIRouter

# backend 루트를 import 경로에 부트스트랩
_BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

from api.evaluation import router as evaluation_router  # noqa: E402
from api.infer import router as infer_router  # noqa: E402
from api.upload import router as upload_router  # noqa: E402


def get_api_router() -> APIRouter:
    """모든 서브 라우터가 등록된 최상위 API 라우터를 반환한다.

    Returns:
        '/api' prefix가 적용된 통합 APIRouter.
    """
    router = APIRouter(prefix="/api")
    router.include_router(upload_router, tags=["upload"])
    router.include_router(infer_router, tags=["infer"])
    router.include_router(evaluation_router, tags=["evaluation"])
    return router
