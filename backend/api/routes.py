"""라우터 통합 등록 모듈.

upload / infer / evaluation 서브 라우터를 하나의 APIRouter로 묶어 main.py에 노출한다.
"""

from __future__ import annotations

from fastapi import APIRouter


def get_api_router() -> APIRouter:
    """모든 서브 라우터가 등록된 최상위 API 라우터를 반환한다.

    Returns:
        '/api' prefix가 적용된 통합 APIRouter.
    """
    pass
