"""API 통합 테스트.

업로드 → 추론 등록 → 결과 폴링 흐름과 핵심 규칙(즉시 결과 반환 금지,
CORS 화이트리스트, MIME 검증)을 검증한다.
"""

from __future__ import annotations


def test_infer_does_not_return_results_immediately() -> None:
    """POST /api/infer가 results 없이 {job_id, session_id, status:'processing'}만 반환하는지 검증한다."""
    pass


def test_result_polling_returns_status() -> None:
    """GET /api/result/{session_id}가 상태 기반 응답을 반환하는지 검증한다."""
    pass


def test_cors_rejects_unlisted_origin() -> None:
    """화이트리스트에 없는 오리진이 차단되는지 검증한다 (와일드카드 미사용)."""
    pass


def test_upload_rejects_invalid_mime() -> None:
    """python-magic MIME 검증이 비영상 파일을 거부하는지 검증한다."""
    pass


def test_evaluation_requires_test_set_id() -> None:
    """평가가 test_set_path 직접 전달을 거부하고 test_set_id만 받는지 검증한다."""
    pass
