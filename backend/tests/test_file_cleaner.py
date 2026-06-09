"""file_cleaner 단위 테스트.

설계문서 13절 — 즉시 삭제(delete_upload_file), 1시간 경과 파일 정리,
연속 실패 카운팅을 검증한다. (실제 1시간 대기 없이 mtime을 조작한다.)
"""

from __future__ import annotations

import os
import sys
import time

# backend 패키지 루트를 import 경로에 추가
_BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

from services import file_cleaner as fc  # noqa: E402


def test_delete_upload_file_success(tmp_path) -> None:
    """존재하는 파일을 즉시 삭제하고 True를 반환하는지 검증한다."""
    f = tmp_path / "sample.mp4"
    f.write_bytes(b"data")
    assert fc.delete_upload_file(str(f)) is True
    assert not f.exists()


def test_delete_upload_file_failure_returns_false() -> None:
    """존재하지 않는 파일 삭제 시 False를 반환한다 (예외 미전파)."""
    assert fc.delete_upload_file("non_existent_file_xyz.tmp") is False


def test_cleanup_removes_only_expired(tmp_path, monkeypatch) -> None:
    """1시간 이상 경과 파일만 삭제하고 최근 파일은 남기는지 검증한다."""
    monkeypatch.setattr(fc, "TMP_UPLOAD_DIR", str(tmp_path))

    old = tmp_path / "old.mp4"
    old.write_bytes(b"old")
    new = tmp_path / "new.mp4"
    new.write_bytes(b"new")

    # old 파일의 수정시각을 2시간 전으로 조작
    two_hours_ago = time.time() - 7200
    os.utime(old, (two_hours_ago, two_hours_ago))

    deleted, failed = fc._cleanup_expired_files()
    assert deleted == 1
    assert failed == 0
    assert not old.exists()
    assert new.exists()


def test_cleanup_no_dir_returns_zero(tmp_path, monkeypatch) -> None:
    """대상 디렉토리가 없으면 (0, 0)을 반환한다."""
    monkeypatch.setattr(fc, "TMP_UPLOAD_DIR", str(tmp_path / "missing"))
    assert fc._cleanup_expired_files() == (0, 0)


def test_consecutive_failure_critical(tmp_path, monkeypatch) -> None:
    """정리 실패가 연속 3회 누적되면 CRITICAL 임계치에 도달하는지 검증한다."""
    monkeypatch.setattr(fc, "_consecutive_failures", 0)
    # 항상 실패(파일 1건 실패)를 반환하도록 패치
    monkeypatch.setattr(fc, "_cleanup_expired_files", lambda *a, **k: (0, 1))

    for _ in range(3):
        fc._run_cleanup()

    assert fc._consecutive_failures >= fc._CRITICAL_FAILURE_THRESHOLD


def test_success_resets_failure_counter(tmp_path, monkeypatch) -> None:
    """성공한 정리 작업은 연속 실패 카운터를 0으로 초기화한다."""
    monkeypatch.setattr(fc, "_consecutive_failures", 2)
    monkeypatch.setattr(fc, "_cleanup_expired_files", lambda *a, **k: (5, 0))
    fc._run_cleanup()
    assert fc._consecutive_failures == 0


def test_scheduler_start_and_shutdown() -> None:
    """스케줄러가 1시간 주기 작업과 함께 시작/종료되는지 검증한다."""
    scheduler = fc.start_cleanup_scheduler()
    try:
        assert scheduler.running
        assert scheduler.get_job("tmp_upload_cleanup") is not None
    finally:
        fc.shutdown_cleanup_scheduler()
    assert fc._scheduler is None
