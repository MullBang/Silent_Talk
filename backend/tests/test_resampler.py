"""resampler 단위 테스트.

설계문서 1.2절 'FPS 리샘플링 처리' 검증.

타임스탬프 기반 리샘플링이 목표 FPS로 균등 샘플링하는지, MAX_DURATION_SEC
초과 시 예외가 발생하는지 cv2.VideoWriter로 생성한 실제 영상으로 검증한다.
"""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np
import pytest

# backend 패키지 루트를 import 경로에 추가 (config / preprocessing 직접 import)
_BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

from config import FPS, MAX_DURATION_SEC  # noqa: E402
from preprocessing.resampler import (  # noqa: E402
    get_video_duration_sec,
    resample_to_fps,
)

_FRAME_W = 32
_FRAME_H = 32


def _make_video(path: str, fps: float, num_frames: int) -> None:
    """지정 FPS/프레임 수의 테스트 영상을 생성한다.

    각 프레임에 프레임 인덱스를 밝기로 인코딩해 내용이 모두 다르도록 한다.
    코덱 호환을 위해 mp4v/.mp4 조합을 사용한다.
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (_FRAME_W, _FRAME_H))
    assert writer.isOpened(), "VideoWriter 초기화 실패 (코덱 확인 필요)"
    try:
        for i in range(num_frames):
            value = (i * 3) % 256
            frame = np.full((_FRAME_H, _FRAME_W, 3), value, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()


def test_resample_30fps_to_25fps_frame_count(tmp_path) -> None:
    """30fps 영상을 25fps로 리샘플링한 후 프레임 수를 검증한다.

    기대 프레임 수 = round(duration * 25), duration = 원본프레임수 / 원본FPS.
    """
    video = str(tmp_path / "v30.mp4")
    written = 60  # 30fps × 2초
    _make_video(video, fps=30.0, num_frames=written)

    frames, original_fps = resample_to_fps(video, target_fps=FPS)

    duration = written / original_fps
    expected = int(round(duration * FPS))
    assert len(frames) == expected
    # 30fps 2초 → 25fps 약 50프레임
    assert len(frames) == 50


def test_resample_60fps_to_25fps_timestamp_interval(tmp_path) -> None:
    """60fps 영상을 25fps로 리샘플링한 후 타임스탬프 간격(1/25초)을 검증한다.

    출력 프레임은 목표 FPS 타임라인에 균등 배치되므로 평균 간격 ≈ 1/25 = 0.04초.
    """
    video = str(tmp_path / "v60.mp4")
    written = 120  # 60fps × 2초
    _make_video(video, fps=60.0, num_frames=written)

    frames, original_fps = resample_to_fps(video, target_fps=FPS)

    duration = written / original_fps
    # 출력 길이로 환산한 실효 프레임 간격이 1/25초에 수렴하는지 검증
    effective_interval = duration / len(frames)
    assert effective_interval == pytest.approx(1.0 / FPS, rel=0.05)
    # 출력 프레임 수도 목표 FPS 기준과 일치
    assert len(frames) == int(round(duration * FPS))


def test_duration_exceeds_max_raises(tmp_path) -> None:
    """MAX_DURATION_SEC를 초과하는 영상에서 ValueError가 발생하는지 검증한다.

    저용량 유지를 위해 낮은 FPS·소수 프레임으로 긴 재생시간을 구성한다
    (예: 2fps × (초과 길이) → 파일은 작지만 duration은 한도 초과).
    """
    video = str(tmp_path / "long.mp4")
    fps = 2.0
    # MAX_DURATION_SEC를 확실히 초과하도록 +2초 여유
    num_frames = int((MAX_DURATION_SEC + 2) * fps)
    _make_video(video, fps=fps, num_frames=num_frames)

    with pytest.raises(ValueError):
        get_video_duration_sec(video)


def test_duration_within_limit_ok(tmp_path) -> None:
    """허용 범위 내 영상은 정상적으로 재생 시간을 반환하는지 검증한다."""
    video = str(tmp_path / "ok.mp4")
    _make_video(video, fps=25.0, num_frames=50)  # 2초

    duration = get_video_duration_sec(video)
    assert duration == pytest.approx(2.0, abs=0.1)


def test_invalid_path_raises() -> None:
    """존재하지 않는 경로에 대해 예외가 발생하는지 검증한다."""
    with pytest.raises((FileNotFoundError, ValueError)):
        resample_to_fps("non_existent_video.mp4")
