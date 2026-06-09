"""FPS 타임스탬프 기반 리샘플링.

cap.set(CAP_PROP_FPS)는 저장 영상에 효과가 없으므로 절대 사용하지 않는다.
각 프레임의 실제 타임스탬프를 기준으로 목표 FPS에 맞춰 균등 샘플링한다.
"""

from __future__ import annotations

import numpy as np

from config import FPS


def resample_to_fps(
    frames: list[np.ndarray],
    timestamps_ms: list[float],
    target_fps: int = FPS,
) -> list[np.ndarray]:
    """타임스탬프 기반 nearest-neighbor 균등 샘플링.

    cap.set(CAP_PROP_FPS)는 저장 영상에 효과 없으므로 사용 금지.
    target_fps 간격의 이상적 타임라인을 만들고 각 지점에 가장 가까운
    실제 프레임을 선택한다.

    Args:
        frames: 디코딩된 원본 프레임 리스트.
        timestamps_ms: 각 프레임의 밀리초 타임스탬프 (frames와 동일 길이).
        target_fps: 목표 프레임 레이트 (기본 FPS=25).

    Returns:
        target_fps로 리샘플된 프레임 리스트.
    """
    pass
