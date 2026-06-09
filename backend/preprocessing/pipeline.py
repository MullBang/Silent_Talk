"""전처리 파이프라인 오케스트레이터.

resampler → segmenter → roi_extractor 순으로 호출하여 원본 영상을 모델 입력
텐서(B, T, H, W, C) = (B, 75, 96, 96, 3)로 변환한다.
"""

from __future__ import annotations

import numpy as np


def preprocess_video(video_path: str) -> np.ndarray:
    """원본 영상을 전처리하여 모델 입력 직전 텐서로 변환한다.

    단계: 디코딩 → 25fps 타임스탬프 리샘플링 → 3초 세그먼트 분할 →
    MediaPipe ROI 크롭/정규화 → last-frame 패딩.

    Args:
        video_path: 입력 영상 파일 경로.

    Returns:
        PREPROC_SHAPE (B, 75, 96, 96, 3) 형상의 float32 ndarray.
        모델 입력 전 permute((0, 4, 1, 2, 3)) 변환 필요.
    """
    pass
