"""MediaPipe ROI 크롭 + 정규화.

Face Mesh 랜드마크(ROI_CROP_IDX 20개)로 입술 바운딩 박스를 구하고
ROI_MARGIN(20%)을 추가한 뒤 ROI_SIZE(96×96)로 resize, ImageNet 정규화한다.
VVAD용 mouth_ratio도 함께 계산한다.
"""

from __future__ import annotations

import numpy as np

from config import (
    NORMALIZE_MEAN,
    NORMALIZE_STD,
    ROI_CROP_IDX,
    ROI_MARGIN,
    ROI_SIZE,
    VVAD_IDX_BOT,
    VVAD_IDX_TOP,
)


def extract_landmarks(frame: np.ndarray) -> np.ndarray | None:
    """단일 프레임에서 MediaPipe Face Mesh 랜드마크를 추출한다.

    Args:
        frame: BGR/RGB 단일 프레임.

    Returns:
        (468, 2) 정규화 좌표 배열. 얼굴 미검출 시 None.
    """
    pass


def crop_roi(frame: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """ROI_CROP_IDX 랜드마크로 입술 영역을 크롭한다.

    바운딩 박스에 상하좌우 ROI_MARGIN(20%)을 추가한 뒤 ROI_SIZE(96×96)로
    resize한다.

    Args:
        frame: 원본 프레임.
        landmarks: extract_landmarks 결과.

    Returns:
        ROI_SIZE 크기로 크롭/리사이즈된 ROI 프레임.
    """
    pass


def compute_mouth_ratio(landmarks: np.ndarray) -> float:
    """VVAD용 mouth_ratio를 계산한다.

    d_raw = VVAD_IDX_TOP 평균좌표 ↔ VVAD_IDX_BOT 평균좌표 유클리드 거리.
    mouth_ratio = d_raw / W (W: 좌우 광대 랜드마크 거리).

    Args:
        landmarks: extract_landmarks 결과.

    Returns:
        정규화된 입 벌림 비율.
    """
    pass


def normalize_roi(roi: np.ndarray) -> np.ndarray:
    """ROI 프레임을 ImageNet 평균/표준편차로 정규화한다.

    (x/255 - NORMALIZE_MEAN) / NORMALIZE_STD 채널별 적용.

    Args:
        roi: ROI_SIZE 크기 uint8 프레임.

    Returns:
        정규화된 float32 ROI (H, W, C).
    """
    pass
