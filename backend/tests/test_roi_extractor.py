"""roi_extractor 단위 테스트.

설계문서 1.1절 랜드마크 기준 검증.

ROI 크롭 bbox/margin 계산, ImageNet 정규화, 모델 텐서 변환, 그리고 얼굴
미검출 세그먼트 폐기(None) 동작을 검증한다.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

# backend 패키지 루트를 import 경로에 추가
_BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

from config import (  # noqa: E402
    NORMALIZE_MEAN,
    NORMALIZE_STD,
    ROI_SIZE,
    SEQ_LEN,
)
from preprocessing.roi_extractor import (  # noqa: E402
    MAX_CONSECUTIVE_MISS,
    _compute_bbox_with_margin,
    _fill_missing_rois,
    extract_roi_from_segment,
    normalize_roi,
    to_model_tensor,
)


def test_bbox_margin_applied() -> None:
    """bbox에 상하좌우 20% margin이 적용되는지 검증한다."""
    # x: 40~60 (너비 20), y: 40~60 (높이 20) → margin 20% = 4px 확장
    points = np.array([[40, 40], [60, 60], [50, 50]], dtype=np.float32)
    x1, y1, x2, y2 = _compute_bbox_with_margin(points, img_w=200, img_h=200)

    assert x1 == 36  # 40 - 4
    assert y1 == 36
    assert x2 == 64  # 60 + 4
    assert y2 == 64


def test_bbox_clamped_to_image_bounds() -> None:
    """margin 적용 후 좌표가 이미지 경계로 클램프되는지 검증한다."""
    points = np.array([[2, 2], [98, 98]], dtype=np.float32)
    x1, y1, x2, y2 = _compute_bbox_with_margin(points, img_w=100, img_h=100)

    assert x1 >= 0 and y1 >= 0
    assert x2 <= 100 and y2 <= 100
    assert x2 > x1 and y2 > y1


def test_normalize_roi_values_and_shape() -> None:
    """정규화 값/형상/dtype을 검증한다 ((x/255 - mean)/std)."""
    roi = np.full((SEQ_LEN, *ROI_SIZE, 3), 255, dtype=np.uint8)
    out = normalize_roi(roi)

    assert out.shape == (SEQ_LEN, ROI_SIZE[0], ROI_SIZE[1], 3)
    assert out.dtype == np.float32

    # 255 → 1.0 → (1 - mean)/std 채널별
    for c in range(3):
        expected = (1.0 - NORMALIZE_MEAN[c]) / NORMALIZE_STD[c]
        assert out[..., c] == pytest.approx(expected, rel=1e-5)


def test_normalize_roi_rejects_bad_channel() -> None:
    """채널 수가 3이 아니면 예외가 발생한다."""
    bad = np.zeros((SEQ_LEN, 96, 96, 1), dtype=np.uint8)
    with pytest.raises(ValueError):
        normalize_roi(bad)


def test_to_model_tensor_shape() -> None:
    """(75,96,96,3) → (1,3,75,96,96) 변환 형상을 검증한다."""
    roi = np.zeros((SEQ_LEN, ROI_SIZE[0], ROI_SIZE[1], 3), dtype=np.float32)
    tensor = to_model_tensor(roi)

    assert tuple(tensor.shape) == (1, 3, SEQ_LEN, ROI_SIZE[0], ROI_SIZE[1])
    assert tensor.dtype.is_floating_point


def test_to_model_tensor_preserves_values() -> None:
    """permute 후 채널/시간 축 값이 보존되는지 검증한다."""
    roi = np.random.rand(SEQ_LEN, ROI_SIZE[0], ROI_SIZE[1], 3).astype(np.float32)
    tensor = to_model_tensor(roi)

    # tensor[0, c, t, h, w] == roi[t, h, w, c]
    np_back = tensor[0].permute(1, 2, 3, 0).numpy()
    assert np.allclose(np_back, roi, atol=1e-6)


def test_fill_missing_forward_and_back() -> None:
    """일시적 미검출(None)이 forward/back-fill로 채워지는지 검증한다."""
    a = np.full((*ROI_SIZE, 3), 10, dtype=np.uint8)
    b = np.full((*ROI_SIZE, 3), 20, dtype=np.uint8)
    rois = [None, a, None, b, None]

    filled = _fill_missing_rois(rois)
    assert filled.shape == (5, ROI_SIZE[0], ROI_SIZE[1], 3)
    assert np.array_equal(filled[0], a)  # 선두 None → 최초 유효(back-fill)
    assert np.array_equal(filled[2], a)  # forward-fill
    assert np.array_equal(filled[4], b)  # forward-fill


def test_fill_missing_all_none_returns_none() -> None:
    """유효 ROI가 전혀 없으면 None을 반환한다."""
    assert _fill_missing_rois([None, None, None]) is None


def test_no_face_segment_returns_none() -> None:
    """얼굴이 없는(단색) 세그먼트는 연속 미검출 한계 초과로 None을 반환한다."""
    # MAX_CONSECUTIVE_MISS 이상 연속 미검출 → 폐기
    frames = np.zeros((SEQ_LEN, 64, 64, 3), dtype=np.uint8)
    result = extract_roi_from_segment(frames)
    assert result is None


def test_extract_empty_raises() -> None:
    """빈 세그먼트는 예외를 발생시킨다."""
    with pytest.raises(ValueError):
        extract_roi_from_segment(np.empty((0, 64, 64, 3), dtype=np.uint8))


def test_max_consecutive_miss_constant() -> None:
    """연속 미검출 폐기 기준이 설계문서 값(12)과 일치하는지 확인한다."""
    assert MAX_CONSECUTIVE_MISS == 12
