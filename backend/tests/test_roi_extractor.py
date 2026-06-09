"""roi_extractor 단위 테스트.

ROI 크롭 크기(96×96), margin 적용, mouth_ratio 계산, 정규화 범위를 검증한다.
"""

from __future__ import annotations


def test_crop_output_size() -> None:
    """크롭 결과가 ROI_SIZE(96, 96)인지 검증한다."""
    pass


def test_roi_margin_applied() -> None:
    """바운딩 박스에 ROI_MARGIN(20%)이 적용되었는지 검증한다."""
    pass


def test_mouth_ratio_range() -> None:
    """mouth_ratio가 d_raw / W 정의에 부합하는 양수인지 검증한다."""
    pass


def test_normalize_uses_imagenet_stats() -> None:
    """정규화가 NORMALIZE_MEAN/STD를 사용하는지 검증한다."""
    pass


def test_no_face_returns_none() -> None:
    """얼굴 미검출 프레임에서 None을 반환하는지 검증한다."""
    pass
