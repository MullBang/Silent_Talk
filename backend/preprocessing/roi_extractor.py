"""MediaPipe ROI 크롭 + 정규화 + 모델 텐서 변환.

설계문서 1.1절 랜드마크 기준 구현.

MediaPipe Face Mesh 외측 입술 랜드마크(ROI_CROP_IDX 20개)로 입 영역 바운딩
박스를 구하고 상하좌우 ROI_MARGIN(20%)을 더한 뒤 ROI_SIZE(96×96)로 resize한다.
화면에 여러 명이 있으면 바운딩 박스 면적이 가장 큰 1인만 추적하며, 연속 12프레임
이상 얼굴 미검출 시 해당 세그먼트를 None으로 폐기한다.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch

try:  # mediapipe는 선택적 import (단위 테스트가 헬퍼만 검증할 수 있도록)
    import mediapipe as mp
except ImportError:  # pragma: no cover
    mp = None

from config import (
    NORMALIZE_MEAN,
    NORMALIZE_STD,
    ROI_CROP_IDX,
    ROI_MARGIN,
    ROI_SIZE,
    SEQ_LEN,
)

# 연속 미검출 허용 한계. 이 값 이상 연속으로 얼굴이 없으면 세그먼트를 폐기한다.
MAX_CONSECUTIVE_MISS: int = 12


def _compute_bbox_with_margin(
    points_px: np.ndarray,
    img_w: int,
    img_h: int,
    margin: float = ROI_MARGIN,
) -> tuple[int, int, int, int]:
    """랜드마크 픽셀 좌표로 margin이 적용된 바운딩 박스를 계산한다.

    바운딩 박스 너비/높이에 각각 margin 비율을 상하좌우로 추가한 뒤 이미지
    경계로 클램프한다.

    Args:
        points_px: (N, 2) 픽셀 좌표 배열.
        img_w: 원본 이미지 너비.
        img_h: 원본 이미지 높이.
        margin: 상하좌우 추가 margin 비율 (기본 ROI_MARGIN=0.20).

    Returns:
        (x1, y1, x2, y2) 정수 좌표. x2 > x1, y2 > y1 보장.
    """
    x_min = float(points_px[:, 0].min())
    y_min = float(points_px[:, 1].min())
    x_max = float(points_px[:, 0].max())
    y_max = float(points_px[:, 1].max())

    bw = x_max - x_min
    bh = y_max - y_min

    x1 = int(np.floor(x_min - bw * margin))
    y1 = int(np.floor(y_min - bh * margin))
    x2 = int(np.ceil(x_max + bw * margin))
    y2 = int(np.ceil(y_max + bh * margin))

    # 이미지 경계 클램프
    x1 = max(0, min(x1, img_w - 1))
    y1 = max(0, min(y1, img_h - 1))
    x2 = max(x1 + 1, min(x2, img_w))
    y2 = max(y1 + 1, min(y2, img_h))

    return x1, y1, x2, y2


def _largest_face_landmarks(multi_face_landmarks):
    """검출된 여러 얼굴 중 ROI 바운딩 박스 면적이 최대인 얼굴을 선택한다.

    Args:
        multi_face_landmarks: MediaPipe results.multi_face_landmarks.

    Returns:
        면적 최대 얼굴의 landmark 객체. 입력이 비면 None.
    """
    if not multi_face_landmarks:
        return None

    best = None
    best_area = -1.0
    for face in multi_face_landmarks:
        xs = [face.landmark[i].x for i in ROI_CROP_IDX]
        ys = [face.landmark[i].y for i in ROI_CROP_IDX]
        area = (max(xs) - min(xs)) * (max(ys) - min(ys))
        if area > best_area:
            best_area = area
            best = face
    return best


def _crop_roi(frame: np.ndarray, face_landmarks) -> np.ndarray:
    """단일 프레임에서 입 ROI를 크롭하고 ROI_SIZE로 리사이즈한다.

    Args:
        frame: RGB 단일 프레임 (H, W, 3).
        face_landmarks: 선택된 얼굴의 MediaPipe landmark 객체.

    Returns:
        ROI_SIZE 크기의 uint8 ROI (96, 96, 3).
    """
    h, w = frame.shape[:2]
    points_px = np.array(
        [[face_landmarks.landmark[i].x * w, face_landmarks.landmark[i].y * h]
         for i in ROI_CROP_IDX],
        dtype=np.float32,
    )
    x1, y1, x2, y2 = _compute_bbox_with_margin(points_px, w, h)
    crop = frame[y1:y2, x1:x2]
    return cv2.resize(crop, ROI_SIZE, interpolation=cv2.INTER_LINEAR)


def extract_roi_from_segment(frames: np.ndarray) -> np.ndarray | None:
    """세그먼트 프레임 시퀀스에서 입 ROI 시퀀스를 추출한다.

    MediaPipe Face Mesh(static_image_mode=False)로 프레임별 얼굴을 추적하고,
    화면에 여러 명이면 바운딩 박스 면적 최대 1인만 사용한다. 개별 프레임의
    일시적 미검출은 직전(또는 최초) 유효 ROI로 채우되, 연속 MAX_CONSECUTIVE_MISS
    프레임 이상 미검출되면 세그먼트 전체를 폐기(None)한다.

    Args:
        frames: 세그먼트 프레임 (T, H, W, 3). 보통 T == SEQ_LEN(75).
                cv2(BGR) 프레임을 가정하며 내부에서 RGB로 변환한다.

    Returns:
        (T, 96, 96, 3) uint8 ROI 배열. 미검출 한계 초과 시 None.

    Raises:
        RuntimeError: mediapipe가 설치되지 않은 경우.
        ValueError: frames가 비어 있는 경우.
    """
    if mp is None:  # pragma: no cover
        raise RuntimeError("mediapipe가 설치되어 있지 않습니다.")

    n_frames = len(frames)
    if n_frames == 0:
        raise ValueError("빈 세그먼트입니다.")

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=4,
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    rois: list[np.ndarray | None] = []
    consecutive_miss = 0
    try:
        for frame in frames:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)
            face = _largest_face_landmarks(
                getattr(results, "multi_face_landmarks", None)
            )

            if face is None:
                consecutive_miss += 1
                if consecutive_miss >= MAX_CONSECUTIVE_MISS:
                    return None
                rois.append(None)  # 일시적 미검출 → 추후 채움
            else:
                consecutive_miss = 0
                rois.append(_crop_roi(rgb, face))
    finally:
        face_mesh.close()

    return _fill_missing_rois(rois)


def _fill_missing_rois(rois: list[np.ndarray | None]) -> np.ndarray | None:
    """일시적 미검출(None) 프레임을 직전/최초 유효 ROI로 채운다.

    zero-padding을 피하기 위해 forward-fill 후, 선두의 None은 최초 유효 ROI로
    back-fill한다.

    Args:
        rois: 프레임별 ROI 또는 None 리스트.

    Returns:
        (T, 96, 96, 3) uint8 배열. 유효 ROI가 하나도 없으면 None.
    """
    # forward-fill
    last_valid: np.ndarray | None = None
    for i, roi in enumerate(rois):
        if roi is not None:
            last_valid = roi
        elif last_valid is not None:
            rois[i] = last_valid

    if last_valid is None:
        return None  # 유효 프레임 전무

    # 선두 None back-fill
    first_valid = next(roi for roi in rois if roi is not None)
    for i, roi in enumerate(rois):
        if roi is None:
            rois[i] = first_valid

    return np.stack(rois, axis=0).astype(np.uint8)


def normalize_roi(roi_uint8: np.ndarray) -> np.ndarray:
    """ROI를 [0,1] 스케일 후 ImageNet 평균/표준편차로 정규화한다.

    (x / 255 - NORMALIZE_MEAN) / NORMALIZE_STD 를 채널별로 적용한다.

    Args:
        roi_uint8: (T, 96, 96, 3) uint8 ROI 배열.

    Returns:
        (T, 96, 96, 3) float32 정규화 배열.

    Raises:
        ValueError: 마지막 채널 차원이 3이 아닌 경우.
    """
    if roi_uint8.shape[-1] != 3:
        raise ValueError(f"채널 차원은 3이어야 합니다: {roi_uint8.shape}")

    mean = np.array(NORMALIZE_MEAN, dtype=np.float32)
    std = np.array(NORMALIZE_STD, dtype=np.float32)

    scaled = roi_uint8.astype(np.float32) / 255.0
    normalized = (scaled - mean) / std
    return normalized.astype(np.float32)


def to_model_tensor(roi_normalized: np.ndarray) -> torch.Tensor:
    """정규화 ROI를 PyTorch 3D-CNN 입력 텐서로 변환한다.

    (T, H, W, C) → permute → (C, T, H, W) → unsqueeze(0) →
    (1, C, T, H, W). T==SEQ_LEN, (H,W)==ROI_SIZE, C==3 기준이면 최종 형상은
    (1, 3, 75, 96, 96)로 MODEL_INPUT_SHAPE (B, C, T, H, W)와 일치한다.

    Args:
        roi_normalized: (T, 96, 96, 3) float32 정규화 ROI.

    Returns:
        (1, 3, T, 96, 96) float32 torch.Tensor.

    Raises:
        ValueError: 마지막 채널 차원이 3이 아닌 경우.
    """
    if roi_normalized.shape[-1] != 3:
        raise ValueError(f"채널 차원은 3이어야 합니다: {roi_normalized.shape}")

    tensor = torch.from_numpy(np.ascontiguousarray(roi_normalized)).float()
    # (T, H, W, C) -> (C, T, H, W)
    tensor = tensor.permute(3, 0, 1, 2).contiguous()
    # (C, T, H, W) -> (1, C, T, H, W)
    return tensor.unsqueeze(0)
