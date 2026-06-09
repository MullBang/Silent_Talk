"""FPS 타임스탬프 기반 리샘플링.

설계문서 1.2절 'FPS 리샘플링 처리' 구현.

cap.set(cv2.CAP_PROP_FPS)는 저장된(녹화) 영상에 아무런 효과가 없으므로 절대
사용하지 않는다. 대신 원본 프레임 각각의 타임스탬프(index / original_fps)를
기준으로, 목표 FPS의 이상적 타임라인에 가장 가까운 프레임을 nearest-neighbor로
선택한다.
"""

from __future__ import annotations

import cv2
import numpy as np

from config import FPS, MAX_DURATION_SEC


def _read_all_frames(video_path: str) -> tuple[list[np.ndarray], float]:
    """영상의 모든 프레임과 원본 FPS를 읽어 반환한다.

    Args:
        video_path: 입력 영상 파일 경로.

    Returns:
        (프레임 리스트, 원본 FPS).

    Raises:
        FileNotFoundError: 영상을 열 수 없는 경우.
        ValueError: 원본 FPS가 유효하지 않거나(<= 0) 프레임이 없는 경우.
    """
    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            raise FileNotFoundError(f"영상을 열 수 없습니다: {video_path}")

        original_fps = float(cap.get(cv2.CAP_PROP_FPS))
        if original_fps <= 0:
            raise ValueError(
                f"유효하지 않은 원본 FPS({original_fps})입니다: {video_path}"
            )

        frames: list[np.ndarray] = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)

        if not frames:
            raise ValueError(f"프레임이 존재하지 않습니다: {video_path}")

        return frames, original_fps
    finally:
        cap.release()


def get_video_duration_sec(video_path: str) -> float:
    """영상 총 재생 시간(초)을 반환한다.

    duration = frame_count / original_fps 로 계산하며, MAX_DURATION_SEC를
    초과하면 예외를 발생시킨다 (업로드 제약 강제).

    cap.get(CAP_PROP_FRAME_COUNT)가 신뢰 불가(<= 0)한 코덱의 경우 프레임을
    직접 카운트하는 폴백을 사용한다.

    Args:
        video_path: 입력 영상 파일 경로.

    Returns:
        영상 재생 시간(초).

    Raises:
        FileNotFoundError: 영상을 열 수 없는 경우.
        ValueError: FPS가 유효하지 않거나 재생 시간이 MAX_DURATION_SEC를 초과하는 경우.
    """
    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            raise FileNotFoundError(f"영상을 열 수 없습니다: {video_path}")

        original_fps = float(cap.get(cv2.CAP_PROP_FPS))
        if original_fps <= 0:
            raise ValueError(
                f"유효하지 않은 원본 FPS({original_fps})입니다: {video_path}"
            )

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            # 메타데이터를 신뢰할 수 없으면 직접 카운트
            frame_count = 0
            while True:
                ok, _ = cap.read()
                if not ok:
                    break
                frame_count += 1

        if frame_count <= 0:
            raise ValueError(f"프레임이 존재하지 않습니다: {video_path}")

        duration = frame_count / original_fps
        if duration > MAX_DURATION_SEC:
            raise ValueError(
                f"영상 길이({duration:.2f}s)가 허용 최대치"
                f"({MAX_DURATION_SEC}s)를 초과했습니다."
            )

        return duration
    finally:
        cap.release()


def resample_to_fps(
    video_path: str,
    target_fps: int = FPS,
) -> tuple[list[np.ndarray], float]:
    """타임스탬프 기반 nearest-neighbor 균등 리샘플링.

    cap.set(cv2.CAP_PROP_FPS)는 저장 영상에 효과가 없으므로 사용 금지.
    원본 FPS는 cap.get(cv2.CAP_PROP_FPS)로 읽는다.

    절차:
      1. 원본 프레임 전체와 original_fps 로드.
      2. duration = len(frames) / original_fps.
      3. n_frames = round(duration * target_fps) (목표 프레임 수).
      4. target_timestamps = [i / target_fps for i in range(n_frames)].
      5. 각 target_timestamp에 대해 원본 타임스탬프(j / original_fps)가 가장
         가까운 프레임을 nearest-neighbor로 선택.

    Args:
        video_path: 입력 영상 파일 경로.
        target_fps: 목표 프레임 레이트 (기본 FPS=25).

    Returns:
        (리샘플된 프레임 리스트, 원본 FPS).

    Raises:
        FileNotFoundError: 영상을 열 수 없는 경우.
        ValueError: 원본 FPS가 유효하지 않거나 프레임이 없는 경우.
    """
    if target_fps <= 0:
        raise ValueError(f"target_fps는 양수여야 합니다: {target_fps}")

    frames, original_fps = _read_all_frames(video_path)

    n_original = len(frames)
    duration = n_original / original_fps

    # 목표 프레임 수 (3초 영상 × 25fps == 75 형태)
    n_frames = int(round(duration * target_fps))
    if n_frames <= 0:
        n_frames = 1

    # 원본 프레임 타임스탬프 (정렬된 오름차순)
    original_timestamps = np.arange(n_original, dtype=np.float64) / original_fps

    resampled: list[np.ndarray] = []
    for i in range(n_frames):
        target_ts = i / float(target_fps)
        # nearest-neighbor: 정렬된 배열에서 삽입 위치 양옆 비교
        pos = int(np.searchsorted(original_timestamps, target_ts))
        if pos == 0:
            nearest_idx = 0
        elif pos >= n_original:
            nearest_idx = n_original - 1
        else:
            before = original_timestamps[pos - 1]
            after = original_timestamps[pos]
            nearest_idx = pos - 1 if (target_ts - before) <= (after - target_ts) else pos
        resampled.append(frames[nearest_idx])

    return resampled, original_fps
