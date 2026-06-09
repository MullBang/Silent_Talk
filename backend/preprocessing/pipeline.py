"""전처리 파이프라인 오케스트레이터.

설계문서 2단계 — resampler · segmenter · roi_extractor를 연결한다.

처리 흐름:
  ① get_video_duration_sec()  → MAX_DURATION_SEC 초과 시 예외
  ② resample_to_fps()         → 25fps 프레임 리스트
  ③ split_into_segments()     → 3초 세그먼트 리스트
  ④ extract_roi_from_segment()→ None(미검출)이면 해당 세그먼트 skip
  ⑤ normalize_roi() → to_model_tensor() → (1, 3, 75, 96, 96) 텐서
"""

from __future__ import annotations

import os
import sys

# backend 루트를 import 경로에 부트스트랩.
# - 프로젝트 루트에서 `from backend.preprocessing.pipeline import ...` 호출 시
# - pytest 실행 시
# 모두 `from config import ...` / `from preprocessing.X import ...`가 동작하도록 한다.
_BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

from preprocessing.resampler import (  # noqa: E402
    get_video_duration_sec,
    resample_to_fps,
)
from preprocessing.roi_extractor import (  # noqa: E402
    extract_roi_from_segment,
    normalize_roi,
    to_model_tensor,
)
from preprocessing.segmenter import split_into_segments  # noqa: E402

_FAIL_TEXT = "[검출 실패 구간]"


def run_preprocessing_pipeline(video_path: str) -> list[dict]:
    """영상을 전처리하여 세그먼트별 모델 입력 텐서 리스트를 생성한다.

    Args:
        video_path: 입력 영상 파일 경로.

    Returns:
        세그먼트 dict 리스트.
          - 처리 성공: {tensor: torch.Tensor(1,3,75,96,96), start_ms, end_ms,
                        skipped: False}
          - 검출 실패: {tensor: None, start_ms, end_ms, skipped: True,
                        text: '[검출 실패 구간]', confidence: None}

    Raises:
        FileNotFoundError: 영상을 열 수 없는 경우.
        ValueError: 재생 시간이 MAX_DURATION_SEC를 초과하거나 영상이 비정상인 경우.
    """
    # ① 길이 검증 (초과 시 ValueError)
    get_video_duration_sec(video_path)

    # ② 25fps 리샘플링
    frames, _original_fps = resample_to_fps(video_path)

    # ③ 3초 세그먼트 분할
    segments = split_into_segments(frames)

    results: list[dict] = []
    for segment in segments:
        start_ms = segment["start_ms"]
        end_ms = segment["end_ms"]

        try:
            # ④ ROI 추출 (연속 미검출 시 None)
            roi_uint8 = extract_roi_from_segment(segment["frames"])
        except Exception:
            roi_uint8 = None

        if roi_uint8 is None:
            results.append(
                {
                    "tensor": None,
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "skipped": True,
                    "text": _FAIL_TEXT,
                    "confidence": None,
                }
            )
            continue

        # ⑤ 정규화 → 모델 입력 텐서
        normalized = normalize_roi(roi_uint8)
        tensor = to_model_tensor(normalized)

        results.append(
            {
                "tensor": tensor,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "skipped": False,
            }
        )

    return results
