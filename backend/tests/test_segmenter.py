"""segmenter 단위 테스트.

설계문서 1.3절 '세그먼트 분할 처리' 검증.

SEQ_LEN(75) 순차 분할, last-frame padding 정책(zero-padding 금지),
세그먼트 타임스탬프(start_ms/end_ms)를 검증한다.
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

from config import FPS, SEQ_LEN  # noqa: E402
from preprocessing.segmenter import split_into_segments  # noqa: E402

_H = 8
_W = 8


def _make_frames(n: int) -> list[np.ndarray]:
    """프레임마다 고유 밝기값을 가진 프레임 리스트를 생성한다."""
    return [
        np.full((_H, _W, 3), (i * 2) % 256, dtype=np.uint8) for i in range(n)
    ]


def test_150_frames_two_segments() -> None:
    """150프레임 → 2 세그먼트(각 75프레임)로 분할되는지 검증한다."""
    frames = _make_frames(150)
    segments = split_into_segments(frames, fps=FPS, seq_len=SEQ_LEN)

    assert len(segments) == 2
    for seg in segments:
        assert seg["frames"].shape == (SEQ_LEN, _H, _W, 3)


def test_100_frames_two_segments_with_padding() -> None:
    """100프레임 → 2 세그먼트, 마지막은 25프레임 last-frame padding 검증."""
    frames = _make_frames(100)
    segments = split_into_segments(frames, fps=FPS, seq_len=SEQ_LEN)

    assert len(segments) == 2
    last = segments[1]["frames"]
    assert last.shape == (SEQ_LEN, _H, _W, 3)

    # 마지막 유효 프레임(원본 99번)이 패딩 구간(인덱스 25~74)에 반복되어야 한다
    last_valid = frames[99]
    for i in range(25, SEQ_LEN):
        assert np.array_equal(last[i], last_valid)

    # zero-padding이 아님을 확인 (패딩 프레임이 전부 0이 아니어야 함)
    assert last[SEQ_LEN - 1].any()


def test_segment_timestamps() -> None:
    """start_ms/end_ms가 3초(3000ms) 단위로 부여되는지 검증한다."""
    frames = _make_frames(150)
    segments = split_into_segments(frames, fps=FPS, seq_len=SEQ_LEN)

    assert segments[0]["start_ms"] == 0
    assert segments[0]["end_ms"] == 3000
    assert segments[1]["start_ms"] == 3000
    assert segments[1]["end_ms"] == 6000


def test_exact_single_segment_no_padding() -> None:
    """정확히 75프레임이면 패딩 없이 1 세그먼트로 분할된다."""
    frames = _make_frames(SEQ_LEN)
    segments = split_into_segments(frames)

    assert len(segments) == 1
    assert segments[0]["frames"].shape == (SEQ_LEN, _H, _W, 3)
    # 마지막 프레임이 원본 그대로(패딩 없음)
    assert np.array_equal(segments[0]["frames"][-1], frames[-1])


def test_empty_input_returns_empty() -> None:
    """빈 입력은 빈 리스트를 반환한다."""
    assert split_into_segments([]) == []


def test_invalid_params_raise() -> None:
    """fps/seq_len이 양수가 아니면 예외가 발생한다."""
    frames = _make_frames(10)
    with pytest.raises(ValueError):
        split_into_segments(frames, fps=0)
    with pytest.raises(ValueError):
        split_into_segments(frames, seq_len=0)
