"""3초 세그먼트 분할.

리샘플된 프레임 시퀀스를 SEQ_LEN(75프레임 = 3초) 단위로 분할한다.
마지막 세그먼트가 부족하면 zero-padding 금지 — last-frame 복제로 채운다.
"""

from __future__ import annotations

import numpy as np

from config import SEQ_LEN


def segment_frames(
    frames: list[np.ndarray],
    seq_len: int = SEQ_LEN,
) -> list[list[np.ndarray]]:
    """프레임 시퀀스를 seq_len 단위 세그먼트로 분할한다.

    부족한 마지막 세그먼트는 PADDING='last-frame' 정책에 따라 마지막
    프레임을 복제하여 채운다 (zero-padding 절대 금지).

    Args:
        frames: 25fps로 리샘플된 프레임 리스트.
        seq_len: 세그먼트당 프레임 수 (기본 SEQ_LEN=75).

    Returns:
        각각 seq_len 길이를 갖는 세그먼트들의 리스트.
    """
    pass


def pad_last_frame(
    segment: list[np.ndarray],
    seq_len: int = SEQ_LEN,
) -> list[np.ndarray]:
    """세그먼트를 마지막 프레임 복제로 seq_len까지 패딩한다.

    Args:
        segment: 길이가 seq_len 이하인 프레임 세그먼트.
        seq_len: 목표 길이 (기본 SEQ_LEN=75).

    Returns:
        정확히 seq_len 길이의 패딩된 세그먼트.
    """
    pass
