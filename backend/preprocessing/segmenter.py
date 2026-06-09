"""3초 세그먼트 분할.

설계문서 1.3절 '세그먼트 분할 처리' 구현.

리샘플된 프레임 시퀀스를 seq_len(75프레임 = 3초) 단위로 오버랩 없이 순차
분할한다. 마지막 세그먼트가 seq_len 미만이면 zero-padding 대신 last-frame
padding(마지막 유효 프레임 반복)을 적용한다.
"""

from __future__ import annotations

import numpy as np

from config import FPS, SEQ_LEN


def _pad_last_frame(
    segment: list[np.ndarray],
    seq_len: int,
) -> list[np.ndarray]:
    """세그먼트를 마지막 유효 프레임 복제로 seq_len까지 패딩한다.

    PADDING='last-frame' 정책 — zero-padding 절대 금지.

    Args:
        segment: 길이가 seq_len 이하인 (비어 있지 않은) 프레임 리스트.
        seq_len: 목표 길이 (기본 SEQ_LEN=75).

    Returns:
        정확히 seq_len 길이로 패딩된 프레임 리스트.

    Raises:
        ValueError: 입력 세그먼트가 비어 있는 경우(패딩 기준 프레임 없음).
    """
    if not segment:
        raise ValueError("빈 세그먼트는 패딩할 수 없습니다 (기준 프레임 없음).")

    if len(segment) >= seq_len:
        return segment[:seq_len]

    last_frame = segment[-1]
    pad_count = seq_len - len(segment)
    return segment + [last_frame.copy() for _ in range(pad_count)]


def split_into_segments(
    frames: list[np.ndarray],
    fps: int = FPS,
    seq_len: int = SEQ_LEN,
) -> list[dict]:
    """리샘플된 프레임을 3초(seq_len) 단위 세그먼트로 분할한다.

    오버랩 없이 순차 분할하며, 각 세그먼트의 시작/종료 타임스탬프(ms)를 함께
    부여한다. 마지막 세그먼트가 seq_len 미만이면 last-frame padding을 적용한다
    (zero-padding 절대 금지).

    Args:
        frames: 리샘플된 프레임 리스트 (각 프레임은 (H, W, 3) ndarray).
        fps: 프레임 레이트 (기본 FPS=25). 세그먼트 시간 길이 계산에 사용.
        seq_len: 세그먼트당 프레임 수 (기본 SEQ_LEN=75).

    Returns:
        세그먼트 dict 리스트. 각 dict는
        {
            'frames': np.ndarray,  # (seq_len, H, W, 3)
            'start_ms': int,
            'end_ms': int,
        }.
        입력 frames가 비어 있으면 빈 리스트를 반환한다.

    Raises:
        ValueError: fps 또는 seq_len이 양수가 아닌 경우.
    """
    if fps <= 0:
        raise ValueError(f"fps는 양수여야 합니다: {fps}")
    if seq_len <= 0:
        raise ValueError(f"seq_len은 양수여야 합니다: {seq_len}")

    if not frames:
        return []

    # 세그먼트 1개의 시간 길이(ms): 75 / 25 * 1000 == 3000
    segment_ms = int(round(seq_len / fps * 1000))

    segments: list[dict] = []
    for seg_idx, start in enumerate(range(0, len(frames), seq_len)):
        chunk = frames[start:start + seq_len]
        padded = _pad_last_frame(chunk, seq_len)

        start_ms = seg_idx * segment_ms
        end_ms = start_ms + segment_ms

        segments.append(
            {
                "frames": np.stack(padded, axis=0),  # (seq_len, H, W, 3)
                "start_ms": start_ms,
                "end_ms": end_ms,
            }
        )

    return segments
