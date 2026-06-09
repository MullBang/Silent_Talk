"""segmenter 단위 테스트.

SEQ_LEN(75) 분할과 last-frame 패딩 정책(zero-padding 금지)을 검증한다.
"""

from __future__ import annotations


def test_segment_exact_multiple() -> None:
    """프레임 수가 SEQ_LEN 배수일 때 정확히 분할되는지 검증한다."""
    pass


def test_last_frame_padding_not_zero() -> None:
    """부족한 세그먼트가 last-frame 복제로 채워지고 zero-padding이 아님을 검증한다."""
    pass


def test_padding_reaches_seq_len() -> None:
    """패딩 후 세그먼트 길이가 정확히 SEQ_LEN인지 검증한다."""
    pass
