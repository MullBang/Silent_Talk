"""CTC 디코더: Beam Search + 자모음 복원.

모델의 CTC log-prob 출력을 Beam Search로 디코딩하고 python-jamo로 자모를
완성형 한글 음절로 복원한다. confidence(0~1)와 raw_score(log-prob 원본)를 분리 반환.
"""

from __future__ import annotations

import torch


def ctc_beam_search(
    log_probs: torch.Tensor,
    beam_width: int = 10,
) -> tuple[list[int], float]:
    """CTC Beam Search 디코딩.

    Args:
        log_probs: (T, num_classes) CTC log-softmax 출력.
        beam_width: 빔 폭.

    Returns:
        (디코딩된 클래스 인덱스 시퀀스, raw_score=CTC log-prob 원본).
    """
    pass


def restore_jamo(indices: list[int]) -> str:
    """자모 인덱스 시퀀스를 완성형 한글 문자열로 복원한다.

    python-jamo를 사용하여 초성/중성/종성을 결합한다.

    Args:
        indices: CTC 디코딩 결과 클래스 인덱스 시퀀스.

    Returns:
        복원된 한글 텍스트.
    """
    pass


def decode(
    log_probs: torch.Tensor,
    beam_width: int = 10,
) -> tuple[str, float, float]:
    """디코딩 전체 파이프라인 (Beam Search → 자모 복원 → 신뢰도 산출).

    Args:
        log_probs: (T, num_classes) CTC log-softmax 출력.
        beam_width: 빔 폭.

    Returns:
        (transcript, confidence[0~1 정규화], raw_score[CTC log-prob 원본]).
    """
    pass
