"""CTC 디코더: Prefix Beam Search + 자모음 → 음절 복원.

모델의 CTC log-prob 출력을 prefix beam search로 디코딩하고, 위치별 Unicode 자모로
자모(초성/중성/종성)를 완성형 한글 음절로 복원한다. confidence(0~1)와
raw_score(CTC log-prob 원본)를 분리 반환한다.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from . import BLANK_IDX, IDX_TO_GRAPHEME, NUM_CLASSES
from .tokenizer import TokenizerMismatchError, compose_jamos


_NEG_INF = -1e30


def _to_numpy(log_probs) -> np.ndarray:
    """log_probs를 (T, C) numpy 배열로 변환한다.

    (T, B, C)가 들어오면 첫 번째 배치만 사용한다.
    """
    arr = log_probs.detach().cpu().numpy() if hasattr(log_probs, "detach") else np.asarray(log_probs)
    if arr.ndim == 3:
        arr = arr[:, 0, :]
    if arr.ndim != 2:
        raise ValueError(f"log_probs 형상이 (T, C)가 아닙니다: {arr.shape}")
    return arr


def ctc_beam_search_decode(log_probs, beam_width: int = 10) -> str:
    """CTC prefix beam search 디코딩 후 한글로 복원한다.

    Args:
        log_probs: (T, C) 또는 (T, B, C) CTC log-softmax 출력.
        beam_width: 빔 폭.

    Returns:
        복원된 한국어 텍스트 문자열.
    """
    lp = _to_numpy(log_probs)
    t_len, n_class = lp.shape
    if n_class != NUM_CLASSES:
        raise TokenizerMismatchError(f"모델 출력은 {NUM_CLASSES}클래스여야 합니다 (수신 {n_class}).")
    if beam_width < 1:
        raise ValueError("beam_width는 1 이상이어야 합니다.")

    # prefix(tuple[int]) -> (log p_blank, log p_nonblank)
    beams: dict[tuple, tuple[float, float]] = {(): (0.0, _NEG_INF)}

    for t in range(t_len):
        nxt: dict[tuple, list[float]] = defaultdict(lambda: [_NEG_INF, _NEG_INF])
        for prefix, (p_b, p_nb) in beams.items():
            p_total = np.logaddexp(p_b, p_nb)
            last = prefix[-1] if prefix else -1

            # blank → prefix 유지
            cur = nxt[prefix]
            cur[0] = np.logaddexp(cur[0], p_total + lp[t, BLANK_IDX])

            for c in range(n_class):
                if c == BLANK_IDX:
                    continue
                p = lp[t, c]
                if c == last:
                    # 같은 문자 반복: blank 경로에서만 새 토큰으로 확장
                    ext = nxt[prefix + (c,)]
                    ext[1] = np.logaddexp(ext[1], p_b + p)
                    # 같은 prefix 유지(반복 흡수)는 nonblank 경로에서
                    same = nxt[prefix]
                    same[1] = np.logaddexp(same[1], p_nb + p)
                else:
                    ext = nxt[prefix + (c,)]
                    ext[1] = np.logaddexp(ext[1], p_total + p)

        # 점수 상위 beam_width개만 유지
        scored = sorted(
            nxt.items(),
            key=lambda kv: np.logaddexp(kv[1][0], kv[1][1]),
            reverse=True,
        )
        beams = {k: (v[0], v[1]) for k, v in scored[:beam_width]}

    best_prefix = max(
        beams.items(), key=lambda kv: np.logaddexp(kv[1][0], kv[1][1])
    )[0]
    graphemes = [IDX_TO_GRAPHEME[i] for i in best_prefix]
    return compose_jamos(graphemes)


def ctc_greedy_decode(log_probs) -> str:
    """프레임별 최댓값의 연속 반복을 접고 blank를 제거한다(학습 중 빠른 확인용)."""
    lp = _to_numpy(log_probs)
    if lp.shape[1] != NUM_CLASSES:
        raise TokenizerMismatchError(f"모델 출력은 {NUM_CLASSES}클래스여야 합니다.")
    tokens = []
    previous = None
    for value in lp.argmax(axis=-1).tolist():
        if value != previous and value != BLANK_IDX:
            tokens.append(IDX_TO_GRAPHEME[value])
        previous = value  # blank도 기록해야 blank 사이의 같은 토큰 두 개를 보존한다.
    return compose_jamos(tokens)


def get_confidence(log_probs) -> tuple[float, float]:
    """디코딩 신뢰도와 raw_score를 계산한다.

    - raw_score: 프레임별 최대 log-prob의 합 (CTC log-prob 원본, 음수).
    - confidence: exp(raw_score / T) = 프레임별 최대 확률의 기하평균 → [0,1].

    Args:
        log_probs: (T, C) 또는 (T, B, C) CTC log-softmax 출력.

    Returns:
        (confidence[0~1 정규화], raw_score[CTC log-prob 합]).
    """
    lp = _to_numpy(log_probs)
    t_len = lp.shape[0]
    if t_len == 0:
        return 0.0, 0.0
    best_per_frame = lp.max(axis=1)  # (T,)
    raw_score = float(best_per_frame.sum())
    confidence = float(np.exp(raw_score / t_len))
    confidence = max(0.0, min(1.0, confidence))
    return confidence, raw_score


def decode(log_probs, beam_width: int = 10) -> tuple[str, float, float]:
    """디코딩 전체 파이프라인 (Beam Search → 자모 복원 → 신뢰도 산출).

    Args:
        log_probs: (T, C) 또는 (T, B, C) CTC log-softmax 출력.
        beam_width: 빔 폭.

    Returns:
        (transcript, confidence[0~1], raw_score[CTC log-prob 원본]).
    """
    text = ctc_beam_search_decode(log_probs, beam_width=beam_width)
    confidence, raw_score = get_confidence(log_probs)
    return text, confidence, raw_score
