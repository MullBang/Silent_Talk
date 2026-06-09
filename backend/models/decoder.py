"""CTC 디코더: Prefix Beam Search + 자모음 → 음절 복원.

모델의 CTC log-prob 출력을 prefix beam search로 디코딩하고, python-jamo로
자모(초성/중성/종성)를 완성형 한글 음절로 복원한다. confidence(0~1)와
raw_score(CTC log-prob 원본)를 분리 반환한다.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from jamo import j2h

from . import (
    BLANK_IDX,
    IDX_TO_GRAPHEME,
    VALID_TAILS,
    is_consonant,
    is_vowel,
)

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


def compose_jamos(graphemes: list[str]) -> str:
    """자모 시퀀스를 완성형 한글 문자열로 복원한다.

    초성(자음) + 중성(모음) [+ 종성(자음)] 패턴으로 그리디 결합한다.
    - 자음 뒤 자음이 오면, 다음이 모음이면 그 자음은 다음 음절 초성(종성 아님).
    - 종성으로 불가능한 쌍자음(ㄸ/ㅃ/ㅉ)은 종성으로 쓰지 않는다.

    Args:
        graphemes: 디코딩된 자모 문자 리스트.

    Returns:
        복원된 한글 텍스트 (결합 불가 자모는 제외).
    """
    out: list[str] = []
    i = 0
    n = len(graphemes)
    while i < n:
        lead = graphemes[i]
        # 초성(자음) + 중성(모음) 필요
        if is_consonant(lead) and i + 1 < n and is_vowel(graphemes[i + 1]):
            vowel = graphemes[i + 1]
            i += 2
            tail = ""
            # 종성 후보: 다음 자음이 (끝이거나 그 다음이 자음일 때) 종성
            if i < n and is_consonant(graphemes[i]) and graphemes[i] in VALID_TAILS:
                is_last = i + 1 >= n
                next_is_consonant = (not is_last) and is_consonant(graphemes[i + 1])
                if is_last or next_is_consonant:
                    tail = graphemes[i]
                    i += 1
            try:
                out.append(j2h(lead, vowel, tail) if tail else j2h(lead, vowel))
            except Exception:  # noqa: BLE001 - 잘못된 조합은 건너뜀
                pass
        else:
            # 단독 자모/특수 토큰 제거
            i += 1
    return "".join(out)


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
