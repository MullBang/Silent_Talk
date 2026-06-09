"""decoder 단위 테스트.

CTC prefix beam search 디코딩 + 자모→음절 복원, 신뢰도/raw_score 산출을 검증한다.
"""

from __future__ import annotations

import os
import sys

import numpy as np

# backend 패키지 루트를 import 경로에 추가
_BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

from models import GRAPHEME_TO_IDX  # noqa: E402
from models.decoder import (  # noqa: E402
    compose_jamos,
    ctc_beam_search_decode,
    get_confidence,
)


def _confident_log_probs(target_indices: list[int], n_class: int = 41) -> np.ndarray:
    """각 프레임에서 target index에 확신을 둔 log-softmax 배열을 만든다."""
    logits = np.zeros((len(target_indices), n_class), dtype=np.float64)
    for t, idx in enumerate(target_indices):
        logits[t, idx] = 12.0
    # log-softmax
    m = logits.max(axis=1, keepdims=True)
    log_sum = m + np.log(np.exp(logits - m).sum(axis=1, keepdims=True))
    return logits - log_sum


def test_compose_single_syllable() -> None:
    """[ㅇ, ㅏ, ㄴ] → '안'."""
    assert compose_jamos(["ㅇ", "ㅏ", "ㄴ"]) == "안"


def test_compose_multi_syllable() -> None:
    """[ㄱ, ㅏ, ㄴ, ㄷ, ㅏ] → '간다' (종성/초성 분리 그리디)."""
    assert compose_jamos(["ㄱ", "ㅏ", "ㄴ", "ㄷ", "ㅏ"]) == "간다"


def test_compose_no_tail() -> None:
    """[ㄱ, ㅏ] → '가' (종성 없음)."""
    assert compose_jamos(["ㄱ", "ㅏ"]) == "가"


def test_beam_search_decodes_known_syllable() -> None:
    """확신 있는 log_probs [ㅇ,ㅏ,ㄴ] → '안'."""
    idxs = [GRAPHEME_TO_IDX["ㅇ"], GRAPHEME_TO_IDX["ㅏ"], GRAPHEME_TO_IDX["ㄴ"]]
    lp = _confident_log_probs(idxs)
    assert ctc_beam_search_decode(lp, beam_width=10) == "안"


def test_beam_search_collapses_repeats_and_blanks() -> None:
    """blank/반복 프레임이 섞여도 [ㄱ,ㅏ] → '가'로 축약된다."""
    g = GRAPHEME_TO_IDX["ㄱ"]
    a = GRAPHEME_TO_IDX["ㅏ"]
    # ㄱ ㄱ blank ㅏ ㅏ → '가'
    lp = _confident_log_probs([g, g, 0, a, a])
    assert ctc_beam_search_decode(lp, beam_width=10) == "가"


def test_get_confidence_high_for_confident() -> None:
    """확신 입력은 confidence가 높고 raw_score는 0에 가깝다(음수)."""
    idxs = [GRAPHEME_TO_IDX["ㅇ"], GRAPHEME_TO_IDX["ㅏ"], GRAPHEME_TO_IDX["ㄴ"]]
    conf, raw = get_confidence(_confident_log_probs(idxs))
    assert 0.0 <= conf <= 1.0
    assert conf > 0.5
    assert raw <= 0.0


def test_get_confidence_low_for_uniform() -> None:
    """균일 분포(불확실) 입력은 confidence가 낮다."""
    lp = np.log(np.full((5, 41), 1.0 / 41))
    conf, raw = get_confidence(lp)
    assert conf < 0.1
    assert raw <= 0.0


def test_decode_accepts_3d_input() -> None:
    """(T, B, C) 입력도 첫 배치를 사용해 처리한다."""
    idxs = [GRAPHEME_TO_IDX["ㄱ"], GRAPHEME_TO_IDX["ㅏ"]]
    lp = _confident_log_probs(idxs)[:, None, :]  # (T, 1, C)
    assert ctc_beam_search_decode(lp) == "가"
