"""decoder 단위 테스트.

CTC prefix beam search 디코딩 + 자모→음절 복원, 신뢰도/raw_score 산출을 검증한다.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import unicodedata

# backend 패키지 루트를 import 경로에 추가
_BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

from models import GRAPHEME_TO_IDX, NUM_CLASSES  # noqa: E402
from models.tokenizer import encode_text, TokenizerMismatchError
from models.decoder import (  # noqa: E402
    compose_jamos,
    ctc_beam_search_decode,
    ctc_greedy_decode,
    get_confidence,
)


def _confident_log_probs(target_indices: list[int], n_class: int = NUM_CLASSES) -> np.ndarray:
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
    assert compose_jamos(unicodedata.normalize("NFD", "안")) == "안"


def test_compose_multi_syllable() -> None:
    """[ㄱ, ㅏ, ㄴ, ㄷ, ㅏ] → '간다' (종성/초성 분리 그리디)."""
    assert compose_jamos(unicodedata.normalize("NFD", "간다")) == "간다"


def test_compose_no_tail() -> None:
    """[ㄱ, ㅏ] → '가' (종성 없음)."""
    assert compose_jamos(unicodedata.normalize("NFD", "가")) == "가"


def test_beam_search_decodes_known_syllable() -> None:
    """확신 있는 log_probs [ㅇ,ㅏ,ㄴ] → '안'."""
    idxs = encode_text("안")
    lp = _confident_log_probs(idxs)
    assert ctc_beam_search_decode(lp, beam_width=10) == "안"


def test_beam_search_collapses_repeats_and_blanks() -> None:
    """blank/반복 프레임이 섞여도 [ㄱ,ㅏ] → '가'로 축약된다."""
    g = encode_text("가")[0]
    a = encode_text("가")[1]
    # ㄱ ㄱ blank ㅏ ㅏ → '가'
    lp = _confident_log_probs([g, g, 0, a, a])
    assert ctc_beam_search_decode(lp, beam_width=10) == "가"


def test_get_confidence_high_for_confident() -> None:
    """확신 입력은 confidence가 높고 raw_score는 0에 가깝다(음수)."""
    idxs = encode_text("안")
    conf, raw = get_confidence(_confident_log_probs(idxs))
    assert 0.0 <= conf <= 1.0
    assert conf > 0.5
    assert raw <= 0.0


def test_get_confidence_low_for_uniform() -> None:
    """균일 분포(불확실) 입력은 confidence가 낮다."""
    lp = np.log(np.full((5, NUM_CLASSES), 1.0 / NUM_CLASSES))
    conf, raw = get_confidence(lp)
    assert conf < 0.1
    assert raw <= 0.0


def test_decode_accepts_3d_input() -> None:
    """(T, B, C) 입력도 첫 배치를 사용해 처리한다."""
    idxs = encode_text("가")
    lp = _confident_log_probs(idxs)[:, None, :]  # (T, 1, C)
    assert ctc_beam_search_decode(lp) == "가"


def test_beam_preserves_compound_tails_and_spaces():
    text = "값이 없었어"
    frames = []
    for idx in encode_text(text):
        frames.extend([idx, idx, 0])
    assert ctc_beam_search_decode(_confident_log_probs(frames)) == text


def test_legacy_41_class_output_is_rejected():
    with pytest.raises(TokenizerMismatchError):
        ctc_beam_search_decode(np.zeros((3, 41)))


def test_greedy_decodes_repeated_frames_and_compound_tail():
    text = "값이 없었어"
    frames = [value for token in encode_text(text) for value in (token, token, 0)]
    assert ctc_greedy_decode(_confident_log_probs(frames)[:, None, :]) == text


def test_greedy_preserves_tokens_separated_by_blank():
    # 두 초성이 연속이면 잘못된 자모이지만, 둘을 임의로 합쳐 오류를 숨기면 안 된다.
    token = encode_text("가")[0]
    assert ctc_greedy_decode(_confident_log_probs([token, token, 0, token])) == "��"


def test_greedy_blank_and_empty_outputs():
    assert ctc_greedy_decode(_confident_log_probs([0, 0, 0])) == ""
    assert ctc_greedy_decode(np.empty((0, NUM_CLASSES))) == ""
    with pytest.raises(TokenizerMismatchError):
        ctc_greedy_decode(np.zeros((2, 41)))
