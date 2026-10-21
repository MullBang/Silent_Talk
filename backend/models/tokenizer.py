"""손실 없는 현대 한글 토큰화. 영상/torch/jamo 의존성이 없는 공용 계약.

NFC + 문장부호를 공백으로 치환 + 공백 축약 후 현대 한글 음절만 허용한다.
숫자/영문/X/기호/단독 자모는 추정하거나 삭제하지 않고 명시적으로 거부한다.
"""

from __future__ import annotations

import hashlib
import json
from numbers import Integral
from typing import Iterable, Mapping
import unicodedata

TOKENIZER_VERSION = "ko-jamo-v2"
NORMALIZATION_VERSION = "ko-text-v1"
LEADS = tuple(chr(i) for i in range(0x1100, 0x1113))
VOWELS = tuple(chr(i) for i in range(0x1161, 0x1176))
TAILS = tuple(chr(i) for i in range(0x11A8, 0x11C3))
SPACE_TOKEN = " "
BLANK_TOKEN = "<blank>"
BLANK_IDX = 0
GRAPHEME_VOCAB = [BLANK_TOKEN, *LEADS, *VOWELS, *TAILS, SPACE_TOKEN]
NUM_CLASSES = len(GRAPHEME_VOCAB)
GRAPHEME_TO_IDX = {g: i for i, g in enumerate(GRAPHEME_VOCAB)}
IDX_TO_GRAPHEME = dict(enumerate(GRAPHEME_VOCAB))
VOCAB_SHA256 = hashlib.sha256(
    json.dumps(GRAPHEME_VOCAB, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
).hexdigest()


class UnsupportedTextError(ValueError):
    """정답을 임의 삭제 없이 인코딩할 수 없을 때 발생한다."""


class TokenizerMismatchError(ValueError):
    """서로 다른 토큰 사전의 캐시/체크포인트 사용을 차단한다."""


def normalize_text(text: str) -> str:
    """정해진 규칙으로 정규화하며 지원하지 않는 문자는 오류로 보고한다."""
    if not isinstance(text, str):
        raise TypeError("정답 텍스트는 문자열이어야 합니다.")
    text = unicodedata.normalize("NFC", text)
    cleaned, unsupported = [], set()
    for ch in text:
        if "가" <= ch <= "힣":
            cleaned.append(ch)
        elif ch.isspace() or unicodedata.category(ch).startswith("P"):
            cleaned.append(" ")
        else:
            unsupported.add(ch)
    if unsupported:
        details = ", ".join(f"{ch!r}(U+{ord(ch):04X})" for ch in sorted(unsupported))
        raise UnsupportedTextError(f"지원하지 않는 정답 문자: {details}. 발음/표기 확인이 필요합니다.")
    return " ".join("".join(cleaned).split())


def encode_text(text: str) -> list[int]:
    """정규화된 문장을 위치가 구분된 Unicode 자모 + 공백 인덱스로 변환한다."""
    return [GRAPHEME_TO_IDX[ch] for ch in unicodedata.normalize("NFD", normalize_text(text))]


def compose_jamos(graphemes: Iterable[str], *, strict: bool = False) -> str:
    """초성+중성[+종성]을 조합하며 잘못된 예측 자모는 U+FFFD로 표시한다.

    strict=True는 정답 검증용이다. 추론의 불완전한 자모는 삭제하지 않고
    '�'로 남겨 평가 시 인식 오류로 계산한다.
    """
    tokens = list(graphemes)
    out: list[str] = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token == SPACE_TOKEN:
            out.append(token)
            i += 1
        elif token in LEADS and i + 1 < len(tokens) and tokens[i + 1] in VOWELS:
            parts = tokens[i:i + 2]
            i += 2
            if i < len(tokens) and tokens[i] in TAILS:
                parts.append(tokens[i])
                i += 1
            out.append(unicodedata.normalize("NFC", "".join(parts)))
        else:
            if strict:
                raise ValueError(f"잘못된 자모 순서/토큰 (위치 {i}): {token!r}")
            out.append("\ufffd")
            i += 1
    return "".join(out)


def decode_labels(indices: Iterable[int]) -> str:
    """CTC 타깃 인덱스를 엄격하게 복원한다 (프레임 반복 축약은 하지 않음)."""
    tokens = []
    for idx in indices:
        if not isinstance(idx, Integral) or not 0 < int(idx) < NUM_CLASSES:
            raise ValueError(f"잘못된 정답 토큰 인덱스: {idx!r}")
        tokens.append(IDX_TO_GRAPHEME[int(idx)])
    return compose_jamos(tokens, strict=True)


def ctc_min_input_len(indices: Iterable[int]) -> int:
    """CTC가 타깃을 정렬할 수 있는 최소 프레임 수를 반환한다."""
    labels = list(indices)
    return len(labels) + sum(a == b for a, b in zip(labels, labels[1:]))


def tokenizer_metadata() -> dict[str, str | int]:
    """NPZ와 체크포인트에서 공통으로 검증할 직렬화 가능한 메타데이터."""
    return {"tokenizer_version": TOKENIZER_VERSION,
            "normalization_version": NORMALIZATION_VERSION,
            "vocab_sha256": VOCAB_SHA256, "num_classes": NUM_CLASSES}


def validate_tokenizer_metadata(metadata: Mapping, source: str = "데이터") -> None:
    """버전 없는 구형 41클래스 자료와 다른 사전을 명시적으로 거부한다."""
    for key, expected in tokenizer_metadata().items():
        if key not in metadata or str(metadata[key]) != str(expected):
            raise TokenizerMismatchError(
                f"{source}: 토큰화 메타데이터 불일치({key}). "
                "구형 41클래스 캐시는 migrate_labels.py로 변환하고 가중치는 새로 학습해야 합니다."
            )


def _metric_text(text: str) -> str:
    """평가용 정규화. 잘못된 예측 문자도 삭제하지 않고 오류로 계산한다."""
    return " ".join("".join(
        " " if ch.isspace() or unicodedata.category(ch).startswith("P") else ch
        for ch in unicodedata.normalize("NFC", text)
    ).split())


def _edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    """두 토큰 열의 편집 거리."""
    previous = list(range(len(hypothesis) + 1))
    for i, ref in enumerate(reference, 1):
        current = [i]
        for j, hyp in enumerate(hypothesis, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (ref != hyp)))
        previous = current
    return previous[-1]


def _error_rate(reference: list[str], hypothesis: list[str]) -> float:
    """빈 정답을 포함하는 오류율 계산."""
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return _edit_distance(reference, hypothesis) / len(reference)


def cer(reference: str, hypothesis: str) -> float:
    """위치별 자모 및 단일 공백을 포함한 CER. 잘못된 예측은 오류로 유지."""
    ref = list(unicodedata.normalize("NFD", normalize_text(reference)))
    hyp = list(unicodedata.normalize("NFD", _metric_text(hypothesis)))
    return _error_rate(ref, hyp)


def wer(reference: str, hypothesis: str) -> float:
    """동일 문장부호/공백 정규화 후 어절 WER."""
    return _error_rate(normalize_text(reference).split(), _metric_text(hypothesis).split())
