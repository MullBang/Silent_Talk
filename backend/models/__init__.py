"""모델 패키지 — 한국어 자모(grapheme) 어휘 정의 포함.

CTC 클래스 = 자음 19 + 모음 21 + blank 1 = 41 (설계문서 1.5절).
blank은 인덱스 0으로 고정한다 (CTC 표준).
"""

from __future__ import annotations

# 초성/단독 자음 19개 (한글 초성 순서)
CONSONANTS: list[str] = [
    "ㄱ", "ㄲ", "ㄴ", "ㄷ", "ㄸ", "ㄹ", "ㅁ", "ㅂ", "ㅃ", "ㅅ",
    "ㅆ", "ㅇ", "ㅈ", "ㅉ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ",
]

# 중성 모음 21개 (한글 중성 순서)
VOWELS: list[str] = [
    "ㅏ", "ㅐ", "ㅑ", "ㅒ", "ㅓ", "ㅔ", "ㅕ", "ㅖ", "ㅗ", "ㅘ",
    "ㅙ", "ㅚ", "ㅛ", "ㅜ", "ㅝ", "ㅞ", "ㅟ", "ㅠ", "ㅡ", "ㅢ", "ㅣ",
]

BLANK_TOKEN: str = "<blank>"
BLANK_IDX: int = 0

# 인덱스 0 = blank, 1~19 = 자음, 20~40 = 모음
GRAPHEME_VOCAB: list[str] = [BLANK_TOKEN] + CONSONANTS + VOWELS
NUM_CLASSES: int = len(GRAPHEME_VOCAB)  # 41

GRAPHEME_TO_IDX: dict[str, int] = {g: i for i, g in enumerate(GRAPHEME_VOCAB)}
IDX_TO_GRAPHEME: dict[int, str] = {i: g for i, g in enumerate(GRAPHEME_VOCAB)}

# 종성(받침)으로 올 수 없는 쌍자음 (ㄸ, ㅃ, ㅉ) 제외
VALID_TAILS: set[str] = set(CONSONANTS) - {"ㄸ", "ㅃ", "ㅉ"}

_CONSONANT_SET = set(CONSONANTS)
_VOWEL_SET = set(VOWELS)


def is_consonant(grapheme: str) -> bool:
    """자음 여부."""
    return grapheme in _CONSONANT_SET


def is_vowel(grapheme: str) -> bool:
    """모음 여부."""
    return grapheme in _VOWEL_SET
