"""한국어 CTC 어휘: 초성 19 + 중성 21 + 종성 27 + 공백 + blank = 69."""

from .tokenizer import (
    BLANK_IDX, BLANK_TOKEN, GRAPHEME_TO_IDX, GRAPHEME_VOCAB,
    IDX_TO_GRAPHEME, NUM_CLASSES,
)
