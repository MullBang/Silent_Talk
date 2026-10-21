"""겹받침, 공백, 전체 현대 한글, 잘못된 입력 및 평가 계약 회귀 검사."""

import sys
import unicodedata
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.tokenizer import (
    GRAPHEME_TO_IDX, LEADS, TAILS, NUM_CLASSES, UnsupportedTextError,
    TokenizerMismatchError, cer, wer, compose_jamos, ctc_min_input_len,
    decode_labels, encode_text, normalize_text, tokenizer_metadata, validate_tokenizer_metadata,
)


@pytest.mark.parametrize("text", [
    "나이가 많다고", "인상은 없었어", "값이 비싸다", "책을 읽다",
    "앉다 않다 삶 넓다 외곬 핥다 읊다 싫다", "괜찮아 정말 괜찮아", "가가 까까 아아",
])
def test_roundtrip_preserves_text(text):
    assert decode_labels(encode_text(text)) == text


def test_all_11172_modern_hangul_syllables_roundtrip():
    for codepoint in range(0xAC00, 0xD7A4):
        text = chr(codepoint)
        assert decode_labels(encode_text(text)) == text


def test_normalization_is_explicit_and_idempotent():
    original = "  ‘많다고,’\t말했어.\n괜찮아?  "
    normalized = "많다고 말했어 괜찮아"
    assert normalize_text(original) == normalized
    assert normalize_text(normalized) == normalized
    assert decode_labels(encode_text(unicodedata.normalize("NFD", original))) == normalized


@pytest.mark.parametrize("text", ["X 이유", "아무XX 쓸모", "3명", "AI 기술", "ㄱㅏ", "좋아🙂", "가\x00나"])
def test_unsupported_characters_are_not_silently_deleted(text):
    with pytest.raises(UnsupportedTextError):
        encode_text(text)


def test_initial_and_final_consonants_are_distinct_tokens():
    assert NUM_CLASSES == 69
    assert GRAPHEME_TO_IDX[LEADS[0]] != GRAPHEME_TO_IDX[TAILS[0]]
    assert decode_labels(encode_text("각 가")) == "각 가"


@pytest.mark.parametrize("indices", [[0], [-1], [69], [1.5]])
def test_invalid_label_indices_are_rejected(indices):
    with pytest.raises(ValueError):
        decode_labels(indices)


def test_bad_predictions_count_as_errors_instead_of_disappearing():
    assert compose_jamos([LEADS[0]]) == "�"
    assert cer("가", "�가") > 0
    with pytest.raises(ValueError):
        decode_labels([GRAPHEME_TO_IDX[TAILS[0]]])


def test_ctc_minimum_includes_repeated_tokens():
    assert ctc_min_input_len([1, 1, 2, 2]) == 6
    assert ctc_min_input_len([]) == 0


def test_metrics_share_normalization_and_keep_word_boundaries():
    assert cer("많다고 말했어.", "많다고 말했어") == 0
    assert wer("많다고 말했어!", "많다고 말했어") == 0
    assert cer("많다고", "마다고") > 0
    assert wer("값이 비싸다", "값이비싸다") == 1
    assert cer("", "가") == 1


def test_legacy_or_wrong_vocabulary_metadata_is_rejected():
    validate_tokenizer_metadata(tokenizer_metadata())
    with pytest.raises(TokenizerMismatchError):
        validate_tokenizer_metadata({"num_classes": 41})
    bad = {**tokenizer_metadata(), "vocab_sha256": "wrong-order"}
    with pytest.raises(TokenizerMismatchError):
        validate_tokenizer_metadata(bad)
