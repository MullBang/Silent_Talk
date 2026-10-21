"""Pure helpers for transparent diagnostic output and micro CER counts."""
from __future__ import annotations

import unicodedata


def collapse_ctc(path: list[int], blank_id: int = 3) -> list[int]:
    """Collapse adjacent repeats BEFORE blank removal, preserving EOS for audit."""
    result, previous = [], None
    for token in path:
        if token != previous and token != blank_id:
            result.append(token)
        previous = token
    return result


def edit_distance(reference, prediction) -> int:
    """Levenshtein distance over explicit characters or token sequences."""
    previous = list(range(len(prediction) + 1))
    for i, ref in enumerate(reference, 1):
        current = [i]
        for j, pred in enumerate(prediction, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (ref != pred)))
        previous = current
    return previous[-1]


def canonical_text(text: str) -> str:
    """NFC plus normalized whitespace; no spell correction or punctuation erasure."""
    return " ".join(unicodedata.normalize("NFC", text).split())


def text_counts(reference: str, prediction: str) -> dict:
    """Store numerator/denominator so pooled CER cannot be confused with mean CER."""
    reference, prediction = canonical_text(reference), canonical_text(prediction)
    return {
        "character_edits_with_spaces": edit_distance(reference, prediction),
        "reference_characters_with_spaces": len(reference),
        "character_edits_without_spaces": edit_distance(reference.replace(" ", ""), prediction.replace(" ", "")),
        "reference_characters_without_spaces": len(reference.replace(" ", "")),
        "exact_sentence": reference == prediction,
    }
