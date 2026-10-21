"""Explicit text metrics for full-cohort evaluation, including failed inputs."""
from __future__ import annotations

import unicodedata

from diagnostic_utils import edit_distance


def counts(reference: str, prediction: str) -> dict:
    """Keep internal spaces in primary CER; WER uses whitespace-separated words."""
    reference = unicodedata.normalize("NFC", reference).strip()
    prediction = unicodedata.normalize("NFC", prediction).strip()
    normalized_reference, normalized_prediction = " ".join(reference.split()), " ".join(prediction.split())
    return {
        "character_edits": edit_distance(reference, prediction),
        "reference_characters": len(reference),
        "word_edits": edit_distance(reference.split(), prediction.split()),
        "reference_words": len(reference.split()),
        "exact": reference == prediction,
        "normalized_character_edits": edit_distance(normalized_reference, normalized_prediction),
        "normalized_reference_characters": len(normalized_reference),
        "normalized_exact": normalized_reference == normalized_prediction,
    }


def aggregate(records: list[dict]) -> dict:
    """Use every frozen record. Failure records must carry an empty prediction."""
    if not records:
        raise ValueError("Cannot evaluate an empty cohort")
    if len({r["id"] for r in records}) != len(records):
        raise ValueError("Duplicate IDs in cohort")
    keys = ["character_edits", "reference_characters", "word_edits", "reference_words",
            "normalized_character_edits", "normalized_reference_characters"]
    values = {key: sum(r["metrics"][key] for r in records) for key in keys}
    if not values["reference_characters"] or not values["reference_words"]:
        raise ValueError("Empty reference denominator")
    values.update({
        "clips": len(records), "failed_inputs": sum(r["status"] != "ok" for r in records),
        "exact_sentences": sum(r["metrics"]["exact"] for r in records),
        "normalized_exact_sentences": sum(r["metrics"]["normalized_exact"] for r in records),
        "cer_percent": 100 * values["character_edits"] / values["reference_characters"],
        "wer_percent": 100 * values["word_edits"] / values["reference_words"],
        "normalized_cer_percent": 100 * values["normalized_character_edits"] / values["normalized_reference_characters"],
    })
    return values
