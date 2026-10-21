"""Pure CPU prediction decoding and metric definitions for the future comparison.

No network is imported or executed. Invalid predicted jamo remain error
characters, and metric aggregation keeps internal spaces and all failed rows.
"""
from __future__ import annotations

from itertools import groupby
from numbers import Integral
from pathlib import Path
import sys
from typing import Iterable, Sequence
import unicodedata

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from backend.models.tokenizer import IDX_TO_GRAPHEME, NUM_CLASSES, compose_jamos


def metric_text(text: str) -> str:
    """Use the fixed NFC/outer-strip rule, without removing erroneous characters."""
    if not isinstance(text, str):
        raise TypeError("Metric text must be a string")
    return unicodedata.normalize("NFC", text).strip()


def decode_prediction(frame_ids: Iterable[int]) -> dict:
    """Collapse CTC frame repeats before removing blanks; preserve invalid jamo."""
    raw = list(frame_ids)
    if any(isinstance(v, bool) or not isinstance(v, Integral) or not 0 <= int(v) < NUM_CLASSES for v in raw):
        raise ValueError("Prediction IDs must be integers within the fixed69 vocabulary")
    raw = [int(v) for v in raw]
    collapsed = [ident for ident, _ in groupby(raw) if ident != 0]
    text = compose_jamos((IDX_TO_GRAPHEME[v] for v in collapsed), strict=False)
    return {"raw_argmax_ids": raw, "collapsed_ids": collapsed,
            "raw_text": text, "metric_text": metric_text(text)}


def edit_distance(reference: Sequence, prediction: Sequence) -> int:
    """Compute ordinary Levenshtein distance with one dynamic-programming row."""
    row = list(range(len(prediction)+1))
    for i, ref in enumerate(reference, 1):
        next_row = [i]
        for j, pred in enumerate(prediction, 1):
            next_row.append(min(next_row[-1]+1, row[j]+1, row[j-1]+(ref != pred)))
        row = next_row
    return row[-1]


def score_prediction(reference: str, prediction: str, *, failed: bool = False) -> dict:
    """Count edits without silently dropping empty/failed predictions."""
    ref, pred = metric_text(reference), metric_text(prediction)
    if not ref:
        raise ValueError("The fixed pilot requires nonempty reference text")
    if type(failed) is not bool or (failed and pred):
        raise ValueError("Failed inference must have an explicit empty prediction")
    runs = [len(list(values)) for char, values in groupby(pred) if not char.isspace()]
    maximum_run = max(runs, default=0)
    return {"reference": ref, "prediction": pred, "failed": failed,
            "character_edits": edit_distance(ref, pred), "reference_characters": len(ref),
            "word_edits": edit_distance(ref.split(), pred.split()), "reference_words": len(ref.split()),
            "exact": ref == pred and not failed, "replacement_characters": pred.count("\ufffd"),
            "maximum_same_nonspace_character_run": maximum_run, "repetition_ge5": maximum_run >= 5}


def corpus_scores(rows: Sequence[dict]) -> dict:
    """Aggregate edit totals, never the unweighted mean of clip error rates."""
    if not rows:
        raise ValueError("Cannot score an empty cohort")
    chars = sum(r["reference_characters"] for r in rows)
    words = sum(r["reference_words"] for r in rows)
    char_errors = sum(r["character_edits"] for r in rows)
    word_errors = sum(r["word_edits"] for r in rows)
    return {"count": len(rows), "character_edits": char_errors, "reference_characters": chars,
            "word_edits": word_errors, "reference_words": words,
            "CER": char_errors/chars, "WER": word_errors/words,
            "exact_count": sum(r["exact"] for r in rows), "failure_count": sum(r["failed"] for r in rows),
            "repetition_ge5_count": sum(r["repetition_ge5"] for r in rows)}
