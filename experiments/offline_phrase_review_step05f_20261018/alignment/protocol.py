"""Exact annotation-audio geometry; never derives a visual training frame count."""
from __future__ import annotations

from fractions import Fraction
import hashlib
import importlib.util
import json
from pathlib import Path
import re
from typing import Any
import unicodedata

BASE = Path(__file__).resolve().parent
STAGE = BASE.parent
ROOT = STAGE.parents[1]
OLD = ROOT / "experiments/offline_alignment_step02_20261012"
RATE = 16000
HOP = 320
FIELD = 400
EDGE = (FIELD - HOP) // 2
CTC_SHA = "862fd5f7b6c891b693687ea89a9d95690ac0425b409a6610e3a3271843f281c1"


def sha(path: Path) -> str:
    """Hash a local artifact without deserializing model tensors."""
    try:
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()
    except OSError as exc:
        raise RuntimeError(f"Cannot hash {path}") from exc


def read(path: Path) -> Any:
    """Read JSON without treating its contents as executable instructions."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Cannot read JSON {path}") from exc


def write(path: Path, value: Any) -> None:
    """Create a new evidence file, refusing any existing destination."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
    except (OSError, ValueError, TypeError) as exc:
        raise RuntimeError(f"Cannot create JSON {path}") from exc


def binding(path: Path) -> dict[str, Any]:
    """Describe a file's exact local identity."""
    return {"path": str(path.resolve()), "sha256": sha(path), "bytes": path.stat().st_size}


def verify_binding(spec: dict[str, Any]) -> Path:
    """Reject changed bytes before interpreting a bound artifact."""
    path = Path(spec["path"])
    if sha(path) != spec["sha256"] or ("bytes" in spec and path.stat().st_size != spec["bytes"]):
        raise ValueError(f"Bound artifact changed: {path}")
    return path


def ratio(value: Fraction) -> dict[str, int]:
    """Serialize rational time exactly, without rounding it to a video frame."""
    return {"numerator": value.numerator, "denominator": value.denominator}


def fraction(value: dict[str, int]) -> Fraction:
    """Validate a JSON rational; floats and booleans cannot represent sample counts."""
    if set(value) != {"numerator", "denominator"}:
        raise ValueError("A rational needs numerator and denominator")
    n, d = value["numerator"], value["denominator"]
    if type(n) is not int or type(d) is not int or d <= 0:
        raise ValueError("A rational requires integral values and positive denominator")
    return Fraction(n, d)


def load_ctc() -> Any:
    """Load the frozen, audited DP helper only after checking its original SHA."""
    path = OLD / "ctc_alignment.py"
    if sha(path) != CTC_SHA:
        raise ValueError("Frozen CTC helper changed")
    spec = importlib.util.spec_from_file_location("stage5f_frozen_ctc", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot import the bound CTC helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def checked_text(text: str) -> str:
    """Require the already chosen canonical text; never normalize it silently."""
    if not text or text != unicodedata.normalize("NFC", text) or text != " ".join(text.split()):
        raise ValueError("Expected a nonempty NFC transcript with single spaces")
    return text


def acoustic_oov(text: str, vocab: dict[str, int]) -> list[str]:
    """Report unsupported acoustic characters without changing the visual labels."""
    return sorted(set(checked_text(text).replace(" ", "|")) - set(vocab))


def span_geometry(start_frame: int, end_frame: int, samples: int,
                  origin: Fraction, offset: Fraction) -> dict[str, Any]:
    """Map CTC occupancy bin edges to exact audio time and video time once."""
    if any(type(x) is not int for x in (start_frame, end_frame, samples)):
        raise ValueError("Emission indices and PCM sample count must be integers")
    start, end = start_frame * HOP + EDGE, end_frame * HOP + EDGE
    if not 0 <= start_frame < end_frame or not 0 <= start < end <= samples:
        raise ValueError("CTC occupancy proposal exceeds the fixed parent audio")
    wav_start, wav_end = origin + Fraction(start, RATE), origin + Fraction(end, RATE)
    return {
        "local_audio_start_sample_16k": start, "local_audio_end_sample_16k": end,
        "local_audio_sample_count": end - start,
        "wav_start_seconds_exact": ratio(wav_start), "wav_end_seconds_exact": ratio(wav_end),
        "video_start_seconds_exact": ratio(wav_start + offset),
        "video_end_seconds_exact": ratio(wav_end + offset), "video_offset_applied_count": 1,
        "boundary_semantics": "CTC nonblank-occupancy bin-edge proposal, not verified phonetic or articulatory boundary",
        "frame_count": None, "ctc_feasible": None,
        "visual_frame_selection": "deferred: recording-origin 25fps rational grid and actual PTS nearest, ties earlier",
        "actual_training_clip_created": False, "human_phrase_audio_boundary_approved": False,
        "human_phrase_visual_boundary_approved": False, "training_ready": False,
    }


def selected_phrase(parent: str, phrase: str, word_start: int, word_end: int,
                    alignment: dict[str, Any], samples: int, origin: Fraction,
                    offset: Fraction) -> dict[str, Any]:
    """Use one specified phrase's character spans; preserve separate parent review."""
    checked_text(parent)
    checked_text(phrase)
    words = list(re.finditer(r"\S+", parent))
    if type(word_start) is not int or type(word_end) is not int or not 0 <= word_start < word_end <= len(words):
        raise ValueError("The proposed word span is outside its parent transcript")
    start_char, end_char = words[word_start].start(), words[word_end - 1].end()
    if parent[start_char:end_char] != phrase:
        raise ValueError("Selected phrase no longer agrees with the canonical parent")
    spans = alignment["token_spans"]
    if len(spans) != len(parent):
        raise ValueError("Expected one acoustic target symbol per transcript character")
    result = span_geometry(spans[start_char]["start_frame"], spans[end_char - 1]["end_frame"],
                           samples, origin, offset)
    result.update({"text": phrase, "word_start": word_start, "word_end": word_end,
                   "char_start": start_char, "char_end": end_char,
                   "phrase_equals_whole_parent_text": phrase == parent,
                   "parent_review_is_separate_from_phrase_boundary_review": True})
    return result
