"""Pure NumPy temporal-control and CTC diagnostics; no model or torch imports.

These measurements describe input sensitivity and CTC output patterns. They do
not establish correct speech recognition, causal feature importance or a training
failure. All controls keep T; a model can still exploit sequence length.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import unicodedata

import numpy as np

_BACKEND = str(Path(__file__).resolve().parents[1] / "backend")
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from models.decoder import ctc_beam_search_decode, ctc_greedy_decode  # noqa: E402
from models.tokenizer import (  # noqa: E402
    BLANK_IDX, IDX_TO_GRAPHEME, NUM_CLASSES, compose_jamos, ctc_min_input_len,
)

CONDITIONS = ("normal", "static_middle", "shuffle", "reversed", "constant_gray")
SHUFFLE_VERSION = "lipnet-temporal-shuffle-v1"

METRIC_DEFINITIONS = {
    "posterior": "exp(log_probs), renormalized per frame only after verifying the sum is within 2e-5 of one; rounding correction, not a softmax on logits",
    "blank_posterior_mean": "mean over T of probability of blank token0",
    "argmax_blank_fraction": "fraction of T argmax token IDs that equal blank0",
    "mean_frame_entropy_nats": "mean over T of -sum(p*ln(p)); zero-probability terms contribute0",
    "mean_frame_entropy_normalized": "mean_frame_entropy_nats divided by ln69",
    "greedy_collapsed_token_ids": "merge adjacent identical argmax IDs first, then remove blank; blank-separated repetitions remain",
    "decoded_nfd_length": "Unicode NFD length of composed output text, including spaces and U+FFFD; this is not the hidden beam prefix token count",
    "invalid_jamo_count": "number of U+FFFD replacement markers in composed text, not number of invalid raw tokens",
    "top_nonblank_token_fraction": "largest nonblank argmax-token count divided by all nonblank argmax positions;0 when there are none",
    "collapsed_ngram_repetition": "on collapsed raw token IDs, count overlapping n-token windows for n2/3/4; repeat_fraction=(windows-unique_windows)/windows,0 when windows=0",
    "reference_ctc_min_frames": "reference token length plus number of adjacent equal reference IDs",
    "ctc_slack_frames": "input T minus reference_ctc_min_frames; negative means no CTC alignment is possible",
    "mean_total_variation": "mean over aligned time positions of0.5*sum(abs(normal_p-control_p))",
    "argmax_path_agreement_fraction": "fraction of aligned time positions with the same argmax ID, including blank",
    "text_equal": "exact equality of caller-supplied normal/control decoded strings; usually beam outputs",
    "blank_posterior_mean_difference": "control mean blank posterior minus normal mean blank posterior",
}


def transform_frames(frames: np.ndarray, condition: str, *, sample_id: str,
                     seed: int = 42) -> tuple[np.ndarray, dict]:
    """Return a fresh uint8 RGB array and reproducible control metadata.

    static_middle repeats index T//2 (right middle for even T). Shuffle uses a
    local PCG64 generator seeded by stable SHA256 of version/sample-ID/seed; it
    does not depend on model identity, process hash or prior calls. A permutation
    may legitimately be the identity, especially for short or single-frame clips.
    """
    if (not isinstance(frames, np.ndarray) or frames.dtype != np.uint8
            or frames.ndim != 4 or frames.shape[-1] != 3
            or any(value < 1 for value in frames.shape)):
        raise ValueError("frames must be nonempty uint8 RGB with shape (T,H,W,3)")
    if condition not in CONDITIONS:
        raise ValueError(f"Unsupported condition: {condition}")
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError("sample_id must be a nonempty string")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    t = len(frames)
    metadata = {"condition": condition, "sample_id": sample_id, "seed": int(seed),
                "input_shape": list(frames.shape), "dtype": "uint8",
                "input_frames": t, "permutation": None, "source_frame_indices": None}
    if condition == "normal":
        indices = np.arange(t)
        transformed = frames.copy()
    elif condition == "static_middle":
        middle = t // 2
        indices = np.full(t, middle, dtype=np.int64)
        transformed = np.repeat(frames[middle:middle+1], t, axis=0)
        metadata["middle_frame_index"] = middle
    elif condition == "shuffle":
        material = json.dumps({"version": SHUFFLE_VERSION, "sample_id": sample_id,
                               "seed": int(seed)}, ensure_ascii=False,
                              sort_keys=True, separators=(",", ":")).encode("utf-8")
        digest = hashlib.sha256(material).digest()
        generator = np.random.Generator(np.random.PCG64(int.from_bytes(digest, "big")))
        indices = generator.permutation(t)
        transformed = frames[indices].copy()
        metadata.update(shuffle_version=SHUFFLE_VERSION, generator="numpy.PCG64",
                        derived_seed_sha256=digest.hex(), permutation=indices.tolist(),
                        permutation_is_identity=bool(np.array_equal(indices, np.arange(t))))
    elif condition == "reversed":
        indices = np.arange(t-1, -1, -1)
        transformed = frames[::-1].copy()
        metadata["permutation"] = indices.tolist()
    else:
        indices = None
        transformed = np.full_like(frames, 128)
        metadata["gray_uint8_value"] = 128
    if indices is not None:
        metadata["source_frame_indices"] = indices.tolist()
    metadata["pixel_changed_frame_count"] = int(np.any(frames != transformed, axis=(1,2,3)).sum())
    return np.ascontiguousarray(transformed), metadata


def _validated_posteriors(log_probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Validate normalized log probabilities and absorb small rounding error."""
    if (not isinstance(log_probs, np.ndarray) or log_probs.ndim != 2
            or log_probs.shape[0] < 1 or log_probs.shape[1] != NUM_CLASSES
            or not np.issubdtype(log_probs.dtype, np.number)
            or np.issubdtype(log_probs.dtype, np.complexfloating)):
        raise ValueError(f"log_probs must have numeric shape (T,{NUM_CLASSES}), T>=1")
    values = log_probs.astype(np.float64, copy=False)
    if not np.all(np.isfinite(values) | np.isneginf(values)) or np.any(values > 1e-5):
        raise ValueError("log_probs contains NaN, positive infinity or positive log probability")
    probabilities = np.exp(values)
    total = probabilities.sum(axis=1, keepdims=True)
    if not np.allclose(total, 1.0, atol=2e-5, rtol=0):
        raise ValueError("log_probs must already be normalized log probabilities, not logits")
    return values, probabilities / total


def _collapse_path(raw_ids: list[int]) -> list[int]:
    """Apply CTC adjacency collapse before removing blank IDs."""
    result, previous = [], None
    for token in raw_ids:
        if token != previous and token != BLANK_IDX:
            result.append(token)
        previous = token
    return result


def _ngram_repetition(tokens: list[int], n: int) -> dict:
    """Count overlapping windows beyond their first occurrence."""
    windows = [tuple(tokens[start:start+n]) for start in range(max(0, len(tokens)-n+1))]
    count, unique = len(windows), len(set(windows))
    return {"windows": count, "unique_windows": unique, "repeated_windows": count-unique,
            "repeat_fraction": (count-unique)/count if count else 0.0}


def analyze_ctc(log_probs: np.ndarray, labels: np.ndarray, *, beam_text: str | None = None,
                beam_width: int = 10) -> dict:
    """Measure an unpadded T×69 output and the actual reference token sequence.

    Supplied beam_text avoids decoding an existing evaluation twice. The reported
    beam length is composed-text NFD length, not a reconstructed beam token path.
    Blank and repeated tokens are valid CTC behavior; these rates alone are not
    error diagnoses. Labels may be infeasible for T so negative slack is reported.
    """
    values, probabilities = _validated_posteriors(log_probs)
    if (not isinstance(labels, np.ndarray) or labels.ndim != 1
            or not np.issubdtype(labels.dtype, np.integer)
            or np.any(labels <= BLANK_IDX) or np.any(labels >= NUM_CLASSES)):
        raise ValueError("labels must be1D integer nonblank token IDs within the vocabulary")
    if isinstance(beam_width, bool) or not isinstance(beam_width, (int, np.integer)) or beam_width < 1:
        raise ValueError("beam_width must be a positive integer")
    if beam_text is not None and not isinstance(beam_text, str):
        raise ValueError("beam_text must be a string or None")
    raw = probabilities.argmax(axis=1).tolist()
    collapsed = _collapse_path(raw)
    greedy = ctc_greedy_decode(values)
    beam = ctc_beam_search_decode(values, beam_width=int(beam_width)) if beam_text is None else beam_text
    nonblank = [token for token in raw if token != BLANK_IDX]
    counts = Counter(nonblank)
    top = min(counts, key=lambda token: (-counts[token], token)) if counts else None
    logs = np.zeros_like(probabilities)
    np.log(probabilities, out=logs, where=probabilities > 0)
    entropy = float(-(probabilities*logs).sum(axis=1).mean())
    minimum = ctc_min_input_len(labels.tolist())
    return {
        "input_frames": len(values), "num_classes": NUM_CLASSES,
        "reference_token_length": len(labels),
        "reference_adjacent_repeats": int(np.sum(labels[1:] == labels[:-1])),
        "reference_ctc_min_frames": minimum, "ctc_slack_frames": len(values)-minimum,
        "reference_ctc_feasible": len(values) >= minimum,
        "blank_posterior_mean": float(probabilities[:, BLANK_IDX].mean()),
        "argmax_blank_fraction": float(np.mean(np.asarray(raw) == BLANK_IDX)),
        "mean_frame_entropy_nats": entropy,
        "mean_frame_entropy_normalized": entropy / float(np.log(NUM_CLASSES)),
        "greedy_raw_token_ids": raw, "greedy_collapsed_token_ids": collapsed,
        "greedy_collapsed_token_count": len(collapsed),
        "greedy_collapsed_unique_token_ids": sorted(set(collapsed)),
        "greedy_collapsed_unique_token_count": len(set(collapsed)),
        "greedy_text": greedy,
        "greedy_decoded_nfd_length": len(unicodedata.normalize("NFD", greedy)),
        "greedy_invalid_jamo_count": greedy.count("\ufffd"),
        "beam_text": beam, "beam_decoded_nfd_length": len(unicodedata.normalize("NFD", beam)),
        "beam_invalid_jamo_count": beam.count("\ufffd"), "beam_width": int(beam_width),
        "beam_text_source": "supplied" if beam_text is not None else "decoded_here",
        "argmax_nonblank_count": len(nonblank), "top_nonblank_token_id": top,
        "top_nonblank_token_fraction": counts[top]/len(nonblank) if top is not None else 0.0,
        "collapsed_ngram_repetition": {str(n):_ngram_repetition(collapsed,n) for n in (2,3,4)},
    }


def compare_posteriors(normal_log_probs: np.ndarray, control_log_probs: np.ndarray, *,
                       normal_text: str, control_text: str) -> dict:
    """Compare equal-T posterior sequences; positions are not realigned.

    In shuffled/reversed videos, equal time positions can contain different source
    frames. A difference therefore describes sensitivity, not semantic causality.
    """
    _, normal = _validated_posteriors(normal_log_probs)
    _, control = _validated_posteriors(control_log_probs)
    if normal.shape != control.shape:
        raise ValueError("Normal and control posterior arrays must have identical shapes")
    if not isinstance(normal_text, str) or not isinstance(control_text, str):
        raise ValueError("Normal and control decoded texts must be strings")
    tv = .5*np.abs(normal-control).sum(axis=1)
    normal_raw, control_raw = normal.argmax(axis=1), control.argmax(axis=1)
    normal_ids, control_ids = _collapse_path(normal_raw.tolist()), _collapse_path(control_raw.tolist())
    normal_greedy = compose_jamos([IDX_TO_GRAPHEME[token] for token in normal_ids])
    control_greedy = compose_jamos([IDX_TO_GRAPHEME[token] for token in control_ids])
    return {
        "input_frames": len(normal), "mean_total_variation": float(tv.mean()),
        "max_total_variation": float(tv.max()),
        "argmax_path_agreement_fraction": float(np.mean(normal_raw == control_raw)),
        "greedy_collapsed_token_ids_equal": normal_ids == control_ids,
        "greedy_text_equal": normal_greedy == control_greedy,
        "text_equal": normal_text == control_text,
        "blank_posterior_mean_difference": float(control[:,BLANK_IDX].mean()-normal[:,BLANK_IDX].mean()),
    }
