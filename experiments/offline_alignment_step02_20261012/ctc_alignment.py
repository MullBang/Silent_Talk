"""NumPy FP64 reference CTC alignment without audio or timestamp assumptions.

The returned token spans describe nonblank state occupancy, not verified speech
boundaries. Callers must check vocabulary coverage and transcript provenance.
"""

from __future__ import annotations

from numbers import Integral
from typing import Any

import numpy as np


class CTCAlignmentError(ValueError):
    """The input or requested full-frame CTC alignment is invalid or impossible."""


def collapse_ctc(path: list[int], blank: int) -> list[int]:
    """Collapse adjacent repeats before removing blanks; no vocabulary is inferred."""
    try:
        if isinstance(blank, bool) or not isinstance(blank, Integral) or blank < 0:
            raise CTCAlignmentError("blank must be a nonnegative integer")
        result: list[int] = []
        previous: int | None = None
        for value in path:
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
                raise CTCAlignmentError("path entries must be nonnegative integer token IDs")
            token = int(value)
            if token != previous and token != blank:
                result.append(token)
            previous = token
        return result
    except CTCAlignmentError:
        raise
    except Exception as exc:
        raise CTCAlignmentError("Could not collapse the CTC path") from exc


def _validate_inputs(
    log_probs: np.ndarray, tokens: list[int], blank: int
) -> tuple[np.ndarray, list[int], int]:
    """Validate IDs and FP64 scores, allowing only negative infinity among infinities."""
    try:
        if not isinstance(log_probs, np.ndarray) or log_probs.ndim != 2:
            raise CTCAlignmentError("log_probs must be a two-dimensional NumPy array [T,V]")
        if log_probs.dtype.kind not in "fiu":
            raise CTCAlignmentError("log_probs must contain real numeric values")
        emissions = np.asarray(log_probs, dtype=np.float64)
        frames, vocabulary = emissions.shape
        if frames < 1 or vocabulary < 2:
            raise CTCAlignmentError("log_probs requires at least one frame and two classes")
        if np.isnan(emissions).any() or np.isposinf(emissions).any():
            raise CTCAlignmentError("NaN and positive infinity are forbidden in log_probs")
        if isinstance(blank, bool) or not isinstance(blank, Integral) or not 0 <= blank < vocabulary:
            raise CTCAlignmentError("blank must be an in-vocabulary integer ID")
        if not isinstance(tokens, list) or not tokens:
            raise CTCAlignmentError("A nonempty list of target token IDs is required")
        validated: list[int] = []
        for token in tokens:
            if isinstance(token, bool) or not isinstance(token, Integral):
                raise CTCAlignmentError("Target token IDs must be integers")
            if not 0 <= token < vocabulary:
                raise CTCAlignmentError("Target token ID is outside the emission vocabulary")
            if token == blank:
                raise CTCAlignmentError("Target transcript must not contain the CTC blank")
            validated.append(int(token))
        minimum = len(validated) + sum(a == b for a, b in zip(validated, validated[1:]))
        if frames < minimum:
            raise CTCAlignmentError(f"Impossible CTC length: T={frames}, minimum={minimum}")
        return emissions, validated, int(blank)
    except CTCAlignmentError:
        raise
    except Exception as exc:
        raise CTCAlignmentError("Could not validate CTC alignment inputs") from exc


def forced_align(log_probs: np.ndarray, tokens: list[int], blank: int) -> dict[str, Any]:
    """Find the highest scoring full-frame CTC path for a nonempty transcript.

    The standard blank-interleaved states permit staying, advancing one state,
    or skipping one blank when adjacent transcript tokens differ. Every frame is
    consumed. The final state must be the last token or final blank. Transition
    ties prefer staying, then advancing one state, then skipping two; final-state
    ties prefer the last token. Spans are half-open emission-frame indices.
    Input rows are assumed to be log probabilities; normalization is the caller's
    responsibility. Impossible paths raise CTCAlignmentError rather than emit
    non-finite JSON scores. No timestamps, padding, or transcript edits are made.
    """
    try:
        emissions, target, blank_id = _validate_inputs(log_probs, tokens, blank)
        frames = emissions.shape[0]
        expanded = np.full(2 * len(target) + 1, blank_id, dtype=np.int64)
        expanded[1::2] = target
        states = len(expanded)
        previous = np.full(states, -np.inf, dtype=np.float64)
        previous[0] = emissions[0, blank_id]
        previous[1] = emissions[0, target[0]]
        backpointers = np.full((frames, states), -1, dtype=np.int64)
        skip_allowed = np.zeros(states, dtype=np.bool_)
        skip_allowed[2:] = (expanded[2:] != blank_id) & (expanded[2:] != expanded[:-2])
        state_indices = np.arange(states, dtype=np.int64)

        for frame in range(1, frames):
            transitions = np.full((3, states), -np.inf, dtype=np.float64)
            transitions[0] = previous
            transitions[1, 1:] = previous[:-1]
            transitions[2, 2:] = np.where(skip_allowed[2:], previous[:-2], -np.inf)
            choice = np.argmax(transitions, axis=0)
            best_previous = transitions[choice, state_indices]
            current = best_previous + emissions[frame, expanded]
            reachable = np.isfinite(current)
            backpointers[frame, reachable] = state_indices[reachable] - choice[reachable]
            previous = current

        final_candidates = [states - 2, states - 1]
        final_state = final_candidates[int(np.argmax(previous[final_candidates]))]
        best_score = float(previous[final_state])
        if not np.isfinite(best_score):
            raise CTCAlignmentError("No finite full-frame path reaches the complete transcript")
        path_states = [0] * frames
        path_states[-1] = final_state
        for frame in range(frames - 1, 0, -1):
            predecessor = int(backpointers[frame, path_states[frame]])
            if predecessor < 0:
                raise CTCAlignmentError("A reachable alignment state has no predecessor")
            path_states[frame - 1] = predecessor
        if path_states[0] not in (0, 1):
            raise CTCAlignmentError("The recovered path does not start at the transcript beginning")
        recovered = [int(expanded[state]) for state in path_states]
        if collapse_ctc(recovered, blank_id) != target:
            raise CTCAlignmentError("Recovered path does not collapse to the exact target transcript")

        path_array = np.asarray(path_states, dtype=np.int64)
        token_spans: list[dict[str, int]] = []
        for index, token_id in enumerate(target):
            occupied = np.flatnonzero(path_array == 2 * index + 1)
            if not len(occupied) or np.any(np.diff(occupied) != 1):
                raise CTCAlignmentError("A target token lacks one contiguous occupied span")
            token_spans.append({"index": index, "token_id": token_id,
                                "start_frame": int(occupied[0]), "end_frame": int(occupied[-1]) + 1})
        return {"path_states": path_states, "expanded_tokens": expanded.tolist(),
                "token_spans": token_spans, "best_log_score": best_score}
    except CTCAlignmentError:
        raise
    except Exception as exc:
        raise CTCAlignmentError("CTC Viterbi reference alignment failed") from exc
