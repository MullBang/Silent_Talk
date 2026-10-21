"""Scalar FP64 CTC Viterbi verification independent of the production helper."""
from __future__ import annotations

import math
from typing import Any
import numpy as np


def collapse(tokens: list[int], blank: int) -> list[int]:
    result: list[int] = []
    for index, token in enumerate(tokens):
        if token != blank and (index == 0 or token != tokens[index - 1]):
            result.append(token)
    return result


def align(log_probs: np.ndarray, target: list[int], blank: int) -> dict[str, Any]:
    """Enumerate legal predecessors per scalar state, with documented ties."""
    scores = np.asarray(log_probs, dtype=np.float64)
    if scores.ndim != 2 or not target or np.isnan(scores).any() or np.isposinf(scores).any():
        raise ValueError("Invalid emissions or empty target")
    frames, vocab_size = scores.shape
    if blank in target or not 0 <= blank < vocab_size or any(not 0 <= t < vocab_size for t in target):
        raise ValueError("Invalid target or blank ID")
    expanded = [blank]
    for token in target:
        expanded.extend([token, blank])
    states = len(expanded)
    table = [[-math.inf] * states for _ in range(frames)]
    predecessors = [[None] * states for _ in range(frames)]
    table[0][0] = float(scores[0, blank])
    table[0][1] = float(scores[0, target[0]])
    for time in range(1, frames):
        for state, token in enumerate(expanded):
            candidates = [state]
            if state >= 1:
                candidates.append(state - 1)
            if state >= 2 and token != blank and token != expanded[state - 2]:
                candidates.append(state - 2)
            # max is stable, preferring stay, advance-one, then advance-two.
            predecessor = max(candidates, key=lambda index: table[time - 1][index])
            value = table[time - 1][predecessor] + float(scores[time, token])
            if math.isfinite(value):
                table[time][state] = value
                predecessors[time][state] = predecessor
    final = max([states - 2, states - 1], key=lambda index: table[-1][index])
    if not math.isfinite(table[-1][final]):
        raise ValueError("No complete CTC path")
    path = [final]
    for time in range(frames - 1, 0, -1):
        previous = predecessors[time][path[-1]]
        if previous is None:
            raise ValueError("Missing reachable predecessor")
        path.append(previous)
    path.reverse()
    assert collapse([expanded[state] for state in path], blank) == target
    spans = []
    for index, token in enumerate(target):
        occupied = [t for t, state in enumerate(path) if state == 2 * index + 1]
        assert occupied and occupied == list(range(occupied[0], occupied[-1] + 1))
        spans.append({"index": index, "token_id": token,
                      "start_frame": occupied[0], "end_frame": occupied[-1] + 1})
    return {"expanded_tokens": expanded, "path_states": path,
            "token_spans": spans, "best_log_score": table[-1][final]}
