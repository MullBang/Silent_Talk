"""Independent CPU tensor/CTC/text audit primitives, without producer imports."""
from __future__ import annotations

from itertools import product
import hashlib
import math
import struct
import unicodedata

import numpy as np
import torch


def unpack(value):
    """Decode the declared safe NumPy state serialization."""
    if isinstance(value, dict):
        if value.get("__step03_numpy_array_v1__") is True:
            assert set(value) == {"__step03_numpy_array_v1__", "dtype", "shape", "bytes"}
            dtype = np.dtype(value["dtype"])
            tensor = value["bytes"]
            assert not dtype.hasobject and tensor.dtype == torch.uint8 and tensor.ndim == 1
            assert tensor.numel() == math.prod(value["shape"]) * dtype.itemsize
            return np.frombuffer(tensor.numpy().tobytes(), dtype=dtype).reshape(value["shape"]).copy()
        return {key: unpack(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(unpack(item) for item in value)
    return value


def fingerprint(value) -> str:
    """Reproduce the published type-tagged state digest from its byte contract."""
    digest = hashlib.sha256()

    def put(blob: bytes) -> None:
        digest.update(struct.pack("<Q", len(blob)))
        digest.update(blob)

    def visit(item) -> None:
        if isinstance(item, torch.Tensor):
            array = item.detach().cpu().contiguous()
            for blob in [b"torch", str(array.dtype).encode(), str(tuple(array.shape)).encode(), array.reshape(-1).view(torch.uint8).numpy().tobytes()]:
                put(blob)
        elif isinstance(item, np.ndarray):
            assert not item.dtype.hasobject
            for blob in [b"numpy", item.dtype.str.encode(), str(item.shape).encode(), np.ascontiguousarray(item).tobytes()]:
                put(blob)
        elif isinstance(item, dict):
            put(b"dict")
            for key in sorted(item, key=lambda key: (type(key).__name__, repr(key))):
                visit(key)
                visit(item[key])
            put(b"end_dict")
        elif isinstance(item, (list, tuple)):
            put(type(item).__name__.encode())
            for child in item:
                visit(child)
            put(b"end_sequence")
        elif item is None:
            put(b"none")
        elif isinstance(item, bool):
            put(b"bool")
            put(str(item).encode())
        elif isinstance(item, int):
            put(b"int")
            put(str(item).encode())
        elif isinstance(item, float):
            assert math.isfinite(item)
            put(b"float")
            put(struct.pack("<d", item))
        elif isinstance(item, str):
            put(b"str")
            put(item.encode())
        else:
            raise AssertionError(type(item))

    visit(value)
    return digest.hexdigest()


def state_hashes(state: dict) -> dict:
    return {**{key: fingerprint(state[key]) for key in ["model", "optimizer", "rng"]}, "complete": fingerprint(state)}


def evaluation_digest(value) -> str:
    """Independently encode the evaluator's documented state-byte format."""
    def encode(item) -> bytes:
        if isinstance(item, torch.Tensor):
            return b"torch:" + encode(item.detach().cpu().contiguous().numpy())
        if isinstance(item, np.ndarray):
            assert not item.dtype.hasobject
            return b"numpy:" + item.dtype.str.encode() + repr(item.shape).encode() + b":" + np.ascontiguousarray(item).tobytes()
        if isinstance(item, dict):
            ordered = sorted(item, key=lambda key: (type(key).__name__, repr(key)))
            return b"dict{" + b"".join(encode(key) + encode(item[key]) for key in ordered) + b"}"
        if isinstance(item, (tuple, list)):
            return type(item).__name__.encode() + b"[" + b"".join(encode(child) for child in item) + b"]"
        assert item is None or isinstance(item, (str, int, float, bool))
        payload = repr(item).encode()
        return type(item).__name__.encode() + b":" + str(len(payload)).encode() + b":" + payload
    return hashlib.sha256(encode(value)).hexdigest()


def collapsed_ids(raw: list[int]) -> list[int]:
    result = []
    previous = -1
    for token in raw:
        assert 0 <= token < 69
        if token != previous and token != 0:
            result.append(token)
        previous = token
    return result


def compose_ids(tokens: list[int]) -> str:
    """Construct Hangul syllables arithmetically; retain each invalid token."""
    result = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token == 68:
            result.append(" ")
            i += 1
        elif 1 <= token <= 19 and i + 1 < len(tokens) and 20 <= tokens[i + 1] <= 40:
            lead, vowel, tail = token - 1, tokens[i + 1] - 20, 0
            i += 2
            if i < len(tokens) and 41 <= tokens[i] <= 67:
                tail = tokens[i] - 40
                i += 1
            result.append(chr(0xAC00 + (lead * 21 + vowel) * 28 + tail))
        else:
            assert 1 <= token < 69
            result.append("\ufffd")
            i += 1
    return "".join(result)


def metric_text(value: str) -> str:
    return unicodedata.normalize("NFC", value).strip()


def edit_distance(reference, prediction) -> int:
    """Full matrix independent of the producer's one-row implementation."""
    matrix = np.zeros((len(reference) + 1, len(prediction) + 1), dtype=np.int32)
    matrix[:, 0] = np.arange(len(reference) + 1)
    matrix[0, :] = np.arange(len(prediction) + 1)
    for i, left in enumerate(reference, 1):
        for j, right in enumerate(prediction, 1):
            matrix[i, j] = min(matrix[i - 1, j] + 1, matrix[i, j - 1] + 1, matrix[i - 1, j - 1] + (left != right))
    return int(matrix[-1, -1])


def score(reference: str, prediction: str, failed: bool = False) -> dict:
    reference, prediction = metric_text(reference), metric_text(prediction)
    assert reference and (not failed or not prediction)
    longest = current = 0
    previous = None
    for char in prediction:
        current = current + 1 if char == previous and not char.isspace() else 0 if char.isspace() else 1
        longest, previous = max(longest, current), char
    return {
        "reference": reference, "prediction": prediction, "failed": failed,
        "character_edits": edit_distance(reference, prediction), "reference_characters": len(reference),
        "word_edits": edit_distance(reference.split(), prediction.split()), "reference_words": len(reference.split()),
        "exact": reference == prediction and not failed, "replacement_characters": prediction.count("\ufffd"),
        "maximum_same_nonspace_character_run": longest, "repetition_ge5": longest >= 5,
    }


def corpus(rows: list[dict]) -> dict:
    assert rows
    counts = {key: sum(row[key] for row in rows) for key in ["character_edits", "reference_characters", "word_edits", "reference_words"]}
    return {"count": len(rows), **counts, "CER": counts["character_edits"] / counts["reference_characters"],
            "WER": counts["word_edits"] / counts["reference_words"], "exact_count": sum(row["exact"] for row in rows),
            "failure_count": sum(row["failed"] for row in rows), "repetition_ge5_count": sum(row["repetition_ge5"] for row in rows)}


def ctc_nll(log_probs: np.ndarray, target: list[int]) -> float:
    """Float64 forward sum over CTC states, no torch CTC or model call."""
    values = np.asarray(log_probs, dtype=np.float64)
    assert values.ndim == 2 and len(target) and np.isfinite(values).all()
    extended = np.zeros(2 * len(target) + 1, dtype=np.int64)
    extended[1::2] = target
    skip_allowed = np.zeros(len(extended), dtype=bool)
    skip_allowed[2:] = (extended[2:] != 0) & (extended[2:] != extended[:-2])
    previous = np.full(len(extended), -np.inf)
    previous[0], previous[1] = values[0, 0], values[0, target[0]]
    for frame in values[1:]:
        one = np.concatenate(([-np.inf], previous[:-1]))
        two = np.concatenate(([-np.inf, -np.inf], previous[:-2]))
        two[~skip_allowed] = -np.inf
        previous = np.logaddexp(np.logaddexp(previous, one), two) + frame[extended]
    return -float(np.logaddexp(previous[-1], previous[-2]))


def self_check() -> dict:
    """Small analytical or exhaustive checks before auditing real outputs."""
    assert collapsed_ids([1, 1, 0, 1, 20, 20, 0, 68]) == [1, 1, 20, 68]
    assert compose_ids([1, 20, 41, 68, 1, 20]) == "각 가"
    assert compose_ids([20, 1, 68, 41]) == "\ufffd\ufffd \ufffd"
    assert edit_distance("abc", "axbc") == 1 and edit_distance("aa", "") == 2
    assert score("가  나", "가 나")["character_edits"] == 1
    assert score("말", "", True)["character_edits"] == 1
    probabilities = np.array([[.2, .5, .3], [.4, .2, .4], [.1, .6, .3]], dtype=np.float64)
    tested = []
    for target in [[1], [1, 1], [1, 2], [2, 1]]:
        total = 0.
        for raw in product(range(3), repeat=3):
            if collapsed_ids(list(raw)) == target:
                total += float(np.prod([probabilities[i, x] for i, x in enumerate(raw)]))
        actual = ctc_nll(np.log(probabilities), target)
        assert abs(actual + math.log(total)) < 1e-12
        tested.append(target)
    return {"status": "passed", "CTC_exhaustive_paths_per_target": 27, "targets": tested,
            "Unicode_composition_cases": 2, "CTC_collapse_case": 1, "metric_edge_cases": 4}


if __name__ == "__main__":
    print(self_check())
