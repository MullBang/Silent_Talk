"""Pure CPU checks of frame transformations, decoding and failure accounting."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location("offline_stage5_evaluator_tests", Path(__file__).with_name("evaluator.py"))
evaluator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluator)


def fixture_input() -> tuple[np.ndarray, dict]:
    pixels = np.empty((3, 5, 96, 96), dtype=np.float32)
    for channel in range(3):
        for frame in range(5):
            pixels[channel, frame] = channel * 10 + frame
    pixels.flags.writeable = False
    return pixels, {"T": 5, "static_frame_index": 2, "shuffled_frame_indices": [4, 0, 3, 1, 2]}


@pytest.mark.parametrize("condition,indices", [("normal", [0, 1, 2, 3, 4]), ("static", [2, 2, 2, 2, 2]), ("shuffled", [4, 0, 3, 1, 2])])
def test_temporal_controls_are_exact_and_do_not_mutate_source(condition: str, indices: list[int]) -> None:
    pixels, item = fixture_input()
    before = evaluator.digest_state(pixels)
    result, actual = evaluator.condition_input(pixels, item, condition)
    assert result.dtype == np.float32 and result.flags.c_contiguous
    assert actual.tolist() == indices
    np.testing.assert_array_equal(result, pixels[:, indices])
    assert evaluator.digest_state(pixels) == before
    assert not np.shares_memory(result, pixels)


@pytest.mark.parametrize("fault", ["float64", "layout", "nonfinite", "duplicate_shuffle", "wrong_static", "unknown"])
def test_invalid_inputs_and_transformations_fail_closed(fault: str) -> None:
    pixels, item = fixture_input()
    condition = "shuffled"
    if fault == "float64":
        pixels = pixels.astype(np.float64)
    elif fault == "layout":
        pixels = pixels.transpose(1, 0, 2, 3)
    elif fault == "nonfinite":
        pixels = pixels.copy()
        pixels[0, 0, 0, 0] = np.nan
    elif fault == "duplicate_shuffle":
        item["shuffled_frame_indices"] = [0, 0, 2, 3, 4]
    elif fault == "wrong_static":
        item["static_frame_index"] = 1
        condition = "static"
    else:
        condition = "reversed"
    with pytest.raises(ValueError):
        evaluator.condition_input(pixels, item, condition)


def test_ctc_blank_separates_repeats_and_invalid_jamo_survives() -> None:
    isolated_initial = next(i for i, token in evaluator.contract.IDX_TO_GRAPHEME.items() if token == "ᄀ")
    result = evaluator.contract.decode_prediction([isolated_initial, isolated_initial, 0, isolated_initial])
    assert result["collapsed_ids"] == [isolated_initial, isolated_initial]
    assert result["raw_text"] == "\ufffd\ufffd"


def test_failure_is_full_deletions_and_corpus_uses_weighted_totals() -> None:
    rows = [
        {"clip_id": "a", "split": "dev", "kind": "full_sentence", "condition": "normal",
         "metrics": evaluator.contract.score_prediction("가 나", "", failed=True)},
        {"clip_id": "b", "split": "dev", "kind": "full_sentence", "condition": "normal",
         "metrics": evaluator.contract.score_prediction("가", "가")},
    ]
    score = evaluator.summarize_rows(rows)["cohorts"]["dev/full_sentence/normal"]
    assert score["count"] == 2 and score["failure_count"] == 1
    assert score["character_edits"] == 3 and score["reference_characters"] == 4 and score["CER"] == .75
    assert score["word_edits"] == 2 and score["reference_words"] == 3 and score["WER"] == 2 / 3
    assert score["exact_count"] == 1
    with pytest.raises(ValueError, match="Duplicate"):
        evaluator.summarize_rows([rows[0], rows[0]])


def test_plan_rejects_changed_conditions_or_checkpoint() -> None:
    plan = json.loads(evaluator.PLAN_PATH.read_text(encoding="utf-8"))
    identity = {"arm": "A_sentence", "seed": 42, "completed_update": 420}
    evaluator.validate_plan(plan, identity)
    changed = copy.deepcopy(plan)
    changed["items"][0]["shuffled_frame_indices"].reverse()
    with pytest.raises(ValueError, match="differs"):
        evaluator.validate_plan(changed, identity)
    with pytest.raises(ValueError, match="outside"):
        evaluator.validate_plan(plan, {**identity, "completed_update": 419})


def test_state_hash_is_type_and_shape_sensitive() -> None:
    assert evaluator.digest_state(np.asarray([1, 2], dtype=np.int64)) != evaluator.digest_state(np.asarray([[1, 2]], dtype=np.int64))
    assert evaluator.digest_state([1, 2]) != evaluator.digest_state((1, 2))
    assert evaluator.digest_state({"b": 2, "a": 1}) == evaluator.digest_state({"a": 1, "b": 2})


def test_state_rng_cpu_capture_restore_without_neural_execution(monkeypatch) -> None:
    # Keep this test pure CPU even on a CUDA machine; production captures all GPUs.
    monkeypatch.setattr(evaluator.torch.cuda, "is_available", lambda: False)
    original = evaluator.capture_rng()
    digest = evaluator.digest_state(original)
    evaluator.random.random()
    evaluator.np.random.random()
    evaluator.torch.rand(2)
    assert evaluator.digest_state(evaluator.capture_rng()) != digest
    evaluator.restore_rng(original)
    assert evaluator.digest_state(evaluator.capture_rng()) == digest
