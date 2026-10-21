"""Stage15 fixed-checkpoint contracts and synthetic-only inference verification.

These tests do not open any real clip cache or run any trained checkpoint.
The CUDA smoke test initializes a compact model and uses generated pixels.
"""
from __future__ import annotations

import copy
import itertools
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "backend")]
import diagnose_generalization as runner
from learning_budget_state import capture_rng_state, compare_nested
from models.baseline import LipNetBaseline
from models.tokenizer import encode_text


def checkpoint_metadata(key="B_reduce9600"):
    ids = [f"training-{i:02d}" for i in range(64)]
    branch = "A_keep" if key == "A_keep9600" else "B_reduce"
    metadata = {"architecture_version": "lipnet-compact-v1", "num_classes": 69,
                "training_data": {"mode": "smoke_only", "manifest_sha256": runner.MANIFEST_SHA,
                    "sample_ids": ids.copy(), "deployment": False, "branch": branch,
                    "purpose": "train_subset_budget_diagnostic" if key == "source6400" else "train_subset_lr_diagnostic"},
                "training_run": {"inference_only": True, "selection": "fixed_attempt_schedule",
                    "plan_sha256": "synthetic-plan-sha", "attempt_updates": 6400 if key == "source6400" else 9600,
                    "branch": branch, "settings": {"learning_rate": .0001 if key == "B_reduce9600" else .0003}}}
    return metadata, ids


@pytest.mark.parametrize("key", list(runner.MODEL_SPECS))
def test_accepts_three_exact_fixed_metadata_contracts(key):
    metadata, ids = checkpoint_metadata(key)
    runner.validate_checkpoint_metadata(metadata, key, ids, "synthetic-plan-sha")


@pytest.mark.parametrize("path,value,message", [
    (("architecture_version",), "other", "architecture"),
    (("num_classes",), 70, "architecture"),
    (("training_data", "mode"), "train", "lineage"),
    (("training_data", "manifest_sha256"), "other", "lineage"),
    (("training_data", "purpose"), "deployment", "purpose"),
    (("training_data", "deployment"), True, "diagnostic"),
    (("training_run", "inference_only"), False, "diagnostic"),
    (("training_run", "selection"), "best_val", "diagnostic"),
    (("training_run", "plan_sha256"), "other", "diagnostic"),
    (("training_run", "attempt_updates"), 9601, "attempt"),
    (("training_data", "branch"), "A_keep", "branch"),
    (("training_run", "branch"), "A_keep", "branch"),
    (("training_run", "settings", "learning_rate"), .0003, "LR"),
])
def test_rejects_mismatched_checkpoint_metadata(path, value, message):
    metadata, ids = checkpoint_metadata()
    target = metadata
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value
    with pytest.raises(ValueError, match=message):
        runner.validate_checkpoint_metadata(metadata, "B_reduce9600", ids, "synthetic-plan-sha")


def test_rejects_same_training_set_with_different_order():
    metadata, ids = checkpoint_metadata()
    metadata["training_data"]["sample_ids"].reverse()
    with pytest.raises(ValueError, match="lineage"):
        runner.validate_checkpoint_metadata(metadata, "B_reduce9600", ids, "synthetic-plan-sha")


def test_rejects_duplicate_ids_even_when_checkpoint_matches_the_duplicate_list():
    metadata, ids = checkpoint_metadata()
    ids[-1] = ids[0]
    metadata["training_data"]["sample_ids"] = ids.copy()
    with pytest.raises(ValueError, match="lineage"):
        runner.validate_checkpoint_metadata(metadata, "B_reduce9600", ids, "synthetic-plan-sha")


def test_rejects_unapproved_checkpoint_key():
    metadata, ids = checkpoint_metadata()
    with pytest.raises(ValueError, match="Unapproved"):
        runner.validate_checkpoint_metadata(metadata, "new_candidate", ids, "synthetic-plan-sha")


@pytest.mark.parametrize("source", [
    {"id": "test-id", "split": "test"},
    {"id": "unselected", "split": "train"},
    {"id": "unselected", "split": "val"},
    {"id": "selected-train", "split": "val"},
])
def test_forbidden_cache_is_rejected_before_verifier_can_run(monkeypatch, source):
    calls = []
    monkeypatch.setattr(runner, "verify_cache", lambda row: calls.append(row))
    cohorts = {"train60": ["selected-train", "test-id"], "val20": ["selected-val"]}
    with pytest.raises(ValueError, match="cache access forbidden"):
        runner.cache_for_evaluation(source, cohorts)
    assert calls == []


@pytest.mark.parametrize("split,cohort", [("train", "train60"), ("val", "val20")])
def test_selected_cache_delegates_exact_row_once(monkeypatch, split, cohort):
    calls, item = [], {"synthetic": True}
    def verifier(source):
        calls.append(source)
        return item
    monkeypatch.setattr(runner, "verify_cache", verifier)
    source = {"id": "selected", "split": split}
    cohorts = {"train60": [], "val20": []}
    cohorts[cohort] = [source["id"]]
    assert runner.cache_for_evaluation(source, cohorts) is item
    assert calls == [source]


def test_tensor_digest_is_serialization_order_independent_and_exact():
    first = {"weight": torch.tensor([[1., 2.], [3., 4.]]), "bias": torch.tensor([.2, .4])}
    second = {"bias": first["bias"].clone(), "weight": first["weight"].clone()}
    original = runner.state_digest(first)
    assert runner.state_digest(second) == original
    second["weight"][0, 0] = torch.nextafter(second["weight"][0, 0], torch.tensor(float("inf")))
    assert runner.state_digest(second) != original
    assert runner.state_digest({"weight": first["weight"].to(torch.float64), "bias": first["bias"]}) != original
    assert runner.state_digest({"weight": first["weight"].flatten(), "bias": first["bias"]}) != original
    assert runner.state_digest({"renamed": first["weight"], "bias": first["bias"]}) != original


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_tensor_digest_rejects_nonfinite_weights(value):
    with pytest.raises(ValueError, match="Nonfinite"):
        runner.state_digest({"weight": torch.tensor([value])})


def synthetic_clip(frames=4, text="가", size=4):
    source = {"id": "synthetic-clip", "split": "train", "speaker_id": "synthetic-speaker",
              "recording_id": "synthetic-recording", "text": text, "frames": frames}
    pixels = np.repeat(np.arange(frames, dtype=np.uint8)[:, None, None, None], size*size*3, axis=1)
    item = {"frames": pixels.reshape(frames, size, size, 3), "label": np.array(encode_text(text)), "text": text}
    return source, item


def exhaustive_ctc_probability(probabilities, labels):
    """Enumerate tiny valid alphabet paths independently of torch CTC."""
    total = 0.0
    for path in itertools.product([0] + list(labels), repeat=len(probabilities)):
        merged = [token for i, token in enumerate(path) if i == 0 or token != path[i-1]]
        collapsed = [token for token in merged if token != 0]
        if collapsed == list(labels):
            total += float(np.prod([probabilities[i, token] for i, token in enumerate(path)]))
    return total


def synthetic_log_probabilities():
    labels = encode_text("가")
    probabilities = np.full((4, 69), .02/68, dtype=np.float64)
    for index, token in enumerate([labels[0], 0, labels[1], 0]):
        probabilities[index, token] = .98
    return np.log(probabilities).astype(np.float32)


def test_three_conditions_beam_normal_only_and_ctc_against_exhaustive_paths(monkeypatch):
    source, item = synthetic_clip()
    original_pixels = item["frames"].copy()
    log_probs = synthetic_log_probabilities()
    inputs, beam_calls = [], []
    actual_beam = runner.ctc_beam_search_decode
    def beam(probs, beam_width):
        beam_calls.append((probs.copy(), beam_width))
        return actual_beam(probs, beam_width=beam_width)
    monkeypatch.setattr(runner, "ctc_beam_search_decode", beam)
    def synthetic_forward(model, pixels, capture):
        assert capture is False
        inputs.append(pixels.copy())
        return log_probs[pixels[:, 0, 0, 0]].copy(), {}
    rows, arrays = runner.evaluate_clip(torch.nn.Identity().eval(), source, item,
        key="source6400", cohort="train60", forward_fn=synthetic_forward)
    assert [r["condition"] for r in rows] == ["normal", "static_middle", "shuffle_42"]
    assert len(arrays) == len(inputs) == 3
    assert len(beam_calls) == 1 and beam_calls[0][1] == 10
    assert rows[0]["beam"] is not None and rows[1]["beam"] is rows[2]["beam"] is None
    assert rows[0]["greedy"]["prediction"] == "가"
    assert rows[0]["greedy"]["cer"] == rows[0]["greedy"]["wer"] == 0
    assert rows[0]["greedy"]["exact"] is True
    assert rows[1]["greedy"]["prediction"] == "�"
    assert rows[1]["greedy"]["cer"] == 1 and rows[1]["greedy"]["jamo_edit_distance"] == 2
    assert rows[1]["greedy"]["wer"] == 1
    assert rows[0]["comparison_to_normal"] is None
    assert all(r["comparison_to_normal"] is not None for r in rows[1:])
    np.testing.assert_array_equal(item["frames"], original_pixels)
    np.testing.assert_array_equal(inputs[0], original_pixels)
    np.testing.assert_array_equal(inputs[1], np.repeat(original_pixels[2:3], 4, axis=0))
    assert sorted(inputs[2][:, 0, 0, 0].tolist()) == list(range(4))
    for row in rows:
        log_p = arrays[row["posterior_key"]]
        assert log_p.dtype == np.float32 and log_p.shape == (4, 69)
        np.testing.assert_allclose(np.exp(log_p.astype(np.float64)).sum(axis=1), 1., atol=2e-7)
        probability = exhaustive_ctc_probability(np.exp(log_p.astype(np.float64)), item["label"])
        expected_loss = -np.log(probability)/len(item["label"])
        assert row["ctc_loss_per_target_token"] == pytest.approx(expected_loss, rel=1e-6)
        assert row["frames"] == row["transform"]["input_frames"] == 4


@pytest.mark.parametrize("kind", ["dtype", "shape", "nan", "unnormalized"])
def test_invalid_posterior_rejected(kind):
    source, item = synthetic_clip()
    probs = synthetic_log_probabilities()
    if kind == "dtype":
        probs = probs.astype(np.float64)
    elif kind == "shape":
        probs = probs[:, :-1]
    elif kind == "nan":
        probs[0, 0] = np.nan
    else:
        probs += np.float32(.1)
    with pytest.raises(ValueError):
        runner.evaluate_clip(torch.nn.Identity().eval(), source, item, key="source6400", cohort="train60",
                             forward_fn=lambda *args, **kwargs: (probs, {}))


def test_training_mode_rejected_before_forward():
    source, item = synthetic_clip()
    calls = []
    with pytest.raises(ValueError, match="Evaluation mode"):
        runner.evaluate_clip(torch.nn.Identity().train(), source, item, key="source6400", cohort="train60",
                             forward_fn=lambda *args, **kwargs: calls.append(True))
    assert calls == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_actual_compact_cuda_synthetic_clip_preserves_state_rng_and_temporal_shape():
    torch.set_num_threads(2)
    runner.seed_training(42)
    model = LipNetBaseline().to("cuda").eval()
    # 각 has three reference jamo; no real cache or checkpoint is read.
    source, item = synthetic_clip(frames=8, text="각", size=96)
    assert len(item["label"]) == 3
    before_weights = runner.state_digest(model.state_dict())
    before_rng = capture_rng_state()
    rows, arrays = runner.evaluate_clip(model, source, item, key="source6400", cohort="train60")
    assert len(rows) == len(arrays) == 3
    assert [r["condition"] for r in rows] == runner.CONDITIONS
    assert sum(r["beam"] is not None for r in rows) == 1
    assert all(p.shape == (8, 69) and p.dtype == np.float32 and np.isfinite(p).all() for p in arrays.values())
    assert all(np.allclose(np.exp(p).sum(axis=1), 1, atol=2e-5) for p in arrays.values())
    assert runner.state_digest(model.state_dict()) == before_weights
    assert compare_nested(before_rng, capture_rng_state())["exact"]
    assert all(not module.training for module in model.modules())
    assert all(parameter.grad is None for parameter in model.parameters())
    del model
    torch.cuda.empty_cache()
