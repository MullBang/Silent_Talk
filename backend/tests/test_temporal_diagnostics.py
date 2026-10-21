"""Temporal controls and CTC statistics without loading any model or GPU."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from temporal_diagnostics import (  # noqa: E402
    CONDITIONS, analyze_ctc, compare_posteriors, transform_frames,
)


def one_hot_logs(token_ids: list[int]) -> np.ndarray:
    result = np.full((len(token_ids), 69), -np.inf)
    result[np.arange(len(token_ids)), token_ids] = 0
    return result


def probability_logs(values: np.ndarray) -> np.ndarray:
    result = np.full_like(values, -np.inf, dtype=np.float64)
    np.log(values, out=result, where=values > 0)
    return result


@pytest.mark.parametrize("condition", CONDITIONS)
def test_controls_preserve_original_dtype_shape_and_have_independent_storage(condition):
    # Noncontiguous input also exercises reversed indexing without mutating its owner.
    original = np.arange(6*2*3*3, dtype=np.uint8).reshape(6,2,3,3)[::-1]
    before = original.copy()
    result, metadata = transform_frames(original, condition, sample_id="clip1", seed=42)
    assert result.shape == original.shape and result.dtype == np.uint8
    assert result.flags.c_contiguous and not np.shares_memory(result, original)
    np.testing.assert_array_equal(original, before)
    assert metadata["condition"] == condition and metadata["input_frames"] == 6
    if condition == "normal":
        np.testing.assert_array_equal(result, original)
    elif condition == "static_middle":
        assert metadata["middle_frame_index"] == 3
        np.testing.assert_array_equal(result, np.repeat(original[3:4],6,axis=0))
    elif condition == "reversed":
        np.testing.assert_array_equal(result, original[::-1])
    elif condition == "constant_gray":
        assert np.all(result == 128)
    else:
        permutation = metadata["permutation"]
        assert sorted(permutation) == list(range(6))
        np.testing.assert_array_equal(result, original[permutation])
        # Exact multiset preservation includes every pixel of every frame.
        assert sorted(frame.tobytes() for frame in result) == sorted(frame.tobytes() for frame in original)
    assert metadata["pixel_changed_frame_count"] == int(np.any(result != original,axis=(1,2,3)).sum())
    result.fill(0)
    np.testing.assert_array_equal(original, before)


def test_shuffle_is_independent_of_call_order_model_and_global_random_state():
    original = np.arange(20*3, dtype=np.uint8).reshape(20,1,1,3)
    np.random.seed(17)
    expected_next_global_random = np.random.random()
    np.random.seed(17)
    first, first_meta = transform_frames(original,"shuffle",sample_id="same_clip",seed=91)
    transform_frames(original,"shuffle",sample_id="other_clip",seed=91)
    second, second_meta = transform_frames(original,"shuffle",sample_id="same_clip",seed=91)
    assert np.random.random() == expected_next_global_random
    np.testing.assert_array_equal(first,second)
    assert first_meta == second_meta
    _, changed_id = transform_frames(original,"shuffle",sample_id="different_clip",seed=91)
    _, changed_seed = transform_frames(original,"shuffle",sample_id="same_clip",seed=92)
    assert first_meta["permutation"] != changed_id["permutation"]
    assert first_meta["permutation"] != changed_seed["permutation"]


@pytest.mark.parametrize("hash_seed", ["1", "999"])
def test_import_remains_torch_free_and_shuffle_is_stable_across_processes(hash_seed):
    code = "\n".join([
        "import json, sys, numpy as np",
        f"sys.path.insert(0, {str(ROOT / 'scripts')!r})",
        "import temporal_diagnostics as d",
        "frames=np.arange(60,dtype=np.uint8).reshape(20,1,1,3)",
        "_, metadata=d.transform_frames(frames,'shuffle',sample_id='sample_17',seed=42)",
        "print(json.dumps({'torch_loaded':'torch' in sys.modules,'permutation':metadata['permutation']}))",
    ])
    env = {**os.environ, "PYTHONHASHSEED":hash_seed}
    child = subprocess.run([sys.executable,"-c",code],check=True,capture_output=True,text=True,env=env)
    actual = json.loads(child.stdout)
    _, metadata = transform_frames(np.arange(60,dtype=np.uint8).reshape(20,1,1,3),
                                    "shuffle",sample_id="sample_17",seed=42)
    assert actual == {"torch_loaded":False,"permutation":metadata["permutation"]}


def test_single_frame_shuffle_records_identity_instead_of_claiming_a_change():
    original = np.full((1,2,2,3),70,dtype=np.uint8)
    actual, metadata = transform_frames(original,"shuffle",sample_id="one")
    assert metadata["permutation"] == [0]
    assert metadata["permutation_is_identity"] is True
    assert metadata["pixel_changed_frame_count"] == 0
    np.testing.assert_array_equal(actual,original)


@pytest.mark.parametrize("frames", [np.zeros((0,2,2,3),dtype=np.uint8),
    np.zeros((2,2,2,1),dtype=np.uint8), np.zeros((2,2,2,3),dtype=np.float32)])
def test_invalid_frame_contract_is_rejected(frames):
    with pytest.raises(ValueError,match="uint8 RGB"):
        transform_frames(frames,"normal",sample_id="clip")


def test_ctc_collapse_preserves_blank_separated_repeated_tokens_and_reports_slack():
    raw = [1,1,0,1,20,20,0,1,20]
    labels = np.array([1,1,20,1,20],dtype=np.int64)
    result = analyze_ctc(one_hot_logs(raw),labels,beam_text="가\ufffd")
    assert result["greedy_raw_token_ids"] == raw
    assert result["greedy_collapsed_token_ids"] == [1,1,20,1,20]
    assert result["greedy_collapsed_token_count"] == 5
    assert result["greedy_collapsed_unique_token_ids"] == [1,20]
    assert result["greedy_collapsed_unique_token_count"] == 2
    assert result["greedy_text"] == "\ufffd가가"
    assert result["greedy_invalid_jamo_count"] == 1
    assert result["beam_text"] == "가\ufffd" and result["beam_text_source"] == "supplied"
    assert result["beam_decoded_nfd_length"] == 3 and result["beam_invalid_jamo_count"] == 1
    assert result["reference_adjacent_repeats"] == 1
    assert result["reference_ctc_min_frames"] == 6 and result["ctc_slack_frames"] == 3
    assert result["reference_ctc_feasible"] is True
    assert result["argmax_blank_fraction"] == pytest.approx(2/9)
    assert result["blank_posterior_mean"] == pytest.approx(2/9)
    assert result["mean_frame_entropy_nats"] == 0
    assert result["argmax_nonblank_count"] == 7
    assert result["top_nonblank_token_id"] == 1
    assert result["top_nonblank_token_fraction"] == pytest.approx(4/7)
    assert result["collapsed_ngram_repetition"]["2"] == {
        "windows":4,"unique_windows":3,"repeated_windows":1,"repeat_fraction":.25}


def test_beam_decoder_reuse_and_ngram_rates_have_explicit_overlapping_windows():
    result = analyze_ctc(one_hot_logs([1,0,20,0,1,0,20,0]), np.array([1,20,1,20]))
    assert result["greedy_text"] == result["beam_text"] == "가가"
    assert result["greedy_decoded_nfd_length"] == result["beam_decoded_nfd_length"] == 4
    assert result["beam_text_source"] == "decoded_here"
    assert result["collapsed_ngram_repetition"]["2"]["repeat_fraction"] == pytest.approx(1/3)
    assert result["collapsed_ngram_repetition"]["3"]["windows"] == 2
    assert result["collapsed_ngram_repetition"]["4"]["repeat_fraction"] == 0


def test_uniform_posteriors_entropy_blank_probability_and_argmax_are_distinct():
    result = analyze_ctc(np.full((3,69),-np.log(69)), np.array([1,20]),beam_text="")
    assert result["blank_posterior_mean"] == pytest.approx(1/69)
    # Equal scores choose token0 for greedy; a low blank posterior can coexist with blank argmax.
    assert result["argmax_blank_fraction"] == 1
    assert result["mean_frame_entropy_nats"] == pytest.approx(np.log(69))
    assert result["mean_frame_entropy_normalized"] == pytest.approx(1)
    assert result["top_nonblank_token_id"] is None
    assert result["top_nonblank_token_fraction"] == 0
    assert result["collapsed_ngram_repetition"]["2"]["windows"] == 0
    assert result["collapsed_ngram_repetition"]["2"]["repeat_fraction"] == 0


def test_infeasible_reference_is_reported_as_negative_slack_not_hidden():
    result = analyze_ctc(one_hot_logs([1,20]),np.array([1,1,20]),beam_text="가")
    assert result["reference_ctc_min_frames"] == 4
    assert result["ctc_slack_frames"] == -2 and result["reference_ctc_feasible"] is False


def test_comparison_matches_total_variation_math_and_separates_beam_text_agreement():
    normal = np.zeros((2,69))
    control = np.zeros((2,69))
    normal[0,0],normal[1,0],normal[1,1] = 1,.5,.5
    control[0,1],control[1,0],control[1,1] = 1,.25,.75
    result = compare_posteriors(probability_logs(normal),probability_logs(control),
                                normal_text="same beam",control_text="same beam")
    assert result["mean_total_variation"] == pytest.approx((1+.25)/2)
    assert result["max_total_variation"] == 1
    assert result["argmax_path_agreement_fraction"] == 0
    assert result["greedy_collapsed_token_ids_equal"] is False
    assert result["greedy_text_equal"] is False
    assert result["text_equal"] is True
    assert result["blank_posterior_mean_difference"] == pytest.approx(-.625)
    identity = compare_posteriors(probability_logs(normal),probability_logs(normal),
                                  normal_text="",control_text="")
    assert identity["mean_total_variation"] == 0 and identity["argmax_path_agreement_fraction"] == 1


@pytest.mark.parametrize("bad", [np.zeros((2,69)), np.full((2,69),-np.inf),
    np.full((2,69),np.nan), np.zeros((2,68)), np.zeros((0,69))])
def test_invalid_posterior_inputs_do_not_silently_become_probabilities(bad):
    with pytest.raises(ValueError):
        analyze_ctc(bad,np.array([1,20]),beam_text="")


@pytest.mark.parametrize("labels", [np.array([0,20]),np.array([1,69]),np.array([1.,20.])])
def test_reference_tokens_must_be_nonblank_integer_vocabulary_ids(labels):
    with pytest.raises(ValueError,match="labels"):
        analyze_ctc(one_hot_logs([1,20]),labels,beam_text="")


def test_posterior_comparison_requires_matching_time_axes():
    with pytest.raises(ValueError,match="identical shapes"):
        compare_posteriors(one_hot_logs([1,20]),one_hot_logs([1]),normal_text="가",control_text="")
