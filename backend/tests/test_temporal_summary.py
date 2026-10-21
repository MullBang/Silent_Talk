"""시간축 진단 집계의 집단 분리·반복실험 해석·변조 거부를 작은 행으로 검증한다."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "backend"), str(ROOT / "scripts")]

import summarize_temporal_diagnostics as summary
from models.tokenizer import encode_text
from temporal_diagnostics import analyze_ctc


@pytest.fixture
def overlapping_cohorts():
    """train a는 층화집단과 선행학습집단 양쪽에 속한다."""
    cohorts = {"train_stratified": ["a", "b"], "short16": ["a", "c"], "val": ["v"],
               "train_controls_union": ["a", "b", "c"]}
    rows = [{"id": sample_id, "split": "val" if sample_id == "v" else "train", "condition": condition}
            for sample_id in ("a", "b", "c", "v") for condition in ("normal", "static_middle")]
    return cohorts, rows


def test_short_only_model_never_appears_as_full_training_or_validation(overlapping_cohorts):
    cohorts, rows = overlapping_cohorts
    grouped = summary.cohort_rows(rows, cohorts, short_only=True)
    assert set(grouped) == {"short16"}
    assert {row["id"] for row in grouped["short16"]} == {"a", "c"}
    assert all(row["split"] == "train" for row in grouped["short16"])


def test_stratified_and_memorization_overlap_is_not_added_to_training_population(overlapping_cohorts):
    cohorts, rows = overlapping_cohorts
    original = copy.deepcopy(rows)
    grouped = summary.cohort_rows(rows, cohorts, short_only=False)
    assert set(grouped) == {"val", "train_all_normal", "train_stratified", "short16"}
    assert [row["id"] for row in grouped["train_all_normal"]] == ["a", "b", "c"]
    assert all(row["condition"] == "normal" for row in grouped["train_all_normal"])
    assert {row["id"] for row in grouped["train_stratified"]} == {"a", "b"}
    assert {row["id"] for row in grouped["short16"]} == {"a", "c"}
    assert len(grouped["train_all_normal"]) == 3  # 2+2=4로 중복 합산하지 않는다.
    assert rows == original


def _shuffle_condition(score: float, count: int = 60) -> dict:
    """집단별 실제 표본 수와 scalar 요약을 가진 합성 seed 결과."""
    feature = {"stages": {stage: {"relative_rms_difference": {"mean": score}}
                           for stage in summary.STAGES}}
    return {
        "beam_metrics": {"clips": count, "mean_sentence_cer": score, "corpus_wer": score * 2,
                         "exact_count": round(score * 10)},
        "greedy_metrics": {"mean_sentence_cer": score},
        "ctc": {"blank_posterior_mean": {"mean": score}, "mean_frame_entropy_nats": {"mean": score},
                "beam_decoded_nfd_length": {"mean": score * 100}},
        "normal_beam_text_agreement_fraction": 1 - score,
        "posterior_comparison_same_position": {"mean_total_variation": {"mean": score}},
        "feature_comparison_same_position": copy.deepcopy(feature),
        "feature_comparison_source_frame_correspondence": copy.deepcopy(feature),
    }


def test_shuffle_seeds_are_three_repetitions_of_same_sixty_samples_not_180_independent_samples():
    result = summary.shuffle_summary({f"shuffle_{seed}": _shuffle_condition(score)
                                      for seed, score in zip((42, 137, 2026), (.2, .4, .6))})
    assert result["seeds"] == [42, 137, 2026]
    assert result["repeats"] == 3 and result["clips_per_seed"] == 60
    metric = result["metrics_across_seeds"]["beam_mean_sentence_cer"]
    assert metric["count"] == 3
    assert metric["mean"] == pytest.approx(.4)
    assert metric["minimum"] == .2 and metric["maximum"] == .6
    assert "not confidence intervals" in result["note"]
    assert "confidence_interval" not in metric and "pooled_clips" not in result
    assert result["feature_relative_rms_across_seeds"]["feature_comparison_source_frame_correspondence"]["cnn"]["mean"] == pytest.approx(.4)


def test_shuffle_rejects_changed_sample_count_across_seeds():
    values = {f"shuffle_{seed}": _shuffle_condition(.2) for seed in (42, 137, 2026)}
    values["shuffle_2026"]["beam_metrics"]["clips"] = 59
    with pytest.raises(ValueError, match="표본 수"):
        summary.shuffle_summary(values)


@pytest.mark.parametrize("x,y", [([1., 1., 1.], [2., 3., 4.]), ([1., 2.], [9., 9.]), ([1.], [2.]), ([], [])])
def test_constant_or_insufficient_vectors_have_no_correlation_coefficient(x, y):
    result = summary.pearson(x, y)
    assert result["pearson_r"] is None
    assert result["pairs"] == len(x)


def test_pearson_reports_direction_without_claiming_causation():
    result = summary.pearson([1., 2., 3.], [6., 4., 2.])
    assert result["pearson_r"] == pytest.approx(-1.0)
    with pytest.raises(ValueError, match="길이"):
        summary.pearson([1., 2.], [1.])


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_summary_values_are_rejected(bad):
    with pytest.raises(ValueError, match="비유한"):
        summary.describe([1.0, bad])
    with pytest.raises(ValueError, match="수치 불일치"):
        summary.compare_values({"blank_mean": bad}, {"blank_mean": .5}, "ctc")


def _ctc_row() -> dict:
    """실제 모델 없이 정규화된 posterior로 유효한 짧은 CTC 진단 행을 만든다."""
    logits = np.full((4, 69), -8., dtype=np.float32)
    for index, token in enumerate([1, 0, 20, 0]):
        logits[index, token] = 0.
    log_probs = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
    reference = "가"
    row = summary.prediction_row({"id": "sample", "text": reference, "frames": 4}, "�")
    row.update({"ctc": analyze_ctc(log_probs, np.asarray(encode_text(reference), dtype=np.int64), beam_text="�"),
                "greedy_prediction": "가", "greedy_beam_same": False, "greedy_exact": True,
                "ctc_loss_per_target_token": .2, "features": {}, "condition": "normal"})
    return row


def test_altered_ctc_metric_path_or_structure_fails_recursive_validation():
    expected = _ctc_row()["ctc"]
    for field, value in (("blank_posterior_mean", .01), ("greedy_collapsed_token_ids", [1, 21]),
                         ("beam_text", "가")):
        altered = copy.deepcopy(expected)
        altered[field] = value
        with pytest.raises(ValueError, match="불일치"):
            summary.compare_values(altered, expected, "raw_posterior_check")
    missing = copy.deepcopy(expected)
    missing.pop("input_frames")
    with pytest.raises(ValueError, match="필드"):
        summary.compare_values(missing, expected, "raw_posterior_check")


def test_beam_replacement_marker_frequency_is_preserved_without_reencoding_invalid_prediction():
    row = _ctc_row()
    result = summary.token_frequency([row])
    assert result["reference_token_count"] == 2
    assert result["greedy_collapsed_token_count"] == 2
    assert result["beam_decoded_nfd_counts"] == {"U+FFFD": 1}
    assert result["greedy_counts_by_token_id"] == {"1": 1, "20": 1}


def test_normal_aggregate_separates_beam_errors_greedy_errors_and_identity_comparison():
    row = _ctc_row()
    result = summary.summarize_rows([row], {row["id"]: row})
    assert result["beam_metrics"]["exact_count"] == 0
    assert result["greedy_metrics"]["exact_count"] == 1
    assert result["normal_beam_text_agreement_fraction"] == 1.
    assert result["posterior_comparison_same_position"]["mean_total_variation"]["mean"] == 0.
    assert result["features"]["observed_clips"] == 0
    assert result["features"]["cohort_clips"] == 1
    assert result["length_correlations"]["frames_vs_beam_decoded_nfd_length"]["pearson_r"] is None


@pytest.mark.parametrize("status,plan_hash", [("running", "unused"), ("completed", "tampered")])
def test_incomplete_or_changed_plan_is_rejected_before_summary_output(tmp_path, monkeypatch, status, plan_hash):
    run = tmp_path / "results" / "run01"
    run.mkdir(parents=True)
    (run / "plan.json").write_text("{}", encoding="utf-8")
    (run / "completed.json").write_text(json.dumps({"status": status, "plan_sha256": plan_hash}), encoding="utf-8")
    output = tmp_path / "docs" / "summary.json"
    monkeypatch.setattr(summary, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["summarize", "--run", str(run), "--out", str(output)])
    with pytest.raises(ValueError, match="완료 상태/계획 해시"):
        summary.main()
    assert not output.exists() and not output.parent.exists()


def test_existing_summary_is_never_overwritten(tmp_path, monkeypatch):
    run = tmp_path / "results" / "run01"
    run.mkdir(parents=True)
    output = tmp_path / "summary.json"
    output.write_bytes(b"original output")
    monkeypatch.setattr(summary, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["summarize", "--run", str(run), "--out", str(output)])
    with pytest.raises(FileExistsError, match="덮어쓰지"):
        summary.main()
    assert output.read_bytes() == b"original output"
