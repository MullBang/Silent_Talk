"""완료된 시간축 진단을 원시 posterior와 대조해 검증하고 간결한 집계 JSON을 만든다.

모델 추론이나 학습을 실행하지 않는다. 저장된 beam 문자열을 재사용하며 greedy,
CTC/entropy/blank/대조 수치는 원시 log posterior에서 다시 계산한다. 특징 원배열은
진단 산출물에 없으므로 특징 차이는 해시로 보호된 행의 유효성 검사 후 집계한다.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import importlib.util
import json
import math
from pathlib import Path
import sys
import unicodedata

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "backend"), str(ROOT / "scripts")]

from models import GRAPHEME_VOCAB
from models.dataset_manifest import TEMPORAL_METHOD, load_manifest, sha256_file
from models.tokenizer import cer, wer, encode_text, validate_tokenizer_metadata
from test_model import aggregate_predictions, validate_evaluation_provenance
from temporal_diagnostics import METRIC_DEFINITIONS, analyze_ctc, compare_posteriors, transform_frames
from diagnose_temporal_use import (
    CONDITIONS, MANIFEST, SOURCE_PATHS, condition_name, length_bin, prediction_row, select_cohorts,
)

MODEL_KEYS = ("compact", "residual", "compact_short16", "residual_short16")
STAGES = {"cnn": 1536, "projection": 256, "lstm": 512, "logits": 69}
CTC_SCALARS = (
    "blank_posterior_mean", "argmax_blank_fraction", "mean_frame_entropy_nats",
    "mean_frame_entropy_normalized", "reference_token_length", "reference_ctc_min_frames",
    "ctc_slack_frames", "greedy_collapsed_token_count", "greedy_collapsed_unique_token_count",
    "greedy_decoded_nfd_length", "beam_decoded_nfd_length", "greedy_invalid_jamo_count",
    "beam_invalid_jamo_count", "top_nonblank_token_fraction",
)


def require(condition: bool, message: str) -> None:
    """최적화 실행에서도 자료 검증을 생략하지 않는다."""
    if not condition:
        raise ValueError(message)


def read_json(path: Path) -> dict:
    """UTF-8 JSON 객체만 읽는다."""
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON 객체 필요: {path}")
    return value


def compare_values(actual, expected, location: str, *, atol: float = 1e-8, rtol: float = 1e-7) -> None:
    """복합 진단 값을 재계산 결과와 비교하고 유한 실수만 허용한다."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and actual.keys() == expected.keys(), f"필드 불일치: {location}")
        for key, value in expected.items():
            compare_values(actual[key], value, f"{location}.{key}", atol=atol, rtol=rtol)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), f"배열 길이 불일치: {location}")
        for index, (a, b) in enumerate(zip(actual, expected)):
            compare_values(a, b, f"{location}[{index}]", atol=atol, rtol=rtol)
    elif isinstance(expected, (float, np.floating)):
        require(isinstance(actual, (float, int)) and not isinstance(actual, bool)
                and math.isfinite(actual) and math.isfinite(expected)
                and math.isclose(actual, float(expected), abs_tol=atol, rel_tol=rtol),
                f"수치 불일치: {location}: {actual!r} != {expected!r}")
    else:
        require(actual == expected, f"값 불일치: {location}: {actual!r} != {expected!r}")


def describe(values: list[float]) -> dict:
    """문장별 수치의 평균·중앙값·범위를 요약한다. 빈 집합은 평균을 만들지 않는다."""
    if not values:
        return {"count": 0}
    array = np.asarray(values, dtype=np.float64)
    require(np.isfinite(array).all(), "비유한 집계 입력")
    return {"count": len(values), "mean": float(array.mean()), "median": float(np.median(array)),
            "minimum": float(array.min()), "maximum": float(array.max())}


def pearson(x: list[float], y: list[float]) -> dict:
    """상관은 인과 효과가 아니며 상수/소수 표본에는 계수를 만들지 않는다."""
    require(len(x) == len(y), "상관 입력 길이 불일치")
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return {"pairs": len(x), "pearson_r": None, "reason": "fewer_than_two_or_constant_variable"}
    value = float(np.corrcoef(np.asarray(x, dtype=float), np.asarray(y, dtype=float))[0, 1])
    require(math.isfinite(value), "비유한 상관")
    return {"pairs": len(x), "pearson_r": value}


def slack_bin(slack: int) -> str:
    """정답 CTC 최소 정렬 길이보다 여유 있는 프레임 수를 사전 고정 구간으로 나눈다."""
    if slack < 0:
        return "negative_infeasible"
    if slack < 20:
        return "000-019"
    if slack < 50:
        return "020-049"
    if slack < 100:
        return "050-099"
    return "100+"


def token_frequency(rows: list[dict]) -> dict:
    """참조 토큰/greedy 원시 토큰과 beam 출력 NFD 문자를 구분하여 집계한다."""
    reference, greedy, beam_nfd = Counter(), Counter(), Counter()
    for row in rows:
        reference.update(encode_text(row["reference"]))
        greedy.update(row["ctc"]["greedy_collapsed_token_ids"])
        beam_nfd.update(unicodedata.normalize("NFD", row["prediction"]))
    def top(counter: Counter) -> list[dict]:
        total = sum(counter.values())
        return [{"token_id": token, "token": GRAPHEME_VOCAB[token], "count": count,
                 "fraction": count / total if total else 0.0,
                 "reference_count": reference[token], "greedy_count": greedy[token]}
                for token, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))[:10]]
    return {
        "reference_token_count": sum(reference.values()), "greedy_collapsed_token_count": sum(greedy.values()),
        "reference_counts_by_token_id": {str(k): v for k, v in sorted(reference.items())},
        "greedy_counts_by_token_id": {str(k): v for k, v in sorted(greedy.items())},
        "beam_decoded_nfd_counts": {f"U+{ord(k):04X}": v for k, v in sorted(beam_nfd.items())},
        "top_reference_tokens": top(reference), "top_greedy_tokens": top(greedy),
        "reference_tokens_absent_from_greedy_output": sorted(set(reference) - set(greedy)),
        "beam_length_note": "NFD text includes spaces and U+FFFD; not hidden beam prefix token IDs",
    }


def feature_summary(rows: list[dict], key: str) -> dict:
    """관측된 특징 행만 집계하고 전체 집단 중 실제 관측 수를 명시한다."""
    observed = [row for row in rows if row.get(key)]
    result = {"observed_clips": len(observed), "cohort_clips": len(rows), "stages": {}}
    for stage in STAGES:
        values = [row[key][stage] for row in observed]
        if values:
            names = [name for name in values[0] if name != "shape"]
            result["stages"][stage] = {name: describe([value[name] for value in values]) for name in names}
    return result


def summarize_rows(rows: list[dict], normal_by_id: dict, *, detailed: bool = True) -> dict:
    """동일 집단의 오류율·출력 패턴·입력 민감도를 문장 단위로 집계한다."""
    require(bool(rows), "빈 집단을 요약할 수 없습니다.")
    greedy_rows = [prediction_row({"id": row["id"], "text": row["reference"], "frames": row["frames"]},
                                  row["greedy_prediction"]) for row in rows]
    normal_text_equal = [row["prediction"] == normal_by_id[row["id"]]["prediction"] for row in rows]
    greedy_equal = [row["greedy_prediction"] == normal_by_id[row["id"]]["greedy_prediction"] for row in rows]
    result = {
        "beam_metrics": aggregate_predictions(rows), "greedy_metrics": aggregate_predictions(greedy_rows),
        "normal_beam_text_agreement_count": sum(normal_text_equal),
        "normal_beam_text_agreement_fraction": sum(normal_text_equal) / len(rows),
        "normal_greedy_text_agreement_fraction": sum(greedy_equal) / len(rows),
        "greedy_beam_same_fraction": sum(row["greedy_beam_same"] for row in rows) / len(rows),
        "ctc_loss_per_target_token": describe([row["ctc_loss_per_target_token"] for row in rows]),
        "ctc": {name: describe([row["ctc"][name] for row in rows]) for name in CTC_SCALARS},
        "collapsed_ngram_repetition": {
            str(n): describe([row["ctc"]["collapsed_ngram_repetition"][str(n)]["repeat_fraction"] for row in rows])
            for n in (2, 3, 4)},
        "beam_nfd_to_reference_token_length_ratio": describe([
            row["ctc"]["beam_decoded_nfd_length"] / row["ctc"]["reference_token_length"] for row in rows]),
        "greedy_collapsed_to_reference_token_length_ratio": describe([
            row["ctc"]["greedy_collapsed_token_count"] / row["ctc"]["reference_token_length"] for row in rows]),
    }
    changes = [row["comparison_to_normal"] for row in rows if "comparison_to_normal" in row]
    if changes:
        result["posterior_comparison_same_position"] = {
            name: describe([change[name] for change in changes]) for name in (
                "mean_total_variation", "max_total_variation", "argmax_path_agreement_fraction",
                "blank_posterior_mean_difference")}
        result["posterior_comparison_same_position"]["observed_clips"] = len(changes)
    else:
        result["posterior_comparison_same_position"] = {
            "observed_clips": len(rows), "mean_total_variation": describe([0.0] * len(rows)),
            "argmax_path_agreement_fraction": describe([1.0] * len(rows)), "normal_identity": True}
    if detailed:
        result["features"] = feature_summary(rows, "features")
        result["feature_comparison_same_position"] = feature_summary(rows, "feature_comparison_same_position")
        result["feature_comparison_source_frame_correspondence"] = feature_summary(
            rows, "feature_comparison_source_frame_correspondence")
        result["token_frequency"] = token_frequency(rows)
        result["length_correlations"] = {
            "frames_vs_beam_decoded_nfd_length": pearson(
                [row["frames"] for row in rows], [row["ctc"]["beam_decoded_nfd_length"] for row in rows]),
            "frames_vs_greedy_collapsed_token_count": pearson(
                [row["frames"] for row in rows], [row["ctc"]["greedy_collapsed_token_count"] for row in rows]),
            "reference_token_length_vs_beam_decoded_nfd_length": pearson(
                [row["ctc"]["reference_token_length"] for row in rows],
                [row["ctc"]["beam_decoded_nfd_length"] for row in rows]),
            "note": "correlation is descriptive, not evidence that length causes the prediction",
        }
    return result


def cohort_rows(rows: list[dict], cohorts: dict, short_only: bool) -> dict:
    """본학습의 전체/층화/암기 이력 집단과 암기 모델의16개 집단을 분리한다."""
    result = {}
    names = ("short16",) if short_only else ("val", "train_all_normal", "train_stratified", "short16")
    for name in names:
        if name == "train_all_normal":
            result[name] = [row for row in rows if row["split"] == "train" and row["condition"] == "normal"]
        else:
            selected = set(cohorts[name])
            result[name] = [row for row in rows if row["id"] in selected]
    return result


def shuffle_summary(by_condition: dict) -> dict:
    """같은 표본에서 세 seed의 평균·범위를 표시하며 독립 표본 수로 합치지 않는다."""
    names = [f"shuffle_{seed}" for seed in (42, 137, 2026)]
    if not all(name in by_condition for name in names):
        return {}
    values = [by_condition[name] for name in names]
    require(len({value["beam_metrics"]["clips"] for value in values}) == 1,
            "shuffle seed별 표본 수가 다릅니다.")
    result = {"seeds": [42, 137, 2026], "repeats": 3,
              "clips_per_seed": values[0]["beam_metrics"]["clips"],
              "note": "same clips repeated; ranges are not confidence intervals"}
    getters = {
        "beam_mean_sentence_cer": lambda value: value["beam_metrics"]["mean_sentence_cer"],
        "beam_corpus_wer": lambda value: value["beam_metrics"]["corpus_wer"],
        "beam_exact_count": lambda value: value["beam_metrics"]["exact_count"],
        "greedy_mean_sentence_cer": lambda value: value["greedy_metrics"]["mean_sentence_cer"],
        "blank_posterior_mean": lambda value: value["ctc"]["blank_posterior_mean"]["mean"],
        "mean_frame_entropy_nats": lambda value: value["ctc"]["mean_frame_entropy_nats"]["mean"],
        "beam_decoded_nfd_length": lambda value: value["ctc"]["beam_decoded_nfd_length"]["mean"],
        "normal_beam_text_agreement_fraction": lambda value: value["normal_beam_text_agreement_fraction"],
        "mean_total_variation": lambda value: value["posterior_comparison_same_position"]["mean_total_variation"]["mean"],
    }
    result["metrics_across_seeds"] = {name: describe([getter(value) for value in values])
                                      for name, getter in getters.items()}
    result["feature_relative_rms_across_seeds"] = {}
    for alignment in ("feature_comparison_same_position", "feature_comparison_source_frame_correspondence"):
        result["feature_relative_rms_across_seeds"][alignment] = {
            stage: describe([value[alignment]["stages"][stage]["relative_rms_difference"]["mean"] for value in values])
            for stage in STAGES}
    return result


def validate_features(row: dict, capture_expected: bool) -> None:
    """특징 원배열 부재를 숨기지 않고 저장 통계의 형상·범위만 검사한다."""
    features = row["features"]
    require(set(features) == (set(STAGES) if capture_expected else set()), "특징 관측 범위 불일치")
    for stage, values in features.items():
        require(values["shape"] == [row["frames"], STAGES[stage]], f"특징 형상 불일치: {stage}")
        require(set(values) == {"shape", "rms", "temporal_centered_rms", "temporal_centered_relative_rms",
                                "adjacent_rms", "adjacent_relative_rms"}, "특징 통계 필드 불일치")
        require(all(math.isfinite(value) and value >= 0 for key, value in values.items() if key != "shape"),
                "비유한/음수 특징 통계")
    control = row["condition"] != "normal"
    permutation = row["condition"].startswith("shuffle_") or row["condition"] == "reversed"
    for field, expected in (("feature_comparison_same_position", control),
                            ("feature_comparison_source_frame_correspondence", permutation)):
        require((field in row) == expected, f"특징 비교 범위 불일치: {field}")
        if expected:
            require(set(row[field]) == set(STAGES), "특징 비교 단계 누락")
            for values in row[field].values():
                require(set(values) == {"rms_difference", "relative_rms_difference", "mean_frame_cosine"},
                        "특징 차이 필드 불일치")
                require(all(math.isfinite(value) for value in values.values())
                        and values["rms_difference"] >= 0 and values["relative_rms_difference"] >= 0
                        and -1.000001 <= values["mean_frame_cosine"] <= 1.000001, "특징 차이 범위 오류")


def verify_transform(row: dict, condition_spec: dict) -> None:
    """픽셀을 다시 열지 않고 ID/seed/길이에 따른 순열·정지 위치 메타데이터를 대조한다."""
    _, expected = transform_frames(np.zeros((row["frames"], 1, 1, 3), dtype=np.uint8),
                                  condition_spec["transform"], sample_id=row["id"], seed=condition_spec["seed"])
    expected["input_shape"] = [row["frames"], 96, 96, 3]
    actual = row["transform"]
    require(set(actual) == set(expected), "변환 메타데이터 필드 불일치")
    for key, value in expected.items():
        if key != "pixel_changed_frame_count":
            compare_values(actual[key], value, f"transform.{key}")
    require(isinstance(actual["pixel_changed_frame_count"], int)
            and 0 <= actual["pixel_changed_frame_count"] <= row["frames"], "변경 프레임 수 범위 오류")
    if row["condition"] == "normal":
        require(actual["pixel_changed_frame_count"] == 0, "정상 입력 변환 오류")


def verify_reference_reports(spec: dict, checkpoint: dict, records: list[dict], summary: dict) -> dict:
    """정상 train1000/val20 결과와 모델 출처·예측 문자열·집계의 재사용 근거를 확인한다."""
    baseline = {}
    for split in ("train", "val"):
        path = Path(spec["folder"]) / f"best_{split}.json"
        require(sha256_file(path) == spec["reference_reports"][split], "기준선 JSON 변경")
        data = read_json(path)
        require(data["checkpoint_sha256"] == spec["checkpoint_sha256"]
                and Path(data["checkpoint"]).resolve() == Path(spec["checkpoint"]).resolve()
                and data["manifest_sha256"] == summary["manifest_sha256"]
                and data["training_data"] == checkpoint["training_data"]
                and data["training_run"] == checkpoint["training_run"]
                and data["decoder"] == "beam" and data["beam_width"] == 10
                and data["control"] is None and data["split"] == split
                and data["test_inference_performed"] is False
                and data["cross_preprocessing_diagnostic"] is False, "기준선 출처/설정 불일치")
        expected = [record for record in records if record["split"] == split]
        require([row["id"] for row in data["predictions"]] == [row["id"] for row in expected], "기준선 ID 불일치")
        for row, original in zip(data["predictions"], expected):
            calculated = prediction_row(original, row["prediction"])
            require(all(row[key] == value for key, value in calculated.items()), "기준선 문장 지표 불일치")
            baseline[row["id"]] = row
        compare_values(data["metrics"], aggregate_predictions(data["predictions"]), f"baseline.{split}")
    return baseline


def representative_examples(rows: list[dict], cohorts: dict, short_only: bool) -> dict:
    """미리 정한 명세 순서 예시와 사후 오류 최대 예시를 구분해 제시한다."""
    normal = {row["id"]: row for row in rows if row["condition"] == "normal"}
    if short_only:
        selections = {"first_two_selected_short16": cohorts["short16"][:2]}
    else:
        val = [normal[key] for key in cohorts["val"]]
        selections = {"first_two_validation_in_manifest_order": cohorts["val"][:2],
                      "highest_normal_validation_cer_then_id": [row["id"] for row in sorted(
                          val, key=lambda row: (-row["cer"], row["id"]))[:3]]}
    output = {"selection_rules": selections, "not_statistically_representative": True, "examples": {}}
    for sample_id in dict.fromkeys(key for values in selections.values() for key in values):
        source = normal[sample_id]
        output["examples"][sample_id] = {"reference": source["reference"], "frames": source["frames"],
            "speaker_id": source["speaker_id"], "conditions": {
                row["condition"]: {"prediction": row["prediction"], "cer": row["cer"],
                                   "greedy_prediction": row["greedy_prediction"], "greedy_cer": row["greedy_cer"],
                                   "blank_posterior_mean": row["ctc"]["blank_posterior_mean"],
                                   "greedy_collapsed_token_count": row["ctc"]["greedy_collapsed_token_count"]}
                for row in rows if row["id"] == sample_id}}
    return output


def process_model(key: str, plan: dict, completed: dict, document: dict, summary: dict, run: Path) -> tuple[dict, dict]:
    """하나의 완료 모델 산출물을 전수 검증하되 beam/model 추론은 수행하지 않는다."""
    spec, folder = plan["models"][key], run / key
    short_only = key.endswith("_short16")
    report_path = folder / "report.json"
    report = read_json(report_path)
    require(report == completed["models"][key], "완료 인덱스와 모델 보고서 불일치")
    require(report["status"] == "completed" and report["model"] == key
            and report["checkpoint"] == spec and report["manifest_sha256"] == summary["manifest_sha256"]
            and report["training_performed"] is False and report["test_inference_performed"] is False
            and report["short_overfit_diagnostic_only"] == short_only
            and report["device"] == "cuda" and report["precision"] == "fp32"
            and report["batch_size"] == 1 and report["beam_width"] == 10, "모델 보고서 설정 불일치")
    checkpoint_path = Path(spec["checkpoint"])
    require(sha256_file(checkpoint_path) == spec["checkpoint_sha256"], "체크포인트 변경")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    validate_tokenizer_metadata(checkpoint, str(checkpoint_path))
    require(checkpoint["architecture_version"] == ("lipnet-compact-v1" if key.startswith("compact") else "lipnet-residual-v1")
            and checkpoint["vocab"] == GRAPHEME_VOCAB and checkpoint["lstm_hidden"] == 256, "모델 구조/토큰 불일치")
    require(report["training_data"] == checkpoint["training_data"] and report["training_run"] == checkpoint["training_run"],
            "체크포인트와 보고서 학습 출처 불일치")
    by_id = {record["id"]: record for record in document["records"]}
    if short_only:
        provenance = checkpoint["training_data"]
        require(provenance.get("mode") == "smoke_only" and provenance.get("purpose") == "train_subset_overfit"
                and provenance.get("manifest_sha256") == summary["manifest_sha256"]
                and provenance.get("sample_ids") == plan["cohorts"]["short16"]
                and checkpoint["training_run"].get("overfit_only") is True
                and checkpoint["training_run"].get("step") == 1600, "short16 출처 불일치")
        todo = [by_id[sample_id] for sample_id in plan["cohorts"]["short16"]]
        baseline = {}
    else:
        validate_evaluation_provenance(checkpoint["training_data"], summary)
        training_path = Path(spec["folder"]) / "training.json"
        require(sha256_file(training_path) == spec["training_report_sha256"], "본 학습 기록 변경")
        training = read_json(training_path)
        require(training["status"] == "completed" and training["epochs_completed"] == 20
                and training["initial_training_data"]["sample_ids"] == plan["cohorts"]["short16"]
                and checkpoint["training_run"].get("bounded_smoke") is False
                and checkpoint["training_run"]["epoch"] == training["best_epoch"], "본 학습 출처 불일치")
        todo = [record for record in document["records"] if record["split"] in ("train", "val")]
        baseline = verify_reference_reports(spec, checkpoint, document["records"], summary)
    del checkpoint
    rows_path, posterior_path = folder / "rows.jsonl", folder / "posteriors.npz"
    require(sha256_file(rows_path) == report["rows_sha256"] and sha256_file(posterior_path) == report["posteriors_sha256"],
            "행 또는 원시 posterior 해시 불일치")
    rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    controls = set(plan["cohorts"]["train_controls_union"])
    expected_pairs = [(record["id"], condition_name(condition, seed)) for record in todo
                      for condition, seed in (CONDITIONS if (short_only or record["split"] == "val"
                                                             or record["id"] in controls) else CONDITIONS[:1])]
    require([(row["id"], row["condition"]) for row in rows] == expected_pairs, "진단 행 순서/표본/조건 누락")
    require(report["clips"] == len(todo) and report["forwards"] == len(rows), "진단 개수 불일치")
    require(report["normal_prediction_reproduction_matches"] == (0 if short_only else len(todo)), "정상 재현 개수 불일치")
    conditions = {item["name"]: item for item in plan["conditions"]}
    normal_by_id, normal_arrays = {}, {}
    max_loss_difference = 0.0
    with np.load(posterior_path, allow_pickle=False) as arrays:
        require(set(arrays.files) == {f"{sample_id}__{name}" for sample_id, name in expected_pairs}, "원시 posterior 키 불일치")
        for row in rows:
            source = by_id[row["id"]]
            require(source["split"] in ("train", "val"), "test 진단 금지")
            check = prediction_row(source, row["prediction"])
            require(all(row[name] == value for name, value in check.items()), "beam 문자열/정답/평가 수치 불일치")
            require(row["split"] == source["split"] and row["speaker_id"] == source["speaker_id"]
                    and row["length_bin"] == length_bin(source["frames"])
                    and row["short16"] == (row["id"] in plan["cohorts"]["short16"])
                    and row["stratified"] == (row["id"] in plan["cohorts"]["train_stratified"]), "진단 집단 메타데이터 불일치")
            verify_transform(row, conditions[row["condition"]])
            capture = short_only or source["split"] == "val" or source["id"] in controls
            validate_features(row, capture)
            probs = arrays[f"{row['id']}__{row['condition']}"]
            require(probs.dtype == np.float32 and probs.shape == (source["frames"], 69)
                    and np.isfinite(probs).all(), "원시 log posterior 형상/정밀도/유한성 불일치")
            labels = np.asarray(encode_text(source["text"]), dtype=np.int64)
            recalculated = analyze_ctc(probs, labels, beam_text=row["prediction"])
            compare_values(row["ctc"], recalculated, f"{key}/{row['id']}/{row['condition']}/ctc")
            require(row["greedy_prediction"] == recalculated["greedy_text"]
                    and row["greedy_cer"] == cer(source["text"], recalculated["greedy_text"])
                    and row["greedy_exact"] == (source["text"] == recalculated["greedy_text"])
                    and row["greedy_beam_same"] == (recalculated["greedy_text"] == row["prediction"]), "greedy 진단 불일치")
            loss = torch.nn.functional.ctc_loss(
                torch.from_numpy(probs[:, None, :]), torch.from_numpy(labels),
                torch.tensor([len(probs)]), torch.tensor([len(labels)]),
                blank=0, reduction="none", zero_infinity=False).item() / len(labels)
            require(math.isfinite(loss), "비유한 CTC loss")
            compare_values(row["ctc_loss_per_target_token"], loss, "ctc_loss_per_target_token", atol=1e-5, rtol=1e-5)
            max_loss_difference = max(max_loss_difference, abs(row["ctc_loss_per_target_token"] - loss))
            if row["condition"] == "normal":
                require("comparison_to_normal" not in row, "정상 행의 대조 메타데이터 오류")
                normal_by_id[row["id"]] = row
                normal_arrays[row["id"]] = probs
                if not short_only:
                    require(row["prediction"] == baseline[row["id"]]["prediction"], "기존 정상 예측 문자열 불일치")
            else:
                require(row["id"] in normal_arrays, "정상 posterior보다 앞선 대조 입력")
                difference = compare_posteriors(normal_arrays[row["id"]], probs,
                    normal_text=normal_by_id[row["id"]]["prediction"], control_text=row["prediction"])
                compare_values(row["comparison_to_normal"], difference, "comparison_to_normal")
    del normal_arrays
    prior_comparisons = {}
    if short_only:
        old_path = checkpoint_path.parent / "result.json"
        old = read_json(old_path)
        require(old["passed"] is True and old["checkpoint_sha256"] == spec["checkpoint_sha256"]
                and old["training_data"] == report["training_data"]
                and [item["id"] for item in old["selected"]] == plan["cohorts"]["short16"],
                "기존 short16 정상 결과 출처 불일치")
        require(len(old["final_gpu"]["samples"]) == 16, "기존 short16 결과 수 불일치")
        for selected, prior in zip(old["selected"], old["final_gpu"]["samples"]):
            current = normal_by_id[selected["id"]]
            require(prior["reference"] == current["reference"]
                    and prior["beam"] == current["prediction"]
                    and prior["greedy"] == current["greedy_prediction"], "기존 short16 정상 출력 재현 불일치")
        prior_comparisons["short16_result"] = {"path": str(old_path), "sha256": sha256_file(old_path),
                                                "normal_beam_and_greedy_prediction_matches": 16}
    else:
        # 계획은 정상 train/val JSON을 고정했다. 추가로 이전에 저장된 같은 가중치의
        # 회색/역순 val 예측도 확인하고 그 파일 해시를 이번 집계 결과에 남긴다.
        for control in ("constant_gray", "reversed"):
            old_path = Path(spec["folder"]) / f"best_val_{control}.json"
            old = read_json(old_path)
            current = [row for row in rows if row["split"] == "val" and row["condition"] == control]
            require(old["checkpoint_sha256"] == spec["checkpoint_sha256"]
                    and old["manifest_sha256"] == summary["manifest_sha256"]
                    and old["training_data"] == report["training_data"]
                    and old["training_run"] == report["training_run"]
                    and old["split"] == "val" and old["control"] == control
                    and old["decoder"] == "beam" and old["beam_width"] == 10
                    and old["test_inference_performed"] is False
                    and old["cross_preprocessing_diagnostic"] is False, "기존 대조 결과 출처 불일치")
            require([item["id"] for item in old["predictions"]] == [row["id"] for row in current],
                    "기존 대조 평가 ID 순서 불일치")
            for prior, actual in zip(old["predictions"], current):
                check = prediction_row(by_id[actual["id"]], prior["prediction"])
                require(all(prior[name] == value for name, value in check.items())
                        and actual["prediction"] == prior["prediction"], "기존 대조 예측/문장 지표 불일치")
            compare_values(old["metrics"], aggregate_predictions(current), f"prior.{control}.metrics")
            prior_comparisons[control] = {"path": str(old_path), "sha256": sha256_file(old_path),
                                          "beam_prediction_matches": len(current)}
    grouped = cohort_rows(rows, plan["cohorts"], short_only)
    report_groups, summarized = {}, {}
    for cohort, chosen in grouped.items():
        by_condition = {}
        for condition in conditions:
            subset = [row for row in chosen if row["condition"] == condition]
            if subset:
                report_groups[f"{cohort}/{condition}"] = aggregate_predictions(subset)
                by_condition[condition] = summarize_rows(subset, normal_by_id)
        summarized[cohort] = {"conditions": by_condition, "shuffle_repetitions": shuffle_summary(by_condition)}
    compare_values(report["groups"], report_groups, "runner.groups")
    result = {
        "checkpoint": spec, "architecture_version": "lipnet-compact-v1" if key.startswith("compact") else "lipnet-residual-v1",
        "scope": "memorization_training16_only" if short_only else "full_training_checkpoint_diagnosis",
        "training_run": report["training_run"], "clips": report["clips"], "forwards": report["forwards"],
        "elapsed_seconds": report["elapsed_seconds"], "cohorts": summarized,
        "representative_examples": representative_examples(rows, plan["cohorts"], short_only),
        "verification": {"report_sha256": sha256_file(report_path), "rows_sha256": report["rows_sha256"],
                         "posteriors_sha256": report["posteriors_sha256"],
                         "raw_posterior_rows_recalculated": len(rows), "ctc_loss_max_absolute_difference": max_loss_difference,
                         "normal_baseline_prediction_matches": report["normal_prediction_reproduction_matches"],
                         "additional_prior_result_checks": prior_comparisons,
                         "beam_redecoded": False, "feature_raw_arrays_available": False,
                         "feature_values_verification": "hashed row provenance, expected shapes, finite values and ranges only"},
    }
    if not short_only:
        normal_train = [row for row in rows if row["condition"] == "normal" and row["split"] == "train"]
        result["full_train_normal_breakdowns"] = {}
        for name, getter in (("length_bins", lambda row: row["length_bin"]),
                             ("speakers", lambda row: row["speaker_id"]),
                             ("ctc_slack_bins", lambda row: slack_bin(row["ctc"]["ctc_slack_frames"]))):
            groups = defaultdict(list)
            for row in normal_train:
                groups[getter(row)].append(row)
            result["full_train_normal_breakdowns"][name] = {
                group: summarize_rows(items, normal_by_id, detailed=False) for group, items in sorted(groups.items())}
    require(sha256_file(checkpoint_path) == spec["checkpoint_sha256"]
            and sha256_file(rows_path) == report["rows_sha256"]
            and sha256_file(posterior_path) == report["posteriors_sha256"], "집계 도중 원본 산출물 변경")
    print(f"[검증] {key}: {len(rows)} posterior행, 정상 일치 {report['normal_prediction_reproduction_matches']}", flush=True)
    return result, normal_by_id


def main() -> None:
    """완료된 진단만 집계하여 새 JSON 파일 한 개에 기록한다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="완료된 results/step11-temporal/run01 등")
    parser.add_argument("--out", required=True, help="새 집계 JSON 경로; 기존 파일 덮어쓰기 금지")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    torch.set_num_threads(2)
    run, output = Path(args.run).resolve(), Path(args.out).resolve()
    require(run.is_relative_to(ROOT / "results") and run.is_dir(), "프로젝트 results 하위 완료 폴더가 필요합니다.")
    require(output.is_relative_to(ROOT) and output.suffix == ".json" and not output.is_relative_to(run),
            "원본 진단 폴더 밖의 프로젝트 JSON 경로가 필요합니다.")
    if output.exists():
        raise FileExistsError(f"기존 집계 파일을 덮어쓰지 않습니다: {output}")
    plan_path, completed_path = run / "plan.json", run / "completed.json"
    plan, completed = read_json(plan_path), read_json(completed_path)
    plan_hash, completed_hash = sha256_file(plan_path), sha256_file(completed_path)
    require(completed["status"] == "completed" and completed["plan_sha256"] == plan_hash
            and completed["training_performed"] is False and completed["test_inference_performed"] is False,
            "완료 상태/계획 해시 불일치")
    require(plan["stage"] == 11 and plan["purpose"] == "fixed_checkpoint_temporal_and_ctc_diagnosis"
            and plan["training_performed"] is False and plan["test_inference_performed"] is False
            and plan["model_selection_performed"] is False, "11단계 고정 가중치 진단 계획이 아닙니다.")
    require(set(plan["models"]) == set(completed["models"]) == set(MODEL_KEYS), "진단 모델 누락/추가")
    require(Path(plan["manifest"]).resolve() == MANIFEST.resolve(), "v4 이외 명세")
    document, summary = load_manifest(MANIFEST, verify_files=False)
    require(plan["manifest_sha256"] == summary["manifest_sha256"]
            and document["policy"]["temporal_method"] == TEMPORAL_METHOD, "명세 해시/시간축 불일치")
    require({split: info["clips"] for split, info in summary["splits"].items()}
            == {"train": 1000, "val": 20, "test": 50}, "고정 분할 크기 불일치")
    compare_values(plan["cohorts"], select_cohorts(document["records"]), "plan.cohorts")
    compare_values(plan["conditions"], [{"name": condition_name(c, s), "transform": c, "seed": s}
                                      for c, s in CONDITIONS], "plan.conditions")
    require(set(plan["sources"]) == set(SOURCE_PATHS), "진단 소스 목록 불일치")
    for path, expected in plan["sources"].items():
        require(sha256_file(ROOT / path) == expected, f"진단 소스 변경: {path}")
    verified_caches = 0
    for row in document["records"]:
        if row["split"] in ("train", "val"):
            require(sha256_file(MANIFEST.parent / row["cache_path"]) == row["cache_sha256"], "train/val 캐시 변경")
            verified_caches += 1
    models, normals = {}, {}
    for key in MODEL_KEYS:
        models[key], normals[key] = process_model(key, plan, completed, document, summary, run)
    pretraining = {}
    short_ids = plan["cohorts"]["short16"]
    for architecture in ("compact", "residual"):
        before = [normals[f"{architecture}_short16"][sample_id] for sample_id in short_ids]
        after = [normals[architecture][sample_id] for sample_id in short_ids]
        pretraining[architecture] = {
            "scope": "same 16 training samples, different checkpoints; not unseen-speaker performance",
            "before": summarize_rows(before, normals[f"{architecture}_short16"]),
            "after": summarize_rows(after, normals[architecture]),
            "beam_text_unchanged_count": sum(a["prediction"] == b["prediction"] for a, b in zip(before, after)),
            "beam_exact_retained_count": sum(a["exact"] and b["exact"] for a, b in zip(before, after)),
            "beam_exact_lost_count": sum(a["exact"] and not b["exact"] for a, b in zip(before, after)),
            "greedy_exact_retained_count": sum(a["greedy_exact"] and b["greedy_exact"] for a, b in zip(before, after)),
        }
    result = {
        "stage": 11, "status": "completed", "summary_created_at": datetime.now(timezone.utc).isoformat(),
        "run": str(run), "plan_sha256": plan_hash, "completed_sha256": completed_hash,
        "manifest": str(MANIFEST), "manifest_sha256": summary["manifest_sha256"],
        "verified_train_val_cache_hashes": verified_caches, "cohorts": plan["cohorts"],
        "conditions": plan["conditions"], "models": models, "short16_before_after_full_training": pretraining,
        "metric_definitions": METRIC_DEFINITIONS, "vocabulary_by_token_id": dict(enumerate(GRAPHEME_VOCAB)),
        "aggregation": {"ctc_and_feature_means": "equal weight per clip unless metric explicitly says corpus",
                        "length_bins": ["001-050", "051-100", "101-150", "151-200"],
                        "ctc_slack_bins": ["negative_infeasible", "000-019", "020-049", "050-099", "100+"],
                        "overlap_stratified_short16": len(set(plan["cohorts"]["train_stratified"]) & set(short_ids)),
                        "shuffle": "same clips, 3 deterministic permutations; means/ranges, not confidence intervals"},
        "verification": {"source_sha256": plan["sources"], "summarizer_sha256": sha256_file(Path(__file__)),
                         "posterior_statistics_recalculated": True, "ctc_loss_recalculated_on_cpu": True,
                         "beam_redecoded": False, "feature_raw_arrays_available": False,
                         "test_cache_opened": False, "matplotlib_available": importlib.util.find_spec("matplotlib") is not None},
        "training_performed": False, "model_inference_performed_by_summarizer": False,
        "test_inference_performed": False, "model_selection_performed": False, "deployed": False,
        "limitations": [*plan["limits"],
                        "feature arrays are not saved; feature statistics are aggregated after hash/shape/range checks, not recomputed",
                        "high blank or repetition rates alone do not establish a CTC training failure",
                        "beam text was checked against stored references and edit metrics; full beam decoding was not repeated",
                        "full train1000 normal, stratified60 and short16 overlap but are separate populations",
                        "correlations between length and output length do not establish causation"],
    }
    require(sha256_file(plan_path) == plan_hash and sha256_file(completed_path) == completed_hash
            and sha256_file(MANIFEST) == summary["manifest_sha256"], "집계 도중 계획/완료/명세 변경")
    for path, expected in plan["sources"].items():
        require(sha256_file(ROOT / path) == expected, f"집계 중 소스 변경: {path}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"[완료] {output}", flush=True)


if __name__ == "__main__":
    main()
