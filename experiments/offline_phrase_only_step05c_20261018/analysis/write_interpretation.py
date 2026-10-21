"""Analyze saved stage5c outputs only; never construct or call a neural model."""
from __future__ import annotations

from collections import Counter, defaultdict
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "experiments/offline_phrase_only_step05c_20261018"
OLD_CTC = ROOT / "experiments/offline_diagnostic_step05b_20261018/diagnose_ctc.py"
OLD_CTC_SHA = "0206e6691da3141403cbb600ba3102a5825ae0f29f12bd48fe8f7a3cefa1ed58"


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


ctc = module("stage5c_readonly_ctc_diagnostic", OLD_CTC)
assert ctc.sha(OLD_CTC) == OLD_CTC_SHA
agg = module("stage5c_readonly_result_aggregator", BASE / "evaluation/aggregate.py")
ev = agg.ev


def edit_distance(reference, prediction):
    previous = list(range(len(prediction) + 1))
    for i, left in enumerate(reference, 1):
        current = [i]
        for j, right in enumerate(prediction, 1):
            current.append(min(previous[j] + 1, current[-1] + 1, previous[j - 1] + (left != right)))
        previous = current
    return previous[-1]


def maximum_run(items):
    longest = current = 0
    last = object()
    for item in items:
        current = current + 1 if item == last else 1
        longest = max(longest, current)
        last = item
    return longest


def source_paths():
    return [BASE / f"runs/seed{s}/evaluations/update_{u:04d}/evaluation_summary.json"
            for s in [42, 43] for u in [420, 840, 1260, 1680]]


def main():
    paths = source_paths()
    aggregate_path = BASE / "results/aggregate_metrics.json"
    aggregate = ctc.read(aggregate_path)
    assert aggregate == agg.aggregate_summaries(paths)
    plan = ctc.read(BASE / "evaluation_plan.json")
    records = plan["items"]
    result_rows = []
    source_refs = [ctc.binding(aggregate_path), ctc.binding(BASE / "evaluation_plan.json"), ctc.binding(OLD_CTC)]
    source_refs.extend(ctc.binding(path) for path in paths)
    for path in paths:
        result = ctc.read(path)
        identity = result["identity"]
        for row in result["rows"]:
            ctc.check(row["posterior"])
            with np.load(row["posterior"]["path"], allow_pickle=False) as saved:
                lp = saved["log_probs"].copy()
                raw = saved["raw_argmax_ids"].tolist()
                collapsed = saved["collapsed_ids"].tolist()
            assert lp.shape == (row["T"], 69) and lp.dtype == np.float32 and np.isfinite(lp).all()
            assert lp.argmax(axis=1).tolist() == raw == row["raw_argmax_ids"]
            assert ctc.collapse(raw) == collapsed == row["collapsed_ids"]
            assert ev.digest_state(lp) == row["log_probs_array_sha256"]
            reference = row["reference_label_ids"]
            assert ctc.minimum_frames(reference) <= len(lp)
            target_nll = ctc.ctc_nll(lp, reference)
            greedy_nll = ctc.ctc_nll(lp, collapsed)
            assert np.isfinite(target_nll) and np.isfinite(greedy_nll)
            posterior_blank_sum = float(np.exp(lp.astype(np.float64)[:, 0]).sum())
            result_rows.append({
                "seed": identity["seed"], "completed_update": identity["completed_update"],
                "clip_id": row["clip_id"], "parent_id": row["parent_id"],
                "speaker_id": row["parent_id"].split("_")[5], "condition": row["condition"],
                "T": row["T"], "reference_text": row["reference_text"], "prediction_text": row["raw_text"],
                "metrics": row["metrics"], "reference_labels": reference, "collapsed_labels": collapsed,
                "reference_tokens": len(reference), "predicted_tokens": len(collapsed),
                "token_edits": edit_distance(reference, collapsed), "predicted_to_reference_tokens": len(collapsed) / len(reference),
                "ctc_minimum_frames": ctc.minimum_frames(reference), "frames_per_target_token": len(lp) / len(reference),
                "argmax_blank_frames": raw.count(0), "argmax_blank_ratio": raw.count(0) / len(raw),
                "posterior_blank_sum": posterior_blank_sum, "mean_blank_probability": posterior_blank_sum / len(lp),
                "maximum_collapsed_same_token_run": maximum_run(collapsed),
                "reference_ctc_nll": target_nll, "reference_ctc_nll_per_token": target_nll / len(reference),
                "greedy_ctc_nll": greedy_nll, "target_minus_greedy_nll": target_nll - greedy_nll,
                "posterior": row["posterior"],
            })
    assert len(result_rows) == 144 and sum(r["T"] for r in result_rows) == 5772
    groups = defaultdict(list)
    for row in result_rows:
        groups[(row["seed"], row["completed_update"], row["condition"])].append(row)
    cohorts = []
    for (seed, update, condition), selected in sorted(groups.items()):
        assert len(selected) == 12
        frames = sum(r["T"] for r in selected)
        labels = sum(r["reference_tokens"] for r in selected)
        assert frames == 481 and labels == 272
        cohorts.append({
            "seed": seed, "completed_update": update, "condition": condition, "count": len(selected),
            "metrics": ev.contract.corpus_scores([r["metrics"] for r in selected]),
            "frames": frames, "reference_tokens": labels, "predicted_tokens": sum(r["predicted_tokens"] for r in selected),
            "token_edits": sum(r["token_edits"] for r in selected),
            "token_error_rate": sum(r["token_edits"] for r in selected) / labels,
            "predicted_to_reference_tokens": sum(r["predicted_tokens"] for r in selected) / labels,
            "argmax_blank_frames": sum(r["argmax_blank_frames"] for r in selected),
            "argmax_blank_ratio": sum(r["argmax_blank_frames"] for r in selected) / frames,
            "mean_blank_probability": sum(r["posterior_blank_sum"] for r in selected) / frames,
            "target_nll_per_token_micro": sum(r["reference_ctc_nll"] for r in selected) / labels,
            "target_nll_per_token_clip_mean": float(np.mean([r["reference_ctc_nll_per_token"] for r in selected])),
            "replacement_characters": sum(r["metrics"]["replacement_characters"] for r in selected),
            "repetition_ge5_count": sum(r["metrics"]["repetition_ge5"] for r in selected),
            "target_probability_below_greedy_count": sum(r["target_minus_greedy_nll"] > 1e-9 for r in selected),
            "target_probability_above_greedy_count": sum(r["target_minus_greedy_nll"] < -1e-9 for r in selected),
            "target_probability_tied_greedy_count": sum(abs(r["target_minus_greedy_nll"]) <= 1e-9 for r in selected),
        })
    loss_curves = []
    for seed in [42, 43]:
        log_path = BASE / f"runs/seed{seed}/progress.jsonl"
        source_refs.append(ctc.binding(log_path))
        logs = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
        assert [r["update"] for r in logs] == list(range(1, 1681))
        assert all(r["kind"] == "phrase" and np.isfinite(r["loss"]) for r in logs)
        windows = []
        for start in range(0, 1680, 60):
            group = logs[start:start + 60]
            losses = [r["loss"] for r in group]
            windows.append({"first_update": start + 1, "last_update": start + 60,
                            "loss_mean": float(np.mean(losses)), "loss_minimum": min(losses), "loss_maximum": max(losses),
                            "learning_rates": sorted({r["learning_rate"] for r in group}),
                            "gradient_clipped_updates": sum(r["gradient_norm_before_clip"] > 1 for r in group)})
        loss_curves.append({"seed": seed, "count": len(logs), "windows": windows,
                            "last60_loss_mean": windows[-1]["loss_mean"], "previous60_loss_mean": windows[-2]["loss_mean"],
                            "finite_losses": len(logs), "updates_with_zero_gradient_tensor": sum(r["nonzero_gradient_tensors"] < r["parameter_tensors"] for r in logs)})
    speaker_records = defaultdict(list)
    length_records = defaultdict(list)
    for row in records:
        speaker_records[row["parent_id"].split("_")[5]].append(row)
        length_records[row["T"]].append(row)
    final_map = {(r["seed"], r["clip_id"], r["condition"]): r for r in result_rows if r["completed_update"] == 1680}
    retention = []
    for seed in [42, 43]:
        for row in records:
            conditions = {condition: final_map[(seed, row["clip_id"], condition)] for condition in ["normal", "static", "shuffled"]}
            speaker = row["parent_id"].split("_")[5]
            retention.append({"seed": seed, "clip_id": row["clip_id"], "speaker_id": speaker, "reference_text": row["text"],
                              "T": row["T"], "phrases_for_speaker": len(speaker_records[speaker]),
                              "unique_texts_for_speaker": len({r["text"] for r in speaker_records[speaker]}),
                              "clips_with_same_T": len(length_records[row["T"]]),
                              "exact_by_condition": {c: r["metrics"]["exact"] for c, r in conditions.items()},
                              "prediction_by_condition": {c: r["prediction_text"] for c, r in conditions.items()},
                              "all_three_exact": all(r["metrics"]["exact"] for r in conditions.values()),
                              "static_equals_normal_output": conditions["static"]["prediction_text"] == conditions["normal"]["prediction_text"],
                              "shuffled_equals_normal_output": conditions["shuffled"]["prediction_text"] == conditions["normal"]["prediction_text"]})
    by_speaker = []
    for speaker, items in sorted(speaker_records.items()):
        by_speaker.append({"speaker_id": speaker, "phrase_count": len(items), "unique_texts": len({r["text"] for r in items}),
                           "clips": [{"clip_id": r["clip_id"], "text": r["text"], "T": r["T"]} for r in items],
                           "final_by_seed": [{"seed": seed,
                             "exact_counts": {c: sum(final_map[(seed, r["clip_id"], c)]["metrics"]["exact"] for r in items)
                                              for c in ["normal", "static", "shuffled"]}}
                                             for seed in [42, 43]]})
    differences = []
    for seed in [42, 43]:
        selected = {r["condition"]: r for r in cohorts if r["seed"] == seed and r["completed_update"] == 1680}
        differences.append({"seed": seed,
            "static_minus_normal_CER_percentage_points": 100 * (selected["static"]["metrics"]["CER"] - selected["normal"]["metrics"]["CER"]),
            "shuffled_minus_normal_CER_percentage_points": 100 * (selected["shuffled"]["metrics"]["CER"] - selected["normal"]["metrics"]["CER"]),
            "static_exact_retained": sum(r["seed"] == seed and r["exact_by_condition"]["static"] for r in retention),
            "shuffled_exact_retained": sum(r["seed"] == seed and r["exact_by_condition"]["shuffled"] for r in retention),
            "all_three_exact": sum(r["seed"] == seed and r["all_three_exact"] for r in retention)})
    gate_failure = [r for r in result_rows if r["condition"] == "normal" and r["completed_update"] in [1260, 1680] and not r["metrics"]["exact"]]
    assert len(gate_failure) == 1 and gate_failure[0]["seed"] == 43 and gate_failure[0]["completed_update"] == 1260
    finding = {
        "schema": "offline-stage5c-saved-result-interpretation-v1", "client_date": "2026-10-18",
        "status": "completed_saved_output_analysis", "sources": source_refs,
        "source_code": ctc.binding(__file__), "fit_gate": aggregate["fit_gate"],
        "cohorts": cohorts, "loss_curves": loss_curves, "rows": result_rows,
        "gate_failure_rows": gate_failure,
        "final_normal_errors": [r for r in result_rows if r["completed_update"] == 1680 and r["condition"] == "normal" and not r["metrics"]["exact"]],
        "final_control_errors": [r for r in result_rows if r["completed_update"] == 1680 and r["condition"] != "normal" and not r["metrics"]["exact"]],
        "temporal_differences": differences, "final_retention_by_clip": retention,
        "speaker_distribution": by_speaker,
        "length_distribution": [{"T": length, "count": len(items), "clip_ids": [r["clip_id"] for r in items]} for length, items in sorted(length_records.items())],
        "dataset_counts": {"clips": len(records), "unique_texts": len({r["text"] for r in records}),
                           "speakers": len(speaker_records), "distinct_T": len(length_records),
                           "speakers_with_single_phrase": sum(len(items) == 1 for items in speaker_records.values()),
                           "clips_with_unique_T": sum(len(items) == 1 for items in length_records.values())},
        "interpretation_ko": [
            "최종1680회에는 두 초기값 모두 학습12구절을 정확히 재현했다. 작은 구절집합을 학습할 수 없다는 결론은 지지되지 않는다.",
            "seed43의1260회에 한 글자 오류가 남아 두 시점 모두 정확해야 하는 사전 기술적 기준은 미충족이다. 최종 성공을 숨기거나 전체모델 실패로 확대하지 않는다.",
            "정상 영상보다 정지·순서섞기 오류율이 높지만, 정지에서도 두 모델 모두8/12를 맞혔다. 정상학습 재현만으로 입술의 시간적 움직임을 읽었다고 판단할 수 없다.",
            "정지 조건은 가운데 입모양·개인 외양·총프레임수를 유지하고, 순서섞기는 프레임집합과길이를 유지한다. 해당단서 암기 가능성은 남지만 이 분석만으로 특정단서의 인과효과를 증명하지 못한다.",
            "평가144건은 동일12학습구절의 반복 관측이다. 새 구절·새 문장·새 화자·웹캠 성능은 측정하지 않았다.",
            "기존420 상태 재현을 전제로, 문장으로 전환하지 않고 구절학습을 지속하는 설계가 학습집합 적합도를 크게 높였다는 관측이다. 추가학습량과입력유지가함께바뀌므로 순수순서효과가 아니다.",
            "다음은 보유자료에서 같은화자의 다른 구절과 녹화·전사 분리를 검토해 새구절 진단을 설계하는 방향이 타당하다. 이번 기준에 맞추려 자동연장하거나 문장단계로 승격하지 않는다."
        ],
        "actual_work": {"saved_posterior_arrays_read": len(result_rows), "saved_posterior_frames": sum(r["T"] for r in result_rows),
                        "ctc_float64_calls": 2 * len(result_rows), "training_log_rows": 3360,
                        "new_neural_calls": 0, "new_optimizer_updates": 0, "checkpoint_tensor_loads": 0,
                        "source_media_reads": 0, "new_downloads": 0, "service_changes": 0},
        "statistics_definitions": {
            "CER": "Frozen NFC+outer-strip corpus character edits/ref characters; inner spaces retained.",
            "token_error_rate": "Levenshtein on actual collapsed 69-vocabulary IDs divided by total reference IDs; not NFC CER.",
            "argmax_blank_ratio": "Blank0 argmax frames / all481 frames in each condition cohort; frame-weighted.",
            "mean_blank_probability": "Sum of exp(saved log probability blank0) / all481 frames; not argmax fraction.",
            "target_nll_per_token_micro": "Sum of exact CPU Float64 CTC reference NLL /272 reference tokens; no renormalization of saved logits.",
            "training_loss_windows": "60 optimizer-update arithmetic means from original logs, each update has batch2; not held-out loss.",
            "temporal_control_limitation": "Static/shuffled comparisons on training clips alone do not identify speaker, appearance or duration causal contributions."
        }
    }
    ctc.save(BASE / "analysis/result_interpretation.json", finding)
    print(json.dumps({"status": finding["status"], "rows": len(result_rows), "fit_gate_passed": aggregate["fit_gate"]["passed"],
                      "final_normal_errors": len(finding["final_normal_errors"]), "dataset_counts": finding["dataset_counts"]}))


if __name__ == "__main__":
    main()
