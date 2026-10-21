"""Post-hoc analysis of all18 saved dev-probe posteriors; no neural calls."""
from collections import defaultdict
import importlib.util
import json
from pathlib import Path
import unicodedata

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "experiments/offline_unseen_phrase_step05d_20261018"
TRAIN_PLAN = ROOT / "experiments/offline_phrase_only_step05c_20261018/evaluation_plan.json"
CTC_PATH = ROOT / "experiments/offline_diagnostic_step05b_20261018/diagnose_ctc.py"
CTC_SHA = "0206e6691da3141403cbb600ba3102a5825ae0f29f12bd48fe8f7a3cefa1ed58"


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


ctc = module("stage5d_saved_ctc", CTC_PATH)
assert ctc.sha(CTC_PATH) == CTC_SHA
agg = module("stage5d_saved_aggregation", BASE / "evaluation/aggregate.py")
ev = agg.ev


def distance(reference, prediction):
    previous = list(range(len(prediction) + 1))
    for i, left in enumerate(reference, 1):
        current = [i]
        for j, right in enumerate(prediction, 1):
            current.append(min(previous[j] + 1, current[-1] + 1, previous[j - 1] + (left != right)))
        previous = current
    return previous[-1]


def nearest_records(prediction, train, speaker=None):
    selected = [r for r in train if speaker is None or r["speaker_id"] == speaker]
    assert selected
    scored = [{**r, "character_edit_distance_to_prediction": distance(ev.contract.metric_text(r["text"]), ev.contract.metric_text(prediction))}
              for r in selected]
    minimum = min(r["character_edit_distance_to_prediction"] for r in scored)
    return {"candidate_clip_count": len(scored), "candidate_unique_texts": len({r["text"] for r in scored}),
            "minimum_character_edits": minimum,
            "all_tied_nearest_records": [r for r in scored if r["character_edit_distance_to_prediction"] == minimum]}


def main():
    paths = [BASE / f"runs/seed{s}/evaluation_summary.json" for s in [42, 43]]
    aggregate_path = BASE / "results/aggregate_metrics.json"
    aggregate = ctc.read(aggregate_path)
    assert aggregate == agg.aggregate_summaries(paths)
    train = ctc.read(TRAIN_PLAN)["items"]
    assert len(train) == 12 and all(r["split"] == "train" and r["kind"] == "phrase" for r in train)
    train = [{"clip_id": r["clip_id"], "parent_id": r["parent_id"], "speaker_id": r["parent_id"].split("_")[5],
              "text": r["text"], "T": r["T"]} for r in train]
    rows = []
    for path in paths:
        result = ctc.read(path)
        for row in result["rows"]:
            ctc.check(row["posterior"])
            with np.load(row["posterior"]["path"], allow_pickle=False) as saved:
                lp = saved["log_probs"].copy()
                raw = saved["raw_argmax_ids"].tolist()
                collapsed = saved["collapsed_ids"].tolist()
            assert lp.dtype == np.float32 and lp.shape == (row["T"], 69) and np.isfinite(lp).all()
            assert lp.argmax(axis=1).tolist() == raw == row["raw_argmax_ids"]
            assert ctc.collapse(raw) == collapsed == row["collapsed_ids"]
            assert ev.digest_state(lp) == row["log_probs_array_sha256"]
            decoded = ev.contract.decode_prediction(raw)
            assert decoded["raw_text"] == row["raw_text"]
            labels = row["reference_label_ids"]
            target_nll = ctc.ctc_nll(lp, labels)
            greedy_nll = ctc.ctc_nll(lp, collapsed)
            assert np.isfinite(target_nll) and np.isfinite(greedy_nll)
            speaker = row["parent_id"].split("_")[5]
            nearest_all = nearest_records(row["raw_text"], train)
            nearest_same = nearest_records(row["raw_text"], train, speaker)
            rows.append({
                "seed": result["identity"]["seed"], "completed_update": 1680, "clip_id": row["clip_id"],
                "speaker_id": speaker, "condition": row["condition"], "T": row["T"],
                "reference_text": row["reference_text"], "prediction_text": row["raw_text"], "metrics": row["metrics"],
                "reference_labels": labels, "collapsed_labels": collapsed,
                "reference_tokens": len(labels), "predicted_tokens": len(collapsed), "predicted_to_reference_tokens": len(collapsed) / len(labels),
                "token_edits": distance(labels, collapsed), "argmax_blank_frames": raw.count(0), "argmax_blank_ratio": raw.count(0) / len(raw),
                "posterior_blank_sum": float(np.exp(lp.astype(np.float64)[:, 0]).sum()),
                "ctc_minimum_frames": ctc.minimum_frames(labels), "frames_per_target_token": len(lp) / len(labels),
                "target_ctc_nll": target_nll, "target_ctc_nll_per_token": target_nll / len(labels),
                "greedy_ctc_nll": greedy_nll, "target_minus_greedy_nll": target_nll - greedy_nll,
                "target_probability_below_greedy": target_nll - greedy_nll > 1e-9,
                "nearest_train_all": nearest_all, "nearest_train_same_speaker": nearest_same,
                "nearest_all_contains_same_speaker": any(r["speaker_id"] == speaker for r in nearest_all["all_tied_nearest_records"]),
                "exactly_matches_train_text": any(ev.contract.metric_text(row["raw_text"]) == ev.contract.metric_text(r["text"]) for r in train),
                "same_speaker_training_phrases": [r for r in train if r["speaker_id"] == speaker],
                "posterior": row["posterior"],
            })
    assert len(rows) == 18 and sum(r["T"] for r in rows) == 576
    cohorts = []
    for seed in [42, 43]:
        for condition in ["normal", "static", "shuffled"]:
            selected = [r for r in rows if r["seed"] == seed and r["condition"] == condition]
            assert len(selected) == 3 and sum(r["T"] for r in selected) == 96
            labels = sum(r["reference_tokens"] for r in selected)
            cohorts.append({"seed": seed, "condition": condition, "count": 3,
                "metrics": ev.contract.corpus_scores([r["metrics"] for r in selected]),
                "frames": 96, "reference_tokens": labels, "predicted_tokens": sum(r["predicted_tokens"] for r in selected),
                "predicted_to_reference_tokens": sum(r["predicted_tokens"] for r in selected) / labels,
                "token_edits": sum(r["token_edits"] for r in selected), "token_error_rate": sum(r["token_edits"] for r in selected) / labels,
                "argmax_blank_ratio": sum(r["argmax_blank_frames"] for r in selected) / 96,
                "mean_blank_probability": sum(r["posterior_blank_sum"] for r in selected) / 96,
                "target_nll_per_token_micro": sum(r["target_ctc_nll"] for r in selected) / labels,
                "target_probability_below_greedy_count": sum(r["target_probability_below_greedy"] for r in selected),
                "replacement_character_count": sum(r["metrics"]["replacement_characters"] for r in selected),
                "repetition_ge5_count": sum(r["metrics"]["repetition_ge5"] for r in selected),
                "exact_train_text_output_count": sum(r["exactly_matches_train_text"] for r in selected),
                "nearest_train_contains_same_speaker_count": sum(r["nearest_all_contains_same_speaker"] for r in selected)})
    controls = []
    for seed in [42, 43]:
        ids = sorted({r["clip_id"] for r in rows if r["seed"] == seed})
        for clip_id in ids:
            values = {r["condition"]: r for r in rows if r["seed"] == seed and r["clip_id"] == clip_id}
            normal = values["normal"]
            controls.append({"seed": seed, "clip_id": clip_id, "reference_text": normal["reference_text"],
                             "predictions": {c: r["prediction_text"] for c, r in values.items()},
                             "static_identical_text_to_normal": values["static"]["prediction_text"] == normal["prediction_text"],
                             "shuffled_identical_text_to_normal": values["shuffled"]["prediction_text"] == normal["prediction_text"],
                             "static_identical_tokens_to_normal": values["static"]["collapsed_labels"] == normal["collapsed_labels"],
                             "shuffled_identical_tokens_to_normal": values["shuffled"]["collapsed_labels"] == normal["collapsed_labels"],
                             "character_edits": {c: r["metrics"]["character_edits"] for c, r in values.items()}})
    temporal = []
    for seed in [42, 43]:
        metrics = {r["condition"]: r["metrics"] for r in cohorts if r["seed"] == seed}
        temporal.append({"seed": seed,
                        "static_minus_normal_CER_percentage_points": 100 * (metrics["static"]["CER"] - metrics["normal"]["CER"]),
                        "shuffled_minus_normal_CER_percentage_points": 100 * (metrics["shuffled"]["CER"] - metrics["normal"]["CER"]),
                        "static_same_text_count": sum(r["seed"] == seed and r["static_identical_text_to_normal"] for r in controls),
                        "shuffled_same_text_count": sum(r["seed"] == seed and r["shuffled_identical_text_to_normal"] for r in controls)})
    normal = [r for r in rows if r["condition"] == "normal"]
    interpretation = {
        "schema": "offline-stage5d-saved-output-interpretation-v1", "client_date": "2026-10-18", "status": "completed_saved_output_analysis",
        "sources": [ctc.binding(path) for path in [aggregate_path, TRAIN_PLAN, CTC_PATH, BASE / "evaluation_plan.json", *paths]],
        "analysis_code": ctc.binding(__file__), "rows": rows, "normal_predictions": normal,
        "cohorts": cohorts, "temporal_differences": temporal, "condition_output_comparisons": controls,
        "train_metadata": train,
        "normal_summary": {"observations": 6, "independent_dev_clips": 3, "correct": sum(r["metrics"]["exact"] for r in normal),
                           "mean_seed_CER": aggregate["primary"]["mean_CER"],
                           "target_probability_below_greedy_count": sum(r["target_probability_below_greedy"] for r in normal),
                           "exact_train_text_output_count": sum(r["exactly_matches_train_text"] for r in normal),
                           "nearest_train_contains_same_speaker_count": sum(r["nearest_all_contains_same_speaker"] for r in normal),
                           "replacement_character_count": sum(r["metrics"]["replacement_characters"] for r in normal)},
        "interpretation_ko": [
            "학습12구절의 최종 재현은 두 모델 모두12/12였으나, 이번 다른구절3개는 두 모델 모두0/3이다. 이 소규모조건에서 학습집합 적합도와 새구절 인식의 차이를 확인했다.",
            "정상 입력이 정지·섞기보다 나은 문자오류율을 보이지 않았다. 이 재사용dev3에서는 시간순서 활용의 인식상 이점을 확인하지 못했지만 모든상황에서 움직임을 무시한다는 인과결론은 아니다.",
            "정답/greedy 전체CTC확률은 저장확률의 진단이다. 확률차는 새디코더나언어모델이해결할성능을측정한것이아니다.",
            "최근접학습구절은 실제예측문자열과의편집거리 비교다. 길이·후보수·주제 차이가 있으며 화자암기의 증거나 open-vocabulary정확도로 해석하지 않는다.",
            "이 dev3은 기존에 분석·설계에 사용된 같은화자 자료이다. 새화자나독립시험,30초웹캠성능은측정하지않았다. 18평가는독립18문장이아니다.",
            "5c의엄격한두시점기준미충족상태와원자료provisional/training_readyfalse를유지한다. 이번결과로자동추가학습·문장확대·서비스교체를진행하지않는다."
        ],
        "next_priority_ko": [
            "같은화자에게서 여러서로다른구절이반복적으로관측되도록보유자료후보를조사하고,녹화와전사를분리한추가검수계획을먼저고정한다.",
            "학습문구와정확히일치하는추측만이아닌문자·자모조합실패가있으므로,현재고정모델의입력길이·정지영상영향을비교하는작은별도진단을설계할수있다. 아직실행하지않았다.",
            "새평가자료는원래검수범위·미응답·보류를보존하며확인된자료만추가한다.254개메타데이터를즉시학습자료로승격하지않고,대용량추가다운로드보다보유자료검토를우선한다."
        ],
        "method_limits": {
            "statistics": "NFC outer-strip character CER includes spaces and U+FFFD. Token error uses actual collapsed vocabulary IDs.",
            "nearest_train": "Raw character edit distance to prediction; all ties retained. All-pool12clips/11texts vs same-speaker1clip for each dev speaker are unequal candidate sets.",
            "CTC": "NumPyFloat64 CTC of savedFP32 log probabilities, no renormalization; target-minus-greedy NLL positive means reference has lower total CTC probability.",
            "blank": "Argmax frame count ratio and mean posterior probability are distinct statistics.",
            "no_decoder_counterfactual": "Beam search and language models were not executed."
        },
        "actual_work": {"saved_posteriors_read": 18, "saved_posterior_frames": 576, "ctc_float64_calls": 36,
                        "train_metadata_records": 12, "new_neural_calls": 0, "new_optimizer_updates": 0,
                        "checkpoint_tensor_loads": 0, "source_media_reads": 0, "new_downloads": 0, "service_changes": 0},
        "prior5c_strict_fit_gate_remains_failed": True,
        "exploratory_reused_development_probe": True, "generalization_established": False,
    }
    ctc.save(BASE / "analysis/result_interpretation.json", interpretation)
    print(json.dumps(interpretation["normal_summary"]))


if __name__ == "__main__":
    main()
