"""Post-main CPU-only reporting using the already frozen evaluator aggregation.

This script requires the complete 5,040-update execution summary. It does not
load checkpoints, run neural inference, change frozen code, or select weights.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
EXECUTION_PATH = BASE / "results/execution_summary.json"
AGGREGATE_PATH = BASE / "results/aggregate_metrics.json"
INTERPRETATION_PATH = BASE / "analysis/result_interpretation.json"
AGGREGATOR = BASE / "evaluation/aggregate.py"
EVALUATOR = BASE / "evaluation/evaluator.py"
FROZEN_SHA = {
    AGGREGATOR: "2c4bcae55c4dd0d3d5cd84fef2a87c5575009daa5bfbf5ff90f3196c8b975acc",
    EVALUATOR: "1f75bcc10865d07c6d67c8806a9e082b5e23c4c3ca3c6c276694839aeaa29277",
}


def sha(path: Path) -> str:
    """Compute SHA256 for an existing evidence artifact."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_ref(binding: dict) -> dict:
    """Follow only a hash-bound stage5 evidence reference."""
    path = Path(binding["path"]).resolve()
    require(path.is_relative_to(BASE), "Evidence reference outside stage5")
    require(sha(path) == binding["sha256"], f"Evidence reference changed: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def save(path: Path, value: dict) -> dict:
    """Create an output without replacing an earlier report."""
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size}


def main() -> None:
    """Recompute fixed aggregates and preserve actual final dev predictions."""
    require(not AGGREGATE_PATH.exists() and not INTERPRETATION_PATH.exists(), "Reporting already attempted; preserve prior outputs")
    for path, expected in FROZEN_SHA.items():
        require(sha(path) == expected, f"Frozen evaluation source changed: {path}")
    execution = json.loads(EXECUTION_PATH.read_text(encoding="utf-8"))
    require(execution["status"] == "completed" and execution["physical_main_optimizer_updates"] == 5040
            and execution["evaluation_rows"] == 1080 and execution["checkpoint_count"] == 12,
            "All fixed main runs must complete before aggregation")
    paths = []
    records = []
    for run_binding in execution["runs"]:
        run = read_ref(run_binding)
        require(run["status"] == "completed" and run["completed_updates"] == 1260, "Incomplete comparison run")
        for result in run["evaluations"]:
            records.append(read_ref(result["summary"]))
            paths.append(Path(result["summary"]["path"]))
    spec = importlib.util.spec_from_file_location("stage5_frozen_aggregate_for_report", AGGREGATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    aggregate = module.aggregate_evaluations(paths)
    require(aggregate["failures"] == 0, "Failure-containing comparison requires explicit separate interpretation")
    aggregate_ref = save(AGGREGATE_PATH, aggregate)
    finals = [r for r in records if r["identity"]["completed_update"] == 1260]
    final_normal = [row for record in finals for row in record["rows"]
                    if row["split"] == "dev" and row["kind"] == "full_sentence" and row["condition"] == "normal"]
    dev_cases = []
    for clip_id in sorted({r["clip_id"] for r in final_normal}):
        case = [r for r in final_normal if r["clip_id"] == clip_id]
        require(len(case) == 4 and len({r["reference_text"] for r in case}) == 1, "Final dev example coverage changed")
        dev_cases.append({"clip_id": clip_id, "reference": case[0]["reference_text"],
            "predictions": [{"seed": r["identity"]["seed"], "arm": r["identity"]["arm"],
                "prediction": r["raw_text"], "character_edits": r["metrics"]["character_edits"],
                "reference_characters": r["metrics"]["reference_characters"],
                "CER_percent": 100 * r["metrics"]["character_edits"] / r["metrics"]["reference_characters"],
                "exact": r["metrics"]["exact"], "repetition_ge5": r["metrics"]["repetition_ge5"],
                "replacement_characters": r["metrics"]["replacement_characters"]}
                for r in sorted(case, key=lambda r: (r["identity"]["seed"], r["identity"]["arm"]))]})
    headline = []
    paired = aggregate["paired_primary"]
    mean = aggregate["mean_primary"]
    if mean["B_lower_CER_both_seeds"]:
        headline.append("현재 고정 내부 검증 3문장에서 구절→문장 방식의 최종 CER가 두 초기값 모두 낮았다. 소규모 추가 연구 근거이며 실용적 인식이나 미지 화자 성능을 입증하지 않는다.")
    elif all(p["B_CER"] >= p["A_CER"] for p in paired):
        headline.append("현재 고정 내부 검증 3문장에서 구절→문장 방식의 최종 CER는 두 초기값 모두 문장 직접 학습보다 낮지 않았다. 이번 조건에서 단계 학습의 개선을 확인하지 못했다.")
    else:
        headline.append("최종 내부 검증 CER의 A/B 우열이 초기값에 따라 달랐다. 두 초기값에서 일관된 단계 학습 개선을 확인하지 못했다.")
    models = aggregate["final_models"]
    if all(m["primary_dev_sentence_normal"]["exact_count"] == 0 for m in models):
        headline.append("네 모델 모두 학습에 넣지 않은 내부 검증 문장 완전 일치가 0/3이다. 현재 결과를 임의의 한국어 웹캠 발화 인식 성공으로 설명할 수 없다.")
    if all(m["cohorts"]["train/full_sentence/normal"]["exact_count"] == 12 for m in models):
        headline.append("네 모델 모두 학습 문장 12개를 맞혀 암기는 가능했지만 새 문장 인식과는 차이가 남았다.")
    else:
        counts = ", ".join(f"{m['arm']} seed{m['seed']}={m['cohorts']['train/full_sentence/normal']['exact_count']}/12" for m in models)
        headline.append(f"최종 학습 문장 완전 일치는 {counts}이다. 학습 문장 재현과 새 문장 성능을 따로 판단해야 한다.")
    temporal = [{"seed": m["seed"], "arm": m["arm"],
                 "static_minus_normal_CER_pp": m["primary_static_minus_normal_CER_pp"],
                 "shuffled_minus_normal_CER_pp": m["primary_shuffled_minus_normal_CER_pp"]} for m in models]
    interpretation = {"schema": "offline-stage5-result-interpretation-v1", "status": "completed_fixed_final_analysis",
        "execution": {"path": str(EXECUTION_PATH), "sha256": sha(EXECUTION_PATH)}, "aggregate": aggregate_ref,
        "analysis_code": {"path": str(Path(__file__).resolve()), "sha256": sha(Path(__file__).resolve())},
        "headlines_ko": headline, "paired_primary": paired, "mean_primary": mean,
        "final_train_dev_scores": [{"seed": m["seed"], "arm": m["arm"],
            "train_sentence": m["cohorts"]["train/full_sentence/normal"],
            "train_phrase": m["cohorts"]["train/phrase/normal"],
            "dev_sentence": m["cohorts"]["dev/full_sentence/normal"], "dev_phrase": m["cohorts"]["dev/phrase/normal"],
            "output_diagnostics": m["output_diagnostics"]} for m in models],
        "final_dev_sentence_actual_predictions": dev_cases, "temporal_control_differences": temporal,
        "interpretation_limits_ko": ["내부 검증은 같은 기존 화자 범위의 3문장뿐이다. 최종 시험이나 미지 화자 평가가 아니다.",
            "A/B의 업데이트 수와 부모별 노출 수는 같지만 B의 실제 학습 프레임은 약20.04% 적다. 순서 효과만 분리한 실험이 아니다.",
            "두 경로 모두 새 compact 모델과 CTC만 사용했다. 이전 교사 특징 전달 모델과 수치를 직접 비교하지 않는다.",
            "중간420/840 결과로 가중치를 고르거나 계획을 바꾸지 않았다. 최종1260을 사전 규약대로 비교했다.",
            "정지/섞기 대조의 오류율 차이는 시간 정보 활용 진단이며 조음 이해나 최종 웹캠 성능의 증명은 아니다.",
            "분석은 저장된 결과를 읽기만 했으며 추가 신경망 실행, 재학습, 서비스 가중치 교체는 없다."],
        "planned_next_step_gate_ko": "원래 6단계의 문장·여러 문장 확장은 새 발화 인식 개선이 확인될 때 진행한다. 현재 작은 개발군 결과와 실패 양상을 먼저 해석하고 후속 설계를 별도 고정해야 한다.",
        "new_neural_calls": 0, "new_optimizer_updates": 0, "checkpoint_selection": "fixed_final_1260_only"}
    interpretation_ref = save(INTERPRETATION_PATH, interpretation)
    print(json.dumps({"aggregate": aggregate_ref, "interpretation": interpretation_ref,
                      "paired_primary": paired, "mean_primary": mean, "headlines_ko": headline}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    main()
