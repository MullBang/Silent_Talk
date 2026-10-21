"""Publish the audited stage04 development result and frozen-rule decision."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
LABELS = {
    "annotation_bbox": "제공 입술 좌표(참고 조건)",
    "rectangle_control": "기존 자동 직사각형",
    "square": "자동 정사각형",
    "context_square": "자동 정사각형 + 주변 1.5배",
    "stable_context_square": "자동 정사각형 + 주변 1.5배 + EMA",
}


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    result_path = ROOT / "results/step04.json"
    audit_path = ROOT / "results/step04_metrics_audit.json"
    result, audit = read(result_path), read(audit_path)
    assert result["status"] == "completed" and len(result["records"]) == 100
    assert audit["status"] == "passed" and audit["result_sha256"] == sha(result_path)
    assert audit["verified"]["output_metric_records"] == 200
    aggregate = result["aggregate"]
    hybrid = {a: aggregate[f"{a}/corrected_hybrid"] for a in LABELS}
    plain, stable = hybrid["context_square"], hybrid["stable_context_square"]
    checks = {
        "strictly_fewer_total_character_edits": stable["character_edits"] < plain["character_edits"],
        "no_speaker_worse": all(stable["per_speaker"][s]["character_edits"] <= v["character_edits"]
                                for s, v in plain["per_speaker"].items()),
        "no_extra_failed_inputs": stable["failed_inputs"] <= plain["failed_inputs"],
    }
    candidates = [a for a in LABELS if a != "annotation_bbox"]
    if not all(checks.values()):
        candidates.remove("stable_context_square")
    selected = min(candidates, key=lambda a: (hybrid[a]["character_edits"], candidates.index(a)))
    assert selected == "stable_context_square", "Update narrative if underlying validated result changes"
    lookup = {(r["arm"], r["id"]): r for r in result["records"]}
    paired = {}
    for base in ["rectangle_control", "context_square"]:
        deltas = [lookup[selected, sid]["outputs"]["corrected_hybrid"]["metrics"]["character_edits"]
                  - lookup[base, sid]["outputs"]["corrected_hybrid"]["metrics"]["character_edits"]
                  for sid in result["plan"]["source_ids"]]
        paired[base] = {"improved": sum(x < 0 for x in deltas), "tied": sum(x == 0 for x in deltas),
                        "worsened": sum(x > 0 for x in deltas), "total_edit_delta": sum(deltas)}
    decision = {
        "schema": "olkavs-step04-development-decision-v1", "status": "audited_development_candidate",
        "plan_sha256": result["plan_sha256"], "result_sha256": sha(result_path),
        "independent_audit_sha256": sha(audit_path), "report_script_sha256": sha(Path(__file__)),
        "selection_rule": result["plan"]["selection"], "smoothing_checks": checks,
        "selected_automatic_arm": selected, "selected_decoder": "corrected_hybrid",
        "geometry": result["plan"]["geometry"][selected], "paired_sentence_counts": paired,
        "selected_metrics": hybrid[selected], "production_model_changed": False,
        "original_labels_changed": False, "new_training": False, "new_aihub_download": False,
        "test50_used": False, "prior_old_model_reassessment_required": True,
        "limits": ["Repeated development20 from two speakers, not final test or live webcam.",
                   "Public pretraining overlap unknown; source audio used only for clock audit.",
                   "C025 error remains high; offset alone does not resolve all recognition errors.",
                   "These two C025 recordings were validation, not train; training corruption is unproven."]}
    decision_path = ROOT / "results/step04_decision.json"
    decision_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    report_path = PROJECT / "docs/공식모델_도입검증/04_실행결과.md"
    old = report_path.read_text(encoding="utf-8")
    diagnosis = old.split("## 진행 중 발견한 정답 연결 문제", 1)[1].split("## 기록 파일", 1)[0]
    diagnosis = diagnosis.replace("짧은 파형을 MP4의 음성과 비교하여 이동량과 일관성을 확인한다.",
                                  "짧은 파형을 MP4의 음성과 비교하여 이동량과 일관성을 확인했다.")
    diagnosis = diagnosis.replace("확정된 파형 근거가 있을 때만 별도 시간 매핑 파일을 만들며 원본 JSON과 기존 실험 결과는 보존한다.",
                                  "파형 근거를 확보한 뒤 별도 시간 매핑 파일을 만들었고 원본 JSON과 기존 실험 결과는 보존했다.")
    diagnosis = diagnosis.replace("아직 실행하지 않은 개선 효과를 성과로 기록하지 않는다.",
                                  "효과는 위의 동일 조건 비교 결과로만 보고하며 범용 성능이나 신규 신경망의 성과로 확대하지 않는다.")
    lines = ["# 새 계획 4단계 — 시간 정렬 재점검과 자동 입술 영역 교정", "",
             "실행일: 2026-09-24(KST). **완료 및 독립 검산 통과.** 보유 영상 네 개의 20문장을 사용했다. "
             "C025 두 녹화의 라벨 시간과 MP4 시간 사이에 −8.362초 차이를 확인해 평가용 매핑을 교정했다. "
             "같은 교정 시간에서 자동 입술 처리 개선으로 문자 오류율(CER)이 **75.11% → 33.58%**, 문장 완전 일치는 **0/20 → 5/20**이 됐다. "
             "이는 두 화자의 개발 자료 결과이며 웹캠 완성 성능을 뜻하지 않는다.", "",
             "새 학습·AIHub 데이터 다운로드·서비스 모델 교체는 진행하지 않았다. 원본 라벨과 기존 실험 결과를 보존했고, "
             "최종 test50은 사용하지 않았다. 모델의 입력은 영상뿐이다.", "",
             "## 완료 결과와 선택", "",
             "20문장 × 5가지 입력 × 2가지 디코딩 = **200개 출력**을 모두 평가했다. "
             "주 지표는 NFC 정규화·앞뒤 공백 제거 후 내부 공백을 포함한 총 문자 편집거리/667자이며, "
             "WER는 총 어절 편집거리/189어절이다. 문장별 CER의 단순 평균이나 ‘정확도’가 아니다. 모든 입력·출력이 성공했다.", "",
             "| 입력 방식 | Hybrid 문자 오류 | CER | WER | 완전 일치 | CTC 직접 출력 CER |",
             "|---|---:|---:|---:|---:|---:|"]
    for arm, label in LABELS.items():
        h, c = hybrid[arm], aggregate[f"{arm}/ctc_greedy"]
        lines.append(f"| {label} | {h['character_edits']}/667 | {h['cer_percent']:.2f}% | {h['wer_percent']:.2f}% | "
                     f"{h['exact_sentences']}/20 | {c['cer_percent']:.2f}% |")
    lines += ["", "CTC 직접 출력의 문장 완전 일치는 다섯 조건 모두 0/20이다. Hybrid는 기존에 수정·고정한 CTC/attention 결합 검색이며 이번 단계에서 튜닝하지 않았다.", "",
              "**다음 개발 실험의 자동 처리 후보는 `stable_context_square`다.** 정사각형·주변 1.5배·중심과 크기의 인과적 EMA(alpha 0.4)를 적용한다. "
              "EMA 추가 전후 총 문자 오류는 234→224개로 줄었고 C025는 218→214개, C294는 16→10개로 모두 감소했다. "
              "따라서 추론 전에 정한 ‘전체 오류 엄격 감소, 화자별 악화 없음, 실패 증가 없음’ 기준을 만족했다. "
              "기존 자동 직사각형보다 20문장 모두 개선됐지만, EMA만의 효과는 8문장 개선·5문장 동일·7문장 악화로 작고 고르지 않다. "
              "더 많은 독립 자료에서도 EMA가 우수하다고 확정하지 않는다. 선택 기록은 `results/step04_decision.json`이며 서비스 기본값은 바꾸지 않았다.", "",
              "| 입력 방식 | C025 CER(10문장·402자) | C294 CER(10문장·265자) |",
              "|---|---:|---:|"]
    for arm, label in LABELS.items():
        h = hybrid[arm]
        lines.append(f"| {label} | {h['per_speaker']['C025']['cer_percent']:.2f}% | {h['per_speaker']['C294']['cer_percent']:.2f}% |")
    lines += ["", "C025는 시간 교정 후에도 선택 후보의 CER가 53.23%이고 10문장 완전 일치가 0개다. "
              "시간 불일치와 자동 영역 차이는 확인된 원인이지만, 남은 오류를 모두 설명하지는 않는다. "
              "발화 내용·입술/음성 동기·화자 및 촬영 조건·학습 분포 차이를 다음 점검에서 구분해야 한다.", "",
              "### 시간 교정 효과와 영역 처리 효과의 구분", "",
              "제공 좌표·같은 모델/디코더의 3단계 기록은 345/667(CER 51.72%)였고 시간 교정 후에는 218/667(32.68%)이다. "
              "C025는 330→202개 오류, C294는 15→16개 오류였다. 이것은 평가 입력과 정답의 시간 연결을 바로잡은 효과이며 새 모델 학습의 성과가 아니다. "
              "영역 처리 효과는 이번에 교정한 동일 프레임에서 직사각형 501개 → 선택 후보 224개 오류로 비교한다. "
              "3단계의 잘못된 구간과 이번 후보를 직접 비교해 전부 ROI 개선 효과라고 보고하지 않는다.", "",
              "3단계의 산술 기록은 보존하지만 기존 모델과 공식 후보의 일반화 성능 차이를 확정하는 근거로 사용할 수 없다. "
              "기존 17단계의 개발 평가도 정확히 같은 val20을 썼다. 다만 현재 v4 학습 집합에는 C025/C294가 없고, "
              "17단계는 검증 점수 기반 조기 종료·체크포인트 선택을 사용하지 않았으므로 이번 발견만으로 기존 가중치가 이 오류 때문에 학습됐거나 선택됐다고 판단하지 않는다. "
              "실제 17단계 학습은 8화자·59녹화에서 뽑은 256/512문장이었으며, 해당 학습 녹화의 WAV/MP4 정렬은 아직 검사하지 않았다. "
              "전체 v4 학습 자료는 1,000문장·96녹화다. 관련 범위 감사는 `results/step04_legacy_impact_scope.json`에 기록했다.", "",
              "[20문장·200개 전체 출력](04_문장별_출력.md)에서 성공과 실패 사례를 모두 확인할 수 있다.", "",
              "## 진행 중 발견한 정답 연결 문제" + diagnosis,
              "## 기록 파일", "",
              "아래 파일은 프로젝트 `experiments/olkavs_official` 기준이다.", "",
              "| 목적 | 파일 |", "|---|---|",
              "| 실행 전 보류된 최초 계획 | `step04_eval_plan.json` |",
              "| 내용·메타데이터 진단 | `alignment_audit_step04.py`, `inspect_timeline_step04.py`, `results/step04_C025_alignment.json`, `results/step04_timeline_metadata.json` |",
              "| 사용자 청취 확인 | `prepare_step04_listening.py`, `results/step04_human_listening.json` |",
              "| 파형 근거와 독립 검산 | `align_step04_audio.py`, `results/step04_audio_alignment.json`, `results/step04_audio_alignment_independent_audit.json` |",
              "| 시간 매핑 | `build_step04_audio_offsets.py`, `results/step04_audio_offset_candidates.json` |",
              "| 고정한 실험 계획 | `step04_aligned_eval_plan.json` |",
              "| 영역 처리·입력 준비 | `roi_geometry_step04.py`, `prepare_step04_aligned_inputs.py` |",
              "| 입력·QA | `inputs_step04_aligned/` 아래 `manifest.json`, `input_qa.json`, `input_summary.json`, `contact_sheet.png` |",
              "| 추론·전체 결과 | `evaluate_step04.py`, `results/step04.json`, `results/step04_features/` |",
              "| 독립 결과 검산 | `audit_step04_results.py`, `results/step04_metrics_audit.json` |",
              "| 후보 선택·보고 | `results/step04_decision.json`, `summarize_step04.py`, `write_step04_sentence_report.py` |", "",
              "## 검증·재현과 시간 비용", "",
              "입력 100개 모두 전체 디코딩·파일 해시·정답 ID·시간 경계를 검사했다. 각 조건은 2,574프레임이고 자동 검출은 2,574/2,574 성공했다. "
              "이전 검출 좌표 재사용과 제공 좌표 fallback은 없었다. 5조건의 원본 프레임 해시는 문장마다 같았다. "
              "정사각형/EMA 7,722프레임의 기하 계산을 독립 재계산했으며, 대표 프레임 100장의 입술 표시도 확인했다.", "",
              "독립 검산은 200개 출력의 문자·어절 편집거리, 화자별/전체 집계, 토큰→문자열, "
              "100개 특징 파일의 유한값·형상·해시와 CTC argmax/중복 제거를 확인했다. 모델을 다시 실행해 유리한 결과를 고르지 않았다. "
              "Hybrid 최대 탐색 깊이는 33~139였고 길이 제한 도달은 0개다. geometry 12개·기존 준비 8개·보정 준비 7개로 고유 단위 테스트 27개가 통과했다. "
              "보정 준비 실행 로그의 15개는 기존 준비 8개를 포함하므로 중복 집계하지 않는다.", "",
              "100개 crop 영상 준비는 약 52.22초였으며 영상 파일 합계는 약 3.11MB다. 이 안의 원본 디코딩 측정 합계는 10.23초, 검출은 15.10초였다. "
              "선택 후보의 GPU 준비 입력→특징 추출+hybrid 시간은 문장당 중앙값 2.73초, 평균 3.34초, 범위 0.78~9.19초였다. "
              "CTC 직접 출력은 중앙값 0.111초였지만 CER 41.38%이고 문장 일치는 0/20이다. "
              "측정에는 촬영·발화 종료 대기·crop·전송이 포함되지 않고 첫 호출 예열도 없으므로 실시간 웹캠 지연으로 해석하지 않는다.", "",
              "ROI는 미래 프레임을 쓰지 않지만, 현재 후보 모델 실험은 정답 발화 경계와 발화 전체 정규화를 사용한다. "
              "따라서 이 모듈의 인과성만으로 전체 시스템이 실시간 스트리밍이라고 주장할 수 없다.", "",
              "저장 결과의 재검산과 보고서 재생성 명령(PowerShell, 프로젝트 루트). 감사기는 덮어쓰기를 거부하므로 새 이름을 사용한다. "
              "재검산 파일은 별도로 남고 보고서는 기존 확정 감사 파일을 계속 참조한다:", "", "```powershell",
              "$step04AuditPath = 'experiments/olkavs_official/results/step04_metrics_recheck_' + (Get-Date -Format 'yyyyMMdd_HHmmss') + '.json'",
              ".\\.venv-olkavs\\Scripts\\python.exe experiments/olkavs_official/audit_step04_results.py --output $step04AuditPath",
              ".\\.venv-olkavs\\Scripts\\python.exe experiments/olkavs_official/summarize_step04.py",
              ".\\.venv-olkavs\\Scripts\\python.exe experiments/olkavs_official/write_step04_sentence_report.py", "```", "",
              "새 추론은 기존 결과를 덮어쓰지 않도록 평가 실행기가 거부한다. 재실험은 별도 실행 계획·출력 경로를 준비해야 한다.", "",
              "| 고정 산출물 | SHA256 |", "|---|---|"]
    for label, path in [("보정 실험 계획", ROOT / "step04_aligned_eval_plan.json"),
                        ("입력 manifest", ROOT / "inputs_step04_aligned/manifest.json"),
                        ("추론 결과", result_path), ("독립 결과 감사", audit_path),
                        ("시간 매핑", ROOT / "results/step04_audio_offset_candidates.json")]:
        lines.append(f"| {label} | `{sha(path)}` |")
    lines += ["", "## 다음에 진행할 작업", "",
              "1. **기존 모델을 보정된 동일 개발 문장에서 다시 평가.** 기존 모델의 69토큰/RGB/25fps 계약과 공식 후보의 72토큰/BGR/native-frame 계약을 각각 지키면서 "
              "정답과 실제 발화 구간을 일치시킨다. 이번 결과만으로 이전 모델이 가망 없다고 결론내리지 않는다. 필요 시 시간축을 확인한 소량 자료로 재학습 여부를 판단한다.",
              "2. **기존 학습 자료의 시간축 감사.** 확인한 네 녹화는 별도 매핑으로 관리하고, 다른 녹화는 각자의 원본 WAV/MP4 파형을 확인한다. "
              "−8.362초를 전체 파일에 적용하지 않는다. 실제 학습에 사용한 59녹화에서 소수부터 점검하고, 필요 시 전체 96녹화로 넓힌다. "
              "시간 불일치가 확인된 경우 검증된 매핑으로 별도 캐시를 만들고 제한된 재학습을 비교한다. 기존 v4와 가중치는 과거 재현용으로 남긴다.",
              "3. **자동 입력 후보의 독립 검증.** 여기서 고른 주변 영역+EMA 설정을 고정한 뒤 새 화자·실제 웹캠에서 발화 검출, 촬영 조건, 인식과 지연을 확인한다. "
              "반복 사용한 val20은 개발 자료이며 최종 test는 설정 확정 전까지 보존한다. 공개 사전학습 모델과 평가 원본의 중복 여부도 미확인이다.", "",
              "현재 단계에 새 대용량 학습 데이터는 필요하지 않았다. 후속 점검에서 원본 WAV가 없거나 독립 화자 자료가 부족한 경우 "
              "필요한 파일·이유·최소 용량을 먼저 특정해서 알린다. 이번 정상 파일의 시간 연결 문제는 TS2·TS20·TS200의 잘린 MP4 복구 문제와 별개다.", "",
              "졸업 프로젝트에서는 공개 모델 출처와 사용 범위를 밝히고, 직접 구현한 시간축 검증·자동 입술 처리·비교 실험·웹캠 시스템을 "
              "개발 기여로 구분해 제시할 수 있다. 정사각형/EMA를 새로운 학습 모델로 표현하지 않으며 학과의 인정 여부를 확정하지 않는다.", ""]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"report": str(report_path), "decision": str(decision_path), "selected": selected,
                      "checks": checks}, ensure_ascii=False))


if __name__ == "__main__":
    main()
