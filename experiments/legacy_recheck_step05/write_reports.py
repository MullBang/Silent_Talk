"""Render audited legacy reevaluation, retaining every prediction unchanged."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import statistics

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
DOCS = ROOT / "docs/공식모델_도입검증"
BRANCH = {"A_keep256": "A: 256문장 학습", "B_expand512": "B: 512문장 학습"}
ARM = {"old_timing": "기존 시각", "aligned_timing": "보정 시각"}
METHOD = {"greedy": "greedy", "beam10": "beam10"}


def read(path: Path) -> dict:
    """Load a frozen local result."""
    return json.loads(path.read_text(encoding="utf8"))


def sha(path: Path) -> str:
    """Digest an evidence file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """Create the full output table and evidence-backed interpretation."""
    result_path = BASE / "results/result.json"
    audit_path = BASE / "results/independent_audit.json"
    result, audit = read(result_path), read(audit_path)
    if result["status"] != "completed" or audit["status"] != "passed" or audit["result_sha256"] != sha(result_path):
        raise ValueError("Completed result and matching independent audit required")
    if len(result["records"]) != 80 or not result["historical_replay"]["passed"]:
        raise ValueError("Complete two-model, two-timing,20-target experiment required")
    plan, aggregate = result["plan"], result["aggregate"]
    source = read(Path(plan["files"]["source_plan"]["path"]))
    lookup = {(r["branch"], r["arm"], r["id"]): r for r in result["records"]}
    pairs = {}
    for branch in BRANCH:
        for method in METHOD:
            delta = [lookup[branch, "aligned_timing", sid]["outputs"][method]["metrics"]["character_edits"] -
                     lookup[branch, "old_timing", sid]["outputs"][method]["metrics"]["character_edits"] for sid in plan["source_ids"]]
            pairs[f"{branch}/{method}"] = {"improved": sum(v < 0 for v in delta), "tied": sum(v == 0 for v in delta),
                                           "worsened": sum(v > 0 for v in delta), "edit_delta": sum(delta)}
    report = ["# 새 계획 5단계 — 기존 모델의 시간 보정 입력 재평가", "",
              "2026-09-24 실행 완료. **기존 평가의 시간 연결을 고쳐도 이전 두 모델의 인식 성능은 회복되지 않았다.** "
              "256문장 학습 A와 512문장 학습 B 모두 보정된 20문장의 greedy 문자 오류율이 **84.56%**, 완전 일치는 **0/20**이었다. "
              "기존 입력과 출력의 정확한 재현을 먼저 확인했으므로, 이번 비교에서 실행 환경이나 다른 전처리 변경을 시간 보정 효과로 섞지 않았다.", "",
              "이 결과는 **현재 고정 가중치의 성능이 부족하다**는 근거다. 같은 모델 구조를 검증된 자료로 다시 학습해도 가능성이 없다는 증거는 아니다. "
              "학습에 쓰인 원본들의 WAV/MP4 시간 정렬은 아직 미검증이며, 다음은 그 자료를 점검할 차례다.", "",
              "## 비교 결과", "",
              "동일한 20개 ID·정답, 두 화자(C025/C294 각각10문장)를 사용했다. "
              "주 지표는 총 NFC 완성형 문자 편집거리/정답667자이며 내부 공백도 센다. 낮을수록 좋고 ‘1−오류율’을 인식 정확도로 부르지 않는다. "
              "실패나 불리한 문장을 제외하지 않았다.", "",
              "| 고정 모델 | 입력 시각 | greedy 문자 오류 | greedy CER | beam10 CER | 문장 일치(greedy/beam) |",
              "|---|---|---:|---:|---:|---:|"]
    for branch, label in BRANCH.items():
        for arm, timing in ARM.items():
            g, b = (aggregate[f"{branch}/{arm}/{m}"] for m in METHOD)
            report.append(f"| {label} | {timing} | {g['character_edits']}/667 | {g['cer_percent']:.2f}% | {b['cer_percent']:.2f}% | "
                          f"{g['exact_sentences']}/20 · {b['exact_sentences']}/20 |")
    report += ["", "같은 문장별 greedy 문자 오류는 A에서 4개 개선·10개 동일·6개 악화(총1자 오류 증가), "
               "B에서 5개 개선·4개 동일·11개 악화(총16자 오류 증가)였다. 정답 구간을 잘못 연결했던 과거 점수가 우연히 더 낮을 수 있다. "
               "그 점수를 되찾으려고 잘못된 시각으로 돌아가거나 정답을 모델 출력에 맞춰 바꾸지 않는다.", "",
               "| 모델·디코더 | C025 CER: 기존 → 보정 | C294 CER: 기존 → 보정 | WER: 기존 → 보정 |",
               "|---|---:|---:|---:|"]
    for branch, label in BRANCH.items():
        for method in METHOD:
            old, new = (aggregate[f"{branch}/{a}/{method}"] for a in ARM)
            speakers = [f"{old['per_speaker'][s]['cer_percent']:.2f}% → {new['per_speaker'][s]['cer_percent']:.2f}%" for s in ["C025", "C294"]]
            report.append(f"| {label} / {method} | {speakers[0]} | {speakers[1]} | {old['wer_percent']:.2f}% → {new['wer_percent']:.2f}% |")
    report += ["", "WER는 총 어절 편집거리/189어절이며 삽입 오류가 많으면 100%를 넘을 수 있다. "
               "한 화자의 변화만으로 전체 개선을 주장하지 않는다.", "",
               "### 이전 17단계 지표와 구분", "",
               "17단계는 공백/문장부호 정규화 후 한글을 초성·중성·종성으로 분해한 **문장 평균 자모 CER**를 주로 보고했다. "
               "이번 공식 후보 비교의 **전체 완성형 문자 CER**와 단위·정규화·집계가 다르므로 같은 숫자처럼 비교할 수 없다.", "",
               "| 모델·디코더 | 문장 평균 자모 CER: 기존 → 보정 | 전체 자모 CER: 기존 → 보정 |",
               "|---|---:|---:|"]
    for branch, label in BRANCH.items():
        for method in METHOD:
            old, new = (aggregate[f"{branch}/{a}/{method}"] for a in ARM)
            report.append(f"| {label} / {method} | {old['mean_jamo_cer_percent']:.2f}% → {new['mean_jamo_cer_percent']:.2f}% | "
                          f"{old['pooled_jamo_cer_percent']:.2f}% → {new['pooled_jamo_cer_percent']:.2f}% |")
    report += ["", "B의 greedy 자모 평균은 78.03%→76.55%로 조금 낮아졌지만 완성형 문자 CER는 82.16%→84.56%로 높아졌다. "
               "일부 자모의 일치와 사용 가능한 한국어 문장 복원은 다르다. 어느 집계에서도 완전 일치 문장은 0개이며, 유리한 지표만 골라 성능이 회복됐다고 해석하지 않는다.", "",
               "## 시간 외의 조건을 유지한 방법", "",
               "비교 대상은 기존 17단계의 사전 지정 종료점 `A_keep256/step_6400.pt`, `B_expand512/step_6400.pt` 두 개다. "
               "두 모델 모두 누적19,200회 시도의 종료 가중치이며 중간 점수를 보고 더 좋은 가중치를 고르지 않았다. "
               "구조는 3,287,781개 변수의 compact 3D CNN → 프레임별 GroupNorm → 투영층 → 2층 양방향 LSTM →69토큰 CTC다.", "",
               "입력은 기존 준비 함수 `_roi_seq_from_boxes`를 그대로 호출했다. 프레임 수는 `round((끝−시작)*25)`, "
               "선택 원본 프레임은 `round((시작+k/25)*실제FPS)`다. 영상 프레임 번호에 대응하는 제공 입술 좌표에 "
               "상하좌우 각각20% 여백을 더하고 RGB96×96으로 보간한다. NPZ의 uint8 원본을 /255한 뒤 ImageNet 평균·표준편차로 정규화한다. "
               "모델은 FP32·패딩 없는 batch1·eval 모드·시간 길이 유지이며, 외부 언어모델 없이 greedy와 기존 prefix beam10을 사용했다.", "",
               "보정 입력은 기존 라벨 시작과 끝에 4단계에서 파형으로 확정한 이동량만 더했다. "
               "C025 A006/A007은 −8.362초, C294 A001/A003은 +0.046375초다. 원본 라벨과 문장은 고치지 않았다. "
               "조건별2,144프레임·문장당35~178프레임으로 기존과 길이가 같고 CTC 최소길이 조건도 유지했다. "
               "공식 후보의 BGR·native-frame·margin0/자동영역·MP4 입력은 기존 모델의 입력으로 재사용하지 않았다.", "",
               "### 재현 검증", "",
               "1. 기존 시각20개를 원본 영상에서 다시 추출해 원래 v4 캐시의 모든 uint8 픽셀·정답·토큰과 정확히 일치함을 확인했다.",
               "2. 두 모델로 원래 입력을 다시 추론해 **40개 CTC 로그 확률 배열과80개 greedy/beam 문자열**이 저장된17단계 결과와 정확히 같음을 확인했다. "
               "모든 원래 결과가 재현되기 전에는 보정 입력 추론을 시작하지 않도록 실행기를 구성했다.",
               "3. 보정 입력까지 총80회 추론·160개 출력에서 실행 실패가 없었다. "
               "독립 검산은40입력·4,288개 프레임 선택 기록,80확률 배열의 형상/유한값/확률합,160출력의 NFC/NFD 편집거리 및 전체/화자 집계를 확인했다. "
               "greedy는 별도 축약·유니코드 조합 구현으로 확인했고 beam10은 고정한 기존 디코더로 재실행했다. beam 알고리즘 자체를 독립 구현한 것은 아니다.", "",
               "입력 준비의 필수 테스트9개가 통과했다. 짝지은40개 대표 프레임을 육안으로 확인했고 입술 영역과 RGB 색상이 정상이었다. "
               "대표 이미지 검수가 모든 프레임의 발화 전사 정확성까지 보증하지는 않는다.", "",
               "이번 준비는 약63.64초였고 새로 저장한20개 NPZ는36.09MB다. 원본 전체 사본이나 새 데이터셋 다운로드를 만들지 않았다. "
               "기존 모델·원본·캐시·서비스 코드는 변경하지 않았으며 학습과test50 추론도 실행하지 않았다.", "",
               "## 공식 후보와 비교할 수 있는 범위", "",
               "같은 시간 매핑·20개 정답을 사용한4단계 공식 후보는 제공 좌표 경로CER32.68%·5/20문장 일치, "
               "개선한 자동 영역 경로CER33.58%·5/20문장 일치였다. 이번 기존 A/B의 greedy는84.56%·0/20이다. "
               "**현재 가중치와 입력·디코딩을 합친 전체 처리 경로의 차이**를 보여준다. "
               "모델 크기·사전학습량·토큰·프레임 속도·색상·crop·디코더가 다르므로 이 차이를 특정 신경망 구조 하나의 우열로 분리할 수 없다.", "",
               "val20은 반복 사용한 두 화자의 개발 자료다. 공식 가중치의 사전학습 자료와 평가 원본의 중복 여부는 미확인이다. "
               "두 시스템 모두 처음 보는 사용자의 실제 웹캠 성능으로 일반화할 수 없다.", "",
               "[4단계 보고서](04_실행결과.md) · [이번 20문장·160개 전체 출력](05_문장별_출력.md)", "",
               "## 남은 문제와 다음 단계", "",
               "현재 고정 가중치는 발화 구간을 바로잡아도 한국어 문장 복원이 부족하다. "
               "예를 들어 B의 보정 입력에서 정답 ‘아주 기쁘고 감사해’는 ‘하수 우 낡아 결혼해’로 출력됐다. "
               "출력의 `�`는 저장 파일 인코딩 오류가 아니라 기존 토크나이저가 조합할 수 없는 예측 자모를 U+FFFD로 표시한 것이다. "
               "이를 삭제하거나 문법적으로 다듬지 않고 그대로 오류로 계산했다.", "",
               "1. **다음6단계는 학습 원본의 시간축 점검이다.** 실제 A/B 학습에 사용한59개 녹화를 대상으로 원본 WAV 보유 여부를 먼저 확인한다. "
               "8명의 화자별 소수 녹화부터 앞·중간·뒤의 파형을 점검해 범위를 파악하고, 새 학습에 쓸 녹화는 각각 매핑을 검증한다. "
               "검사량을 나누고 필요한 짧은 구간만 읽으며 Drive가 원본을 캐시할 수 있는 점도 고려한다.",
               "2. 시간 불일치가 확인되면 해당 원본의 검증된 이동량으로 **별도 캐시**를 만들고, 기존 구조의 작은 복원/학습 대조부터 진행한다. "
               "이상 여부를 확인하기 전에 C025의 −8.362초를 다른 화자에 적용하거나59개를 모두 잘못된 자료로 취급하지 않는다. "
               "현재 확인된 C025/C294는 A/B 학습 집합에 없었으므로 이번 발견만으로 학습 오염을 주장할 수 없다.",
               "3. 학습 정렬이 정상이라면 데이터 다양성·사전학습/특징 전달·토큰 출력·시간 정보 활용을 순서대로 비교한다. "
               "같은 자료와 설정으로 반복 학습만 늘리는 것을 기본 다음 행동으로 삼지 않는다. 공개 후보는 비교 기준으로 보존하고 기존 모델을 개선할 여지도 구분해서 검증한다.", "",
               "현재 단계에 새 학습 데이터는 필요하지 않았다. 다음 점검에서 원본 WAV가 빠져 있다면 필요한 파일명·용도·최소 용량을 먼저 특정해서 알린다. "
               "이번 실험은 모델 폐기나 공식 모델의 서비스 채택을 결정하지 않았다.", "",
               "## 산출물과 재현", "",
               "프로젝트 `experiments/legacy_recheck_step05/` 아래:", "",
               "| 파일 | 내용 |", "|---|---|",
               "| `preparation_plan.json`, `prepare_inputs.py` | 고정한 입력 준비 계약과 원래 함수 재사용 |",
               "| `test_preparation.py`, `verify_inputs.py` | 단위 테스트9개와 입력 전체 검증 |",
               "| `inputs/manifest.json`, `inputs/input_qa.json` | 원래/보정40입력·재현 검증·파일 해시 |",
               "| `inputs/contact_sheet.png`, `inputs/traces/` | 짝지은 대표 이미지·선택 프레임/PTS 기록 |",
               "| `eval_plan.json`, `evaluate.py` | 모델·실행 환경·지표·출력 재현 조건 고정 |",
               "| `results/result.json`, `results/*.npz` | 전체 결과80행·160문자열·CTC 로그 확률 |",
               "| `audit_results.py`, `results/independent_audit.json` | 독립 지표·출력 검산 |",
               "| `write_reports.py` | 저장한 결과를 문서로 정리 |", "",
               "평가 실행기는 이미 있는 계획/결과를 덮어쓰지 않는다. 새 실험은 별도 실행 경로와 계획이 필요하다. "
               "다음 명령은 확정된 결과와 감사 해시를 확인하고 두 보고서만 다시 만든다:", "", "```powershell",
               ".\\.venv-gpu\\Scripts\\python.exe experiments/legacy_recheck_step05/write_reports.py", "```", "",
               "| 증거 | SHA256 |", "|---|---|"]
    for label, path in [("입력 계획", BASE / "preparation_plan.json"), ("입력 명세", BASE / "inputs/manifest.json"),
                        ("평가 계획", BASE / "eval_plan.json"), ("전체 결과", result_path), ("독립 감사", audit_path)]:
        report.append(f"| {label} | `{sha(path)}` |")
    report += [""]
    (DOCS / "05_실행결과.md").write_text("\n".join(report), encoding="utf8")
    sentences = ["# 새 계획 5단계 — 기존 모델 20문장 전체 출력", "",
                 "2026-09-24. 두 고정 모델 × 원래/보정 시각 × greedy/beam10의160개 출력을 모두 싣는다. "
                 "원문·예측은 저장한 그대로이며 오타·중복 공백·U+FFFD를 고치거나 삭제하지 않았다. "
                 "`�`는 잘못 조합된 예측 자모의 표시다. CER는 총 NFC 문자 편집거리이며 내부 공백을 포함한다.", "",
                 "[전체 분석](05_실행결과.md)", ""]
    for i, source_row in enumerate(source["records"], 1):
        sid = source_row["id"]
        sentences += [f"## {i:02d}. {sid}", "", f"정답: {source_row['text']}", "",
                      "| 모델·시각·디코더 | 저장된 출력 | 문자 오류 | CER |", "|---|---|---:|---:|"]
        for branch, b_name in BRANCH.items():
            for arm, a_name in ARM.items():
                for method in METHOD:
                    out = lookup[branch, arm, sid]["outputs"][method]
                    prediction = out["prediction"].replace("|", "\\|").replace("\n", "<br>")
                    m = out["metrics"]
                    sentences.append(f"| {b_name} / {a_name} / {method} | {prediction} | "
                                     f"{m['character_edits']}/{m['reference_characters']} | {100*m['character_edits']/m['reference_characters']:.2f}% |")
        sentences.append("")
    (DOCS / "05_문장별_출력.md").write_text("\n".join(sentences), encoding="utf8")
    comparison = {"schema": "legacy-step05-interpretation-v1", "result_sha256": sha(result_path),
                  "audit_sha256": sha(audit_path), "report_script_sha256": sha(Path(__file__)),
                  "paired_character_errors": pairs, "current_checkpoints_recovered": False,
                  "architecture_has_no_future_proven": False, "training_misalignment_proven": False,
                  "next_step": "Per-recording training WAV/MP4 timeline audit before new training or architecture conclusions",
                  "new_data_needed_this_stage": False, "model_adoption_or_deployment": False}
    (BASE / "results/interpretation.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    print(json.dumps({"report": str(DOCS / "05_실행결과.md"), "sentences": str(DOCS / "05_문장별_출력.md")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
