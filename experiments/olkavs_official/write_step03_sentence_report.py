"""Render all frozen stage3 predictions without choosing or changing results."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]


def main() -> None:
    result = json.loads((ROOT / "results" / "step03.json").read_text(encoding="utf-8"))
    if result["status"] != "completed" or len(result["records"]) != 40:
        raise ValueError("Complete paired20 result required")
    lookup = {(r["id"], r["arm"]): r for r in result["records"]}
    if len(lookup) != 40:
        raise ValueError("Duplicate result")
    labels = {"annotation_bbox": "제공 좌표", "automatic_mediapipe": "자동 추출"}
    methods = {"corrected_hybrid": "수정 hybrid", "ctc_greedy": "CTC 직접 출력"}
    lines = ["# 새 계획 3단계 — 20문장 전체 출력", "",
             "2026-09-24 실행 결과. 정답·예측은 저장된 원문 그대로이며, 예측의 오타나 문법을 고치지 않았다. "
             "CER는 내부 공백을 포함한 문자 편집거리/정답 문자 수다. 실패·불리한 사례를 제외하지 않았다.", "",
             "[전체 분석](03_실행결과.md)", ""]
    for i, sid in enumerate(result["plan"]["source_ids"], 1):
        ref = lookup[(sid, "annotation_bbox")]["reference"]
        lines += [f"## {i:02d}. {sid}", "", f"정답: {ref}", "",
                  "| 입력·출력 방식 | 모델 출력 | 문자 오류 | CER |", "|---|---|---:|---:|"]
        for arm in result["plan"]["arms"]:
            for method in result["plan"]["methods"]:
                out = lookup[(sid, arm)]["outputs"][method]
                met = out["metrics"]
                edits, total = met["character_edits"], met["reference_characters"]
                pred = out["prediction"].replace("|", "\\|").replace("\n", "<br>")
                lines.append(f"| {labels[arm]} / {methods[method]} | {pred} | {edits}/{total} | {100*edits/total:.2f}% |")
        lines.append("")
    target = PROJECT / "docs" / "공식모델_도입검증" / "03_문장별_출력.md"
    target.write_text("\n".join(lines), encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()
