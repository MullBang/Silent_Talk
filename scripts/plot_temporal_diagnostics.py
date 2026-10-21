"""완료된11단계 진단의 검증 입력 대조를 그린다. 모델 추론 없음."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "tmp/matplotlib-step11"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


def main() -> None:
    """동일20문장의 CER와 정상 대비 문자열 일치율을 구분해 표시한다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    run, out = Path(args.run), Path(args.out)
    if out.exists():
        raise FileExistsError(out)
    complete = json.loads((run / "completed.json").read_text(encoding="utf-8"))
    if complete["status"] != "completed" or complete["test_inference_performed"]:
        raise ValueError("train/val 고정 진단 완료 기록이 필요합니다.")
    font = Path("C:/Windows/Fonts/malgun.ttf")
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({"axes.unicode_minus": False, "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.7), gridspec_kw={"width_ratios": [1.2, 1]})
    names = ["normal", "static_middle", "shuffle", "reversed", "constant_gray"]
    labels = ["정상", "중간 프레임\n반복", "순서 섞기\n3회 평균", "역순", "회색"]
    all_rows = {}
    for key in ("compact", "residual"):
        path = run / key / "rows.jsonl"
        if hashlib.sha256(path.read_bytes()).hexdigest() != complete["models"][key]["rows_sha256"]:
            raise ValueError(f"결과 변경: {key}")
        all_rows[key] = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    for index, (key, label, color) in enumerate((("compact", "경량", "#2463a6"), ("residual", "잔차", "#a05195"))):
        selected = [row for row in all_rows[key] if row["split"] == "val"]
        cers, c_lo, c_hi, agreements, a_lo, a_hi = [], [], [], [], [], []
        for name in names:
            conditions = [f"shuffle_{seed}" for seed in (42, 137, 2026)] if name == "shuffle" else [name]
            scores, same = [], []
            for condition in conditions:
                rows = [r for r in selected if r["condition"] == condition]
                if len(rows) != 20:
                    raise ValueError("고정 검증20문장이 필요합니다.")
                scores.append(100 * np.mean([r["cer"] for r in rows]))
                if name != "normal":
                    same.append(100 * np.mean([r["comparison_to_normal"]["text_equal"] for r in rows]))
            cers.append(np.mean(scores))
            c_lo.append(np.mean(scores) - min(scores))
            c_hi.append(max(scores) - np.mean(scores))
            if same:
                agreements.append(np.mean(same))
                a_lo.append(np.mean(same) - min(same))
                a_hi.append(max(same) - np.mean(same))
        offset = (index - .5) * .36
        bars = axes[0].bar(np.arange(5) + offset, cers, .34, label=label, color=color,
                           yerr=[c_lo, c_hi], capsize=3)
        axes[0].bar_label(bars, labels=[f"{v:.1f}" for v in cers], padding=4, fontsize=9)
        bars = axes[1].bar(np.arange(4) + offset, agreements, .34, label=label, color=color,
                           yerr=[a_lo, a_hi], capsize=3)
        axes[1].bar_label(bars, labels=[f"{v:.0f}" for v in agreements], padding=4, fontsize=9)
    axes[0].set(title="같은 검증20문장의 자모 오류율", ylabel="문장 평균 CER (%) · 낮을수록 좋음",
                xticks=np.arange(5), xticklabels=labels, ylim=(0, 108))
    axes[1].set(title="입력을 바꿔도 정상과 같은 예측 문자열", ylabel="정상 대비 문자열 일치율 (%)",
                xticks=np.arange(4), xticklabels=labels[1:], ylim=(0, 110))
    for ax in axes:
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
        ax.legend(loc="upper right", frameon=False)
    fig.suptitle("11단계 · 움직임 제거와 순서 변경에 대한 모델 반응", fontsize=17, weight="bold", y=.98)
    fig.text(.04, .045, "고정된 best 가중치 · beam10 · 새 학습 없음 · 최종 test 제외\n"
             "오차막대는 순서 섞기 3개 seed의 최솟값~최댓값이며 신뢰구간이 아닙니다. 문자열 일치는 정답률이 아닙니다.",
             fontsize=10, color="#444444")
    fig.tight_layout(rect=(0, .13, 1, .92))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, facecolor="white")
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
