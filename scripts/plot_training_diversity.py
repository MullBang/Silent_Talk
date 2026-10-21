"""Render verified stage16 development curves without opening clips or models."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ["MPLCONFIGDIR"] = str(ROOT / "tmp/matplotlib-step16")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D

BRANCHES = ("A_keep64", "B_expand256")
COHORTS = ("original64", "additional192", "train60", "val20")
COUNTS = {"original64": 64, "additional192": 192, "train60": 60, "val20": 20}
STEPS = (0, 800, 1600, 3200)
CONDITIONS = ("normal", "static_middle", "shuffle_42")
COLORS = {"A_keep64": "#2675a5", "B_expand256": "#cc651d"}
LABELS = {"A_keep64": "A · 학습 64개 유지", "B_expand256": "B · 학습 256개 확대"}


def require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def read_json(path: Path) -> dict:
    def unique(items):
        result = {}
        for key, value in items:
            require(key not in result, f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique)


def group_values(group: dict, expected: int) -> dict:
    require(type(group["count"]) is int and group["count"] == expected, "Unexpected cohort size")
    greedy = group["greedy"]
    require(greedy["count"] == expected, "Greedy denominator mismatch")
    cer = greedy["mean_sentence_jamo_cer"]
    require(type(cer) in (int, float) and math.isfinite(cer) and cer >= 0, "Invalid CER")
    exact = greedy["exact_count"]
    require(type(exact) is int and 0 <= exact <= expected, "Invalid exact count")
    return {"count": expected, "cer": float(cer), "exact": exact}


def prepare_plot_data(summary: dict) -> dict:
    require(summary.get("schema") == "lipnet-training-diversity-summary-v1"
            and summary.get("status") == "completed_and_verified", "Verified stage16 summary required")
    require(set(summary["branches"]) == set(BRANCHES), "Unexpected branches")
    output = {"branches": {}}
    for branch in BRANCHES:
        snapshots = summary["branches"][branch]["snapshots"]
        require(set(snapshots) == {f"{step:04d}" for step in STEPS}, "Unexpected evaluation steps")
        values = {"normal": {}, "controls": {}}
        for cohort in COHORTS:
            values["normal"][cohort] = [group_values(
                snapshots[f"{step:04d}"]["cohorts"][cohort]["normal"], COUNTS[cohort]) for step in STEPS]
        for cohort in ("train60", "val20"):
            groups = snapshots["3200"]["cohorts"][cohort]
            require(set(groups) == set(CONDITIONS), "Endpoint controls required")
            values["controls"][cohort] = {condition: group_values(groups[condition], COUNTS[cohort])
                                          for condition in CONDITIONS}
        output["branches"][branch] = values
    for cohort in COHORTS:
        require(output["branches"][BRANCHES[0]]["normal"][cohort][0]
                == output["branches"][BRANCHES[1]]["normal"][cohort][0], "Different starting model results")
    return output


def render_figure(data: dict, font_path: Path):
    require(font_path.is_file(), f"Korean font missing: {font_path}")
    font_manager.fontManager.addfont(str(font_path))
    font_name = font_manager.FontProperties(fname=str(font_path)).get_name()
    style = {"font.family": font_name, "font.size": 11, "axes.titlesize": 13,
             "axes.labelsize": 10.5, "axes.spines.top": False, "axes.spines.right": False,
             "axes.unicode_minus": False, "legend.fontsize": 10}
    with plt.rc_context(style):
        fig, axes = plt.subplots(2, 3, figsize=(19.2, 12.0))
        normal_axes = [axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0]]
        titles = {
            "original64": "A. 기존 학습 문장 64개",
            "additional192": "B. 추가 문장 192개 · 확대군만 학습",
            "train60": "C. 고정 미학습 영상 60개 · 익숙한 화자",
            "val20": "D. 개발 검증 영상 20개 · 다른 화자",
        }
        max_cer = max(
            [v["cer"] * 100 for branch in data["branches"].values()
             for series in branch["normal"].values() for v in series]
            + [v["cer"] * 100 for branch in data["branches"].values()
               for conditions in branch["controls"].values() for v in conditions.values()])
        # Reserve headroom for both endpoint legends above the highest curve.
        maximum = max(110., max_cer * 1.32)
        for ax, cohort in zip(normal_axes, COHORTS):
            for branch in BRANCHES:
                rows = data["branches"][branch]["normal"][cohort]
                end = rows[-1]
                label = f"{branch[0]} 종료 {100*end['cer']:.2f}% · 완전 일치 {end['exact']}/{end['count']}"
                ax.plot(STEPS, [100*row["cer"] for row in rows], color=COLORS[branch],
                        marker="o", markersize=5, linewidth=2, label=label)
            ax.set_title(titles[cohort], loc="left", fontweight="bold", pad=14)
            ax.set_xticks(STEPS, ["0", "800", "1,600", "3,200"])
            ax.set_xlabel("이번 단계의 추가 갱신 시도 수")
            ax.set_xlim(-90, 3300)
            ax.legend(loc="upper left", frameon=False, handlelength=1.6)
        for ax, cohort, title in ((axes[1, 1], "train60", "E. 종료 모델 입력 대조 · 미학습 60개"),
                                  (axes[1, 2], "val20", "F. 종료 모델 입력 대조 · 개발 검증 20개")):
            for branch in BRANCHES:
                rows = data["branches"][branch]["controls"][cohort]
                yy = [100*rows[condition]["cer"] for condition in CONDITIONS]
                label = f"{branch[0]}: " + " / ".join(f"{value:.2f}" for value in yy) + "%"
                ax.plot(range(3), yy, color=COLORS[branch], marker="o", markersize=5,
                        linewidth=2, label=label)
            ax.set_title(title, loc="left", fontweight="bold", pad=14)
            ax.set_xticks(range(3), ["정상", "중간 프레임 반복", "순서 섞기 42"])
            ax.legend(loc="upper left", frameon=False, handlelength=1.6, fontsize=9.5)
        for ax in axes.flat:
            ax.set_ylim(0, maximum)
            ax.set_ylabel("문장 평균 자모 CER (%)")
            ax.grid(axis="y", alpha=.18)
            ax.set_axisbelow(True)
        fig.suptitle("16단계 | 학습 문장 확대가 새 발화로 이어지는가", fontsize=22, fontweight="bold", y=.98)
        fig.text(.5, .939, "14단계 B의 9,600회 전체 학습 상태에서 재개 · 학습률 0.0001 · 군별 추가 3,200회 시도",
                 ha="center", fontsize=12, color="#40505a")
        fig.legend(handles=[Line2D([0], [0], color=COLORS[branch], marker="o", linewidth=2,
                                  label=LABELS[branch]) for branch in BRANCHES],
                   loc="upper center", bbox_to_anchor=(.5, .910), ncol=2, frameon=False, fontsize=12)
        footer = (
            "모든 패널은 greedy 문장 평균 자모 오류율(CER)입니다. 낮을수록 좋으며, 100-CER를 인식 정확도로 해석하지 않습니다.\n"
            "두 군은 같은 가중치·Adam·AMP scaler·RNG에서 시작했습니다. 추가 표본 제시는 각 6,400회, 문장당 반복은 A 100회 / B 25회입니다.\n"
            "추가 192개는 B에서만 학습합니다. 길이·화자 구성·처리 프레임·반복 빈도도 달라지므로 순수 데이터 개수의 인과 효과는 아닙니다.\n"
            "80개는 반복 사용하는 고정 개발 평가 자료입니다. 단일 시작점 비교이며, test50·실제 웹캠 평가·배포는 수행하지 않았습니다."
        )
        fig.text(.044, .026, footer, va="bottom", fontsize=10, linespacing=1.8, color="#465159")
        fig.subplots_adjust(left=.062, right=.978, bottom=.237, top=.824, wspace=.32, hspace=.63)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--font", type=Path, default=Path("C:/Windows/Fonts/malgun.ttf"))
    args = parser.parse_args()
    require(args.out.suffix.lower() == ".png" and not args.out.exists(), "New PNG output path required")
    data = prepare_plot_data(read_json(args.summary))
    figure = render_figure(data, args.font)
    try:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("xb") as stream:
            figure.savefig(stream, format="png", dpi=170, facecolor="white", metadata={
                "Title": "Stage16 training diversity diagnostic",
                "Description": "Fixed development evaluation; sentence-mean greedy jamo CER"})
    finally:
        plt.close(figure)
    print(args.out.resolve())


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
