"""Render verified stage17 development curves without opening clips or models."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ["MPLCONFIGDIR"] = str(ROOT / "tmp/matplotlib-step17")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D

BRANCHES = ("A_keep256", "B_expand512")
COHORTS = ("original64", "additional192", "additional256", "train60", "val20")
COUNTS = {"original64": 64, "additional192": 192, "additional256": 256, "train60": 60, "val20": 20}
STEPS = (0, 1600, 3200, 6400)
CONDITIONS = ("normal", "static_middle", "shuffle_42")
COLORS = {"A_keep256": "#2675a5", "B_expand512": "#cc651d"}
LABELS = {"A_keep256": "A · 학습 256개 유지", "B_expand512": "B · 학습 512개 확대"}


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
    require(summary.get("schema") == "lipnet-training-expansion-summary-v1"
            and summary.get("status") == "completed_and_verified", "Verified stage17 summary required")
    require(set(summary["branches"]) == set(BRANCHES), "Unexpected branches")
    output = {"branches": {}}
    distribution = summary["protocol"]["selection"]["distributions"]
    for name, count in (("existing256", 256), ("expanded512", 512)):
        require(distribution[name]["clips"] == count and len(distribution[name]["speaker"]) == 8
                and distribution[name]["recordings"] == 59, "Unexpected training speaker/recording scope")
    for branch in BRANCHES:
        snapshots = summary["branches"][branch]["snapshots"]
        require(set(snapshots) == {f"{step:04d}" for step in STEPS}, "Unexpected evaluation steps")
        values = {"normal": {}, "controls": {}}
        for cohort in COHORTS:
            values["normal"][cohort] = [group_values(
                snapshots[f"{step:04d}"]["cohorts"][cohort]["normal"], COUNTS[cohort]) for step in STEPS]
        for cohort in ("train60", "val20"):
            groups = snapshots["6400"]["cohorts"][cohort]
            require(set(groups) == set(CONDITIONS), "Endpoint controls required")
            values["controls"][cohort] = {condition: group_values(groups[condition], COUNTS[cohort])
                                          for condition in CONDITIONS}
        compute = summary["branches"][branch]["compute"]
        extension = summary["branches"][branch]["extension"]
        require(compute["attempted_clip_presentations"] == 12800, "Unexpected clip presentation budget")
        successful, skipped = extension["optimizer_steps"], extension["skipped_optimizer_steps"]
        require(type(successful) is int and type(skipped) is int and successful >= 0 and skipped >= 0
                and successful + skipped == 6400, "Unexpected optimizer attempt budget")
        raw, padded = compute["attempted_unpadded_frames"], compute["attempted_padded_frames"]
        require(type(raw) is int and type(padded) is int and 0 < raw <= padded, "Invalid frame workload")
        values["compute"] = {"successful": successful, "skipped": skipped, "raw_frames": raw,
                             "padded_frames": padded, "presentations": compute["attempted_clip_presentations"]}
        output["branches"][branch] = values
    for cohort in COHORTS:
        require(output["branches"][BRANCHES[0]]["normal"][cohort][0]
                == output["branches"][BRANCHES[1]]["normal"][cohort][0], "Different starting model results")
    return output


def render_figure(data: dict, font_path: Path):
    require(font_path.is_file(), f"Korean font missing: {font_path}")
    font_manager.fontManager.addfont(str(font_path))
    font_name = font_manager.FontProperties(fname=str(font_path)).get_name()
    style = {"font.family": font_name, "font.size": 10, "axes.titlesize": 12,
             "axes.labelsize": 10, "axes.spines.top": False, "axes.spines.right": False,
             "axes.unicode_minus": False, "legend.fontsize": 9}
    with plt.rc_context(style):
        fig, axes = plt.subplots(2, 4, figsize=(22.4, 11.8))
        flat = list(axes.flat)
        titles = {
            "original64": "A. 초기 학습 문장 64개",
            "additional192": "B. 16단계 추가 문장 192개",
            "additional256": "C. 17단계 추가 256개 · B만 학습",
            "train60": "D. 미학습 영상 60개 · 익숙한 화자",
            "val20": "E. 개발 검증 20개 · 다른 화자",
        }
        max_cer = max(
            [v["cer"] * 100 for branch in data["branches"].values()
             for series in branch["normal"].values() for v in series]
            + [v["cer"] * 100 for branch in data["branches"].values()
               for conditions in branch["controls"].values() for v in conditions.values()])
        maximum = max(110., max_cer * 1.34)
        for ax, cohort in zip(flat[:5], COHORTS):
            for branch in BRANCHES:
                rows = data["branches"][branch]["normal"][cohort]
                end = rows[-1]
                label = f"{branch[0]} 종료 {100*end['cer']:.2f}% · 일치 {end['exact']}/{end['count']}"
                ax.plot(STEPS, [100*row["cer"] for row in rows], color=COLORS[branch],
                        marker="o", markersize=4.5, linewidth=2, label=label)
            ax.set_title(titles[cohort], loc="left", fontweight="bold", pad=14)
            ax.set_xticks(STEPS, ["0", "1,600", "3,200", "6,400"])
            ax.set_xlabel("이번 단계의 추가 갱신 시도 수")
            ax.set_xlim(-180, 6600)
            ax.legend(loc="upper left", frameon=False, handlelength=1.4)
        for ax, cohort, title in ((flat[5], "train60", "F. 종료 모델 입력 대조 · 미학습 60개"),
                                  (flat[6], "val20", "G. 종료 모델 입력 대조 · 검증 20개")):
            for branch in BRANCHES:
                rows = data["branches"][branch]["controls"][cohort]
                yy = [100*rows[condition]["cer"] for condition in CONDITIONS]
                label = f"{branch[0]}: " + " / ".join(f"{value:.2f}" for value in yy) + "%"
                ax.plot(range(3), yy, color=COLORS[branch], marker="o", markersize=4.5,
                        linewidth=2, label=label)
            ax.set_title(title, loc="left", fontweight="bold", pad=14)
            ax.set_xticks(range(3), ["정상", "중간 프레임 반복", "순서 섞기 42"], fontsize=9)
            ax.legend(loc="upper left", frameon=False, handlelength=1.4, fontsize=8.8)
        for ax in flat[:7]:
            ax.set_ylim(0, maximum)
            ax.set_ylabel("문장 평균 자모 CER (%)")
            ax.grid(axis="y", alpha=.18)
            ax.set_axisbelow(True)
        ax = flat[7]
        ax.set_title("H. 같은 시도 예산 · 다른 반복과 처리량", loc="left", fontweight="bold", pad=14)
        ax.axis("off")
        a = data["branches"][BRANCHES[0]]["compute"]
        b = data["branches"][BRANCHES[1]]["compute"]
        cells = [
            ["학습 문장 수", "256", "512"],
            ["추가 갱신 시도", "6,400", "6,400"],
            ["실제 갱신 / 건너뜀", f"{a['successful']:,} / {a['skipped']}", f"{b['successful']:,} / {b['skipped']}"],
            ["문장당 추가 제시", "50회", "25회"],
            ["실제 입력 프레임", f"{a['raw_frames']:,}", f"{b['raw_frames']:,}"],
            ["패딩 포함 프레임", f"{a['padded_frames']:,}", f"{b['padded_frames']:,}"],
            ["학습 화자 / 녹화 수", "8 / 59", "8 / 59"],
        ]
        table = ax.table(cellText=cells, colLabels=["이번 단계", "A 유지", "B 확대"],
                         colWidths=[.46, .27, .27], cellLoc="center", bbox=[0., .10, 1., .90])
        table.auto_set_font_size(False)
        table.set_fontsize(9.2)
        for (r, c), cell in table.get_celld().items():
            cell.set_edgecolor("#d9e1e5")
            cell.set_linewidth(.7)
            cell.set_facecolor("#eaf0f3" if r == 0 else ("#f6f8f9" if r % 2 == 0 else "white"))
            if r == 0:
                cell.set_text_props(weight="bold")
        fig.suptitle("17단계 | 더 학습할 것인가, 학습 문장을 늘릴 것인가", fontsize=23, fontweight="bold", y=.98)
        fig.text(.5, .939, "16단계 B의 종료 상태에서 재개(누적 12,800회) · 학습률 0.0001 · 군별 추가 6,400회 시도",
                 ha="center", fontsize=12, color="#40505a")
        fig.legend(handles=[Line2D([0], [0], color=COLORS[branch], marker="o", linewidth=2,
                                  label=LABELS[branch]) for branch in BRANCHES],
                   loc="upper center", bbox_to_anchor=(.5, .910), ncol=2, frameon=False, fontsize=12)
        footer = (
            "A–G 패널은 greedy 문장 평균 자모 오류율(CER), '일치'는 문장 완전 일치 수입니다. 낮은 CER가 좋으며, 100-CER는 인식 정확도가 아닙니다.\n"
            "두 군은 같은 가중치·Adam·AMP scaler·RNG에서 시작했습니다. 추가 표본 제시는 각 12,800회, 문장당 추가 반복은 A 50회 / B 25회입니다.\n"
            "새 256개는 B에서만 학습합니다. 추가 화자나 녹화는 없으며, 길이 구성·처리 프레임·반복 빈도도 달라져 순수 데이터 수의 인과 효과는 아닙니다.\n"
            "80개는 반복 사용하는 고정 개발 평가 자료입니다. 단일 계보·seed 비교이며, test50·실제 웹캠 평가·배포는 수행하지 않았습니다."
        )
        fig.text(.037, .028, footer, va="bottom", fontsize=10, linespacing=1.8, color="#465159")
        fig.subplots_adjust(left=.049, right=.983, bottom=.233, top=.818, wspace=.35, hspace=.62)
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
                "Title": "Stage17 training diversity diagnostic",
                "Description": "Fixed development evaluation; sentence-mean greedy jamo CER"})
    finally:
        plt.close(figure)
    print(args.out.resolve())


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
