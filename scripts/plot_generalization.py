"""Plot verified stage15 fixed-model development evaluation; no data/model access.

The prior training48 results are a reference from stage14, not stage15 inference.
CER is sentence-mean greedy jamo edit error and is not recognition accuracy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ["MPLCONFIGDIR"] = str(ROOT / "tmp/matplotlib-step15")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
import numpy as np

MODELS = ("source6400", "A_keep9600", "B_reduce9600")
LABELS = {"source6400": "13단계 · 6,400회", "A_keep9600": "A 유지 · 9,600회",
          "B_reduce9600": "B 감소 · 9,600회"}
COLORS = {"source6400": "#75838b", "A_keep9600": "#2273aa", "B_reduce9600": "#cf651f"}
COHORTS = ("train60", "val20")
COUNTS = {"train60": 60, "val20": 20}
CONDITIONS = ("normal", "static_middle", "shuffle_42")
BINS = ("001-050", "051-100", "101-150", "151-200")


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


def group_values(group: dict, expected_count: int | None = None) -> dict:
    count = group["count"]
    require(isinstance(count, int) and not isinstance(count, bool) and count > 0,
            "Positive integer cohort count required")
    require(expected_count is None or count == expected_count, "Unexpected cohort count")
    greedy = group["greedy"]
    require(greedy["count"] == count, "Greedy count mismatch")
    cer = greedy["mean_sentence_jamo_cer"]
    require(isinstance(cer, (int, float)) and not isinstance(cer, bool)
            and math.isfinite(cer) and cer >= 0, "Finite nonnegative CER required")
    exact = greedy["exact_count"]
    require(isinstance(exact, int) and not isinstance(exact, bool) and 0 <= exact <= count,
            "Invalid exact count")
    return {"count": count, "cer": float(cer), "exact": exact}


def prepare_plot_data(summary: dict, prior: dict) -> dict:
    require(summary.get("schema") == "lipnet-generalization-summary-v1"
            and summary.get("status") == "completed_and_verified", "Verified stage15 summary required")
    require(prior.get("schema") == "lipnet-learning-rate-summary-v1"
            and prior.get("status") == "completed_and_verified", "Verified stage14 training reference required")
    require(set(summary["models"]) == set(MODELS), "Unexpected fixed model set")
    previous = {
        "source6400": prior["common"]["cohorts"]["additional48"]["normal"],
        "A_keep9600": prior["branches"]["A_keep"]["snapshots"]["9600"]["cohorts"]["additional48"]["normal"],
        "B_reduce9600": prior["branches"]["B_reduce"]["snapshots"]["9600"]["cohorts"]["additional48"]["normal"],
    }
    data = {"models": {}, "bin_counts": {}, "speaker_counts": {}}
    for name in MODELS:
        model = summary["models"][name]
        require(set(model["cohorts"]) == set(COHORTS), "Unexpected evaluation cohort set")
        values = {"training48": group_values(previous[name], 48), "cohorts": {}, "length": {}, "speakers": {}}
        for cohort in COHORTS:
            groups = model["cohorts"][cohort]
            require(set(groups) == set(CONDITIONS), "Unexpected control set")
            values["cohorts"][cohort] = {condition: group_values(groups[condition], COUNTS[cohort])
                                          for condition in CONDITIONS}
            lengths = model["normal_by_length"][cohort]
            require(set(lengths) <= set(BINS) and lengths, "Unexpected length bin set")
            values["length"][cohort] = {key: group_values(value["summary"]) for key, value in lengths.items()}
            bin_counts = {key: row["count"] for key, row in values["length"][cohort].items()}
            require(sum(bin_counts.values()) == COUNTS[cohort], "Length bins do not cover cohort")
            if cohort not in data["bin_counts"]:
                data["bin_counts"][cohort] = bin_counts
            require(data["bin_counts"][cohort] == bin_counts, "Length counts differ across fixed models")
            speakers = model["normal_by_speaker"][cohort]
            values["speakers"][cohort] = {key: group_values(value["summary"]) for key, value in speakers.items()}
            speaker_counts = {key: row["count"] for key, row in values["speakers"][cohort].items()}
            require(sum(speaker_counts.values()) == COUNTS[cohort], "Speaker groups do not cover cohort")
            require(len(speaker_counts) == (8 if cohort == "train60" else 2), "Unexpected speaker count")
            if cohort not in data["speaker_counts"]:
                data["speaker_counts"][cohort] = speaker_counts
            require(data["speaker_counts"][cohort] == speaker_counts, "Speaker counts differ across fixed models")
        data["models"][name] = values
    require(not (set(data["speaker_counts"]["train60"]) & set(data["speaker_counts"]["val20"])),
            "Evaluation speaker cohorts overlap")
    return data


def setup_axis(ax, title: str, maximum: float) -> None:
    ax.set_title(title, loc="left", fontweight="bold", pad=14)
    ax.set_ylabel("문장 평균 자모 CER (%)")
    ax.set_ylim(0, max(10., maximum * 1.20))
    ax.grid(axis="y", alpha=.18)
    ax.set_axisbelow(True)


def render_figure(data: dict, font_path: Path):
    require(font_path.is_file(), f"Korean font not found: {font_path}")
    font_manager.fontManager.addfont(str(font_path))
    font_name = font_manager.FontProperties(fname=str(font_path)).get_name()
    style = {"font.family": font_name, "font.size": 10, "axes.titlesize": 12,
             "axes.labelsize": 10, "axes.spines.top": False, "axes.spines.right": False,
             "axes.unicode_minus": False, "legend.fontsize": 10}
    with plt.rc_context(style):
        fig, axes = plt.subplots(2, 3, figsize=(19.2, 12.0))
        a, b, c, d, e, f = axes.flat
        all_cer = [value["cer"] * 100 for model in data["models"].values()
                   for cohort in COHORTS for value in model["cohorts"][cohort].values()]
        main_maximum = max(all_cer)
        width = .24
        for index, name in enumerate(MODELS):
            model = data["models"][name]
            normal = [model["training48"], model["cohorts"]["train60"]["normal"],
                      model["cohorts"]["val20"]["normal"]]
            xx = np.arange(3) + (index - 1) * width
            bars = a.bar(xx, [100 * row["cer"] for row in normal], width=width,
                         color=COLORS[name], label=LABELS[name])
            for bar, row in zip(bars, normal):
                a.annotate(f"{100*row['cer']:.1f}", (bar.get_x()+bar.get_width()/2, bar.get_height()),
                           ha="center", va="bottom", xytext=(0, 3), textcoords="offset points", fontsize=9)
            for ax, cohort in ((b, "train60"), (c, "val20")):
                yy = [100 * model["cohorts"][cohort][condition]["cer"] for condition in CONDITIONS]
                ax.plot(range(3), yy, marker="o", linewidth=2, markersize=5,
                        color=COLORS[name], label=LABELS[name])
            for ax, cohort in ((d, "train60"), (e, "val20")):
                yy = [100 * model["length"][cohort][key]["cer"]
                      if key in model["length"][cohort] else np.nan for key in BINS]
                ax.plot(range(4), yy, marker="o", linewidth=2, markersize=5, color=COLORS[name])
            speaker_rows = [(cohort, speaker) for cohort in COHORTS
                            for speaker in sorted(data["speaker_counts"][cohort])]
            f.plot(range(len(speaker_rows)), [100 * model["speakers"][cohort][speaker]["cer"]
                                             for cohort, speaker in speaker_rows],
                   marker="o", linewidth=1.6, markersize=4, color=COLORS[name])
        setup_axis(a, "A. 정상 영상: 학습 복원과 새 발화", main_maximum)
        a.set_xticks(range(3), ["학습했던 48개\n이전 결과 참조", "미학습 train60\n익숙한 8화자", "개발 val20\n미학습 2화자"])
        for ax, cohort, title in ((b, "train60", "B. 익숙한 화자 · 다른 녹화 60개"),
                                  (c, "val20", "C. 미학습 화자 · 개발 영상 20개")):
            setup_axis(ax, title, main_maximum)
            ax.set_xticks(range(3), ["정상", "중간 프레임 반복", "순서 섞기 42"])
        for ax, cohort, title in ((d, "train60", "D. train60 길이별 정상 영상"),
                                  (e, "val20", "E. val20 길이별 정상 영상")):
            maximum = max(100 * row["cer"] for name in MODELS
                          for row in data["models"][name]["length"][cohort].values())
            setup_axis(ax, title, maximum)
            ax.set_xticks(range(4), [f"{int(key[:3])}–{int(key[4:])}\nn={data['bin_counts'][cohort].get(key, 0)}"
                                      for key in BINS])
            ax.set_xlabel("입력 프레임 수 (25fps)")
        speaker_rows = [(cohort, speaker) for cohort in COHORTS
                        for speaker in sorted(data["speaker_counts"][cohort])]
        maximum = max(100 * model["speakers"][cohort][speaker]["cer"]
                      for model in data["models"].values() for cohort, speaker in speaker_rows)
        setup_axis(f, "F. 화자별 정상 영상", maximum)
        f.axvline(7.5, linestyle="--", color="#8f9ca2", linewidth=1)
        f.set_xticks(range(len(speaker_rows)),
                     [f"{speaker}\n{data['speaker_counts'][cohort][speaker]}" for cohort, speaker in speaker_rows],
                     fontsize=8)
        f.set_xlabel("화자 ID / 표본 수   ·   점선 오른쪽은 val20")
        legend = [Line2D([0], [0], color=COLORS[name], marker="o", linewidth=2, label=LABELS[name])
                  for name in MODELS]
        fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(.5, .903), ncol=3, frameon=False)
        fig.suptitle("15단계 | 학습 문장 복원과 미학습 영상 전이", fontsize=22, fontweight="bold", y=.98)
        fig.text(.5, .94, "고정 모델 3개 × 미학습 영상 80개 × 정상·정지·섞기   |   추가 학습 없이 비교",
                 ha="center", fontsize=12, color="#40505a")
        footer = (
            "모든 패널: greedy 문장 평균 자모 CER. 낮을수록 좋으며, 100-CER를 인식 정확도로 해석하지 않습니다.\n"
            "학습 48개는 14단계 저장 결과를 참조했습니다. train60은 학습 64개와 ID·녹화·정규화 문장이 겹치지 않도록 사전 선정했습니다.\n"
            "val20은 기존 개발 평가 자료입니다. 화자·문장·길이·녹화 조건이 함께 달라지므로 두 평가군의 차이는 화자만의 인과 효과가 아닙니다.\n"
            "작은 층별 표본의 기술 통계이며 독립 반복실험이 아닙니다. test50·실제 웹캠 평가와 배포는 수행하지 않았습니다."
        )
        fig.text(.046, .03, footer, va="bottom", fontsize=9.5, linespacing=1.75, color="#465159")
        fig.subplots_adjust(left=.063, right=.974, bottom=.235, top=.827, wspace=.32, hspace=.64)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--training-reference", type=Path, default=ROOT / "docs/단계별_개선/14_검증결과.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--font", type=Path, default=Path("C:/Windows/Fonts/malgun.ttf"))
    args = parser.parse_args()
    require(args.out.suffix.lower() == ".png" and not args.out.exists(), "A new PNG output path is required")
    summary = read_json(args.summary)
    reference_sha = hashlib.sha256(args.training_reference.read_bytes()).hexdigest()
    require(summary["provenance"]["prior_files"]["docs/단계별_개선/14_검증결과.json"] == reference_sha,
            "Training reference is not the prior summary frozen in stage15")
    data = prepare_plot_data(summary, read_json(args.training_reference))
    figure = render_figure(data, args.font)
    try:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("xb") as stream:
            figure.savefig(stream, format="png", dpi=170, facecolor="white", metadata={
                "Title": "Stage15 fixed-model generalization diagnostic",
                "Description": "Development evaluation; sentence-mean greedy jamo CER; prior train48 reference"})
    finally:
        plt.close(figure)
    print(args.out.resolve())


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
