"""Plot verified stage13 budget results without model or dataset access.

All CER values are sentence-mean greedy CER. The plotted checkpoints belong to
one fixed training trajectory, not independent repeats or an external test set.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ["MPLCONFIGDIR"] = str(ROOT / "tmp/matplotlib-step13")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

STEPS = [0, 1, 5, 10, 25, 50, 100, 200, 400, 800, 1600, 2400, 3200, 4800, 6400]
CONTROL_STEPS = [0, 100, 400, 1600, 3200, 6400]
CONDITIONS = ("normal", "static_middle", "shuffle_42")
WINDOWS = ((1, 100), (101, 400), (401, 800), (801, 1600),
           (1601, 2400), (2401, 3200), (3201, 4800), (4801, 6400))
BLUE, ORANGE, GREEN, GRAY = "#1764a0", "#c45e21", "#24836a", "#707070"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def number(value, label: str, *, fraction: bool = False, nullable: bool = False) -> float:
    if value is None and nullable:
        return float("nan")
    require(isinstance(value, (int, float)) and not isinstance(value, bool), f"Numeric {label} required")
    value = float(value)
    require(math.isfinite(value) and value >= 0, f"Finite nonnegative {label} required")
    require(not fraction or value <= 1, f"{label} must be in [0,1]")
    return value


def prepare_plot_data(summary: dict) -> dict:
    """Validate the plotting contract and expose every plotted value explicitly."""
    require(summary.get("schema") == "lipnet-learning-budget-summary-v1"
            and summary.get("status") == "completed_and_verified", "Verified stage13 summary required")
    protocol = summary["protocol"]
    for name, expected in {"attempt_updates": 6400, "batch_size": 2, "learning_rate": .0003,
                           "seed": 42, "grad_clip": 1., "normal_steps": STEPS,
                           "control_steps": CONTROL_STEPS, "precision": "amp_fp16_ctc_fp32"}.items():
        require(protocol["settings"].get(name) == expected, f"Unexpected setting: {name}")
    for name, expected in {"train_only": True, "validation_cache_access": False,
                           "test_cache_access": False, "generalization_evaluation": False,
                           "deployment": False}.items():
        require(protocol["scope"].get(name) is expected, f"Unexpected scope: {name}")
    snapshots = summary["snapshots"]
    require(set(snapshots) == {f"{step:04d}" for step in STEPS}, "Snapshot inventory mismatch")

    def group(step: int, cohort: str, condition: str = "normal") -> dict:
        snapshot = snapshots[f"{step:04d}"]
        require(snapshot["step"] == step, "Snapshot step mismatch")
        value = snapshot["cohorts"][cohort][condition]
        expected = 16 if cohort == "short16" else 48
        require(value["count"] == value["greedy"]["count"] == expected, "Cohort count mismatch")
        return value

    def cer(value: dict) -> float:
        return number(value["greedy"]["mean_sentence_jamo_cer"], "sentence-mean greedy CER")

    def exact(value: dict) -> int:
        count = value["greedy"]["exact_count"]
        require(isinstance(count, int) and not isinstance(count, bool)
                and 0 <= count <= value["count"], "Invalid exact count")
        return count

    data = {"steps": list(STEPS), "old_cer": [], "old_exact": [], "new_cer": [],
            "new_exact": [], "new_top4": [], "new_blank": [], "windows": [],
            "control_steps": list(CONTROL_STEPS), "controls": {c: [] for c in CONDITIONS}}
    for step in STEPS:
        old, new = group(step, "short16"), group(step, "additional48")
        data["old_cer"].append(cer(old))
        data["old_exact"].append(exact(old))
        data["new_cer"].append(cer(new))
        data["new_exact"].append(exact(new))
        frequency = new["token_frequency"]
        counts = frequency["collapsed_counts_by_token_id"]
        for token, count in counts.items():
            require(isinstance(token, str) and token.isdigit() and 1 <= int(token) <= 68
                    and isinstance(count, int) and not isinstance(count, bool) and count >= 0,
                    "Invalid collapsed nonblank token frequency")
        total = sum(counts.values())
        require(total == frequency["collapsed_token_count"], "Collapsed token count mismatch")
        top4 = sum(sorted(counts.values(), reverse=True)[:4]) / total if total else 0.
        require(math.isclose(top4, number(frequency["pooled_top4_fraction"], "top4", fraction=True),
                             abs_tol=1e-12), "Pooled top4 mismatch")
        data["new_top4"].append(top4)
        data["new_blank"].append(number(new["ctc"]["blank_posterior_mean"]["mean"], "blank", fraction=True))
    for step in CONTROL_STEPS:
        for condition in CONDITIONS:
            data["controls"][condition].append(cer(group(step, "additional48", condition)))
    windows = summary["fixed_windows"]
    require(set(windows) == {f"{start:04d}-{end:04d}" for start, end in WINDOWS}, "Window inventory mismatch")
    for start, end in WINDOWS:
        value = windows[f"{start:04d}-{end:04d}"]
        require(value["first_attempt"] == start and value["last_attempt"] == end
                and value["attempts_observed"] == end-start+1, "Window bounds/count mismatch")
        total = value["groups"]["total"]
        data["windows"].append({"start": start, "end": end,
            "gradient": number(total["preclip"]["l2_norm"]["median"], "gradient median", nullable=True),
            "update": number(total["parameter_update"]["relative_l2"]["median"], "update median", nullable=True),
            "skips": int(value["skipped_count"])})
    data["reference_top4"] = number(protocol["cohort_metadata"]["additional48"]["reference_pooled_top4_fraction"],
                                    "reference top4", fraction=True)
    data["skips"] = sum(row["skips"] for row in data["windows"])
    data["milestones"] = [{"step": step, "old_exact": exact(group(step, "short16")),
                           "new_exact": exact(group(step, "additional48")),
                           "new_cer": cer(group(step, "additional48"))} for step in (1600, 3200, 6400)]
    return data


def timeline(ax) -> None:
    ax.set_xlim(-100, 6650)
    ax.set_xticks([0, 1600, 3200, 4800, 6400])
    ax.set_xlabel("Attempted optimizer updates")
    ax.grid(alpha=.17, linewidth=.6)
    ax.set_axisbelow(True)
    ax.axvline(1600, color="#999999", linewidth=1.0, linestyle="--", alpha=.75, zorder=1)


def metric_axis(ax, values: list[float], floor: float) -> None:
    """Keep numerical ticks visible even when a log-range spans less than one decade."""
    finite = [value for value in values if math.isfinite(value)]
    positive = [value for value in finite if value > 0]
    if not positive:
        ax.set_ylim(-.03, 1.)
        ax.set_yticks([0., .5, 1.])
        return
    low, high = min(positive), max(positive)
    low, high = low*.85, high*1.18
    ax.set_yscale("symlog", linthresh=floor, linscale=.5)
    ticks = list(np.geomspace(low, high, 4))
    if 0. in finite:
        ticks = [0., *ticks]
        low = -floor*.08
    ax.set_ylim(low, high)
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{value:.2g}" for value in ticks], fontsize=8)


def render_figure(data: dict, font_path: Path | None = None):
    """Create six panels and a fixed-budget result strip; caller writes the PNG."""
    if font_path is not None:
        require(font_path.is_file(), f"Font not found: {font_path}")
        font_manager.fontManager.addfont(str(font_path))
        font_name = font_manager.FontProperties(fname=str(font_path)).get_name()
    else:
        font_name = "DejaVu Sans"
    style = {"font.family": font_name, "font.size": 10, "axes.titlesize": 12,
             "axes.labelsize": 10, "axes.spines.top": False, "axes.spines.right": False,
             "axes.unicode_minus": True, "legend.fontsize": 8}
    with plt.rc_context(style):
        fig, axes = plt.subplots(2, 3, figsize=(17.3, 11.3))
        a, b, c, d, e, f = axes.flat
        old_error_axis = a.twinx()
        update_axis = e.twinx()
        for ax in (old_error_axis, update_axis):
            ax.spines["right"].set_visible(True)
        for ax in axes.flat:
            timeline(ax)
        x = data["steps"]
        a.plot(x, data["old_exact"], color=BLUE, marker="o", markersize=3, label="Exact sentences")
        old_error_axis.plot(x, 100*np.asarray(data["old_cer"]), color=ORANGE, linestyle=":",
                            marker="s", markersize=2.7, label="CER")
        a.set(title="A | Original 16: retained transcription", ylabel="Exact matches / 16",
              ylim=(-.5, 18.7), yticks=[0, 4, 8, 12, 16])
        old_error_axis.set(ylabel="Mean greedy CER (%)", ylim=(-.2, max(8., 125*max(data["old_cer"]))))
        a.yaxis.label.set_color(BLUE)
        old_error_axis.yaxis.label.set_color(ORANGE)
        handles, labels = a.get_legend_handles_labels()
        h2, l2 = old_error_axis.get_legend_handles_labels()
        a.legend(handles+h2, labels+l2, loc="center right", frameon=False)
        a.text(.03, .97, "Early retention dip is compressed on the linear time axis.",
               transform=a.transAxes, fontsize=7.5, color=GRAY, va="top")

        b.plot(x, 100*np.asarray(data["new_cer"]), color=BLUE, marker="o", markersize=3.8)
        b.set(title="B | Additional 48: transcription error", ylabel="Mean greedy CER (%)", ylim=(0, None))
        c.plot(x, data["new_exact"], color=BLUE, marker="o", markersize=3.8)
        c.set(title="C | Additional 48: exact sentences", ylabel="Exact matches / 48",
              ylim=(-1, 52), yticks=[0, 12, 24, 36, 48])
        for row in data["milestones"]:
            align = "right" if row["step"] == 6400 else "center"
            b.annotate(f"{100*row['new_cer']:.2f}%", (row["step"], 100*row["new_cer"]),
                       xytext=(0, 10), textcoords="offset points", ha=align, color=BLUE, fontsize=9)
            c.annotate(f"{row['new_exact']}/48", (row["step"], row["new_exact"]),
                       xytext=(0, 9), textcoords="offset points", ha=align, color=BLUE, fontsize=9)

        labels = {"normal": "Normal", "static_middle": "Static middle", "shuffle_42": "Shuffle (42)"}
        for condition, color, marker in zip(CONDITIONS, (BLUE, ORANGE, GREEN), ("o", "s", "^")):
            d.plot(data["control_steps"], 100*np.asarray(data["controls"][condition]),
                   color=color, marker=marker, markersize=4, label=labels[condition])
        d.set(title="D | Additional 48: time controls", ylabel="Mean greedy CER (%)", ylim=(0, None))
        d.legend(frameon=False, loc="upper right")

        windows = data["windows"]
        ends = [row["end"] for row in windows]
        e.plot(ends, [row["gradient"] for row in windows], color=BLUE, marker="o", markersize=4,
               label="Gradient L2")
        update_axis.plot(ends, [row["update"] for row in windows], color=ORANGE, marker="s",
                         markersize=3.5, linestyle=":", label="Relative update")
        e.set(title="E | Gradients and actual parameter updates", ylabel="Pre-clip L2 (window median)")
        metric_axis(e, [row["gradient"] for row in windows], 1e-5)
        update_axis.set(ylabel="Relative update (window median)")
        metric_axis(update_axis, [row["update"] for row in windows], 1e-8)
        e.yaxis.label.set_color(BLUE)
        update_axis.yaxis.label.set_color(ORANGE)
        handles, labels = e.get_legend_handles_labels()
        h2, l2 = update_axis.get_legend_handles_labels()
        e.legend(handles+h2, labels+l2, loc="upper right", frameon=False)

        f.plot(x, 100*np.asarray(data["new_top4"]), color=BLUE, marker="o", markersize=3,
               label="Output top-4")
        f.plot(x, 100*np.asarray(data["new_blank"]), color=ORANGE, marker="s", markersize=3,
               linestyle=":", label="Blank posterior")
        f.axhline(100*data["reference_top4"], color=GREEN, linestyle="--", linewidth=1.3,
                  label="Reference top-4")
        f.set(title="F | Additional 48: output concentration", ylabel="Fraction (%)", ylim=(-2, 104))
        f.legend(frameon=False, loc="upper right")

        fig.suptitle("Stage 13 | More training on the same 64 clips", fontsize=19, fontweight="bold", y=.97)
        fig.text(.5, .935, "Same initial weights and batch sequence  |  Adam lr 0.0003  |  Batch 2  |  AMP  |  Clip 1",
                 ha="center", fontsize=11, color="#454545")
        strip = "     |     ".join(f"{row['step']:,} attempts: old {row['old_exact']}/16, new {row['new_exact']}/48, new CER {100*row['new_cer']:.2f}%"
                                   for row in data["milestones"])
        fig.text(.5, .147, strip, ha="center", fontsize=10,
                 bbox=dict(boxstyle="round,pad=.55", facecolor="#edf3f7", edgecolor="none"))
        footer = (
            "One train-only trajectory, seed 42; checkpoints are not independent repetitions. No validation/test evaluation or deployment.\n"
            "Dashed vertical line: the reproduced 1,600-attempt endpoint. Planned per-clip presentations: 50 / 100 / 200 at 1,600 / 3,200 / 6,400. "
            f"AMP skips: {data['skips']}.\n"
            "CER is sentence-mean greedy CER; controls preserve frame count. Top-4 pools collapsed nonblank tokens within the additional 48. "
            "Blank is averaged over clips.\n"
            "Window medians use unscaled gradients and actual Adam updates; null norms are omitted, zero updates retained. "
            "Lines connect fixed observations on a linear time axis."
        )
        fig.text(.043, .032, footer, va="bottom", fontsize=8.8, linespacing=1.55, color="#444444")
        fig.subplots_adjust(left=.066, right=.95, bottom=.245, top=.865, wspace=.56, hspace=.43)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--font", type=Path)
    args = parser.parse_args()
    require(args.out.suffix.lower() == ".png" and not args.out.exists(), "A new PNG output path is required")
    data = prepare_plot_data(json.loads(args.summary.read_text(encoding="utf-8")))
    fig = render_figure(data, args.font)
    try:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("xb") as stream:
            fig.savefig(stream, format="png", dpi=170, facecolor="white",
                        metadata={"Title": "Stage13 fixed train64 learning budget",
                                  "Description": "Verified summary; one trajectory; no held-out evaluation"})
    finally:
        plt.close(fig)
    print(args.out.resolve())


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
