"""Plot a verified stage14 learning-rate comparison, without model or data access.

Every CER shown is sentence-mean greedy jamo CER on the same training clips.
The two continuations are one paired experiment, not independent seed repeats.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ["MPLCONFIGDIR"] = str(ROOT / "tmp/matplotlib-step14")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
import numpy as np

BRANCHES = ("A_keep", "B_reduce")
STEPS = [6400, 6401, 6500, 6800, 7200, 8000, 8800, 9600]
CONTROL_STEPS = [6400, 7200, 8000, 9600]
WINDOWS = ((6401, 6500), (6501, 6800), (6801, 7200), (7201, 8000),
           (8001, 8800), (8801, 9600))
LABELS = {"A_keep": "A: lr 0.0003", "B_reduce": "B: lr 0.0001"}
COLORS = {"A_keep": "#1764a0", "B_reduce": "#c45e21"}
CONDITIONS = ("normal", "static_middle", "shuffle_42")
GRAY, GREEN = "#707070", "#24836a"


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


def greedy_cer(group: dict) -> float:
    return number(group["greedy"]["mean_sentence_jamo_cer"], "sentence-mean greedy CER")


def exact_count(group: dict) -> int:
    count = group["greedy"]["exact_count"]
    require(isinstance(count, int) and not isinstance(count, bool) and 0 <= count <= group["count"],
            "Invalid exact count")
    return count


def token_concentration(group: dict) -> float:
    frequency = group["token_frequency"]
    counts = frequency["collapsed_counts_by_token_id"]
    for token, count in counts.items():
        require(isinstance(token, str) and token.isdigit() and 1 <= int(token) <= 68
                and isinstance(count, int) and not isinstance(count, bool) and count >= 0,
                "Invalid collapsed nonblank token count")
    total = sum(counts.values())
    require(total == frequency["collapsed_token_count"], "Collapsed token total mismatch")
    value = sum(sorted(counts.values(), reverse=True)[:4]) / total if total else 0.
    require(math.isclose(value, number(frequency["pooled_top4_fraction"], "top4", fraction=True),
                         abs_tol=1e-12), "Pooled top4 mismatch")
    return value


def prepare_plot_data(summary: dict) -> dict:
    """Validate the completed summary contract and expose all plotted quantities."""
    require(summary.get("schema") == "lipnet-learning-rate-summary-v1"
            and summary.get("status") == "completed_and_verified", "Verified stage14 summary required")
    protocol = summary["protocol"]
    settings = protocol["settings"]
    steps = settings["normal_steps"]
    require(steps == STEPS, "Unexpected fixed observation steps")
    for name, expected in {"batch_size": 2, "seed": 42, "grad_clip": 1.,
                           "precision": "amp_fp16_ctc_fp32", "control_steps": CONTROL_STEPS}.items():
        require(settings.get(name) == expected, f"Unexpected setting: {name}")
    for name, expected in {"train_only": True, "validation_cache_access": False,
                           "test_cache_access": False, "generalization_evaluation": False,
                           "deployment": False}.items():
        require(protocol["scope"].get(name) is expected, f"Unexpected scope: {name}")
    require(set(summary["branches"]) == set(BRANCHES), "Expected the fixed A/B branches")
    data = {"steps": steps, "branches": {}, "reference_top4": number(
        protocol["cohort_metadata"]["additional48"]["reference_pooled_top4_fraction"],
        "reference top4", fraction=True)}

    def group(snapshot: dict, cohort: str, condition: str = "normal") -> dict:
        value = snapshot["cohorts"][cohort][condition]
        expected = 16 if cohort == "short16" else 48
        require(value["count"] == value["greedy"]["count"] == expected, "Cohort count mismatch")
        return value

    baseline_groups = []
    bin_contract = None
    for branch in BRANCHES:
        expected_lr = .0003 if branch == "A_keep" else .0001
        require(protocol["branches"][branch]["learning_rate"] == expected_lr
                and protocol["branches"][branch]["settings"]["learning_rate"] == expected_lr,
                "Unexpected branch learning rate")
        result = summary["branches"][branch]
        snapshots = result["snapshots"]
        require(set(snapshots) == {str(step) for step in steps}, "Snapshot inventory mismatch")
        values = {"old_cer": [], "old_exact": [], "new_cer": [], "new_exact": [],
                  "top4": [], "blank": [], "controls": [], "windows": [], "length_bins": {}}
        for step in steps:
            snapshot = snapshots[str(step)]
            require(snapshot["step"] == step, "Snapshot step mismatch")
            old, new = group(snapshot, "short16"), group(snapshot, "additional48")
            values["old_cer"].append(greedy_cer(old))
            values["old_exact"].append(exact_count(old))
            values["new_cer"].append(greedy_cer(new))
            values["new_exact"].append(exact_count(new))
            values["top4"].append(token_concentration(new))
            values["blank"].append(number(new["ctc"]["blank_posterior_mean"]["mean"], "blank", fraction=True))
        baseline_groups.append(snapshots["6400"]["cohorts"])
        for condition in CONDITIONS:
            values["controls"].append(greedy_cer(group(snapshots["9600"], "additional48", condition)))
        windows = result["fixed_windows"]
        require(set(windows) == {f"{first:04d}-{last:04d}" for first, last in WINDOWS},
                "Fixed window inventory mismatch")
        next_start = 6401
        for key, window in sorted(windows.items()):
            start, end = window["first_attempt"], window["last_attempt"]
            require(key == f"{start:04d}-{end:04d}" and start == next_start
                    and window["attempts_observed"] == end-start+1, "Window inventory mismatch")
            total = window["groups"]["total"]
            skipped = window["skipped_count"]
            require(isinstance(skipped, int) and not isinstance(skipped, bool)
                    and 0 <= skipped <= end-start+1, "Invalid skipped attempt count")
            values["windows"].append({"start": start, "end": end, "skips": skipped,
                "gradient": number(total["preclip"]["l2_norm"]["median"], "gradient median", nullable=True),
                "update": number(total["parameter_update"]["relative_l2"]["median"], "update median", nullable=True)})
            next_start = end+1
        require(next_start == 9601, "Incomplete 6401-9600 telemetry windows")
        for step in (6400, 9600):
            bins = result["length_bins"][str(step)]
            contract, values["length_bins"][step] = [], []
            for key, value in sorted(bins.items()):
                first, last = value["minimum_frames"], value["maximum_frames"]
                group_value = value["summary"]
                ids = value["sample_ids"]
                count = group_value["count"]
                require((first, last) in ((1, 50), (51, 100), (101, 150), (151, 200))
                        and key == f"{first:03d}-{last:03d}" and len(ids) == len(set(ids)) == count
                        and count == group_value["greedy"]["count"], "Length bin mismatch")
                contract.append((key, ids))
                values["length_bins"][step].append({"label": f"{first}-{last}", "count": count,
                    "cer": greedy_cer(group_value)})
            require(sum(row["count"] for row in values["length_bins"][step]) == 48,
                    "Length bins must cover additional48")
            if bin_contract is None:
                bin_contract = contract
            require(contract == bin_contract, "Length bin cohorts must be identical across branches/steps")
        values["skips"] = sum(window["skips"] for window in values["windows"])
        data["branches"][branch] = values
    require(baseline_groups[0] == baseline_groups[1], "A/B must share the same baseline metrics")
    require(summary["common"]["step"] == 6400 and summary["common"]["cohorts"] == baseline_groups[0],
            "Shared baseline summary mismatch")
    return data


def timeline(ax, steps) -> None:
    ax.set_xlim(6320, 9760)
    ax.set_xticks([6400, 7200, 8000, 8800, 9600])
    ax.set_xlabel("Cumulative attempted optimizer updates")
    ax.grid(alpha=.17, linewidth=.6)
    ax.set_axisbelow(True)


def metric_axis(ax, values: list[float], floor: float) -> None:
    finite = [value for value in values if math.isfinite(value)]
    positive = [value for value in finite if value > 0]
    if not positive:
        ax.set_ylim(-.03, 1.)
        ax.set_yticks([0., .5, 1.])
        return
    # Reserve room for the four-entry legend on both log-scaled axes.
    low, high = min(positive)*.8, max(positive)*1.8
    ax.set_yscale("symlog", linthresh=floor, linscale=.5)
    ticks = list(np.geomspace(low, high, 4))
    if 0. in finite:
        ticks = [0., *ticks]
        low = -floor*.08
    ax.set_ylim(low, high)
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{value:.2g}" for value in ticks], fontsize=8)


def render_figure(data: dict, font_path: Path | None = None):
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
        fig, axes = plt.subplots(2, 3, figsize=(18.4, 11.8))
        a, b, c, d, e, f = axes.flat
        update_axis = e.twinx()
        update_axis.spines["right"].set_visible(True)
        for ax in (a, b, e, f):
            timeline(ax, data["steps"])
        x = data["steps"]
        for branch in BRANCHES:
            values, color = data["branches"][branch], COLORS[branch]
            a.plot(x, 100*np.asarray(values["new_cer"]), color=color, marker="o", markersize=4,
                   label=LABELS[branch])
            b.plot(x, 100*np.asarray(values["new_exact"])/48, color=color, marker="o", markersize=4,
                   label=f"{branch[0]}: additional 48")
            b.plot(x, 100*np.asarray(values["old_exact"])/16, color=color, marker="s", markersize=3,
                   linestyle=":" if branch == "A_keep" else "--", label=f"{branch[0]}: original 16")
            windows = values["windows"]
            ends = [row["end"] for row in windows]
            e.plot(ends, [row["gradient"] for row in windows], color=color, marker="o", markersize=4)
            update_axis.plot(ends, [row["update"] for row in windows], color=color, marker="s", markersize=3.5,
                             linestyle=":")
            f.plot(x, 100*np.asarray(values["top4"]), color=color, marker="o", markersize=3.5,
                   label=f"{branch[0]}: output top-4")
            f.plot(x, 100*np.asarray(values["blank"]), color=color, marker="s", markersize=3,
                   linestyle=":", label=f"{branch[0]}: blank posterior")

        a.set(title="A | Additional 48: transcription error", ylabel="Mean greedy CER (%)", ylim=(0, None))
        a.legend(frameon=False, loc="upper right")
        b.set(title="B | Exact transcription: two cohorts", ylabel="Exact sentences / cohort size (%)",
              ylim=(-3, 122), yticks=[0, 25, 50, 75, 100])
        b.legend(frameon=False, ncols=2, loc="upper center")

        condition_positions = np.arange(len(CONDITIONS))
        for offset, branch in zip((-.19, .19), BRANCHES):
            values = 100*np.asarray(data["branches"][branch]["controls"])
            bars = c.bar(condition_positions+offset, values, width=.34, color=COLORS[branch], label=LABELS[branch])
            c.bar_label(bars, labels=[f"{value:.1f}" for value in values], padding=3, fontsize=8.5)
        c.set(title="C | At 9,600: time controls", ylabel="Additional 48 mean greedy CER (%)",
              xticks=condition_positions, xticklabels=["Normal", "Static middle", "Shuffle (42)"])
        c.set_ylim(0, max(max(data["branches"][branch]["controls"]) for branch in BRANCHES)*130+1)
        c.legend(frameon=False, loc="upper left")
        c.grid(axis="y", alpha=.17, linewidth=.6)
        c.set_axisbelow(True)

        baseline_bins = data["branches"]["A_keep"]["length_bins"][6400]
        bin_positions = np.arange(len(baseline_bins))
        for offset, label, color, bins in [(-.25, "Shared 6,400", "#b8bfc4", baseline_bins),
                (0., "A at 9,600", COLORS["A_keep"], data["branches"]["A_keep"]["length_bins"][9600]),
                (.25, "B at 9,600", COLORS["B_reduce"], data["branches"]["B_reduce"]["length_bins"][9600])]:
            d.bar(bin_positions+offset, [100*row["cer"] for row in bins], width=.22, label=label, color=color)
        d.set(title="D | Additional 48: error by clip length", ylabel="Mean greedy CER (%)",
              xlabel="Input frames at 25 fps; fixed groups", xticks=bin_positions,
              xticklabels=[f"{row['label']}\nn = {row['count']}" for row in baseline_bins], ylim=(0, None))
        d.legend(frameon=False, fontsize=7.5, loc="upper left")
        d.grid(axis="y", alpha=.17, linewidth=.6)
        d.set_axisbelow(True)

        gradients = [row["gradient"] for branch in BRANCHES for row in data["branches"][branch]["windows"]]
        updates = [row["update"] for branch in BRANCHES for row in data["branches"][branch]["windows"]]
        e.set(title="E | Gradients and actual Adam updates", ylabel="Pre-clip L2 (window median)")
        update_axis.set_ylabel("Relative update (window median)")
        metric_axis(e, gradients, 1e-5)
        metric_axis(update_axis, updates, 1e-8)
        e.legend(handles=[Line2D([0], [0], color=COLORS[branch], label=LABELS[branch]) for branch in BRANCHES]
                 + [Line2D([0], [0], color=GRAY, marker="o", label="Gradient L2"),
                    Line2D([0], [0], color=GRAY, marker="s", linestyle=":", label="Relative update")],
                 loc="upper right", frameon=False, fontsize=7.5)
        f.axhline(100*data["reference_top4"], color=GREEN, linestyle="--", linewidth=1.3, label="Reference top-4")
        fractions = [data["reference_top4"], *[value for branch in BRANCHES
            for field in ("top4", "blank") for value in data["branches"][branch][field]]]
        low = max(0., 5*math.floor((100*min(fractions)-5)/5))
        high = min(105., 5*math.ceil((100*max(fractions)+15)/5))
        f.set(title="F | Additional 48: output concentration", ylabel="Fraction (%)", ylim=(low, high))
        f.legend(frameon=False, loc="upper right", ncols=2, fontsize=7.5)

        fig.suptitle("Stage 14 | Learning rate comparison from the same saved state",
                     fontsize=19, fontweight="bold", y=.976)
        fig.text(.5, .941, "Same model, Adam moments, scaler, RNG and batch order  |  6,400 to 9,600 attempts  |  64 training clips",
                 ha="center", fontsize=11, color="#454545")
        baseline = data["branches"]["A_keep"]
        lines = [f"Shared 6,400: new CER {100*baseline['new_cer'][0]:.2f}%, {baseline['new_exact'][0]}/48 exact; old {baseline['old_exact'][0]}/16 exact"]
        lines.append("     |     ".join(f"{LABELS[branch]} at 9,600: new CER {100*data['branches'][branch]['new_cer'][-1]:.2f}%, "
            f"{data['branches'][branch]['new_exact'][-1]}/48 exact; old {data['branches'][branch]['old_exact'][-1]}/16 exact"
            for branch in BRANCHES))
        fig.text(.5, .16, "\n".join(lines), ha="center", fontsize=10, linespacing=1.8,
                 bbox=dict(boxstyle="round,pad=.6", facecolor="#edf3f7", edgecolor="none"))
        footer = (
            "One paired continuation, seed 42; no independent seed repetitions, validation/test evaluation or deployment. Training fit does not establish webcam accuracy.\n"
            "Each branch adds 3,200 attempted updates and 100 planned presentations per clip. "
            f"Additional AMP skips: A {data['branches']['A_keep']['skips']}, B {data['branches']['B_reduce']['skips']}.\n"
            "CER is sentence-mean greedy jamo CER. Time controls preserve frame count. Length groups contain the same clips at baseline and both endpoints.\n"
            "Gradients are unscaled; updates are actual parameter changes. Window medians omit null norms and retain zero updates. "
            "Top-4 pools collapsed nonblank tokens; blank is a mean over clips."
        )
        fig.text(.045, .032, footer, va="bottom", fontsize=8.4, linespacing=1.55, color="#444444")
        fig.subplots_adjust(left=.061, right=.96, bottom=.285, top=.87, wspace=.48, hspace=.49)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--font", type=Path)
    args = parser.parse_args()
    require(args.out.suffix.lower() == ".png" and not args.out.exists(), "A new PNG output path is required")
    data = prepare_plot_data(json.loads(args.summary.read_text(encoding="utf-8")))
    fig = render_figure(data, args.font)
    try:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("xb") as stream:
            fig.savefig(stream, format="png", dpi=170, facecolor="white", metadata={
                "Title": "Stage14 fixed train64 learning rate comparison",
                "Description": "Verified summary; paired continuation; no held-out evaluation"})
    finally:
        plt.close(fig)
    print(args.out.resolve())


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
