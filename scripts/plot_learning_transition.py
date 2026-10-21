"""Plot the completed stage12 train-only diagnostic; no model or dataset I/O.

Usage (CPU environment):
    .venv-tokenizer/Scripts/python scripts/plot_learning_transition.py \
        --summary results/step12-transition/run01/summary.json --out tmp/step12.png

The input must be the verified summary produced by summarize_learning_transition.py.
All CER curves use sentence-mean greedy CER, not beam CER or corpus WER.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ["MPLCONFIGDIR"] = str(ROOT / "tmp/matplotlib-step12")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

STEPS = [0, 1, 5, 10, 25, 50, 100, 200, 400, 800, 1600]
BRANCHES = ("A_keep16", "B_expand64")
LABELS = {"A_keep16": "A: keep 16", "B_expand64": "B: expand to 64"}
COLORS = {"A_keep16": "#1764a0", "B_expand64": "#cb6020"}
CONDITIONS = ("normal", "static_middle", "shuffle_42")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def number(value, label: str, *, fraction: bool = False, nullable: bool = False) -> float:
    """Reject silently truncated, missing, negative or nonfinite plotted values."""
    if value is None and nullable:
        return float("nan")
    require(isinstance(value, (int, float)) and not isinstance(value, bool), f"Numeric {label} required")
    value = float(value)
    require(math.isfinite(value) and value >= 0, f"Finite nonnegative {label} required")
    require(not fraction or value <= 1, f"{label} must be a fraction in [0,1]")
    return value


def prepare_plot_data(summary: dict) -> dict:
    """Translate the verified summary into the small, explicit plotting contract."""
    require(summary.get("schema") == "lipnet-learning-transition-summary-v1"
            and summary.get("status") == "completed_and_verified",
            "A completed_and_verified stage12 summary is required")
    protocol = summary["protocol"]
    settings, scope = protocol["settings"], protocol["scope"]
    for key, expected in {"attempt_updates": 1600, "batch_size": 2, "learning_rate": .0003,
                          "seed": 42, "grad_clip": 1., "normal_steps": STEPS,
                          "fresh_optimizer_per_branch": True,
                          "precision": "amp_fp16_ctc_fp32"}.items():
        require(settings.get(key) == expected, f"Plot protocol mismatch: {key}")
    for key, expected in {"train_only": True, "validation_cache_access": False,
                          "test_cache_access": False, "generalization_evaluation": False,
                          "deployment": False}.items():
        require(scope.get(key) is expected, f"Forbidden or missing plot scope: {key}")
    require(set(summary["branches"]) == set(BRANCHES), "Expected exactly the A/B branches")
    require(summary["provenance"]["train_cache_count_verified"] == 64,
            "A verified train64 summary is required")
    common = summary["common_step0"]
    require(common["step"] == 0, "Shared initial snapshot must be step0")

    def group(snapshot: dict, cohort: str, condition: str = "normal") -> dict:
        value = snapshot["cohorts"][cohort][condition]
        expected = 16 if cohort == "short16" else 48
        require(value["count"] == value["greedy"]["count"] == expected,
                f"Unexpected cohort count: {cohort}/{condition}")
        return value

    def cer(value: dict) -> float:
        return number(value["greedy"]["mean_sentence_jamo_cer"], "sentence-mean greedy CER")

    def exact(value: dict) -> int:
        result = value["greedy"]["exact_count"]
        require(isinstance(result, int) and not isinstance(result, bool)
                and 0 <= result <= value["count"], "Invalid exact sentence count")
        return result

    def whole64(snapshot: dict, condition: str = "normal") -> tuple[float, float, float]:
        values = [group(snapshot, cohort, condition) for cohort in ("short16", "additional48")]
        # The top four identities can differ by cohort; averaging top4 fractions
        # would be wrong. Recombine complete token counters and rank them anew.
        pooled = Counter()
        for value in values:
            frequency = value["token_frequency"]
            counts = frequency["collapsed_counts_by_token_id"]
            for token_id, count in counts.items():
                require(isinstance(token_id, str) and token_id.isdigit() and 1 <= int(token_id) <= 68,
                        "Invalid nonblank token ID")
                require(isinstance(count, int) and not isinstance(count, bool) and count >= 0,
                        "Invalid collapsed-token frequency")
            require(sum(counts.values()) == frequency["collapsed_token_count"], "Token total mismatch")
            pooled.update(counts)
        total = sum(pooled.values())
        top4 = sum(sorted(pooled.values(), reverse=True)[:4]) / total if total else 0.
        blank = sum(value["count"] * number(value["ctc"]["blank_posterior_mean"]["mean"],
                      "clip-mean blank probability", fraction=True) for value in values) / 64
        mean_cer = sum(value["count"] * cer(value) for value in values) / 64
        return top4, blank, mean_cer

    series = {}
    for branch in BRANCHES:
        values = summary["branches"][branch]
        require(values["attempt_updates"] == 1600, "Incomplete branch")
        skips = values["skipped_optimizer_steps"]
        require(isinstance(skips, int) and not isinstance(skips, bool) and 0 <= skips <= 1600,
                "Invalid AMP skip count")
        require(values["optimizer_steps"] + skips == 1600, "Attempt/update/skip mismatch")
        snapshots = values["snapshots"]
        require(set(snapshots) == {f"{step:04d}" for step in STEPS[1:]}, "Snapshot inventory mismatch")
        schedule = protocol["schedules"][branch]
        expected_size, expected_exposure = (16, 200) if branch == "A_keep16" else (64, 50)
        require(schedule["sample_count"] == expected_size and schedule["total_presentations"] == 3200
                and len(schedule["exposures"]) == expected_size
                and set(schedule["exposures"].values()) == {expected_exposure}, "Unequal-exposure protocol mismatch")
        row = {"steps": list(STEPS), "old_cer": [], "old_exact": [], "additional_cer": [],
               "top4": [], "blank": [], "windows": [], "skips": skips}
        for step in STEPS:
            snapshot = common if step == 0 else snapshots[f"{step:04d}"]
            require(snapshot["step"] == step, "Snapshot step mismatch")
            old, additional = group(snapshot, "short16"), group(snapshot, "additional48")
            row["old_cer"].append(cer(old))
            row["old_exact"].append(exact(old))
            row["additional_cer"].append(cer(additional))
            top4, blank, _ = whole64(snapshot)
            row["top4"].append(top4)
            row["blank"].append(blank)
        windows = values["fixed_windows"]
        bounds = [(1, 100), (101, 400), (401, 800), (801, 1600)]
        require(set(windows) == {f"{start:04d}-{end:04d}" for start, end in bounds},
                "Fixed telemetry-window inventory mismatch")
        for start, end in bounds:
            window = windows[f"{start:04d}-{end:04d}"]
            require(window["first_attempt"] == start and window["last_attempt"] == end,
                    "Window bounds mismatch")
            total = window["groups"]["total"]
            row["windows"].append({"start": start, "end": end,
                "preclip": number(total["preclip"]["l2_norm"]["median"], "gradient norm median", nullable=True),
                "relative_update": number(total["parameter_update"]["relative_l2"]["median"],
                                          "relative update median", nullable=True)})
        row["final_control_cer"] = [whole64(snapshots["1600"], condition)[2] for condition in CONDITIONS]
        series[branch] = row
    return {"series": series}


def _timeline_axis(ax) -> None:
    ax.set_xscale("symlog", linthresh=1, linscale=.7, base=10)
    ax.set_xlim(-.12, 1880)
    ax.set_xticks(STEPS)
    ax.set_xticklabels([str(step) for step in STEPS], rotation=40, ha="right", fontsize=8)
    ax.set_xlabel("Attempted optimizer updates")
    ax.grid(axis="both", alpha=.17, linewidth=.6)
    ax.set_axisbelow(True)


def render_figure(data: dict, font_path: Path | None = None):
    """Return one six-panel figure. This function does not write output files."""
    if font_path is not None:
        require(font_path.is_file(), f"Font file not found: {font_path}")
        font_manager.fontManager.addfont(str(font_path))
        font_name = font_manager.FontProperties(fname=str(font_path)).get_name()
    else:
        font_name = "DejaVu Sans"
    style = {"font.family": font_name, "font.size": 10, "axes.titlesize": 12,
             "axes.labelsize": 10, "axes.spines.top": False, "axes.spines.right": False,
             "axes.unicode_minus": True, "legend.fontsize": 8}
    with plt.rc_context(style):
        fig, axes = plt.subplots(2, 3, figsize=(17.2, 10.9))
        a, b, c, d, e, f = axes.flat
        update_axis = e.twinx()
        update_axis.spines["right"].set_visible(True)
        for ax in (a, b, c, d, e):
            _timeline_axis(ax)
        for branch in BRANCHES:
            values, color = data["series"][branch], COLORS[branch]
            x = values["steps"]
            common = dict(color=color, marker="o", markersize=4, linewidth=1.7,
                          label=LABELS[branch], zorder=3)
            # Different dash patterns keep coincident branch traces distinguishable.
            line_style = "-" if branch == "A_keep16" else "--"
            a.plot(x, 100 * np.asarray(values["old_cer"]), linestyle=line_style, **common)
            b.plot(x, values["old_exact"], linestyle=line_style, **common)
            c.plot(x, 100 * np.asarray(values["additional_cer"]), linestyle=line_style, **common)
            d.plot(x, 100 * np.asarray(values["top4"]), color=color, marker="o", markersize=3,
                   linewidth=1.7, label=f"{branch[0]} top-4", linestyle="-")
            d.plot(x, 100 * np.asarray(values["blank"]), color=color, marker="s", markersize=3,
                   linewidth=1.5, label=f"{branch[0]} blank", linestyle=":")
            windows = values["windows"]
            ends = [window["end"] for window in windows]
            e.plot(ends, [window["preclip"] for window in windows], color=color, marker="o",
                   markersize=3, linewidth=1.5, label=f"{branch[0]} gradient")
            update_axis.plot(ends, [window["relative_update"] for window in windows], color=color,
                             marker="s", markersize=3, linewidth=1.5, linestyle=":",
                             label=f"{branch[0]} update")
            positions = np.arange(len(CONDITIONS)) + (-.19 if branch == "A_keep16" else .19)
            bars = f.bar(positions, 100 * np.asarray(values["final_control_cer"]), width=.36,
                         color=color, alpha=.9, label=LABELS[branch])
            f.bar_label(bars, fmt="%.1f", padding=3, fontsize=9)

        a.set(title="A | Original 16: transcription error", ylabel="Mean greedy CER (%)")
        b.set(title="B | Original 16: exact sentences", ylabel="Exact matches / 16",
              ylim=(-.5, 17), yticks=[0, 4, 8, 12, 16])
        c.set(title="C | Additional 48: transcription error", ylabel="Mean greedy CER (%)")
        d.set(title="D | All 64: output concentration", ylabel="Fraction (%)", ylim=(-2, 104))
        e.set(title="E | Gradients and actual updates", ylabel="Pre-clip gradient L2 (window median)")
        e.set_yscale("symlog", linthresh=1e-6, linscale=.5)
        update_axis.set_yscale("symlog", linthresh=1e-8, linscale=.5)
        update_axis.set_ylabel("Relative update\n(window median)", fontsize=9, labelpad=3)
        e.set_ylim(top=e.get_ylim()[1] * 1.4)
        update_axis.set_ylim(top=update_axis.get_ylim()[1] * 1.4)
        e.text(.03, .035, "Unscaled gradients; actual Adam deltas\nFour fixed windows; null norms omitted, zero retained",
               transform=e.transAxes, fontsize=8, color="#505050", va="bottom")
        f.set(title="F | Final controls: all 64, greedy", ylabel="Mean greedy CER (%)",
              xticks=np.arange(len(CONDITIONS)),
              xticklabels=["Normal", "Static middle", "Shuffle (42)"])
        f.grid(axis="y", alpha=.17, linewidth=.6)
        f.set_axisbelow(True)
        for ax in (a, c, f):
            ax.set_ylim(bottom=0)
            ax.margins(y=.15)
        largest_control = max(100 * value for branch in BRANCHES
                              for value in data["series"][branch]["final_control_cer"])
        f.set_ylim(0, max(110., largest_control * 1.28))
        for ax in (a, b, c):
            ax.legend(frameon=False, loc="best")
        d.legend(frameon=False, ncol=2, loc="best")
        handles1, labels1 = e.get_legend_handles_labels()
        handles2, labels2 = update_axis.get_legend_handles_labels()
        e.legend(handles1 + handles2, labels1 + labels2, frameon=False, ncol=2, loc="upper left")
        f.legend(frameon=False, loc="upper right")
        c.text(.03, .55, "A: not trained on these 48\nB: trained on them",
               transform=c.transAxes, va="top", fontsize=8, color="#505050")
        d.text(.03, .035, "Top-4: pooled collapsed nonblank tokens\nBlank: mean posterior, averaged over clips",
               transform=d.transAxes, va="bottom", fontsize=8, color="#505050")

        fig.suptitle("Stage 12 | From 16 memorized clips to 64 training clips",
                     fontsize=19, fontweight="bold", x=.5, y=.97)
        fig.text(.5, .935, "Same compact checkpoint  |  Fresh Adam, lr 0.0003  |  Batch 2  |  AMP  |  Clip 1",
                 ha="center", fontsize=11, color="#454545")
        skips = ", ".join(f"{key[0]}={data['series'][key]['skips']}" for key in BRANCHES)
        footer = (
            "Train-only diagnostic; no validation/test evaluation or deployment. Single seed (42); no confidence intervals.\n"
            "Fixed 1600 attempted updates / 3200 presentations per arm; unequal per-clip exposure: A=200, B=50. "
            f"AMP skips: {skips}.\n"
            "Expansion also changes length, vocabulary and speaker mix. Additional 48 are not a held-out test set. "
            "X uses a symmetric-log scale (linear through 0-1); lines connect observed checkpoints."
        )
        fig.text(.04, .029, footer, va="bottom", fontsize=9, linespacing=1.6, color="#444444")
        fig.subplots_adjust(left=.07, right=.95, bottom=.19, top=.865, wspace=.52, hspace=.48)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path, help="New PNG output path")
    parser.add_argument("--font", type=Path, help="Optional local TTF/OTF font file")
    args = parser.parse_args()
    require(args.out.suffix.lower() == ".png", "Output must be a PNG file")
    require(not args.out.exists(), f"Refusing to overwrite: {args.out}")
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    data = prepare_plot_data(summary)
    fig = render_figure(data, args.font)
    try:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation also prevents a competing task's output being replaced.
        with args.out.open("xb") as stream:
            fig.savefig(stream, format="png", dpi=170, facecolor="white",
                        metadata={"Title": "Stage12 train-only learning transition",
                                  "Description": "Verified summary; fixed attempts, unequal exposure, seed42"})
    finally:
        plt.close(fig)
    print(args.out.resolve())


if __name__ == "__main__":
    main()
