"""Draw only the fixed stage5 checkpoint CER values, without model execution.

The source is the completed aggregate JSON. Neither a best checkpoint nor a
smoothed trajectory is selected. A machine-readable copy of every plotted
point preserves exact error counts, denominators and source provenance.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any

BASE = Path(__file__).resolve().parent
STEPS = (420, 840, 1260)
ARMS = ("A_sentence", "B_curriculum")
SEEDS = (42, 43)
COHORTS = (("dev/full_sentence/normal", "Internal development: 3 sentences", 3),
           ("train/full_sentence/normal", "Training subset: 12 sentences", 12))
COLORS = {"A_sentence": "#1474b8", "B_curriculum": "#e87919"}
ARM_LABELS = {"A_sentence": "A: sentence", "B_curriculum": "B: phrase to sentence"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def extract_points(aggregate: dict[str, Any]) -> list[dict[str, Any]]:
    """Check the complete fixed cohort and derive percentages from integer counts."""
    require(aggregate["schema"] == "offline-stage5-fixed-comparison-aggregate-v1", "Wrong aggregate schema")
    require(aggregate["status"] in ("completed", "completed_with_explicit_failures"), "Aggregate is incomplete")
    require(aggregate["primary_checkpoint"] == 1260 and aggregate["best_checkpoint_selection"] is False,
            "The chart must retain the prespecified final checkpoint")
    require(aggregate["checkpoints"] == 12 and aggregate["evaluations"] == 1080, "Fixed comparison scope differs")
    expected = {(seed, arm, step) for seed in SEEDS for arm in ARMS for step in STEPS}
    records = aggregate["checkpoint_diagnostics"]
    identities = [(record["seed"], record["arm"], record["completed_update"]) for record in records]
    require(len(identities) == len(set(identities)) == 12 and set(identities) == expected,
            "Need all twelve checkpoints exactly once")
    indexed = dict(zip(identities, records))
    points = []
    denominators = {}
    for cohort, _, expected_count in COHORTS:
        for arm in ARMS:
            for seed in SEEDS:
                for step in STEPS:
                    metrics = indexed[(seed, arm, step)]["cohorts"][cohort]
                    errors, characters, count = metrics["character_edits"], metrics["reference_characters"], metrics["count"]
                    require(type(errors) is int and errors >= 0 and type(characters) is int and characters > 0,
                            "Corpus character counts must be positive integer denominators and nonnegative errors")
                    require(type(count) is int and count == expected_count, "Sentence cohort count changed")
                    denominator = (count, characters)
                    require(cohort not in denominators or denominators[cohort] == denominator,
                            "Checkpoint corpus denominator changed")
                    denominators[cohort] = denominator
                    cer = metrics["CER"]
                    require(isinstance(cer, (int, float)) and math.isfinite(cer) and cer >= 0,
                            "CER must be a finite nonnegative ratio")
                    recomputed = errors / characters
                    require(math.isclose(cer, recomputed, rel_tol=1e-12, abs_tol=1e-12),
                            "Stored CER differs from integer corpus errors / characters")
                    require(type(metrics["failure_count"]) is int and 0 <= metrics["failure_count"] <= count,
                            "Invalid failed-row count")
                    points.append({"cohort": cohort, "arm": arm, "seed": seed, "completed_update": step,
                        "sentence_count": count, "character_edits": errors, "reference_characters": characters,
                        "CER": cer, "CER_percent": 100 * cer, "recomputed_CER": recomputed,
                        "exact_count": metrics["exact_count"], "failure_count": metrics["failure_count"],
                        "final_prespecified_checkpoint": step == 1260})
    require(len(points) == 24, "Expected 2 panels x 2 arms x 2 seeds x 3 points")
    return points


def write_plot(source: Path, output_dir: Path) -> dict[str, Any]:
    """Render a two-panel PNG and save the exact plotted points and their checks."""
    source = source.resolve()
    output_dir = output_dir.resolve()
    require(output_dir.is_relative_to(BASE), "Outputs must stay under this stage's reporting directory")
    aggregate = json.loads(source.read_text(encoding="utf-8"))
    points = extract_points(aggregate)
    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / "sentence_cer_trajectory.png"
    plotted = output_dir / "sentence_cer_plotted_points.json"
    verification = output_dir / "plot_verification.json"
    require(not any(path.exists() for path in (png, plotted, verification)), "Do not overwrite an existing plot or verification")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titlesize": 12, "axes.labelsize": 11,
                         "savefig.facecolor": "white", "figure.facecolor": "white"}):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.8), sharey=True)
        ymax = max(100, math.ceil((max(point["CER_percent"] for point in points) + 8) / 20) * 20)
        for axis, (cohort, title, _) in zip(axes, COHORTS):
            for arm in ARMS:
                for seed in SEEDS:
                    series = [point for point in points if point["cohort"] == cohort
                              and point["arm"] == arm and point["seed"] == seed]
                    require(tuple(point["completed_update"] for point in series) == STEPS, "Plot point order changed")
                    x = [point["completed_update"] for point in series]
                    y = [point["CER_percent"] for point in series]
                    axis.plot(x, y, color=COLORS[arm], linestyle="-" if seed == 42 else "--",
                              linewidth=1.9, marker="o", markersize=5,
                              markerfacecolor=COLORS[arm] if seed == 42 else "white", zorder=3)
                    axis.scatter([1260], [y[-1]], color=COLORS[arm], marker="*",
                                 s=145 if seed == 42 else 105, edgecolors="black", linewidths=.55, zorder=4)
            axis.axvline(1260, color="#777777", linewidth=.8, linestyle=":", zorder=1)
            axis.set_title(title, pad=12)
            axis.set_xlabel("Optimizer updates")
            axis.set_xticks(STEPS)
            axis.set_xlim(360, 1320)
            axis.set_ylim(0, ymax)
            axis.grid(axis="y", color="#dddddd", linewidth=.7)
            axis.spines[["top", "right"]].set_visible(False)
        axes[0].set_ylabel("Corpus character error rate (CER, %)")
        handles = [Line2D([0], [0], color=COLORS[arm], linestyle="-" if seed == 42 else "--",
                         marker="o", markerfacecolor=COLORS[arm] if seed == 42 else "white",
                         linewidth=1.9, label=f"{ARM_LABELS[arm]} | seed {seed}") for arm in ARMS for seed in SEEDS]
        fig.suptitle("Visual-only lipreading: fixed checkpoint comparison", fontsize=15, y=.975)
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .922), ncol=2,
                   frameon=False, fontsize=10)
        caption = ("Three measured checkpoints only; lines connect observations. Stars mark the prespecified final update 1,260.\n"
                   "Internal pilot: 3 development sentences and 12 training sentences; this is not unseen-speaker or webcam accuracy.\n"
                   "A: sentence-only. B: phrase 420 updates, then sentence 840 updates. Both use the same parents and update count.")
        fig.text(.5, .045, caption, ha="center", va="bottom", fontsize=9, color="#444444", linespacing=1.5)
        fig.subplots_adjust(left=.08, right=.975, top=.77, bottom=.265, wspace=.16)
        fig.savefig(png, dpi=180)
        plt.close(fig)
    source_ref = {"path": str(source), "sha256": sha256(source), "bytes": source.stat().st_size}
    plotted.write_text(json.dumps({"schema": "offline-stage5-CER-plotted-points-v1", "source": source_ref,
        "metric": "NFC corpus character error rate; ratio times100", "points": points,
        "interpolation_or_smoothing": False, "best_checkpoint_selection": False,
        "final_checkpoint": 1260}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = {"schema": "offline-stage5-plot-verification-v1", "status": "passed_numeric_and_scope_checks",
        "created_utc": datetime.now(timezone.utc).isoformat(), "source": source_ref,
        "script_sha256": sha256(Path(__file__)), "plotted_point_count": len(points), "panel_count": 2,
        "point_counts_per_series": 3, "stored_CER_vs_error_denominator_all_passed": True,
        "CER_not_clamped_to_100_percent": True, "new_model_calls": 0, "source_mutations": 0,
        "png": {"path": str(png), "sha256": sha256(png), "bytes": png.stat().st_size},
        "plotted_points": {"path": str(plotted), "sha256": sha256(plotted), "bytes": plotted.stat().st_size},
        "visual_inspection": "pending_separate_human_or_agent_view"}
    verification.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate", type=Path, default=BASE.parent / "results/aggregate_metrics.json")
    parser.add_argument("--output-dir", type=Path, default=BASE)
    args = parser.parse_args()
    report = write_plot(args.aggregate, args.output_dir)
    print(json.dumps({"status": report["status"], "png": report["png"]["path"],
                      "points": report["plotted_point_count"]}))


if __name__ == "__main__":
    main()
