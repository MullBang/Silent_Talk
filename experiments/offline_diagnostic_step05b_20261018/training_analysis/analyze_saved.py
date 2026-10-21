"""Read-only stage5 log/posterior diagnostic; no torch or model imports."""
from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "experiments/offline_comparison_step05_20261013"
OUT = Path(__file__).resolve().parent


def ref(path):
    path = Path(path)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}


def save(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def stats(values):
    a = np.asarray(values, dtype=np.float64)
    assert a.ndim == 1 and a.size and np.isfinite(a).all()
    return {"n": int(a.size), "mean": float(a.mean()), "median": float(np.median(a)),
            "minimum": float(a.min()), "maximum": float(a.max()), "population_std": float(a.std()),
            "p10": float(np.quantile(a, .1)), "p90": float(np.quantile(a, .9))}


def log_window(rows):
    assert rows and all(r["optimizer_updated"] for r in rows)
    losses = [r["loss"] for r in rows]
    x = np.asarray([r["update"] for r in rows], dtype=np.float64)
    y = np.asarray(losses, dtype=np.float64)
    centered = x - x.mean()
    return {"first_update": int(x.min()), "last_update": int(x.max()), "updates": len(rows),
            "kinds": sorted({r["kind"] for r in rows}), "learning_rates": sorted({r["learning_rate"] for r in rows}),
            "loss": stats(losses), "loss_linear_slope_per_100_updates": float(np.dot(centered, y-y.mean()) / np.dot(centered, centered)*100),
            "gradient_norm_before_clip": stats([r["gradient_norm_before_clip"] for r in rows]),
            "gradient_norm_after_clip": stats([r["gradient_norm_after_clip"] for r in rows]),
            "gradient_clipping_updates": sum(r["gradient_norm_before_clip"] > 1 for r in rows),
            "updates_with_zero_gradient_tensor": sum(r["nonzero_gradient_tensors"] < r["parameter_tensors"] for r in rows),
            "true_frames": sum(r["true_frames"] for r in rows), "dense_frames": sum(r["dense_frames"] for r in rows),
            "target_tokens": sum(r["target_tokens"] for r in rows)}


def posterior_statistics(lp):
    assert lp.ndim == 2 and lp.shape[1] == 69 and np.isfinite(lp).all()
    p = np.exp(lp.astype(np.float64))
    sums = p.sum(axis=1)
    assert np.max(np.abs(sums-1)) < 1e-5
    p /= sums[:, None]
    logp = lp.astype(np.float64) - np.log(sums)[:, None]
    entropy = -np.sum(p*logp, axis=1)
    raw = lp.argmax(axis=1)
    return {"T": int(len(lp)), "blank_argmax_frames": int(np.count_nonzero(raw == 0)),
            "blank_probability_sum": float(p[:, 0].sum()), "entropy_nats_sum": float(entropy.sum()),
            "maximum_probability_sum": float(p.max(axis=1).sum()),
            "raw_probability_sum_max_abs_error": float(np.max(np.abs(sums-1)))}


def output_group(rows):
    n = len(rows)
    frames = sum(r["T"] for r in rows)
    chars = sum(r["reference_characters"] for r in rows)
    words = sum(r["reference_words"] for r in rows)
    tokens = sum(r["reference_tokens"] for r in rows)
    ent = sum(r["entropy_nats_sum"] for r in rows)
    return {"clips": n, "frames": frames, "reference_characters": chars, "reference_words": words,
            "reference_tokens": tokens, "character_edits": sum(r["character_edits"] for r in rows),
            "CER_percent": 100 * sum(r["character_edits"] for r in rows)/chars,
            "WER_percent": 100 * sum(r["word_edits"] for r in rows)/words,
            "exact_clips": sum(r["exact"] for r in rows), "empty_predictions": sum(r["prediction"] == "" for r in rows),
            "all_blank_argmax_clips": sum(r["blank_argmax_frames"] == r["T"] for r in rows),
            "replacement_characters": sum(r["replacement_characters"] for r in rows),
            "repetition_ge5_clips": sum(r["repetition_ge5"] for r in rows),
            "blank_argmax_frames": sum(r["blank_argmax_frames"] for r in rows),
            "blank_argmax_frame_percent": 100*sum(r["blank_argmax_frames"] for r in rows)/frames,
            "mean_blank_probability_per_frame": sum(r["blank_probability_sum"] for r in rows)/frames,
            "mean_entropy_nats_per_frame": ent/frames,
            "mean_entropy_div_log69_per_frame": ent/frames/math.log(69),
            "mean_maximum_probability_per_frame": sum(r["maximum_probability_sum"] for r in rows)/frames,
            "predicted_tokens": sum(r["predicted_tokens"] for r in rows),
            "predicted_to_reference_token_ratio": sum(r["predicted_tokens"] for r in rows)/tokens,
            "predicted_to_reference_character_ratio": sum(len(r["prediction"]) for r in rows)/chars,
            "unique_prediction_texts": len({r["prediction"] for r in rows})}


def run():
    sources = []
    def read(path):
        sources.append(ref(path))
        return json.loads(path.read_text(encoding="utf-8"))
    def lines(path):
        sources.append(ref(path))
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    completion = read(SOURCE / "results/completion_manifest.json")
    # Root final audit independently checks the whole manifest; this analyzer hashes every input it consumes.
    runs = []
    output_rows = []
    posterior_sources = []
    windows = []
    for seed in [42, 43]:
        for arm in ["A_sentence", "B_curriculum"]:
            base = SOURCE / f"runs/seed{seed}/{arm}"
            rows = lines(base / "progress.jsonl")
            assert len(rows) == 1260 and [r["update"] for r in rows] == list(range(1,1261))
            assert all(r["kind"] == ("phrase" if arm == "B_curriculum" and r["update"] <= 420 else "full_sentence") for r in rows)
            for first in range(1,1261,60):
                windows.append({"seed": seed, "arm": arm, **log_window(rows[first-1:first+59])})
            phase_lrs = []
            start = 0
            for i in range(1,1261):
                if i == 1260 or (rows[i]["kind"],rows[i]["learning_rate"]) != (rows[start]["kind"],rows[start]["learning_rate"]):
                    phase_lrs.append(log_window(rows[start:i]))
                    start = i
            runs.append({"seed": seed, "arm": arm, "all_updates": log_window(rows), "phase_and_lr_windows": phase_lrs,
                "first60": log_window(rows[:60]), "pre_transition60": log_window(rows[360:420]),
                "post_transition60": log_window(rows[420:480]), "penultimate60": log_window(rows[1140:1200]),
                "final60": log_window(rows[1200:1260]),
                "final60_minus_penultimate60_mean_loss": float(np.mean([r["loss"] for r in rows[1200:]])-np.mean([r["loss"] for r in rows[1140:1200]])),
                "single_transition_updates": [{k:r[k] for k in ("update","kind","clip_ids","loss","learning_rate","target_tokens","gradient_norm_before_clip")} for r in rows[419:422]],
                "transition_loss_is_different_target_kind": arm == "B_curriculum"})
            for update in [420,840,1260]:
                erows = lines(base / f"evaluations/update_{update:04d}/rows.jsonl")
                assert len(erows) == 90
                for row in erows:
                    assert row["status"] == "success"
                    path = Path(row["posterior"]["path"])
                    actual = ref(path)
                    assert actual == row["posterior"]
                    posterior_sources.append(actual)
                    with np.load(path, allow_pickle=False) as z:
                        lp = z["log_probs"]
                        assert np.array_equal(lp.argmax(axis=1), row["raw_argmax_ids"])
                        assert np.array_equal(z["collapsed_ids"], row["collapsed_ids"])
                        ps = posterior_statistics(lp)
                    assert ps["T"] == row["T"]
                    metric = row["metrics"]
                    output_rows.append({"seed": seed, "arm": arm, "update": update,
                        **{k:row[k] for k in ("clip_id","parent_id","split","kind","condition")},
                        "reference": row["reference_text"], "prediction": row["metric_text"],
                        **{k:metric[k] for k in ("character_edits","reference_characters","word_edits","reference_words","exact","replacement_characters","repetition_ge5")},
                        "reference_tokens": len(row["reference_label_ids"]), "predicted_tokens": len(row["collapsed_ids"]),
                        **ps, "posterior_sha256": actual["sha256"]})
    assert len(output_rows) == 1080 and len(windows) == 84
    groups = defaultdict(list)
    for r in output_rows:
        groups[(r["seed"],r["arm"],r["update"],r["split"],r["kind"],r["condition"])].append(r)
    output_groups = [{**dict(zip(("seed","arm","update","split","kind","condition"), k)), **output_group(v)} for k,v in sorted(groups.items())]
    assert len(output_groups) == 144
    # Reconstruct the historical aggregate metrics from rows without importing production evaluator.
    aggregate = read(SOURCE / "results/aggregate_metrics.json")
    final_primary = [r for r in output_groups if r["update"] == 1260 and r["split"] == "dev" and r["kind"] == "full_sentence" and r["condition"] == "normal"]
    assert [round(r["CER_percent"],8) for r in final_primary] == [79.72972973,87.83783784,77.02702703,75.67567568]
    normal_phrase = [r for r in output_rows if r["arm"] == "B_curriculum" and r["split"] == "train" and r["kind"] == "phrase" and r["condition"] == "normal"]
    selected = [r for r in output_groups if r["condition"] == "normal"]
    summary = {"schema": "offline-stage5b-saved-loss-output-diagnostic-v1", "status": "completed_saved_log_and_posterior_analysis",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {"main_training_log_updates": 5040, "evaluation_rows": 1080, "posterior_files": 1080,
            "posterior_frames": sum(r["T"] for r in output_rows), "group_rows": 144, "groups_are_repeated_evaluations_not_independent_examples": True,
            "unique_parent_recordings": 15, "paired_clips": 30, "train_parents": 12, "dev_parents": 3,
            "new_optimizer_updates": 0, "new_model_forward_calls": 0, "checkpoint_tensor_loads": 0,
            "source_video_or_audio_reads": 0, "downloads": 0, "service_changes": 0},
        "definitions": {
            "training_loss": "Stored training-mode batch-of-2 CPU FP32 CTC reduction=mean: arithmetic mean of each sequence negative log CTC probability divided by its target token length; measured before that update. Not eval-mode CTC, CER, or per-frame loss.",
            "loss_summary": "Arithmetic mean over update scalar losses; each update weighs two clips equally after target-length normalization. Phase/target changes are not causal estimates.",
            "loss_slope": "Ordinary least-squares slope against actual update number, multiplied by 100; descriptive, no significance claim.",
            "gradient_norm": "Logged total norm before and after max_norm=1 clipping; greater than 1 is clipping activation, not evidence of exploding gradients by itself.",
            "probabilities": "float64 exp of stored FP32 log probabilities, renormalized per frame; original row-sum deviation recorded and checked below 1e-5.",
            "entropy": "Natural-log categorical entropy over all 69 symbols including blank; also divided by log(69). Group means weighted by frame count.",
            "blank": "Blank index=0. Argmax blank frame share differs from mean probability mass of blank. Blank is normal in CTC; high blank share alone does not prove collapse.",
            "CER": "Sum stored NFC character edit counts / sum reference character counts, retaining internal spaces and U+FFFD. Reconstructed from frozen successful rows.",
            "length_ratio": "Sum collapsed predicted token lengths / sum reference token lengths; character counterpart uses metric prediction string lengths. Not coverage alignment.",
            "confidence": "Maximum token probability and entropy describe frame output sharpness, not calibrated recognition correctness.",
            "train_eval": "Evaluation in fixed eval mode on train clips; training-mode batch loss is not directly equal to this evaluation or CER."},
        "training_runs": runs, "normal_output_groups": selected,
        "all_output_groups_path": str(OUT / "output_groups.json"),
        "phrase_per_clip_trajectory_path": str(OUT / "train_phrase_trajectory.json"),
        "all_posterior_row_statistics_path": str(OUT / "posterior_row_statistics.json"),
        "all_loss_windows_path": str(OUT / "loss_windows.json"),
        "maximum_probability_sum_abs_error": max(r["raw_probability_sum_max_abs_error"] for r in output_rows),
        "limits": ["Post-hoc diagnostic on the fixed historical comparison, not a new training intervention or proof of a unique failure cause.",
            "Only three internal dev parent recordings and two seeds; not unseen-speaker, final-test, or webcam performance.",
            "B has fewer full-sentence exposures and approximately 20.04% fewer true training frames than A; not a pure curriculum-order effect.",
            "Training phrase criterion was not met at update 420, but no extended phrase-training counterfactual exists.",
            "No significance tests or causal claims from repeated evaluation rows. Original final update1260 endpoint remains unchanged.",
            "No per-update saved train-mode posteriors exist here, so blank/entropy at intermediate optimizer updates cannot be reconstructed.",
            "Original provisional/training_ready=false/boundary flags and excluded OOV parent remain unchanged."],
        "sources": sources, "posterior_sources_path": str(OUT / "posterior_sources.json"), "producer": ref(__file__)}
    save(OUT/"loss_windows.json", windows)
    save(OUT/"output_groups.json", output_groups)
    save(OUT/"posterior_row_statistics.json", output_rows)
    save(OUT/"train_phrase_trajectory.json", normal_phrase)
    save(OUT/"posterior_sources.json", posterior_sources)
    save(OUT/"loss_output_analysis.json", summary)
    print(json.dumps({"status": summary["status"], **summary["scope"]}))


if __name__ == "__main__":
    run()
