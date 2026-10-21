"""Check report-critical loss windows and B train-phrase outputs from sources."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import unicodedata

import numpy as np

from audit_candidates import close, encode
from independent_ctc import collapse
from verify_preservation import binding, check, read

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ROOT = HERE.parents[2]
S5 = ROOT / "experiments/offline_comparison_step05_20261013"


def edits(a, b):
    previous = list(range(len(b) + 1))
    for i, first in enumerate(a, 1):
        current = [i]
        for j, second in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j-1] + (first != second)))
        previous = current
    return previous[-1]


def inspect():
    source_path = BASE / "training_analysis/loss_output_analysis.json"
    value = read(source_path)
    frozen = {x["path"]: x for x in read(S5 / "results/completion_manifest.json")["artifacts"]}
    loss_checks, source_bindings = [], []
    assert len(value["training_runs"]) == 4
    for run in value["training_runs"]:
        log_path = S5 / "runs" / f"seed{run['seed']}" / run["arm"] / "progress.jsonl"
        check(frozen[str(log_path)])
        source_bindings.append(binding(log_path))
        rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
        assert [r["update"] for r in rows] == list(range(1, 1261))
        assert all(math.isfinite(r["loss"]) and r["optimizer_updated"] for r in rows)
        assert all(r["nonzero_gradient_tensors"] == r["parameter_tensors"] == 31 for r in rows)
        assert sum(r["loss"] for r in rows) > 0
        means = {}
        for name, first, last in [("penultimate60", 1141, 1200), ("final60", 1201, 1260)]:
            chosen = [r for r in rows if first <= r["update"] <= last]
            assert len(chosen) == 60
            declared = run[name]
            assert (declared["first_update"], declared["last_update"], declared["updates"]) == (first, last, 60)
            assert all(r["kind"] == "full_sentence" and r["learning_rate"] == .0001 for r in chosen)
            means[name] = math.fsum(r["loss"] for r in chosen) / len(chosen)
            close(declared["loss"]["mean"], means[name])
        delta = means["final60"] - means["penultimate60"]
        close(run["final60_minus_penultimate60_mean_loss"], delta)
        loss_checks.append({"seed": run["seed"], "arm": run["arm"], "updates_checked": len(rows),
                            **{k + "_mean": v for k, v in means.items()}, "final_minus_penultimate": delta,
                            "updates_with_zero_gradient_tensor": 0})
    source_groups = {tuple(g[x] for x in ["seed", "arm", "update", "split", "kind", "condition"]): g
                     for g in value["normal_output_groups"]}
    assert len(source_groups) == 48
    output_checks = []
    for seed in [42, 43]:
        for update in [420, 840, 1260]:
            p = S5 / "runs" / f"seed{seed}" / "B_curriculum/evaluations" / f"update_{update:04d}/evaluation_summary.json"
            check(frozen[str(p)])
            source_bindings.append(binding(p))
            evaluation = read(p)
            rows = [r for r in evaluation["rows"] if r["condition"] == "normal" and r["split"] == "train" and r["kind"] == "phrase"]
            assert len(rows) == 12 and len({r["clip_id"] for r in rows}) == 12
            declared = source_groups[(seed, "B_curriculum", update, "train", "phrase", "normal")]
            chars = words = char_errors = word_errors = frames = blank = refs = preds = exact = 0
            blank_mass = entropy = confidence = 0.
            for row in rows:
                check(row["posterior"])
                with np.load(row["posterior"]["path"], allow_pickle=False) as saved:
                    log_probs = saved["log_probs"].astype(np.float64)
                    assert log_probs.shape == (row["T"], 69)
                    tokens = list(collapse(log_probs.argmax(axis=1).tolist()))
                    assert tokens == saved["collapsed_ids"].tolist() == row["collapsed_ids"]
                probability = np.exp(log_probs)
                probability /= probability.sum(axis=1, keepdims=True)
                frames += row["T"]
                blank += int((log_probs.argmax(axis=1) == 0).sum())
                blank_mass += float(probability[:, 0].sum())
                entropy += float((-(probability * np.log(probability)).sum(axis=1)).sum())
                confidence += float(probability.max(axis=1).sum())
                reference = unicodedata.normalize("NFC", row["reference_text"]).strip()
                prediction = unicodedata.normalize("NFC", row["raw_text"]).strip()
                assert encode(reference) == row["reference_label_ids"]
                chars += len(reference)
                words += len(reference.split())
                char_errors += edits(reference, prediction)
                word_errors += edits(reference.split(), prediction.split())
                exact += reference == prediction
                refs += len(row["reference_label_ids"])
                preds += len(tokens)
            expected = {"clips": 12, "frames": frames, "reference_characters": chars, "reference_words": words,
                        "reference_tokens": refs, "predicted_tokens": preds, "character_edits": char_errors,
                        "CER_percent": 100 * char_errors / chars, "WER_percent": 100 * word_errors / words,
                        "exact_clips": exact, "blank_argmax_frames": blank,
                        "blank_argmax_frame_percent": 100 * blank / frames,
                        "mean_blank_probability_per_frame": blank_mass / frames,
                        "mean_entropy_nats_per_frame": entropy / frames,
                        "mean_entropy_div_log69_per_frame": entropy / frames / math.log(69),
                        "mean_maximum_probability_per_frame": confidence / frames,
                        "predicted_to_reference_token_ratio": preds / refs}
            for field, expected_value in expected.items():
                if isinstance(expected_value, int):
                    assert declared[field] == expected_value
                else:
                    close(declared[field], expected_value)
            output_checks.append({"seed": seed, "arm": "B_curriculum", "update": update,
                                  "split": "train", "kind": "phrase", "condition": "normal", **expected})
    return {"schema": "offline-stage5b-independent-critical-loss-output-audit-v1", "status": "passed",
            "created_utc": datetime.now(timezone.utc).isoformat(), "source": binding(source_path),
            "source_bindings": source_bindings, "training_log_updates_checked": 5040,
            "loss_window_checks": loss_checks, "B_train_phrase_checks": output_checks,
            "posterior_rows_checked": 72, "comparison_tolerance": "1e-8 + 1e-10 * abs(expected)",
            "scope_note": "Report-critical 60-update windows and all B normal train phrase checkpoints; not a second complete audit of every descriptive field.",
            "neural_forward_calls": 0, "optimizer_updates": 0, "checkpoint_tensor_loads": 0,
            "producer_imports": 0, "GPU_calls": 0, "source_media_reads": 0}


if __name__ == "__main__":
    output = HERE / "loss_output_audit.json"
    assert not output.exists()
    value = inspect()
    output.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": value["status"], "training_log_updates_checked": value["training_log_updates_checked"],
                      "posterior_rows_checked": value["posterior_rows_checked"], "loss_window_checks": value["loss_window_checks"],
                      "phrase_checks": [{k: r[k] for k in ["seed", "update", "CER_percent", "exact_clips", "blank_argmax_frame_percent", "predicted_to_reference_token_ratio"]} for r in value["B_train_phrase_checks"]]}))
