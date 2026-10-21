"""Independent saved-result audit: NumPy only, no model/checkpoint imports.

Reconstruct input transforms, numeric-ID Hangul decoding and full-matrix edit
distances independently of the training runner and evaluation contract.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import unicodedata

import numpy as np

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def check(value, label):
    if not value:
        raise ValueError(label)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def bound(binding):
    path = Path(binding["path"])
    check(digest(path) == binding["sha256"], "File SHA mismatch: " + str(path))
    if "bytes" in binding:
        check(path.stat().st_size == binding["bytes"], "File size mismatch")
    return path


def array_digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def text(value):
    return unicodedata.normalize("NFC", value).strip()


def decode(raw):
    collapsed = [int(v) for i, v in enumerate(raw) if v != 0 and (i == 0 or v != raw[i-1])]
    output, i = [], 0
    while i < len(collapsed):
        first = collapsed[i]
        if first == 68:
            output.append(" ")
            i += 1
        elif 1 <= first <= 19 and i+1 < len(collapsed) and 20 <= collapsed[i+1] <= 40:
            vowel = collapsed[i+1]
            i += 2
            tail = 0
            if i < len(collapsed) and 41 <= collapsed[i] <= 67:
                tail = collapsed[i]-40
                i += 1
            output.append(chr(0xAC00 + (first-1)*588 + (vowel-20)*28 + tail))
        else:
            output.append("\ufffd")
            i += 1
    return collapsed, "".join(output)


def distance(reference, prediction):
    matrix = np.zeros((len(reference)+1, len(prediction)+1), np.int64)
    matrix[:, 0] = np.arange(len(reference)+1)
    matrix[0, :] = np.arange(len(prediction)+1)
    for i in range(1, len(reference)+1):
        for j in range(1, len(prediction)+1):
            matrix[i, j] = min(matrix[i-1, j]+1, matrix[i, j-1]+1,
                               matrix[i-1, j-1]+int(reference[i-1] != prediction[j-1]))
    return int(matrix[-1, -1])


def score(reference, prediction):
    ref, pred = text(reference), text(prediction)
    maximum, run, previous = 0, 0, None
    for char in pred:
        run = run+1 if char == previous else 1
        previous = char
        if not char.isspace():
            maximum = max(maximum, run)
    return {"reference": ref, "prediction": pred, "failed": False,
            "character_edits": distance(ref, pred), "reference_characters": len(ref),
            "word_edits": distance(ref.split(), pred.split()), "reference_words": len(ref.split()),
            "exact": ref == pred, "replacement_characters": pred.count("\ufffd"),
            "maximum_same_nonspace_character_run": maximum, "repetition_ge5": maximum >= 5}


def aggregate(scores):
    total = {key: sum(s[key] for s in scores) for key in
             ["character_edits", "reference_characters", "word_edits", "reference_words"]}
    return {"count": len(scores), **total,
            "CER": total["character_edits"]/total["reference_characters"],
            "WER": total["word_edits"]/total["reference_words"],
            "exact_count": sum(s["exact"] for s in scores), "failure_count": 0,
            "repetition_ge5_count": sum(s["repetition_ge5"] for s in scores)}


def run():
    plan = read(BASE/"plan.json")
    data = read(bound(plan["data_plan"]))
    for binding in plan["bindings"]:
        bound(binding)
    rows = {row["clip_id"]: row for row in data["records"]}
    check(len(rows) == 17 and len(data["dev"]) == 3, "Data count")
    check({r["review_id"] for r in data["held"]} == {f"Q{i:02}" for i in range(3, 9)}, "Held scope")
    inputs, labels = {}, {}
    for sid, row in rows.items():
        meta = read(bound({"path": row["metadata_path"], "sha256": row["metadata_sha256"]}))
        source = bound({"path": row["npz_path"], "sha256": row["npz_sha256"], "bytes": row["npz_bytes"]})
        with np.load(source, allow_pickle=False) as packed:
            rgb = packed["frames_rgb"]
            check(array_digest(rgb) == meta["rgb_uint8_sha256"], "Source RGB hash")
            check(packed["training_ready"].item() is False and packed["provisional"].item() is True, "Source flags")
            check(rgb.shape == (row["T"], 96, 96, 3), "Source shape")
            labels[sid] = packed["label"].copy()
            pixels = rgb.astype(np.float32)/np.float32(255)
            pixels -= np.array([.485, .456, .406], np.float32)
            pixels /= np.array([.229, .224, .225], np.float32)
            inputs[sid] = np.ascontiguousarray(np.moveaxis(pixels, -1, 0))
    initial, summaries, schedule_rows = {}, {}, {}
    posterior_count, frame_count, update_count, checkpoints = 0, 0, 0, 0
    for seed in [42, 43]:
        for arm in ["A12", "B14"]:
            name = f"{arm}_seed{seed}"
            directory = BASE/"results"/name
            result, contract = read(directory/"result.json"), read(directory/"run_contract.json")
            check(result["status"] == "completed" and result["physical_updates"] == result["total_history"] == 1680, name+" completion")
            check(contract["resume"] is None, "This audit expects the four fresh runs")
            check(result["identity"]["plan_sha256"] == digest(BASE/"plan.json"), "Plan identity")
            check(contract["optimizer_ids"] == data["arms"][arm], "Optimizer IDs")
            initial[name] = result["initial_state_hashes"]
            check(initial[name] == contract["initial_state_hashes"], "Initial state record")
            tasks = read(bound(plan["schedules"][arm][str(seed)]))["updates"]
            schedule_rows[name] = tasks
            logs = [json.loads(line) for line in (directory/"progress.jsonl").read_text(encoding="utf-8").splitlines()]
            check(len(tasks) == len(logs) == 1680, "Schedule/progress length")
            for task, log in zip(tasks, logs):
                check(log["update"] == task["update"] and log["clip_ids"] == task["clip_ids"], "Progress order")
                check(log["learning_rate"] == task["lr"] and log["lengths"] == task["lengths"], "Progress LR/length")
                check(log["optimizer_updated"] and np.isfinite(log["loss"]), "Progress update/loss")
                for field in ["true_frames", "dense_frames"]:
                    check(log[field] == task[field], "Progress "+field)
                check(set(log["clip_ids"]) <= set(data["arms"][arm]), "Nontrain optimizer input")
            exposure = Counter(sid for task in tasks for sid in task["clip_ids"])
            check(set(exposure.values()) == {280 if arm == "A12" else 240}, "Exposure balance")
            for field in ["true_frames", "dense_frames"]:
                check(sum(t[field] for t in tasks) == result[field] == plan["planned_accounting"][name][field], "Final "+field)
            for binding in result["checkpoints"]:
                bound(binding)  # File hashes only; no tensor deserialization.
                checkpoints += 1
            check(len(result["evaluations"]) == 2, "Evaluation checkpoints")
            summaries[name] = {}
            for update, link in zip([840, 1680], result["evaluations"]):
                summary_path = bound(link["summary"])
                summary = read(summary_path)
                eval_dir = summary_path.parent
                stored = [json.loads(line) for line in (eval_dir/"rows.jsonl").read_text(encoding="utf-8").splitlines()]
                check(stored == summary["rows"], "Summary/JSONL rows")
                isolation = read(eval_dir/"isolation.json")
                for field in ["state", "gradients", "modes"]:
                    check(isolation[field+"_before"] == isolation[field+"_after"], "Isolation "+field)
                wanted_ids = data["arms"][arm] if update == 840 else sorted(rows)
                wanted_conditions = ["normal"] if update == 840 else ["normal", "static", "shuffled"]
                expected = {(sid, condition) for sid in wanted_ids for condition in wanted_conditions}
                actual = {(r["clip_id"], r["condition"]) for r in stored}
                check(len(stored) == len(actual) and actual == expected, "Evaluation missing/duplicate pairs")
                scores = defaultdict(list)
                for row in stored:
                    sid, condition = row["clip_id"], row["condition"]
                    source, length = rows[sid], rows[sid]["T"]
                    check(row["status"] == "success" and row["audio_input"] is False, "Evaluation status/input")
                    check(row["arm"] == arm and row["seed"] == seed and row["update"] == update, "Evaluation identity")
                    check(row["reference"] == source["text"] and row["reference_label_ids"] == labels[sid].tolist(), "Target identity")
                    check(row["seen_in_optimizer"] == (sid in data["arms"][arm]), "Training exposure annotation")
                    expected_cohort = "dev3" if sid in data["dev"] else "new2" if sid in data["new2"] else "train_common12"
                    check(row["cohort"] == expected_cohort, "Cohort label")
                    if condition == "normal":
                        indices = np.arange(length)
                    elif condition == "static":
                        indices = np.repeat(length//2, length)
                    else:
                        local_seed = int(hashlib.sha256(("stage5h/"+sid).encode()).hexdigest()[:16], 16)
                        indices = np.random.default_rng(local_seed).permutation(length)
                    check(row["frame_indices"] == indices.tolist(), "Temporal control")
                    check(row["input_sha256"] == array_digest(inputs[sid][:, indices]), "Actual transformed input hash")
                    check(row["input_audit"]["normalized_input_sha256"] == array_digest(inputs[sid]), "Normalized input hash")
                    with np.load(bound(row["posterior"]), allow_pickle=False) as packed:
                        values = packed["log_probs"]
                        check(values.dtype == np.float32 and values.shape == (length, 69) and np.isfinite(values).all(), "Posterior shape/finite")
                        check(np.allclose(np.exp(values.astype(np.float64)).sum(1), 1, atol=1e-5, rtol=1e-5), "Posterior normalization")
                        check(array_digest(values) == row["posterior_array_sha256"], "Posterior hash")
                        check(np.array_equal(packed["frame_indices"], indices) and np.array_equal(packed["reference_labels"], labels[sid]), "Posterior metadata")
                        argmax = values.argmax(1).tolist()
                        check(argmax == packed["raw_argmax_ids"].tolist() == row["raw_argmax_ids"], "Argmax")
                    collapsed, prediction = decode(argmax)
                    check(collapsed == row["collapsed_ids"] and prediction == row["raw_text"] and text(prediction) == row["metric_text"], "CTC/Hangul/UFFFD decoding")
                    independent = score(source["text"], prediction)
                    check(independent == row["metrics"], "Independent edit counts")
                    scores[f"{expected_cohort}/{condition}"].append(independent)
                    posterior_count += 1
                    frame_count += length
                recalculated = {key: aggregate(values) for key, values in scores.items()}
                check(recalculated == summary["cohorts"] == link["cohorts"], "Aggregate metrics")
                check(summary["count"] == link["count"] == len(stored) and summary["failures"] == link["failures"] == 0, "Summary counts")
                check(summary["frames"] == link["frames"] == sum(r["T"] for r in stored), "Evaluation frame accounting")
                summaries[name][str(update)] = recalculated
            update_count += len(logs)
        check(initial[f"A12_seed{seed}"] == initial[f"B14_seed{seed}"], "A/B paired initial state")
        for epoch in range(1, 241):
            a = [s for row in schedule_rows[f"A12_seed{seed}"] if row["epoch"] == epoch for s in row["clip_ids"]]
            b = [s for row in schedule_rows[f"B14_seed{seed}"] if row["epoch"] == epoch for s in row["clip_ids"] if s in data["arms"]["A12"]]
            check(a == b, "Shared old-parent order")
    check(update_count == 6720 and posterior_count == 256 and checkpoints == 8, "Total accounting")
    preflight = read(BASE/"results/preflight/result.json")
    check(preflight["status"] == "passed" and preflight["physical_updates"] == 5, "Preflight record")
    preserved = {}
    prior = read(ROOT/"experiments/offline_phrase_review_step05f_20261018/audit/source_preservation_final.json")
    manifests = {"scope": bound(prior["service_and_prior_scope_binding"]),
                 "review_r8": ROOT/"experiments/offline_phrase_boundary_step05g_20261018/results/review_snapshot_manifest_r8.json"}
    for name, path in manifests.items():
        entries = read(path)["artifacts"]
        for entry in entries:
            if entry.get("exists", True):
                bound(entry)
            else:
                check(not Path(entry["path"]).exists(), "Previously absent scope artifact appeared")
        preserved[name] = len(entries)
    return {"schema": "offline-stage5h-independent-saved-results-audit-v1", "status": "passed",
            "client_date": "2026-10-19", "created_utc": datetime.now(timezone.utc).isoformat(),
            "plan_sha256": digest(BASE/"plan.json"), "auditor_sha256": digest(__file__),
            "cache_inputs_checked": 17, "posteriors_checked": posterior_count, "posterior_frames": frame_count,
            "main_updates_checked": update_count, "preflight_updates_recorded": 5,
            "total_physical_updates_recorded": update_count+5, "checkpoint_file_hashes_checked": checkpoints,
            "paired_initial_states_equal": True, "shared_order_epochs_per_seed": 240,
            "preservation_counts": preserved, "cohorts": summaries,
            "decoder": "independent numeric Hangul composition with U+FFFD preservation",
            "metrics": "independent full-matrix Levenshtein, NFC outer-strip, corpus totals",
            "network_forward_calls": 0, "checkpoint_tensor_loads": 0, "original_media_reads": 0,
            "limitations": "Saved-state hashes and outputs validated; no independent rerun or checkpoint-tensor reconstruction."}


if __name__ == "__main__":
    target = BASE/"audit/final_saved_results_check.json"
    check(not target.exists(), "Audit already exists")
    report = run()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: v for k, v in report.items() if k != "cohorts"}, ensure_ascii=False))
