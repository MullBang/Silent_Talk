"""Independently audit stage05 evidence, all posteriors, strings and metrics.

No model forward, original media, test payload or torch import is required.
The full-matrix text metrics and greedy Unicode composer are independent of the
runner. Beam10 is replayed using the frozen shared decoder, not a second beam
algorithm. Pixel hashes verify prepared inputs; original source extraction is
not rerun by this audit.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time
import unicodedata

import numpy as np

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from models.decoder import ctc_beam_search_decode  # shared, explicitly not independent

BRANCHES = ["A_keep256", "B_expand512"]
ARMS = ["old_timing", "aligned_timing"]
METHODS = ["greedy", "beam10"]
LEADS = [chr(x) for x in range(0x1100, 0x1113)]
VOWELS = [chr(x) for x in range(0x1161, 0x1176)]
TAILS = [chr(x) for x in range(0x11A8, 0x11C3)]
VOCAB = ["<blank>", *LEADS, *VOWELS, *TAILS, " "]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def local(path):
    path = Path(path).resolve()
    require(path.is_relative_to(ROOT), f"Audit refuses non-workspace path: {path}")
    return path


def sha(path):
    with local(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(local(path).read_text(encoding="utf8"))


def verify(item):
    require(sha(item["path"]) == item["sha256"], f"Evidence digest differs: {item['path']}")


def equal(actual, expected, context="value"):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), f"Keys differ: {context}")
        for key in expected:
            equal(actual[key], expected[key], f"{context}.{key}")
    elif isinstance(expected, float):
        require(isinstance(actual, (int, float)) and math.isfinite(actual)
                and math.isclose(actual, expected, abs_tol=1e-12, rel_tol=1e-12), f"Number differs: {context}")
    else:
        require(actual == expected, f"Value differs: {context}: {actual!r} != {expected!r}")


def distance(reference, prediction):
    matrix = [[0] * (len(prediction) + 1) for _ in range(len(reference) + 1)]
    for i in range(len(reference) + 1):
        matrix[i][0] = i
    for j in range(len(prediction) + 1):
        matrix[0][j] = j
    for i, left in enumerate(reference, 1):
        for j, right in enumerate(prediction, 1):
            matrix[i][j] = min(matrix[i - 1][j] + 1, matrix[i][j - 1] + 1,
                               matrix[i - 1][j - 1] + (left != right))
    return matrix[-1][-1]


def primary_counts(reference, prediction):
    ref = unicodedata.normalize("NFC", reference).strip()
    pred = unicodedata.normalize("NFC", prediction).strip()
    ref_normal, pred_normal = " ".join(ref.split()), " ".join(pred.split())
    return {"character_edits": distance(ref, pred), "reference_characters": len(ref),
            "word_edits": distance(ref.split(), pred.split()), "reference_words": len(ref.split()),
            "exact": ref == pred, "normalized_character_edits": distance(ref_normal, pred_normal),
            "normalized_reference_characters": len(ref_normal), "normalized_exact": ref_normal == pred_normal}


def metric_text(value):
    return " ".join("".join(" " if c.isspace() or unicodedata.category(c).startswith("P") else c
                            for c in unicodedata.normalize("NFC", value)).split())


def historical_counts(source, prediction):
    reference = source["text"]
    ref, pred = metric_text(reference), metric_text(prediction)
    ref_jamo = unicodedata.normalize("NFD", ref)
    pred_jamo = unicodedata.normalize("NFD", pred)
    jamo_edits, word_edits = distance(ref_jamo, pred_jamo), distance(ref.split(), pred.split())
    return {"id": source["id"], "reference": reference, "prediction": prediction,
            "frames": source["frames"], "cer": jamo_edits / len(ref_jamo),
            "wer": word_edits / len(ref.split()), "exact": reference == prediction,
            "reference_jamo_count": len(ref_jamo), "reference_word_count": len(ref.split()),
            "jamo_edit_distance": jamo_edits, "word_edit_distance": word_edits}


def independent_greedy(lp):
    """Collapse repeats before blank0 removal, then independently compose tokens."""
    path = np.argmax(lp, axis=1).tolist()
    collapsed = [token for i, token in enumerate(path) if token != 0 and (i == 0 or token != path[i - 1])]
    symbols = [VOCAB[token] for token in collapsed]
    output, i = [], 0
    while i < len(symbols):
        if symbols[i] == " ":
            output.append(" ")
            i += 1
        elif symbols[i] in LEADS and i + 1 < len(symbols) and symbols[i + 1] in VOWELS:
            lead, vowel = LEADS.index(symbols[i]), VOWELS.index(symbols[i + 1])
            i += 2
            tail = 0
            if i < len(symbols) and symbols[i] in TAILS:
                tail = TAILS.index(symbols[i]) + 1
                i += 1
            output.append(chr(0xAC00 + (lead * 21 + vowel) * 28 + tail))
        else:
            output.append("\ufffd")
            i += 1
    return "".join(output)


def summarize(rows, method):
    entries = [r["outputs"][method] for r in rows]
    summed = {key: sum(e["metrics"][key] for e in entries) for key in
              ["character_edits", "reference_characters", "word_edits", "reference_words",
               "normalized_character_edits", "normalized_reference_characters"]}
    summed.update(clips=len(rows), failed_inputs=sum(e["status"] != "ok" for e in entries),
                  exact_sentences=sum(e["metrics"]["exact"] for e in entries),
                  normalized_exact_sentences=sum(e["metrics"]["normalized_exact"] for e in entries))
    summed["cer_percent"] = 100 * summed["character_edits"] / summed["reference_characters"]
    summed["wer_percent"] = 100 * summed["word_edits"] / summed["reference_words"]
    summed["normalized_cer_percent"] = 100 * summed["normalized_character_edits"] / summed["normalized_reference_characters"]
    summed["mean_jamo_cer_percent"] = 100 * statistics.mean(e["legacy_metrics"]["cer"] for e in entries)
    summed["jamo_edits"] = sum(e["legacy_metrics"]["jamo_edit_distance"] for e in entries)
    summed["reference_jamos"] = sum(e["legacy_metrics"]["reference_jamo_count"] for e in entries)
    summed["pooled_jamo_cer_percent"] = 100 * summed["jamo_edits"] / summed["reference_jamos"]
    summed["replacement_characters"] = sum(e["prediction"].count("\ufffd") for e in entries)
    return summed


def audit():
    started = time.perf_counter()
    plan_path, result_path = BASE / "eval_plan.json", BASE / "results/result.json"
    destination = BASE / "results/independent_audit.json"
    require(not destination.exists(), "Preserve existing independent audit")
    plan, result = read(plan_path), read(result_path)
    require(result["status"] == "completed", "Run must complete before final audit")
    require(result["plan_sha256"] == sha(plan_path) and result["plan"] == plan, "Plan binding differs")
    equal(plan["branches"], BRANCHES); equal(plan["arms"], ARMS); equal(plan["methods"], METHODS)
    for evidence in [*plan["files"].values(), plan["runner"], *plan["metric_sources"].values()]:
        verify(evidence)
    for name, digest in plan["legacy_source_hashes"].items():
        require(sha(ROOT / name) == digest, f"Historical source changed: {name}")
    for specification in plan["models"].values():
        for evidence in specification.values():
            verify(evidence)
    source_plan = read(plan["files"]["source_plan"]["path"])
    sources = {r["id"]: r for r in source_plan["records"]}
    ids = plan["source_ids"]
    require(len(ids) == len(set(ids)) == len(sources) == 20 and list(sources) == ids, "Cohort differs")
    require(all(sources[s]["split"] == "val" for s in ids), "Non-validation input")
    require(sum(len(sources[s]["text"]) for s in ids) == 667
            and sum(len(sources[s]["text"].split()) for s in ids) == 189, "Denominator differs")
    historical_plan = read(plan["files"]["legacy_plan"]["path"])
    require(historical_plan["cohorts"]["val20"] == ids, "Historical cohort differs")
    historical_sources = {r["id"]: r for r in historical_plan["records"]}
    for sid in ids:
        require(all(historical_sources[sid][key] == sources[sid][key] for key in
                    ["text", "source_text", "split", "frames", "speaker_id"]), "Historical reference differs")
    manifest = read(plan["files"]["input_manifest"]["path"])
    require(manifest["status"] == "completed" and manifest["reproduction_gate"]["passed"], "Inputs incomplete")
    for field in ["source_plan", "alignment_sidecar", "preparation_plan"]:
        verify(manifest[field])
    preparation = read(manifest["preparation_plan"]["path"])
    for item in preparation["code"]:
        verify(item)
    verify(preparation["parent_manifest"])
    offsets_document = read(plan["files"]["alignment_sidecar"]["path"])
    require(offsets_document["status"] == "validated", "Unvalidated clock mapping")
    for item in offsets_document["evidence_files"]:
        verify(item)
    offsets = {r["source_name"]: r["label_to_video_offset_seconds"] for r in offsets_document["sources"]}
    inputs = {(r["arm"], r["id"]): r for r in manifest["records"]}
    require(len(inputs) == len(manifest["records"]) == 40 and set(inputs) == {(a, s) for a in ARMS for s in ids}, "Input inventory differs")
    trace_frames = 0
    for (arm, sid), entry in inputs.items():
        source = sources[sid]
        path = local(entry["npz_path"])
        require(path == local(source["legacy_cache_absolute_path"]) if arm == "old_timing"
                else path.is_relative_to(BASE / "inputs"), "Unplanned payload location")
        verify({"path": str(path), "sha256": entry["npz_sha256"]})
        with np.load(path, allow_pickle=False) as payload:
            frames = payload["frames"]
            require(frames.dtype == np.uint8 and frames.shape == (source["frames"], 96, 96, 3), "Input shape differs")
            require(hashlib.sha256(frames.tobytes()).hexdigest() == entry["pixel_sha256"], "Pixel digest differs")
            require(str(payload["text"]) == entry["reference"] == source["text"]
                    and str(payload["source_text"]) == source["source_text"], "Input transcript differs")
            tokens = [VOCAB.index(c) for c in unicodedata.normalize("NFD", source["text"])]
            require(payload["label"].tolist() == tokens, "Input label tokens differ")
            require(len(frames) >= len(tokens) + sum(a == b for a, b in zip(tokens, tokens[1:])), "CTC impossible")
            require(hashlib.sha256(payload["label"].astype("<i8").tobytes()).hexdigest() == entry["label_tokens_sha256"], "Label hash differs")
        verify({"path": entry["selected_trace_path"], "sha256": entry["selected_trace_sha256"]})
        trace = read(entry["selected_trace_path"])
        offset = 0.0 if arm == "old_timing" else offsets[Path(source["source_video"]["path"]).name]
        start, end = source["start_time"] + offset, source["end_time"] + offset
        require(entry["label_to_video_offset_seconds"] == trace["label_to_video_offset_seconds"] == offset, "Offset differs")
        require(entry["video_start_time"] == trace["video_start_time"] == start
                and entry["video_end_time"] == trace["video_end_time"] == end, "Time bounds differ")
        require(len(trace["selected"]) == entry["frame_count"] == source["frames"] == round((end - start) * 25), "Input frame count differs")
        require(entry["native_fps"] == trace["native_fps"] == source["video_fps"]
                and entry["target_fps"] == trace["target_fps"] == 25, "Input frame rate differs")
        for k, point in enumerate(trace["selected"]):
            target = start + k / 25
            index = round(target * source["video_fps"])
            require(point["source_frame_index"] == point["annotation_box_index"] == index
                    and point["target_video_seconds"] == target, "Trace selection differs")
            require(abs(point["actual_pts_seconds"] - index / source["video_fps"]) <= .001, "Trace PTS differs")
        trace_frames += len(trace["selected"])
        if arm == "old_timing":
            require(entry["old_timing_reproduction_exact"] is True and entry["npz_sha256"] == source["cache_sha256"], "Old pixels not bound to original cache")
    inventory = {(r["branch"], r["arm"], r["id"]): r for r in result["records"]}
    require(len(inventory) == len(result["records"]) == 80
            and set(inventory) == {(b, a, s) for b in BRANCHES for a in ARMS for s in ids}, "Result inventory differs")
    gate = {(r["branch"], r["id"]): r for r in result["historical_replay"]["records"]}
    require(result["historical_replay"]["passed"] and len(gate) == len(result["historical_replay"]["records"]) == 40, "Replay gate inventory differs")
    posterior_count = output_count = replay_count = 0
    max_probability_residual = 0.0
    for branch in BRANCHES:
        spec = plan["models"][branch]
        history = [json.loads(line) for line in local(spec["rows"]["path"]).read_text(encoding="utf8").splitlines()]
        history = [r for r in history if r["cohort"] == "val20" and r["condition"] == "normal"]
        require(len(history) == 20 and {r["id"] for r in history} == set(ids), "Historical rows differ")
        historical_rows = {r["id"]: r for r in history}
        with np.load(local(spec["posteriors"]["path"]), allow_pickle=False) as old_posteriors:
            for arm in ARMS:
                for sid in ids:
                    row, source = inventory[branch, arm, sid], sources[sid]
                    require(row["reference"] == source["text"] and row["speaker_id"] == source["speaker_id"]
                            and row["frames"] == source["frames"], "Result source differs")
                    require(set(row["outputs"]) == set(METHODS), "Decoder inventory differs")
                    verify(row["posterior"])
                    with np.load(local(row["posterior"]["path"]), allow_pickle=False) as arrays:
                        require(arrays.files == ["log_probs"], "Posterior array inventory differs")
                        lp = arrays["log_probs"]
                    require(lp.shape == (source["frames"], 69) and lp.dtype == np.float32 and np.isfinite(lp).all(), "Invalid posterior")
                    residual = float(np.max(np.abs(np.exp(lp.astype(np.float64)).sum(axis=1) - 1)))
                    max_probability_residual = max(max_probability_residual, residual)
                    require(residual < 2e-6, "Posterior does not normalize")
                    equal(row["blank_fraction"], float(np.mean(lp.argmax(axis=-1) == 0)), "blank_fraction")
                    require(math.isfinite(row["ctc_loss_per_target_token"]) and row["ctc_loss_per_target_token"] >= 0, "Invalid saved CTC loss")
                    decoded = {"greedy": independent_greedy(lp), "beam10": ctc_beam_search_decode(lp, beam_width=10)}
                    for method, prediction in decoded.items():
                        entry = row["outputs"][method]
                        require(entry["status"] == "ok" and entry["prediction"] == prediction, f"Decoder replay differs: {branch}/{arm}/{sid}/{method}")
                        equal(entry["metrics"], primary_counts(source["text"], prediction), "NFC metrics")
                        equal(entry["legacy_metrics"], historical_counts(source, prediction), "NFD metrics")
                        output_count += 1
                    if arm == "old_timing":
                        old = historical_rows[sid]
                        require(old["reference"] == source["text"] and np.array_equal(lp, old_posteriors[old["posterior_key"]]), "Historical posterior differs")
                        require(decoded["greedy"] == old["greedy"]["prediction"]
                                and decoded["beam10"] == old["beam"]["prediction"], "Historical text differs")
                        equal(gate[branch, sid], {"branch": branch, "id": sid, "posterior_exact": True,
                              "max_abs_posterior_difference": 0.0, "greedy_exact": True, "beam10_exact": True}, "Replay record")
                        replay_count += 1
                    posterior_count += 1
                print(f"Audited {branch}/{arm}: {posterior_count}/80 posteriors, {output_count}/160 outputs", flush=True)
    aggregate = {}
    for branch in BRANCHES:
        for arm in ARMS:
            rows = [inventory[branch, arm, sid] for sid in ids]
            for method in METHODS:
                item = summarize(rows, method)
                item["per_speaker"] = {s: summarize([r for r in rows if r["speaker_id"] == s], method)
                                       for s in sorted({r["speaker_id"] for r in rows})}
                aggregate[f"{branch}/{arm}/{method}"] = item
    equal(result["aggregate"], aggregate, "Aggregates")
    require("torch" not in sys.modules, "Audit unexpectedly imported torch")
    report = {"schema": "legacy-step05-independent-audit-v1", "status": "passed",
              "created_utc": datetime.now(timezone.utc).isoformat(), "plan_sha256": sha(plan_path),
              "result_sha256": sha(result_path), "audit_script_sha256": sha(Path(__file__)),
              "verified": {"input_hashes_and_payloads": 40, "trace_selected_frames": trace_frames,
                           "historical_posterior_replays": replay_count, "posteriors": posterior_count,
                           "greedy_independent_replays": posterior_count, "shared_beam10_replays": posterior_count,
                           "full_matrix_NFC_and_NFD_output_metrics": output_count,
                           "full_cohort_aggregates": len(aggregate), "speaker_aggregates": 16,
                           "max_probability_sum_residual": max_probability_residual},
              "scope": {"model_forward": False, "torch_import": False, "original_media_read": False,
                        "test_payload_read": False, "training": False, "beam_algorithm_independent": False},
              "limitations": ["Beam search replay uses the frozen shared decoder; it is not an independent beam algorithm.",
                              "Input pixels and saved selection traces are verified; original source extraction is not rerun.",
                              "Historical posterior equality is verified; this audit does not rerun the neural network.",
                              "CTC loss is checked finite/nonnegative, not independently recomputed."],
              "seconds": time.perf_counter() - started, "aggregate": aggregate}
    with destination.open("x", encoding="utf8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"audit": str(destination), "status": "passed", "verified": report["verified"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    audit()
