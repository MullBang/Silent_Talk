"""Verify stage15 fixed-model development evaluation from saved posteriors.

No model forward or training occurs here. All normal beams are decoded again;
greedy text, CTC loss and controls are recalculated for every saved posterior.
Only the selected train60 and existing development val20 cache metadata/labels
are read. Test caches are never accessed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "backend"), str(ROOT / "scripts")]
from models import GRAPHEME_VOCAB
from models.dataset_manifest import load_manifest, sha256_file
from models.decoder import ctc_beam_search_decode
from models.tokenizer import encode_text, validate_tokenizer_metadata
from diagnose_temporal_use import prediction_row, length_bin
from temporal_diagnostics import analyze_ctc, compare_posteriors, transform_frames
from summarize_learning_transition import (
    require, finite_tree, read_json, read_rows, compare, describe, checked_hash,
    inside, summarize_group,
)

MODEL_PATHS = {
    "source6400": "results/step13-budget/run01/step_6400.pt",
    "A_keep9600": "results/step14-learning-rate/run01/A_keep/step_9600.pt",
    "B_reduce9600": "results/step14-learning-rate/run01/B_reduce/step_9600.pt",
}
MODEL_SHAS = {
    "source6400": "af2a579832b585556967d6f885170610d8e7beb1a8a56967f5350ba059d8e519",
    "A_keep9600": "7d5a8bcf1dea50729ba02734deacacb3f38289b4307c8cb71b10fc62915c1c5f",
    "B_reduce9600": "2b5c43f1f0dc1a79300b04f0d0d6a5059de0143b241b067633fa7439a07864ed",
}
CONDITIONS = ["normal", "static_middle", "shuffle_42"]
COHORTS = ["train60", "val20"]
MANIFEST_SHA = "214ce935150b367a15f24f0896901d34c54c6d4fe73d493a9d397c4c94fe59a6"


def ordered_ids(plan: dict) -> list[str]:
    return [sample_id for cohort in COHORTS for sample_id in plan["selection"]["cohorts"][cohort]]


def state_digest(state: dict) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        header = json.dumps([name, str(value.dtype), list(value.shape)], separators=(",", ":")).encode()
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def validate_checkpoint(key: str, spec: dict, plan: dict) -> dict:
    """Bind each diagnostic checkpoint to its previous fixed-step completion."""
    path = Path(spec["checkpoint"]).resolve()
    require(path == (ROOT / MODEL_PATHS[key]).resolve(), "Unexpected fixed checkpoint path")
    compare(spec["checkpoint_sha256"], MODEL_SHAS[key], "Fixed original checkpoint hash")
    digest = checked_hash(path, spec["checkpoint_sha256"])
    previous = read_json(path.parent / "completed.json")
    matches = [row for row in previous["snapshots"] if row["checkpoint"] == path.name]
    require(len(matches) == 1, "Checkpoint missing from previous completed snapshots")
    compare(matches[0]["checkpoint_sha256"], digest, "Previous completion checkpoint hash")
    relative_completed = (path.parent / "completed.json").relative_to(ROOT).as_posix()
    require(relative_completed in plan["prior_files"], "Previous completion not frozen in plan")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    validate_tokenizer_metadata(checkpoint, str(path))
    require(checkpoint["architecture_version"] == "lipnet-compact-v1" and checkpoint["num_classes"] == 69
            and checkpoint["lstm_hidden"] == 256 and checkpoint["vocab"] == GRAPHEME_VOCAB,
            "Unexpected model architecture or tokenizer")
    tensors = checkpoint["state_dict"]
    require(isinstance(tensors, dict) and tensors and all(isinstance(v, torch.Tensor)
            and v.is_floating_point() and bool(torch.isfinite(v).all()) for v in tensors.values()),
            "Invalid checkpoint model tensors")
    require(sum(v.numel() for v in tensors.values()) == 3287781, "Unexpected compact parameter inventory")
    compare(spec["state_sha256"], state_digest(tensors), "Checkpoint tensor digest")
    for field in ("training_data", "training_run"):
        compare(checkpoint[field], spec[field], f"Fixed checkpoint {field}")
    data, run = checkpoint["training_data"], checkpoint["training_run"]
    previous_plan_path = (ROOT / "results/step13-budget/run01/plan.json" if key == "source6400"
                          else ROOT / "results/step14-learning-rate/run01/plan.json")
    compare(run["plan_sha256"], sha256_file(previous_plan_path), "Checkpoint prior training plan")
    compare(data["sample_ids"], plan["training_ids"], "Same training64 lineage")
    require(data["mode"] == "smoke_only" and data["manifest_sha256"] == MANIFEST_SHA
            and data["generalization_evaluation"] is False and data["deployment"] is False,
            "Unexpected checkpoint diagnostic provenance")
    require(run["inference_only"] is True and run["selection"] == "fixed_attempt_schedule"
            and run["attempt_updates"] == (6400 if key == "source6400" else 9600),
            "Unexpected checkpoint training endpoint")
    compare(data["purpose"], "train_subset_budget_diagnostic" if key == "source6400"
            else "train_subset_lr_diagnostic", "Checkpoint diagnostic purpose")
    compare(run["settings"]["learning_rate"], .0001 if key == "B_reduce9600" else .0003, "Checkpoint learning rate")
    if key != "source6400":
        compare(data["branch"], "A_keep" if key == "A_keep9600" else "B_reduce", "Checkpoint branch")
        compare(run["branch"], data["branch"], "Checkpoint training branch")
    return {"checkpoint": str(path), "checkpoint_sha256": digest,
            "architecture_version": checkpoint["architecture_version"], "parameters": 3287781,
            "training_data": data, "training_run": run}


def validate_plan(plan: dict) -> tuple[dict, dict]:
    """Reconstruct metadata-only selection, provenance and selected cache labels."""
    from generalization_selection import select_generalization

    require(plan["schema"] == "lipnet-generalization-v1", "Unsupported stage15 schema")
    require(plan["manifest_sha256"] == MANIFEST_SHA, "Unexpected timed manifest")
    checked_hash(Path(plan["manifest"]), MANIFEST_SHA)
    require(set(plan["models"]) == set(MODEL_PATHS), "Unexpected model inventory")
    require({"scripts/diagnose_generalization.py", "scripts/generalization_selection.py",
             "scripts/temporal_diagnostics.py", "backend/models/decoder.py",
             "backend/models/tokenizer.py"}.issubset(plan["source_hashes"]), "Missing frozen source dependencies")
    for relative, digest in plan["source_hashes"].items():
        checked_hash(inside(ROOT, relative), digest)
    require(bool(plan["prior_files"]), "Missing previous artifact bindings")
    for relative, digest in plan["prior_files"].items():
        checked_hash(inside(ROOT, relative), digest)
    for key, expected in {"conditions": CONDITIONS, "shuffle_seed": 42, "beam_width": 10,
                          "beam_conditions": ["normal"], "precision": "fp32_unpadded_B1",
                          "primary_decoder": "greedy", "batch_size": 1, "training": False,
                          "model_selection": False, "test_cache_access": False,
                          "expected_forwards": 720, "expected_beam_rows": 240,
                          "cudnn_deterministic": True, "cudnn_benchmark": False, "tf32": False}.items():
        compare(plan["settings"][key], expected, f"Fixed evaluation settings.{key}")
    training_ids = plan["training_ids"]
    require(len(training_ids) == len(set(training_ids)) == 64, "Expected original training64 IDs")
    document, _ = load_manifest(plan["manifest"], verify_files=False)
    rows = document["records"]
    selection = select_generalization(rows, training_ids, count=60, seed=42)
    compare(plan["selection"], selection, "Predeclared metadata-only selection")
    require(len(selection["cohorts"]["train60"]) == 60 and len(selection["cohorts"]["val20"]) == 20,
            "Expected train60 plus complete development val20")
    all_records = {row["id"]: row for row in rows}
    ids = ordered_ids(plan)
    require(len(ids) == len(set(ids)) == 80 and not set(ids).intersection(training_ids), "Evaluation/train ID leakage")
    records = {sample_id: all_records[sample_id] for sample_id in ids}
    compare(plan["records"], list(records.values()), "Selected plan metadata")
    train_records = [all_records[sample_id] for sample_id in training_ids]
    familiar = {r["speaker_id"] for r in train_records}
    require(len(familiar) == 8 and {records[s]["speaker_id"] for s in selection["cohorts"]["train60"]} == familiar,
            "Familiar-speaker cohort scope mismatch")
    val_speakers = {records[s]["speaker_id"] for s in selection["cohorts"]["val20"]}
    require(len(val_speakers) == 2 and not val_speakers.intersection(familiar), "Unseen-speaker scope mismatch")
    training_recordings = {r["recording_id"] for r in train_records}
    require(not training_recordings.intersection(r["recording_id"] for r in records.values()), "Training recording leakage")
    for sample_id, row in records.items():
        expected_split = "train" if sample_id in selection["cohorts"]["train60"] else "val"
        compare(row["split"], expected_split, "Selected cache split before access")
        path = inside(Path(plan["manifest"]).parent, row["cache_path"])
        checked_hash(path, row["cache_sha256"])
        with np.load(path, allow_pickle=False) as cache:
            validate_tokenizer_metadata(cache, str(path))
            for field in ("text", "source_text", "speaker_id", "utterance_id", "recording_id", "angle",
                          "source_partition", "roi_method", "label_sha256", "temporal_method", "video_fps", "label_fps"):
                require(str(cache[field]) == str(row[field]), f"Selected cache metadata mismatch: {sample_id}/{field}")
            compare(cache["label"].tolist(), encode_text(row["text"]), "Selected cache tokens")
    checkpoint_details = {key: validate_checkpoint(key, plan["models"][key], plan) for key in MODEL_PATHS}
    return records, checkpoint_details


def validate_runtime_audit(report: dict, spec: dict) -> None:
    """Check hash-bound before/after tensor digests and stored exact comparisons."""
    for field in ("state_before_sha256", "state_after_sha256"):
        compare(report[field], spec["state_sha256"], f"Immutable runtime model.{field}")
    compare(report["evaluation_mode_unchanged"], True, "Preserved evaluation mode")
    for name in ("state_audit", "rng_audit"):
        audit = report[name]
        for field, expected in {"exact": True, "difference_count": 0, "max_abs_numeric_difference": 0.,
                               "nonfinite_difference_count": 0, "differences": [], "excluded_keys": [],
                               "details_truncated": False}.items():
            compare(audit[field], expected, f"Runtime comparison {name}.{field}")
        require(type(audit["compared_leaves"]) is int and audit["compared_leaves"] > 0,
                "Empty runtime comparison")
    state = report["state_audit"]
    require(state["expected_tensor_count"] == state["actual_tensor_count"] == state["matching_tensor_count"]
            == state["compared_leaves"] == len(state["tensors"]) > 0, "Incomplete model state comparison")
    for audit in state["tensors"].values():
        compare(audit, {"exact": True, "compared_leaves": 1, "difference_count": 0,
            "max_abs_numeric_difference": 0., "nonfinite_difference_count": 0,
            "differences": [], "excluded_keys": [], "details_truncated": False}, "Individual state tensor comparison")


def validate_model(folder: Path, plan: dict, records: dict, key: str, plan_sha: str) -> tuple[dict, list[dict]]:
    """Re-decode and rescore one exact clip-by-condition grid from saved log p."""
    report = read_json(folder / "report.json")
    ids = ordered_ids(plan)
    expected_fields = {"status": "completed", "model": key, "plan_sha256": plan_sha,
        "checkpoint_sha256": plan["models"][key]["checkpoint_sha256"], "clips": len(ids),
        "forwards": len(ids)*len(CONDITIONS), "beam_rows": len(ids), "weights_unchanged": True,
        "rng_unchanged": True, "training_performed": False, "test_cache_access": False}
    for field, expected in expected_fields.items():
        compare(report[field], expected, f"Model completion.{field}")
    require(report["seconds"] >= 0, "Negative evaluation duration")
    validate_runtime_audit(report, plan["models"][key])
    checked_hash(folder / "rows.jsonl", report["rows_sha256"])
    checked_hash(folder / "posteriors.npz", report["posteriors_sha256"])
    rows = read_rows(folder / "rows.jsonl")
    expected_pairs = [[sample_id, condition] for sample_id in ids for condition in CONDITIONS]
    compare([[r["id"], r["condition"]] for r in rows], expected_pairs, "Exact clip/condition grid and order")
    normal = {}
    with np.load(folder / "posteriors.npz", allow_pickle=False) as posteriors:
        require(len({r["posterior_key"] for r in rows}) == len(rows)
                and set(posteriors.files) == {r["posterior_key"] for r in rows}, "Posterior inventory mismatch")
        for row in rows:
            sample_id, condition = row["id"], row["condition"]
            source = records[sample_id]
            cohort = "train60" if sample_id in plan["selection"]["cohorts"]["train60"] else "val20"
            for field, value in {"model": key, "split": source["split"], "cohort": cohort,
                "speaker_id": source["speaker_id"], "recording_id": source["recording_id"],
                "length_bin": length_bin(source["frames"]), "reference": source["text"],
                "frames": source["frames"], "posterior_key": sample_id + "__" + condition}.items():
                compare(row[field], value, f"Evaluation row.{field}")
            logp = posteriors[row["posterior_key"]]
            require(logp.dtype == np.float32 and logp.shape == (source["frames"], 69)
                    and bool(np.isfinite(logp).all()), "Invalid finite FP32 unpadded posterior")
            labels = np.asarray(encode_text(source["text"]), dtype=np.int64)
            has_beam = condition == "normal"
            require((row["beam"] is not None) == has_beam, "Beam condition mismatch")
            beam = ctc_beam_search_decode(logp, beam_width=10) if has_beam else ""
            diagnostic = analyze_ctc(logp, labels, beam_text=beam)
            if not has_beam:
                diagnostic = {name: value for name, value in diagnostic.items() if not name.startswith("beam")}
            compare(row["ctc"], diagnostic, "Recomputed CTC diagnostics")
            compare(row["greedy"], prediction_row(source, diagnostic["greedy_text"]), "Recomputed greedy scoring")
            if has_beam:
                compare(row["beam"], prediction_row(source, beam), "Independently decoded beam scoring")
            loss = torch.nn.functional.ctc_loss(torch.from_numpy(logp[:, None, :]), torch.from_numpy(labels),
                torch.tensor([len(logp)]), torch.tensor([len(labels)]), blank=0, reduction="mean", zero_infinity=False)
            compare(row["ctc_loss_per_target_token"], float(loss), "Recomputed CTC loss", atol=2e-5, rtol=2e-5)
            if condition == "normal":
                require(row["comparison_to_normal"] is None, "Normal row has control comparison")
                normal[sample_id] = (logp, diagnostic["greedy_text"])
            else:
                normal_logp, normal_text = normal[sample_id]
                compare(row["comparison_to_normal"], compare_posteriors(normal_logp, logp,
                    normal_text=normal_text, control_text=diagnostic["greedy_text"]), "Recomputed normal/control change")
            _, transform = transform_frames(np.zeros((source["frames"], 1, 1, 3), np.uint8),
                "shuffle" if condition == "shuffle_42" else condition, sample_id=sample_id, seed=42)
            transform["input_shape"] = [source["frames"], 96, 96, 3]
            changed = row["transform"]["pixel_changed_frame_count"]
            require(type(changed) is int and 0 <= changed <= source["frames"], "Invalid changed-pixel frame count")
            if condition == "normal":
                require(changed == 0, "Normal transform changed pixels")
            transform["pixel_changed_frame_count"] = changed
            compare(row["transform"], transform, "Reconstructed transform indices")
    cohorts = {cohort: {condition: summarize_group([r for r in rows if r["cohort"] == cohort
                and r["condition"] == condition]) for condition in CONDITIONS} for cohort in COHORTS}
    by_speaker, by_length = {}, {}
    for cohort in COHORTS:
        selected = [r for r in rows if r["cohort"] == cohort and r["condition"] == "normal"]
        for field, destination in (("speaker_id", by_speaker), ("length_bin", by_length)):
            destination[cohort] = {}
            for group in sorted({r[field] for r in selected}):
                values = [r for r in selected if r[field] == group]
                destination[cohort][group] = {"sample_ids": [r["id"] for r in values], "summary": summarize_group(values)}
    return {"model": key, "checkpoint_sha256": report["checkpoint_sha256"], "clips": len(ids),
            "forwards": len(rows), "beam_rows": len(ids), "seconds": report["seconds"],
            "rows_sha256": report["rows_sha256"], "posteriors_sha256": report["posteriors_sha256"],
            "report_sha256": sha256_file(folder / "report.json"), "cohorts": cohorts,
            "normal_by_speaker": by_speaker, "normal_by_length": by_length}, rows


def paired_normal(rows_by_model: dict) -> dict:
    """Matched-clip differences; negative CER delta means fewer errors later."""
    pairs = [("source6400", "A_keep9600"), ("source6400", "B_reduce9600"), ("A_keep9600", "B_reduce9600")]
    result = {}
    for before, after in pairs:
        result[after + "_minus_" + before] = {}
        for cohort in COHORTS:
            a = {r["id"]: r for r in rows_by_model[before] if r["cohort"] == cohort and r["condition"] == "normal"}
            b = {r["id"]: r for r in rows_by_model[after] if r["cohort"] == cohort and r["condition"] == "normal"}
            require(a.keys() == b.keys() and bool(a), "Unmatched paired evaluation clips")
            metrics = {}
            for decoder in ("greedy", "beam"):
                deltas = [b[s][decoder]["cer"] - a[s][decoder]["cer"] for s in a]
                metrics[decoder] = {"cer_delta": describe(deltas),
                    "improved_count": sum(d < -1e-12 for d in deltas),
                    "unchanged_count": sum(abs(d) <= 1e-12 for d in deltas),
                    "worsened_count": sum(d > 1e-12 for d in deltas),
                    "text_agreement_count": sum(a[s][decoder]["prediction"] == b[s][decoder]["prediction"] for s in a),
                    "before_exact": sum(a[s][decoder]["exact"] for s in a),
                    "after_exact": sum(b[s][decoder]["exact"] for s in b)}
            result[after + "_minus_" + before][cohort] = {"before": before, "after": after, "count": len(a), **metrics}
    return result


def build_summary(run: Path) -> dict:
    run = run.resolve()
    plan, completed = read_json(run / "plan.json"), read_json(run / "completed.json")
    plan_sha = checked_hash(run / "plan.json", completed["plan_sha256"])
    records, checkpoint_details = validate_plan(plan)
    for field, value in {"status": "completed", "cache_ids_opened": ordered_ids(plan), "forwards": 720,
                         "beam_rows": 240, "training_performed": False, "test_cache_access": False,
                         "source_hashes_unchanged": True}.items():
        compare(completed[field], value, f"Completed experiment.{field}")
    require(set(completed["models"]) == set(MODEL_PATHS), "Completed model inventory mismatch")
    checked_hash(run / "cache_audit.json", completed["cache_audit_sha256"])
    compare(read_json(run / "cache_audit.json"), {"plan_sha256": plan_sha,
        "cache_ids_opened": ordered_ids(plan), "cache_sha256": {key: records[key]["cache_sha256"] for key in ordered_ids(plan)},
        "test_cache_access": False}, "Selected cache access audit")
    models, rows_by_model = {}, {}
    for key in MODEL_PATHS:
        checked_hash(run / key / "report.json", completed["models"][key])
        models[key], rows_by_model[key] = validate_model(run / key, plan, records, key, plan_sha)
    require(completed["seconds"] >= 0, "Negative experiment duration")
    predictions = {}
    for cohort in COHORTS:
        predictions[cohort] = []
        lookup = {key: {r["id"]: r for r in rows if r["condition"] == "normal"} for key, rows in rows_by_model.items()}
        for sample_id in plan["selection"]["cohorts"][cohort]:
            record = records[sample_id]
            predictions[cohort].append({"id": sample_id, "speaker_id": record["speaker_id"],
                "recording_id": record["recording_id"], "reference": record["text"], "frames": record["frames"],
                "models": {key: {field: lookup[key][sample_id][field] for field in
                    ("greedy", "beam", "ctc_loss_per_target_token")} for key in MODEL_PATHS}})
    result = {"schema": "lipnet-generalization-summary-v1", "status": "completed_and_verified",
        "created_utc": datetime.now(timezone.utc).isoformat(), "run": str(run),
        "provenance": {"plan_sha256": plan_sha, "completed_sha256": sha256_file(run / "completed.json"),
            "manifest_sha256": plan["manifest_sha256"], "source_hashes": plan["source_hashes"],
            "prior_files": plan["prior_files"], "summarizer_sha256": sha256_file(Path(__file__)),
            "selected_cache_metadata_and_labels_verified": len(records), "test_cache_access": False,
            "model_forward_calls": 0, "training_performed": False, "saved_posterior_rows_verified": 720,
            "saved_normal_beams_independently_redecoded": 240,
            "raw_posterior_recalculation": ["greedy and normal beam texts and CER/WER", "CTC loss",
                "blank/entropy/token statistics", "normal/control comparison"],
            "limits": ["No model forward is repeated during this audit.",
                "Model before/after tensor digests bind to checkpoint weights; stored exact state/RNG comparisons are checked, not replayed.",
                "Cache hashes and transform indices are checked; changed-pixel counts are range-checked, not recomputed."]},
        "protocol": {"settings": plan["settings"], "selection": plan["selection"],
            "training_ids": plan["training_ids"], "checkpoints": checkpoint_details,
            "metric_definitions": {"mean_sentence_jamo_cer": "Arithmetic mean of per-sentence NFD jamo edit distance divided by reference NFD length; may exceed 1.",
                "corpus_jamo_cer": "Sum of NFD jamo edit distances divided by sum of reference NFD lengths.",
                "exact_count": "Number of predictions exactly equal to the full reference text.",
                "ctc_loss_per_target_token": "FP32 CTC loss with reduction mean and zero_infinity false on one unpadded clip.",
                "paired_delta": "After CER minus before CER on the same clip; negative means improvement."}},
        "models": models, "paired_normal": paired_normal(rows_by_model), "normal_predictions": predictions,
        "runtime": {"seconds": completed["seconds"], "environment": completed.get("environment")},
        "interpretation": [
            "train60 contains clips and recordings absent from these fixed checkpoints' training64, with the same eight speakers.",
            "val20 contains two unseen speakers but is a development set used in previous work, not an untouched final test.",
            "The two cohorts also differ in content, length and recording, so their gap does not isolate speaker identity.",
            "The train60 sample is metadata-stratified and not a population-random estimate; scarce short clips limit balance.",
            "Static/shuffle effects measure sensitivity. Posterior change alone does not prove correct lip-motion decoding.",
            "One shared training lineage and one shuffled sequence per clip do not establish repeatability across training seeds.",
            "No training, best-checkpoint selection, test50 inference, checkpoint replacement or webcam deployment is performed.",
            "These controlled dataset results do not measure webcam accuracy. CER complements exact sentences; 1-CER is not program accuracy.",
        ]}
    finite_tree(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", "--run", dest="run", type=Path, default=ROOT / "results/step15-generalization/run01")
    parser.add_argument("--output", "--out", dest="output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite summary: {args.output}")
    torch.set_num_threads(2)
    result = build_summary(args.run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output.resolve()), "status": result["status"]}, ensure_ascii=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
